<#
.SYNOPSIS
    Windows/PowerShell launch harness for scripts/run_v2_official_test.sh.

.DESCRIPTION
    The official entry point remains the bash script
    `scripts/run_v2_official_test.sh`; it is what the team and the grading
    environment use.  This file exists only because this workstation has no
    usable bash: Git Bash is absent and the WSL distribution has no Docker
    integration, so the bash runner cannot be executed here.

    It performs the SAME steps with the SAME environment variables and the SAME
    inner scripts (run_v2_server.sh / run_v2_perception.sh /
    run_v2_decision_client.sh), and applies the same fail-closed referee check.
    Keep the two in sync when either changes.

.EXAMPLE
    pwsh -File scripts/run_v2_official_test.ps1 -DurationSec 900
#>
[CmdletBinding()]
param(
    [string]$HostRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$Root = '/workspace/baseline',
    [int]$DurationSec = 600,
    [string]$ServerImage = 'supermarket_sorting:server',
    [string]$ClientImage = 'supermarket_sorting:client',
    [string]$ContainerPrefix = 'supermarket_sorting_official',
    [int]$ServerStartupSec = 90,
    [int]$ServerReadySec = 240,
    [string]$UseGs = '1',
    [string]$Seed = '11',
    [string]$ObstacleSeed = '11',
    [int]$TaskCount = 5,
    [switch]$KeepContainers
)

# Windows PowerShell 5.1 promotes a native command's stderr to a TERMINATING
# error when ErrorActionPreference is 'Stop' (docker writes routine notices such
# as "No such container" there), which killed this script on its very first
# cleanup call.  Use 'Continue' and check $LASTEXITCODE explicitly at every step
# that matters instead.
$ErrorActionPreference = 'Continue'

$logDirHost = Join-Path $HostRoot 'logs_official'
$logDirContainer = "$Root/logs_official"
$serverContainer = "${ContainerPrefix}_server"
$clientContainer = "${ContainerPrefix}_client"

function Write-Step($msg) { Write-Host "[official_test] $msg" }
function Write-Err($msg) { Write-Host "[official_test] $msg" -ForegroundColor Red }

function Remove-RunContainers {
    docker rm -f $serverContainer $clientContainer 2>$null | Out-Null
}

function Test-ContainerRunning([string]$name) {
    $state = docker inspect -f '{{.State.Running}}' $name 2>$null
    return ($state -eq 'true')
}

function Copy-ClientLog([string]$name) {
    docker cp "${clientContainer}:$logDirContainer/$name" (Join-Path $logDirHost $name) 2>$null | Out-Null
}

# --- guards: mirror the bash script's refusal to run a non-formal test -------
if ($env:SUPERMARKET_ALLOW_RUNTIME_LAYOUT -eq '1' -or $env:SUPERMARKET_TEST_ORACLE -eq '1') {
    Write-Err 'refusing to run with development truth enabled'
    exit 2
}
if ($env:SUPERMARKET_DETECT_BACKEND -eq 'gt') {
    Write-Err 'refusing the development-only gt detector'
    exit 2
}

New-Item -ItemType Directory -Force -Path $logDirHost | Out-Null
Remove-RunContainers
# NOTE: no PowerShell.Exiting handler here.  Event-action script blocks do not
# capture the enclosing scope's variables, so the usual
# `Register-EngineEvent -Action { docker rm -f $serverContainer }` silently
# removes nothing.  Cleanup is explicit on every exit path below, and
# Remove-RunContainers at the top of the next run mops up after a hard kill.

$serverArgs = @(
    'run', '-dit', '--gpus', 'all', '--network', 'host', '--ipc', 'host',
    '--name', $serverContainer,
    '-e', 'ROS_DOMAIN_ID=99',
    '-e', 'RMW_IMPLEMENTATION=rmw_cyclonedds_cpp',
    '-e', 'MUJOCO_GL=egl',
    '-e', 'SUPERMARKET_HEADLESS=1',
    '-e', 'SUPERMARKET_ENABLE_RENDER=1',
    '-e', "SUPERMARKET_USE_GS=$UseGs",
    '-e', 'TORCH_CUDA_ARCH_LIST=8.9',
    '-e', 'TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions',
    '-e', 'SUPERMARKET_ENABLE_SCORE=1',
    '-e', 'SUPERMARKET_ENABLE_LIDAR=1',
    '-e', 'SUPERMARKET_RANDOMIZE=1',
    '-e', 'SUPERMARKET_RANDOMIZE_OBSTACLES=1',
    '-e', "SUPERMARKET_SEED=$Seed",
    '-e', "SUPERMARKET_OBSTACLE_SEED=$ObstacleSeed",
    '-e', "SUPERMARKET_TASK_COUNT=$TaskCount",
    '-e', 'SUPERMARKET_TASK_ANONYMOUS=1',
    '-e', "SUPERMARKET_TARGETS=$($env:SUPERMARKET_TARGETS)",
    '-e', 'SUPERMARKET_RENDER_FPS=6',
    '-e', 'SUPERMARKET_RENDER_WIDTH=640',
    '-e', 'SUPERMARKET_RENDER_HEIGHT=480',
    '-v', "${HostRoot}:$Root",
    '-v', 'supermarket_sorting_cache:/root/.cache',
    $ServerImage,
    'bash', '-lc', "cd $Root && ./scripts/run_v2_server.sh"
)

Write-Step "starting server container (USE_GS=$UseGs seed=$Seed obstacle_seed=$ObstacleSeed)"
& docker @serverArgs | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Err 'failed to start server container'; exit 1 }

Write-Step "waiting for server task publication (up to ${ServerStartupSec}s)"
$taskProbe = Join-Path $logDirHost 'task_probe.log'
Remove-Item -Force $taskProbe -ErrorAction SilentlyContinue
$deadline = (Get-Date).AddSeconds($ServerStartupSec)
$published = $false
while ((Get-Date) -lt $deadline) {
    if (-not (Test-ContainerRunning $serverContainer)) {
        Write-Err 'server exited before publishing a task'
        docker logs $serverContainer 2>&1 | Select-Object -Last 40 | Write-Host
        exit 1
    }
    $hit = docker logs $serverContainer 2>&1 | Select-String -SimpleMatch '[server] task published:' | Select-Object -Last 1
    if ($hit) {
        $hit.Line | Set-Content -Path $taskProbe -Encoding utf8
        Write-Step 'server task handshake passed'
        $published = $true
        break
    }
    Start-Sleep -Seconds 1
}
if (-not $published) {
    Write-Err 'server did not publish /supermarket_sorting/task'
    docker logs $serverContainer 2>&1 | Select-Object -Last 60 | Write-Host
    exit 1
}

Write-Step "waiting for live odometry (up to ${ServerReadySec}s)"
$readyProbe = Join-Path $logDirHost 'odom_probe.log'
Remove-Item -Force $readyProbe -ErrorAction SilentlyContinue
$readyDeadline = (Get-Date).AddSeconds($ServerReadySec)
$odomOk = $false
while ((Get-Date) -lt $readyDeadline) {
    $out = docker exec $serverContainer bash -lc "source /opt/ros/humble/setup.bash && timeout 5 ros2 topic echo /slamware_ros_sdk_server_node/odom --once" 2>&1
    if ($out -match 'pose:') { $odomOk = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $odomOk) {
    Write-Err 'server did not publish live odometry'
    docker logs $serverContainer 2>&1 | Select-Object -Last 60 | Write-Host
    exit 1
}
Write-Step 'server odometry handshake passed'

$clientArgs = @(
    'run', '-dit', '--gpus', 'all', '--network', 'host', '--ipc', 'host',
    '--name', $clientContainer,
    '-e', 'ROS_DOMAIN_ID=99',
    '-e', 'RMW_IMPLEMENTATION=rmw_cyclonedds_cpp',
    '-e', 'SUPERMARKET_ORDER=official',
    '-e', 'SUPERMARKET_ALLOW_RUNTIME_LAYOUT=0',
    '-e', 'SUPERMARKET_DETECT_BACKEND=yolo',
    '-e', "SUPERMARKET_YOLO_WEIGHTS=$Root/examples/supermarket_sorting/perception/checkpoints/supermarket_multiclass.pt",
    '-e', 'SUPERMARKET_YOLO_REQUIRE_OFFICIAL_CLASSES=1',
    '-e', 'SUPERMARKET_YOLO_DEVICE=auto',
    '-e', 'SUPERMARKET_ENABLE_AVOIDANCE=1',
    '-e', 'SUPERMARKET_ENABLE_DEPTH_AVOIDANCE=1',
    '-e', 'SUPERMARKET_DELIVERY_USE_ASTAR=1',
    '-e', 'SUPERMARKET_TEST_ORACLE=0',
    '-e', 'SUPERMARKET_REQUEST_SERVER_RESET=0',
    '-e', 'SUPERMARKET_TASK_WAIT_TIMEOUT=60',
    '-e', 'SUPERMARKET_SEARCH_CLASS_MIN_SAMPLES=3',
    '-e', 'SUPERMARKET_SEARCH_CLASS_MIN_RATIO=0.67',
    '-e', "SUPERMARKET_MISSION_METRICS=$Root/examples/supermarket_sorting/mission_metrics.jsonl",
    '-v', "${HostRoot}:$Root",
    '-v', 'supermarket_sorting_cache:/root/.cache',
    $ClientImage,
    # Keep-alive.  The bash runner writes this as
    #     bash -lc 'trap "exit 0" TERM INT; while true; do sleep 3600; done'
    # but Windows PowerShell 5.1 strips the inner double quotes when it builds
    # the native command line, so bash received `trap exit` and the container
    # exited immediately (verified: Cmd became
    # ["bash","-lc","trap exit","0 TERM INT; ..."]).  No inner quotes here, and
    # `docker rm -f` is used for teardown so no TERM trap is needed.
    'bash', '-lc', 'while true; do sleep 3600; done'
)

Write-Step 'starting client container'
& docker @clientArgs | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Err 'failed to start client container'; exit 1 }

docker exec -d $clientContainer bash -lc "cd $Root && mkdir -p $logDirContainer && SUPERMARKET_ALLOW_RUNTIME_LAYOUT=0 SUPERMARKET_DETECT_BACKEND=yolo ./scripts/run_v2_perception.sh > $logDirContainer/perception.log 2>&1"

Write-Step 'waiting for first perception heartbeat (up to 90s)'
$perceptionReady = $false
$perceptionDeadline = (Get-Date).AddSeconds(90)
while ((Get-Date) -lt $perceptionDeadline) {
    docker exec $clientContainer bash -lc "source /opt/ros/humble/setup.bash && timeout 4 ros2 topic echo /supermarket_sorting/detections --once" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { $perceptionReady = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $perceptionReady) {
    Write-Err 'perception published no heartbeat'
    Copy-ClientLog 'perception.log'
    docker exec $clientContainer bash -lc "cat $logDirContainer/perception.log" 2>&1 | Write-Host
    Write-Err "client container state: $(docker inspect -f '{{.State.Status}}' $clientContainer 2>$null)"
    docker logs $clientContainer 2>&1 | Select-Object -Last 20 | Write-Host
    exit 1
}
Write-Step 'perception heartbeat passed'

docker exec -d $clientContainer bash -lc "cd $Root && mkdir -p $logDirContainer && ./scripts/run_v2_decision_client.sh > $logDirContainer/decision_client.log 2>&1"

Write-Step "official test running for ${DurationSec}s; logs: $logDirHost"
$runDeadline = (Get-Date).AddSeconds($DurationSec)
while ((Get-Date) -lt $runDeadline) {
    if (-not (Test-ContainerRunning $serverContainer)) {
        Write-Err 'server exited during run; stopping early'
        docker logs $serverContainer > (Join-Path $logDirHost 'server.log') 2>&1
        Copy-ClientLog 'perception.log'
        Copy-ClientLog 'decision_client.log'
        exit 1
    }
    if (-not (Test-ContainerRunning $clientContainer)) {
        Write-Err 'client exited during run; stopping early'
        docker logs $serverContainer > (Join-Path $logDirHost 'server.log') 2>&1
        exit 1
    }
    Start-Sleep -Seconds 2
}

Write-Step 'run window finished; collecting logs'
docker logs $serverContainer > (Join-Path $logDirHost 'server.log') 2>&1
Copy-ClientLog 'perception.log'
Copy-ClientLog 'decision_client.log'
docker exec $clientContainer bash -lc "cd $Root && python3 examples/supermarket_sorting/analyze_mission_metrics.py" 2>&1 |
    Tee-Object -FilePath (Join-Path $logDirHost 'mission_summary.txt')

$refereeLog = Join-Path $logDirHost 'referee_final_state.log'
docker exec $serverContainer bash -lc "source /opt/ros/humble/setup.bash && timeout 8 ros2 topic echo /referee/state --once" > $refereeLog 2>&1
$completed = 0
$m = Select-String -Path $refereeLog -Pattern '"completed"\s*:\s*(\d+)' | Select-Object -First 1
if ($m) { $completed = [int]$m.Matches[0].Groups[1].Value }
$score = 0
$ms = Select-String -Path $refereeLog -Pattern '"score"\s*:\s*(-?\d+)' | Select-Object -First 1
if ($ms) { $score = [int]$ms.Matches[0].Groups[1].Value }

Write-Step "referee completed=$completed/$TaskCount score=$score"

if (-not $KeepContainers) { Remove-RunContainers }

if ($completed -lt $TaskCount) {
    Write-Err 'incomplete formal run; refusing success status'
    exit 1
}
Write-Step 'formal run complete'
exit 0
