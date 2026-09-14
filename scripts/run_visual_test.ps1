<#
.SYNOPSIS
    Live visual test of the retail scene on a real X window, with no time limit.

.DESCRIPTION
    This is a WATCHING harness, not a formal entry.  The formal entry stays
    scripts/run_v2_official_test.sh (+) its PowerShell port; those refuse the
    gt detector and the runtime layout on purpose.  This script deliberately
    enables both, because a live window cannot afford the renderer the real
    detector needs:

      measured (scripts/probe_render_speed.py, window path via VcXsrv):
        MuJoCo rasteriser (SUPERMARKET_USE_GS=0)  0.19 s/frame  ->  5.3 fps
        3DGS              (SUPERMARKET_USE_GS=1)  not usable on software GL
                                                  (also needs a Hugging Face
                                                  fetch that fails offline)

    The shipped checkpoint is only accurate with 3DGS (8/10 vs 0/10, see
    scripts/probe_yolo_live_pose.py), so with the rasteriser this run uses the
    gt detector and the runtime layout.  What it therefore validates is the
    NAVIGATION, GRASP, CARRY, DELIVER and PLACE pipeline and the referee
    scoring - which is exactly what the recent fixes touched.  It does NOT
    validate the detector; use the formal runner for that.

.PREREQUISITE
    An X server on the Windows host.  VcXsrv is used here:
      & "C:\Program Files\VcXsrv\vcxsrv.exe" :0 -ac -multiwindow -listen tcp `
          -screen 0 1600x900x24
    It must be running before this script, and it binds TCP :6000.

.EXAMPLE
    pwsh -File scripts/run_visual_test.ps1
    pwsh -File scripts/run_visual_test.ps1 -Targets "kele" -Anonymous 0
#>
[CmdletBinding()]
param(
    [string]$HostRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$Root = '/workspace/baseline',
    [string]$ServerImage = 'supermarket_sorting:server',
    [string]$ClientImage = 'supermarket_sorting:client',
    [string]$ServerContainer = 'visual_server',
    [string]$ClientContainer = 'visual_client',
    [string]$LogDir = 'logs_visual',
    [string]$Display = 'host.docker.internal:0',
    [string]$Seed = '11',
    [string]$ObstacleSeed = '11',
    [int]$TaskCount = 5,
    # '' = let the referee take the first five slots (the real official task)
    [string]$Targets = '',
    [int]$Anonymous = 0,
    [int]$RenderFps = 10,
    [switch]$Stop
)

$ErrorActionPreference = 'Continue'

function Write-Step($m) { Write-Host "[visual] $m" }

if ($Stop) {
    docker rm -f $ServerContainer $ClientContainer 2>$null | Out-Null
    Write-Step 'stopped'
    exit 0
}

function Test-ContainerRunning([string]$name) {
    return ((docker inspect -f '{{.State.Running}}' $name 2>$null) -eq 'true')
}

$logPath = Join-Path $HostRoot $LogDir
New-Item -ItemType Directory -Force -Path $logPath | Out-Null

# --- an X display must exist, otherwise the sim silently renders nowhere ----
$tcp = Test-NetConnection -ComputerName 127.0.0.1 -Port 6000 -WarningAction SilentlyContinue
if (-not $tcp.TcpTestSucceeded) {
    Write-Host "[visual] no X server on TCP :6000." -ForegroundColor Red
    Write-Host "[visual] start VcXsrv first:" -ForegroundColor Red
    Write-Host '  & "C:\Program Files\VcXsrv\vcxsrv.exe" :0 -ac -multiwindow -listen tcp -screen 0 1600x900x24' -ForegroundColor Red
    exit 2
}
Write-Step "X server detected on :6000, DISPLAY=$Display"

docker rm -f $ServerContainer $ClientContainer 2>$null | Out-Null

$serverArgs = @(
    'run', '-dit', '--gpus', 'all', '--network', 'host', '--ipc', 'host',
    '--name', $ServerContainer,
    '-e', "DISPLAY=$Display",
    '-e', 'MUJOCO_GL=glfw',
    '-e', 'PYOPENGL_PLATFORM=glx',
    '-e', 'ROS_DOMAIN_ID=99',
    '-e', 'RMW_IMPLEMENTATION=rmw_cyclonedds_cpp',
    '-e', 'SUPERMARKET_HEADLESS=0',
    '-e', 'SUPERMARKET_ENABLE_RENDER=1',
    '-e', 'SUPERMARKET_USE_GS=0',
    '-e', 'SUPERMARKET_RENDER_ALL_CAMERAS=1',
    '-e', 'SUPERMARKET_ENABLE_SCORE=1',
    '-e', 'SUPERMARKET_ENABLE_LIDAR=1',
    '-e', 'SUPERMARKET_RANDOMIZE=1',
    '-e', 'SUPERMARKET_RANDOMIZE_OBSTACLES=1',
    '-e', "SUPERMARKET_SEED=$Seed",
    '-e', "SUPERMARKET_OBSTACLE_SEED=$ObstacleSeed",
    '-e', "SUPERMARKET_TASK_COUNT=$TaskCount",
    '-e', "SUPERMARKET_TASK_ANONYMOUS=$Anonymous",
    '-e', "SUPERMARKET_TARGETS=$Targets",
    '-e', "SUPERMARKET_RENDER_FPS=$RenderFps",
    '-e', 'SUPERMARKET_RENDER_WIDTH=640',
    '-e', 'SUPERMARKET_RENDER_HEIGHT=480',
    '-v', "${HostRoot}:$Root",
    '-v', 'supermarket_sorting_cache:/root/.cache',
    $ServerImage,
    'bash', '-lc', "cd $Root && ./scripts/run_v2_server.sh"
)

Write-Step "starting server (window on $Display, rasteriser, seed=$Seed)"
& docker @serverArgs | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host '[visual] server container failed to start' -ForegroundColor Red; exit 1 }

Write-Step 'waiting for task publication (up to 120s)'
$deadline = (Get-Date).AddSeconds(120)
$published = $false
while ((Get-Date) -lt $deadline) {
    if (-not (Test-ContainerRunning $ServerContainer)) {
        Write-Host '[visual] server exited during startup:' -ForegroundColor Red
        docker logs $ServerContainer 2>&1 | Select-Object -Last 30 | Write-Host
        exit 1
    }
    if (docker logs $ServerContainer 2>&1 | Select-String -SimpleMatch '[server] task published:') {
        $published = $true
        break
    }
    Start-Sleep -Seconds 1
}
if (-not $published) {
    Write-Host '[visual] server never published a task' -ForegroundColor Red
    docker logs $ServerContainer 2>&1 | Select-Object -Last 40 | Write-Host
    exit 1
}
Write-Step 'task published; the scene window should be visible now'

Write-Step 'waiting for live odometry (up to 240s)'
$odomDeadline = (Get-Date).AddSeconds(240)
$odomOk = $false
while ((Get-Date) -lt $odomDeadline) {
    $out = docker exec $ServerContainer bash -lc "source /opt/ros/humble/setup.bash && timeout 5 ros2 topic echo /slamware_ros_sdk_server_node/odom --once" 2>&1
    if ($out -match 'pose:') { $odomOk = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $odomOk) { Write-Host '[visual] no odometry' -ForegroundColor Red; exit 1 }
Write-Step 'odometry live'

$clientArgs = @(
    'run', '-dit', '--gpus', 'all', '--network', 'host', '--ipc', 'host',
    '--name', $ClientContainer,
    '-e', 'ROS_DOMAIN_ID=99',
    '-e', 'RMW_IMPLEMENTATION=rmw_cyclonedds_cpp',
    '-e', 'SUPERMARKET_ORDER=official',
    '-e', "SUPERMARKET_TASK_FALLBACK_ORDER=$($env:SUPERMARKET_TASK_FALLBACK_ORDER)",
    '-e', 'SUPERMARKET_DETECT_BACKEND=gt',
    '-e', 'SUPERMARKET_ALLOW_RUNTIME_LAYOUT=1',
    '-e', 'SUPERMARKET_STATIC_LAYOUT_ASSOCIATION=0',
    '-e', 'SUPERMARKET_DIRECT_TASK_GEOMETRY_FALLBACK=1',
    '-e', 'SUPERMARKET_TEST_ORACLE=1',
    '-e', 'SUPERMARKET_ENABLE_AVOIDANCE=1',
    '-e', 'SUPERMARKET_ENABLE_DEPTH_AVOIDANCE=0',
    '-e', 'SUPERMARKET_DELIVERY_USE_ASTAR=1',
    '-e', 'SUPERMARKET_CARRY_TUCK_ENABLED=0',
    '-e', 'SUPERMARKET_WRIST_SERVO=1',
    '-e', "SUPERMARKET_MISSION_METRICS=$Root/$LogDir/mission_metrics.jsonl",
    '-v', "${HostRoot}:$Root",
    '-v', 'supermarket_sorting_cache:/root/.cache',
    $ClientImage,
    # no inner quotes: Windows PowerShell 5.1 strips them when building the
    # native command line and bash then receives a different command.
    'bash', '-lc', 'while true; do sleep 3600; done'
)
& docker @clientArgs | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Host '[visual] client container failed to start' -ForegroundColor Red; exit 1 }

docker exec -d $ClientContainer bash -lc "cd $Root && mkdir -p $LogDir && SUPERMARKET_ALLOW_RUNTIME_LAYOUT=1 SUPERMARKET_DETECT_BACKEND=gt ./scripts/run_v2_perception.sh > $Root/$LogDir/perception.log 2>&1"
Start-Sleep -Seconds 6
docker exec -d $ClientContainer bash -lc "cd $Root && ./scripts/run_v2_decision_client.sh > $Root/$LogDir/decision_client.log 2>&1"

Write-Step 'RUNNING - no time limit. Both containers stay up until stopped.'
Write-Step "logs: $logPath"
Write-Step "stop with: pwsh -File scripts/run_visual_test.ps1 -Stop"
exit 0
