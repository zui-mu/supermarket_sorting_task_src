# Launch the detector retrain (dataset over both render domains + training) in a
# throwaway SERVER-image container.
#
# Mirrors scripts/run_v2_retrain_detector.sh for hosts without a usable bash.
# Prerequisite: no other GPU job is running (the formal test containers must be
# stopped first, otherwise this contends for the same GPU).
#
#   pwsh -File scripts/run_v2_retrain_detector.ps1 -Frames 1400 -Epochs 60 -Batch 4
[CmdletBinding()]
param(
    [string]$HostRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$Root = '/workspace/baseline',
    [string]$ServerImage = 'supermarket_sorting:server',
    [string]$Container = 'retrain_detector',
    [int]$Frames = 1400,
    [int]$Variants = 1,
    [int]$Epochs = 60,
    [int]$Batch = 4
)

$ErrorActionPreference = 'Continue'

docker rm -f $Container 2>$null | Out-Null

$already = docker ps --format '{{.Names}}' 2>$null
foreach ($busy in @('supermarket_sorting_official_server', 'supermarket_sorting_official_client')) {
    if ($already -contains $busy) {
        Write-Host "[retrain] refusing to start: $busy is still running (GPU contention)" -ForegroundColor Red
        exit 2
    }
}

Write-Host "[retrain] frames=$Frames variants=$Variants epochs=$Epochs batch=$Batch"

$args = @(
    'run', '--rm', '--gpus', 'all', '--ipc', 'host', '--name', $Container,
    '-e', 'MUJOCO_GL=egl',
    '-e', 'PYOPENGL_PLATFORM=egl',
    '-e', 'TORCH_CUDA_ARCH_LIST=8.9',
    '-e', 'TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions',
    '-e', "RETRAIN_FRAMES=$Frames",
    '-e', "RETRAIN_VARIANTS=$Variants",
    '-e', "RETRAIN_EPOCHS=$Epochs",
    '-e', "RETRAIN_BATCH=$Batch",
    '-v', "${HostRoot}:$Root",
    '-v', 'supermarket_sorting_cache:/root/.cache',
    $ServerImage,
    'bash', "$Root/scripts/retrain_detector_in_container.sh"
)

& docker @args 2>&1
exit $LASTEXITCODE
