param(
    [int]$PollSeconds = 30,
    [switch]$Once,
    [switch]$Cached,
    [switch]$ObserveOnly,
    [switch]$SettleOnly,
    [switch]$Report
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "C:\AA_Projects\wc-predictor\.venv\Scripts\python.exe"
$Slate = Join-Path $ProjectRoot "outputs\us_open_2026_latest_slate.json"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python runtime not found at $Python"
}
if (-not $SettleOnly -and -not $Report -and -not (Test-Path -LiteralPath $Slate)) {
    throw "Prediction slate missing. Run .\scripts\predict_all.ps1 first."
}

$Arguments = @(
    "-m", "us_open_model.cli", "agent",
    "--slate", $Slate,
    "--poll-seconds", $PollSeconds
)
if (-not $Once -and -not $SettleOnly -and -not $Report) {
    $Arguments += "--watch"
}
if ($Cached) {
    $Arguments += "--cached"
}
if ($ObserveOnly) {
    $Arguments += "--observe-only"
}
if ($SettleOnly -or $Report) {
    $Arguments += "--settle-only"
}
if ($Report) {
    $Arguments += "--report"
}

Push-Location $ProjectRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Trading agent failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
