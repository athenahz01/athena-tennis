param(
    [int]$Simulations = 20000,
    [string]$Fixtures = "data\fixtures\us_open_2026_2026-09-01.csv",
    [switch]$NoKalshi
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "C:\AA_Projects\wc-predictor\.venv\Scripts\python.exe"
$Artifact = Join-Path $ProjectRoot "artifacts\uso_hybrid_v1.joblib"
$FixturePath = Join-Path $ProjectRoot $Fixtures
$Output = Join-Path $ProjectRoot "outputs\us_open_2026_latest_slate.json"
$WebOutput = Join-Path $ProjectRoot "web\public\data\slate.json"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python runtime not found at $Python"
}
if (-not (Test-Path -LiteralPath $Artifact)) {
    throw "Model artifact missing. Run the training command in README.md first."
}
if (-not (Test-Path -LiteralPath $FixturePath)) {
    throw "Fixture slate not found: $FixturePath"
}

$Arguments = @(
    "-m", "us_open_model.cli", "slate",
    "--fixtures", $FixturePath,
    "--artifact", $Artifact,
    "--sims", $Simulations,
    "--output", $Output,
    "--web-output", $WebOutput
)
if (-not $NoKalshi) {
    $Arguments += "--kalshi"
}

Push-Location $ProjectRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Prediction command failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "All predictions complete."
Write-Host "Website data: $WebOutput"
Write-Host "Start the website with: cd web; npm.cmd run dev"
