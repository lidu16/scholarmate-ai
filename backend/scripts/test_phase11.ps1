$ErrorActionPreference = "Stop"
$backendRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $backendRoot "venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $python = $pythonCommand.Source
}

Push-Location $backendRoot
try {
    & $python -m compileall -q app tests
    if ($LASTEXITCODE -ne 0) {
        throw "Backend compilation failed with exit code $LASTEXITCODE."
    }

    & $python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        throw "Backend tests failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
