param(
    [string]$Python = "",
    [switch]$InstallPython,
    [switch]$UpgradePackagingTools
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Find-Python {
    if ($Python) {
        return $Python
    }

    $LocalPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
    if (Test-Path $LocalPython) {
        return $LocalPython
    }

    $Command = Get-Command python -ErrorAction SilentlyContinue
    if ($Command -and ($Command.Source -notlike "*\Microsoft\WindowsApps\python.exe")) {
        return $Command.Source
    }

    return ""
}

$PythonExe = Find-Python

if (-not $PythonExe -and $InstallPython) {
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements --scope user
    $PythonExe = Find-Python
}

if (-not $PythonExe) {
    throw "No runnable CPython found. Re-run with -InstallPython or pass -Python C:\Path\To\python.exe."
}

Write-Host "Using Python: $PythonExe"
& $PythonExe --version

if (-not (Test-Path $VenvPython)) {
    & $PythonExe -m venv $VenvDir
}

if ($UpgradePackagingTools) {
    & $VenvPython -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
}

& $VenvPython -m pip install --disable-pip-version-check -r (Join-Path $ProjectRoot "requirements.txt")

& $VenvPython -c "import cv2, numpy; from importlib.metadata import version; print('OpenCV', cv2.__version__); print('NumPy', numpy.__version__); print('Flask', version('flask'))"

Write-Host ""
Write-Host "Environment ready."
Write-Host "Activate with:"
Write-Host "  cd $ProjectRoot"
Write-Host "  .\.venv\Scripts\Activate.ps1"
