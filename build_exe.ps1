param(
    [switch]$Install,
    [switch]$Clean,
    [string]$Name = "RescueVision",
    [string]$Entry = "vision_detect.py"
)

if ($Install) {
    Write-Host "Installing PyInstaller..."
    python -m pip install --upgrade pip
    python -m pip install pyinstaller
}

if ($Clean) {
    Write-Host "Cleaning previous build artifacts..."
    if (Test-Path .\build) { Remove-Item -Recurse -Force .\build }
    if (Test-Path .\dist) { Remove-Item -Recurse -Force .\dist }
    if (Test-Path .\$Name.spec) { Remove-Item -Force .\$Name.spec }
}

Write-Host "Building $Name from $Entry"
pyinstaller --onefile --clean --name $Name $Entry

Write-Host "Done. EXE is in dist\\$Name.exe"
