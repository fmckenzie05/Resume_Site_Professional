Write-Host "Starting local web server..." -ForegroundColor Green
Write-Host ""

Set-Location $PSScriptRoot

# Try to find Python installation
$pythonCmd = $null

# Try python3 first
if (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
    Write-Host "Using python3..." -ForegroundColor Green
}
# Try python
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
    Write-Host "Using python..." -ForegroundColor Green
}
# Try py launcher
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py -3"
    Write-Host "Using py launcher..." -ForegroundColor Green
}

if ($pythonCmd) {
    Write-Host "Your website will be available at: " -NoNewline
    Write-Host "http://localhost:8000" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
    Write-Host ""
    
    # Start the server bound to localhost
    & $pythonCmd -m http.server 8000 --bind 127.0.0.1
}
else {
    Write-Host "ERROR: Python is not installed or not in PATH!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3 from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}