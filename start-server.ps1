Write-Host "Starting local web server..." -ForegroundColor Green
Write-Host ""
Write-Host "Your website will be available at: " -NoNewline
Write-Host "http://localhost:8000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

Set-Location $PSScriptRoot
python -m http.server 8000