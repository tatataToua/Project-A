<#
Starts the whole dev stack: Postgres (docker), backend (uvicorn), frontend (vite),
each backend/frontend in its own new PowerShell window so you can see their logs
and Ctrl+C them independently.

Usage:  .\start-dev.ps1
#>

$root = $PSScriptRoot

Write-Host "Starting Postgres..." -ForegroundColor Cyan
docker compose -f "$root\docker-compose.yml" up -d

Write-Host "Starting backend (new window)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\backend'; .\.venv\Scripts\Activate.ps1; uvicorn app.main:app --reload --port 8000"
)

Write-Host "Starting frontend (new window)..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\frontend'; npm run dev"
)

Write-Host "Both launched in separate windows. Use Ctrl+C in each window to stop them, or run .\stop-dev.ps1 to force-stop both by port." -ForegroundColor Cyan
