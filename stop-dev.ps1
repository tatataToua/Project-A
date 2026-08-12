<#
Stops anything bound to the app's dev ports (backend :8000, frontend :5173),
including orphaned uvicorn/vite processes left behind from a terminal window
that was closed with the X instead of Ctrl+C.

Usage:  .\stop-dev.ps1
#>

$ports = 8000, 5173

foreach ($port in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        Write-Host "Port $port : nothing listening." -ForegroundColor DarkGray
        continue
    }
    foreach ($c in $conns) {
        $procId = $c.OwningProcess
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        $name = if ($proc) { $proc.ProcessName } else { "unknown" }
        Write-Host "Port $port : killing PID $procId ($name) and its child processes" -ForegroundColor Yellow
        taskkill /PID $procId /T /F | Out-Null
    }
}

Write-Host "Done. Postgres (docker) was left running — stop it separately with 'docker compose down' if you want it down too." -ForegroundColor Cyan
