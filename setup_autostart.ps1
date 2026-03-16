# Setup-Autostart.ps1 — регистрирует watchdog в Task Scheduler
# Запуск: PowerShell -ExecutionPolicy Bypass -File setup_autostart.ps1

$PythonExe = "C:\Users\morco\AppData\Local\Programs\Python\Python312\python.exe"
$WatchdogScript = "C:\Users\morco\repo\ouroboros\watchdog.py"
$WorkDir = "C:\Users\morco\repo\ouroboros"

$action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "$WatchdogScript --loop" `
    -WorkingDirectory $WorkDir

$triggerLogon = New-ScheduledTaskTrigger -AtLogOn
$triggerBoot  = New-ScheduledTaskTrigger -AtStartup

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit 0 `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 2) `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName "OuroborosWatchdog" `
    -Action $action `
    -Trigger @($triggerLogon, $triggerBoot) `
    -Settings $settings `
    -RunLevel Highest `
    -Force | Out-Null

# Устанавливаем env vars на уровне пользователя
[System.Environment]::SetEnvironmentVariable("HTTPS_PROXY", "socks5://proxy_user:nmFZhByC9rNNOhz9@64.188.72.89:1080", "User")
[System.Environment]::SetEnvironmentVariable("HTTP_PROXY",  "socks5://proxy_user:nmFZhByC9rNNOhz9@64.188.72.89:1080", "User")
[System.Environment]::SetEnvironmentVariable("ALL_PROXY",   "socks5://proxy_user:nmFZhByC9rNNOhz9@64.188.72.89:1080", "User")

Write-Host ""
Write-Host "=== OuroborosWatchdog registered ===" -ForegroundColor Green
Get-ScheduledTask -TaskName "OuroborosWatchdog" | Select-Object TaskName, State
Write-Host ""
Write-Host "Env vars set: HTTPS_PROXY, HTTP_PROXY, ALL_PROXY" -ForegroundColor Cyan
Write-Host ""
Write-Host "Start now (no reboot needed):" -ForegroundColor Yellow
Write-Host "  Start-ScheduledTask -TaskName OuroborosWatchdog"
