@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0PUSH_TO_GITHUB.ps1"
echo.
pause
