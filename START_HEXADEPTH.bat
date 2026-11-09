@echo off
setlocal
cd /d "%~dp0"
echo.
echo  HexaDepth Priority-First prototype
echo  Folder: %CD%
echo.
python --version
if errorlevel 1 (
  echo Python could not be found. Install Python 3.11 or newer and add it to PATH.
  pause
  exit /b 1
)
if not exist "requirements.txt" (
  echo requirements.txt missing; check that the ZIP was fully extracted.
  pause
  exit /b 1
)
echo Installing project requirements...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency install failed. Copy the error output for troubleshooting.
  pause
  exit /b 1
)
echo.
echo Starting HexaDepth at http://localhost:8000
python -m uvicorn server:app --host 127.0.0.1 --port 8000
pause
