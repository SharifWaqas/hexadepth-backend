@echo off
setlocal
cd /d "%~dp0"
echo Installing optional AI detector dependencies...
python -m pip install -r requirements-yolo.txt
if errorlevel 1 (
  echo Could not install YOLO dependencies.
  pause
  exit /b 1
)
python -m pip install --upgrade huggingface_hub
if errorlevel 1 (
  echo Could not install Hugging Face client.
  pause
  exit /b 1
)
echo Downloading your private model checkpoint (local authentication may be needed)...
python scripts\download_sonar_model.py
if errorlevel 1 (
  echo Could not install checkpoint. Check Hugging Face login and bucket permissions.
  pause
  exit /b 1
)
echo.
echo AI model installed. Launch START_HEXADEPTH.bat, then visit http://localhost:8000/api/model/status
pause
