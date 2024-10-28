@echo off
python.exe -m pip install -r requirements.txt

IF %ERRORLEVEL% NEQ 0 (
    echo Failed to install requirements.
    exit /b 1
)


start "" "Installer\installer.exe"
echo Installation and execution completed.
pause
