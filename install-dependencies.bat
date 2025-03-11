@echo off
REM For Windows

echo Installing Windows dependencies...
pip install -r requirements-windows.txt

REM Check if installation was successful
if %ERRORLEVEL% NEQ 0 (
    echo Error installing dependencies
    exit /b 1
)

echo Dependencies installed successfully 