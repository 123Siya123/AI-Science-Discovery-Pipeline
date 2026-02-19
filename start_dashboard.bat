@echo off
TITLE AI Science Discovery Team
echo 🔬 Starting AI Science Discovery Team...
cd /d "%~dp0"

:: Open the browser first
echo 🌍 Opening Dashboard at http://localhost:5050...
start "" "http://localhost:5050"

:: Start the Python application
echo 🚀 Running app.py...
python app.py

:: If the app stops, keep the window open to see errors
if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ The application stopped unexpectedly.
    pause
)
