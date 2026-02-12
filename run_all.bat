@echo off
setlocal

echo ===================================================
echo     AR Museum Guide - All-in-One Launcher
echo ===================================================
echo.

:: Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo Error: Virtual environment not found at venv\Scripts\python.exe
    echo Please make sure you are running this from the project root and venv is set up.
    pause
    exit /b 1
)

echo Starting servers...
echo.

:: 1. Start ML API (Port 8000)
echo [1/4] Starting ML API (Port 8000)...
start "AR Museum Guide - ML API" venv\Scripts\python.exe backend\ml_api.py

:: 2. Start Training API (Port 8001)
echo [2/4] Starting Training API (Port 8001)...
start "AR Museum Guide - Training API" venv\Scripts\python.exe backend\training_api.py

:: 3. Start QA API (Port 8002)
echo [3/4] Starting QA API (Port 8002)...
start "AR Museum Guide - QA API" venv\Scripts\python.exe backend\qa_api.py

:: 4. Start Web Server (Port 8080)
echo [4/4] Starting Web Server (Port 8080)...
start "AR Museum Guide - Web Server" venv\Scripts\python.exe run_ar_server.py

echo.
echo ===================================================
echo All servers launched in separate windows!
echo ===================================================
echo.
echo APIs:
echo - ML API:       http://localhost:8000/docs
echo - Training API: http://localhost:8001/docs
echo - QA API:       http://localhost:8002/docs
echo.
echo Web Interface:
echo - Mobile Demo:  http://localhost:8080
echo - Admin Panel:  http://localhost:8080/admin_panel.html
echo.
echo Waiting 5 seconds for servers to initialize...
timeout /t 5 >nul

echo Opening Admin Panel...
start http://localhost:8080/admin_panel.html

echo.
echo You can minimize this window, but do not close the other server windows.
echo To stop everything, simply close all the command windows.
echo.
pause
