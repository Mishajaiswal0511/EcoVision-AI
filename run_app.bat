@echo off
setlocal EnableDelayedExpansion
title EcoVision AI - Launcher

echo ============================================================
echo         EcoVision AI - Intelligent Waste Detection
echo                One-Click Windows Launcher
echo ============================================================
echo.

cd /d "%~dp0"

:: 1. Check Python Virtual Environment
echo [*] Checking Python virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Please create the virtual environment using Python 3.12:
    echo    uv venv --python 3.12 .venv
    echo    .venv\Scripts\pip install -r requirements-app.txt
    echo.
    pause
    exit /b 1
)
echo [OK] Virtual environment detected.

:: 2. Check Model Checkpoint
echo [*] Verifying YOLO11s V5 model checkpoint...
if not exist "models\experiments\v5_taxonomy\training\yolo11s_v5_best.pt" (
    echo [ERROR] Model checkpoint not found at models\experiments\v5_taxonomy\training\yolo11s_v5_best.pt
    echo Please place your trained YOLO weights at models\experiments\v5_taxonomy\training\yolo11s_v5_best.pt
    echo.
    pause
    exit /b 1
)
echo [OK] Model weights found at models\experiments\v5_taxonomy\training\yolo11s_v5_best.pt

:: 3. Check Frontend Dependencies
echo [*] Checking frontend dependencies...
if not exist "frontend\node_modules" (
    echo [*] Installing frontend dependencies in frontend\node_modules...
    pushd frontend
    call npm install
    if errorlevel 1 (
        popd
        echo [ERROR] npm install failed. Please ensure Node.js is installed.
        echo.
        pause
        exit /b 1
    )
    popd
)
echo [OK] Frontend dependencies verified.

:: 4. Launch FastAPI Backend in a new window
echo.
echo [*] Starting FastAPI backend (7-class V5 staging) on http://127.0.0.1:8000 ...
start "EcoVision AI - Backend API (FastAPI)" cmd /k "title EcoVision AI - Backend (Port 8000) && cd /d "%~dp0" && .venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir models/experiments/v5_taxonomy/preproduction/staging/backend"

:: Brief wait for backend initialization
ping 127.0.0.1 -n 4 >nul

:: 5. Launch Vite Frontend in a new window
echo [*] Starting Vite React frontend on http://127.0.0.1:5173 ...
start "EcoVision AI - Frontend UI (Vite)" cmd /k "title EcoVision AI - Frontend (Port 5173) && cd /d "%~dp0frontend" && npm run dev -- --host 127.0.0.1 --port 5173"

:: Brief wait for Vite server readiness
ping 127.0.0.1 -n 3 >nul

:: 6. Launch Default Browser
echo [*] Opening EcoVision AI in your default web browser...
start http://127.0.0.1:5173

echo.
echo ============================================================
echo   EcoVision AI is running!
echo.
echo   Frontend UI:     http://127.0.0.1:5173
echo   Backend API:     http://127.0.0.1:8000
echo   API Swagger UI:  http://127.0.0.1:8000/docs
echo ============================================================
echo.
echo   SHUTDOWN INSTRUCTIONS:
echo   To stop the application, simply close the two open server
echo   terminal windows or press Ctrl+C in each of them.
echo.
echo Press any key to exit this launcher window...
pause
