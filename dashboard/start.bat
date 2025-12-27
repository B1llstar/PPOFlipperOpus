@echo off
REM Quick start script for PPO Flipper Dashboard (Windows)

echo ================================================
echo PPO Flipper Training Dashboard
echo ================================================
echo.

REM Check if ge_prices.db exists
if not exist "..\ge_prices.db" (
    echo X Error: ge_prices.db not found in project root
    echo Please run data collection first:
    echo   cd ..\data
    echo   python collect_all.py
    pause
    exit /b 1
)

echo - Database found
echo.

REM Check backend dependencies
echo Checking backend dependencies...
cd backend
python -c "import fastapi, uvicorn, torch" 2>nul
if errorlevel 1 (
    echo Warning - Missing backend dependencies. Installing...
    pip install -r requirements.txt
)

REM Check frontend dependencies
echo Checking frontend dependencies...
cd ..\frontend
if not exist "node_modules" (
    echo Warning - Missing frontend dependencies. Installing...
    call npm install
)

echo.
echo ================================================
echo Starting services...
echo ================================================
echo.

REM Start backend in new window
echo Starting backend on http://localhost:8000 ...
cd ..\backend
start "PPO Backend" cmd /k python server.py

REM Wait for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend in current window
echo Starting frontend...
cd ..\frontend
call npm run dev

pause
