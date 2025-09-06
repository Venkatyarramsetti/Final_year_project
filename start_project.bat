@echo off

echo Starting Hazard Spotter AI project...

echo.
echo Step 1: Starting the backend server
echo ===================================
start cmd /k "cd backend && start_server.bat"

echo.
echo Step 2: Starting the frontend development server
echo ===============================================
start cmd /k "cd frontend && npm run dev"

echo.
echo Hazard Spotter AI is now starting!
echo.
echo * Backend API: http://localhost:8000
echo * Frontend: Check console for URL (typically http://localhost:8081)
echo.
echo Press any key to exit this launcher (servers will continue running)...
pause > nul
