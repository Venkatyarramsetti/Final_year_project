#!/bin/bash

echo "Starting Hazard Spotter AI project..."

echo ""
echo "Step 1: Starting the backend server"
echo "==================================="
gnome-terminal -- bash -c "cd backend && bash start_server.sh" &

echo ""
echo "Step 2: Starting the frontend development server"
echo "==============================================="
gnome-terminal -- bash -c "cd frontend && npm run dev" &

echo ""
echo "Hazard Spotter AI is now starting!"
echo ""
echo "* Backend API: http://localhost:8000"
echo "* Frontend: Check console for URL (typically http://localhost:8081)"
echo ""
echo "Press any key to exit this launcher (servers will continue running)..."
read -n 1 -s
