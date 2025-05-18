#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Car Damage Detection System${NC}"
echo -e "${BLUE}=================================${NC}"

# Create output directories if they don't exist
mkdir -p ../output/car_damage_detection/uploads
mkdir -p ../output/car_damage_detection/results

# Function to cleanup processes on exit
cleanup() {
    echo -e "\n${GREEN}Stopping services...${NC}"
    kill $FLASK_PID $REACT_PID 2>/dev/null
    exit 0
}

# Trap SIGINT (ctrl+c) and call cleanup
trap cleanup INT

# Start Flask backend with car_damage_env
echo -e "\n${GREEN}Starting Flask backend...${NC}"
cd ..
source car_damage_env/bin/activate && python app.py &
FLASK_PID=$!

# Wait for Flask to start
sleep 2

# Start React frontend in development mode
echo -e "\n${GREEN}Starting React frontend...${NC}"
cd car-damage-ui && npm start &
REACT_PID=$!

echo -e "\n${GREEN}All services started!${NC}"
echo -e "- Flask API: http://localhost:5000"
echo -e "- React App: http://localhost:3000"
echo -e "\n${BLUE}Press Ctrl+C to stop all services${NC}"

# Wait for background processes to finish
wait 