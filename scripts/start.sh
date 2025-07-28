#!/bin/bash

# Start the application with Docker Compose
echo "Starting Brev-Hack application..."

# Build and start services
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

echo "Application started!"
echo "Backend: http://localhost:3000"
echo "Frontend: http://localhost:3001" 