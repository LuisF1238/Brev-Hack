#!/bin/bash

# Stop the application
echo "Stopping Brev-Hack application..."

docker-compose -f docker-compose.yml -f docker-compose.dev.yml down

echo "Application stopped!" 