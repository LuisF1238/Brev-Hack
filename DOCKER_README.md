# Docker Compose Setup

This project uses Docker Compose to orchestrate the backend Python service and frontend Next.js application.

## Services

- **Backend**: Python Flask application with AI models (Whisper, DialoGPT) running on port 3000
- **Frontend**: Next.js application running on port 3001

## Quick Start

### Using the provided scripts:

```bash
# Start the application
./scripts/start.sh

# Stop the application  
./scripts/stop.sh
```

### Manual Docker Compose commands:

```bash
# Start services in development mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Start services in detached mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

# Stop services
docker-compose -f docker-compose.yml -f docker-compose.dev.yml down

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

## Access Points

- **Backend API**: http://localhost:3000
- **Frontend**: http://localhost:3001
- **Backend Health Check**: http://localhost:3000/health

## Development Features

- Hot reload for both frontend and backend
- Volume mounts for live code changes
- Separate development environment configuration
- Network isolation between services

## Environment Variables

### Backend
- `PORT`: Server port (default: 3000)
- `SECRET_KEY`: Flask secret key
- `FLASK_ENV`: Flask environment (development/production)
- `FLASK_DEBUG`: Enable Flask debug mode

### Frontend
- `NODE_ENV`: Node environment (development/production)
- `NEXT_PUBLIC_API_URL`: Backend API URL for client-side requests

## Architecture

The services communicate through a Docker network called `app-network`. The frontend can reach the backend using the service name `backend` as the hostname.

## Troubleshooting

1. **Port conflicts**: Make sure ports 3000 and 3001 are not in use
2. **Build issues**: Try `docker-compose down` followed by `docker-compose up --build`
3. **Permission issues**: Ensure the scripts have execute permissions: `chmod +x scripts/*.sh`
4. **Volume issues**: Try removing volumes: `docker-compose down -v` 