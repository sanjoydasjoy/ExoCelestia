# Docker Deployment Guide

## Quick Start

### Development

```bash
# Start all services
docker-compose up

# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

Access:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Redis: localhost:6379

### Production

```bash
# Set environment variables
cp .env.example .env
# Edit .env with production values

# Build and start
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop
docker-compose -f docker-compose.prod.yml down
```

## Services

### Backend (Python 3.11)
- **Base Image**: `python:3.11-slim`
- **Port**: 8000
- **Framework**: FastAPI with Uvicorn
- **Features**:
  - Health checks every 30s
  - Non-root user for security
  - Model directory mounted from `ml/models`
  - Redis connection for caching

### Frontend (Node.js 18)
- **Base Image**: `node:18-alpine`
- **Port**: 3000
- **Framework**: Next.js
- **Features**:
  - Multi-stage build for optimization
  - Health checks every 30s
  - Non-root user for security
  - Environment-based API URL

### Redis
- **Base Image**: `redis:7-alpine`
- **Port**: 6379
- **Purpose**: Caching and session storage
- **Features**:
  - Persistent data with volumes
  - Health checks every 10s
  - Password protection in production

## Commands

### Build Images

```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build backend

# Build with no cache
docker-compose build --no-cache
```

### Start/Stop Services

```bash
# Start all
docker-compose up

# Start specific service
docker-compose up backend

# Start in background
docker-compose up -d

# Stop all
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100
```

### Execute Commands

```bash
# Backend shell
docker-compose exec backend bash

# Run Python command
docker-compose exec backend python -c "print('Hello')"

# Frontend shell
docker-compose exec frontend sh

# Redis CLI
docker-compose exec redis redis-cli
```

### Database/Cache Management

```bash
# Clear Redis cache
docker-compose exec redis redis-cli FLUSHALL

# Backup Redis data
docker-compose exec redis redis-cli SAVE

# Check Redis keys
docker-compose exec redis redis-cli KEYS '*'
```

## Development Workflow

### Hot Reload

Both backend and frontend support hot reload in development:

**Backend**: Changes to Python files trigger automatic reload (uvicorn `--reload`)

**Frontend**: Changes to React/TypeScript files trigger fast refresh (Next.js dev)

### Adding Dependencies

**Backend**:
```bash
# Add to requirements.txt
echo "new-package==1.0.0" >> backend/requirements.txt

# Rebuild
docker-compose build backend
docker-compose up -d backend
```

**Frontend**:
```bash
# Install in running container
docker-compose exec frontend npm install new-package

# Or rebuild
docker-compose build frontend
docker-compose up -d frontend
```

### Running Tests

**Backend**:
```bash
docker-compose exec backend pytest tests/ -v
```

**ML**:
```bash
# Mount ml directory and run tests
docker run --rm -v $(pwd)/ml:/app python:3.11-slim \
  bash -c "cd /app && pip install -r requirements.txt && pytest tests/"
```

**Frontend**:
```bash
docker-compose exec frontend npm test
```

## Production Deployment

### Prerequisites

1. Set environment variables in `.env`
2. Update `NEXT_PUBLIC_API_URL` to your domain
3. Prepare SSL certificates (for nginx)
4. Ensure models are in `ml/models/`

### Deploy

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check health
docker-compose -f docker-compose.prod.yml ps
```

### Nginx Configuration (Optional)

Create `nginx/nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }

    upstream frontend {
        server frontend:3000;
    }

    server {
        listen 80;
        server_name yourdomain.com;

        # Backend API
        location /api {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs backend

# Check container status
docker-compose ps

# Restart service
docker-compose restart backend
```

### Port already in use

```bash
# Find process using port
lsof -i :8000  # or :3000, :6379

# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Maps container 8000 to host 8001
```

### Out of disk space

```bash
# Remove unused containers
docker container prune

# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Nuclear option (removes everything)
docker system prune -a --volumes
```

### Permission errors

```bash
# Fix model directory permissions
sudo chown -R 1000:1000 ml/models

# Or run as root (not recommended)
docker-compose exec -u root backend bash
```

### Health check failing

```bash
# Test health endpoint manually
curl http://localhost:8000/health

# Check if requests is installed
docker-compose exec backend python -c "import requests"

# View detailed health check logs
docker inspect exoplanet-backend | grep Health -A 10
```

## Optimization

### Reduce Image Size

**Backend**:
- Use multi-stage builds
- Remove build dependencies after installation
- Use `.dockerignore` to exclude unnecessary files

**Frontend**:
- Already using multi-stage build
- Standalone output mode for Next.js
- Alpine base image

### Speed Up Builds

```bash
# Use BuildKit (faster builds)
DOCKER_BUILDKIT=1 docker-compose build

# Cache dependencies
# (Already configured in Dockerfiles)
```

### Production Performance

**Backend**:
```yaml
command: gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Frontend**:
- Build optimization already enabled
- Consider CDN for static assets
- Enable Nginx caching

## Monitoring

### Container Stats

```bash
# Real-time stats
docker stats

# Check specific container
docker stats exoplanet-backend
```

### Logs

```bash
# Follow all logs
docker-compose logs -f

# Since timestamp
docker-compose logs --since 2024-01-01T00:00:00

# Save logs to file
docker-compose logs > logs.txt
```

### Health Checks

```bash
# Check all services
docker-compose ps

# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:3000

# Redis health
docker-compose exec redis redis-cli ping
```

## Scaling

### Horizontal Scaling

```bash
# Scale backend to 3 instances
docker-compose up -d --scale backend=3

# With load balancer (nginx required)
```

### Vertical Scaling

Update resource limits in `docker-compose.yml`:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## Security Best Practices

✅ **Implemented**:
- Non-root users in containers
- Health checks for all services
- .dockerignore to exclude secrets
- Read-only volumes for models in production
- Network isolation (separate networks)

🔐 **Additional Recommendations**:
- Use secrets management (Docker Swarm secrets or Kubernetes)
- Scan images for vulnerabilities (`docker scan`)
- Keep base images updated
- Use private registry for production images
- Enable Docker Content Trust
- Implement rate limiting

## Backup & Restore

### Redis Data

**Backup**:
```bash
docker-compose exec redis redis-cli SAVE
docker cp exoplanet-redis:/data/dump.rdb ./backup/
```

**Restore**:
```bash
docker cp ./backup/dump.rdb exoplanet-redis:/data/
docker-compose restart redis
```

### Models

```bash
# Backup
tar -czf models-backup.tar.gz ml/models/

# Restore
tar -xzf models-backup.tar.gz
```

## CI/CD Integration

Add to `.github/workflows/deploy.yml`:

```yaml
- name: Build and push Docker images
  run: |
    docker-compose -f docker-compose.prod.yml build
    docker tag exoplanet-backend:latest registry.example.com/backend:${{ github.sha }}
    docker push registry.example.com/backend:${{ github.sha }}
```

## Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker Security](https://docs.docker.com/engine/security/)
- [FastAPI in Docker](https://fastapi.tiangolo.com/deployment/docker/)
- [Next.js Docker Deployment](https://nextjs.org/docs/deployment#docker-image)

