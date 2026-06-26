# deploy/deploy.sh
#!/usr/bin/env bash
set -euo pipefail

echo "═══════════════════════════════════════"
echo "  ISRO PS12 — Deployment Script"
echo "═══════════════════════════════════════"

# Check Docker
command -v docker >/dev/null 2>&1 || { echo "Docker not found"; exit 1; }
command -v docker compose >/dev/null 2>&1 || { echo "Docker Compose not found"; exit 1; }

# Check NVIDIA runtime
if docker info 2>/dev/null | grep -q "nvidia"; then
    echo "✓ NVIDIA Docker runtime detected"
else
    echo "⚠ NVIDIA runtime not found — GPU disabled"
fi

# Create .env if missing
if [ ! -f .env ]; then
    cp deploy/.env.example .env
    echo "✓ Created .env from .env.example"
fi

# Create dirs
mkdir -p data/goes19/raw data/insat/raw models/checkpoints \
         models/pretrained outputs runs

# Build & start
echo "Building images..."
docker compose build --no-cache

echo "Starting services..."
docker compose up -d

echo ""
echo "✓ Backend  → http://localhost:8000"
echo "✓ Frontend → http://localhost:3000"
echo "✓ API Docs → http://localhost:8000/docs"
echo ""
echo "Logs: docker compose logs -f"