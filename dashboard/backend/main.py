import logging, sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dashboard.backend.config import settings
from dashboard.backend.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description="ISRO PS12 — Satellite Temporal Interpolation Dashboard",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {"status": "ok", "app": settings.app_name, "version": "2.0.0"}


@app.get("/api/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("dashboard.backend.main:app",
                host=settings.host, port=settings.port,
                reload=settings.debug, log_level="info")