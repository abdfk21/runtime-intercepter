"""FastAPI Application entrypoint."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.core.config import settings
from app.core.logging import logger
from app.storage.database import init_db
from app.api.routes import router as api_router

app = FastAPI(
    title="Runtime Interceptor for Multi-Agent LLMs",
    description="Adaptive, Architecture-Aware Error Containment at Agent Handoff Boundaries",
    version="1.0.0",
)

# Initialize database tables on startup
@app.on_event("startup")
def on_startup():
    logger.info("Initializing database tables...")
    init_db()
    logger.info("Database initialized successfully.")

# Mount API routes
app.include_router(api_router)

# Mount static files and templates
dashboard_dir = Path(__file__).resolve().parent / "dashboard"
static_dir = dashboard_dir / "static"
template_file = dashboard_dir / "templates" / "index.html"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def get_dashboard():
    if template_file.exists():
        return FileResponse(template_file)
    return {"message": "Runtime Interceptor API is operational. Visit /docs for OpenAPI specifications."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=settings.debug)
