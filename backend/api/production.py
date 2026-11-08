"""Production ASGI entrypoint serving the UI and API on one origin."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api.main import app as api_app

DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_production_app(api: FastAPI = api_app, dist: Path = DIST) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if not (dist / "index.html").is_file():
            raise RuntimeError(f"Frontend build missing: {dist / 'index.html'}")
        async with api.router.lifespan_context(api):
            yield

    application = FastAPI(
        title="Whack", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    application.mount("/api", api)

    @application.api_route("/{asset_path:path}", methods=["GET", "HEAD"])
    async def frontend(request: Request, asset_path: str):
        candidate = (dist / asset_path).resolve()
        if not candidate.is_relative_to(dist.resolve()):
            raise StarletteHTTPException(status_code=404)
        if candidate.is_file():
            return FileResponse(candidate)
        navigation = not Path(asset_path).suffix and not asset_path.startswith("assets/")
        if navigation and (not asset_path or "text/html" in request.headers.get("accept", "")):
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})
        raise StarletteHTTPException(status_code=404, detail="Asset not found")

    return application


app = create_production_app()
