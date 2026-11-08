FROM node:22.14.0-slim@sha256:1c18d9ab3af4585870b92e4dbc5cac5a0dc77dd13df1a5905cea89fc720eb05b AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
COPY contracts/ /build/contracts/
RUN npm run build

FROM python:3.12.9-slim@sha256:48a11b7ba705fd53bf15248d1f94d36c39549903c5d59edcfa2f3f84126e7b44 AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LLM_MODE=fake LANGSMITH_TRACING=false
WORKDIR /app
COPY requirements-runtime.txt ./
RUN python -m pip install --no-cache-dir -r requirements-runtime.txt
COPY pyproject.toml README.md ./
COPY backend/ backend/
COPY contracts/ contracts/
COPY replays/ replays/
RUN python -m pip install --no-cache-dir --no-deps . && python -m pip check
COPY --from=frontend-build /build/frontend/dist frontend/dist
RUN useradd --create-home --uid 10001 whack && chown -R whack:whack /app
USER whack
EXPOSE 8000
CMD ["sh", "-c", "exec python -m uvicorn backend.api.production:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
