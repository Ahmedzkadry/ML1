FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY production_app ./production_app
COPY models ./models

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "production_app.api:app", "--host", "0.0.0.0", "--port", "8000"]