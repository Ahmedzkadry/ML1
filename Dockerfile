FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

COPY production_app ./production_app
COPY models ./models
COPY run.py ./

EXPOSE 8000

CMD ["uvicorn", "production_app.api:app", "--host", "0.0.0.0", "--port", "8000"]
