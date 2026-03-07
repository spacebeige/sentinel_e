FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Tell Python that /app is the project root so `import backend.X` resolves
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Render injects $PORT at runtime.  Default to 8000 for local Docker runs.
# Module path must be backend.main:app because WORKDIR=/app and main.py
# lives at /app/backend/main.py — NOT /app/main.py.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --log-level info"]