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

# Render root dir is /backend, so COPY . . places main.py at /app/main.py directly.
# PYTHONPATH=/app lets all sibling packages (gateway/, evaluation/, etc.) resolve.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --log-level info"]