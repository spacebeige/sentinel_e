FROM python:3.11-slim

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg libsndfile1 libportaudio2 aplay \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create media dir for Asterisk sound files
RUN mkdir -p /mnt/asterisk_media/ai-generated

EXPOSE 8090 15000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:15000/health || exit 1

CMD ["python", "main.py"]
