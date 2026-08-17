FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install first (Docker layer caching)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy BOTH backend code AND src/ into the image — this is the actual fix
COPY backend/ ./backend/
COPY src/ ./src/

WORKDIR /app/backend
RUN useradd --create-home --shell /usr/sbin/nologin pxddi
USER pxddi

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
