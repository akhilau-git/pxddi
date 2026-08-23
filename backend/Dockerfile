FROM python:3.10-slim
ARG PYTORCH_VERSION=2.13.0
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# The project uses pre-built wheels.  libgomp is the small runtime dependency
# required by numerical wheels; a full compiler toolchain is unnecessary.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install first (Docker layer caching)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "torch==${PYTORCH_VERSION}" \
    && pip install --no-cache-dir -r requirements.txt

# Copy BOTH backend code AND src/ into the image — this is the actual fix
COPY backend/ ./backend/
COPY src/ ./src/

WORKDIR /app/backend
RUN useradd --create-home --shell /usr/sbin/nologin pxddi
USER pxddi

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/ready', timeout=2)"

# The default is the Compose port, while hosted services can inject PORT.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
