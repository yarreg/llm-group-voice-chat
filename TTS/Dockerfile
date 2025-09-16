# CUDA-runtime образ. Версию CUDA/torch при необходимости уточнить под окружение.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    ffmpeg libsndfile1 git git-lfs \
 && git lfs install \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create virtual environment
RUN python3 -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip install "torch==2.4.0+cu124" "torchaudio==2.4.0+cu124" --extra-index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy application code
COPY app /app/app
COPY config.yaml /app/config.yaml

# Create necessary directories
RUN mkdir -p /app/models /app/voices /app/cache

# Set permissions
RUN chown -R 1000:1000 /app
USER 1000:1000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/ping || exit 1

# Run the application
CMD ["uvicorn", "app.main:api", "--host", "0.0.0.0", "--port", "8000"]