FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

RUN huggingface-cli download Misha24-10/F5-TTS_RUSSIAN F5TTS_v1_Base_v2/model_last_inference.safetensors --local-dir /app/models --local-dir-use-symlinks False
RUN huggingface-cli download Misha24-10/F5-TTS_RUSSIAN F5TTS_v1_Base/vocab.txt --local-dir /app/models --local-dir-use-symlinks False

RUN huggingface-cli download charactr/vocos-mel-24khz config.yaml --local-dir /app/models/vocos --local-dir-use-symlinks False
RUN huggingface-cli download charactr/vocos-mel-24khz pytorch_model.bin --local-dir /app/models/vocos --local-dir-use-symlinks False

RUN huggingface-cli download ruaccent/accentuator --local-dir /app/models/ruaccent --local-dir-use-symlinks False \
    && PYTHON_RUACCENT_PATH=$(python3 -c "import ruaccent; import pathlib; print(pathlib.Path(ruaccent.__file__).parent)") \
    && ln -s /app/models/ruaccent/koziev $PYTHON_RUACCENT_PATH/koziev

COPY api.py /app/
COPY voices.yaml /app/
COPY voices /app/voices/

WORKDIR /app/

EXPOSE 8080
CMD ["python3", "api.py"]