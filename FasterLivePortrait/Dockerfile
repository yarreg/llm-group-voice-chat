FROM shaoguo/faster_liveportrait:v3

# Use bash for all RUNs and read profile
SHELL ["/bin/bash","-ic"]

ARG FASTERLIVEPORTRAIT_REPO="8aad3602177547aaa5e4beec0c3ef5b7944e7a1f"

# Clone FasterLivePortrait repository
RUN cd /root && \
    git clone https://github.com/warmshao/FasterLivePortrait.git && \
    cd FasterLivePortrait && \
    git checkout ${FASTERLIVEPORTRAIT_REPO} && \
    pip install -r requirements.txt && \
    mkdir -p /checkpoints && \
    ln -s /checkpoints checkpoints

# Download all required FasterLivePortrait models and convert to TensorRT
RUN cd /root/FasterLivePortrait && \
    huggingface-cli download warmshao/FasterLivePortrait --local-dir ./checkpoints && \
    huggingface-cli download hexgrad/Kokoro-82M --local-dir ./checkpoints/Kokoro-82M && \
    huggingface-cli download TencentGameMate/chinese-hubert-base --local-dir ./checkpoints/chinese-hubert-base && \
    huggingface-cli download jdh-algo/JoyVASA --local-dir ./checkpoints/JoyVASA

# Create script to build TensorRT engines
RUN printf '#!/bin/bash\n\
sh scripts/all_onnx2trt.sh\n\
#sh scripts/all_onnx2trt_animal.sh\n\
' > /root/FasterLivePortrait/build_trt.sh && \
    chmod +x /root/FasterLivePortrait/build_trt.sh

COPY api_v2.py /root/FasterLivePortrait/api_v2.py

WORKDIR /root/FasterLivePortrait

