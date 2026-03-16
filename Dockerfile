FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone VEnhancer
RUN git clone https://github.com/Vchitect/VEnhancer.git

# Install VEnhancer dependencies
RUN pip install --no-cache-dir \
    -r /workspace/VEnhancer/requirements.txt

RUN pip install --no-cache-dir \
    runpod \
    boto3 \
    requests

# Download model weights
RUN mkdir -p /models && \
    wget -q -O /models/venhancer_v2.pth \
    "https://huggingface.co/Vchitect/VEnhancer/resolve/main/venhancer_v2.pth" && \
    wget -q -O /models/venhancer_paper.pth \
    "https://huggingface.co/Vchitect/VEnhancer/resolve/main/venhancer_paper.pth"

# Copy handler
COPY handler.py /workspace/handler.py

CMD ["python", "/workspace/handler.py"]
