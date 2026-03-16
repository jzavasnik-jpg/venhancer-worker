FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    git wget ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /workspace

# Clone VEnhancer
RUN git clone https://github.com/Vchitect/VEnhancer.git

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

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
