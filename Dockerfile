FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone VEnhancer
RUN git clone https://github.com/Vchitect/VEnhancer.git

# Install dependencies
RUN pip install --no-cache-dir -r /workspace/VEnhancer/requirements.txt
RUN pip install --no-cache-dir runpod boto3 requests huggingface_hub

# Download VEnhancer model weights (public repo)
RUN mkdir -p /models && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('jwhejwhe/VEnhancer', 'venhancer_v2.pt', local_dir='/models')" && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download('jwhejwhe/VEnhancer', 'venhancer_paper.pt', local_dir='/models')" && \
    ls -lh /models/

COPY handler.py /workspace/handler.py

CMD ["python", "/workspace/handler.py"]
