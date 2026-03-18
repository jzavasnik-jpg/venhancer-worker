"""
RunPod Serverless Handler for VEnhancer v2

Accepts a video URL, upscales it using VEnhancer's generative space-time
enhancement, and returns the upscaled video URL.
"""

import os
import sys
import time
import subprocess
import tempfile
import json
import traceback
import base64

# Add VEnhancer to Python path
sys.path.insert(0, "/workspace/VEnhancer")

# Set CUDA memory management before any torch imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

import torch
import runpod
import requests as http_requests
import boto3
from uuid import uuid4

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = "/models"
VENHANCER_REPO = "/workspace/VEnhancer"

S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
S3_BUCKET = os.environ.get("S3_BUCKET", "gravity-signal-assets")
S3_PUBLIC_URL = os.environ.get("S3_PUBLIC_URL", "")


# ---------------------------------------------------------------------------
# Startup: Load model eagerly so worker is ready before accepting jobs
# ---------------------------------------------------------------------------

print("[INIT] Loading VEnhancer model...", flush=True)
_enhancer = None

try:
    from enhance_a_video import VEnhancer

    model_path = f"{MODEL_DIR}/venhancer_v2.pt"
    if not os.path.exists(model_path):
        print(f"[INIT] ERROR: Model not found at {model_path}", flush=True)
        print(f"[INIT] /models contents: {os.listdir(MODEL_DIR)}", flush=True)
    else:
        print(f"[INIT] Model file size: {os.path.getsize(model_path) / 1e9:.2f} GB", flush=True)
        _enhancer = VEnhancer(
            result_dir="/tmp/venhancer_output",
            version="v2",
            model_path=model_path,
            solver_mode="fast",
            steps=15,
            guide_scale=7.5,
        )
        torch.cuda.empty_cache()
        print(f"[INIT] VEnhancer loaded successfully!", flush=True)
        print(f"[INIT] GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
        print(f"[INIT] GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB", flush=True)
except Exception as e:
    print(f"[INIT] ERROR loading VEnhancer: {e}", flush=True)
    print(traceback.format_exc(), flush=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def download_video(url: str, dest: str) -> str:
    response = http_requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest


def upload_video(local_path: str) -> str:
    if not S3_ENDPOINT or not S3_ACCESS_KEY or not S3_SECRET_KEY:
        # Return base64 for small files, error for large
        file_size = os.path.getsize(local_path)
        if file_size > 50 * 1024 * 1024:
            return f"error://file_too_large ({file_size} bytes, S3 not configured)"
        with open(local_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:video/mp4;base64,{b64}"

    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )
    key = f"upscaled/{uuid4().hex}.mp4"
    s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    if S3_PUBLIC_URL:
        return f"{S3_PUBLIC_URL}/{key}"
    return f"{S3_ENDPOINT}/{S3_BUCKET}/{key}"


def get_video_info(path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    duration = float(info.get("format", {}).get("duration", 0))
    stream = next((s for s in info.get("streams", []) if s["codec_type"] == "video"), {})
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    return {"duration": duration, "width": width, "height": height}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(event: dict) -> dict:
    start_time = time.time()

    try:
        job_input = event.get("input", {})

        # Diagnostic mode
        if job_input.get("diagnostic"):
            return {
                "status": "ok",
                "model_loaded": _enhancer is not None,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1) if torch.cuda.is_available() else 0,
                "gpu_memory_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
                "model_v2_exists": os.path.exists(f"{MODEL_DIR}/venhancer_v2.pt"),
                "model_v2_size_mb": round(os.path.getsize(f"{MODEL_DIR}/venhancer_v2.pt") / 1e6, 1) if os.path.exists(f"{MODEL_DIR}/venhancer_v2.pt") else 0,
                "s3_configured": bool(S3_ENDPOINT and S3_ACCESS_KEY),
            }

        if _enhancer is None:
            return {"error": "VEnhancer model failed to load at startup. Check worker logs."}

        video_url = job_input.get("video_url")
        if not video_url:
            return {"error": "video_url is required"}

        version = job_input.get("version", "v2")
        up_scale = job_input.get("up_scale", 4)
        fps = job_input.get("fps", 24)
        steps = job_input.get("steps", 15)
        noise_aug = job_input.get("noise_aug", 200)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Download source video
            input_path = os.path.join(tmpdir, "input.mp4")
            print(f"[JOB] Downloading video from {video_url[:100]}...", flush=True)
            download_video(video_url, input_path)
            input_info = get_video_info(input_path)
            print(f"[JOB] Input: {input_info['width']}x{input_info['height']}, {input_info['duration']:.1f}s", flush=True)

            # Clear GPU cache before processing
            torch.cuda.empty_cache()
            print(f"[JOB] GPU memory before: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

            # Run VEnhancer
            _enhancer.result_dir = tmpdir
            print(f"[JOB] Running VEnhancer (up_scale={up_scale}, steps={steps})...", flush=True)
            _enhancer.enhance_a_video(
                video_path=input_path,
                prompt="",
                up_scale=up_scale,
                target_fps=fps,
                noise_aug=noise_aug,
            )

            # Find output
            output_files = [f for f in os.listdir(tmpdir) if f.endswith(".mp4") and f != "input.mp4"]
            if not output_files:
                return {"error": "VEnhancer produced no output file"}

            output_path = os.path.join(tmpdir, output_files[0])
            info = get_video_info(output_path)
            print(f"[JOB] Output: {info['width']}x{info['height']}, {info['duration']:.1f}s", flush=True)

            # Upload
            public_url = upload_video(output_path)

            # Clear GPU cache after processing
            torch.cuda.empty_cache()

        return {
            "video_url": public_url,
            "duration_seconds": info["duration"],
            "resolution": [info["width"], info["height"]],
            "processing_time_seconds": round(time.time() - start_time, 1),
        }

    except Exception as e:
        torch.cuda.empty_cache()
        return {
            "error": str(e),
            "traceback": traceback.format_exc()[-1500:],
            "processing_time_seconds": round(time.time() - start_time, 1),
        }


runpod.serverless.start({"handler": handler})
