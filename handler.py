"""
RunPod Serverless Handler for VEnhancer v2
"""

import os
import sys
import time
import subprocess
import tempfile
import json
import traceback
import base64

sys.path.insert(0, "/workspace/VEnhancer")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

import runpod
import requests as http_requests
import boto3
from uuid import uuid4

MODEL_DIR = "/models"
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
S3_BUCKET = os.environ.get("S3_BUCKET", "gravity-signal-assets")
S3_PUBLIC_URL = os.environ.get("S3_PUBLIC_URL", "")


def download_video(url, dest):
    r = http_requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)


def upload_video(path):
    if not S3_ENDPOINT or not S3_ACCESS_KEY or not S3_SECRET_KEY:
        size = os.path.getsize(path)
        if size > 50 * 1024 * 1024:
            return f"error://too_large_{size}_bytes_s3_not_configured"
        with open(path, "rb") as f:
            return f"data:video/mp4;base64,{base64.b64encode(f.read()).decode()}"
    s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT,
                       aws_access_key_id=S3_ACCESS_KEY,
                       aws_secret_access_key=S3_SECRET_KEY)
    key = f"upscaled/{uuid4().hex}.mp4"
    s3.upload_file(path, S3_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    return f"{S3_PUBLIC_URL}/{key}" if S3_PUBLIC_URL else f"{S3_ENDPOINT}/{S3_BUCKET}/{key}"


def get_video_info(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True)
    info = json.loads(r.stdout)
    fmt = info.get("format", {})
    s = next((x for x in info.get("streams", []) if x["codec_type"] == "video"), {})
    return {"duration": float(fmt.get("duration", 0)),
            "width": int(s.get("width", 0)),
            "height": int(s.get("height", 0)),
            "fps": s.get("r_frame_rate", "24/1")}


def trim_video(input_path, output_path, max_seconds=10):
    """Trim video to max_seconds to avoid OOM on long videos."""
    info = get_video_info(input_path)
    if info["duration"] <= max_seconds:
        return input_path
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-t", str(max_seconds), "-c", "copy", output_path
    ], capture_output=True)
    return output_path


def handler(event):
    import torch
    start = time.time()

    try:
        inp = event.get("input", {})

        # Diagnostic mode — instant response
        if inp.get("diagnostic"):
            return {
                "status": "ok",
                "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                "vram_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1) if torch.cuda.is_available() else 0,
                "model_exists": os.path.exists(f"{MODEL_DIR}/venhancer_v2.pt"),
                "model_size_gb": round(os.path.getsize(f"{MODEL_DIR}/venhancer_v2.pt") / 1e9, 2) if os.path.exists(f"{MODEL_DIR}/venhancer_v2.pt") else 0,
                "s3_configured": bool(S3_ENDPOINT and S3_ACCESS_KEY),
            }

        url = inp.get("video_url")
        if not url:
            return {"error": "video_url is required"}

        up_scale = inp.get("up_scale", 4)
        fps = inp.get("fps", 24)
        steps = inp.get("steps", 15)
        version = inp.get("version", "v2")
        noise_aug = inp.get("noise_aug", 200)
        max_seconds = inp.get("max_seconds", 10)

        with tempfile.TemporaryDirectory() as tmp:
            # Download
            raw = os.path.join(tmp, "raw.mp4")
            download_video(url, raw)

            # Trim to avoid OOM
            trimmed = os.path.join(tmp, "input.mp4")
            actual_input = trim_video(raw, trimmed, max_seconds)
            input_info = get_video_info(actual_input)

            # Load VEnhancer lazily per job
            torch.cuda.empty_cache()
            from enhance_a_video import VEnhancer

            model_file = "venhancer_v2.pt" if version == "v2" else "venhancer_paper.pt"
            enhancer = VEnhancer(
                result_dir=tmp,
                version=version,
                model_path=f"{MODEL_DIR}/{model_file}",
                solver_mode="fast",
                steps=steps,
                guide_scale=7.5,
            )

            # Enhance
            enhancer.enhance_a_video(
                video_path=actual_input,
                prompt="",
                up_scale=up_scale,
                target_fps=fps,
                noise_aug=noise_aug,
            )

            # Cleanup model from GPU
            del enhancer
            torch.cuda.empty_cache()

            # Find output
            outs = [f for f in os.listdir(tmp) if f.endswith(".mp4") and f not in ("raw.mp4", "input.mp4")]
            if not outs:
                return {"error": "VEnhancer produced no output"}

            out_path = os.path.join(tmp, outs[0])
            out_info = get_video_info(out_path)
            video_url = upload_video(out_path)

        return {
            "video_url": video_url,
            "input_resolution": [input_info["width"], input_info["height"]],
            "output_resolution": [out_info["width"], out_info["height"]],
            "duration_seconds": out_info["duration"],
            "processing_time_seconds": round(time.time() - start, 1),
        }

    except Exception as e:
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
        return {
            "error": str(e),
            "traceback": traceback.format_exc()[-1500:],
            "time": round(time.time() - start, 1),
        }


runpod.serverless.start({"handler": handler})
