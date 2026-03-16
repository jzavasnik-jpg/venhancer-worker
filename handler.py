"""
RunPod Serverless Handler for VEnhancer v2

Accepts a video URL, upscales it using VEnhancer's generative space-time
enhancement, and returns the upscaled video URL.

VEnhancer is purpose-built for AI-generated video (Veo, HeyGen, Wan, etc.)
and performs spatial SR + temporal SR + artifact refinement in one pass.

Input schema:
{
  "input": {
    "video_url": "https://...",           # Required: source video URL
    "target_resolution": [3840, 2160],    # Optional: [width, height], default 4K
    "version": "v2",                      # Optional: "v1" (creative) or "v2" (detail-preserving)
    "up_scale": 4,                        # Optional: upscale factor (1-8), default 4
    "fps": 30,                            # Optional: target FPS, default 30
    "steps": 15                           # Optional: diffusion steps (more = better, slower)
  }
}

Output schema:
{
  "video_url": "https://...",             # Uploaded result URL
  "duration_seconds": 10.5,
  "resolution": [3840, 2160],
  "processing_time_seconds": 62.3
}
"""

import os
import time
import subprocess
import tempfile
import requests
import runpod
import boto3
from uuid import uuid4


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = "/models"
VENHANCER_REPO = "/workspace/VEnhancer"

# R2/S3-compatible storage for output uploads
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
S3_BUCKET = os.environ.get("S3_BUCKET", "gravity-signal-assets")
S3_PUBLIC_URL = os.environ.get("S3_PUBLIC_URL", "")


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )


def download_video(url: str, dest: str) -> str:
    """Download a video from URL to local path."""
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest


def upload_video(local_path: str) -> str:
    """Upload a video to R2/S3 and return the public URL."""
    s3 = get_s3_client()
    key = f"upscaled/{uuid4().hex}.mp4"
    s3.upload_file(
        local_path,
        S3_BUCKET,
        key,
        ExtraArgs={"ContentType": "video/mp4"},
    )
    if S3_PUBLIC_URL:
        return f"{S3_PUBLIC_URL}/{key}"
    return f"{S3_ENDPOINT}/{S3_BUCKET}/{key}"


def get_video_info(path: str) -> dict:
    """Get video duration and resolution via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", path,
        ],
        capture_output=True, text=True,
    )
    import json
    info = json.loads(result.stdout)
    duration = float(info.get("format", {}).get("duration", 0))
    stream = next((s for s in info.get("streams", []) if s["codec_type"] == "video"), {})
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    return {"duration": duration, "width": width, "height": height}


def run_venhancer(
    input_path: str,
    output_path: str,
    version: str = "v2",
    up_scale: int = 4,
    target_fps: int = 30,
    steps: int = 15,
    target_resolution: list | None = None,
) -> str:
    """Run VEnhancer inference."""
    model_ckpt = f"{MODEL_DIR}/venhancer_{version}.pth"

    cmd = [
        "python", f"{VENHANCER_REPO}/enhance.py",
        "--input_path", input_path,
        "--save_dir", os.path.dirname(output_path),
        "--model_path", model_ckpt,
        "--up_scale", str(up_scale),
        "--target_fps", str(target_fps),
        "--steps", str(steps),
        "--solver_mode", "fast",
        "--guide_scale", "7.5",
    ]

    if target_resolution:
        cmd.extend(["--target_width", str(target_resolution[0])])
        cmd.extend(["--target_height", str(target_resolution[1])])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        raise RuntimeError(f"VEnhancer failed: {result.stderr[-500:]}")

    # VEnhancer outputs to save_dir with a generated filename
    # Find the output file
    save_dir = os.path.dirname(output_path)
    output_files = [
        f for f in os.listdir(save_dir)
        if f.endswith(".mp4") and f != os.path.basename(input_path)
    ]
    if not output_files:
        raise RuntimeError("VEnhancer produced no output file")

    actual_output = os.path.join(save_dir, output_files[0])

    # Rename to expected path if different
    if actual_output != output_path:
        os.rename(actual_output, output_path)

    return output_path


# ---------------------------------------------------------------------------
# RunPod Handler
# ---------------------------------------------------------------------------

def handler(event: dict) -> dict:
    """RunPod serverless handler for VEnhancer."""
    start_time = time.time()
    job_input = event.get("input", {})

    video_url = job_input.get("video_url")
    if not video_url:
        return {"error": "video_url is required"}

    target_resolution = job_input.get("target_resolution", [3840, 2160])
    version = job_input.get("version", "v2")
    up_scale = job_input.get("up_scale", 4)
    fps = job_input.get("fps", 30)
    steps = job_input.get("steps", 15)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download source video
        input_path = os.path.join(tmpdir, "input.mp4")
        download_video(video_url, input_path)

        # Run VEnhancer
        output_path = os.path.join(tmpdir, "output.mp4")
        run_venhancer(
            input_path=input_path,
            output_path=output_path,
            version=version,
            up_scale=up_scale,
            target_fps=fps,
            steps=steps,
            target_resolution=target_resolution,
        )

        # Get output info
        info = get_video_info(output_path)

        # Upload to R2/S3
        public_url = upload_video(output_path)

    processing_time = time.time() - start_time

    return {
        "video_url": public_url,
        "duration_seconds": info["duration"],
        "resolution": [info["width"], info["height"]],
        "processing_time_seconds": round(processing_time, 1),
    }


runpod.serverless.start({"handler": handler})
