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
import sys
import time
import subprocess
import tempfile
import requests
import runpod
import boto3
from uuid import uuid4

# Add VEnhancer to Python path so we can import its modules
sys.path.insert(0, "/workspace/VEnhancer")


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
    """Upload a video to R2/S3 and return the public URL.
    Falls back to RunPod's built-in upload if S3 is not configured."""
    if not S3_ENDPOINT or not S3_ACCESS_KEY or not S3_SECRET_KEY:
        # Use runpod's upload utility (returns a presigned URL)
        return runpod.serverless.modules.rp_upload.upload_file_to_bucket(
            file_name=os.path.basename(local_path),
            file_location=local_path,
        )

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


def get_venhancer(version: str = "v2", steps: int = 15):
    """Get or create the VEnhancer model instance (cached)."""
    global _venhancer_instance, _venhancer_version
    if _venhancer_instance is not None and _venhancer_version == version:
        return _venhancer_instance

    from enhance_a_video import VEnhancer

    # Model files are .pt (not .pth) — downloaded from jwhejwhe/VEnhancer
    version_map = {"v2": "venhancer_v2.pt", "v1": "venhancer_paper.pt"}
    model_file = version_map.get(version, "venhancer_v2.pt")
    model_path = f"{MODEL_DIR}/{model_file}"

    _venhancer_instance = VEnhancer(
        result_dir="/tmp/venhancer_output",
        version=version,
        model_path=model_path,
        solver_mode="fast",
        steps=steps,
        guide_scale=7.5,
    )
    _venhancer_version = version
    return _venhancer_instance


# Global model cache
_venhancer_instance = None
_venhancer_version = None


def run_venhancer(
    input_path: str,
    output_path: str,
    version: str = "v2",
    up_scale: int = 4,
    target_fps: int = 30,
    steps: int = 15,
    target_resolution: list | None = None,
) -> str:
    """Run VEnhancer inference using the Python API."""
    enhancer = get_venhancer(version=version, steps=steps)

    # VEnhancer.enhance_a_video returns output frames saved to result_dir
    save_dir = os.path.dirname(output_path)
    enhancer.result_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Run enhancement — prompt can be empty for pure upscaling
    enhancer.enhance_a_video(
        video_path=input_path,
        prompt="",
        up_scale=up_scale,
        target_fps=target_fps,
        noise_aug=200,
    )

    # Find the output file VEnhancer created
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

    try:
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

            # Upload to R2/S3 (or RunPod's built-in storage)
            public_url = upload_video(output_path)

        processing_time = time.time() - start_time

        return {
            "video_url": public_url,
            "duration_seconds": info["duration"],
            "resolution": [info["width"], info["height"]],
            "processing_time_seconds": round(processing_time, 1),
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()[-1000:],
            "processing_time_seconds": round(time.time() - start_time, 1),
        }


runpod.serverless.start({"handler": handler})
