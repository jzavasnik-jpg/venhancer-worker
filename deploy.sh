#!/bin/bash
#
# Deploy VEnhancer to RunPod Serverless
#
# Prerequisites:
#   1. RunPod account with API key
#   2. Docker installed locally (or use RunPod's build system)
#   3. Set env vars: RUNPOD_API_KEY, DOCKERHUB_USERNAME
#
# Usage:
#   ./deploy.sh
#
# After deployment, add these to Signal's .env.local:
#   RUNPOD_API_KEY=your-runpod-api-key
#   RUNPOD_VENHANCER_ENDPOINT_ID=<endpoint-id-from-deploy>
#
# The worker uses R2 for output storage (same bucket as Signal).
# Set these env vars on the RunPod endpoint template:
#   S3_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
#   S3_ACCESS_KEY=<r2-access-key>
#   S3_SECRET_KEY=<r2-secret-key>
#   S3_BUCKET=gravity-signal-assets
#   S3_PUBLIC_URL=https://assets.yourdomain.com
#

set -euo pipefail

IMAGE_NAME="${DOCKERHUB_USERNAME:-gravityculture}/venhancer-worker"
IMAGE_TAG="v2"

echo "=== Building VEnhancer Docker image ==="
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo "=== Pushing to Docker Hub ==="
docker push "${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "=== Image pushed: ${IMAGE_NAME}:${IMAGE_TAG} ==="
echo ""
echo "Next steps:"
echo "  1. Go to https://www.runpod.io/console/serverless"
echo "  2. Create a new Serverless Endpoint"
echo "  3. Set Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  4. GPU: NVIDIA A6000 (48GB) or A100 (40/80GB)"
echo "  5. Min workers: 0, Max workers: 3"
echo "  6. Idle timeout: 60s"
echo "  7. Set environment variables:"
echo "     - S3_ENDPOINT"
echo "     - S3_ACCESS_KEY"
echo "     - S3_SECRET_KEY"
echo "     - S3_BUCKET=gravity-signal-assets"
echo "     - S3_PUBLIC_URL"
echo "  8. Copy the Endpoint ID and add to Signal .env.local:"
echo "     RUNPOD_API_KEY=<your-key>"
echo "     RUNPOD_VENHANCER_ENDPOINT_ID=<endpoint-id>"
echo ""
echo "Estimated cost: ~\$0.39/hr (A6000) or ~\$0.69/hr (A100)"
echo "Billed per-second. Scales to zero when idle."
