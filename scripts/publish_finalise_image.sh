#!/usr/bin/env bash
set -euo pipefail

# Build and publish the finalise image (training base + mlx-vlm) to GHCR.
: "${GHCR_TOKEN:?set GHCR_TOKEN to a GitHub token with write:packages}"
: "${BASE_IMAGE_REF:?set BASE_IMAGE_REF to the pinned training image digest}"

image="${IMAGE_REPO:-ghcr.io/hammertoe/qwen3-omni-barbados-dapt}"
tag="${IMAGE_TAG:-finalise-v3}"

printf '%s\n' "$GHCR_TOKEN" | docker login ghcr.io --username hammertoe --password-stdin
docker buildx build \
    --platform linux/amd64 \
    --provenance=false \
    --build-arg "BASE_IMAGE=${BASE_IMAGE_REF}" \
    --file containers/Dockerfile.finalise \
    --tag "${image}:${tag}" \
    --push \
    .

docker buildx imagetools inspect "${image}:${tag}"
