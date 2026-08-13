#!/usr/bin/env bash
set -euo pipefail

# Build a Linux image and publish an immutable content-addressed reference for
# Hugging Face Jobs. GHCR_TOKEN needs `write:packages` for the hammertoe owner.
: "${GHCR_TOKEN:?set GHCR_TOKEN to a GitHub token with write:packages}"

image="${IMAGE_REPO:-ghcr.io/hammertoe/qwen3-omni-barbados-dapt}"
tag="${IMAGE_TAG:-v3}"

printf '%s\n' "$GHCR_TOKEN" | docker login ghcr.io --username hammertoe --password-stdin
docker buildx build \
    --platform linux/amd64 \
    --provenance=false \
    --file hf_space/Dockerfile \
    --tag "${image}:${tag}" \
    --push \
    .

docker buildx imagetools inspect "${image}:${tag}"
