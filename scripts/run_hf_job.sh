#!/usr/bin/env bash
set -euo pipefail

# Launch a bounded H200 Job using the immutable GHCR image. The checkpoint
# bucket is created by the supervisor and holds only private run artefacts.
: "${HF_TOKEN:?set HF_TOKEN to a token with Jobs, Hub, and bucket access}"
: "${IMAGE_REF:?set IMAGE_REF to the published GHCR image digest}"
if [[ "$IMAGE_REF" != *@sha256:* ]]; then
    printf '%s\n' "IMAGE_REF must be pinned by digest (image@sha256:...)" >&2
    exit 2
fi

stage="${1:-doctor}"
run_id="${RUN_ID:-barbados-dapt-v3-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
timeout="${HF_JOB_TIMEOUT:-30m}"
flavor="${HF_JOB_FLAVOR:-h200}"
bucket="${HF_BUCKET_REPO_ID:-hammertoe/barbados-dapt-checkpoints-v3}"

case "$stage" in
    doctor)
        args=(python -m hf_space.supervisor doctor)
        ;;
    smoke)
        args=(python -m hf_space.supervisor smoke)
        ;;
    train)
        timeout="${HF_JOB_TIMEOUT:-13h}"
        args=(python -m hf_space.supervisor train --budget "${DAPT_BUDGET_HOURS:-12}")
        ;;
    fuse)
        timeout="${HF_JOB_TIMEOUT:-6h}"
        flavor="${HF_JOB_FLAVOR:-cpu-performance}"
        args=(python -m hf_space.supervisor fuse)
        ;;
    mlx)
        timeout="${HF_JOB_TIMEOUT:-6h}"
        flavor="${HF_JOB_FLAVOR:-cpu-performance}"
        args=(python -m hf_space.supervisor mlx)
        ;;
    finalise)
        timeout="${HF_JOB_TIMEOUT:-12h}"
        flavor="${HF_JOB_FLAVOR:-cpu-performance}"
        args=(python -m hf_space.supervisor finalise)
        ;;
    push-adapter)
        timeout="${HF_JOB_TIMEOUT:-1h}"
        flavor="${HF_JOB_FLAVOR:-cpu-performance}"
        args=(python -m hf_space.supervisor push-adapter --run-id "${DAPT_TRAIN_RUN_ID:?set DAPT_TRAIN_RUN_ID to the finished train run}")
        ;;
    *)
        printf '%s\n' "usage: $0 [doctor|smoke|train|fuse|mlx|finalise|push-adapter]" >&2
        exit 2
        ;;
esac

hf jobs run \
    --detach \
    --label "name=barbados-dapt-v3-${stage}" \
    --label "run_id=${run_id}" \
    --label "stage=${stage}" \
    --flavor "$flavor" \
    --timeout "$timeout" \
    --secrets HF_TOKEN \
    --env "RUN_ID=${run_id}" \
    --env "IMAGE_DIGEST=${IMAGE_REF##*@}" \
    --env "HF_BUCKET_REPO_ID=${bucket}" \
    --env "HF_HOME=/data/v3/hf" \
    --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    "$IMAGE_REF" \
    "${args[@]}"
