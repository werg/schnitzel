#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${SCHNITZELJAGD_IMAGE:-${SDKB_IMAGE:-schnitz-spark:0.4}}"
configured_cache="${SCHNITZELJAGD_CACHE_DIR:-${SDKB_CACHE_DIR:-}}"
if [[ -z "$configured_cache" && -f "$root/.sdkb/cache-dir" ]]; then
  IFS= read -r configured_cache < "$root/.sdkb/cache-dir" || true
elif [[ -z "$configured_cache" && -f "$root/.schnitz/cache-dir" ]]; then
  IFS= read -r configured_cache < "$root/.schnitz/cache-dir" || true
  [[ -n "$configured_cache" ]] || { echo 'Empty .schnitz/cache-dir storage setting.' >&2; exit 1; }
fi
cache="${configured_cache:-$HOME/.cache/schnitz}"
if [[ -z "$configured_cache" && ! -d "$cache" && -d "$HOME/.cache/sdkb" ]]; then cache="$HOME/.cache/sdkb"; fi
configured_runs="${SCHNITZELJAGD_RUNS_DIR:-${SDKB_RUNS_DIR:-}}"
if [[ -z "$configured_runs" && -f "$root/.sdkb/runs-dir" ]]; then
  IFS= read -r configured_runs < "$root/.sdkb/runs-dir" || true
elif [[ -z "$configured_runs" && -f "$root/.schnitz/runs-dir" ]]; then
  IFS= read -r configured_runs < "$root/.schnitz/runs-dir" || true
  [[ -n "$configured_runs" ]] || { echo 'Empty .schnitz/runs-dir storage setting.' >&2; exit 1; }
fi
runs="${configured_runs:-$root/runs}"
archive="${SCHNITZELJAGD_ARCHIVE_DIR:-${SDKB_ARCHIVE_DIR:-}}"
if [[ -z "$archive" && -f "$root/.sdkb/archive-dir" ]]; then
  IFS= read -r archive < "$root/.sdkb/archive-dir" || true
elif [[ -z "$archive" && -f "$root/.schnitz/archive-dir" ]]; then
  IFS= read -r archive < "$root/.schnitz/archive-dir" || true
  [[ -n "$archive" ]] || { echo 'Empty .schnitz/archive-dir storage setting.' >&2; exit 1; }
fi
container="${SCHNITZELJAGD_CONTAINER:-${SDKB_CONTAINER:-schnitz-training}}"
command="${1:-shell}"
shift || true
case "$command" in
  build)
    [[ "$(uname -m)" == aarch64 || "$(uname -m)" == arm64 ]] || {
      echo 'Build natively on Spark/ARM64, not under x86 emulation.' >&2; exit 1;
    }
    base="${SCHNITZELJAGD_BASE_IMAGE:-${SDKB_BASE_IMAGE:-nvcr.io/nvidia/pytorch:25.11-py3}}"
    docker pull --platform linux/arm64 "$base"
    [[ "$(docker image inspect --format '{{.Architecture}}' "$base")" == arm64 ]] || {
      echo 'Base image is not ARM64.' >&2; exit 1;
    }
    pinned="$(docker image inspect --format '{{index .RepoDigests 0}}' "$base")"
    mkdir -p "$root/.schnitz"
    printf '%s\n' "$pinned" >"$root/.schnitz/base-image.txt"
    exec docker build --platform linux/arm64 -f "$root/docker/Dockerfile.spark" \
      --build-arg "BASE_IMAGE=$pinned" -t "$image" "$root"
    ;;
  shell|run|start)
    # An explicit storage location must already exist; a missing mount must not
    # quietly turn into a directory on the internal filesystem.
    if [[ -n "$configured_runs" ]]; then
      [[ -d "$runs" ]] || { echo 'Configured run storage is unavailable; check the mounted disk.' >&2; exit 1; }
    else
      mkdir -p "$runs"
    fi
    if [[ -n "$configured_cache" ]]; then
      [[ -d "$cache" ]] || { echo 'Configured cache storage is unavailable; check the mounted disk.' >&2; exit 1; }
    else
      mkdir -p "$cache"
    fi
    mkdir -p "$cache"/{tmp,xdg,triton,cuda,torchinductor,torch,wandb}
    if [[ "$command" == start || ( ( "${1:-}" == schnitz || "${1:-}" == sdkb ) && ( "${2:-}" == launch || "${2:-}" == train || ( "${2:-}" == runs && "${3:-}" == start ) ) ) ]]; then
      previous=""
      for argument in "$@"; do
        if [[ "$previous" == --output || "$argument" == --output=* ]]; then
          output="${argument#--output=}"
          if [[ "$output" != /* ]]; then
            echo 'Use an absolute container output path, normally /runs/NAME for configured storage.' >&2
            exit 2
          fi
        fi
        previous="$argument"
      done
    fi
    flags=(--init --gpus all --shm-size=8g --stop-timeout=600 \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      --user "$(id -u):$(id -g)" --env HOME=/tmp \
      --env HF_HOME=/cache/huggingface --env NVIDIA_IMEX_CHANNELS=0 \
      --env XDG_CACHE_HOME=/cache/xdg --env TRITON_CACHE_DIR=/cache/triton \
      --env CUDA_CACHE_PATH=/cache/cuda --env TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
      --env TORCH_HOME=/cache/torch --env WANDB_CACHE_DIR=/cache/wandb --env TMPDIR=/cache/tmp \
      --mount "type=bind,src=$root,dst=/workspace/schnitz" \
      --mount "type=bind,src=$cache,dst=/cache" \
      --mount "type=bind,src=$runs,dst=/runs" --workdir /workspace/schnitz)
    if [[ -n "$archive" ]]; then
      [[ -d "$archive" ]] || { echo 'Archive directory must exist on the mounted disk.' >&2; exit 1; }
      flags+=(--mount "type=bind,src=$archive,dst=/archive")
      # Relocated checkpoint links use a host-absolute path. Expose that same
      # path so existing run directories remain readable inside/outside Docker.
      archive_absolute="$(cd "$archive" && pwd -P)"
      [[ "$archive_absolute" == /archive ]] || flags+=(--mount "type=bind,src=$archive_absolute,dst=$archive_absolute")
    fi
    [[ -z "${HF_TOKEN:-}" ]] || flags+=(--env HF_TOKEN)
    [[ -z "${WANDB_API_KEY:-}" ]] || flags+=(--env WANDB_API_KEY)
    [[ -z "${WANDB_BASE_URL:-}" ]] || flags+=(--env WANDB_BASE_URL)
    if [[ "$command" == start ]]; then
      [[ $# -gt 0 ]] || { echo 'Supply schnitz launch arguments after start.' >&2; exit 2; }
      exec docker run --detach --name "$container" "${flags[@]}" "$image" schnitz launch "$@"
    fi
    flags+=(--rm)
    if [[ "$command" == shell ]]; then
      exec docker run -it "${flags[@]}" "$image" bash
    fi
    [[ $# -gt 0 ]] || { echo 'Supply a command after run.' >&2; exit 1; }
    exec docker run "${flags[@]}" "$image" "$@"
    ;;
  status) exec docker inspect --format '{{.State.Status}} exit={{.State.ExitCode}}' "$container" ;;
  logs) exec docker logs --follow "$container" ;;
  stop) exec docker stop --time "${SCHNITZELJAGD_STOP_TIMEOUT:-${SDKB_STOP_TIMEOUT:-600}}" "$container" ;;
  *) echo 'Usage: scripts/spark.sh build|shell|run COMMAND...|start LAUNCH_ARGS...|status|logs|stop' >&2; exit 2 ;;
esac
