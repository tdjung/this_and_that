#!/usr/bin/env bash
# MiniMax-H3 FL2VA를 vLLM-Omni로 서버 실행 (L40S 48GB × 8 서버용)
#
# 사용법:
#   MODEL_DIR=/models/MiniMax-H3 ./serve.sh l40s           # 1순위: GPU 0~3에 TP4
#   MODEL_DIR=/models/MiniMax-H3 ./serve.sh l40s-offload   # 메모리 부족(OOM)이면: 레이어 오프로드 추가
#   MODEL_DIR=/models/MiniMax-H3 ./serve.sh single         # 그래도 안 되면: GPU 1장 + CPU 오프로드 (공식 프로필, 느림)
#
# 참고
# - 공식 레시피에 L40S 전용 프로필은 없어요. l40s 프로필은 공식 RTX 5090(32GB×2) 프로필의
#   "텐서 병렬로 가중치를 나눠 올리는" 방식을 48GB×4에 맞춘 설정이에요.
# - L40S는 NVLink가 없어서 GPU 8장 전체 텐서 병렬보다 4장이 통신 부담이 적어요.
#   남는 GPU 4~7은 키프레임 이미지 생성이나 업스케일에 쓰세요.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:?MODEL_DIR에 다운로드한 MiniMax-H3 루트 경로를 넣어주세요 (FL2VA 하위 폴더 아님)}"
PROFILE="${1:-l40s}"
PORT="${PORT:-8000}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-14400}"
export FLASHINFER_DISABLE_VERSION_CHECK=1

COMMON=(
  --omni --trust-remote-code
  --task-type fl2va
  --host 0.0.0.0 --port "$PORT"
)

case "$PROFILE" in
  l40s)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    vllm serve "$MODEL_DIR" "${COMMON[@]}" \
      --num-gpus 4 --tensor-parallel-size 4 --usp 1 \
      --vae-use-tiling \
      --diffusion-attention-backend FLASH_ATTN
    ;;
  l40s-offload)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    vllm serve "$MODEL_DIR" "${COMMON[@]}" \
      --num-gpus 4 --tensor-parallel-size 4 --usp 1 \
      --vae-use-tiling \
      --enable-distributed-layerwise-offload --dlo-no-use-allgather \
      --enforce-eager \
      --diffusion-attention-backend FLASH_ATTN
    ;;
  single)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    vllm serve "$MODEL_DIR" "${COMMON[@]}" \
      --num-gpus 1 \
      --enable-cpu-offload \
      --diffusion-attention-backend FLASH_ATTN
    ;;
  *)
    echo "프로필은 l40s, l40s-offload, single 중 하나예요"; exit 1 ;;
esac
