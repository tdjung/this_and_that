#!/usr/bin/env bash
# autotrust/JEV-27B를 vLLM으로 서빙합니다 (모델 카드 권장 설정 기준).
#
#   System 2 (일반 생성)  : model = "autotrust/JEV-27B"  → Qwen3.8-27B 원본 그대로
#   System 1 (분류/판정)  : model = "jev-decision"       → LoRA + 결정 헤드
#
# 사용법:
#   MODEL_DIR=/models/JEV-27B bash serve/serve_jev27b.sh            # 분류 전용 (max-model-len 4096)
#   MODE=shared MODEL_DIR=/models/JEV-27B bash serve/serve_jev27b.sh # 생성 겸용 (긴 컨텍스트)
#
# 주요 환경변수 (기본값):
#   MODEL_DIR=./JEV-27B   PORT=8000   HOST=0.0.0.0
#   TP=1                  # tensor-parallel-size. bf16 가중치가 약 54GB라 80GB 미만 GPU는 TP=2 이상
#   GPU_MEM=0.90          # gpu-memory-utilization
#   MODE=decision         # decision | shared
#   MAX_LEN=              # 비우면 decision=4096, shared=131072
#   PREFIX_CACHE=1        # 같은 state로 여러 질문을 할 때 이득 (모델 카드 권장 옵션)
#   API_KEY=              # 지정 시 vLLM이 Bearer 토큰을 요구
#   EXTRA_ARGS=           # vllm serve에 그대로 덧붙일 인자
#
# 요구사항: Qwen3.5(qwen3_5) 아키텍처, lm_head LoRA, --logprobs-mode, allowed_token_ids를
# 지원하는 vLLM 빌드 (모델 카드는 2026년 9월 개발 빌드에서 테스트). 기동에 3~8분 걸립니다.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./JEV-27B}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP="${TP:-1}"
GPU_MEM="${GPU_MEM:-0.90}"
MODE="${MODE:-decision}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"

case "$MODE" in
  decision) MAX_LEN="${MAX_LEN:-4096}" ;;
  shared)   MAX_LEN="${MAX_LEN:-131072}" ;;
  *) echo "MODE는 decision 또는 shared 여야 합니다: $MODE" >&2; exit 1 ;;
esac

# 필수 파일 확인
for f in config.json model.safetensors.index.json tokenizer.json \
         adapter_vllm/adapter_config.json adapter_vllm/adapter_model.safetensors \
         adapter_vllm/decision_head.json calibration.json; do
  [[ -f "$MODEL_DIR/$f" ]] || { echo "누락: $MODEL_DIR/$f  (download_jev27b.sh로 받으세요)" >&2; exit 1; }
done
command -v vllm >/dev/null || { echo "vllm 명령을 찾을 수 없습니다." >&2; exit 1; }

ARGS=(
  serve "$MODEL_DIR"
  --served-model-name autotrust/JEV-27B
  --host "$HOST" --port "$PORT"
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$GPU_MEM"
  --enable-lora --max-lora-rank 32
  --lora-modules "jev-decision=$MODEL_DIR/adapter_vllm"
  --logprobs-mode processed_logprobs     # 필수: allowed_token_ids를 반영한 logprob 반환
  --max-model-len "$MAX_LEN"
)
if [[ "$PREFIX_CACHE" == "1" ]]; then
  ARGS+=(--enable-prefix-caching --mamba-cache-mode align)
fi
if [[ -n "${API_KEY:-}" ]]; then
  ARGS+=(--api-key "$API_KEY")
fi
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=($EXTRA_ARGS)

echo ">> vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"
