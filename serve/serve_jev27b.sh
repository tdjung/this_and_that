#!/usr/bin/env bash
# autotrust/JEV-27B를 vLLM으로 서빙합니다 (모델 카드 권장 설정 기준). 기본은 docker로 실행합니다.
#
#   System 2 (일반 생성)  : model = "autotrust/JEV-27B"  → Qwen3.8-27B 원본 그대로
#   System 1 (분류/판정)  : model = "jev-decision"       → LoRA + 결정 헤드
#
# 사용법:
#   bash serve/serve_jev27b.sh start      # 컨테이너 시작 후 준비될 때까지 대기 (기본 동작)
#   bash serve/serve_jev27b.sh stop       # 컨테이너 중지·삭제
#   bash serve/serve_jev27b.sh restart
#   bash serve/serve_jev27b.sh status     # 컨테이너 상태 + /v1/models
#   bash serve/serve_jev27b.sh logs       # docker logs -f
#   DRY_RUN=1 bash serve/serve_jev27b.sh  # 실행할 명령만 출력
#
# 예시 (기존 qwen3.8-27b 서버와 같은 형태):
#   MODEL_DIR=/data/models/JEV-27B GPUS=0,1,2,3 HOST_PORT=8000 bash serve/serve_jev27b.sh
#   MODE=shared GPUS=4,5,6,7 HOST_PORT=8001 NAME=jev-27b-server-2 bash serve/serve_jev27b.sh
#
# 환경변수 (기본값):
#   RUNTIME=docker          # docker | native (native는 호스트의 vllm 명령으로 포그라운드 실행)
#   MODEL_DIR=./JEV-27B     # 호스트의 모델 폴더 (download_jev27b.sh로 받은 곳)
#   IMAGE=vllm/vllm-openai:latest
#                           # Qwen3.5(qwen3_5), lm_head LoRA, --logprobs-mode, allowed_token_ids를 지원하는
#                           # 이미지여야 합니다 (모델 카드는 2026년 9월 개발 빌드에서 테스트).
#                           # 기존 qwen3.8-27b 서버가 쓰는 이미지가 가장 안전합니다.
#   NAME=jev-27b-server-1   # 컨테이너 이름
#   GPUS=0,1,2,3            # --gpus '"device=..."' 에 들어갈 GPU 번호
#   TP=                     # tensor-parallel-size. 비우면 GPUS 개수
#   HOST_PORT=8000          # 호스트 포트 (컨테이너 안은 항상 8000)
#   GPU_MEM=0.90            # gpu-memory-utilization
#   MODE=decision           # decision: 분류 전용 / shared: 생성 겸용 (27b 풀의 서버로도 사용)
#   MAX_LEN=                # 비우면 decision=4096, shared=131072 (shared는 다른 27b 서버와 맞추세요)
#   PREFIX_CACHE=1          # --enable-prefix-caching --mamba-cache-mode align
#   API_KEY=                # 지정 시 vLLM이 Bearer 토큰을 요구
#   RESTART=unless-stopped  # docker 재시작 정책 (빈 값이면 미지정)
#   WAIT=1                  # start 후 /v1/models 응답까지 대기 (최대 WAIT_TIMEOUT초)
#   WAIT_TIMEOUT=900
#   EXTRA_DOCKER_ARGS=      # docker run에 덧붙일 인자 (기존 스크립트의 -e, -v 옵션 등)
#   EXTRA_ARGS=             # vllm serve에 덧붙일 인자 (기존 스크립트의 vllm 옵션 등)
set -euo pipefail

ACTION="${1:-start}"
RUNTIME="${RUNTIME:-docker}"
MODEL_DIR="${MODEL_DIR:-./JEV-27B}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
NAME="${NAME:-jev-27b-server-1}"
GPUS="${GPUS:-0,1,2,3}"
HOST_PORT="${HOST_PORT:-8000}"
GPU_MEM="${GPU_MEM:-0.90}"
MODE="${MODE:-decision}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
RESTART="${RESTART-unless-stopped}"
WAIT="${WAIT:-1}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-900}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a _gpu_list <<<"$GPUS"
TP="${TP:-${#_gpu_list[@]}}"

case "$MODE" in
  decision) MAX_LEN="${MAX_LEN:-4096}" ;;
  shared)   MAX_LEN="${MAX_LEN:-131072}" ;;
  *) echo "MODE는 decision 또는 shared 여야 합니다: $MODE" >&2; exit 1 ;;
esac

die() { echo "serve_jev27b: $*" >&2; exit 1; }
run() { if [[ "$DRY_RUN" == "1" ]]; then printf '%q ' "$@"; echo; else "$@"; fi; }

check_model_dir() {
  [[ -d "$MODEL_DIR" ]] || die "모델 폴더가 없습니다: $MODEL_DIR"
  for f in config.json model.safetensors.index.json tokenizer.json \
           adapter_vllm/adapter_config.json adapter_vllm/adapter_model.safetensors \
           adapter_vllm/decision_head.json calibration.json; do
    [[ -f "$MODEL_DIR/$f" ]] || die "누락: $MODEL_DIR/$f  (download_jev27b.sh로 받으세요)"
  done
}

# vllm serve 인자. $1 = 서버가 보는 모델 경로, $2 = 포트
vllm_args() {
  local mdir="$1" port="$2"
  VLLM_ARGS=(
    serve "$mdir"
    --served-model-name autotrust/JEV-27B
    --host 0.0.0.0 --port "$port"
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization "$GPU_MEM"
    --enable-lora --max-lora-rank 32
    --lora-modules "jev-decision=$mdir/adapter_vllm"
    --logprobs-mode processed_logprobs        # 필수: allowed_token_ids를 반영한 logprob 반환
    --max-model-len "$MAX_LEN"
  )
  [[ "$PREFIX_CACHE" == "1" ]] && VLLM_ARGS+=(--enable-prefix-caching --mamba-cache-mode align)
  [[ -n "${API_KEY:-}" ]] && VLLM_ARGS+=(--api-key "$API_KEY")
  # shellcheck disable=SC2206
  [[ -n "${EXTRA_ARGS:-}" ]] && VLLM_ARGS+=($EXTRA_ARGS)
  return 0
}

wait_ready() {
  [[ "$WAIT" == "1" && "$DRY_RUN" != "1" ]] || return 0
  local url="http://127.0.0.1:${HOST_PORT}/v1/models" auth=() t0=$SECONDS
  [[ -n "${API_KEY:-}" ]] && auth=(-H "Authorization: Bearer $API_KEY")
  echo ">> 준비 대기 중 ($url, 보통 3~8분). 진행 상황: bash $0 logs"
  while (( SECONDS - t0 < WAIT_TIMEOUT )); do
    if curl -sf "${auth[@]}" "$url" 2>/dev/null | grep -q jev-decision; then
      echo ">> 준비 완료 ($((SECONDS - t0))초). 확인: python serve/smoke_test.py --url http://127.0.0.1:${HOST_PORT} --model-dir $MODEL_DIR"
      return 0
    fi
    if ! docker ps -q -f "name=^${NAME}$" | grep -q .; then
      docker logs --tail 40 "$NAME" 2>&1 || true
      die "컨테이너가 종료되었습니다."
    fi
    sleep 10
  done
  die "${WAIT_TIMEOUT}초 안에 준비되지 않았습니다. 'bash $0 logs'로 확인하세요."
}

docker_start() {
  check_model_dir
  command -v docker >/dev/null || die "docker 명령을 찾을 수 없습니다."
  if docker ps -a -q -f "name=^${NAME}$" | grep -q . && [[ "$DRY_RUN" != "1" ]]; then
    die "같은 이름의 컨테이너가 이미 있습니다: $NAME  ('bash $0 restart' 또는 NAME=... 지정)"
  fi
  local abs_dir mdir=/models/JEV-27B
  abs_dir="$(cd "$MODEL_DIR" && pwd)"
  vllm_args "$mdir" 8000
  local DOCKER_ARGS=(
    run -d
    --gpus "\"device=${GPUS}\""
    -p "${HOST_PORT}:8000"
    --ipc=host
    --name "$NAME"
    -v "${abs_dir}:${mdir}:ro"
    -e HF_HUB_OFFLINE=1                       # HF 접속 없이 로컬 파일만 사용
    -e TRANSFORMERS_OFFLINE=1
  )
  [[ -n "$RESTART" ]] && DOCKER_ARGS+=(--restart "$RESTART")
  # shellcheck disable=SC2206
  [[ -n "${EXTRA_DOCKER_ARGS:-}" ]] && DOCKER_ARGS+=($EXTRA_DOCKER_ARGS)
  DOCKER_ARGS+=(--entrypoint vllm "$IMAGE")   # 이미지 버전마다 엔트리포인트가 달라 vllm으로 고정
  VLLM_ARGS_NO_SERVE=("${VLLM_ARGS[@]:1}")    # entrypoint가 vllm이므로 serve부터 그대로 전달
  echo ">> 컨테이너 $NAME 시작 (GPU $GPUS, TP=$TP, 포트 $HOST_PORT, MODE=$MODE, max-model-len $MAX_LEN)"
  run docker "${DOCKER_ARGS[@]}" serve "${VLLM_ARGS_NO_SERVE[@]}"
  wait_ready
}

case "$RUNTIME:$ACTION" in
  docker:start)   docker_start ;;
  docker:stop)    run docker rm -f "$NAME" ;;
  docker:restart) run docker rm -f "$NAME" >/dev/null 2>&1 || true; docker_start ;;
  docker:logs)    exec docker logs -f "$NAME" ;;
  docker:status)
    docker ps -a -f "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    curl -s "http://127.0.0.1:${HOST_PORT}/v1/models" ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
      | python3 -c 'import sys,json; print("models:", [m["id"] for m in json.load(sys.stdin)["data"]])' \
      2>/dev/null || echo "models: (응답 없음)"
    ;;
  native:start)
    check_model_dir
    [[ "$DRY_RUN" == "1" ]] || command -v vllm >/dev/null || die "vllm 명령을 찾을 수 없습니다."
    vllm_args "$MODEL_DIR" "$HOST_PORT"
    export CUDA_VISIBLE_DEVICES="$GPUS"
    echo ">> CUDA_VISIBLE_DEVICES=$GPUS vllm ${VLLM_ARGS[*]}"
    [[ "$DRY_RUN" == "1" ]] && exit 0
    exec vllm "${VLLM_ARGS[@]}"
    ;;
  *) die "알 수 없는 조합: RUNTIME=$RUNTIME ACTION=$ACTION (start|stop|restart|status|logs)" ;;
esac
