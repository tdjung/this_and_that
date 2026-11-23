#!/usr/bin/env bash
# MiniMax-H3 (FL2VA: 텍스트→영상, 첫 프레임→영상)를 vLLM-Omni docker로 서빙합니다. (L40S 48GB 서버 기준)
#
# 사용법:
#   bash serve.sh setup      # 최초 1회: vLLM-Omni 최신 소스 받기 (이미지의 패키지를 덮어씀)
#   bash serve.sh start      # 컨테이너 시작 후 준비될 때까지 대기 (기본 동작)
#   bash serve.sh stop       # 컨테이너 중지·삭제
#   bash serve.sh restart
#   bash serve.sh status     # 컨테이너 상태 + /health
#   bash serve.sh logs       # docker logs -f
#   DRY_RUN=1 bash serve.sh  # 실행할 docker 명령만 출력
#
# 예시:
#   MODEL_DIR=/data/models/MiniMax-H3 bash serve.sh                          # GPU 0~3, 포트 19300
#   MODEL_DIR=/data/models/MiniMax-H3 PROFILE=l40s-offload bash serve.sh restart   # 메모리 부족(OOM)이면
#   MODEL_DIR=/data/models/MiniMax-H3 PROFILE=single GPUS=4 bash serve.sh restart  # 그래도 안 되면
#
# 환경변수 (기본값):
#   MODEL_DIR=              # 필수. hf download로 받은 MiniMax-H3 "루트" 폴더 (FL2VA 하위 폴더 아님)
#   OMNI_SRC=./vllm-omni    # vLLM-Omni 소스 위치 (setup이 여기에 받음)
#   IMAGE=vllm/vllm-omni:minimax-h3
#                           # 이 태그는 H3 모듈형 파이프라인(vLLM-Omni #5720)보다 오래돼서,
#                           # OMNI_SRC를 마운트하고 PYTHONPATH 앞에 둬서 최신 소스로 덮어씁니다 (공식 레시피 방식).
#   NAME=minimax-h3-fl2va   # 컨테이너 이름
#   GPUS=0,1,2,3            # --gpus '"device=..."' 에 들어갈 GPU 번호
#   HOST_PORT=19300         # 호스트 포트 (컨테이너 안은 항상 8000). 기존 8000/8001 서버와 겹치지 않게
#   PROFILE=l40s            # l40s: GPUS 개수만큼 텐서 병렬
#                           # l40s-offload: + 레이어 오프로드 (메모리 부족 시)
#                           # single: GPU 1장 + CPU 오프로드 (공식 프로필, 느림, 호스트 RAM 많이 필요)
#   RESTART=                # docker 재시작 정책 (영상 만들 때만 띄우는 용도라 기본은 미지정)
#   WAIT=1                  # start 후 /health 응답까지 대기 (최대 WAIT_TIMEOUT초)
#   WAIT_TIMEOUT=1800       # 33B 모델 두 개(DiT, 텍스트 인코더)를 올려서 오래 걸릴 수 있음
#   EXTRA_DOCKER_ARGS=      # docker run에 덧붙일 인자
#   EXTRA_ARGS=             # vllm serve에 덧붙일 인자
#
# 참고: 공식 레시피에 L40S 전용 프로필은 없습니다. l40s 프로필은 RTX 5090(32GB×2) 프로필의
#       텐서 병렬 방식을 48GB×4에 맞춘 설정입니다. L40S는 NVLink가 없어 8장보다 4장이 통신 부담이 적습니다.
set -euo pipefail

ACTION="${1:-start}"
MODEL_DIR="${MODEL_DIR:-}"
OMNI_SRC="${OMNI_SRC:-./vllm-omni}"
IMAGE="${IMAGE:-vllm/vllm-omni:minimax-h3}"
NAME="${NAME:-minimax-h3-fl2va}"
GPUS="${GPUS:-0,1,2,3}"
HOST_PORT="${HOST_PORT:-19300}"
PROFILE="${PROFILE:-l40s}"
RESTART="${RESTART:-}"
WAIT="${WAIT:-1}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a _gpu_list <<<"$GPUS"
NGPU="${#_gpu_list[@]}"

die() { echo "serve.sh: $*" >&2; exit 1; }
run() { if [[ "$DRY_RUN" == "1" ]]; then printf '%q ' "$@"; echo; else "$@"; fi; }

check_inputs() {
  [[ -n "$MODEL_DIR" ]] || die "MODEL_DIR에 MiniMax-H3 루트 폴더를 지정하세요."
  [[ -d "$MODEL_DIR" ]] || die "모델 폴더가 없습니다: $MODEL_DIR"
  [[ -f "$MODEL_DIR/model_index.json" && -d "$MODEL_DIR/FL2VA" ]] \
    || die "루트 폴더가 아닌 것 같습니다 (model_index.json, FL2VA/ 필요): $MODEL_DIR"
  [[ -d "$OMNI_SRC/vllm_omni" ]] || die "vLLM-Omni 소스가 없습니다: $OMNI_SRC  ('bash $0 setup' 먼저 실행)"
}

# vllm serve 인자 (컨테이너 안 경로 기준)
vllm_args() {
  VLLM_ARGS=(
    serve /models/MiniMax-H3
    --omni --trust-remote-code
    --task-type fl2va
    --host 0.0.0.0 --port 8000
  )
  case "$PROFILE" in
    l40s)
      VLLM_ARGS+=(--num-gpus "$NGPU" --tensor-parallel-size "$NGPU" --usp 1
                  --vae-use-tiling --diffusion-attention-backend FLASH_ATTN) ;;
    l40s-offload)
      VLLM_ARGS+=(--num-gpus "$NGPU" --tensor-parallel-size "$NGPU" --usp 1
                  --vae-use-tiling --diffusion-attention-backend FLASH_ATTN
                  --enable-distributed-layerwise-offload --dlo-no-use-allgather --enforce-eager) ;;
    single)
      [[ "$NGPU" == "1" ]] || die "PROFILE=single은 GPU 1장만 지정하세요 (예: GPUS=4)"
      VLLM_ARGS+=(--num-gpus 1 --enable-cpu-offload --diffusion-attention-backend FLASH_ATTN) ;;
    *) die "PROFILE은 l40s, l40s-offload, single 중 하나입니다: $PROFILE" ;;
  esac
  # shellcheck disable=SC2206
  [[ -n "${EXTRA_ARGS:-}" ]] && VLLM_ARGS+=($EXTRA_ARGS)
  return 0
}

wait_ready() {
  [[ "$WAIT" == "1" && "$DRY_RUN" != "1" ]] || return 0
  local url="http://127.0.0.1:${HOST_PORT}/health" t0=$SECONDS
  echo ">> 준비 대기 중 ($url, 모델 로딩에 10분 이상 걸릴 수 있음). 진행 상황: bash $0 logs"
  while (( SECONDS - t0 < WAIT_TIMEOUT )); do
    if curl -sf "$url" >/dev/null 2>&1; then
      echo ">> 준비 완료 ($((SECONDS - t0))초). 테스트: H3_DURATION=4.0 ./gen.sh prompts/key_A.txt test 1"
      return 0
    fi
    if ! docker ps -q -f "name=^${NAME}$" | grep -q .; then
      docker logs --tail 40 "$NAME" 2>&1 || true
      die "컨테이너가 종료되었습니다. 메모리 부족이면 PROFILE=l40s-offload로 다시 시도하세요."
    fi
    sleep 15
  done
  die "${WAIT_TIMEOUT}초 안에 준비되지 않았습니다. 'bash $0 logs'로 확인하세요."
}

docker_start() {
  check_inputs
  command -v docker >/dev/null || die "docker 명령을 찾을 수 없습니다."
  if docker ps -a -q -f "name=^${NAME}$" | grep -q . && [[ "$DRY_RUN" != "1" ]]; then
    die "같은 이름의 컨테이너가 이미 있습니다: $NAME  ('bash $0 restart' 또는 NAME=... 지정)"
  fi
  local abs_model abs_src
  abs_model="$(cd "$MODEL_DIR" && pwd)"
  abs_src="$(cd "$OMNI_SRC" && pwd)"
  vllm_args
  local DOCKER_ARGS=(
    run -d
    --gpus "\"device=${GPUS}\""
    -p "${HOST_PORT}:8000"
    --ipc=host
    --name "$NAME"
    -v "${abs_model}:/models/MiniMax-H3:ro"
    -v "${abs_src}:/opt/vllm-omni:ro"
    -e PYTHONPATH=/opt/vllm-omni              # 최신 vLLM-Omni 소스를 이미지 패키지보다 먼저 사용
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn
    -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400     # 긴 생성이 타임아웃 나지 않게
    -e FLASHINFER_DISABLE_VERSION_CHECK=1
    -e HF_HUB_OFFLINE=1                       # HF 접속 없이 로컬 파일만 사용
    -e TRANSFORMERS_OFFLINE=1
  )
  [[ -n "$RESTART" ]] && DOCKER_ARGS+=(--restart "$RESTART")
  # shellcheck disable=SC2206
  [[ -n "${EXTRA_DOCKER_ARGS:-}" ]] && DOCKER_ARGS+=($EXTRA_DOCKER_ARGS)
  DOCKER_ARGS+=(--entrypoint vllm "$IMAGE")   # 이미지마다 엔트리포인트가 달라 vllm으로 고정
  echo ">> 컨테이너 $NAME 시작 (GPU $GPUS, PROFILE=$PROFILE, 포트 $HOST_PORT)"
  run docker "${DOCKER_ARGS[@]}" "${VLLM_ARGS[@]}"
  wait_ready
}

case "$ACTION" in
  setup)
    if [[ -d "$OMNI_SRC/.git" ]]; then
      run git -C "$OMNI_SRC" pull --ff-only
    else
      run git clone --depth 1 https://github.com/vllm-project/vllm-omni.git "$OMNI_SRC"
    fi
    run docker pull "$IMAGE"
    ;;
  start)   docker_start ;;
  stop)    run docker rm -f "$NAME" ;;
  restart) run docker rm -f "$NAME" >/dev/null 2>&1 || true; docker_start ;;
  logs)    exec docker logs -f "$NAME" ;;
  status)
    docker ps -a -f "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    if curl -sf "http://127.0.0.1:${HOST_PORT}/health" >/dev/null 2>&1; then
      echo "health: OK (http://127.0.0.1:${HOST_PORT})"
    else
      echo "health: (응답 없음)"
    fi
    ;;
  *) die "알 수 없는 동작: $ACTION (setup|start|stop|restart|status|logs)" ;;
esac
