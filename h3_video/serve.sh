#!/usr/bin/env bash
# MiniMax-H3 (FL2VA: 텍스트→영상, 첫 프레임→영상)를 vLLM-Omni docker로 서빙합니다. (L40S 48GB 서버 기준)
#
# 사용법:
#   bash serve.sh check      # 빌드 전에: 기반 이미지(CUDA 13)가 이 서버 드라이버에서 GPU를 쓰는지 확인
#   bash serve.sh setup      # 최초 1회: vLLM-Omni 최신 소스를 받고 docker 이미지 빌드 (vLLM 버전까지 맞춘 이미지)
#   bash serve.sh diag       # 빌드가 패키지 설치에서 실패할 때: 빌드와 같은 조건(프록시, 미러)으로 setuptools 설치를 시험
#   bash serve.sh build      # 소스는 그대로 두고 이미지만 다시 빌드 (프록시·미러 설정을 바꿨을 때)
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
#   OMNI_SRC=./vllm-omni    # vLLM-Omni 소스 위치 (setup이 여기에 받고 이미지를 빌드)
#   IMAGE=vllm-omni-h3:local
#                           # setup이 vLLM-Omni 저장소의 docker/Dockerfile.cuda로 빌드하는 이미지.
#                           # 소스와 vLLM 버전이 짝이 맞아야 해서, 최신 소스만 마운트하는 방식은 쓰지 않습니다
#                           # (vllm/vllm-omni:minimax-h3 + 최신 소스 마운트 → vLLM API 불일치 에러).
#                           # Docker Hub의 vllm/vllm-omni:nightly-<커밋> 이미지를 지정해도 됩니다.
#   NAME=minimax-h3-fl2va   # 컨테이너 이름
#   GPUS=0,1,2,3            # --gpus '"device=..."' 에 들어갈 GPU 번호
#   HOST_PORT=19300         # 호스트 포트 (컨테이너 안은 항상 8000). 기존 8000/8001 서버와 겹치지 않게
#   PROFILE=l40s            # l40s: GPUS 개수만큼 텐서 병렬
#                           # l40s-offload: + 레이어 오프로드 (메모리 부족 시)
#                           # single: GPU 1장 + CPU 오프로드 (공식 프로필, 느림, 호스트 RAM 많이 필요)
#   RESTART=                # docker 재시작 정책 (영상 만들 때만 띄우는 용도라 기본은 미지정)
#   WAIT=1                  # start 후 /health 응답까지 대기 (최대 WAIT_TIMEOUT초)
#   WAIT_TIMEOUT=1800       # 33B 모델 두 개(DiT, 텍스트 인코더)를 올려서 오래 걸릴 수 있음
#   --- setup(이미지 빌드)용 ---
#   HTTP_PROXY/HTTPS_PROXY/NO_PROXY (소문자도 가능)
#                           # 셸에 설정돼 있으면 빌드 안의 apt, pip(uv)에 그대로 전달합니다.
#                           # docker 데몬의 프록시 설정은 이미지 pull에만 쓰이고 빌드 RUN 단계에는 전달되지 않습니다.
#   PIP_INDEX_URL=          # 사내 PyPI 미러 주소 (예: http://nexus.local/repository/pypi/simple). 빌드 안의 pip, uv에 사용
#   PIP_TRUSTED_HOST=       # pip.conf의 trusted-host 값 (예: nexus.local). 여러 개면 공백으로 구분.
#                           # 빌드는 pip가 아니라 uv로 설치해서, uv의 UV_INSECURE_HOST로도 같이 넘깁니다.
#   SKIP_NVIDIA_APT=1       # 빌드 중 NVIDIA apt 저장소(developer.download.nvidia.com)를 빼고 apt-get update
#                           # (빌드에 필요한 git, jq는 Ubuntu 저장소에 있음)
#   BUILD_NETWORK=host      # docker build --network. 프록시가 호스트의 127.0.0.1에 떠 있어도 닿도록 host
#   EXTRA_BUILD_ARGS=       # docker build에 덧붙일 인자
#
#   SKIP_VERIFY=0           # 1이면 start 전에 컨테이너 안에서 모델을 확인하는 단계를 건너뜀
#   EXTRA_DOCKER_ARGS=      # docker run에 덧붙일 인자
#   EXTRA_ARGS=             # vllm serve에 덧붙일 인자
#
# 참고: 공식 레시피에 L40S 전용 프로필은 없습니다. l40s 프로필은 RTX 5090(32GB×2) 프로필의
#       텐서 병렬 방식을 48GB×4에 맞춘 설정입니다. L40S는 NVLink가 없어 8장보다 4장이 통신 부담이 적습니다.
set -euo pipefail

ACTION="${1:-start}"
MODEL_DIR="${MODEL_DIR:-}"
OMNI_SRC="${OMNI_SRC:-./vllm-omni}"
IMAGE="${IMAGE:-vllm-omni-h3:local}"
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
  if [[ "$DRY_RUN" != "1" ]] && ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    die "이미지가 없습니다: $IMAGE  ('bash $0 setup'으로 빌드하거나 IMAGE=... 지정)"
  fi

  # 컨테이너 안에서도 호스트와 같은 절대경로로 마운트합니다.
  # HF 캐시(.../snapshots/<rev>)처럼 파일이 ../../blobs를 가리키는 심볼릭 링크면
  # 그 폴더만 마운트하면 링크가 깨지므로, 링크가 가리키는 상위 폴더까지 함께 마운트합니다.
  ABS_MODEL="$(cd "$MODEL_DIR" && pwd -P)"
  MOUNT_ROOT="$ABS_MODEL"
  local link target
  while IFS= read -r -d '' link; do
    target="$(readlink -f "$link")" || continue
    while [[ "$target" != "$MOUNT_ROOT"/* && "$MOUNT_ROOT" != "/" ]]; do
      MOUNT_ROOT="$(dirname "$MOUNT_ROOT")"
    done
  done < <(find "$ABS_MODEL" -type l -print0 2>/dev/null)
  [[ "$MOUNT_ROOT" == "/" ]] && die "모델 파일 링크가 너무 넓은 범위를 가리킵니다. hf download --local-dir로 받은 실제 폴더를 쓰세요."
  [[ "$MOUNT_ROOT" != "$ABS_MODEL" ]] && echo ">> 심볼릭 링크 대상까지 포함해 마운트: $MOUNT_ROOT"

  local d missing=()
  for d in text_encoder video_vae audio_vae; do
    [[ -e "$ABS_MODEL/$d" || -e "$ABS_MODEL/FL2VA/$d" ]] || missing+=("$d")
  done
  if (( ${#missing[@]} )); then
    echo "!! 공용 구성 요소 폴더를 찾지 못했습니다: ${missing[*]}" >&2
    echo "   FL2VA만 받았다면 전체를 받으세요: hf download MiniMaxAI/MiniMax-H3 --local-dir $ABS_MODEL" >&2
  fi
  return 0
}

# 서버를 띄우기 전에 같은 이미지·마운트로 vLLM-Omni가 모델을 알아보는지 확인합니다 (몇 초).
# "did not resolve to a registered Omni pipeline" 에러를 미리 잡기 위한 단계입니다.
verify_model_in_container() {
  [[ "$DRY_RUN" == "1" || "${SKIP_VERIFY:-0}" == "1" ]] && return 0
  echo ">> 컨테이너 안에서 모델 경로 확인: $ABS_MODEL"
  docker run --rm -v "${MOUNT_ROOT}:${MOUNT_ROOT}:ro" -e HF_HUB_OFFLINE=1 \
    --entrypoint python3 "$IMAGE" - "$ABS_MODEL" <<'PY' || die "컨테이너 안에서 모델을 인식하지 못했습니다 (위 내용 참고)"
import json, sys
from pathlib import Path
m = Path(sys.argv[1])
ok = True
idx = m / "model_index.json"
if not idx.is_file():
    print("  model_index.json: 컨테이너에서 보이지 않음 (마운트/링크 문제)"); sys.exit(1)
cls = json.load(open(idx)).get("_class_name")
print("  model_index.json _class_name:", cls)
print("  FL2VA/model_index.json:", (m / "FL2VA" / "model_index.json").is_file())
broken = [str(p.relative_to(m)) for p in m.rglob("*") if p.is_symlink() and not p.exists()][:5]
if broken:
    print("  깨진 링크:", broken); ok = False
from vllm_omni.diffusion.data import resolve_model_class_name
from vllm_omni.diffusion.registry import DiffusionModelRegistry
resolved = resolve_model_class_name(str(m))
supported = resolved in DiffusionModelRegistry.get_supported_archs()
print("  vLLM-Omni가 읽은 파이프라인:", resolved, "| 지원:", supported)
if not supported:
    print("  → 이 이미지의 vLLM-Omni가 이 체크포인트 형식을 모릅니다. 'bash serve.sh setup'으로 최신 소스로 다시 빌드하세요.")
    ok = False
sys.exit(0 if ok else 1)
PY
}

# vllm serve 인자 (컨테이너 안 경로 기준)
vllm_args() {
  VLLM_ARGS=(
    serve "$ABS_MODEL"
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
  verify_model_in_container
  vllm_args
  local DOCKER_ARGS=(
    run -d
    --gpus "\"device=${GPUS}\""
    -p "${HOST_PORT}:8000"
    --ipc=host
    --name "$NAME"
    -v "${MOUNT_ROOT}:${MOUNT_ROOT}:ro"         # 호스트와 같은 경로로 마운트 (링크 유지)
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn
    -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400     # 긴 생성이 타임아웃 나지 않게
    -e FLASHINFER_DISABLE_VERSION_CHECK=1
    -e HF_HUB_OFFLINE=1                       # HF 접속 없이 로컬 파일만 사용
    -e TRANSFORMERS_OFFLINE=1
  )
  [[ -n "$RESTART" ]] && DOCKER_ARGS+=(--restart "$RESTART")
  # shellcheck disable=SC2206
  [[ -n "${EXTRA_DOCKER_ARGS:-}" ]] && DOCKER_ARGS+=($EXTRA_DOCKER_ARGS)
  DOCKER_ARGS+=(--entrypoint vllm "$IMAGE")   # 이미지마다 엔트리포인트가 달라 vllm으로 고정 (빌드 이미지는 비어 있음)
  echo ">> 컨테이너 $NAME 시작 (GPU $GPUS, PROFILE=$PROFILE, 포트 $HOST_PORT)"
  run docker "${DOCKER_ARGS[@]}" "${VLLM_ARGS[@]}"
  wait_ready
}

# 사내 미러(PIP_INDEX_URL)와 trusted-host는 사내 주소라 프록시를 거치면 안 됩니다.
# 거치면 사내 프록시의 차단 페이지(403)가 돌아옵니다. 그래서 NO_PROXY/no_proxy에 자동으로 추가합니다.
add_internal_hosts_to_no_proxy() {
  local hosts="" h
  if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    h="${PIP_INDEX_URL#*://}"; h="${h%%/*}"; h="${h%%:*}"
    hosts+=" $h"
  fi
  hosts+=" ${PIP_TRUSTED_HOST:-}"
  NO_PROXY="${NO_PROXY:-${no_proxy:-}}"; no_proxy="${no_proxy:-$NO_PROXY}"   # 대소문자 중 하나만 있어도 맞춤
  for h in $hosts; do
    h="${h%%:*}"
    [[ -z "$h" ]] && continue
    [[ ",${NO_PROXY:-}," == *",$h,"* ]] || NO_PROXY="${NO_PROXY:+$NO_PROXY,}$h"
    [[ ",${no_proxy:-}," == *",$h,"* ]] || no_proxy="${no_proxy:+$no_proxy,}$h"
  done
  # 프록시가 설정된 경우에만 의미가 있으므로 그때만 내보냄
  if [[ -n "${HTTPS_PROXY:-}${https_proxy:-}${HTTP_PROXY:-}${http_proxy:-}" ]]; then
    export NO_PROXY no_proxy
  fi
}

# 사내망용으로 조정한 Dockerfile을 만들고 빌드합니다. 원본(docker/Dockerfile.cuda)은 건드리지 않습니다.
#  - FROM 바로 뒤에: NVIDIA apt 저장소 제거(SKIP_NVIDIA_APT=1), 사내 PyPI 미러 지정(PIP_INDEX_URL)
#  - 프록시는 docker 기본 build-arg(HTTP_PROXY 등)로 넘겨서 모든 RUN 단계에 적용
build_image() {
  local src_df="$OMNI_SRC/docker/Dockerfile.cuda"
  local out_dir; out_dir="$(cd "$(dirname "$0")" && pwd)/.build"
  local df="$out_dir/Dockerfile.h3"
  local inject=""
  if [[ "${SKIP_NVIDIA_APT:-1}" == "1" ]]; then
    inject+=$'RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list\n'
  fi
  if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    inject+="ENV PIP_INDEX_URL=${PIP_INDEX_URL} UV_DEFAULT_INDEX=${PIP_INDEX_URL} UV_INDEX_URL=${PIP_INDEX_URL}"$'\n'
  fi
  if [[ -n "${PIP_TRUSTED_HOST:-}" ]]; then
    inject+="ENV PIP_TRUSTED_HOST=\"${PIP_TRUSTED_HOST}\" UV_INSECURE_HOST=\"${PIP_TRUSTED_HOST}\""$'\n'
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$out_dir"
    # 첫 FROM 줄 바로 뒤에 inject 내용을 넣음
    awk -v inject="$inject" '!done && /^FROM /{print; printf "%s", inject; done=1; next} {print}' "$src_df" > "$df"
  fi

  if [[ -n "${PIP_INDEX_URL:-}" && "${PIP_INDEX_URL%/}" != */simple ]]; then
    echo "!! PIP_INDEX_URL이 /simple로 끝나지 않습니다: $PIP_INDEX_URL  (pip/uv는 'simple' 형식 주소가 필요해요)" >&2
  fi
  add_internal_hosts_to_no_proxy
  local BUILD_ARGS=(build -f "$df" -t "$IMAGE")
  [[ -n "${BUILD_NETWORK-host}" ]] && BUILD_ARGS+=(--network "${BUILD_NETWORK-host}")
  local v
  for v in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
    [[ -n "${!v:-}" ]] && BUILD_ARGS+=(--build-arg "$v=${!v}")
  done
  # shellcheck disable=SC2206
  [[ -n "${EXTRA_BUILD_ARGS:-}" ]] && BUILD_ARGS+=($EXTRA_BUILD_ARGS)
  BUILD_ARGS+=("$OMNI_SRC")

  echo ">> 이미지 빌드: $IMAGE (LMCache 컴파일 포함이라 수십 분 걸릴 수 있음)"
  echo "   프록시: ${HTTPS_PROXY:-${https_proxy:-(없음)}} | PyPI 미러: ${PIP_INDEX_URL:-(없음)} | trusted-host: ${PIP_TRUSTED_HOST:-(없음)} | NVIDIA apt 제외: ${SKIP_NVIDIA_APT:-1}"
  [[ -n "${NO_PROXY:-}" ]] && echo "   NO_PROXY: $NO_PROXY"
  run docker "${BUILD_ARGS[@]}"
}

# 빌드 안과 같은 조건(기반 이미지, --network, 프록시, 미러)으로 패키지 인덱스 접근을 시험합니다.
diag_index() {
  local base="${CHECK_IMAGE:-vllm/vllm-openai:v0.34.0}"
  add_internal_hosts_to_no_proxy
  local DARGS=(run --rm --entrypoint bash)
  [[ -n "${BUILD_NETWORK-host}" ]] && DARGS+=(--network "${BUILD_NETWORK-host}")
  local v
  for v in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
    [[ -n "${!v:-}" ]] && DARGS+=(-e "$v=${!v}")
  done
  if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    DARGS+=(-e "PIP_INDEX_URL=$PIP_INDEX_URL" -e "UV_DEFAULT_INDEX=$PIP_INDEX_URL" -e "UV_INDEX_URL=$PIP_INDEX_URL")
  fi
  if [[ -n "${PIP_TRUSTED_HOST:-}" ]]; then
    DARGS+=(-e "PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST" -e "UV_INSECURE_HOST=$PIP_TRUSTED_HOST")
  fi
  echo ">> $base 안에서 패키지 인덱스 확인 (프록시: ${HTTPS_PROXY:-${https_proxy:-(없음)}}, 미러: ${PIP_INDEX_URL:-(없음, pypi.org)})"
  run docker "${DARGS[@]}" "$base" -c '
    echo "--- 이미지/전달된 설정 (UV_*, PIP_*, *_proxy)"
    env | grep -iE "^(uv_|pip_|https?_proxy|no_proxy)" | sed "s/^/  /" || true
    [ -f /etc/pip.conf ] && { echo "--- /etc/pip.conf"; sed "s/^/  /" /etc/pip.conf; }
    [ -f /etc/uv/uv.toml ] && { echo "--- /etc/uv/uv.toml"; sed "s/^/  /" /etc/uv/uv.toml; }
    IDX="${UV_DEFAULT_INDEX:-https://pypi.org/simple}"
    echo "--- 인덱스 응답: ${IDX%/}/setuptools/"
    python3 - "${IDX%/}/setuptools/" <<PY
import os, ssl, sys, urllib.request
url = sys.argv[1]
ctx = None
if os.environ.get("UV_INSECURE_HOST"):   # trusted-host와 같은 효과: 인증서 검증 생략
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
try:
    r = urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "text/html"}), timeout=30, context=ctx)
    body = r.read().decode("utf-8", "replace")
    n = body.count("setuptools-")
    print("  HTTP", r.status, "| setuptools 파일 링크", n, "개", "(정상)" if n else "(링크 없음: 인덱스 주소가 /simple 형식이 아니거나 미러에 패키지가 없음)")
    if not n: print("  응답 앞부분:", body[:300].replace(chr(10), " "))
except Exception as e:
    print("  접속 실패:", e)
    if "403" in str(e):
        print("  → 사내 프록시가 막은 응답입니다. 사내 미러라면 NO_PROXY에 들어갔는지(위 설정 목록), 외부 pypi.org라면 PIP_INDEX_URL로 사내 미러를 지정하세요.")
PY
    echo "--- uv로 setuptools 설치 시험 (실제 설치는 안 함)"
    uv pip install --dry-run --python "$(command -v python3)" "setuptools>=77.0.3,<81.0.0" wheel "setuptools-scm>=8" 2>&1 | tail -8 | sed "s/^/  /"
  '
}

case "$ACTION" in
  diag) diag_index ;;
  build)
    [[ -f "$OMNI_SRC/docker/Dockerfile.cuda" ]] || die "vLLM-Omni 소스가 없습니다: $OMNI_SRC ('bash $0 setup' 먼저 실행)"
    build_image
    ;;
  setup)
    if [[ -d "$OMNI_SRC/.git" ]]; then
      run git -C "$OMNI_SRC" pull --ff-only
    else
      run git clone --depth 1 https://github.com/vllm-project/vllm-omni.git "$OMNI_SRC"
    fi
    [[ "$DRY_RUN" == "1" || -f "$OMNI_SRC/docker/Dockerfile.cuda" ]] || die "Dockerfile이 없습니다: $OMNI_SRC/docker/Dockerfile.cuda"
    build_image
    ;;
  check)
    # 호스트 드라이버가 CUDA 12.x(R535 등)여도, 데이터센터 GPU(L40S 등)는 이미지 안의 forward compatibility
    # 라이브러리로 CUDA 13 이미지를 실행할 수 있습니다. 빌드(수십 분) 전에 기반 이미지로 먼저 확인합니다.
    CHECK_IMAGE="${CHECK_IMAGE:-vllm/vllm-openai:v0.34.0}"
    echo ">> $CHECK_IMAGE 로 GPU ${_gpu_list[0]} 확인"
    run docker run --rm --gpus "\"device=${_gpu_list[0]}\"" --entrypoint bash "$CHECK_IMAGE" -c '
      ls -d /usr/local/cuda*/compat 2>/dev/null | sed "s/^/compat: /"
      python3 -c "
import torch
print(\"torch\", torch.__version__, \"| CUDA runtime\", torch.version.cuda)
assert torch.cuda.is_available(), \"GPU를 쓸 수 없음 (드라이버/compat 문제)\"
x = torch.randn(2048, 2048, device=\"cuda\")
print(\"GPU:\", torch.cuda.get_device_name(0), \"| matmul OK:\", float((x @ x).sum()) == float((x @ x).sum()))
"'
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
  *) die "알 수 없는 동작: $ACTION (check|diag|setup|build|start|stop|restart|status|logs)" ;;
esac
