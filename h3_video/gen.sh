#!/usr/bin/env bash
# vLLM-Omni로 띄운 MiniMax-H3(FL2VA) 서버에 영상 생성을 요청하는 스크립트
#
# 사용법:
#   ./gen.sh <프롬프트파일> <출력이름> [첫프레임_이미지] [개수] [시작시드]
#
# 예:
#   ./gen.sh prompts/key_A.txt key_A                         # 첫 프레임 없음 = 텍스트로만 생성(T2VA), 3개
#   ./gen.sh prompts/cut_A.txt cut_A frames/key_A.png 3 42   # 첫 프레임 기반(FL2VA), 시드 42/43/44
#   ./gen.sh prompts/key_A.txt test 1                        # 첫 프레임 없이 개수만 지정해도 됨 (세 번째가 숫자면 개수)
#
# 첫 프레임 이미지는 이 PC에서 업로드돼요. 서버 쪽 경로를 맞출 필요 없어요.
set -euo pipefail

HOST="${H3_HOST:-http://127.0.0.1:19300}"   # serve.sh의 HOST_PORT와 맞추기. 다른 서버에서: H3_HOST=http://<GPU서버IP>:19300

# GPU 서버는 사내 주소라 프록시를 거치면 안 됩니다 (거치면 사내 프록시 차단 페이지 403).
_h="${HOST#*://}"; _h="${_h%%/*}"; _h="${_h%%:*}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$_h" no_proxy="${no_proxy:+$no_proxy,}$_h"

for _c in curl jq ffmpeg; do
  command -v "$_c" >/dev/null || { echo "$_c 명령이 필요해요 (예: sudo apt-get install -y $_c)"; exit 1; }
done
PROMPT_FILE="$1"
NAME="$2"
FRAME="${3:-}"
N="${4:-3}"
SEED0="${5:-42}"
# 세 번째 인자가 숫자이고 그런 파일이 없으면 "첫 프레임 없음 + 개수"로 해석 (예: ./gen.sh p.txt test 1)
if [[ "$FRAME" =~ ^[0-9]+$ && ! -f "$FRAME" ]]; then
  SEED0="${4:-42}"; N="$FRAME"; FRAME=""
fi
[[ "$FRAME" == "-" ]] && FRAME=""   # '-'도 첫 프레임 없음
DURATION="${H3_DURATION:-5.0}"   # 초 단위. 우리 컷은 5초로 뽑고 편집에서 자르기
WIDTH="${H3_WIDTH:-1344}"        # 16:9, 짧은 변 768
HEIGHT="${H3_HEIGHT:-768}"
ASPECT="${H3_ASPECT:-16:9}"      # 서버가 t2va에 aspect_ratio를 필수로 요구함
# 첫 프레임을 보내는 필드 이름. 레시피에 FL2VA용 이름이 명시돼 있지 않아서
# 서버의 /openapi.json에서 확인한 이름으로 바꿔 쓰세요 (기본값: input_reference)
IMAGE_FIELD="${H3_IMAGE_FIELD:-input_reference}"

if [ -n "$FRAME" ]; then
  TASK="fl2va"
  [ -f "$FRAME" ] || { echo "첫 프레임 파일이 없어요: $FRAME"; exit 1; }
else
  TASK="t2va"
fi

EXTRA=$(jq -cn --arg task "$TASK" --argjson dur "$DURATION" \
  '{task:$task, duration:$dur, audio_flow_shift:3.0}')

curl --fail -sS "$HOST/health" >/dev/null || { echo "서버가 응답하지 않아요: $HOST"; exit 1; }
mkdir -p out

for i in $(seq 0 $((N - 1))); do
  SEED=$((SEED0 + i))
  OUT="out/${NAME}_s${SEED}.mp4"
  echo "[$NAME] seed=$SEED 생성 중..."

  ARGS=(
    -F "prompt=<$PROMPT_FILE"
    -F "width=$WIDTH"
    -F "height=$HEIGHT"
    -F "aspect_ratio=$ASPECT"
    -F "fps=24"
    -F "num_inference_steps=50"
    -F "flow_shift=12"
    -F "seed=$SEED"
    --form-string "extra_params=$EXTRA"
  )
  if [ "$TASK" = "fl2va" ]; then
    ARGS+=(-F "${IMAGE_FIELD}=@${FRAME}")
  fi

  # 오래된 curl(7.76 미만)에는 --fail-with-body가 없어서, 상태 코드를 직접 받아 판단해요
  T0=$SECONDS
  CODE=$(curl -sS -X POST "$HOST/v1/videos/sync" "${ARGS[@]}" -o "$OUT" -w '%{http_code}') || CODE="000"
  if [[ "$CODE" != "200" ]]; then
    echo "[$NAME] 요청 실패 (HTTP $CODE). 서버 응답:"; head -c 800 "$OUT" 2>/dev/null; echo
    mv -f "$OUT" "${OUT%.mp4}.error.txt" 2>/dev/null || true
    echo "$(date '+%F %T') | $NAME | seed=$SEED | FAILED HTTP $CODE" >> out/log.txt
    continue
  fi

  # 영상과 오디오 트랙이 제대로 들어갔는지 확인
  if ffmpeg -v error -i "$OUT" -map 0:v:0 -map 0:a:0 -f null - 2>/dev/null; then
    echo "저장: $OUT ($((SECONDS - T0))초)"
  else
    echo "확인 실패(영상/오디오 트랙 이상): $OUT"; head -c 300 "$OUT"; echo
  fi

  echo "$(date '+%F %T') | $NAME | task=$TASK | seed=$SEED | frame=${FRAME:-none} | prompt=$PROMPT_FILE" >> out/log.txt
done
