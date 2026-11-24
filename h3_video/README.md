# h3_video: MiniMax-H3 로컬 영상 생성 (vLLM-Omni docker, L40S × 8)

비전 인트로 영상의 AI 컷 3개(A: 작년의 하루, B: 올해의 하루, C: 화이트보드 토론)를 로컬 서버에서 만드는 스크립트입니다.
모든 명령은 이 `h3_video/` 폴더에서 실행합니다.

| 파일 | 내용 |
|---|---|
| `serve.sh` | MiniMax-H3 FL2VA 서버를 docker로 실행 (`setup`/`start`/`stop`/`restart`/`status`/`logs`, 기본 포트 19300) |
| `gen.sh` | 서버에 생성 요청. 첫 프레임 이미지를 주면 FL2VA, 없으면 텍스트만으로 생성(T2VA) |
| `prompts/key_*.txt` | 키프레임용 프롬프트 (거의 정지된 장면을 만들어 프레임을 뽑는 용도) |
| `prompts/cut_*.txt` | 실제 컷 프롬프트 (첫 프레임에서 이어지는 동작과 소리) |

## 실행 순서

```bash
# 1. 모델 다운로드 (Hugging Face 모델 페이지에서 접근 승인 필요). 루트 폴더 전체를 받음
hf download MiniMaxAI/MiniMax-H3 --local-dir /data/models/MiniMax-H3

# 2. 빌드 전에: CUDA 13 기반 이미지가 이 서버 드라이버(CUDA 12.x)에서 GPU를 쓰는지 확인 (몇 분)
bash serve.sh check

#    최초 1회: vLLM-Omni 최신 소스를 받고, vLLM 버전까지 맞춘 docker 이미지 빌드 (수십 분)
#    사내 프록시·PyPI 미러가 있으면 함께 지정 (빌드 안의 apt, pip에 전달됨)
HTTPS_PROXY=http://<프록시>:<포트> HTTP_PROXY=http://<프록시>:<포트> \
PIP_INDEX_URL=http://<사내 PyPI 미러>/simple PIP_TRUSTED_HOST=<pip.conf의 trusted-host> \
bash serve.sh setup
#    설정만 바꿔 다시 빌드할 때 → bash serve.sh build
#    패키지 설치 단계에서 실패하면 → 같은 변수로 bash serve.sh diag (빌드와 같은 조건에서 인덱스 접근을 1분 안에 시험)

# 3. 서버 실행 (GPU 0~3, 텐서 병렬 4, 호스트 포트 19300)
MODEL_DIR=/data/models/MiniMax-H3 bash serve.sh
#    메모리 부족이면 → PROFILE=l40s-offload ... bash serve.sh restart
#    그래도 안 되면 → PROFILE=single GPUS=4 ... bash serve.sh restart
#    GPU·포트 바꾸기 → GPUS=4,5,6,7 HOST_PORT=19310 NAME=minimax-h3-2 ...
#    실행될 docker 명령만 보기 → DRY_RUN=1 MODEL_DIR=... bash serve.sh
bash serve.sh status | logs | stop

# 4. 첫 프레임 필드 이름 확인 (최초 1회). input_reference가 아니면 H3_IMAGE_FIELD로 지정
curl -s http://127.0.0.1:19300/openapi.json | jq '.paths["/v1/videos/sync"]'

# 5. 짧은 테스트
H3_DURATION=4.0 ./gen.sh prompts/key_A.txt test 1

# 6. 키프레임 영상 생성 → 프레임 추출 (B, C도 같은 방식)
./gen.sh prompts/key_A.txt key_A
mkdir -p frames && ffmpeg -ss 2 -i out/key_A_s42.mp4 -frames:v 1 frames/key_A.png

# 7. 실제 컷 생성 (기본 3개, 시드 42/43/44)
./gen.sh prompts/cut_A.txt cut_A frames/key_A.png
./gen.sh prompts/cut_B.txt cut_B frames/key_B.png
./gen.sh prompts/cut_C.txt cut_C frames/key_C.png
```

포트를 바꿨다면 `gen.sh`도 `H3_HOST=http://127.0.0.1:<포트>`로 맞춥니다.
결과는 `out/`에, 설정 기록은 `out/log.txt`에 저장됩니다.

## 주의

- 이미지는 vLLM-Omni 저장소의 `docker/Dockerfile.cuda`로 직접 빌드합니다 (`vllm-omni-h3:local`). vLLM-Omni 소스와 vLLM 버전이 짝이 맞아야 해서, 오래된 `vllm/vllm-omni:minimax-h3` 이미지에 최신 소스만 마운트하면 `compute_mm_prefix_ranges` 같은 vLLM API 불일치 에러가 납니다. 빌드 대신 Docker Hub의 `vllm/vllm-omni:nightly-<커밋>` 이미지를 `IMAGE=`로 지정해도 됩니다.
- 빌드되는 이미지는 CUDA 13 기반입니다. 호스트 드라이버가 CUDA 12.x(R535 등)여도 L40S 같은 데이터센터 GPU는 이미지 안의 forward compatibility 라이브러리로 실행될 수 있어서, 드라이버를 올리기 전에 `bash serve.sh check`로 먼저 확인합니다. `matmul OK`가 나오면 그대로 진행하고, `GPU를 쓸 수 없음`이 나오면 드라이버 업데이트가 필요합니다.
- 공식 레시피에 L40S 전용 프로필은 없습니다. `l40s` 프로필은 RTX 5090 프로필의 텐서 병렬 방식을 48GB × 4장에 맞춘 설정이라 처음 실행 때 조정이 필요할 수 있습니다.
- 로컬 버전은 프롬프트 보정(H3-Context-IR)과 2K 변환이 없어서 출력은 768p(1344×768)입니다. 업스케일은 남는 GPU 4장에서 SeedVR2로 하고, 필름 그레인은 업스케일 뒤에 입힙니다.
- 첫 프레임 필드 이름(`input_reference`)은 레시피에 명시돼 있지 않아 추정값입니다. 첫 결과에서 키프레임이 유지되는지 꼭 확인하세요.
- 길이는 5초로 뽑고 편집에서 3초로 자릅니다 (최소 4초).
- 빌드 안에서는 docker 데몬의 프록시 설정이 적용되지 않습니다. 셸의 `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY`를 build-arg로 넘기고, NVIDIA apt 저장소(`developer.download.nvidia.com`)는 빌드에 필요 없어서 기본으로 빼고 `apt-get update`를 합니다 (`SKIP_NVIDIA_APT=0`이면 유지). 원본 Dockerfile은 그대로 두고 조정한 사본을 `.build/Dockerfile.h3`에 만들어 빌드합니다.
