# h3_video: MiniMax-H3 로컬 영상 생성 (vLLM-Omni, L40S × 8)

비전 인트로 영상의 AI 컷 3개(A: 작년의 하루, B: 올해의 하루, C: 화이트보드 토론)를 로컬 서버에서 만드는 스크립트입니다.
모든 명령은 이 `h3_video/` 폴더에서 실행합니다.

| 파일 | 내용 |
|---|---|
| `serve.sh` | MiniMax-H3 FL2VA 서버 실행 (`l40s` → `l40s-offload` → `single` 순으로 시도) |
| `gen.sh` | 서버에 생성 요청. 첫 프레임 이미지를 주면 FL2VA, 없으면 텍스트만으로 생성(T2VA) |
| `prompts/key_*.txt` | 키프레임용 프롬프트 (거의 정지된 장면을 만들어 프레임을 뽑는 용도) |
| `prompts/cut_*.txt` | 실제 컷 프롬프트 (첫 프레임에서 이어지는 동작과 소리) |

## 실행 순서

```bash
# 1. 설치: H3 지원은 vLLM-Omni에 있음
uv pip install --torch-backend=cu130 vllm==0.33.0
git clone --depth 1 https://github.com/vllm-project/vllm-omni.git && (cd vllm-omni && uv pip install -e .)

# 2. 다운로드 (Hugging Face 모델 페이지에서 접근 승인 필요). 루트 폴더 전체를 받음
hf download MiniMaxAI/MiniMax-H3 --local-dir /models/MiniMax-H3

# 3. 서버 실행 (GPU 0~3, TP4)
MODEL_DIR=/models/MiniMax-H3 ./serve.sh l40s
#    메모리 부족이면 → ./serve.sh l40s-offload, 그래도 안 되면 → ./serve.sh single

# 4. 첫 프레임 필드 이름 확인 (최초 1회). input_reference가 아니면 H3_IMAGE_FIELD로 지정
curl -s http://127.0.0.1:8000/openapi.json | jq '.paths["/v1/videos/sync"]'

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

결과는 `out/`에, 설정 기록은 `out/log.txt`에 저장됩니다.

## 주의

- 공식 레시피에 L40S 전용 프로필은 없습니다. `l40s` 프로필은 RTX 5090 프로필의 텐서 병렬 방식을 48GB × 4장에 맞춘 설정이라 처음 실행 때 조정이 필요할 수 있습니다.
- 로컬 버전은 프롬프트 보정(H3-Context-IR)과 2K 변환이 없어서 출력은 768p(1344×768)입니다. 업스케일은 남는 GPU 4장에서 SeedVR2로 하고, 필름 그레인은 업스케일 뒤에 입힙니다.
- 첫 프레임 필드 이름(`input_reference`)은 레시피에 명시돼 있지 않아 추정값입니다. 첫 결과에서 키프레임이 유지되는지 꼭 확인하세요.
- 길이는 5초로 뽑고 편집에서 3초로 자릅니다 (최소 4초).
