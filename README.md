# claude_auto: JEV-27B 기반 Claude Code 모델 라우팅

Claude Code 요청을 로컬 프록시가 받아, 프롬프트 난이도에 따라 내부 Qwen 모델 또는 Anthropic 모델로 보냅니다.
분류는 [autotrust/JEV-27B](https://huggingface.co/autotrust/JEV-27B)의 System 1(타입 판정)을 사용합니다.

```
claude_auto ──► 로컬 프록시 (127.0.0.1:8787) ──► qwen3.8-27b   (internal)
                 │  룰 + JEV 분류 + 세션 고정      ├► qwen3.8-fn    (internal)
                 └► JEV-27B vLLM (jev-decision)    ├► sonnet        (anthropic)
                                                   ├► opus          (anthropic)
                                                   └► fable         (anthropic)
```

## 구성

| 경로 | 내용 |
|---|---|
| `download_jev27b.sh` | hf CLI 없이 wget으로 JEV-27B 다운로드 + SHA256 검증 |
| `serve/serve_jev27b.sh` | 모델 카드 권장 설정으로 vLLM 서빙 (`decision` / `shared` 모드) |
| `serve/smoke_test.py` | 서빙 확인: 모델 목록, System 2 생성, System 1 판정, 지연시간 |
| `eval/prompts.jsonl` | 기대 티어가 달린 평가용 프롬프트 30개 (한국어 21, 영어 9) |
| `eval/run_eval.py` | 30개 프롬프트 분류 → 정확도, 과소/과대 라우팅, 혼동 행렬, CSV |
| `proxy/server.py` | 라우팅 프록시 (Anthropic Messages API 호환) |
| `proxy/config.example.yaml` | 프록시·티어·룰 설정 예시 |
| `bin/claude_auto` | 프록시를 띄우고 Claude Code를 그 프록시로 실행하는 런처 |
| `jevtools/` | JEV 클라이언트와 라우팅 로직 (프록시와 평가가 공유) |
| `tests/` | 목업 백엔드로 하는 프록시 테스트 |

## 1. JEV-27B 서빙과 확인

기존 qwen3.8-27b 서버와 같은 형태(`docker run -d --gpus '"device=..."' -p ... --ipc=host --name ...`)로 띄웁니다.

```bash
bash download_jev27b.sh /data/models/JEV-27B

# 분류 전용 (max-model-len 4096)
IMAGE=<기존 qwen3.8-27b 서버가 쓰는 vLLM 이미지> \
MODEL_DIR=/data/models/JEV-27B GPUS=0,1,2,3 HOST_PORT=8000 \
  bash serve/serve_jev27b.sh start          # 컨테이너 시작 후 준비될 때까지 대기

# 생성 겸용: 27b 풀의 4번째 서버로도 사용 (max-model-len은 다른 27b 서버와 맞추세요)
MODE=shared MAX_LEN=131072 ... bash serve/serve_jev27b.sh start

bash serve/serve_jev27b.sh status | logs | stop | restart
DRY_RUN=1 bash serve/serve_jev27b.sh       # 실행될 docker 명령만 확인

pip install -r requirements.txt
python serve/smoke_test.py --url http://localhost:8000 --model-dir /data/models/JEV-27B
```

- 모델 폴더는 컨테이너의 `/models/JEV-27B`에 읽기 전용으로 마운트하고, `HF_HUB_OFFLINE=1`로 HF 접속 없이 실행합니다.
- `TP`는 기본으로 `GPUS` 개수를 씁니다.
- 기존 스크립트의 다른 옵션은 `EXTRA_DOCKER_ARGS`(docker run 쪽: `-e`, `-v` 등)와 `EXTRA_ARGS`(vllm serve 쪽)로 그대로 넘기면 됩니다.
- 이미지는 Qwen3.5 계열, `lm_head` LoRA, `--logprobs-mode`, `allowed_token_ids`를 지원해야 합니다. 기존 qwen3.8-27b 서버가 쓰는 이미지를 먼저 시도하세요.
- 호스트에서 바로 실행하려면 `RUNTIME=native`.

`smoke_test.py`는 모델 카드 예시(환불 요청 P(true)≈0.978, 공급사 질문 → dual_source)와 비교해 결과를 판정합니다.

## 2. 30개 프롬프트 분류 테스트

```bash
cp proxy/config.example.yaml ~/.claude_auto/config.yaml     # jev.base_url, jev.model_dir 수정
python eval/run_eval.py --shuffles 3
# 또는 설정 없이 주소만 지정
python eval/run_eval.py --config proxy/config.example.yaml \
    --jev-url http://gpu01:8000 --model-dir /models/JEV-27B --shuffles 3
```

출력에서 볼 것:
- **JEV 단독 vs 게이트+룰 적용** 정확도 비교
- **under-routed** (기대보다 낮은 모델로 감, 품질 위험) — 0에 가까워야 함
- **[ko] / [en]** 별 정확도와 평균 확신도 — JEV-27B는 영어 코퍼스로 학습되어 한국어가 약할 수 있음
- **선택지 순서 변경 시 바뀐 비율** — 모델 카드 기준 약 7%
- 결과 CSV는 `eval/results/`에 저장 (`p_<티어>` 열에 확률)

`--mock`을 붙이면 서버 없이 스크립트 동작만 확인합니다.

티어 설명(`tiers[].criteria`)과 `jev.confidence_threshold`가 결과에 가장 큰 영향을 줍니다. 결과를 보고 이 두 가지를 먼저 조정하세요.

## 3. 프록시와 claude_auto

```bash
pip install -r requirements.txt
mkdir -p ~/.claude_auto && cp proxy/config.example.yaml ~/.claude_auto/config.yaml
# config.yaml 의 TODO 항목 수정 (백엔드 주소, 모델 이름, 인증)
export INTERNAL_LLM_TOKEN=...          # 내부 서버 토큰 (config의 backends.*.auth.env)
ln -s "$PWD/bin/claude_auto" ~/.local/bin/claude_auto

claude_auto                            # 평소 claude 대신 실행
```

- `claude`는 그대로 두고, `claude_auto`로 실행한 세션만 프록시를 거칩니다 (`--settings`로 이 세션에만 주입).
- 프록시가 없으면 백그라운드로 띄우고, 마지막 `claude_auto` 세션이 끝나면 종료합니다 (`CLAUDE_AUTO_KEEP_PROXY=1`로 유지).
- 세션 안에서 `/model opus` → Opus 강제, `/model sonnet` → 자동 라우팅 복귀, `/model claude-tier-fable` → Fable 강제.
- 로그: `~/.claude_auto/proxy.log`, 라우팅 결정: `~/.claude_auto/decisions.jsonl`
- 상태 확인: `curl localhost:8787/health` (티어별 분류·처리 건수 포함), `/sessions`, `/backends` (서버별 부하), `/budget`

### 라우팅 규칙

1. `claude-tier-<이름>` 모델로 요청하면 그 티어로 고정
2. 요청 종류(`x-claude-code-request-class`)별 정책: 부가 요청(제목 생성 등)은 가장 낮은 티어, 컴팩션은 현재 세션 티어, Explore 서브에이전트는 qwen3.8-fn
3. **새 프롬프트의 첫 요청만** 분류 (`x-claude-code-prompt-id`, 없으면 tool_result 여부로 판단). 도구 루프 중간 요청은 세션에 고정된 티어 그대로
4. 분류 = 룰(키워드·파일 확장자, 상향 전용) + JEV choice 판정 + 확신도 낮으면 한 단계 상향
5. 세션 정책 `upgrade_only`: 한 세션에서 티어는 올라가기만 함 (모델 전환에 따른 프롬프트 캐시 손실 방지). 컴팩션 직후에는 다시 내려갈 수 있음
6. 연속 도구 에러 3회 → 한 단계 상향
7. 컨텍스트가 티어의 `max_context`를 넘으면 상향
8. 예산 압박(`budget`)이 있으면 새 프롬프트부터 외부 티어를 내부로 치환 (아래 참고)
9. 서버 선택: 같은 세션은 같은 서버로 (vLLM 프리픽스 캐시 유지). 그 서버가 바쁘거나 죽었으면 같은 풀의 다른 서버 → `overflow` 티어 → 한 단계 위 티어 순서로 시도
10. Anthropic 사용량 한도(429, retry-after 60초 이상) → 가장 높은 내부 티어로 하향하고 30분간 Anthropic 건너뜀

### 서버 풀과 부하

- `backends.<이름>.servers`에 같은 모델 서버를 여러 대 적습니다. 예시 설정은 qwen3.8-27b 3대(+JEV-27B를 `MODE=shared`로 띄우면 4번째, `weight` 낮게), flash-next 1대입니다.
- `metrics: true`면 각 서버의 vLLM `/metrics`(실행 중/대기 중 요청 수)를 5초마다 읽습니다. 프록시가 개발자마다 하나씩 돌기 때문에, 다른 사람의 부하는 서버 지표로만 보입니다. 프록시가 쉬는 동안(2분 이상 요청 없음)은 읽지 않습니다.
- 대기열이 `max_waiting`을 넘으면 '바쁨'입니다. flash-next는 1대라 `max_waiting: 2`로 낮게 두고, 바쁘면 27b 풀로 넘기도록 `overflow: [qwen3.8-27b]`를 걸었습니다. Anthropic으로는 새지 않습니다.
- `/health`의 `stats.decided`(분류 결과)와 `stats.served`(실제 처리)를 비교하면 overflow가 얼마나 일어나는지 보입니다. flash-next 비중이 서버 대수 비율(약 20%)보다 계속 높다면 `tiers[].criteria`를 조정해 일부 작업을 27b 티어로 옮기세요.

### 외부 모델 예산

레벨은 `normal` → `tight` → `critical`이고, 아래 세 신호 중 가장 나쁜 값을 씁니다.

| 신호 | 방식 |
|---|---|
| 예산 조회 (`budget.source`) | HTTP URL 또는 명령어가 JSON을 반환. `{"level": "tight"}` 또는 `{"remaining": 3200, "limit": 10000}`. 남은 예산 비율이 남은 기간 비율 × 0.8보다 작으면 `tight`, 5% 미만이면 `critical` |
| 응답 헤더 | Anthropic 응답의 `anthropic-ratelimit-unified-*`에서 경고/거부 상태, 높은 사용률을 감지 (휴리스틱) |
| 수동 | `curl -XPOST localhost:8787/budget -d '{"level":"tight","minutes":240}'`, 해제는 `{"level":null}` |

레벨별 동작은 `budget.policy`로 정합니다. 기본값은 `tight`에서 sonnet급만 flash-next로, `critical`에서 sonnet/opus/fable급 모두 내부로 보냅니다. `tight`는 새 프롬프트부터만 적용해서 진행 중인 작업은 같은 모델로 끝냅니다. `/model`로 직접 고른 모델은 치환하지 않습니다. 컨텍스트가 내부 모델에 안 들어가는 요청은 치환하지 않습니다.

사내 사용량 조회 수단(API, MCP 등)이 있으면 그것을 감싸 JSON을 출력하는 스크립트를 `source: {type: command}`로 연결하면 됩니다. 이렇게 하면 "월초에 다 쓰고 남은 기간은 내부 모델만" 대신 **월말까지 페이스를 맞춰 쓰는** 방식이 됩니다.

### 도입 순서 권장

1. `mode: shadow` + `shadow_tier: sonnet` 로 1~2주 운영 — 실제로는 지금처럼 동작하고 `decisions.jsonl`의 `would_route`만 쌓임
2. 로그로 티어 설명·임계값·키워드 룰 조정
3. `mode: route` 전환

### 인증 (`client_auth`)

- `claude_login` (기본): Claude Code의 claude.ai 엔터프라이즈 로그인이 그대로 전달되고, anthropic 백엔드는 `auth.mode: passthrough`로 좌석 과금을 그대로 사용합니다. `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` / `apiKeyHelper`가 설정돼 있으면 로그인 대신 그 키가 쓰이므로 런처가 경고합니다.
- `token`: 로컬 더미 토큰을 보내고, 프록시가 백엔드별 실제 키(`api_key`/`bearer`)를 붙입니다.

## 테스트

```bash
python -m pytest -q tests/
```

## 주의

- 내부 백엔드는 Anthropic Messages API(`/v1/messages`) 호환이어야 합니다. 프록시는 `model` 필드만 바꾸고 본문과 SSE 스트림을 그대로 전달합니다.
- Anthropic은 게이트웨이를 통해 Claude Code를 Claude가 아닌 모델로 라우팅하는 구성을 공식 지원하지 않습니다. 내부 모델 경로의 문제는 자체적으로 확인해야 합니다.
- JEV-27B는 영어 코퍼스로 학습되어 한국어 판정 품질은 평가 결과로 확인해야 합니다.
