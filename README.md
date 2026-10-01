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

```bash
bash download_jev27b.sh /models/JEV-27B
MODEL_DIR=/models/JEV-27B PORT=8000 bash serve/serve_jev27b.sh        # 분류 전용
# 80GB 미만 GPU라면 TP=2, 생성 겸용으로 쓰려면 MODE=shared

pip install -r requirements.txt
python serve/smoke_test.py --url http://localhost:8000 --model-dir /models/JEV-27B
```

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
- 상태 확인: `curl localhost:8787/health`, `curl localhost:8787/sessions`

### 라우팅 규칙

1. `claude-tier-<이름>` 모델로 요청하면 그 티어로 고정
2. 요청 종류(`x-claude-code-request-class`)별 정책: 부가 요청(제목 생성 등)은 가장 낮은 티어, 컴팩션은 현재 세션 티어, Explore 서브에이전트는 qwen3.8-fn
3. **새 프롬프트의 첫 요청만** 분류 (`x-claude-code-prompt-id`, 없으면 tool_result 여부로 판단). 도구 루프 중간 요청은 세션에 고정된 티어 그대로
4. 분류 = 룰(키워드·파일 확장자, 상향 전용) + JEV choice 판정 + 확신도 낮으면 한 단계 상향
5. 세션 정책 `upgrade_only`: 한 세션에서 티어는 올라가기만 함 (모델 전환에 따른 프롬프트 캐시 손실 방지). 컴팩션 직후에는 다시 내려갈 수 있음
6. 연속 도구 에러 3회 → 한 단계 상향
7. 컨텍스트가 티어의 `max_context`를 넘으면 상향
8. 내부 백엔드 연결 실패/5xx → 다음 티어로 재시도. Anthropic 사용량 한도(429, retry-after 60초 이상) → 가장 높은 내부 티어로 하향하고 30분간 Anthropic 건너뜀

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
