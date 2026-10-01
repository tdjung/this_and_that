# claude_auto 프로젝트 인수인계 (2026-10-01)

폐쇄망 환경에서 논의를 이어가기 위한 정리입니다. 이 문서만 보고도 맥락을 파악할 수 있게 썼습니다.
코드 저장소: `github.com/tdjung/this_and_that` (README.md에 사용법 상세)

---

## 1. 배경과 목표

- 회사는 Claude Code를 엔터프라이즈 요금제로 제공하지만, 월 사용량이 1~2주 만에 소진됩니다. 남은 기간은 사내 오픈소스 모델(Qwen)을 씁니다.
- 개인당 평균 월 약 $180 수준이고 사람마다 다릅니다. 팀 단위 배정은 먼저 쓰는 사람이 더 많이 쓰는 구조입니다.
- 사람들은 모델 전환을 귀찮아하거나 손해(캐시 등)를 걱정해서 엔터프라이즈 모델부터 소진합니다.
- **목표**: 요청 난이도에 따라 내부 모델과 외부 모델을 자동으로 나눠, 외부 토큰을 아끼면서 품질 손실은 최소화합니다.

### 사용 가능한 모델

| 이름 | 위치 | 서버 | 비고 |
|---|---|---|---|
| qwen3.8-27b | 내부 | 3대 (+JEV-27B 겸용 시 4대) | 1M 컨텍스트 (YaRN factor 4) |
| qwen3.8-flash-next (fn) | 내부 | 1대 | 용량 병목 |
| Sonnet 5.5 | Anthropic | - | 입력 $2 / 출력 $10 / 캐시 읽기 $0.20 (1M 토큰당) |
| Opus 5.5 | Anthropic | - | $4 / $20 / $0.20 |
| Fable 5.1 | Anthropic | - | $10 / $50 / $0.25 → 월 $180로는 사실상 사용 불가 |

---

## 2. 현재 구조

```
claude_auto (런처)
  └─ Claude Code ──ANTHROPIC_BASE_URL──► 로컬 프록시 127.0.0.1:8787 (proxy/server.py)
                                           ├─ 룰 + JEV-27B 분류 (새 프롬프트의 첫 요청만)
                                           ├─ 세션 고정, 실패 시 상향, 예산 정책
                                           ├─► 내부 qwen3.8-27b 풀 / flash-next
                                           └─► Anthropic (Sonnet / Opus / Fable)
```

### 분류기: autotrust/JEV-27B
- TypeSafe의 비공개 분류 모델 Jev 1.13을 Qwen3.8-27B 위에 증류한 오픈 모델입니다. 같은 가중치로 일반 생성(System 2)과 분류(System 1, LoRA `jev-decision`)를 모두 서빙합니다.
- 선택지(최대 16개) 중 하나를 확률과 함께 반환합니다. 한 번의 prefill로 끝나서 빠릅니다.
- 영어 코퍼스로 학습되어, 한국어 품질은 별도 검증이 필요합니다.

### 저장소 구성
| 경로 | 내용 |
|---|---|
| `download_jev27b.sh` | HF CLI 없이 wget으로 모델 다운로드 + SHA256 검증 |
| `serve/serve_jev27b.sh`, `serve/smoke_test.py` | vLLM 서빙 스크립트와 동작 확인 |
| `eval/prompts.jsonl`, `eval/run_eval.py` | 평가 프롬프트 50개(짧은 30 + 긴 20)와 평가 스크립트 |
| `proxy/server.py`, `pool.py`, `budget.py`, `usage.py` | 라우팅 프록시, 서버 풀, 예산, 사용액 계산 |
| `proxy/config.example.yaml` | 티어, 룰, 예산 정책 등 모든 설정 |
| `bin/claude_auto` | 프록시를 띄우고 Claude Code를 그 프록시로 실행 |
| `tests/test_proxy.py` | 목업 백엔드 테스트 35개 (모두 통과) |

### 라우팅 로직 요약
1. `/model claude-tier-<이름>`으로 직접 고르면 그 티어로 고정
2. 요청 종류(`x-claude-code-request-class` 헤더): 부가 요청은 최저 티어, 컴팩션은 현재 세션 티어, Explore 서브에이전트는 fn
3. **새 프롬프트의 첫 요청만 분류**(`x-claude-code-prompt-id`). 도구 루프 중간 요청은 세션 티어를 그대로 씀
4. 분류 = 키워드·확장자 룰(상향 전용, 최소 sonnet) + JEV choice + 확신도 0.55 미만이면 한 단계 상향(최대 sonnet까지)
5. 비용 통제: fable은 자동 선택 안 함(→opus), opus는 P(opus 이상) ≥ 0.75일 때만
6. 세션 정책 `upgrade_only`: 세션 안에서는 티어가 올라가기만 함 (캐시 보호)
7. 연속 도구 에러 3회면 한 단계 상향, 컨텍스트가 티어 한도를 넘으면 상향
8. 예산 레벨(normal/tight/critical)에 따라 외부 티어를 치환. tight: opus→sonnet, critical: 외부 전부→fn
9. 서버 선택: 세션별로 같은 서버(vLLM 프리픽스 캐시 유지). 바쁘면(`/metrics` 대기열) 다른 서버 → overflow 티어. fn과 27b는 서로 overflow로 연결되고, Anthropic으로는 새지 않음
10. Anthropic 429(한도 소진) → 내부로 하향하고 30분간 건너뜀

---

## 3. 현재 상태

### 서빙 (동작 확인됨)
최종 docker 명령 요지:
```bash
docker run -d --gpus '"device=0,1,2,3"' -p 8001:8000 --ipc=host --name <이름> \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e http_proxy=... -e https_proxy=... -e HF_HUB_OFFLINE=1 \
  -v /경로/JEV-27B:/model:ro \
  vllm/vllm-openai:nightly /model \
  --served-model-name jev-27b --tensor-parallel-size 4 --max-model-len 1000000 \
  --hf-overrides '{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}' \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.90 --enable-prefix-caching --mamba-cache-mode align \
  --enable-lora --max-lora-rank 32 --lora-modules jev-decision=/model/adapter_vllm \
  --logprobs-mode processed_logprobs \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3-coder \
  --max-num-batched-tokens 8192 --default-chat-template-kwargs '{"enable_thinking": false}'
```
(YaRN은 `--hf-overrides` 대신 config.json 사본을 덮어 마운트하는 방식도 가능. 실제 적용 여부는 로그로 확인 필요)

### 서빙 중 겪은 문제와 해결
| 증상 | 원인 | 해결 |
|---|---|---|
| `text_config ... num_attention_heads` 에러 | JEV-27B는 텍스트 전용(`qwen3_5_text`)이라 `text_config`가 없음 | `--hf-overrides`를 최상위 `rope_parameters`로 |
| `rope_theta` | 원래 명령의 1,000,000이 틀림 | 모델 config 값 10,000,000 (**기존 27b 서버 설정도 확인 필요**) |
| `Expected Qwen3_5Config but found Qwen3_5TextConfig` | vLLM v0.26.0의 텍스트 전용 Qwen3.5 버그 | `vllm/vllm-openai:nightly` (0.30.1rc1.dev396) 사용 |
| `max_model_len > derived` | 환경변수 누락 | `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` |
| `No adapter found` | 경로 오타 `adaptor_vllm` | `adapter_vllm` |
| 파이썬에서 502/403, curl은 정상 | `http_proxy` 환경변수 때문에 사내 프록시를 경유 | 코드에서 내부 서버는 프록시 우회(`trust_env=False`)하도록 수정 |

### 평가 결과 (짧은 30개)
- 기대값(직접 붙인 기준선)과 JEV 일치 **29/30**. 불일치는 p11("이 에러 의미가 뭐야? TypeError... parser.py:88")로, 기대는 fn, JEV는 27b였습니다. JEV 쪽 해석도 타당합니다.
- 확신도(p01~p30): `1,1,1,1,1,1,0.66,0.98,1,0.97,0.7,0.99,0.99,0.79,0.94,0.9,0.89,0.68,0.88,0.99,0.98,0.84,0.85,0.86,0.98,0.96,0.51,0.95,0.57,0.48`
  - 단순 질문은 전부 1.0. fable 그룹(p25~p30)이 가장 낮음(최저 0.48). 최상위 티어의 경계가 가장 흐릿합니다.
- **주의**: 프롬프트와 티어 설명을 같은 사람이 만들어 표현이 겹치므로, 실제보다 쉬운 시험입니다.
- 긴 프롬프트 20개(코드·로그·파일 목록 포함, 384~4,939자)는 추가만 했고 **아직 실제 서버로 돌리지 않았습니다** (`python eval/run_eval.py --set long`).
- YaRN과 fp8 KV 캐시 상태에서 `smoke_test.py`의 분류 확률이 모델 카드 값(환불 예시 0.978)과 맞는지 **확인 필요**.

---

## 4. 열린 논의 (다음에 이어갈 주제)

### 4-1. 개인별 예산이 다름 → 고정 $180은 부적절
현재는 `budget.monthly_limit_usd: 180` 고정값과, 프록시가 직접 계산한 사용액(`/usage`)으로 페이스를 판단합니다.

문제:
- 사람마다 한도가 다릅니다.
- 프록시는 `claude_auto`를 거친 사용만 볼 수 있습니다. 일반 `claude` 실행이나 다른 PC 사용은 빠집니다.
- 팀 풀이면 개인 잔액이라는 개념 자체가 모호합니다.

방향 후보:
- **(권장) 회사 시스템에서 "개인의 남은 금액/한도"를 받아오기.** 사내 MCP나 API가 있으면 `budget.source`(command 또는 http)로 연결합니다. 페이스 판단은 비율 기반이라 한도가 사람마다 달라도 그대로 동작합니다. 로컬 사용액 계산은 동기화 사이의 추정용으로만 씁니다.
- Anthropic 응답 헤더(`anthropic-ratelimit-unified-*`)에 엔터프라이즈 좌석의 사용률이 실리는지 확인합니다. 실린다면 별도 API 없이 개인 한도 대비 사용률을 알 수 있습니다(현재는 휴리스틱으로만 읽는 중).
- 사용자별로 `monthly_limit_usd`를 설정 파일이나 환경변수로 받는 방식은 최소한의 대안입니다.
- 팀 풀의 선착순 문제는 라우터로 해결되지 않습니다. 운영 정책(1인 기본 할당 + 공용 풀)과 함께 논의해야 합니다.

### 4-2. "대부분 27b로 충분할 수 있다" → 판단 근거를 무엇으로?
현재 5단계 티어 설명은 **사전 가정**이고, JEV는 그 설명에 맞춰 문장을 분류할 뿐입니다. "27b로 충분한가"는 문장만 보고 예측하기보다 **결과로 측정**해야 합니다.

근거를 만드는 방법:
1. **오프라인 재실행(가장 확실)**: 실제 업무 요청을 모아 같은 작업을 27b와 Sonnet에 각각 시키고, 결과를 비교합니다(테스트 통과 여부, 사람 평가). 작업 유형별로 "27b 성공률"이 나옵니다.
2. **온라인 신호**: 섀도 모드나 27b 기본 운영 중 실패 신호를 작업 유형별로 집계합니다. 사용자가 `/model`로 올림, 같은 요청 반복, 연속 도구 에러, 세션 중도 포기 같은 것들입니다.
3. **구조 전환 후보 — "예측" 대신 "기본 27b + 실패 시 상향"(cascade)**:
   - 대부분이 27b로 해결된다면, 모든 요청을 27b로 시작하고 명확히 어려운 신호(룰, 높은 확신도의 hard 판정)가 있을 때만 외부로 보냅니다.
   - 판단의 부담이 "난이도 예측"에서 "실패 감지"로 옮겨갑니다.
   - 대가: 실패한 시도의 시간 낭비, 그리고 모델 전환 시 캐시 손실(전환은 세션 초반에 하거나 요약 후 새로 시작).
4. **분류 단순화**: 5단계 choice 대신 "중형 오픈 모델로 실패할 가능성이 높은가?"라는 yes/no(`noul`) 질문 하나로 바꾸고, 1번 데이터로 임계값을 맞추는 방식이 더 정확하고 해석하기 쉬울 수 있습니다.
5. flash-next는 1대뿐이라 독립 티어보다 27b의 overflow나 특정 용도로만 쓰는 편이 나을 수 있습니다.

### 4-3. 모델 × 추론 강도(effort/thinking) 2차원 라우팅
현재는 모델만 고르고 effort는 다루지 않습니다. 사다리를 "모델 × 생각 강도"로 만들 수 있습니다.
- 예: 27b(no think) → 27b(think) → Sonnet(low) → Sonnet(medium) → Opus(low) → Opus(xhigh)

고려할 점:
- **내부 Qwen의 thinking**: 비용은 $가 아니라 GPU 시간과 지연입니다. 현재 서버 기본값은 `enable_thinking: false`입니다. 요청별로 켜는 방식(`chat_template_kwargs`)이 Claude Code → 내부 서버 경로에서 어떻게 전달되는지 확인이 필요합니다.
- **Anthropic의 effort**: 출력 토큰(가장 비싼 항목)을 크게 늘립니다. 그래서 "Sonnet 높은 effort vs Opus 낮은 effort"처럼 모델 선택보다 effort가 더 큰 비용 레버일 수 있습니다.
- **프록시가 effort를 바꿀 수 있는지**: Claude Code가 이미 effort/thinking 관련 필드를 보내고 있습니다. 프록시가 이를 덮어쓰는 것은 기술적으로 가능하지만, 문서상 게이트웨이는 본문을 바꾸지 않는 것이 원칙입니다(thinking 서명 검증 등). 필드 이름과 영향은 **검증이 필요**합니다.
- **분류**: JEV의 `score`(0~5) 질문으로 "필요한 추론 깊이"를 따로 물어 effort에 매핑할 수 있습니다. 모델 선택(choice)과 effort(score)를 한 번의 요청에서 같이 물을 수 있습니다.
- 제안: 처음에는 비용이 들지 않는 **내부 thinking on/off만** 도입하고, Anthropic effort는 Claude Code 기본값을 유지한 채 사용액 데이터를 본 뒤 결정합니다.

---

## 5. 다음 단계 제안
1. `smoke_test.py`로 YaRN/fp8 상태의 분류 확률 확인 → 문제 있으면 분류용 서버를 YaRN 없이 분리
2. `run_eval.py --set long` 실행 → 긴 입력과 한국어 품질 확인
3. 팀의 실제 요청 20~50개로 평가셋 구성 (사람이 기대 티어 라벨링)
4. 4-2의 오프라인 재실행으로 "27b 성공률"을 작업 유형별로 측정 → 티어 구조 재설계(cascade 또는 yes/no)
5. 개인 예산 데이터 소스 확보 (사내 MCP/API 또는 응답 헤더)
6. `mode: shadow`로 1~2주 운영 → `~/.claude_auto/decisions.jsonl` 분석 후 `mode: route`

## 6. 확인이 필요한 사항
- 기존 qwen3.8-27b 서버의 `rope_theta` 값 (1e6으로 설정돼 있다면 품질 저하 가능)
- 내부 Qwen 서버가 Anthropic `/v1/messages` 형식을 직접 받는지 (지금 Claude Code가 직접 붙어 있으니 그렇다고 가정 중)
- Claude Code는 `claude-auto` 같은 모르는 모델 ID를 200K 컨텍스트로 가정함 → 1M을 실제로 쓰려면 런처/프록시 조정 필요
- Anthropic은 게이트웨이를 통해 Claude Code를 Claude가 아닌 모델로 라우팅하는 구성을 공식 지원하지 않음 (내부 모델 경로의 문제는 자체 확인 필요)
