"""Generates eval/prompts.jsonl entries p31-p50: long prompts with pasted code, file lists and logs.

  python eval/make_long_prompts.py      # rewrites the long set in eval/prompts.jsonl (short set kept)
"""
import json
import random
from pathlib import Path

R = random.Random(7)
P = []


def add(tier, lang, prompt, note=""):
    P.append((tier, lang, prompt, note))


# ---------------------------------------------------------------- helpers to make realistic long text
def app_log(n=60):
    lines = []
    t = 0
    for i in range(n):
        t += R.randint(1, 900)
        lvl = R.choices(["INFO", "INFO", "INFO", "WARN", "ERROR"], k=1)[0]
        msg = R.choice([
            "request completed path=/api/orders status=200 latency_ms={}".format(R.randint(8, 240)),
            "cache miss key=user:{}".format(R.randint(1000, 9999)),
            "retrying upstream call attempt={}".format(R.randint(1, 3)),
            "db pool size=20 active={} idle={}".format(R.randint(5, 20), R.randint(0, 15)),
            "payment webhook received id=evt_{}".format(R.randint(10 ** 5, 10 ** 6)),
            "timeout calling inventory-service after 3000ms",
        ])
        lines.append(f"2026-09-30T10:{(t // 60000) % 60:02d}:{(t // 1000) % 60:02d}.{t % 1000:03d} {lvl:5} {msg}")
    return "\n".join(lines)


FILE_LIST = """src/
  api/
    __init__.py
    routes_orders.py
    routes_users.py
    routes_products.py
    schemas.py
    deps.py
  services/
    order_service.py
    user_service.py
    product_service.py
    notification_service.py
  repositories/
    base.py
    order_repo.py
    user_repo.py
    product_repo.py
  core/
    config.py
    logging.py
    security.py
  utils/
    strings.py
    time.py
    pagination.py
tests/
  test_orders_api.py
  test_users_api.py
  test_order_service.py
  conftest.py"""

PY_FUNC = '''def merge_intervals(intervals, min_gap=0):
    """Merge overlapping [start, end] intervals."""
    if not intervals:
        return []
    items = sorted(intervals, key=lambda x: x[0])
    merged = [list(items[0])]
    for start, end in items[1:]:
        last = merged[-1]
        if start <= last[1] + min_gap:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    result = []
    for s, e in merged:
        if e - s < 0:
            raise ValueError(f"invalid interval {s}-{e}")
        result.append((s, e))
    return result


def total_covered(intervals, min_gap=0):
    return sum(e - s for s, e in merge_intervals(intervals, min_gap))


def find_free_slots(busy, day_start, day_end, min_len):
    free = []
    cursor = day_start
    for s, e in merge_intervals(busy):
        if s - cursor >= min_len:
            free.append((cursor, s))
        cursor = max(cursor, e)
    if day_end - cursor >= min_len:
        free.append((cursor, day_end))
    return free'''

GO_RACE = '''package cache

import (
    "sync"
    "time"
)

type entry struct {
    val     []byte
    expires time.Time
}

type Cache struct {
    mu      sync.RWMutex
    items   map[string]*entry
    hits    int
    misses  int
    onEvict func(key string)
}

func (c *Cache) Get(key string) ([]byte, bool) {
    c.mu.RLock()
    e, ok := c.items[key]
    c.mu.RUnlock()
    if !ok || time.Now().After(e.expires) {
        c.misses++
        if ok {
            c.mu.Lock()
            delete(c.items, key)
            c.mu.Unlock()
            c.onEvict(key)
        }
        return nil, false
    }
    c.hits++
    return e.val, true
}

func (c *Cache) Set(key string, val []byte, ttl time.Duration) {
    c.mu.Lock()
    defer c.mu.Unlock()
    if old, ok := c.items[key]; ok {
        old.val = val
        old.expires = time.Now().Add(ttl)
        return
    }
    c.items[key] = &entry{val: val, expires: time.Now().Add(ttl)}
}

func (c *Cache) janitor(interval time.Duration, stop <-chan struct{}) {
    t := time.NewTicker(interval)
    for {
        select {
        case <-t.C:
            c.mu.Lock()
            for k, e := range c.items {
                if time.Now().After(e.expires) {
                    delete(c.items, k)
                    c.onEvict(k)   // onEvict may call c.Get for metrics
                }
            }
            c.mu.Unlock()
        case <-stop:
            return
        }
    }
}'''

RACE_LOG = '''==================
WARNING: DATA RACE
Read at 0x00c0001a4010 by goroutine 37:
  example.com/svc/cache.(*Cache).Get()
      /src/cache/cache.go:24 +0x1a4
  example.com/svc/api.(*Handler).GetUser()
      /src/api/user.go:88 +0x2b1
Previous write at 0x00c0001a4010 by goroutine 41:
  example.com/svc/cache.(*Cache).Set()
      /src/cache/cache.go:44 +0x13c
  example.com/svc/worker.refresh()
      /src/worker/refresh.go:51 +0x96
==================
fatal error: all goroutines are asleep - deadlock! (observed once in staging, 2026-09-28 03:12)'''

SV_FIFO = '''module async_fifo #(parameter DW = 32, parameter AW = 4) (
  input  logic          wclk, wrst_n, winc,
  input  logic [DW-1:0] wdata,
  output logic          wfull,
  input  logic          rclk, rrst_n, rinc,
  output logic [DW-1:0] rdata,
  output logic          rempty
);
  logic [DW-1:0] mem [0:(1<<AW)-1];
  logic [AW:0] wptr, rptr;
  logic [AW:0] wptr_sync, rptr_sync;

  // write side
  always_ff @(posedge wclk or negedge wrst_n)
    if (!wrst_n) wptr <= '0;
    else if (winc && !wfull) begin
      mem[wptr[AW-1:0]] <= wdata;
      wptr <= wptr + 1'b1;
    end

  // read side
  always_ff @(posedge rclk or negedge rrst_n)
    if (!rrst_n) rptr <= '0;
    else if (rinc && !rempty) rptr <= rptr + 1'b1;

  assign rdata = mem[rptr[AW-1:0]];

  // pointer crossing (single flop, binary pointers)
  always_ff @(posedge rclk) wptr_sync <= wptr;
  always_ff @(posedge wclk) rptr_sync <= rptr;

  assign rempty = (rptr == wptr_sync);
  assign wfull  = (wptr[AW] != rptr_sync[AW]) && (wptr[AW-1:0] == rptr_sync[AW-1:0]);
endmodule'''

V_SMALL = '''// 4비트 카운터 모듈
module counter4 (
  input  wire clk,     // 클럭
  input  wire rst_n,   // 액티브 로우 리셋
  input  wire en,      // 카운트 인에이블
  output reg  [3:0] q  // 카운터 출력
);
  // 리셋 시 0으로 초기화하고, en이 1이면 1씩 증가
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) q <= 4'd0;
    else if (en) q <= q + 4'd1;
  end
endmodule'''

PROFILE = """Showing top 25 nodes out of 312
      flat  flat%   sum%        cum   cum%
    4.21s 31.20% 31.20%      6.80s 50.37%  encoding/json.(*decodeState).object
    1.92s 14.22% 45.42%      1.92s 14.22%  runtime.mallocgc
    1.10s  8.15% 53.57%      2.40s 17.78%  example.com/svc/pricing.(*Engine).applyRules
    0.88s  6.52% 60.09%      0.88s  6.52%  regexp.(*Regexp).doExecute
    0.71s  5.26% 65.35%      0.95s  7.04%  example.com/svc/pricing.matchCoupon
    0.63s  4.67% 70.02%      0.63s  4.67%  runtime.memmove
    0.52s  3.85% 73.87%      3.10s 22.96%  example.com/svc/pricing.(*Engine).Quote
    0.41s  3.04% 76.91%      0.41s  3.04%  sync.(*Mutex).Lock
    0.33s  2.44% 79.35%      0.33s  2.44%  runtime.mapaccess2_faststr
    0.29s  2.15% 81.50%      0.29s  2.15%  strings.ToLower"""

PRICING_GO = '''func (e *Engine) Quote(ctx context.Context, req QuoteRequest) (Quote, error) {
    e.mu.Lock()
    defer e.mu.Unlock()
    var rules []Rule
    if err := json.Unmarshal(e.rawRules, &rules); err != nil {   // rules re-parsed on every call
        return Quote{}, err
    }
    total := 0.0
    for _, item := range req.Items {
        price := item.UnitPrice * float64(item.Qty)
        for _, r := range rules {
            if r.Applies(item) {
                price = r.Apply(price)
            }
        }
        for _, c := range req.Coupons {
            if matchCoupon(strings.ToLower(c), item.SKU) {   // compiles regexp per call
                price *= 0.9
            }
        }
        total += price
    }
    return Quote{Total: total}, nil
}

func matchCoupon(code, sku string) bool {
    re := regexp.MustCompile("^" + code + "-[a-z0-9]+$")
    return re.MatchString(strings.ToLower(sku))
}'''

# ---------------------------------------------------------------- tier 0: qwen3.8-27b (long but easy)
add("qwen3.8-27b", "ko", "아래 애플리케이션 로그를 시간순으로 간단히 요약해줘. 어떤 종류의 메시지가 주로 나오는지 정도만 알면 돼.\n\n"
    + app_log(70), "긴 로그 + 단순 요약")
cfg = {"service": "order-api", "replicas": 3, "env": {f"FEATURE_{i}": bool(i % 2) for i in range(30)},
       "resources": {"cpu": "2", "memory": "4Gi"}, "ports": [{"name": "http", "port": 8080}, {"name": "metrics", "port": 9090}],
       "probes": {"liveness": {"path": "/healthz", "period": 10}, "readiness": {"path": "/ready", "period": 5}},
       "dependencies": [{"name": n, "url": f"http://{n}.svc:8080", "timeout_ms": 3000} for n in
                        ["inventory", "payment", "user", "notification", "pricing", "search"]]}
add("qwen3.8-27b", "ko", "이 JSON 설정을 YAML로 바꿔줘. 키 순서는 그대로 유지해줘.\n\n" + json.dumps(cfg, indent=2),
    "긴 JSON + 형식 변환")
notes = ("Weekly sync notes (Sep 29)\n" + "\n".join(
    f"- {who}: {what}" for who, what in [
        ("Mina", "finished the onboarding checklist draft and shared it in the team drive"),
        ("Joon", "will benchmark the new GPU nodes on Thursday and post numbers"),
        ("Alex", "raised that on-call handover notes are often incomplete; proposal to add a template"),
        ("Sora", "budget review is moved to next Tuesday because finance asked for more detail"),
        ("Mina", "asked everyone to update their OKR progress before Friday"),
        ("Joon", "the staging cluster will be down Saturday 2-4am for maintenance"),
        ("Alex", "new hire starts Monday; needs laptop, accounts and a buddy"),
        ("Sora", "reminder that the security training deadline is Oct 15"),
    ] * 3))
add("qwen3.8-27b", "en", "Translate these meeting notes into Korean, keep the bullet structure.\n\n" + notes,
    "긴 회의록 번역")
sql = ("select o.id, o.created_at, u.email, u.name, sum(oi.qty*oi.unit_price) as total, count(distinct oi.product_id) as products, "
       "max(p.category) as top_category from orders o join users u on u.id=o.user_id join order_items oi on oi.order_id=o.id "
       "join products p on p.id=oi.product_id left join refunds r on r.order_id=o.id where o.created_at >= '2026-09-01' "
       "and o.created_at < '2026-10-01' and r.id is null and u.country in ('KR','JP','US') and o.status in ('PAID','SHIPPED','DELIVERED') "
       "group by o.id, o.created_at, u.email, u.name having sum(oi.qty*oi.unit_price) > 100 order by total desc, o.created_at asc limit 500")
add("qwen3.8-27b", "ko", "이 SQL을 읽기 좋게 줄바꿈하고 들여쓰기해서 정리만 해줘. 로직은 바꾸지 마.\n\n" + sql,
    "긴 SQL 포맷팅")

# ---------------------------------------------------------------- tier 1: qwen3.8-fn
add("qwen3.8-fn", "ko", "아래 코드가 무슨 일을 하는지 함수별로 설명해줘. 특히 min_gap 파라미터가 어떻게 동작하는지.\n\n```python\n"
    + PY_FUNC + "\n```", "코드 붙여넣기 + 설명")
add("qwen3.8-fn", "ko", "아래 함수들에 대한 pytest 단위 테스트를 작성해줘. 경계 조건(빈 입력, 붙어 있는 구간, min_gap)도 포함해서.\n\n```python\n"
    + PY_FUNC + "\n```", "코드 붙여넣기 + 테스트 작성")
add("qwen3.8-fn", "en", "Running find_free_slots gives a wrong result: a slot at the very end of the day is missing when the last "
    "busy interval ends exactly min_len before day_end. Here's the code and the failing case. Fix the bug.\n\n```python\n"
    + PY_FUNC + "\n```\n\nFailing case:\n>>> find_free_slots([(9, 10), (13, 15)], 9, 17, 2)\n[(10, 13)]\nexpected [(10, 13), (15, 17)]",
    "코드 + 명확한 버그")
add("qwen3.8-fn", "ko", "우리 프로젝트 구조가 아래와 같아. 날짜 문자열을 파싱하는 공용 함수를 새로 하나 만들려고 하는데 어느 파일에 두는 게 맞을까? "
    "그리고 그 함수만 간단히 작성해줘.\n\n" + FILE_LIST, "파일 목록 + 작은 판단")
add("qwen3.8-fn", "ko", "아래 Verilog 모듈의 한국어 주석을 영어로 바꿔줘. 코드는 건드리지 마.\n\n```verilog\n" + V_SMALL + "\n```",
    "룰 오탐 확인용 (verilog 키워드지만 주석 번역)")

# ---------------------------------------------------------------- tier 2: sonnet
add("sonnet", "ko", "아래 구조의 FastAPI 프로젝트에서 주문 목록 API(GET /orders)에 커서 기반 페이지네이션을 추가해줘. "
    "routes_orders.py, order_service.py, order_repo.py, utils/pagination.py를 같이 수정해야 하고, 기존 offset 방식을 쓰는 "
    "클라이언트도 한 버전 동안은 깨지지 않게 해야 해. 테스트도 tests/test_orders_api.py에 추가해줘.\n\n" + FILE_LIST,
    "파일 목록 + 다중 파일 기능")
ci_log = "\n".join([
    "============================= test session starts ==============================",
    "collected 412 items",
    "tests/test_orders_api.py ........................................ [ 10%]",
    "tests/test_users_api.py .......................F................ [ 20%]",
    "tests/test_order_service.py .................................... [ 30%]",
] + [f"tests/test_misc_{i}.py ................................ [{30 + i * 5}%]" for i in range(1, 12)] + [
    "=================================== FAILURES ===================================",
    "______________________ test_update_profile_concurrently ________________________",
    "    def test_update_profile_concurrently(client, user):",
    "        with ThreadPoolExecutor(4) as ex:",
    "            futs = [ex.submit(client.patch, f'/users/{user.id}', json={'nickname': f'n{i}'}) for i in range(8)]",
    "        results = [f.result().status_code for f in futs]",
    ">       assert all(r == 200 for r in results)",
    "E       assert False",
    "E        +  where False = all(<generator object ...>)",
    "E       results = [200, 200, 409, 200, 200, 200, 500, 200]",
    "--------------------------- Captured log call ----------------------------------",
    "ERROR    sqlalchemy.pool:pool.py:412 Exception during reset or similar",
    "ERROR    app.api.routes_users:routes_users.py:77 deadlock detected while updating user_profiles",
    "========================= 1 failed, 411 passed in 92.31s =========================",
    "(local run: 412 passed. CI fails roughly 1 in 4 runs)"])
add("sonnet", "ko", "CI에서만 가끔 실패하는 테스트가 있어. 로컬에서는 항상 통과해. 아래 CI 로그 보고 원인 찾아서 고쳐줘. "
    "관련 코드는 src/api/routes_users.py, src/services/user_service.py, src/repositories/user_repo.py야.\n\n" + ci_log,
    "긴 CI 로그 + 간헐적 실패 디버깅")
client_srv = '''# client/payments.py
import httpx

class PaymentsClient:
    def __init__(self, base_url, timeout=3.0):
        self.http = httpx.Client(base_url=base_url, timeout=timeout)

    def charge(self, order_id, amount, currency="KRW"):
        r = self.http.post("/charges", json={"order_id": order_id, "amount": amount, "currency": currency})
        r.raise_for_status()
        return r.json()

    def refund(self, charge_id, amount=None):
        r = self.http.post(f"/charges/{charge_id}/refunds", json={"amount": amount})
        r.raise_for_status()
        return r.json()

# server/routes_charges.py
@router.post("/charges")
def create_charge(body: ChargeIn, db: Session = Depends(get_db)):
    charge = Charge(order_id=body.order_id, amount=body.amount, currency=body.currency, status="PENDING")
    db.add(charge)
    db.commit()
    result = gateway.charge(body.amount, body.currency)
    charge.status = "PAID" if result.ok else "FAILED"
    db.commit()
    return {"id": charge.id, "status": charge.status}

@router.post("/charges/{charge_id}/refunds")
def create_refund(charge_id: int, body: RefundIn, db: Session = Depends(get_db)):
    charge = db.get(Charge, charge_id)
    refund = Refund(charge_id=charge.id, amount=body.amount or charge.amount)
    db.add(refund)
    db.commit()
    gateway.refund(charge.gateway_id, refund.amount)
    return {"id": refund.id}'''
add("sonnet", "ko", "결제 클라이언트와 서버 코드야. 타임아웃이 나면 클라이언트가 재시도하면서 중복 결제가 생기고 있어. "
    "클라이언트에는 지수 백오프 재시도를, 서버에는 Idempotency-Key 헤더 기반 중복 방지를 넣어줘. 환불도 똑같이. "
    "DB 마이그레이션이 필요하면 alembic 리비전도 만들어줘.\n\n```python\n" + client_srv + "\n```",
    "두 모듈 코드 + 다중 파일 구현")
pkg = json.dumps({"name": "web-console", "version": "3.4.0", "private": True,
                  "dependencies": {"react": "^19.0.0", "react-dom": "^19.0.0", "react-router-dom": "^6.22.0",
                                   "@tanstack/react-query": "^4.36.1", "zustand": "^4.5.0", "formik": "^2.4.5",
                                   "styled-components": "^5.3.11", "recharts": "^2.10.0", "dayjs": "^1.11.10"},
                  "devDependencies": {"typescript": "^5.4.0", "vite": "^5.1.0", "@types/react": "^18.2.0",
                                      "eslint": "^8.56.0", "vitest": "^1.3.0"}}, indent=2)
build_err = """src/components/OrderTable.tsx:41:7 - error TS2786: 'Formik' cannot be used as a JSX component.
  Its type '<Values extends FormikValues = FormikValues, ExtraProps = {}>(props: FormikConfig<Values> & ExtraProps) => Element' is not a valid JSX element type.
src/pages/Dashboard.tsx:12:20 - error TS2322: Type '{ children: Element; }' is not assignable to type 'IntrinsicAttributes'.
src/hooks/useOrders.ts:8:3 - error TS2345: Argument of type 'string[]' is not assignable to parameter of type 'QueryKey | QueryFilters'.
node_modules/styled-components/dist/types.d.ts:120:5 - error TS2430: Interface 'StyledComponentBase' incorrectly extends ...
Found 37 errors in 14 files."""
add("sonnet", "ko", "React 18에서 19로 올렸더니 빌드가 깨졌어. package.json이랑 빌드 에러 붙일게. 필요한 라이브러리 업그레이드와 "
    "코드 수정까지 해서 빌드 통과시켜줘.\n\npackage.json:\n" + pkg + "\n\n빌드 에러:\n" + build_err,
    "의존성 업그레이드 + 여러 파일 수정")
spec = """## Spec: Bulk order export endpoint

Endpoint: POST /orders/exports
Purpose: let back-office users export orders matching a filter as CSV, delivered asynchronously.

Request body:
- filter.status: list of order statuses (optional)
- filter.created_from / created_to: ISO dates, max range 92 days (validation error otherwise)
- filter.user_ids: up to 500 ids
- columns: subset of [id, created_at, user_email, total, status, items_count]; default all
Behaviour:
- returns 202 with export_id; job runs in the existing Celery worker (queue: exports)
- job streams rows from the read replica in batches of 5,000 and uploads to S3 bucket `bo-exports/{yyyy}/{mm}/{export_id}.csv`
- GET /orders/exports/{export_id} returns status (queued/running/done/failed), row_count and a presigned URL (15 min) when done
- only users with role `backoffice:export` may call either endpoint; others get 403
- at most 3 running exports per user; the 4th returns 429
Non-functional:
- p95 under 2 minutes for 1M rows
- audit log entry on request and on download URL generation
- tests: API validation, permission checks, job happy path with moto S3, the 429 limit"""
add("sonnet", "en", "Implement this spec in our FastAPI + Celery codebase (structure below). Touch whatever files you need.\n\n"
    + spec + "\n\n" + FILE_LIST, "긴 스펙 문서 + 기능 구현")

# ---------------------------------------------------------------- tier 3: opus
add("opus", "ko", "아래 Go 캐시 코드에서 race detector가 데이터 레이스를 잡았고 스테이징에서 한 번 데드락도 났어. "
    "근본 원인을 분석하고, 락 구조를 다시 설계해서 고쳐줘. hits/misses 통계와 onEvict 콜백 동작은 유지해야 해.\n\n```go\n"
    + GO_RACE + "\n```\n\n" + RACE_LOG, "동시성 코드 + race 로그")
add("opus", "ko", "아래 비동기 FIFO의 CDC(clock domain crossing) 문제를 분석해줘. 포인터 동기화 방식, full/empty 판정, "
    "메타스테이빌리티 측면에서 문제를 짚고, 안전한 구조로 재설계해서 SystemVerilog로 다시 작성해줘.\n\n```systemverilog\n"
    + SV_FIFO + "\n```", "RTL CDC 분석 + 재설계 (룰은 sonnet 바닥)")
arch = """현재 구조 (모두 동기 HTTP 호출):
- order-api → inventory-service (재고 확인, p99 420ms)
- order-api → pricing-service (가격 계산, p99 310ms)
- order-api → payment-service (결제, p99 1.8s, 외부 PG 포함)
- order-api → notification-service (메일/푸시, p99 650ms)
- order-api → search-indexer (주문 검색 인덱싱, p99 900ms)
- 하나라도 실패하면 주문 전체가 500으로 실패, 하루 평균 실패율 2.3%
- 트래픽: 평시 120 rps, 이벤트 시 2,000 rps, 이벤트 때마다 payment 타임아웃 연쇄 발생
- DB: order-api가 단일 PostgreSQL에 주문/주문항목/결제상태를 같이 씀
제약: 결제는 반드시 한 번만, 재고는 초과판매 금지, 고객에게 3초 안에 접수 응답"""
add("opus", "ko", "주문 처리 구조를 이벤트 기반으로 바꾸려고 해. 아래 현재 구조와 제약을 보고, 어떤 호출을 비동기로 빼고 어떤 건 동기로 남길지, "
    "사가/아웃박스 패턴 적용 방식, 실패 보상 흐름, 데이터 일관성 전략까지 포함한 목표 아키텍처를 설계해줘.\n\n" + arch,
    "아키텍처 설계")
add("opus", "en", "Quote() is our hottest endpoint and p99 went from 40ms to 180ms after the coupon feature shipped. "
    "Here's the CPU profile and the code. Find the real bottlenecks, propose a design that removes them "
    "(including concurrency: Quote holds a global mutex), and implement it with benchmarks.\n\n"
    + PROFILE + "\n\n```go\n" + PRICING_GO + "\n```", "프로파일 + 성능 최적화")

# ---------------------------------------------------------------- tier 4: fable
corrupt = """증상: 일부 주문의 total 금액이 실제 항목 합계와 다름. 3~5일에 한 번, 피크 시간대에만, 매번 1~3건.
시스템: order-api(Go) → Kafka(orders.v2, 파티션 24) → billing-consumer(Java, 6개 인스턴스) → PostgreSQL(주/복제 2대, 비동기 복제)
          → Redis(주문 요약 캐시, TTL 10분) → reporting-service가 복제 DB와 Redis를 섞어서 읽음

billing-consumer 로그 (문제 주문 o-88123):
10:41:02.113 partition=7 offset=8812331 order=o-88123 event=ITEM_ADDED qty=2
10:41:02.114 partition=7 offset=8812332 order=o-88123 event=COUPON_APPLIED
10:41:02.511 rebalance: revoked [7,8], assigned [7,8,13]
10:41:02.902 partition=7 offset=8812331 order=o-88123 event=ITEM_ADDED qty=2   (재처리)
10:41:03.004 total updated o-88123 52000 -> 61000

PostgreSQL 복제 지연 (같은 시각): replica lag 1.8s ~ 4.2s
Redis: o-88123 summary set at 10:41:02.950 (from replica read)
order-api: 같은 주문에 대해 10:41:02.700 쿠폰 취소 요청 처리 (COUPON_REMOVED 발행, partition 7 offset 8812339)"""
add("fable", "ko", "몇 달째 못 잡고 있는 데이터 정합성 버그야. 아래 정보를 보고 가능한 원인 가설을 모두 세우고, 각 가설을 확인할 방법, "
    "그리고 근본적인 해결 설계(정확히 한 번 처리, 캐시 일관성, 복제 지연 고려)까지 제시해줘.\n\n" + corrupt,
    "다중 시스템 희귀 버그")
proto = """Protocol LeaseCache (simplified):
  nodes N1..Nk, one leader L holds the authoritative store.
  read(x):  if local lease(x) valid -> return local value
            else request lease from L; L grants lease(x, t) if no write to x is pending
  write(x, v): send to L; L marks x pending, sends INVALIDATE(x) to all lease holders,
               waits for ACKs or lease expiry (whichever first), applies v, clears pending
  clocks: bounded drift epsilon, lease duration T, message delay unbounded but eventually delivered
  leader failover: new leader waits T + epsilon before serving writes"""
add("fable", "en", "Here is a simplified lease-based cache coherence protocol. Formally specify it (TLA+ preferred), "
    "state the safety property (no node returns a value older than the last completed write) and liveness properties, "
    "and either prove them or find a counterexample — pay attention to leader failover and clock drift.\n\n" + proto,
    "형식 명세 + 증명")

assert len(P) == 20, len(P)

path = Path(__file__).with_name("prompts.jsonl")
rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
short = [dict(r, set="short") for r in rows if r.get("set", "short") == "short"][:30]
long_rows = [{"id": f"p{31 + i:02d}", "expected": t, "lang": l, "prompt": p, "note": n, "set": "long"}
             for i, (t, l, p, n) in enumerate(P)]
with open(path, "w", encoding="utf-8") as f:
    for r in short + long_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
lens = [len(r["prompt"]) for r in long_rows]
print(f"short {len(short)} + long {len(long_rows)}; long prompt chars: min {min(lens)}, max {max(lens)}, "
      f"avg {sum(lens) // len(lens)}")
