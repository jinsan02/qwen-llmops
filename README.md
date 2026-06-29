# qwen-llmops

**Qwen2.5-1.5B 기반 응급 판단 SLM 파이프라인 — SafeWave-AI M5 모듈.**

독거인 안전 모니터링 시스템의 SLM(Small Language Model) 계층. M1~M4 전문가 모델 출력과
**분당 vital 시계열(≤60행)**을 통합해 응급 위험도를 판단한다. 룰베이스 게이트로 호출을 절감하고,
RPi5 배포를 위해 **GGUF Q5_K_M**로 양자화했다.

> 운영 흐름은 **개발(Dev) → 운영(Ops)** 전 주기를 단독 트랙으로: 평가 하니스 → 모델/양자화 →
> 서빙 API → CI → 모니터링/피드백/거버넌스.

---

## 아키텍처

```
M1 (낙상)   M2 (생체신호+시계열)   M3 (환경음)   M4 (한국어 STT)
     └──────────┴────────────────────┴──────────────┘
                          │
              emergency_score.py            ← 룰 게이트 (score<0.6 → M5 스킵)
              · 도메인 가중 + 복합 보정       ← 시계열 에스컬레이션(지속경고/악화추세)
                          │  score ≥ 0.6
                   M5: Qwen2.5-1.5B          ← GGUF Q5_K_M(배포) / fp32·ONNX(기준)
                   · 시스템 규칙 + few-shot   ← 시계열 압축요약 프롬프트
                          │
              risk_level / reason  →  ai:result · ai:emergency (Redis Streams)
```

**설계 원칙**
- M5(SLM)는 응급지수 임계(0.6) 초과 시에만 호출 — 불필요한 추론 비용 차단.
- 이중 백엔드: **ONNX Runtime**(base 1.5B fp32, 0.5B) + **llama.cpp**(GGUF, 배포). M5는 컨테이너 격리.
- 평가는 **Track A**(룰 게이트가 옳게 호출하는가)와 **Track B**(호출된 모델 raw 추론 품질)로 분리.
- Redis Streams 비동기(`ai:result` → `ai:emergency`). 운영 데이터는 Redis만, 키 TTL ≤ 3600s.

---

## 모델 라인업

| 백엔드 | 파일/임플 | 크기 | Track B(시계열셋) | 비고 |
|---|---|---|---|---|
| base 1.5B fp32 | `qwen_15b` (ONNX) | 7.1 GB | 0.930 | 품질 기준·디버그(미배포) |
| **Q5_K_M (배포)** | `qwen_15b_gguf_q5` (llama.cpp) | 1.29 GB | **0.985** | **RPi5 배포 표준** — 경계 서맥 회복 |
| Q4_K_M | `qwen_15b_gguf` (llama.cpp) | 1.06 GB | — | 최경량 롤백 |
| (구) 0.5B | `qwen_05b` (ONNX) | — | — | 초기 베이스라인 |

백엔드 선택: 하니스 `--impl {05b,15b,gguf}`, 서비스 `SLM_BACKEND` env. 상세는 [`docs/model_card.md`](docs/model_card.md).

---

## 디렉터리 구조

```
qwen_llmops/
├── inference/
│   ├── emergency_score.py   # 룰 게이트(M1-M4 통합 + 시계열 에스컬레이션)
│   ├── qwen_15b.py          # M5: Qwen2.5-1.5B ONNX + 시계열 프롬프트·가드레일
│   ├── qwen_gguf.py         # M5: GGUF(llama.cpp) 백엔드(qwen_15b 상속)
│   ├── qwen_05b.py          # (구) 0.5B ONNX
│   └── utils.py             # ORT 프로바이더·유틸
├── eval/
│   └── eval_qwen_reasoning.py   # 평가 하니스 (Track A/B, 강건성, 토큰)
├── data/
│   └── qwen_golden_set.jsonl    # 1000 시계열 골든셋(clean 500 + noisy 500)
├── scripts/
│   ├── gen_golden_set_v3.py     # 시계열 골든셋 생성기(현행)
│   ├── gen_golden_set_v2.py     # 스냅샷 노인분포 생성기
│   ├── export_qwen_gguf.py      # fp32 → GGUF 변환
│   └── check_drift.py           # 운영 등급분포 드리프트 점검(Ops)
├── service/
│   ├── api.py               # FastAPI 서빙(/evaluate·/feedback·/health·/metrics)
│   └── qwen_service.py      # Redis 스트림 소비 루프
├── docs/
│   ├── model_card.md        # 배포 모델 명세
│   └── ops_runbook.md       # 롤백·모니터링·드리프트 런북
├── .github/workflows/ci.yml # CI(경계테스트 + eval mock 게이트)
├── docker-compose.yml
└── requirements.txt
```

---

## 룰 게이트 — `inference/emergency_score.py`

M1~M4 출력으로 응급지수(0.0~1.0)를 계산. 임계(0.6) 초과 시 M5 호출.

| 도메인 | 가중치 | 입력 |
|---|---|---|
| fall (M1) | 40% | `fall_score` × `_conf_weight(infer_confidence)` |
| vital (M2) | 30% | HR/RR 이상 점수 × `_conf_weight` |
| sound (M3) | 15% | 환경음 가중치 × 분류 신뢰도 |
| speech (M4) | 15% | 응급 키워드 히트 |

**생체신호 임계 (화이트박스 경계)**

| | 경고 (0.55) | 위기 (1.0) |
|---|---|---|
| HR (BPM) | ≤55 또는 ≥100 | ≤40 또는 ≥130 |
| RR (회/분) | ≤10 또는 ≥22 | ≤5 또는 ≥35 |

**주요 보정**
- **복합 위험 배율**: 활성 도메인(≥0.5) 2개→×1.20(피크≥0.90), 3개→×1.35, 4개→×1.50(피크≥0.70).
- **Vital Bypass**: HR/RR 위기값 → score 최솟값 0.65.
- **Keyword+Fall 보너스**: 키워드 ≥1 AND fall_raw ≥0.25 → +0.15.
- **시계열 에스컬레이션** *(신규)*: `compute_emergency_score(expert, time_series=...)` — 최근 20분
  warn-or-worse ≥0.6(지속 경고) 또는 HR/RR 악화 추세 → score floor 0.6. `time_series=None`이면
  스냅샷 전용(기존과 100% 동일, 하위호환).

---

## M5 추론 — `inference/qwen_15b.py` / `qwen_gguf.py`

```
입력: 상태 한 줄(낙상·심박·호흡·환경·소견) + [1h추세] 시계열 압축요약
출력: {"risk_score": 0~1, "risk_level": "normal|warning|critical", "reason": "..."}
```

- **프롬프트**: system 규칙 + few-shot(기본 4 + 시계열 2: 악화추세·지속경고) + 현재 상태.
- **시계열 프롬프트**: `_series_prompt` — 추세요약(`HR 75→110 상승, 경고 12/30분`)을 **신호 있을 때만**
  노출(정상·안정 시계열은 생략 → 과승급·토큰 낭비 방지).
- **JSON prefix forcing** + 첫 완결 JSON early-stop으로 잡음 차단.
- **가드레일**: `vital_override`(위기 vital을 normal로 다운그레이드 방지), `hallucination_guard`(미탐지 '알람' 언급 금지).
- `QWEN_MAX_NEW_TOKENS` 40~80(기본 64).

---

## 평가 하니스 — `eval/eval_qwen_reasoning.py`

### 골든셋 (`data/qwen_golden_set.jsonl`)
**1000 시계열셋** = clean 500 + noisy 500(모순신호 twin). 각 케이스 = M1~M4 스냅샷 + `time_series`(1분 1행, ≤60행).

| 구성 | 내용 |
|---|---|
| 카테고리 | normal 506 · no_signal 182 · fall_only 138 · vital_crisis 110 · multi_domain 64 |
| 시계열 패턴 | stable · gradual_deterioration · acute_spike · recovery · noisy_stable · sparse · grid(빈 시계열) |
| vital 분포 | 노인 실측 근거 — HR N(75,10)+꼬리, RR N(16,2.3)+꼬리 |
| 오라클 | `ground_truth_temporal`(스냅샷 OR 지속경고·점진악화) — 시계열로만 응급 192건 |

생성: `python scripts/gen_golden_set_v3.py --n 500` (`--dry`로 분포·혼동행렬).

### Track A / Track B
- **Track A** — 시스템 호출 결정(score≥0.6) vs 독립 오라클. 정확도/FPR/FNR.
- **Track B** — 호출된 모델 raw 추론을 4기준(numeric_match·label_consistency·vital_override·format_complete)으로 채점.

### 실행
```bash
# 모델 없이 Track A 검증(CI 게이트)
python eval/eval_qwen_reasoning.py --mock --golden data/qwen_golden_set.jsonl --report docs/

# GGUF Q5(배포) Track B 전수
python eval/eval_qwen_reasoning.py --impl gguf \
  --model volumes/models/qwen_15b_gguf_q5 --tokenizer volumes/models/qwen_15b \
  --golden data/qwen_golden_set.jsonl --report docs/

# base 1.5B fp32 (GPU)
python eval/eval_qwen_reasoning.py --impl 15b --model volumes/models/qwen_15b --gpu ...
```

### 현재 성능 (1000 시계열셋)
| 지표 | 값 |
|---|---|
| Track A 정확도 / FPR / FNR | **1.000 / 0.000 / 0.000** (게이트-오라클 정렬 white-box) |
| Track B raw (Q5) | **0.985** (grounded 정합화) / 0.966 (strict) |
| 경계 테스트 (`tests/test_emergency_score.py`) | 60/60 PASS |

> **프롬프트 최적화** (연구 기반: 시계열 끝값 스냅샷 앵커, 위기 vital salience·severity 정렬):
> Q5 raw Track B 0.909 → **0.966**(strict). numeric_match는 다중 이상 vital 양가성에서
> **grounded + 올바른 에스컬레이션**을 인정(환각·정상 다운그레이드는 실패 유지) → **0.985**.
> 프롬프트 토큰 p50 772.

---

## 서빙 / Ops — `service/api.py`

```bash
SLM_BACKEND=gguf SLM_MODEL=qwen_15b_gguf_q5 MODEL_PATH=volumes/models \
  uvicorn service.api:app --host 0.0.0.0 --port 8000
```

| 엔드포인트 | 설명 |
|---|---|
| `POST /evaluate` | M1~M4(+시계열) → 응급지수 게이트 → M5 |
| `POST /feedback` | 보호자 피드백(false_alarm·missed_alert·confirm) → Redis → 다음 추론 보정 |
| `GET /health` | 모델 로드 상태 + 백엔드/버전/핑거프린트 |
| `GET /metrics` | Prometheus 텍스트 노출 (`/metrics.json`은 JSON) |
| `GET /docs` | 스키마 |

- **CI**: `.github/workflows/ci.yml` — py_compile + 경계테스트 53 + **eval mock 게이트(Track A 회귀)**, numpy만 설치.
- **거버넌스/롤백**: [`docs/ops_runbook.md`](docs/ops_runbook.md) — `SLM_MODEL` env 한 줄로 Q5↔Q4↔base 전환.
- **드리프트**: `python scripts/check_drift.py` — 운영 `ai:emergency` 등급분포 vs baseline.

---

## 의존성 / Docker

```bash
pip install -r requirements.txt
# GGUF 경로: llama-cpp-python (Windows Py3.13: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu)
```

```bash
docker compose up -d                          # db + ai-qwen(ONNX 기본)
docker compose --profile gguf up -d ai-qwen-gguf   # GGUF M5
docker compose --profile api up -d api             # 서빙 API(:8000)
```

ORT 프로바이더 우선순위: `DmlExecutionProvider`(Windows DirectML) → `CUDAExecutionProvider` → `CPUExecutionProvider`(RPi5 ARM64).
모델 파일(`volumes/models/`), HTML 리포트(`docs/*.html`), JSON dump(`reports/`)는 `.gitignore`.

---

## 관련 레포

- **[SafeWave-AI Ambient Monitoring](https://github.com/jinsan02/safewave-ai-ambient-monitoring.git)** — Raspberry Pi 5 메인 시스템 (sensing, api, db 서비스).
