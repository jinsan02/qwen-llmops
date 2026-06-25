# qwen-llmops

Qwen-0.5B ONNX 기반 응급 판단 LLM 파이프라인 — SafeWave-AI M5 모듈.

독거인 안전 모니터링 시스템의 SLM(Small Language Model) 계층으로, M1~M4 전문가 모델 출력을 통합해 응급 위험도를 판단합니다.

---

## 아키텍처

```
M1 (낙상)   M2 (생체신호)   M3 (환경음)   M4 (한국어 STT)
     └──────────┴──────────────┴──────────────┘
                       │
            emergency_score.py       ← 룰베이스 1차 판단 (임계값 < 0.6 → M5 스킵)
                       │
               qwen_05b.py (M5)      ← Qwen-0.5B ONNX, 임계값 ≥ 0.6 시 호출
                       │
              risk_level / reason    → ai:result Redis 스트림
```

**설계 원칙**:
- M5(SLM)는 임계값(0.6) 초과 시에만 호출 — 불필요한 추론 비용 차단
- ONNX Runtime 전용 (PyTorch/TF 런타임 금지)
- Redis Streams 기반 비동기 파이프라인 (`ai:result`, `ai:emergency`)

---

## 디렉터리 구조

```
qwen_llmops/
├── inference/
│   ├── emergency_score.py   # 룰베이스 응급지수 계산 (M1-M4 통합)
│   ├── qwen_05b.py          # M5: Qwen-0.5B ONNX 추론 엔진
│   └── utils.py             # ORT 프로바이더, 유틸
├── eval/
│   ├── eval_qwen_reasoning.py   # 평가 하니스 (정확도/오탐율/미탐율)
│   └── eval_qwen_accuracy.py    # 경량 정확도 검증
├── data/
│   └── qwen_golden_set.jsonl    # 100-case 골든셋 (화이트박스 BVA 포함)
├── scripts/
│   ├── export_m5_qwen_onnx.py   # Qwen → ONNX 변환
│   └── quantize_qwen_int8.py    # INT8 양자화
├── docs/                        # HTML 평가 리포트 (gitignored)
├── reports/                     # JSON 평가 결과 (gitignored)
├── volumes/models/              # ONNX 모델 파일 (gitignored)
└── docker-compose.yml
```

---

## 모듈 상세

### `inference/emergency_score.py` — 룰베이스 응급지수

M1~M4 출력을 받아 응급지수(0.0~1.0)를 계산합니다.

| 도메인 | 가중치 | 입력 |
|---|---|---|
| fall (M1) | 40% | `fall_score` × `_conf_weight(infer_confidence)` |
| vital (M2) | 30% | HR/RR 이상 점수 × `_conf_weight(infer_confidence)` |
| sound (M3) | 15% | 환경음 가중치 × 분류 신뢰도 |
| speech (M4) | 15% | 응급 키워드 히트 기반 |

**주요 보정 로직**:
- **복합 위험 배율**: 활성 도메인(≥0.5) 2개→×1.20, 3개→×1.35, 4개→×1.50
- **Vital Bypass**: HR/RR 극한값(`vital_component=1.0`) → score 최솟값 0.65 보장
- **Keyword+Fall 보너스**: 응급 키워드 ≥1 AND fall_score_raw ≥0.25 → +0.15

**생체신호 임계값 (화이트박스 경계)**:

| | 경고 (0.55) | 위기 (1.0) |
|---|---|---|
| HR (BPM) | ≤55 또는 ≥100 | ≤35 또는 ≥130 |
| RR (회/분) | ≤10 또는 ≥22 | ≤4 또는 ≥35 |

> Dead zone: HR 36~54, 101~129는 연산자 우선순위상 정상(0.0) 반환 — BVA 필수 영역

### `inference/qwen_05b.py` — Qwen-0.5B ONNX

```
입력: 컨텍스트 JSON (M1-M4 결과 + 시간별 이력)
출력: {"risk_level": "...", "reason": "...", "emergency_score": float}
```

- few-shot 프롬프트 7개로 출력 포맷 강제
- `vital_override`: vital_crisis 케이스에서 risk_level=normal 억제
- `hallucination_guard`: 알람/충격음 미탐지 시 '알람' 언급 금지
- `QWEN_MAX_NEW_TOKENS`: 40~80 (기본 56)

### `inference/utils.py` — ORT 프로바이더

```python
ORT_USE_GPU=1   # DirectML → CUDA → CPU 순 fallback
ORT_USE_GPU=0   # CPU 전용 (기본)
```

---

## 평가 하니스

### 골든셋 (`data/qwen_golden_set.jsonl`)

100-case, JSONL 포맷. 화이트박스 경계값 분석(BVA) 지향.

| 카테고리 | 건수 | 설명 |
|---|---|---|
| `normal` | 10 | 오탐 방지 — 정상 상황 |
| `fall_only` | 12 | 낙상 단독, score 경계 |
| `vital_crisis` | 20 | HR/RR 경계값 전수 커버 |
| `multi_domain` | 18 | 2/3/4 도메인 복합 |
| `no_signal` | 10 | `infer_confidence` 경계 (0.0/0.5/1.0) |
| `hallucination_guard` | 12 | env_label 다양화 |
| `boundary_bva` | 18 | 순수 BVA: dead zone, conf_weight, keyword 임계 |

각 케이스 필드:

```json
{
  "id": "vital-01",
  "category": "vital_crisis",
  "ground_truth_emergency": true,
  "input": { "fall": {...}, "vital": {...}, "env_sound": {...}, "speech_ko": {...} },
  "expected": {
    "score_min": 0.63, "score_max": 0.67,
    "m5_called": true,
    "vital_override": true,
    "hallucination_guard": false,
    "env_label": "silence",
    "numeric_hr": 35.0,
    "numeric_rr": null,
    "note": "..."
  }
}
```

### 실행

```bash
# CPU 모드 (mock — 모델 없이 score 검증만)
python eval/eval_qwen_reasoning.py --mock

# CPU 모드 (실제 모델)
python eval/eval_qwen_reasoning.py \
  --golden data/qwen_golden_set.jsonl \
  --report docs/

# GPU 모드 (DirectML / CUDA)
python eval/eval_qwen_reasoning.py \
  --golden data/qwen_golden_set.jsonl \
  --report docs/ \
  --gpu
```

### 메트릭

| 메트릭 | 설명 |
|---|---|
| accuracy | (TP+TN) / 전체 |
| FPR (오탐율) | FP / (FP+TN) — 정상을 응급으로 잘못 판단한 비율 |
| FNR (미탐율) | FN / (FN+TP) — 응급을 놓친 비율 |
| pass rate | 룰베이스 score 범위 통과율 |

리포트: `docs/qwen_eval_<timestamp>.html` + `reports/qwen_eval_<timestamp>.json`

---

## GPU 설정

Windows (RTX 5060 / DirectML):

```bash
pip install onnxruntime-directml
ORT_USE_GPU=1 python eval/eval_qwen_reasoning.py --golden data/qwen_golden_set.jsonl --gpu
```

Linux (CUDA):

```bash
pip install onnxruntime-gpu
ORT_USE_GPU=1 python eval/eval_qwen_reasoning.py --golden data/qwen_golden_set.jsonl --gpu
```

프로바이더 우선순위: `DmlExecutionProvider` → `CUDAExecutionProvider` → `CPUExecutionProvider`

---

## 모델 준비

```bash
# Qwen2-0.5B-Instruct → ONNX 변환
python scripts/export_m5_qwen_onnx.py --output volumes/models/qwen_05b/

# INT8 양자화 (Raspberry Pi 5 배포용)
python scripts/quantize_qwen_int8.py \
  --input  volumes/models/qwen_05b/model.onnx \
  --output volumes/models/qwen_05b/model_int8.onnx
```

모델 파일(`*.onnx`, `volumes/models/`)은 `.gitignore`에 포함 — LFS 또는 별도 관리.

---

## 의존성

```
onnxruntime>=1.18          # CPU
onnxruntime-directml       # Windows GPU (DirectML)
onnxruntime-gpu            # Linux GPU (CUDA)
tokenizers>=0.19
numpy>=1.24
fastapi
redis>=5.0
```

```bash
pip install -r requirements.txt
```

---

## Docker

```bash
# 전체 스택 (CPU)
docker compose up -d

# GPU 타겟
AI_DOCKER_TARGET=gpu-runtime docker compose up -d ai

# 로그
docker compose logs -f ai
```

---

## 관련 레포

- **[SafeWave-AI Ambient Monitoring](https://github.com/jinsan02/safewave-ai-ambient-monitoring.git)** — Raspberry Pi 5 메인 시스템 (sensing, api, db 서비스 포함)
