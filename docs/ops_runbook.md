# Ops 런북 — M5 서빙/롤백/드리프트

## 백엔드 전환 (롤백 포함)
배포 표준은 **Q5_K_M**. 백엔드는 `SLM_MODEL` env 한 줄로 즉시 전환된다(코드 변경 없음).

| 대상 | SLM_MODEL | SLM_BACKEND | 용도 |
|---|---|---|---|
| Q5_K_M (표준) | `qwen_15b_gguf_q5` | `gguf` | 배포 기본 |
| Q4_K_M (최경량 롤백) | `qwen_15b_gguf` | `gguf` | 메모리/속도 우선 |
| 1.5B fp32 (기준) | `qwen_15b` | `15b` | 품질 기준·디버그 |

### 절차
```bash
# 1) compose 환경변수 교체 후 재기동
SLM_MODEL=qwen_15b_gguf docker compose --profile api up -d api   # 예: Q4로 롤백
# 2) 헬스체크 — model_loaded=true, model/version 확인
curl -s http://localhost:8000/health
# 3) 위기 케이스 스모크 (HR=36 → risk_level ≠ normal 기대)
curl -s -X POST http://localhost:8000/evaluate -H 'Content-Type: application/json' \
  -d '{"vital":{"heart_rate":36,"breathing_rate":16}}'
# 4) 회귀 게이트 (모델 무관, Track A)
python eval/eval_qwen_reasoning.py --mock --golden data/qwen_golden_set.jsonl --report docs/
```
`/health`의 `model_sha`(핑거프린트)가 의도한 모델과 일치하는지 확인.

## 모니터링
- `GET /metrics` — Prometheus 텍스트(외부 Prometheus가 스크랩). `GET /metrics.json` — 사람이 읽는 JSON.
- 핵심 지표: `m5_requests_total`, `m5_called_total`, `m5_errors_total`, `m5_level_total{level}`, `m5_call_rate`, `m5_avg_latency_ms`, `m5_model_loaded`.
- Redis(`REDIS_HOST` 설정 시) `m5:metrics:snapshot`에 1h 롤링 스냅샷 지속(TTL 3600) — 재기동 복원.

## 피드백 루프
- `POST /feedback {"feedback":"false_alarm|missed_alert|confirm"}` → Redis `mqtt:feedback:last`(TTL 3600).
- 다음 `evaluate`의 `_apply_feedback_adjustment`가 risk_score를 ±0.08 보정. `m5_feedback_total{type}`로 집계.

## 드리프트 점검
```bash
python scripts/check_drift.py            # 운영 ai:emergency 등급 분포 vs 골든셋 baseline
```
응급 비율·등급 분포가 baseline(응급 ~33%)에서 임계 이상 벗어나면 경고 출력. 운영 데이터는 스캔만(Redis, 저장 안 함).

## 비상 시
- 모델 로드 실패 → `/health status=degraded`, `/evaluate`는 룰 게이트만으로 normal 반환(M5 미호출). 즉시 fp32(`qwen_15b`)로 롤백 후 원인 조사.
- latency 급증 → `m5_avg_latency_ms` 확인, 시계열 프롬프트 토큰(↑) 또는 백엔드 확인. 최경량 Q4 롤백 고려.
