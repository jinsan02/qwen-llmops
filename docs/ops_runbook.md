# Ops 런북 — M5 서빙/롤백/드리프트

## 백엔드 전환 (롤백 포함)
현재 배포 **후보**는 Q5_K_M이며 RPi5 실기 검증 전이다. 백엔드는 `SLM_MODEL` env 한 줄로 전환되도록 구현돼 있다(코드 변경 없음).

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

## 모니터링 (Prometheus + Grafana)
```bash
docker compose --profile monitoring up -d      # Prometheus :9090 · Grafana :3000 (admin/admin)
# API까지 같이 + 호스트 8000을 rp5 api가 쓰고 있으면 API_PORT로 호스트 포트만 바꾼다
API_PORT=18000 docker compose --profile api --profile monitoring up -d db api prometheus grafana
python scripts/monitoring_smoke.py --url http://localhost:18000 --n 60 --feedback 6   # 합성 트래픽으로 패널 채우기
```
- **2026-09-25 기동 확인**: 4컨테이너 기동, `/health` ok·model_loaded, Prometheus 타깃 up, 8패널 전부 값 표시
  ([스크린샷](img/grafana_m5_llmops_20260925.png)). 이때 발견·수정한 결함: gguf-runtime 이미지에 onnxruntime이 없는데
  `qwen_15b.py`가 최상단에서 import해 **M5 로드 실패 → `/health status=degraded`**(룰 게이트만 동작). import를
  ONNX 로드 함수 안으로 옮겼고, CI·CD 스모크에 `from inference.qwen_gguf import QwenLogic`를 추가했다.
  **그 전에 CD가 GHCR에 올린 이미지는 이 결함을 가진다 — 다음 태그 빌드로 교체 필요.**
- Grafana 대시보드 **LLMOps / M5 (Qwen SLM)** 자동 프로비저닝(코드 원본: `monitoring/grafana/dashboards/m5_llmops.json`, UI 수정 불가).
- 패널: 모델 로드·M5 호출률·latency p50/p95·오류·등급분포(드리프트)·토큰 사용량·보호자 피드백.
- `GET /metrics` Prometheus 텍스트 / `GET /metrics.json` 사람이 읽는 JSON.
- 핵심 지표: `m5_requests_total`·`m5_called_total`·`m5_errors_total`·`m5_level_total{level}`·`m5_feedback_total{type}`·
  **`m5_latency_ms_bucket`(히스토그램 → p95)**·`m5_prompt_tokens_total`·`m5_output_tokens_total`·`m5_model_loaded`.
- Redis(`REDIS_HOST` 설정 시) `m5:metrics:snapshot`에 1h 롤링 스냅샷 지속(TTL 3600) — 재기동 복원.

**RPi5 운영 주의**: 모니터링은 `--profile monitoring` opt-in이라 기본 배포엔 안 뜬다.
Prometheus 보존은 **7d / 512MB 상한**으로 SD 마모를 제한. 부하·마모를 아예 없애려면
**모니터링 스택을 오프디바이스에서 띄우고** Pi의 `:8000/metrics`를 원격 스크래핑
(`monitoring/prometheus.yml`의 `m5-api-remote` 주석 참고).

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
