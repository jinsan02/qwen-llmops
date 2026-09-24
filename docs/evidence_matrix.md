# 근거 매트릭스 (Evidence Matrix)

이 저장소의 각 주장을 **네 가지 근거 축**과 **미검증 갭**으로 분리한다. 목적: "구현했다"와 "검증했다",
"합성 평가"와 "실기기 실측", "기능 존재"와 "실운영 실적"을 섞지 않는 것.

> **전제**: 이 프로젝트는 **시스템/LLMOps 엔지니어링 연구**다. **의료기기·임상 검증이 아니다.**
> 어떤 항목도 임상적 유효성·환자 안전을 주장하지 않는다.

## 등급 정의
| 등급 | 의미 |
|---|---|
| **구현** | 코드 경로·설정이 존재한다. 실제 실행 성공을 뜻하지 않는다 |
| **자동검증** | 단위테스트·CI/CD 실행 기록으로 해당 경로를 검증했다 |
| **합성평가** | **합성(생성) 데이터**로 정량 측정됐다 — Track A/B 골든셋은 in-distribution, Track R은 튜닝에 안 쓴 seed(held-out) |
| **실기기평가** | **RPi5(arm64) 실기**에서 측정된 로그가 있다 |
| **미검증 갭** | 위 축으로 입증되지 않은 주장. 아래 별도 목록에 둔다 |

범례: ✅ 있음 · ⚠️ 부분/제약 · ❌ 없음 · — 해당없음

## 매트릭스
| 항목 | 구현 | 자동검증 | 합성평가 | 실기기 | 근거 파일 / 주석 |
|---|:--:|:--:|:--:|:--:|---|
| 룰 게이트 `compute_emergency_score` | ✅ | ✅ | ✅ | ❌ | `inference/emergency_score.py` · `tests/test_emergency_score.py`(73 **경계·로직** 단위테스트, 임상 아님) |
| Track A 게이트 정확도 1.000/0/0 | ✅ | ✅ | ⚠️ | — | **white-box 정책 일관성 점검** — 게이트와 오라클이 같은 정책을 공유. held-out·독립라벨 아님 |
| 게이트·판정표 rp5 동기화 (낙상 확정·긴급 음성 우회, `rubric_level`) | ✅ | ✅ | — | ❌ | rp5와 무작위 입력 20,000건 비교 — 점수·플래그·판정표 불일치 0(`time_series=None`만). 테스트 [12]·[13] 13건 |
| Track R 판정표 준수율 (Q5, 모델만) | ✅ | ❌ | ⚠️ | ❌ | `eval/eval_track_r.py` · seed 4047 **77/98**, seed 5051 M2 꺼짐 **70/102**. 정답이 게이트에서 출발 → 준수율이지 정확도 아님. 튜닝에 안 쓴 seed지만 합성 |
| 시계열 에스컬레이션(지속경고/악화) | ✅ | ✅ | ✅ | ❌ | `emergency_score.py::_temporal_escalation` · test 섹션 [11] · 오라클 `ground_truth_temporal` |
| M5 Qwen 추론 (GGUF Q5_K_M) | ✅ | ⚠️ | ✅ | ❌ | `inference/qwen_gguf.py` · 파싱만 단위검증. Track B grounded **323/328**, strict 재채점 **317/328** |
| Track B 채점 (strict / grounded 분리) | ⚠️ | ❌ | ✅ | — | 코드는 grounded만 저장. strict는 raw reason 사후 재채점이며 별도 필드·자동 회귀테스트 없음 |
| 프롬프트 압축 (system 슬림 + 시계열 요약) | ✅ | — | ⚠️ | — | **규칙기반**(LLM/LLMLingua 미사용). `_SYSTEM` 200→183토큰(Qwen 토크나이저 실측, 94a64f7 수작업 2줄→1줄 병합), 시계열요약 `_series_prompt` ~41토큰 vs raw 60행≈736. 슬림 전후 Track B **0.915→0.909(300→298/328, −2건)** — 로컬 로그 `volumes/models/eval_q5_ts_1000b.log`·`eval_q5_slim_1000.log`(gitignore, 비공개). **무손실 아님**; 두 런 사이 변경이 슬림뿐인지는 git으로 확정 불가 |
| 골든셋 1000 시계열 | ✅ | — | ✅ | — | **전량 합성** `scripts/gen_golden_set_v3.py`(`random.gauss/randint`). 실환자·실측 아님 |
| GGUF 크기·경계 서맥(HR=36/40) | ✅ | ❌ | ✅ | ❌ | `docs/model_card.md` 「백엔드 비교」 — Q5 0.988(회복) / Q4 0.976(미탐) / fp32 1.000 |
| M5 추론 latency | ✅ | ❌ | — | ❌ | 개발 PC raw dump: 06-29 p50/p95 **1431.78/1757.67ms**, 06-30 **2535.10/3214.57ms**. 조건 메타데이터 부족, RPi5 아님 |
| 룰 게이트 latency ~2.4ms(warm) | ✅ | ❌ | — | ❌ | `docs/bench_result_20260624.json`(개발 PC) · `scripts/bench_latency.py` |
| RPi5 latency/메모리 투영 | ✅ | ❌ | — | ❌ | `scripts/bench_latency.py --project` **근사치만**. 실기 로그로 대체 필요 |
| FastAPI 서빙 `/evaluate·/health·/metrics·/feedback` | ✅ | ⚠️ | — | ❌ | `service/api.py` — CI 통합테스트 없음. 2026-09-25 도커 수동 기동: `/health` ok, 스모크 `/evaluate` 60·`/feedback` 6 오류 0. 이때 **gguf 이미지 M5 로드 실패(onnxruntime 최상단 import) 결함 발견·수정** |
| CI (경계73 + Track A mock 게이트) | ✅ | ✅ | — | — | `.github/workflows/ci.yml` · [CI #10 성공, 13초](https://github.com/jinsan02/qwen-llmops/actions/runs/29954017086) |
| CD (GHCR arm64 이미지 빌드·푸시) | ✅ | ✅ | — | ❌ | `.github/workflows/cd.yml` · [CD #1 성공, 3m39s](https://github.com/jinsan02/qwen-llmops/actions/runs/29953228399). **Delivery+이미지 스모크까지만**; 실기 pull·기동 로그 없음. ⚠️ CD #1 이미지는 M5 로드 실패 결함 포함(당시 스모크가 `emergency_score`만 import) — 스모크 강화 후 재빌드 필요 |
| 모니터링 (Prometheus + Grafana) | ✅ | ⚠️ | — | ❌ | 2026-09-25 개발 PC 도커 **수동 기동 1회**: 타깃 up, 코드 프로비저닝, 합성 스모크 60건으로 8패널 값 표시 — [스크린샷](img/grafana_m5_llmops_20260925.png). CI 자동 검증 아님, RPi5·운영 데이터 아님 |
| 보호자 피드백 루프 | ✅ | ❌ | — | ❌ | `service/api.py::/feedback` — 기능 구현. **실사용 피드백 데이터 없음** |
| 드리프트 점검 `check_drift.py` | ✅ | ❌ | ⚠️ | ❌ | baseline은 합성 골든셋 분포. **실운영 `ai:emergency` 스트림 실적 없음** |
| 거버넌스(model_card·runbook·model_sha 롤백) | ✅ | ❌ | — | ❌ | `docs/model_card.md` · `docs/ops_runbook.md` · `/health` 핑거프린트 |

## 미검증 갭 (요약)
정직하게 남은 것 — 포트폴리오에서 "했다"로 말하면 안 되는 항목:
1. **RPi5 실기 실측** — latency·메모리·스로틀링 로그 전무. 배포는 "목표로 구성" 단계(이미지 빌드까지 자동).
2. **held-out 일반화** — Track A/B는 튜닝에 쓴 평가셋. Track R이 튜닝에 안 쓴 seed로 판정표 준수율을 재지만
   합성 데이터이고 정답이 게이트에서 출발한다. 실데이터에서의 판정 정확도는 미측정.
3. **실운영 데이터** — 피드백·드리프트가 돌 실제 `ai:emergency` 스트림 실적 없음.
4. **서빙/모니터링 자동 검증** — 2026-09-25 수동 기동·스크린샷은 있으나 CI에서 도는 API 통합테스트는 없음.
5. **공개 raw evidence** — `reports/`가 Git 제외 대상이라 공개 저장소에는 응답 원본이 없다. 집계·SHA-256만 공개된다.
6. **임상 유효성** — 범위 밖(비목표). 어떤 수치도 의학적 판단 근거로 쓰일 수 없다.

## 수치 원장

Track B 건수·latency·원본 SHA-256은 [`metrics_summary.json`](metrics_summary.json)에 기록한다.
strict 값은 기존 dump에 별도 필드로 저장되지 않았으므로 raw reason과 골든셋 기대값을 사후 재채점한 값이다.
원본 dump는 로컬 `reports/`에 있고 Git에는 포함되지 않으므로, 공개 검증은 이 집계와 해시까지로 제한된다.
