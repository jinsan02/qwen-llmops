# 모델 카드 — M5 Qwen SLM (배포 목표본)

SafeWave-AI 독거인 안전 모니터링의 M5(통합 위험도 판단) 모델 명세.

> **고지**: 이 카드는 **시스템 연구 산출물**이며 의료기기·임상 문서가 아니다. 아래 성능 수치는 전부
> **합성 골든셋(in-distribution)** 기반이고, latency는 **개발 PC(x86)** 측정이다. **RPi5(arm64) 실기 실측 로그는 없다.**
> 근거 등급 분리는 [`evidence_matrix.md`](evidence_matrix.md).

## 배포 목표본
| 항목 | 값 |
|---|---|
| 모델 | Qwen2.5-1.5B-Instruct |
| 백엔드(배포 후보) | **GGUF Q5_K_M** (llama.cpp) — RPi5 실기 검증 전 |
| 파일 | `volumes/models/qwen_15b_gguf_q5/qwen2.5-1.5b-instruct-q5_k_m.gguf` |
| 크기 | 1,285 MB (~1.20 GB) · 추정 RSS ~1.44 GB |
| 토크나이저 | `volumes/models/qwen_15b` (chat_template 재사용) |
| 컨텍스트 | n_ctx 2048 |
| 디코딩 | greedy(temperature 0, top_k 1), max_new_tokens 64, JSON prefix forcing |

## 입력 계약
M1~M4 전문가 출력 JSON(`fall`/`vital`/`env_sound`/`speech_ko`) + 선택 `time_series`(분당 vital ≤60행).
응급지수 룰 게이트(`compute_emergency_score`, 임계 0.6) 통과 시에만 호출.

## 성능 (2026-06-29, **합성** 골든셋 1000 시계열셋)
데이터 전량 합성(`scripts/gen_golden_set_v3.py`). 프롬프트를 이 셋으로 튜닝 → **held-out 아님(in-distribution)**.

| 지표 | 값 | 성격 |
|---|---|---|
| Track A 정확도 / FPR / FNR | 1.000 / 0.000 / 0.000 | **white-box 정책 일관성 점검** — 게이트·오라클이 같은 정책을 공유. held-out·독립라벨 아님 |
| Track B raw (Q5, 시계열셋) — grounded | 323/328 = **0.985** | `20260629` raw dump의 `m5_pass` |
| Track B raw (Q5, 시계열셋) — strict | 317/328 = **0.966** | 저장 raw reason을 기대 위기값 ±5% 기준으로 재채점 |
| Track B raw (Q5, 스냅샷 1000) | 162/164 = **0.988** | 1000개 중 M5 호출 대상 164건 |
| **M5 추론** latency p50 / p95 (2026-06-29) | 1431.78 / 1757.67 ms | 개발 PC(x86), N=328. **RPi5 실측 아님** |
| **M5 추론** latency p50 / p95 (2026-06-30 재실행) | 2535.10 / 3214.57 ms | 개발 PC(x86), N=328. 실행 조건 메타데이터가 충분하지 않아 날짜 간 비교 금지 |
| 룰 게이트 latency (warm p50) | ~2.4ms | `docs/bench_result_20260624.json`(개발 PC) |
| 프롬프트 토큰 (p50) | ~772 | system 슬림 + 시계열 추세요약 |

> **strict / grounded 정의**: 현재 `_score_numeric_match`는 grounded 기준을 구현한다. strict는 저장 raw reason에서
> 기대 위기값 ±5% 일치를 사후 재채점한 값이며 원본 dump에 별도 필드가 없다. **RPi5 latency는 미측정** —
> 투영치는 `scripts/bench_latency.py`의 `--project` 근사이며 실기 로그로 대체돼야 한다.

> 집계·원본 파일 해시는 [`metrics_summary.json`](metrics_summary.json)에 기록했다.
> 중간 프롬프트 실험의 0.909는 당시 HTML 보고서에는 있으나 현재 raw dump만으로 독립 재산출되지 않아 대표 수치에서 제외한다.

## Track R — 판정표 준수율 (2026-09-25, rp5 동기화 후)
정답 = `inference/risk_policy.rubric_level`(게이트 점수에서 출발) → **판정 정확도가 아니라 판정표 준수율**.
운영 구간(게이트≥0.6)만 채점. 케이스는 seed 기반 합성(rp5 생성기와 동일)이며 이 프롬프트 튜닝에 쓰지 않은 seed다.
qwen-llmops에는 판정표 하한 후처리가 없다(rp5 노트북 프로필 전용).

| 조건 (Q5, 개발 PC CPU, 기본 64토큰) | 운영 구간 | 모델만 exact | 과대 / 과소 |
|---|---|---|---|
| seed 4047, M2 켬 | 98 | 77 (0.786) | 21 / 0 — rp5 `rpi5` 프로필 모델만 77/98과 동일 |
| seed 5051, M2 꺼짐(심박·호흡 미측정) | 102 | 70 (0.686) | 32 / 0 — 16건이 낙상 확정 단독 |

동기화 후 Track B 재측정: strict 317/328, grounded 323/328 — 이전과 동일.
상세·판정표 의견·LoRA 판단은 [`rp5_sync_20260925.md`](rp5_sync_20260925.md).

## 백엔드 비교 (배포 선택 근거)
| 백엔드 | 크기 | Track B(스냅샷 1000) | 경계 서맥(HR=36/40) |
|---|---|---|---|
| base 1.5B fp32 | 7.1 GB | 164/164 = 1.000 | 합성 스냅샷 참조 |
| **Q5_K_M (배포 후보)** | 1.29 GB | 162/164 = 0.988 | 합성 경계 서맥 케이스 통과 |
| Q4_K_M | 1.06 GB | 160/164 = 0.976 | 합성 경계 서맥 케이스 미탐 |

→ 합성 스냅샷셋의 통과율과 경계 서맥 케이스를 근거로 Q5를 **배포 후보**로 선택했다. RPi5에서의 메모리·지연 trade-off는 미측정이므로 선택은 실기 검증 전 잠정적이다.

## 버전 식별
`GET /health` → `model_version`(=SLM_MODEL), `model_sha`(파일 name+size+mtime 12자 핑거프린트).
모델 스왑·드리프트 감지에 사용. 롤백 절차는 `docs/ops_runbook.md`.

## 알려진 한계
- **임상 아님**: 시스템/LLMOps 연구 산출물. 의료기기 인증·임상 유효성·환자 안전을 주장하지 않는다.
- **합성 평가**: 성능 전량 in-distribution 합성 골든셋 기준. held-out·실환자 데이터 없음 → 일반화 미검증.
- **white-box 점검**: Track A 1.000은 게이트·오라클이 같은 정책을 공유하는 일관성 점검. 독립 라벨 성능이 아니다.
- **과대 판정 경향**: Track R 오답은 전부 warning→critical 과대. M2 꺼짐에서 낙상 확정 단독을 critical로 올리는 경우가 많다.
  판정표 기준의 오답일 뿐 실제로 무엇이 맞는지는 시나리오 시험으로만 가릴 수 있다.
- **잔여 실패**: 저장 raw 기준 strict 실패 11건, grounded 실패 5건. 실패 원인은 raw 응답과 채점 규칙 단위로 확인해야 하며 임상적 의미를 부여할 수 없다.
- **공개 원본 제한**: raw response dump는 로컬 `reports/`에만 있고 Git에는 포함되지 않는다. 공개 저장소에서는 `metrics_summary.json`의 집계·해시까지만 확인할 수 있다.
- **실기기 미검증**: latency·메모리는 개발 PC(x86). RPi5(arm64) 실측 로그 없음.
- **운영 데이터 부재**: 피드백·드리프트는 기능 구현 완료이나 실운영 스트림 실적 없음.
