# 모델 카드 — M5 Qwen SLM (배포 표준)

SafeWave-AI 독거인 안전 모니터링의 M5(통합 위험도 판단) 배포 모델 명세.

## 배포본
| 항목 | 값 |
|---|---|
| 모델 | Qwen2.5-1.5B-Instruct |
| 백엔드(배포 표준) | **GGUF Q5_K_M** (llama.cpp) |
| 파일 | `volumes/models/qwen_15b_gguf_q5/qwen2.5-1.5b-instruct-q5_k_m.gguf` |
| 크기 | 1,285 MB (~1.20 GB) · 추정 RSS ~1.44 GB |
| 토크나이저 | `volumes/models/qwen_15b` (chat_template 재사용) |
| 컨텍스트 | n_ctx 2048 |
| 디코딩 | greedy(temperature 0, top_k 1), max_new_tokens 64, JSON prefix forcing |

## 입력 계약
M1~M4 전문가 출력 JSON(`fall`/`vital`/`env_sound`/`speech_ko`) + 선택 `time_series`(분당 vital ≤60행).
응급지수 룰 게이트(`compute_emergency_score`, 임계 0.6) 통과 시에만 호출.

## 성능 (2026-06-29, 골든셋 1000 시계열셋)
| 지표 | 값 |
|---|---|
| Track A 정확도 / FPR / FNR | 1.000 / 0.000 / 0.000 (게이트-오라클 정렬 white-box) |
| Track B raw 통과율 (Q5, 시계열셋) | **0.985** (grounded 정합화) / 0.966 (strict) |
| Track B raw 통과율 (Q5, 스냅샷 1000) | 0.988 |
| 추론 latency p50 / p95 | ~1.8s / ~2.3s (개발 PC, 시계열 프롬프트 기준) |
| 프롬프트 토큰 (p50) | ~772 (system 슬림 + 시계열 추세요약) |

> 프롬프트 최적화: 시계열 끝값 스냅샷 앵커 + 위기 vital salience·severity 정렬로 0.909→0.966(strict).
> numeric_match 정합화(grounded+에스컬레이션 인정)로 0.985. 잔여는 모델이 정상 vital을 이상으로
> 오인용하거나 사라진 케이스(환각/누락)로, 채점이 정직하게 실패 처리.

## 백엔드 비교 (배포 선택 근거)
| 백엔드 | 크기 | Track B(스냅샷 1000) | 경계 서맥(HR=36/40) |
|---|---|---|---|
| base 1.5B fp32 | 7.1 GB | 1.000 | 정상 분류 |
| **Q5_K_M (배포)** | 1.29 GB | 0.988 | **정상 분류(회복)** |
| Q4_K_M | 1.06 GB | 0.976 | 미탐(normal) |

→ RPi5 8GB·5초 게이트에서 메모리(+170MB)·속도(-18%) 차이는 무의미. **경계 서맥 안전 마진**으로 Q5 채택.

## 버전 식별
`GET /health` → `model_version`(=SLM_MODEL), `model_sha`(파일 name+size+mtime 12자 핑거프린트).
모델 스왑·드리프트 감지에 사용. 롤백 절차는 `docs/ops_runbook.md`.

## 알려진 한계
- 시계열 지속경고·점진악화 borderline 케이스에서 1.5B raw 추론이 가끔 ambivalent(Track B 0.915의 잔여 실패).
  최종 판정은 결정적 룰 게이트(Track A 1.000) + 가드레일(vital_override)이 보강.
- Track A 1.000은 curated white-box(held-out 아님) — 게이트와 임상 오라클 정렬의 의미.
