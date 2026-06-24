# /test-emergency-score

emergency_score 경계값 테스트를 실행한다.

## 실행

```bash
python tests/test_emergency_score.py -v
```

## 테스트 커버리지

| 섹션 | 케이스 수 | 내용 |
|---|---:|---|
| 기본 케이스 | 2 | 모든 입력 0, 낙상 단독 max |
| M1 낙상 경계 | 4 | fall_score 0 / 0.59 / 0.60 / 1.0 |
| M2 심박(HR) 경계 | 8 | 0 / crit_lo / warn_lo±ε / 정상 / warn_hi±ε / crit_hi |
| M2 호흡수(RR) 경계 | 7 | 0 / crit_lo / warn_lo±ε / 정상 / warn_hi±ε / crit_hi |
| M3 환경음 경계 | 10 | 모든 7 라벨 × conf=1.0, conf 스케일링, 미등록 라벨 |
| M4 음성/키워드 경계 | 4 | speech_detected only, 키워드 1/2개, stt_conf=0 보정 |
| 복합 위험 보정 | 3 | ×1.2 적용 여부, 복합>단독 검증 |
| 점수 클램프 | 2 | 모든 max → ≤1.0 |

## 경계값 핵심 규칙

- `_vital_component`: `val <= warn_lo` 또는 `val >= warn_hi` 이면 **경고(0.55)** 반환
  - HR=55 정확히 → warning (< 55 아님, **≤ 55** 포함)
  - RR=10 정확히 → warning (< 10 아님, **≤ 10** 포함)
- 복합 보정: 2개 이상 도메인 ≥ 0.5 → score × 1.2 (상한 1.0)
- STT 신뢰도 0 → 최소 0.3으로 보정

## 수정 필요 시

새 임계값 추가 시:
1. `inference/emergency_score.py` 상수 수정
2. `tests/test_emergency_score.py` 해당 섹션 경계값 업데이트
3. `/test-emergency-score` 실행으로 검증
