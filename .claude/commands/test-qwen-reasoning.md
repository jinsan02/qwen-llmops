# /test-qwen-reasoning

Qwen 출력 평가 하니스를 실행하고 카테고리별 PASS/FAIL 리포트를 출력한다.
(Task A/D: eval_qwen_reasoning.py + 골든셋 채점)

## 실행

```bash
python eval/eval_qwen_reasoning.py --golden data/qwen_golden_set.jsonl --report docs/
```

## 채점 기준 (4가지 룰베이스)

| 기준 | 설명 |
|---|---|
| 수치 일치 | reason에 언급된 HR/RR 수치가 입력값과 일치하는지 (±5% 허용) |
| 라벨 정합성 | env_label ≠ alarm/impact 시 "알람" 금지어 없는지 |
| vital_override 발동 | HR/RR 극단값 케이스에서 reason에 "normal"/"정상" 미포함 |
| 포맷 완결 | max_new_tokens=56 안에서 문장이 완결되는지 |

## 골든셋 카테고리

| 카테고리 | 케이스 수 | 검증 포인트 |
|---|---:|---|
| normal | 5 | M5 호출 안 됨 검증 (score < 0.6) |
| fall_only | 6 | 낙상 단독 |
| vital_crisis | 8 | vital_override 발동 경계 케이스 |
| multi_domain | 9 | ×1.20/×1.35/×1.50 보정 |
| no_signal | 4 | conf_sound/speech ≈ 0 |
| hallucination_guard | 8 | env_label 후처리 회귀 테스트 |

## 출력

- 콘솔: 카테고리별 PASS/FAIL 표, 실패 케이스 입력/기대값/실제 출력
- 파일: `docs/qwen_eval_report_{YYYYMMDD}.html` (Notion 첨부용)
