# CLAUDE.md — qwen_llmops

SafeWave-AI M5(Qwen-0.5B-Instruct) LLMOps 독립 레포.
rp5의 `ai-qwen` 컨테이너 코드를 분리해 capstone / 개인 연구용으로 관리.

---

## 디렉토리 구조

```
inference/      # M5 추론 코어 (rp5: ai/logic/ 에서 이식)
  utils.py      # ORT 프로바이더, Redis 헬퍼 (rp5: ai/utils/__init__.py)
  emergency_score.py  # M1-M4 응급지수 계산 (rp5: ai/logic/emergency_score.py)
  qwen_05b.py   # QwenLogic 클래스 (rp5: ai/logic/qwen_05b.py)
service/        # Redis 스트림 소비 루프 (rp5: ai/qwen_service.py)
data/           # 골든셋 (Task A)
eval/           # 평가 스크립트
scripts/        # 변환·벤치마크·양자화
tests/          # 경계값 테스트
docs/           # HTML 리포트 (Notion 첨부용)
volumes/models/ # ONNX 모델 파일 (git 제외)
```

## 핵심 제약

- **inference/ 코드는 rp5와 동기화 기준.** 독립 개선 시 rp5 반영 여부를 명시적으로 결정.
- **guard 로직 독립 유지:** `env_label` 후처리, `vital_override`는 qwen_05b.py 하단에 분리. 추론 백엔드 교체 시 함께 건드리지 않는다.
- **Redis 없이도 eval/test 독립 실행 가능.**
- **Task C 결정은 Task A(골든셋 채점) + Task B(RPi5 실측) 완료 후 확정.**

## 자주 쓰는 명령어

```bash
# 경계값 테스트 (Redis 불필요)
python tests/test_emergency_score.py -v

# Qwen 출력 평가 (Task A, 골든셋 채점)
python eval/eval_qwen_reasoning.py --golden data/qwen_golden_set.jsonl --report docs/

# 기존 100케이스 정확도 평가
python eval/eval_qwen_accuracy.py

# RPi5 레이턴시 측정 (Task B, RPi5에서 실행)
python scripts/benchmark_rpi5_qwen.py --model volumes/models/qwen_05b

# 서비스 실행 (Docker)
docker compose up -d
docker compose logs -f ai-qwen
```

## 슬래시커맨드

- `/test-emergency-score` — emergency_score 경계값 40케이스 실행
- `/test-qwen-reasoning` — Qwen 출력 골든셋 채점 + HTML 리포트 (Task D)

## rp5 동기화 메모

| 파일 | rp5 원본 |
|---|---|
| `inference/utils.py` | `ai/utils/__init__.py` |
| `inference/emergency_score.py` | `ai/logic/emergency_score.py` |
| `inference/qwen_05b.py` | `ai/logic/qwen_05b.py` |
| `service/qwen_service.py` | `ai/qwen_service.py` |
