"""
드리프트 점검 — 운영 ai:emergency 등급 분포 vs 골든셋 baseline (Ops ③ 거버넌스).

운영 스트림의 risk_level 분포·응급 비율이 평가셋 baseline에서 임계 이상 벗어나면 경고.
운영 데이터는 Redis 스캔만 한다(저장·복사 없음). 모델/ORT 불필요.

실행:
  REDIS_HOST=db REDIS_PORT=6379 python scripts/check_drift.py
  python scripts/check_drift.py --window-min 60 --threshold 0.15
종료코드: 드리프트 감지 시 1, 정상 0 (cron/수동 점검용).
"""

import argparse
import json
import os
import sys
import time

# 골든셋 baseline (data/qwen_golden_set.jsonl 1000 시계열셋 기준)
_BASELINE = {"emergency_ratio": 0.328, "critical": 0.0, "warning": 1.0, "normal": 0.0}
_EMERGENCY_STREAM = "ai:emergency"


def _ts_ms(msg_id):
    try:
        return int(msg_id.split("-")[0])
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-min", type=int, default=60, help="최근 N분 윈도우")
    ap.add_argument("--threshold", type=float, default=0.15, help="응급비율 허용 편차(절대)")
    ap.add_argument("--min-samples", type=int, default=20, help="이보다 적으면 판정 보류")
    args = ap.parse_args()

    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    try:
        import redis
        r = redis.Redis(host=host, port=port, decode_responses=True, socket_connect_timeout=3)
        r.ping()
    except Exception as e:
        print(f"[ERR] Redis 연결 실패({host}:{port}): {e}")
        sys.exit(2)

    since = int(time.time() * 1000) - args.window_min * 60 * 1000
    try:
        entries = r.xrevrange(_EMERGENCY_STREAM, count=5000)
    except Exception as e:
        print(f"[ERR] {_EMERGENCY_STREAM} 스캔 실패: {e}")
        sys.exit(2)

    levels = {"normal": 0, "warning": 0, "critical": 0}
    emergencies = total = 0
    for msg_id, fields in entries:
        if _ts_ms(msg_id) < since:
            break
        raw = fields.get("data", "")
        try:
            p = json.loads(raw) if raw else {}
        except Exception:
            continue
        lvl = str(p.get("risk_level", "normal"))
        if lvl in levels:
            levels[lvl] += 1
        if p.get("emergency") or lvl in ("warning", "critical"):
            emergencies += 1
        total += 1

    print(f"== 드리프트 점검 (최근 {args.window_min}분, {_EMERGENCY_STREAM}) ==")
    print(f"  표본 {total}건 · 등급 {levels}")
    if total < args.min_samples:
        print(f"  표본 부족(<{args.min_samples}) → 판정 보류")
        sys.exit(0)

    ratio = emergencies / total
    base = _BASELINE["emergency_ratio"]
    dev = abs(ratio - base)
    print(f"  운영 응급비율 {ratio:.3f} vs baseline {base:.3f} (편차 {dev:.3f}, 허용 {args.threshold})")

    drift = dev > args.threshold
    if drift:
        direction = "상승(오탐↑ 의심)" if ratio > base else "하강(미탐↑ 의심)"
        print(f"  ⚠️ 드리프트 감지 — 응급비율 {direction}. 모델/센서/임계 점검 필요.")
        sys.exit(1)
    print("  ✅ 정상 범위")
    sys.exit(0)


if __name__ == "__main__":
    main()
