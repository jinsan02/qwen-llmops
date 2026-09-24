"""
모니터링 스모크 — 서빙 API에 합성 요청을 보내 Prometheus/Grafana 패널에 값을 채운다.

대시보드 스크린샷 재현용이다. 요청은 Track R 생성기(eval/eval_track_r.py)와 같은 합성 케이스이며
운영 데이터가 아니다. 표준 라이브러리만 쓴다.

  python scripts/monitoring_smoke.py --url http://localhost:8000 --n 60 --feedback 6
"""

import argparse
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.eval_track_r import build_expert, random_case_defs


def _post(url: str, body: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description="모니터링 스모크 트래픽(합성)")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--feedback", type=int, default=6)
    args = ap.parse_args()

    called, errors, lat = 0, 0, []
    for d in random_case_defs(args.n, args.seed):
        try:
            out = _post(f"{args.url}/evaluate", build_expert(d))
            called += bool(out.get("m5_called"))
            lat.append(out.get("latency_ms", 0.0))
        except Exception as exc:
            errors += 1
            print(f"  evaluate 실패 {d['id']}: {exc}")

    rng = random.Random(args.seed)
    for _ in range(args.feedback):
        try:
            _post(f"{args.url}/feedback", {"feedback": rng.choice(["false_alarm", "missed_alert", "confirm"])})
        except Exception as exc:
            errors += 1
            print(f"  feedback 실패: {exc}")
        time.sleep(0.2)

    lat.sort()
    p50 = lat[len(lat) // 2] if lat else 0.0
    print(f"evaluate {args.n}건 (M5 호출 {called}) · feedback {args.feedback}건 · 오류 {errors} · "
          f"latency p50 {p50:.0f}ms")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
