"""
RPi5 실측 레이턴시 벤치마크 (Task B)

골든셋에서 케이스를 선발해 cold/warm latency, cpu_pct, rss_mb를 측정한다.
RPi5에서 직접 실행하는 것이 전제. Windows에서는 --mock으로 구조 확인만 가능.

실행:
  python scripts/benchmark_rpi5_qwen.py --model volumes/models/qwen_05b --output bench_result.json
  python scripts/benchmark_rpi5_qwen.py --mock
"""

import argparse
import json
import os
import sys
import time
import datetime
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


_BENCH_CASES = [
    # 카테고리당 1~2케이스 선발 (프롬프트 길이 분포 반영)
    {"id": "normal-03",   "category": "normal",               "input": {"fall": {"fall_score": 0.59}, "vital": {"heart_rate": 72.0, "breathing_rate": 16.0}, "env_sound": {"env_sound_label": "silence", "env_sound_confidence": 0.0}, "speech_ko": {"keywords": [], "stt_confidence": 0.0, "speech_detected": False}}},
    {"id": "fall-03",     "category": "fall_only",            "input": {"fall": {"fall_score": 1.0},  "vital": {"heart_rate": 72.0, "breathing_rate": 16.0}, "env_sound": {"env_sound_label": "alarm",   "env_sound_confidence": 1.0}, "speech_ko": {"keywords": [], "stt_confidence": 0.0, "speech_detected": False}}},
    {"id": "fall-06",     "category": "fall_only",            "input": {"fall": {"fall_score": 1.0},  "vital": {"heart_rate": 72.0, "breathing_rate": 16.0}, "env_sound": {"env_sound_label": "silence",  "env_sound_confidence": 0.0}, "speech_ko": {"keywords": ["살려"], "stt_confidence": 1.0, "speech_detected": True}}},
    {"id": "vital-01",    "category": "vital_crisis",         "input": {"fall": {"fall_score": 0.0},  "vital": {"heart_rate": 35.0, "breathing_rate": 16.0}, "env_sound": {"env_sound_label": "silence",  "env_sound_confidence": 0.0}, "speech_ko": {"keywords": [], "stt_confidence": 0.0, "speech_detected": False}}},
    {"id": "vital-06",    "category": "vital_crisis",         "input": {"fall": {"fall_score": 0.0},  "vital": {"heart_rate": 160.0, "breathing_rate": 16.0}, "env_sound": {"env_sound_label": "silence", "env_sound_confidence": 0.0}, "speech_ko": {"keywords": [], "stt_confidence": 0.0, "speech_detected": False}}},
    {"id": "multi-04",    "category": "multi_domain",         "input": {"fall": {"fall_score": 1.0},  "vital": {"heart_rate": 50.0, "breathing_rate": 16.0}, "env_sound": {"env_sound_label": "alarm",   "env_sound_confidence": 1.0}, "speech_ko": {"keywords": [], "stt_confidence": 0.0, "speech_detected": False}}},
    {"id": "multi-07",    "category": "multi_domain",         "input": {"fall": {"fall_score": 1.0},  "vital": {"heart_rate": 50.0, "breathing_rate": 16.0}, "env_sound": {"env_sound_label": "alarm",   "env_sound_confidence": 1.0}, "speech_ko": {"keywords": ["살려"], "stt_confidence": 0.8, "speech_detected": True}}},
    {"id": "no-signal-02","category": "no_signal",            "input": {"fall": {"fall_score": 0.0},  "vital": {"heart_rate": 130.0, "breathing_rate": 16.0, "infer_confidence": 0.0}, "env_sound": {"env_sound_label": "silence", "env_sound_confidence": 0.0}, "speech_ko": {"keywords": [], "stt_confidence": 0.0, "speech_detected": False}}},
    {"id": "hg-03",       "category": "hallucination_guard",  "input": {"fall": {"fall_score": 0.0},  "vital": {"heart_rate": 35.0, "breathing_rate": 16.0}, "env_sound": {"env_sound_label": "speech",  "env_sound_confidence": 1.0}, "speech_ko": {"keywords": [], "stt_confidence": 0.0, "speech_detected": False}}},
    {"id": "hg-07",       "category": "hallucination_guard",  "input": {"fall": {"fall_score": 1.0},  "vital": {"heart_rate": 50.0, "breathing_rate": 16.0}, "env_sound": {"env_sound_label": "music",   "env_sound_confidence": 0.5}, "speech_ko": {"keywords": ["살려"], "stt_confidence": 1.0, "speech_detected": True}}},
]


def _rss_mb() -> float:
    if not _PSUTIL:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _cpu_pct(interval: float = 0.1) -> float:
    if not _PSUTIL:
        return 0.0
    return psutil.cpu_percent(interval=interval)


def _mock_evaluate(inp: dict) -> tuple[float, str]:
    from inference.emergency_score import compute_emergency_score
    score, _ = compute_emergency_score(inp)
    time.sleep(0.002)  # mock 지연
    return score, "mock: no actual inference"


def _run_case(case: dict, qwen_logic, n_warm: int) -> dict:
    inp  = case["input"]
    cid  = case["id"]
    cat  = case["category"]

    times_ms = []
    for i in range(n_warm + 1):
        rss_before = _rss_mb()
        t0 = time.perf_counter()

        if qwen_logic is not None:
            result = qwen_logic.evaluate(inp)
            elapsed = (time.perf_counter() - t0) * 1000.0
            qwen_ms = result.get("qwen_infer_ms") or elapsed
        else:
            _, _ = _mock_evaluate(inp)
            elapsed = (time.perf_counter() - t0) * 1000.0
            qwen_ms = elapsed

        rss_after = _rss_mb()

        if i == 0:
            cold_ms    = elapsed
            cold_rss   = rss_after
        else:
            times_ms.append(elapsed)

    warm_samples = times_ms if times_ms else [cold_ms]
    return {
        "id":          cid,
        "category":    cat,
        "cold_ms":     round(cold_ms, 2),
        "warm_avg_ms": round(statistics.mean(warm_samples), 2),
        "warm_p50_ms": round(statistics.median(warm_samples), 2),
        "warm_p95_ms": round(sorted(warm_samples)[int(len(warm_samples) * 0.95)], 2),
        "warm_max_ms": round(max(warm_samples), 2),
        "rss_mb":      round(cold_rss, 1),
    }


def _print_bench(results: list[dict]) -> None:
    print()
    print(f"{'ID':<16} {'cold_ms':>9} {'warm_avg':>9} {'warm_p95':>9} {'max_ms':>9} {'rss_mb':>8}")
    print("-" * 68)
    for r in results:
        print(f"  {r['id']:<14} {r['cold_ms']:>9.1f} {r['warm_avg_ms']:>9.1f} "
              f"{r['warm_p95_ms']:>9.1f} {r['warm_max_ms']:>9.1f} {r['rss_mb']:>8.1f}")
    cold_vals = [r["cold_ms"] for r in results]
    warm_vals = [r["warm_avg_ms"] for r in results]
    print("-" * 68)
    print(f"  {'전체 평균':<14} {statistics.mean(cold_vals):>9.1f} {statistics.mean(warm_vals):>9.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="RPi5 Qwen 레이턴시 벤치마크")
    parser.add_argument("--model",  default=None)
    parser.add_argument("--output", default=None, help="결과 JSON 경로 (기본: docs/bench_result_YYYYMMDD.json)")
    parser.add_argument("--n-warm", type=int, default=20, help="워밍업 후 측정 횟수 (기본: 20)")
    parser.add_argument("--mock",   action="store_true", help="모델 없이 구조 검증만")
    args = parser.parse_args()

    qwen_logic = None
    if not args.mock:
        model_path = args.model or os.getenv("SLM_MODEL", "volumes/models/qwen_05b")
        if os.path.exists(model_path):
            from inference.qwen_05b import QwenLogic
            qwen_logic = QwenLogic(model_path)
            print(f"[INFO] 모델 로드: {model_path}")
        else:
            print(f"[WARN] 모델 경로 없음({model_path}), mock 모드로 전환")

    print(f"[INFO] {len(_BENCH_CASES)}케이스 × (1 cold + {args.n_warm} warm) 측정 시작")
    results = [_run_case(c, qwen_logic, args.n_warm) for c in _BENCH_CASES]

    _print_bench(results)

    date_str = datetime.date.today().strftime("%Y%m%d")
    output_path = args.output or os.path.join("docs", f"bench_result_{date_str}.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"date": date_str, "n_warm": args.n_warm, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
