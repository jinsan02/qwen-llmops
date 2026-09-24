"""
Track R — 판정표(rubric v2) 준수율 평가 (rp5 scripts/eval_qwen_accuracy.py 방식 이식).

Track A(독립 오라클)·Track B(raw 추론 4기준)와 별개 트랙이다. 정답이 게이트 점수에서
출발하는 판정표(inference/risk_policy.rubric_level)이므로 '판정 정확도'가 아니라
'판정표 준수율'을 잰다. 세 트랙 수치를 섞어 해석하지 않는다.

- 케이스: seed 기반 무작위 생성(rp5 _random_case_defs 그대로) — 같은 seed면 rp5와 같은 케이스.
  프롬프트 조정에 쓰지 않은 seed로 재면 held-out이다.
- 채점 구간: 운영 구간(게이트 >= 0.6, 실제로 M5가 호출되는 케이스)만.
- 보고: 모델만(raw JSON 등급) / 최종(evaluate() 출력 등급, qwen-llmops 가드레일 포함·판정표 하한 없음).

실행:
  python eval/eval_track_r.py --random 150 --seed 4047 --mock          # 게이트만(케이스 분포 확인)
  python eval/eval_track_r.py --random 150 --seed 4047 --impl gguf \\
      --model volumes/models/qwen_15b_gguf_q5 --tokenizer volumes/models/qwen_15b
  python eval/eval_track_r.py --random 300 --seed 5051 --m2-off ...     # M2 꺼짐(심박·호흡 미측정)
"""

import argparse
import datetime
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.emergency_score import compute_emergency_score
from inference.risk_policy import WARNING_THRESHOLD, rubric_level
from eval.eval_qwen_reasoning import _parse_raw_response

_L = {"normal": 0, "warning": 1, "critical": 2}


# ── 케이스 생성 (rp5 scripts/eval_qwen_accuracy.py 그대로 — seed 동일성 유지) ──

def random_case_defs(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    hr_odd = [25, 31, 38, 44, 52, 57, 103, 115, 126, 134, 148, 165]
    rr_odd = [2, 4, 6, 9, 11, 23, 27, 33, 37, 44]
    envs = [("silence", 0.9), ("speech", 0.8), ("noise", 0.5), ("music", 0.7),
            ("alarm", 0.9), ("alarm", 0.45), ("impact", 0.85), ("impact", 0.62)]
    kws = [[], [], [], [], ["살려"], ["도와"], ["119"], ["응급"], ["화재"], ["아파"]]
    defs = []
    for i in range(n):
        fall_det = rng.random() < 0.35
        fall = round(rng.uniform(0.8, 0.99) if fall_det else rng.uniform(0.0, 0.95), 2)
        env, ec = rng.choice(envs)
        kw = rng.choice(kws)
        if kw and rng.random() < 0.5:
            env, ec = "speech", 0.8
        defs.append({
            "id": f"R-{i + 1:03d}",
            "hr": rng.choice(hr_odd) if rng.random() < 0.5 else rng.choice([61, 69, 77, 86, 94]),
            "rr": rng.choice(rr_odd) if rng.random() < 0.4 else rng.choice([13, 15, 17, 19]),
            "fall": fall, "fall_det": fall_det, "env": env, "env_conf": ec,
            "tx": " ".join(kw), "kw": kw,
        })
    return defs


def build_expert(d: dict) -> dict:
    vc, fc, sc, ec = 0.70, 0.75, 0.55, d.get("env_conf", 0.80)
    kw, tx = d.get("kw", []), d.get("tx", "")
    return {
        "fall": {"fall_score": float(d["fall"]), "fall_detected": bool(d["fall_det"]),
                 "infer_confidence": fc},
        "vital": {"heart_rate": float(d["hr"]), "breathing_rate": float(d["rr"]),
                  "infer_confidence": vc},
        "env_sound": {"label": d["env"], "env_sound_label": d["env"],
                      "confidence": float(ec), "env_sound_confidence": float(ec),
                      "infer_confidence": ec},
        "speech_ko": {"transcript_ko": tx, "speech_detected": bool(tx),
                      "stt_confidence": sc if tx else 0.0, "keywords": kw,
                      "infer_confidence": sc},
    }


# ── 평가 ─────────────────────────────────────────────────────────────────────

def _pattern(er: dict) -> str:
    """과대 오답 유형 분류용 요약 태그."""
    tags = []
    if er["fall"]["fall_detected"]:
        tags.append("낙상확정")
    hr, rr = er["vital"]["heart_rate"], er["vital"]["breathing_rate"]
    if (0 < hr <= 40) or hr >= 130 or (0 < rr <= 5) or rr >= 35:
        tags.append("생체위기")
    elif (0 < hr < 60) or hr > 100 or (0 < rr < 12) or rr > 25:
        tags.append("생체이상")
    if er["env_sound"]["env_sound_label"] in ("alarm", "impact"):
        tags.append("위험음")
    if er["speech_ko"]["keywords"]:
        tags.append("키워드")
    return "+".join(tags) or "신호없음"


def run(defs: list[dict], qwen=None) -> list[dict]:
    recs = []
    for d in defs:
        er = build_expert(d)
        score, bd = compute_emergency_score(er)
        gt, gt_reason = rubric_level(er, score, bd)
        rec = {"id": d["id"], "gate": round(score, 4), "m5_called": score >= WARNING_THRESHOLD,
               "gt": gt, "gt_reason": gt_reason, "pattern": _pattern(er)}
        if qwen is not None and rec["m5_called"]:
            out = qwen.evaluate(er)
            raw = out.get("qwen_response") or ""
            rec.update({
                "raw_level": _parse_raw_response(raw)[0].strip().lower(),
                "final_level": str(out.get("risk_level", "")).lower(),
                "raw": raw, "infer_ms": out.get("qwen_infer_ms"),
                "prompt_tokens": out.get("prompt_tokens"),
            })
        recs.append(rec)
    return recs


def _stats(op: list[dict], key: str) -> dict:
    exact = sum(1 for r in op if r[key] == r["gt"])
    over = sum(1 for r in op if _L.get(r[key], -1) > _L[r["gt"]])
    under = sum(1 for r in op if 0 <= _L.get(r[key], -1) < _L[r["gt"]])
    bad = sum(1 for r in op if r[key] not in _L)
    return {"exact": exact, "over": over, "under": under, "unparsed": bad, "n": len(op)}


def report(recs: list[dict], meta: dict) -> dict:
    op = [r for r in recs if r["m5_called"]]
    print(f"\n=== Track R  seed={meta['seed']}  N={len(recs)}  m2_off={meta['m2_off']} ===")
    print(f"  운영 구간(게이트>=0.6): {len(op)}건   판정표 정답 분포: {dict(Counter(r['gt'] for r in op))}")
    summary = {"n_total": len(recs), "n_op": len(op),
               "gt_dist": dict(Counter(r["gt"] for r in op))}
    if op and "raw_level" in op[0]:
        for key, name in (("raw_level", "모델만(raw)"), ("final_level", "최종(가드레일)")):
            s = _stats(op, key)
            summary[key] = s
            print(f"  {name:14s}: exact {s['exact']}/{s['n']} ({s['exact']/s['n']*100:.1f}%)"
                  f"  과대 {s['over']}  과소 {s['under']}  파싱실패 {s['unparsed']}")
        over_pat = Counter(f"{r['pattern']} ({r['gt']}→{r['raw_level']})"
                           for r in op if _L.get(r["raw_level"], -1) > _L[r["gt"]])
        under_pat = Counter(f"{r['pattern']} ({r['gt']}→{r['raw_level']})"
                            for r in op if 0 <= _L.get(r["raw_level"], -1) < _L[r["gt"]])
        summary["over_patterns"] = dict(over_pat)
        summary["under_patterns"] = dict(under_pat)
        print("  과대 유형(모델만):")
        for p, c in over_pat.most_common():
            print(f"    {c:3d}× {p}")
        if under_pat:
            print("  과소 유형(모델만):")
            for p, c in under_pat.most_common():
                print(f"    {c:3d}× {p}")
        ms = sorted(r["infer_ms"] for r in op if r.get("infer_ms"))
        if ms:
            summary["infer_ms_p50"] = ms[len(ms) // 2]
            print(f"  infer_ms p50 {ms[len(ms) // 2]:.0f}  (개발 PC — RPi5 아님)")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Track R — 판정표 준수율 (rp5 방식)")
    ap.add_argument("--random", type=int, default=150)
    ap.add_argument("--seed", type=int, default=4047)
    ap.add_argument("--m2-off", action="store_true", help="심박·호흡 미측정(0) — 현재 운영 조건")
    ap.add_argument("--mock", action="store_true", help="모델 없이 게이트·정답 분포만")
    ap.add_argument("--impl", default="gguf", choices=["gguf", "15b"])
    ap.add_argument("--model", default="volumes/models/qwen_15b_gguf_q5")
    ap.add_argument("--tokenizer", default="volumes/models/qwen_15b")
    ap.add_argument("--out", default=None, help="결과 JSON 경로(기본 reports/track_r_<seed>.json)")
    args = ap.parse_args()

    defs = random_case_defs(args.random, args.seed)
    if args.m2_off:
        for d in defs:
            d["hr"] = d["rr"] = 0

    qwen = None
    if not args.mock:
        if args.impl == "gguf":
            from inference.qwen_gguf import QwenLogic
            qwen = QwenLogic(args.model, tokenizer_dir=args.tokenizer)
        else:
            from inference.qwen_15b import QwenLogic
            qwen = QwenLogic(args.model)

    meta = {"seed": args.seed, "random": args.random, "m2_off": args.m2_off,
            "impl": None if args.mock else args.impl,
            "model": None if args.mock else args.model,
            "date": f"{datetime.date.today():%Y-%m-%d}"}
    recs = run(defs, qwen)
    summary = report(recs, meta)

    if not args.mock:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = args.out or os.path.join(
            root, "reports", f"track_r_{args.seed}{'_m2off' if args.m2_off else ''}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "summary": summary, "results": recs}, f,
                      ensure_ascii=False, indent=2)
        print(f"[INFO] 결과: {out}")


if __name__ == "__main__":
    main()
