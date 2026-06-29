"""
골든셋 생성기 v3 — 시계열(1h, ≤60행) 고도화 + 시계열 오라클 + 모순신호 noisy 쌍.

v2(=gen_golden_set 헬퍼) 재사용: _i / sample_* / boundary_grid / perturb_conflict / _categorize.
v2 대비 추가:
- 각 케이스에 `time_series`: [{m, hr, rr, event?}] (m=-(L-1)..0, 0=현재=스냅샷). 길이 0~60 랜덤.
- 시계열 패턴(스냅샷 정상/이상에 맞춰 선택): stable·gradual_deterioration·acute_spike·recovery·noisy_stable·sparse.
- `ground_truth_temporal(snapshot, series)`: 스냅샷 오라클 OR 시계열 트리거(지속 경고·점진 악화).
  코드(compute_emergency_score)와 독립 — 게이트와의 FP/FN 갭을 측정·수정하기 위함.
- boundary_grid 케이스는 빈 시계열(스냅샷 전용) → BVA 화이트박스 회귀 보존.

실행:
  python scripts/gen_golden_set_v3.py --n 500            # 1000케이스(clean500+noisy500)
  python scripts/gen_golden_set_v3.py --n 500 --dry      # 분포/혼동행렬만
"""

import argparse
import copy
import json
import os
import random
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _THIS)

import gen_golden_set as G  # noqa: E402
import gen_golden_set_v2 as V2  # noqa: E402  (sample_case, boundary_grid, perturb_conflict, _categorize)
from inference.emergency_score import compute_emergency_score  # noqa: E402

_OUT = os.path.join(_ROOT, "data", "qwen_golden_set.jsonl")
_M5 = 0.6


# ── 시계열 행 생성 ───────────────────────────────────────────────────────────
def _clip_hr(h):
    return int(max(30, min(180, round(h))))


def _clip_rr(r):
    return int(max(3, min(45, round(r))))


def _row(m, hr, rr, event=None):
    d = {"m": int(m), "hr": _clip_hr(hr), "rr": _clip_rr(rr)}
    if event:
        d["event"] = event
    return d


def _snap_event(inp):
    """스냅샷에서 m=0 행에 붙일 이벤트 태그(프롬프트 현실감용; 오라클엔 미사용)."""
    fall = float(inp.get("fall", {}).get("fall_score", 0.0))
    label = inp.get("env_sound", {}).get("env_sound_label", "silence")
    kws = inp.get("speech_ko", {}).get("keywords") or []
    if kws:
        return "긴급"
    if fall >= 0.5:
        return "낙상"
    if label in ("alarm", "impact"):
        return "위험음"
    return None


def _abnormal(hr, rr):
    return (hr > 0 and (hr <= 55 or hr >= 100)) or (rr > 0 and (rr <= 10 or rr >= 22))


def _build_series(inp, pattern, L):
    """패턴에 맞는 ≤60행 시계열을 생성. 마지막 행(m=0)은 스냅샷 vital과 정합."""
    v = inp.get("vital", {})
    hr0 = float(v.get("heart_rate", 0) or 0)
    rr0 = float(v.get("breathing_rate", 0) or 0)
    if hr0 <= 0:
        hr0 = 75.0
    if rr0 <= 0:
        rr0 = 16.0
    if L <= 0:
        return []

    ms = list(range(-(L - 1), 1))  # -(L-1) .. 0
    rows = []

    if pattern == "stable":
        for m in ms:
            rows.append(_row(m, hr0 + random.gauss(0, 2), rr0 + random.gauss(0, 1)))
    elif pattern == "gradual_deterioration":
        hs, rs = 75.0, 16.0  # 정상 베이스라인 → 스냅샷으로 단조 이동
        for i, m in enumerate(ms):
            f = i / max(1, L - 1)
            rows.append(_row(m, hs + (hr0 - hs) * f + random.gauss(0, 1.5),
                             rs + (rr0 - rs) * f + random.gauss(0, 0.8)))
    elif pattern == "acute_spike":
        k = random.randint(2, 4)  # 마지막 k행만 급변
        for i, m in enumerate(ms):
            if i < L - k:
                rows.append(_row(m, 75 + random.gauss(0, 2), 16 + random.gauss(0, 1)))
            else:
                rows.append(_row(m, hr0 + random.gauss(0, 1), rr0 + random.gauss(0, 0.5)))
    elif pattern == "recovery":
        hs = hr0 + random.randint(30, 55)  # 악화 상태 → 스냅샷(정상)으로 회복
        rs = rr0 + random.randint(8, 16)
        for i, m in enumerate(ms):
            f = i / max(1, L - 1)
            rows.append(_row(m, hs + (hr0 - hs) * f + random.gauss(0, 1.5),
                             rs + (rr0 - rs) * f + random.gauss(0, 0.8)))
    elif pattern == "noisy_stable":
        for m in ms:
            rows.append(_row(m, hr0 + random.gauss(0, 6), rr0 + random.gauss(0, 3)))
    else:  # sparse — 짧은 시계열
        for m in ms:
            rows.append(_row(m, hr0 + random.gauss(0, 2), rr0 + random.gauss(0, 1)))

    # 마지막 행을 스냅샷 정확값으로 고정(정합) + 이벤트 태그
    rows[-1] = _row(0, hr0, rr0, _snap_event(inp))
    return rows


def _pick_pattern_and_len(inp):
    v = inp.get("vital", {})
    hr0 = float(v.get("heart_rate", 0) or 0)
    rr0 = float(v.get("breathing_rate", 0) or 0)
    abn = _abnormal(hr0, rr0)
    r = random.random()
    if r < 0.10:
        return "sparse", random.randint(0, 8)
    if abn:
        # 이상 스냅샷: 정상→이상 궤적(지속/악화/급변)
        p = random.choices(["stable", "gradual_deterioration", "acute_spike"],
                           weights=[0.4, 0.4, 0.2])[0]
    else:
        # 정상 스냅샷: 평탄/회복/지터 (시계열-응급 아님)
        p = random.choices(["stable", "recovery", "noisy_stable"],
                           weights=[0.5, 0.25, 0.25])[0]
    return p, random.randint(10, 60)


# ── 시계열 오라클 (코드 독립) ────────────────────────────────────────────────
def _wow(r):
    h = float(r.get("hr", 0) or 0)
    rr = float(r.get("rr", 0) or 0)
    return (h > 0 and (h <= 55 or h >= 100)) or (rr > 0 and (rr <= 10 or rr >= 22))


def _qmean(vals):
    if not vals:
        return 0.0, 0.0
    if len(vals) < 4:
        m = sum(vals) / len(vals)
        return m, m
    q = max(1, len(vals) // 4)
    return sum(vals[:q]) / q, sum(vals[-q:]) / q


def ground_truth_temporal(snapshot, series):
    """스냅샷 임상 오라클 OR 시계열 트리거(지속 경고/점진 악화). compute_emergency_score와 독립."""
    if G.ground_truth(snapshot):
        return True
    if not series or len(series) < 10:
        return False
    recent = series[-20:]
    sustained_ratio = sum(1 for r in recent if _wow(r)) / len(recent)
    hrs = [float(r["hr"]) for r in series if float(r.get("hr", 0) or 0) > 0]
    rrs = [float(r["rr"]) for r in series if float(r.get("rr", 0) or 0) > 0]
    h0, h1 = _qmean(hrs)
    r0, r1 = _qmean(rrs)
    if sustained_ratio >= 0.60:                       # 지속 경고 누적
        return True
    if (h1 - h0) >= 30 and h1 >= 100:                 # 빈맥 점진 악화
        return True
    if (h0 - h1) >= 25 and 0 < h1 <= 58:              # 서맥 점진 악화
        return True
    if (r1 - r0) >= 10 and r1 >= 22:                  # 빈호흡 점진 악화
        return True
    return False


def _sig(inp, series):
    return json.dumps([inp, series], ensure_ascii=False, sort_keys=True)


def build(n_clean):
    seen, clean = set(), []
    # 그리드 먼저(빈 시계열 — 스냅샷 BVA 회귀 보존)
    for inp in V2.boundary_grid():
        s = _sig(inp, [])
        if s not in seen:
            seen.add(s)
            clean.append((inp, [], "grid"))
    # 분포 샘플 + 시계열
    guard = 0
    while len(clean) < n_clean and guard < n_clean * 50:
        guard += 1
        inp = V2.sample_case()
        pat, L = _pick_pattern_and_len(inp)
        series = _build_series(inp, pat, L)
        s = _sig(inp, series)
        if s not in seen:
            seen.add(s)
            clean.append((inp, series, "sampled:" + pat))
    clean = clean[:n_clean]

    rows = []
    for i, (inp, series, source) in enumerate(clean):
        cat = V2._categorize(inp)
        exp = _build_expected_ts(inp, series)
        exp["note"] = f"[{source}] {cat} (ts={len(series)})"
        cid = f"c{i:04d}"
        rows.append({"id": cid, "category": cat,
                     "ground_truth_emergency": ground_truth_temporal(inp, series),
                     "pair_id": cid, "noisy": False, "source": source,
                     "input": inp, "time_series": series, "expected": exp})
    # 모순신호 noisy twin (스냅샷에 비결정 모순 주입; 시계열 동일)
    for r in list(rows):
        ninp, inj = V2.perturb_conflict(r["input"])
        nseries = r["time_series"]
        nexp = _build_expected_ts(ninp, nseries)
        nexp["note"] = f"[모순:{inj or '없음'}] {r['category']} (ts={len(nseries)})"
        rows.append({"id": r["id"] + "-n", "category": r["category"],
                     "ground_truth_emergency": ground_truth_temporal(ninp, nseries),
                     "pair_id": r["id"], "noisy": True, "source": r["source"],
                     "input": ninp, "time_series": nseries, "expected": nexp})
    return rows, clean


def _build_expected_ts(inp, series):
    """스냅샷 기준 expected + 시계열 게이트 점수로 score_min/max·m5_called 덮어쓰기."""
    exp, _ = G._build_expected(inp)
    sc, _ = compute_emergency_score(inp, time_series=series)
    exp["score_min"] = round(max(0.0, sc - 0.025), 3)
    exp["score_max"] = round(min(1.0, sc + 0.025), 3)
    exp["m5_called"] = sc >= _M5
    return exp


def _pattern_hist(clean):
    h = {}
    for _, _, src in clean:
        p = src.split(":", 1)[1] if ":" in src else src
        h[p] = h.get(p, 0) + 1
    return h


def _len_hist(clean):
    b = {"0": 0, "1-9": 0, "10-29": 0, "30-59": 0, "60": 0}
    for _, s, _ in clean:
        n = len(s)
        b["0" if n == 0 else "1-9" if n < 10 else "10-29" if n < 30 else "30-59" if n < 60 else "60"] += 1
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="clean 케이스 수 (총=2N)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed)

    rows, clean = build(args.n)

    # 혼동행렬 (게이트 시계열 점수 vs 시계열 오라클)
    tp = fp = tn = fn = 0
    fp_ids, fn_ids = [], []
    ts_only = 0  # 시계열로만 응급된 케이스 수(스냅샷 오라클은 False)
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        gt = r["ground_truth_emergency"]
        sc = compute_emergency_score(r["input"], time_series=r["time_series"])[0]
        pred = sc >= _M5
        if gt and not G.ground_truth(r["input"]):
            ts_only += 1
        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1; fn_ids.append((r["id"], round(sc, 3)))
        elif (not gt) and pred:
            fp += 1; fp_ids.append((r["id"], round(sc, 3)))
        else:
            tn += 1

    # 모순신호 강건성 flip (시계열 동일, 스냅샷 모순만 — 판정 유지해야 강건)
    flips = []
    for r in rows:
        if r["noisy"]:
            continue
        cp = compute_emergency_score(r["input"], time_series=r["time_series"])[0] >= _M5
        tw = by_id.get(r["id"] + "-n")
        if tw is None:
            continue
        npd = compute_emergency_score(tw["input"], time_series=tw["time_series"])[0] >= _M5
        if cp != npd:
            flips.append(r["id"])

    total = len(rows)
    acc = (tp + tn) / total
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    print(f"총 {total}케이스 (clean {len(clean)} + noisy {len(clean)})")
    print(f"  응급(gt=true)={tp+fn}  정상(gt=false)={tn+fp}  시계열로만 응급={ts_only}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}  정확도={acc:.3f} FPR={fpr:.3f} FNR={fnr:.3f}")
    print(f"  패턴 분포: {_pattern_hist(clean)}")
    print(f"  시계열 길이 분포: {_len_hist(clean)}")
    print(f"  Track A 모순신호 flip: {len(flips)}/{len(clean)} {flips[:12]}{'...' if len(flips)>12 else ''}")
    print(f"  FP({len(fp_ids)}): {fp_ids[:14]}")
    print(f"  FN({len(fn_ids)}): {fn_ids[:14]}")

    if args.dry:
        return
    with open(_OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {total}케이스 → {_OUT}")


if __name__ == "__main__":
    main()
