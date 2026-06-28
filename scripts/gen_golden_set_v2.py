"""
골든셋 프로그래매틱 생성기 v2 — 노인 실측 vital 분포 + 모순신호 노이즈 쌍.

기존 gen_golden_set.py 헬퍼 재사용(_i / _build_expected / ground_truth / _decisive_domains).
- (a) 경계 그리드: BVA 화이트박스 값 명시 주입(결정적, 회귀 보존)
- (b) 분포 샘플링: 노인 실제 HR/RR 분포(대부분 정상 + 현실 유병률 꼬리)
      근거: HR 정상60-100(최빈75)·서맥<60 유병<5%·중증<40 드묾·빈맥>100 / RR 정상12-20·빈호흡≥20흔함·서호흡<10드묾
- (c) 중복 제거 + 목표 N(clean) → 각 clean에 모순신호 noisy twin → 총 2N
- (d) 모순신호: 응급엔 진정맥락(music/발화), 정상엔 약한경보(alarm 저conf) 주입(비결정 → ground_truth 보존)

실행:
  python scripts/gen_golden_set_v2.py --n 250            # 500케이스(clean250+noisy250)
  python scripts/gen_golden_set_v2.py --n 500 --dry      # 1000, 분포/혼동행렬만
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

import gen_golden_set as G  # noqa: E402  (_i, _build_expected, ground_truth, _decisive_domains)
from inference.emergency_score import compute_emergency_score  # noqa: E402

_OUT = os.path.join(_ROOT, "data", "qwen_golden_set.jsonl")
_M5 = 0.6
_DISTRESS = ["살려", "도와", "응급", "위험", "119"]


# ── (b) 노인 실측 분포 샘플러 ────────────────────────────────────────────────
def sample_hr():
    r = random.random()
    if r < 0.015:                       # 중증 서맥 (crit) — 드묾
        return random.randint(30, 39)
    if r < 0.06:                        # 서맥 (warn) — 유병 <5%
        return random.randint(40, 59)
    if r < 0.86:                        # 정상 60-100 (최빈 75)
        return int(min(99, max(60, round(random.gauss(75, 10)))))
    if r < 0.97:                        # 빈맥 (warn)
        return random.randint(100, 129)
    return random.randint(130, 165)     # 중증 빈맥 (crit)


def sample_rr():
    r = random.random()
    if r < 0.012:                       # 중증 서호흡 (crit) — 드묾
        return random.randint(3, 5)
    if r < 0.045:                       # 서호흡 (warn)
        return random.randint(6, 11)
    if r < 0.80:                        # 정상 12-20 (최빈 16)
        return int(min(20, max(12, round(random.gauss(16, 2.3)))))
    if r < 0.96:                        # 빈호흡 (warn) — 흔함
        return random.randint(21, 33)
    return random.randint(35, 45)       # 중증 빈호흡 (crit)


def sample_fall():
    r = random.random()
    if r < 0.72:
        return 0.0
    if r < 0.90:
        return round(random.uniform(0.3, 0.7), 2)      # 중등도 의심
    return round(random.uniform(0.85, 1.0), 2)         # 확정 낙상


def sample_env():
    r = random.random()
    if r < 0.70:
        return "silence", 0.0
    if r < 0.88:
        return random.choice(["noise", "music", "speech"]), round(random.uniform(0.5, 1.0), 2)
    return random.choice(["alarm", "impact"]), round(random.uniform(0.6, 1.0), 2)


def sample_case():
    hr, rr = sample_hr(), sample_rr()
    fall = sample_fall()
    label, sconf = sample_env()
    kws, sdet = [], False
    if random.random() < 0.05:                 # 긴급 키워드 드묾
        kws = [random.choice(_DISTRESS)]
        sdet = True
    elif random.random() < 0.15:               # 일반 발화
        sdet = True
    stt = 1.0 if kws else 0.0
    return G._i(fall=fall, hr=hr, rr=rr, label=label, sconf=sconf, kws=kws, stt=stt, sdet=sdet)


# ── (a) 경계 그리드 (결정적, BVA 회귀) ──────────────────────────────────────
def boundary_grid():
    cases = []
    for hr in (33, 34, 35, 36, 40, 41, 55, 56, 99, 100, 129, 130, 131, 160):
        cases.append(G._i(hr=hr, rr=16))
    for rr in (2, 3, 4, 5, 6, 10, 11, 21, 22, 33, 34, 35, 38):
        cases.append(G._i(hr=72, rr=rr))
    for fall in (0.24, 0.25, 0.59, 0.60, 0.89, 0.90, 1.0):
        cases.append(G._i(fall=fall))
    # 복합/키워드/경보 경계
    cases += [
        G._i(fall=1.0, label="alarm", sconf=1.0),
        G._i(fall=0.6, label="alarm", sconf=0.9, kws=["살려"], stt=1.0, sdet=True),
        G._i(fall=0.25, kws=["살려"], stt=1.0, sdet=True),
        G._i(fall=0.9, hr=45, label="alarm", sconf=1.0),
        G._i(hr=130, label="noise", sconf=1.0),       # vital위기 + 비경보(환각가드)
        G._i(rr=4, label="speech", sconf=1.0),
    ]
    return cases


def _categorize(inp):
    v = inp.get("vital", {})
    hr = float(v.get("heart_rate", 0)); rr = float(v.get("breathing_rate", 0))
    fall = float(inp.get("fall", {}).get("fall_score", 0))
    label = inp.get("env_sound", {}).get("env_sound_label", "silence")
    kws = inp.get("speech_ko", {}).get("keywords") or []
    vital_crisis = (hr and (hr <= 40 or hr >= 130)) or (rr and (rr <= 5 or rr >= 34))
    doms = sum([bool(fall >= 0.5), bool(hr and (hr < 60 or hr > 100)),
                bool(label in ("alarm", "impact")), bool(kws)])
    if vital_crisis:
        return "vital_crisis"
    if doms >= 2:
        return "multi_domain"
    if fall >= 0.5:
        return "fall_only"
    if doms == 1:
        return "no_signal"
    return "normal"


# ── (d) 모순신호 노이즈 ──────────────────────────────────────────────────────
def perturb_conflict(inp):
    """비결정 모순 신호 1개 주입 → noisy twin. gt 보존(모순=비결정) 확인, 깨지면 no-op."""
    gt0 = G.ground_truth(inp)
    out = copy.deepcopy(inp)
    snd = out.get("env_sound", {})
    sp = out.get("speech_ko", {})
    sound_empty = (snd.get("env_sound_label", "silence") == "silence"
                   and not snd.get("env_sound_confidence"))
    speech_empty = not (sp.get("keywords") or sp.get("speech_detected"))
    injected = None
    if gt0:  # 응급 → 진정 맥락(모순)
        if sound_empty:
            out["env_sound"] = {"env_sound_label": "music", "env_sound_confidence": 0.8}
            injected = "음악(진정)"
        elif speech_empty:
            out["speech_ko"] = {**sp, "speech_detected": True}
            injected = "발화(잡담)"
    else:    # 정상 → 약한 경보(모순)
        if sound_empty:
            out["env_sound"] = {"env_sound_label": "alarm", "env_sound_confidence": 0.3}
            injected = "약한경보"
        elif speech_empty:
            out["speech_ko"] = {**sp, "speech_detected": True}
            injected = "발화"
    if injected is None or G.ground_truth(out) != gt0:
        return copy.deepcopy(inp), None
    return out, injected


def _sig(inp):
    return json.dumps(inp, ensure_ascii=False, sort_keys=True)


def build(n_clean):
    seen, clean = set(), []
    # 그리드 먼저(항상 포함)
    for inp in boundary_grid():
        s = _sig(inp)
        if s not in seen:
            seen.add(s); clean.append((inp, "grid"))
    # 분포 샘플(목표까지)
    guard = 0
    while len(clean) < n_clean and guard < n_clean * 50:
        guard += 1
        inp = sample_case()
        s = _sig(inp)
        if s not in seen:
            seen.add(s); clean.append((inp, "sampled"))
    clean = clean[:n_clean]

    rows, drop_stat = [], {}
    for i, (inp, source) in enumerate(clean):
        cat = _categorize(inp)
        exp, _ = G._build_expected(inp)
        exp["note"] = f"[{source}] {cat}"
        cid = f"c{i:04d}"
        rows.append({"id": cid, "category": cat, "ground_truth_emergency": G.ground_truth(inp),
                     "pair_id": cid, "noisy": False, "source": source, "input": inp, "expected": exp})
    for r in list(rows):
        ninp, inj = perturb_conflict(r["input"])
        nexp, _ = G._build_expected(ninp)
        nexp["note"] = f"[모순:{inj or '없음'}] {r['category']}"
        rows.append({"id": r["id"] + "-n", "category": r["category"],
                     "ground_truth_emergency": G.ground_truth(ninp),
                     "pair_id": r["id"], "noisy": True, "source": r["source"],
                     "input": ninp, "expected": nexp})
        drop_stat[inj or "없음"] = drop_stat.get(inj or "없음", 0) + 1
    return rows, clean, drop_stat


def _hist(clean):
    hr_b = {"<40": 0, "40-59": 0, "60-100": 0, "101-129": 0, ">=130": 0}
    rr_b = {"<=5": 0, "6-11": 0, "12-20": 0, "21-34": 0, ">=35": 0}
    for inp, _ in clean:
        hr = inp["vital"]["heart_rate"]; rr = inp["vital"]["breathing_rate"]
        hr_b["<40" if hr < 40 else "40-59" if hr < 60 else "60-100" if hr <= 100 else "101-129" if hr < 130 else ">=130"] += 1
        rr_b["<=5" if rr <= 5 else "6-11" if rr <= 11 else "12-20" if rr <= 20 else "21-34" if rr <= 34 else ">=35"] += 1
    return hr_b, rr_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250, help="clean 케이스 수 (총=2N)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed)

    rows, clean, drop_stat = build(args.n)

    # 혼동행렬 + 강건성
    tp = fp = tn = fn = 0
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        gt = r["ground_truth_emergency"]
        pred = compute_emergency_score(r["input"])[0] >= _M5
        if gt and pred: tp += 1
        elif gt and not pred: fn += 1
        elif (not gt) and pred: fp += 1
        else: tn += 1
    flips = []
    for r in rows:
        if r["noisy"]:
            continue
        cp = compute_emergency_score(r["input"])[0] >= _M5
        npd = compute_emergency_score(by_id[r["id"] + "-n"]["input"])[0] >= _M5
        if cp != npd:
            flips.append(r["id"])
    total = len(rows)
    acc = (tp + tn) / total
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    src = {}
    for inp, s in clean:
        src[s] = src.get(s, 0) + 1
    hr_b, rr_b = _hist(clean)

    print(f"총 {total}케이스 (clean {len(clean)} + noisy {len(clean)}) · source {src}")
    print(f"  응급(gt=true)={tp+fn}  정상(gt=false)={tn+fp}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}  정확도={acc:.3f} 오탐율(FPR)={fpr:.3f} 미탐율(FNR)={fnr:.3f}")
    print(f"  HR 분포: {hr_b}")
    print(f"  RR 분포: {rr_b}")
    print(f"  모순신호 주입 분포: {drop_stat}")
    print(f"  Track A 모순신호 flip: {len(flips)}/{len(clean)} {flips[:12]}{'...' if len(flips)>12 else ''}")

    if args.dry:
        return
    with open(_OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {total}케이스 → {_OUT}")


if __name__ == "__main__":
    main()
