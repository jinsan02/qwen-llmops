"""
qwen_golden_set.jsonl 생성기 (100케이스, 화이트박스 경계값 분석 지향).

- score_min/score_max는 compute_emergency_score 실제 출력에서 ±window로 산출(산술 오류 방지).
- ground_truth_emergency는 코드와 독립적인 '임상/상식' 오라클로 부여 → 정확도/오탐율/미탐율 측정용.
- 기존 40케이스는 입력/메타를 유지하고 ground_truth_emergency만 주입.

실행:
  python scripts/gen_golden_set.py            # data/qwen_golden_set.jsonl 덮어쓰기
  python scripts/gen_golden_set.py --dry      # 혼동행렬만 출력(쓰지 않음)
"""

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.emergency_score import compute_emergency_score

_M5_THRESHOLD = 0.6
_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "qwen_golden_set.jsonl")

_DISTRESS = {"살려", "도와", "응급", "위험", "넘어", "불", "화재", "119"}


# ── 독립 오라클: 실제 응급인가? (코드 점수와 무관한 사람 판단 기준) ──────────────
def ground_truth(inp: dict) -> bool:
    fall = float(inp.get("fall", {}).get("fall_score", 0.0))
    v = inp.get("vital", {})
    hr = float(v.get("heart_rate", 0.0))
    rr = float(v.get("breathing_rate", 0.0))
    snd = inp.get("env_sound", {})
    label = snd.get("env_sound_label") or snd.get("label") or "unknown"
    sp = inp.get("speech_ko", {})
    kw = set(sp.get("keywords") or [])
    has_distress = bool(_DISTRESS & kw)

    # 1) 명백한 생체신호 위기: 심한 서맥/빈맥/무호흡/과호흡
    if hr > 0 and (hr <= 40 or hr >= 130):
        return True
    if rr > 0 and (rr <= 5 or rr >= 34):
        return True
    # 2) 구조 요청(distress) + 낙상 정황
    if has_distress and fall >= 0.5:
        return True
    # 3) 확정 낙상 + 보강 신호(생체 경고 or 충격/경보음)
    vital_warn = (hr > 0 and (40 < hr <= 55 or 100 <= hr < 130)) or \
                 (rr > 0 and (5 < rr <= 10 or 22 <= rr < 34))
    if fall >= 0.9 and (vital_warn or label in ("alarm", "impact")):
        return True
    # 4) 경보음 + 중등도 낙상
    if label == "alarm" and float(snd.get("env_sound_confidence", 0.0)) >= 0.8 and fall >= 0.6:
        return True
    return False


def _vital_override(inp: dict) -> bool:
    # D1 정렬: crit_lo 40/5 (emergency_score와 동일)
    v = inp.get("vital", {})
    hr = float(v.get("heart_rate", 0.0))
    rr = float(v.get("breathing_rate", 0.0))
    return (hr > 0 and (hr <= 40 or hr >= 130)) or (rr > 0 and (rr <= 5 or rr >= 35))


def _numeric(inp, key, lo_crit, hi_crit):
    v = float(inp.get("vital", {}).get(key, 0.0))
    # crisis 값일 때만 reason이 수치를 언급해야 함
    if v > 0 and (v <= lo_crit or v >= hi_crit):
        return v
    return None


def _build_expected(inp: dict, window=0.025, **overrides) -> dict:
    score, _ = compute_emergency_score(inp)
    m5 = score >= _M5_THRESHOLD
    label = inp.get("env_sound", {}).get("env_sound_label") \
        or inp.get("env_sound", {}).get("label") or "silence"
    exp = {
        "score_min": round(max(0.0, score - window), 3),
        "score_max": round(min(1.0, score + window), 3),
        "m5_called": m5,
        "vital_override": _vital_override(inp),
        # hallucination_guard: 경보/충격음이 아닌데 M5까지 간 케이스 → '알람' 환각 금지
        "hallucination_guard": bool(m5 and label not in ("alarm", "impact")),
        "env_label": label,
        "numeric_hr": _numeric(inp, "heart_rate", 40.0, 130.0),
        "numeric_rr": _numeric(inp, "breathing_rate", 5.0, 35.0),
    }
    exp.update(overrides)
    return exp, score


# ── 신규 60케이스 입력 정의 ────────────────────────────────────────────────
def _i(fall=0.0, hr=0.0, rr=0.0, label="silence", sconf=0.0,
       kws=None, stt=0.0, sdet=False, fall_ic=None, vital_ic=None, sound_ic=None, speech_ic=None):
    fb = {"fall_score": fall}
    if fall_ic is not None:
        fb["infer_confidence"] = fall_ic
    vb = {"heart_rate": hr, "breathing_rate": rr}
    if vital_ic is not None:
        vb["infer_confidence"] = vital_ic
    eb = {"env_sound_label": label, "env_sound_confidence": sconf}
    if sound_ic is not None:
        eb["infer_confidence"] = sound_ic
    pb = {"keywords": kws or [], "stt_confidence": stt, "speech_detected": sdet}
    if speech_ic is not None:
        pb["infer_confidence"] = speech_ic
    return {"fall": fb, "vital": vb, "env_sound": eb, "speech_ko": pb}


NEW_CASES = [
    # ── normal +5 (정상 영역 / 오탐 방지) ────────────────────────────────
    ("normal-06", "normal", _i(hr=56, rr=16), "HR=56 → 정상 영역(warn_lo+1), 비응급"),
    ("normal-07", "normal", _i(hr=99, rr=16), "HR=99 → 정상 영역(warn_hi-1), 비응급"),
    ("normal-08", "normal", _i(hr=72, rr=11), "RR=11 → 정상 영역(warn_lo+1), 비응급"),
    ("normal-09", "normal", _i(hr=72, rr=21), "RR=21 → 정상 영역(warn_hi-1), 비응급"),
    ("normal-10", "normal", _i(fall=0.5, hr=72, rr=16, label="noise", sconf=0.4), "중간 낙상+소음, 생체정상 → 비응급"),

    # ── fall_only +6 ────────────────────────────────────────────────────
    ("fall-07", "fall_only", _i(fall=0.249), "fall=0.249 → keyword_fall_min(0.25) 직하, 단독 비응급"),
    ("fall-08", "fall_only", _i(fall=0.60), "fall=0.60 정확(임계 경계), 단독 비응급"),
    ("fall-09", "fall_only", _i(fall=1.0, label="impact", sconf=1.0), "fall=1.0+impact → 0.597, 0.6 직하 미탐 경계"),
    ("fall-10", "fall_only", _i(fall=1.0, label="alarm", sconf=1.0, sdet=True), "fall=1.0+alarm+발화 → 알람 보강 응급"),
    ("fall-11", "fall_only", _i(fall=0.65), "fall=0.65 단독 → 비응급(M5 미호출)"),
    ("fall-12", "fall_only", _i(fall=1.0, hr=56, rr=16), "fall=1.0+HR정상 → 0.40 단독, 비응급"),

    # ── vital_crisis +12 (HR/RR 위기, 모두 bypass) ──────────────────────
    ("vital-09", "vital_crisis", _i(hr=34, rr=16), "HR=34 → crit_lo(35) 직하, bypass"),
    ("vital-10", "vital_crisis", _i(hr=131, rr=16), "HR=131 → crit_hi(130) 직상, bypass"),
    ("vital-11", "vital_crisis", _i(hr=72, rr=3), "RR=3 → crit_lo(4) 직하, bypass"),
    ("vital-12", "vital_crisis", _i(hr=72, rr=36), "RR=36 → crit_hi(35) 직상, bypass"),
    ("vital-13", "vital_crisis", _i(hr=180, rr=30), "HR=180 극단 빈맥+RR경고, bypass"),
    ("vital-14", "vital_crisis", _i(hr=72, rr=2), "RR=2 극단 무호흡, bypass"),
    ("vital-15", "vital_crisis", _i(hr=33, rr=16, sdet=True), "HR=33 서맥+발화감지, bypass"),
    ("vital-16", "vital_crisis", _i(hr=140, rr=16, label="noise", sconf=1.0), "HR=140 빈맥+소음, bypass"),
    ("vital-17", "vital_crisis", _i(hr=72, rr=38), "RR=38 과호흡, bypass"),
    ("vital-18", "vital_crisis", _i(hr=30, rr=8), "HR=30 심한 서맥+RR경고, bypass"),
    ("vital-19", "vital_crisis", _i(fall=0.4, hr=132, rr=16), "HR=132+중간낙상, bypass"),
    ("vital-20", "vital_crisis", _i(fall=0.3, hr=72, rr=3), "RR=3+약한낙상, bypass"),

    # ── multi_domain +9 ─────────────────────────────────────────────────
    ("multi-10", "multi_domain", _i(fall=0.7, hr=50, rr=16, label="noise", sconf=1.0), "fall0.7+HR경고+소음 → 0.57 경계 미탐"),
    ("multi-11", "multi_domain", _i(fall=0.8, hr=50, rr=16, label="alarm", sconf=1.0), "fall0.8+HR경고+alarm → 3도메인"),
    ("multi-12", "multi_domain", _i(fall=0.6, hr=72, rr=24, label="impact", sconf=1.0), "fall0.6+RR경고+impact → 3도메인"),
    ("multi-13", "multi_domain", _i(hr=110, rr=16, label="alarm", sconf=1.0), "HR110경고+alarm → 경증 복합, 비응급"),
    ("multi-14", "multi_domain", _i(fall=0.5, hr=50, rr=16, label="noise", sconf=0.5, sdet=True), "약한 복합신호 → 비응급"),
    ("multi-15", "multi_domain", _i(fall=1.0, hr=72, rr=20, label="music", sconf=1.0), "fall1.0+음악 → 단독수준, 비응급"),
    ("multi-16", "multi_domain", _i(fall=0.9, hr=45, rr=16, label="alarm", sconf=1.0, kws=["살려"], stt=1.0, sdet=True), "fall0.9+HR경고+alarm+키워드 → 4도메인"),
    ("multi-17", "multi_domain", _i(fall=0.7, hr=128, rr=16, label="impact", sconf=1.0, kws=["도와"], stt=1.0, sdet=True), "fall0.7+HR128경고+impact+키워드 → 4도메인"),
    ("multi-18", "multi_domain", _i(fall=0.3, hr=72, rr=16, label="alarm", sconf=0.9, kws=["살려"], stt=1.0, sdet=True), "약한낙상+alarm+키워드 → bonus 경계"),

    # ── no_signal +6 (infer_confidence 경계 0.0/0.5/1.0) ────────────────
    ("no-signal-05", "no_signal", _i(hr=130, rr=16, vital_ic=0.5), "HR=130 위기+vital_conf=0.5 → bypass는 raw기반, 응급"),
    ("no-signal-06", "no_signal", _i(fall=1.0, label="alarm", sconf=1.0, fall_ic=0.5), "fall_conf=0.5 감쇠 → 0.52 미탐 경계"),
    ("no-signal-07", "no_signal", _i(hr=50, rr=16, label="alarm", sconf=1.0), "HR경고+alarm conf1.0 → 0.36 경증, 비응급"),
    ("no-signal-08", "no_signal", _i(fall=1.0, fall_ic=1.0), "fall=1.0 conf1.0 단독 → 0.40, 비응급"),
    ("no-signal-09", "no_signal", _i(hr=50, rr=16, vital_ic=0.0), "HR경고+vital_conf=0.0 → 0.0825, 비응급"),
    ("no-signal-10", "no_signal", _i(fall=1.0, label="alarm", sconf=1.0, sound_ic=0.0), "sound_conf=0.0 → alarm 억제, fall단독 0.40 비응급"),

    # ── hallucination_guard +4 (env_label 다양화, 알람 환각 금지) ─────────
    ("hg-09", "hallucination_guard", _i(hr=130, rr=16, label="noise", sconf=1.0), "HR=130 위기+noise → bypass, '알람' 환각 금지"),
    ("hg-10", "hallucination_guard", _i(fall=1.0, hr=45, rr=16, label="music", sconf=1.0, kws=["살려"], stt=1.0, sdet=True), "고위험+music → 환각 금지"),
    ("hg-11", "hallucination_guard", _i(hr=72, rr=36, label="speech", sconf=1.0), "RR=36 위기+speech → bypass, 환각 금지"),
    ("hg-12", "hallucination_guard", _i(fall=0.9, hr=128, rr=16, label="silence", sconf=0.0, kws=["도와"], stt=1.0, sdet=True), "fall0.9+HR128+키워드+silence → 환각 금지"),

    # ── boundary_bva +18 (순수 화이트박스 경계값) ────────────────────────
    ("bva-01", "boundary_bva", _i(hr=55, rr=16), "HR=55 → warn_lo 포함(≤55) 경고"),
    ("bva-02", "boundary_bva", _i(hr=56, rr=16), "HR=56 → warn_lo+1 정상 전이"),
    ("bva-03", "boundary_bva", _i(hr=100, rr=16), "HR=100 → warn_hi 포함(≥100) 경고"),
    ("bva-04", "boundary_bva", _i(hr=99, rr=16), "HR=99 → warn_hi-1 정상"),
    ("bva-05", "boundary_bva", _i(hr=35, rr=16), "HR=35 → crit_lo 포함(≤35) 위기, bypass"),
    ("bva-06", "boundary_bva", _i(hr=36, rr=16), "HR=36 → crit_lo+1 경고(0.55)만 → 심한 서맥 미탐 경계"),
    ("bva-07", "boundary_bva", _i(hr=130, rr=16), "HR=130 → crit_hi 포함(≥130) 위기, bypass"),
    ("bva-08", "boundary_bva", _i(hr=129, rr=16), "HR=129 → crit_hi-1 경고(0.55)만"),
    ("bva-09", "boundary_bva", _i(hr=72, rr=10), "RR=10 → warn_lo 포함(≤10) 경고"),
    ("bva-10", "boundary_bva", _i(hr=72, rr=11), "RR=11 → warn_lo+1 정상"),
    ("bva-11", "boundary_bva", _i(hr=72, rr=22), "RR=22 → warn_hi 포함(≥22) 경고"),
    ("bva-12", "boundary_bva", _i(hr=72, rr=4), "RR=4 → crit_lo 포함(≤4) 위기, bypass"),
    ("bva-13", "boundary_bva", _i(hr=72, rr=5), "RR=5 → crit_lo+1 경고만 → 심한 서호흡 미탐 경계"),
    ("bva-14", "boundary_bva", _i(hr=72, rr=35), "RR=35 → crit_hi 포함(≥35) 위기, bypass"),
    ("bva-15", "boundary_bva", _i(fall=0.25, kws=["살려"], stt=1.0, sdet=True), "fall=0.25 정확+키워드 → keyword_fall_bonus 발동"),
    ("bva-16", "boundary_bva", _i(fall=0.24, kws=["살려"], stt=1.0, sdet=True), "fall=0.24 → bonus 미발동(0.25 직하)"),
    ("bva-17", "boundary_bva", _i(fall=1.0, hr=50, rr=16, label="alarm", sconf=1.0), "fall1.0+HR경고+alarm 3도메인 → composite ×1.35 경계"),
    ("bva-18", "boundary_bva", _i(fall=0.6, hr=50, rr=16, label="alarm", sconf=0.9, kws=["살려"], stt=0.9, sdet=True), "4도메인 활성 → composite ×1.50 상한 경계"),
]


# ── 결측 노이즈: 비결정(non-decisive) 도메인 1개 제거 → 강건성 쌍 생성 ──────────
def _decisive_domains(inp: dict) -> set:
    """ground_truth()를 만든(=판정을 좌우한) 도메인 집합. 이건 제거하면 안 됨."""
    fall = float(inp.get("fall", {}).get("fall_score", 0.0))
    v = inp.get("vital", {})
    hr = float(v.get("heart_rate", 0.0)); rr = float(v.get("breathing_rate", 0.0))
    snd = inp.get("env_sound", {})
    label = snd.get("env_sound_label") or snd.get("label") or "unknown"
    sconf = float(snd.get("env_sound_confidence", 0.0))
    kw = set(inp.get("speech_ko", {}).get("keywords") or [])
    has_distress = bool(_DISTRESS & kw)
    vital_warn = (hr > 0 and (40 < hr <= 55 or 100 <= hr < 130)) or \
                 (rr > 0 and (5 < rr <= 10 or 22 <= rr < 34))
    dec = set()
    if hr > 0 and (hr <= 40 or hr >= 130):
        dec.add("vital")
    if rr > 0 and (rr <= 5 or rr >= 34):
        dec.add("vital")
    if has_distress and fall >= 0.5:
        dec.update({"speech", "fall"})
    if fall >= 0.9 and (vital_warn or label in ("alarm", "impact")):
        dec.add("fall")
        if label in ("alarm", "impact"):
            dec.add("sound")
        if vital_warn:
            dec.add("vital")
    if label == "alarm" and sconf >= 0.8 and fall >= 0.6:
        dec.update({"sound", "fall"})
    return dec


def _present_domains(inp: dict) -> set:
    pres = set()
    if float(inp.get("fall", {}).get("fall_score", 0.0)) > 0:
        pres.add("fall")
    v = inp.get("vital", {})
    if float(v.get("heart_rate", 0.0)) > 0 or float(v.get("breathing_rate", 0.0)) > 0:
        pres.add("vital")
    snd = inp.get("env_sound", {})
    if (snd.get("env_sound_label") or "silence") != "silence" or float(snd.get("env_sound_confidence", 0.0)) > 0:
        pres.add("sound")
    sp = inp.get("speech_ko", {})
    if (sp.get("keywords") or sp.get("speech_detected")):
        pres.add("speech")
    return pres


_DOMAIN_KEY = {"fall": "fall", "vital": "vital", "sound": "env_sound", "speech": "speech_ko"}


def perturb_missing(inp: dict):
    """비결정 도메인 1개(센서)를 제거한 입력과 제거 도메인명을 반환.
    제거 우선순위 speech→sound→vital→fall (안전 영향 낮은 순). 없으면 (사본, None).
    비결정만 제거하므로 ground_truth는 보존된다(재계산해도 동일)."""
    dec = _decisive_domains(inp)
    pres = _present_domains(inp)
    for dom in ("speech", "sound", "vital", "fall"):
        if dom in pres and dom not in dec:
            out = copy.deepcopy(inp)
            out.pop(_DOMAIN_KEY[dom], None)
            return out, dom
    # 도메인 단위로 못 떨구면(주로 vital 단독 위기) vital 내 '비위기' 서브필드 제거
    v = inp.get("vital", {})
    hr = float(v.get("heart_rate", 0.0)); rr = float(v.get("breathing_rate", 0.0))
    hr_crit = hr > 0 and (hr <= 40 or hr >= 130)
    rr_crit = rr > 0 and (rr <= 5 or rr >= 34)
    if hr_crit and rr > 0 and not rr_crit:   # HR 위기·RR 비위기 → RR 센서 누락
        out = copy.deepcopy(inp); out["vital"].pop("breathing_rate", None); return out, "vital.rr"
    if rr_crit and hr > 0 and not hr_crit:   # RR 위기·HR 비위기 → HR 센서 누락
        out = copy.deepcopy(inp); out["vital"].pop("heart_rate", None); return out, "vital.hr"
    return copy.deepcopy(inp), None  # 단일 신호뿐 → 제거 가능한 비결정 센서 없음(트리비얼 강건)


def load_existing(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    out_rows = []
    _new_ids = {cid for cid, _, _, _ in NEW_CASES}

    # 1) 원본 케이스(NEW_CASES에 없는 것만): 멱등 보장 — NEW와 겹치는 id는 건너뛰고
    #    재append에 의한 중복 방지. expected 윈도우는 현재 코드 출력으로 재생성(코드 동기화),
    #    note는 보존.
    existing = load_existing(_OUT)
    for case in existing:
        if case["id"] in _new_ids:
            continue
        note = case.get("expected", {}).get("note", "")
        exp, _ = _build_expected(case["input"])
        exp["note"] = note
        case["expected"] = exp
        case["ground_truth_emergency"] = ground_truth(case["input"])
        out_rows.append(case)

    # 2) 신규 케이스
    for cid, cat, inp, note in NEW_CASES:
        exp, score = _build_expected(inp)
        exp["note"] = note
        out_rows.append({
            "id": cid,
            "category": cat,
            "ground_truth_emergency": ground_truth(inp),
            "input": inp,
            "expected": exp,
        })

    # 3) clean 태그 + 결측 노이즈 twin 생성 (100 → 200, clean↔noisy 쌍)
    clean_rows = out_rows
    for r in clean_rows:
        r["pair_id"] = r["id"]
        r["noisy"] = False
    noisy_rows = []
    drop_stat = {}
    for r in clean_rows:
        ninp, dropped = perturb_missing(r["input"])
        nexp, _ = _build_expected(ninp)
        nexp["note"] = f"[결측:{dropped or '없음'}] " + r["expected"].get("note", "")
        noisy_rows.append({
            "id": r["id"] + "-n",
            "category": r["category"],
            "ground_truth_emergency": ground_truth(ninp),  # 비결정 제거라 보존됨
            "pair_id": r["id"],
            "noisy": True,
            "input": ninp,
            "expected": nexp,
        })
        drop_stat[dropped or "없음"] = drop_stat.get(dropped or "없음", 0) + 1
    out_rows = clean_rows + noisy_rows

    # 4) 혼동행렬 점검 (system 결정 = score>=0.6)
    tp = fp = tn = fn = 0
    fp_ids, fn_ids = [], []
    for r in out_rows:
        gt = r["ground_truth_emergency"]
        sc, _ = compute_emergency_score(r["input"])
        pred = sc >= _M5_THRESHOLD
        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
            fn_ids.append((r["id"], round(sc, 3)))
        elif (not gt) and pred:
            fp += 1
            fp_ids.append((r["id"], round(sc, 3)))
        else:
            tn += 1

    # 5) Track A 강건성: 쌍별 pred(>=0.6) flip
    by_id = {r["id"]: r for r in out_rows}
    flips = []
    for r in clean_rows:
        cpred = compute_emergency_score(r["input"])[0] >= _M5_THRESHOLD
        npred = compute_emergency_score(by_id[r["id"] + "-n"]["input"])[0] >= _M5_THRESHOLD
        if cpred != npred:
            flips.append(r["id"])

    total = len(out_rows)
    acc = (tp + tn) / total
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    print(f"총 {total}케이스 (clean {len(clean_rows)} + noisy {len(noisy_rows)})")
    print(f"  응급(gt=true)={tp+fn}  정상(gt=false)={tn+fp}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"정확도={acc:.3f}  오탐율(FPR)={fpr:.3f}  미탐율(FNR)={fnr:.3f}")
    print(f"결측 제거 도메인 분포: {drop_stat}")
    print(f"Track A 쌍 flip(센서 누락에 판정 뒤집힘): {len(flips)}/{len(clean_rows)} {flips}")
    print(f"오탐(FP): {fp_ids}")
    print(f"미탐(FN): {fn_ids}")

    if args.dry:
        return

    with open(_OUT, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {total}케이스 → {_OUT}")


if __name__ == "__main__":
    main()
