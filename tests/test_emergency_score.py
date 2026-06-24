"""
경계값 테스트 — emergency_score.compute_emergency_score()

실행: python tests/test_emergency_score.py
실행 (verbose): python tests/test_emergency_score.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.emergency_score import (
    compute_emergency_score,
    _HR_WARN_LO, _HR_WARN_HI, _HR_CRIT_LO, _HR_CRIT_HI,
    _RR_WARN_LO, _RR_WARN_HI, _RR_CRIT_LO, _RR_CRIT_HI,
    _SOUND_WEIGHTS, _CRITICAL_KEYWORDS,
)

VERBOSE = "-v" in sys.argv

PASS = 0
FAIL = 0


def _mk(fall=0.0, hr=0.0, rr=0.0, label="silence", conf=0.0, keywords=None, speech_detected=False, stt_conf=0.0):
    return {
        "fall":      {"fall_score": fall},
        "vital":     {"heart_rate": hr, "breathing_rate": rr},
        "env_sound": {"env_sound_label": label, "env_sound_confidence": conf},
        "speech_ko": {
            "keywords": keywords or [],
            "stt_confidence": stt_conf,
            "speech_detected": speech_detected,
        },
    }


def check(name, score, breakdown, *, score_lo=None, score_hi=None, score_eq=None, atol=0.01, **bd_checks):
    global PASS, FAIL
    errors = []

    if score_eq is not None:
        if abs(score - score_eq) > atol:
            errors.append(f"score={score:.4f} expected={score_eq} (atol={atol})")
    if score_lo is not None and score < score_lo - atol:
        errors.append(f"score={score:.4f} < lo={score_lo}")
    if score_hi is not None and score > score_hi + atol:
        errors.append(f"score={score:.4f} > hi={score_hi}")

    for domain, (lo, hi) in bd_checks.items():
        v = breakdown.get(domain, -1)
        if not (lo - atol <= v <= hi + atol):
            errors.append(f"{domain}={v:.4f} not in [{lo}, {hi}]")

    if errors:
        FAIL += 1
        print(f"  FAIL  {name}")
        for e in errors:
            print(f"         {e}")
    else:
        PASS += 1
        if VERBOSE:
            print(f"  PASS  {name}  score={score:.4f}  {breakdown}")


# ── 기본 케이스 ──────────────────────────────────────────────────────────────
print("[1] 기본 케이스")

s, bd = compute_emergency_score(_mk())
check("모든 입력 0 → 점수 0", s, bd, score_eq=0.0)

s, bd = compute_emergency_score(_mk(fall=1.0, hr=75.0, rr=16.0, label="silence", conf=1.0))
check("낙상 max, 나머지 정상 → fall 도메인만 기여", s, bd, score_lo=0.38, score_hi=0.42, fall=(0.99, 1.0))


# ── M1: 낙상 점수 경계값 ─────────────────────────────────────────────────────
print("[2] M1 낙상 경계값")

for fall_v, expected_lo, expected_hi in [
    (0.0,  0.0,   0.0),
    (0.59, 0.0,   0.27),   # threshold(0.6) 미만
    (0.60, 0.22,  0.26),
    (1.0,  0.38,  0.42),   # fall 40% * 1.0
]:
    s, bd = compute_emergency_score(_mk(fall=fall_v))
    check(f"fall={fall_v}", s, bd, score_lo=expected_lo, score_hi=expected_hi)


# ── M2: 심박 경계값 ──────────────────────────────────────────────────────────
print("[3] M2 심박(HR) 경계값")

cases_hr = [
    (0.0,   0.0,   0.0,   "HR=0 (no-signal)"),
    (_HR_CRIT_LO + 0.1, 0.0,  0.17, "HR 위기 하한 직상 → warning 진입"),
    (_HR_CRIT_LO,       0.63, 0.67, "HR 위기 하한(40, D1) → vital=1.0 → vital_bypass → score=0.65"),
    (_HR_WARN_LO - 0.1, 0.14, 0.18, "HR 경고 하한(55) 미만 → vital=0.55, score~0.165"),
    (_HR_WARN_LO,       0.14, 0.18, "HR=55 정확히 → _vital_component val<=warn_lo True → vital=0.55"),
    (75.0,              0.0,  0.01, "HR 정상(75) → vital=0"),
    (_HR_WARN_HI + 0.1, 0.14, 0.18, "HR 경고 상한 초과 → vital=0.55"),
    (_HR_CRIT_HI,       0.63, 0.67, "HR 위기 상한(130) → vital=1.0 → vital_bypass → score=0.65"),
]
for hr, lo, hi, label in cases_hr:
    s, bd = compute_emergency_score(_mk(hr=hr))
    check(label, s, bd, score_lo=lo, score_hi=hi)


# ── M2: 호흡수 경계값 ────────────────────────────────────────────────────────
print("[4] M2 호흡수(RR) 경계값")

cases_rr = [
    (0.0,              0.0,  0.01, "RR=0 (no-signal)"),
    (_RR_CRIT_LO,      0.63, 0.67, "RR 위기 하한(5, D1) → vital=1.0 → vital_bypass → score=0.65"),
    (_RR_WARN_LO - 0.1,0.14, 0.18, "RR 경고 하한 미만 → vital=0.55"),
    (_RR_WARN_LO,      0.14, 0.18, "RR=10 정확히 → _vital_component val<=warn_lo True → vital=0.55"),
    (16.0,             0.0,  0.01, "RR 정상(16) → vital=0"),
    (_RR_WARN_HI + 0.1,0.14, 0.18, "RR 경고 상한 초과 → vital=0.55"),
    (_RR_CRIT_HI,      0.63, 0.67, "RR 위기 상한(35) → vital=1.0 → vital_bypass → score=0.65"),
]
for rr, lo, hi, label in cases_rr:
    s, bd = compute_emergency_score(_mk(rr=rr))
    check(label, s, bd, score_lo=lo, score_hi=hi)


# ── M3: 환경음 경계값 ────────────────────────────────────────────────────────
print("[5] M3 환경음 경계값")

for label, weight in _SOUND_WEIGHTS.items():
    s, bd = compute_emergency_score(_mk(label=label, conf=1.0))
    expected = weight * 0.15
    check(f"label={label} conf=1.0", s, bd, score_lo=expected - 0.02, score_hi=expected + 0.02)

# conf 스케일링
s, bd = compute_emergency_score(_mk(label="alarm", conf=0.5))
check("alarm conf=0.5 → sound_c=0.45", s, bd, sound=(0.43, 0.47))

s, bd = compute_emergency_score(_mk(label="alarm", conf=0.0))
check("alarm conf=0 → sound_c=0", s, bd, score_eq=0.0)

s, bd = compute_emergency_score(_mk(label="unknown_label", conf=1.0))
check("알 수 없는 라벨 → 기본 0.10 적용", s, bd, sound=(0.08, 0.12))


# ── M4: 음성/키워드 경계값 ──────────────────────────────────────────────────
print("[6] M4 음성/키워드 경계값")

s, bd = compute_emergency_score(_mk(speech_detected=True, stt_conf=0.0))
check("speech_detected only (no keyword) → speech=0.15", s, bd, speech=(0.13, 0.17))

kw = list(_CRITICAL_KEYWORDS)[:1]
s, bd = compute_emergency_score(_mk(keywords=kw, stt_conf=1.0))
check("키워드 1개 + stt_conf=1.0", s, bd, speech=(0.68, 0.72))

kw = list(_CRITICAL_KEYWORDS)[:2]
s, bd = compute_emergency_score(_mk(keywords=kw, stt_conf=1.0))
check("키워드 2개 + stt_conf=1.0", s, bd, speech=(0.88, 0.92))

# stt_conf=0 → 최소 0.3으로 보정
kw = list(_CRITICAL_KEYWORDS)[:1]
s, bd = compute_emergency_score(_mk(keywords=kw, stt_conf=0.0))
check("키워드 1개 + stt_conf=0 → 0.3 보정", s, bd, speech=(0.20, 0.23))


# ── 복합 위험 보정(×1.2) ────────────────────────────────────────────────────
print("[7] 복합 위험 보정(×1.2)")

# fall 고, vital 고 → 2 도메인 ≥ 0.5 → 보정 적용
s_combined, bd = compute_emergency_score(_mk(fall=1.0, hr=_HR_CRIT_LO))
import math
raw = 0.40 * 1.0 + 0.30 * 1.0   # fall + vital (sound=0, speech=0)
expected_with_boost = min(1.0, raw * 1.2)
check("fall=1.0 + HR=critical → 복합 보정 적용", s_combined, bd,
      score_lo=expected_with_boost - 0.02, score_hi=expected_with_boost + 0.02)

# 1개 도메인만 ≥ 0.5 → 보정 미적용
s_single, bd = compute_emergency_score(_mk(fall=1.0))
raw_single = 0.40 * 1.0
check("fall만 1.0 → 보정 없음", s_single, bd,
      score_lo=raw_single - 0.01, score_hi=raw_single + 0.01)

assert s_combined > s_single, "복합 보정 점수가 단독 점수보다 커야 함"
PASS += 1
if VERBOSE:
    print(f"  PASS  복합 > 단독  ({s_combined:.4f} > {s_single:.4f})")


# ── 점수 범위 클램프 ─────────────────────────────────────────────────────────
print("[8] 점수 범위 클램프")

s, bd = compute_emergency_score(_mk(
    fall=1.0,
    hr=_HR_CRIT_LO,
    rr=_RR_CRIT_HI,
    label="alarm", conf=1.0,
    keywords=list(_CRITICAL_KEYWORDS)[:3], stt_conf=1.0,
))
check("모든 도메인 max → score ≤ 1.0", s, bd, score_lo=0.9, score_hi=1.0)
assert 0.0 <= s <= 1.0, f"클램프 실패: {s}"
PASS += 1
if VERBOSE:
    print(f"  PASS  클램프 score={s:.4f}")


# ── infer_confidence 가중치(_conf_weight) ────────────────────────────────────
print("[9] infer_confidence 가중치 적용")

def _mk_conf(fall=0.0, fall_conf=1.0, hr=0.0, vital_conf=1.0,
             label="silence", lconf=0.0, sound_conf=1.0,
             keywords=None, stt_conf=0.0, speech_conf=1.0):
    return {
        "fall":      {"fall_score": fall, "infer_confidence": fall_conf},
        "vital":     {"heart_rate": hr, "breathing_rate": 0.0, "infer_confidence": vital_conf},
        "env_sound": {"env_sound_label": label, "env_sound_confidence": lconf, "infer_confidence": sound_conf},
        "speech_ko": {"keywords": keywords or [], "stt_confidence": stt_conf, "speech_detected": False, "infer_confidence": speech_conf},
    }

# conf_weight(1.0) = 1.0 → 기존과 동일
s_full, _ = compute_emergency_score(_mk_conf(fall=1.0, fall_conf=1.0))
check("fall=1.0 conf=1.0 → _conf_weight(1.0)=1.0 보정 없음", s_full, _, score_lo=0.38, score_hi=0.42)

# conf_weight(0.0) = 0.5 → 점수 절반 감쇠
s_zero, _ = compute_emergency_score(_mk_conf(fall=1.0, fall_conf=0.0))
check("fall=1.0 conf=0.0 → _conf_weight(0.0)=0.5 → fall_c=0.5", s_zero, _, score_lo=0.18, score_hi=0.22)
assert s_zero < s_full, f"저신뢰도 점수({s_zero:.4f})가 고신뢰도({s_full:.4f})보다 낮아야 함"
PASS += 1
if VERBOSE:
    print(f"  PASS  저신뢰도<고신뢰도  ({s_zero:.4f} < {s_full:.4f})")

# breakdown에 conf_* 플랫 키 포함 확인
s_bd, bd_conf = compute_emergency_score(_mk_conf(fall=0.8, fall_conf=0.75, vital_conf=0.70))
for domain in ("fall", "vital", "sound", "speech"):
    assert f"conf_{domain}" in bd_conf, f"conf_{domain} 키 없음"
PASS += 1
if VERBOSE:
    print(f"  PASS  breakdown conf_* 플랫 키 확인: { {k: bd_conf[k] for k in bd_conf if k.startswith('conf_')} }")

# no-audio (infer_confidence=0.0) → _conf_weight(0.0)=0.5 → 점수 절반으로 감쇠 (완전 0 아님)
# alarm(0.9) * env_conf(1.0) * _conf_weight(0.0)=0.5 = 0.45 → score = 0.45*0.15 = 0.0675
s_noaudio, _ = compute_emergency_score(_mk_conf(label="alarm", lconf=1.0, sound_conf=0.0))
check("sound alarm infer_conf=0.0 → _conf_weight=0.5 → 절반 감쇠", s_noaudio, _, score_lo=0.06, score_hi=0.08)


# ── D1/D2/D3 결함 수정 회귀 ─────────────────────────────────────────────────
print("[10] D1/D2/D3 결함 수정 회귀")

# D1: crit_lo 40/5 — HR=36·RR=5 직상값이 이제 crit→vital_bypass (이전 경고 0.165 미탐)
s, bd = compute_emergency_score(_mk(hr=36))
check("D1 HR=36 → crit(40 이하) → vital_bypass score≥0.6", s, bd, score_lo=0.63, score_hi=0.67)
s, bd = compute_emergency_score(_mk(rr=5))
check("D1 RR=5 → crit(5 이하) → vital_bypass score≥0.6", s, bd, score_lo=0.63, score_hi=0.67)
# D1 경계 반대편: HR=41은 여전히 경고(정상 아님, 과승급 아님)
s, bd = compute_emergency_score(_mk(hr=41))
check("D1 HR=41 → 경고(0.55) 유지, bypass 아님", s, bd, score_lo=0.14, score_hi=0.18)

# D2: 확정 낙상(≥0.8) + 경보/충격음(conf≥0.8) → conf 감쇠 무관 fall_hazard_bypass
s, bd = compute_emergency_score(_mk(fall=1.0, label="alarm", conf=1.0))
check("D2 fall=1.0+alarm → fall_hazard_bypass score≥0.6", s, bd, score_lo=0.60, score_hi=1.0)
if bd.get("fall_hazard_bypass"):
    PASS += 1
    VERBOSE and print("  PASS  D2 fall_hazard_bypass 플래그 set")
else:
    FAIL += 1
    print("  FAIL  D2 fall_hazard_bypass 플래그 미set")
# D2 비대상: 낙상만(보강 음향 없음)은 에스컬레이션 안 됨
s, bd = compute_emergency_score(_mk(fall=1.0, label="silence"))
check("D2 fall=1.0 단독(음향 없음) → 0.40, bypass 아님", s, bd, score_lo=0.38, score_hi=0.42)

# D3: composite 부스트는 단일 피크 ≥0.70 필요 — 전부 중등도면 미발동
# fall=0.6(0.6)+RR=24(0.55)+impact(0.65): 피크 0.65 < 0.70 → 부스트 없음 → 0.6 미만
s, bd = compute_emergency_score(_mk(fall=0.6, rr=24, label="impact", conf=1.0))
check("D3 중등도 3도메인(피크<0.70) → 부스트 미발동, score<0.6", s, bd, score_lo=0.0, score_hi=0.59)
# D3 대조: 피크 ≥0.70(alarm 0.9) 있으면 부스트 정상 발동
s, bd = compute_emergency_score(_mk(fall=0.6, hr=50, label="alarm", conf=1.0))
check("D3 피크≥0.70(alarm) 포함 → composite 부스트 발동 score≥0.6", s, bd, score_lo=0.60, score_hi=1.0)


# ── 결과 ────────────────────────────────────────────────────────────────────
print()
total = PASS + FAIL
print(f"결과: {PASS}/{total} PASS  {'OK' if FAIL == 0 else f'FAIL {FAIL}개'}")
if FAIL > 0:
    sys.exit(1)
