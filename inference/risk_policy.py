"""판정표(rubric v2) — rp5 `ai/logic/risk_policy.py`의 rubric_level 부분 동기화.

하니스 Track R(판정표 준수율) 정답을 만든다. rp5와 같은 결과가 나와야 하므로 임계·키워드·
분기 순서를 바꾸지 않는다. (rp5 원본: ea19c97·4fb7407)

주의: 정답이 게이트 점수에서 출발하므로 Track R은 '판정 정확도'가 아니라
'판정표 준수율'이다. 독립 오라클 기반 Track A/B와 섞어 해석하지 않는다.
"""

from __future__ import annotations

from inference.emergency_score import _HR_CRIT_HI, _HR_CRIT_LO, _RR_CRIT_HI, _RR_CRIT_LO
from inference.utils import safe_float

WARNING_THRESHOLD = 0.6
CRITICAL_THRESHOLD = 0.85
# rp5 판정표 키워드. 게이트의 _CRITICAL_KEYWORDS(아파·넘어 포함)와 다르다 — rp5와 동일하게 유지.
EMERGENCY_KEYWORDS = ("살려", "도와", "응급", "위험", "119", "불", "화재")
HAZARD_SOUNDS = ("alarm", "impact")


def voice_emergency_text(speech: dict | None) -> str:
    speech = speech or {}
    heard = str(speech.get("transcript_ko", "") or "").strip()
    return (f"긴급 음성 '{heard}'(≈{speech.get('emergency_phrase', '?')}, "
            f"유사도 {safe_float(speech.get('emergency_phrase_sim'), 0.0):.2f})")


def has_emergency_keyword(speech: dict | None) -> bool:
    speech = speech or {}
    transcript = str(speech.get("transcript_ko", "") or "")
    return any(k in transcript for k in EMERGENCY_KEYWORDS) or \
        any(k in EMERGENCY_KEYWORDS for k in (speech.get("keywords") or []))


def rubric_level(expert_results: dict | None, gate_score: float,
                 gate_breakdown: dict | None = None) -> tuple[str, str]:
    """판정표: 게이트가 M5를 부른 경우 최종 등급의 하한과 그 근거 문장.

    critical ① 위기 생체신호 + (낙상 확정·위험음·긴급키워드)
             ② 낙상 확정 + (위험음·긴급키워드)
             ③ 게이트 점수가 이미 critical
             ④ M4 긴급 음성(환각 필터 통과 + 긴급 문장 유사 매칭)
    warning  그 밖에 게이트 >= 0.6 (M5 호출 구간)
    normal   게이트 < 0.6 (운영에서는 M5를 부르지 않는 구간)
    """
    gate = safe_float(gate_score, default=0.0)
    if gate < WARNING_THRESHOLD:
        return "normal", ""
    er = expert_results or {}
    vital = er.get("vital") or {}
    hr = safe_float(vital.get("heart_rate"), default=0.0)
    rr = safe_float(vital.get("breathing_rate"), default=0.0)
    crisis = []
    if 0 < hr <= _HR_CRIT_LO or hr >= _HR_CRIT_HI:
        crisis.append(f"심박위기(hr={hr:.0f})")
    if 0 < rr <= _RR_CRIT_LO or rr >= _RR_CRIT_HI:
        crisis.append(f"호흡위기(rr={rr:.0f})")
    fall = ["낙상감지"] if (er.get("fall") or {}).get("fall_detected") else []
    sound = er.get("env_sound") or {}
    label = str(sound.get("env_sound_label") or sound.get("label") or "")
    hazard = [f"위험음({label})"] if label in HAZARD_SOUNDS else []
    keyword = ["긴급키워드"] if has_emergency_keyword(er.get("speech_ko")) else []

    if gate >= CRITICAL_THRESHOLD:
        return "critical", f"판정표③ 게이트 점수 {gate:.2f}"
    if crisis and (fall or hazard or keyword):
        return "critical", "판정표① " + "+".join(crisis + fall + hazard + keyword)
    if fall and (hazard or keyword):
        return "critical", "판정표② " + "+".join(fall + hazard + keyword)
    if (er.get("speech_ko") or {}).get("emergency_phrase_detected"):
        return "critical", "판정표④ " + voice_emergency_text(er.get("speech_ko"))
    basis = crisis + fall
    if (gate_breakdown or {}).get("temporal_escalation"):
        basis.append("시계열 악화")
    return "warning", "판정표 warning " + ("+".join(basis) if basis else f"게이트 점수 {gate:.2f}")
