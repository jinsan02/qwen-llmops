"""
M1-M4 전문가 출력 기반 응급지수 계산.

SLM(M5) 호출 여부를 결정하는 경량 알고리즘.
SLM은 임계값(threshold) 초과 시에만 호출된다.
"""

import numpy as np

# M2 생체신호 정상 범위
# D1: crit_lo를 40/5로 상향 — HR=36·RR=5 같은 '직상' 심한 서맥/서호흡이
#     경고(0.55)로만 처리돼 미탐되던 결함 해소 (crit→vital_bypass 에스컬레이션)
_HR_WARN_LO, _HR_WARN_HI = 55.0, 100.0    # BPM
_HR_CRIT_LO, _HR_CRIT_HI = 40.0, 130.0
_RR_WARN_LO, _RR_WARN_HI = 10.0, 22.0    # 분당 호흡수
_RR_CRIT_LO, _RR_CRIT_HI =  5.0,  35.0

# M3 환경음 → 위험 가중치
_SOUND_WEIGHTS = {
    "alarm":   0.90,
    "impact":  0.65,
    "noise":   0.20,
    "speech":  0.10,
    "music":   0.05,
    "silence": 0.00,
    "unknown": 0.10,
}

# M4 응급 키워드 목록
_CRITICAL_KEYWORDS = frozenset(["살려", "도와", "아파", "응급", "위험", "넘어", "불", "화재", "119"])

# 도메인 가중치 (합 = 1.0)
_DOMAIN_WEIGHTS = {"fall": 0.40, "vital": 0.30, "sound": 0.15, "speech": 0.15}

# 복합 위험 보정 배율 (활성 도메인 수 기준)
_COMPOSITE_BOOST = {2: 1.20, 3: 1.35, 4: 1.50}
# D3: 부스트 발동에 최소 단일 도메인 피크 요구 — 전부 중등도(예: fall0.6+vital경고+impact)인데
#     ×1.35로 과승급해 오탐(multi-12)되던 결함 해소
_COMPOSITE_MIN_PEAK = 0.70
# B(개선): 2도메인 부스트는 더 강한 피크(near-확정) 요구 — 중등도 2신호 과승급 차단
_COMPOSITE_MIN_PEAK_2DOM = 0.90

# M2 생체신호 극한값 단일 에스컬레이션: vital_component==1.0이면 score 최솟값
_VITAL_CRIT_BYPASS = 0.65

# D2: 확정 낙상 + 경보/충격음 동시 → infer_confidence 감쇠와 무관하게 에스컬레이션
#     (낙상센서·음향이 저신뢰로 깎여 보강된 복합응급이 0.6 직하로 미탐되던 결함 해소)
_FALL_HAZARD_BYPASS = 0.65
_FALL_CONFIRM_MIN = 0.80
_HAZARD_SOUND_CONF_MIN = 0.80

# M4 긴급 키워드 + M1 낙상 의심 동시 발생 시 score 보너스
_KEYWORD_FALL_BONUS = 0.15
_KEYWORD_FALL_SCORE_MIN = 0.25

# ── 시계열(1h, ≤60행) 에스컬레이션 ──────────────────────────────────────────
# 단발 스냅샷이 놓치는 '지속 경고 누적·점진 악화' 궤적을 M5 임계(0.6)로 올린다.
# time_series=None이면 전혀 관여하지 않음(스냅샷 전용 — 하위호환).
_TS_MIN_ROWS = 10                # 이보다 짧으면 시계열 신호 무시(sparse → 스냅샷 govern)
_TS_RECENT_WINDOW = 20           # 최근 N분 윈도우
_TS_SUSTAINED_RATIO = 0.60       # 최근 윈도우 warn-or-worse 비율 임계 → 지속 경고
_TS_HR_RISE = 30.0               # HR 상승 추세(빈맥 악화) 임계
_TS_HR_END_HI = 100.0
_TS_HR_FALL = 25.0               # HR 하강 추세(서맥 악화) 임계
_TS_HR_END_LO = 58.0
_TS_RR_RISE = 10.0               # RR 상승 추세(빈호흡 악화) 임계
_TS_RR_END_HI = 22.0
_TS_FLOOR = 0.60                 # 에스컬레이션 시 score 최솟값(M5 호출)


def _ts_warn_or_worse(hr: float, rr: float) -> bool:
    """단일 행이 정상 범위를 벗어났는가(warn 이상)."""
    return (hr > 0 and (hr <= _HR_WARN_LO or hr >= _HR_WARN_HI)) or \
           (rr > 0 and (rr <= _RR_WARN_LO or rr >= _RR_WARN_HI))


def _ts_quartile_means(vals: list) -> tuple:
    """시작 1/4 평균, 최근 1/4 평균 (추세 방향 산정)."""
    if len(vals) < 4:
        m = sum(vals) / len(vals) if vals else 0.0
        return m, m
    q = max(1, len(vals) // 4)
    return sum(vals[:q]) / q, sum(vals[-q:]) / q


def _temporal_escalation(time_series) -> tuple:
    """시계열에서 지속 경고/점진 악화를 감지해 (floor, tags)를 반환.
    time_series: [{m, hr, rr, ...}, ...] (m 오름차순, 마지막이 현재). 신호 없으면 (0.0, [])."""
    if not time_series or len(time_series) < _TS_MIN_ROWS:
        return 0.0, []
    rows = list(time_series)
    hrs = [float(r.get("hr", 0) or 0) for r in rows]
    rrs = [float(r.get("rr", 0) or 0) for r in rows]

    recent = rows[-_TS_RECENT_WINDOW:]
    sustained = sum(1 for r in recent
                    if _ts_warn_or_worse(float(r.get("hr", 0) or 0), float(r.get("rr", 0) or 0)))
    sustained_ratio = sustained / len(recent)

    hr_start, hr_end = _ts_quartile_means([h for h in hrs if h > 0] or [0.0])
    rr_start, rr_end = _ts_quartile_means([r for r in rrs if r > 0] or [0.0])
    hr_trend = hr_end - hr_start
    rr_trend = rr_end - rr_start

    tags = []
    if sustained_ratio >= _TS_SUSTAINED_RATIO:
        tags.append("sustained_warn")
    if hr_trend >= _TS_HR_RISE and hr_end >= _TS_HR_END_HI:
        tags.append("hr_rising")
    if hr_trend <= -_TS_HR_FALL and 0 < hr_end <= _TS_HR_END_LO:
        tags.append("hr_falling")
    if rr_trend >= _TS_RR_RISE and rr_end >= _TS_RR_END_HI:
        tags.append("rr_rising")

    return (_TS_FLOOR, tags) if tags else (0.0, [])


def _conf_weight(confidence: float) -> float:
    """추론 신뢰도(0~1) → 점수 가중치 [0.5, 1.0]. 낮은 신뢰도 시 중립 방향으로 감쇠."""
    return 0.5 + 0.5 * float(np.clip(confidence, 0.0, 1.0))


def _vital_component(val: float, crit_lo: float, warn_lo: float, warn_hi: float, crit_hi: float) -> float:
    """단일 생체신호 값의 이상 점수를 반환한다."""
    if val <= 0.0:
        return 0.0
    if val <= crit_lo or val >= crit_hi:
        return 1.0
    if val <= warn_lo or val >= warn_hi:
        return 0.55
    return 0.0


def compute_emergency_score(expert_results: dict, time_series=None) -> tuple[float, dict]:
    """
    M1-M4 출력에서 응급지수(0.0-1.0)를 계산한다.

    time_series(선택): [{m, hr, rr, ...}, ...] 1h(≤60행) 시계열. 주어지면 스냅샷 점수에
      지속 경고·점진 악화 에스컬레이션을 적용(M5 임계 0.6으로 floor). None이면 스냅샷 전용
      (기존과 100% 동일 — 하위호환).

    도메인 가중치: fall 40% + vital 30% + sound 15% + speech 15%
    복합 위험 보정(활성 도메인 수 비례):
      2도메인 >=0.5 -> x1.20 / 3도메인 -> x1.35 / 4도메인 -> x1.50 (상한 1.0)
    M2 생체신호 극한값 단일 에스컬레이션:
      vital_component==1.0 (심박/호흡 위기 범위) -> score 최솟값 0.65
    M4 긴급 키워드 + M1 낙상 의심 동시 발생:
      keyword>=1 & fall_score>=0.25 -> score +0.15

    Args:
        expert_results: M1-M4 전문가 모델 출력 dict

    Returns:
        (score, breakdown)
          - score: float 0.0~1.0 응급지수
          - breakdown: {"fall", "vital", "sound", "speech", "conf_fall", ...}
    """
    fall_out   = expert_results.get("fall")      or {}
    vital_out  = expert_results.get("vital")     or {}
    sound_out  = expert_results.get("env_sound") or {}
    speech_out = expert_results.get("speech_ko") or {}

    # ── M1: 낙상 점수 ────────────────────────────────────────────
    fall_conf = float(np.clip(fall_out.get("infer_confidence", 1.0), 0.0, 1.0))
    fall_c = float(np.clip(fall_out.get("fall_score", 0.0), 0.0, 1.0)) * _conf_weight(fall_conf)

    # ── M2: 생체신호 이상 점수 ────────────────────────────────────
    vital_conf = float(np.clip(vital_out.get("infer_confidence", 1.0), 0.0, 1.0))
    hr = float(vital_out.get("heart_rate", 0.0))
    rr = float(vital_out.get("breathing_rate", 0.0))
    _raw_vital_comp = max(
        _vital_component(hr, _HR_CRIT_LO, _HR_WARN_LO, _HR_WARN_HI, _HR_CRIT_HI),
        _vital_component(rr, _RR_CRIT_LO, _RR_WARN_LO, _RR_WARN_HI, _RR_CRIT_HI),
    )
    vital_c = _raw_vital_comp * _conf_weight(vital_conf)

    # ── M3: 환경음 위험 점수 ──────────────────────────────────────
    sound_conf = float(np.clip(sound_out.get("infer_confidence", 1.0), 0.0, 1.0))
    label = str(sound_out.get("label") or sound_out.get("env_sound_label") or "unknown")
    conf  = float(sound_out.get("confidence") or sound_out.get("env_sound_confidence") or 0.0)
    sound_c = float(np.clip(_SOUND_WEIGHTS.get(label, 0.10) * conf, 0.0, 1.0)) * _conf_weight(sound_conf)

    # ── M4: 음성 응급 키워드 점수 ────────────────────────────────
    speech_conf = float(np.clip(speech_out.get("infer_confidence", 1.0), 0.0, 1.0))
    keywords = list(speech_out.get("keywords") or [])
    stt_conf = float(speech_out.get("stt_confidence", 0.0))
    kw_hits  = len(_CRITICAL_KEYWORDS.intersection(keywords))
    if kw_hits > 0:
        # 키워드 1개당 +0.2, STT 신뢰도 낮아도 최소 0.3으로 보정
        speech_c = float(np.clip(0.5 + kw_hits * 0.2, 0.0, 0.9)) * max(stt_conf, 0.3)
    elif speech_out.get("speech_detected"):
        speech_c = 0.15
    else:
        speech_c = 0.0
    speech_c = speech_c * _conf_weight(speech_conf)

    breakdown = {
        "fall":        round(fall_c,   4),
        "vital":       round(vital_c,  4),
        "sound":       round(sound_c,  4),
        "speech":      round(float(speech_c), 4),
        "conf_fall":   round(fall_conf,   3),
        "conf_vital":  round(vital_conf,  3),
        "conf_sound":  round(sound_conf,  3),
        "conf_speech": round(speech_conf, 3),
    }

    score = sum(_DOMAIN_WEIGHTS[k] * breakdown[k] for k in _DOMAIN_WEIGHTS)

    # ── 복합 위험 보정: 활성 도메인 수에 비례한 차등 배율 ──────────────
    # D3: 활성 도메인 ≥2 AND 최소 한 도메인이 피크일 때만 발동.
    # B(개선): 2도메인 부스트는 near-확정 피크(≥0.90)를 요구 — 중등도 2신호(예: fall0.89+서맥)
    #     가 ×1.20으로 0.6을 넘는 과승급 오탐 방지. 3+ 도메인은 0.70 유지(다신호 복합응급).
    _domain_scores = (breakdown["fall"], breakdown["vital"], breakdown["sound"], breakdown["speech"])
    _active = sum(1 for v in _domain_scores if v >= 0.5)
    _min_peak = _COMPOSITE_MIN_PEAK_2DOM if _active == 2 else _COMPOSITE_MIN_PEAK
    if _active >= 2 and max(_domain_scores) >= _min_peak:
        score = min(1.0, score * _COMPOSITE_BOOST.get(_active, 1.50))

    # ── M4 긴급 키워드 + M1 낙상 의심 동시 발생 보너스 ─────────────────
    _fall_score_raw = float(fall_out.get("fall_score", 0.0))
    if kw_hits >= 1 and _fall_score_raw >= _KEYWORD_FALL_SCORE_MIN:
        score = min(1.0, score + _KEYWORD_FALL_BONUS)
        breakdown["keyword_fall_bonus"] = True

    # ── M2 생체신호 극한값 단일 에스컬레이션 ────────────────────────────
    # vital_component==1.0: 심박/호흡이 위기 범위 → score 최솟값 0.65 보장
    if _raw_vital_comp >= 1.0:
        score = max(score, _VITAL_CRIT_BYPASS)
        breakdown["vital_bypass"] = True

    # ── D2: 확정 낙상 + 경보/충격음 동시 에스컬레이션 (raw 기반, conf 감쇠 무관) ──
    _snd_conf_raw = float(sound_out.get("env_sound_confidence") or sound_out.get("confidence") or 0.0)
    if (_fall_score_raw >= _FALL_CONFIRM_MIN
            and label in ("alarm", "impact")
            and _snd_conf_raw >= _HAZARD_SOUND_CONF_MIN):
        score = max(score, _FALL_HAZARD_BYPASS)
        breakdown["fall_hazard_bypass"] = True

    # ── 시계열 에스컬레이션: 지속 경고/점진 악화 궤적 → M5 임계로 floor ──────────
    if time_series:
        ts_floor, ts_tags = _temporal_escalation(time_series)
        if ts_floor > 0.0 and score < ts_floor:
            score = ts_floor
            breakdown["temporal_escalation"] = ts_tags

    return float(np.clip(score, 0.0, 1.0)), breakdown
