"""
eval/eval_qwen_accuracy.py

Qwen-0.5B 프롬프트/few-shot 정확도 평가 도구.

Ground truth: compute_emergency_score() -> risk_level
  emg_score < 0.60        -> "normal"
  0.60 <= score < 0.85    -> "warning"
  score >= 0.85           -> "critical"
  (vital_bypass 포함 -- emg_score에 이미 반영)

사용법:
  python eval/eval_qwen_accuracy.py            # 전체 100케이스 Qwen 추론 + 채점
  python eval/eval_qwen_accuracy.py --gen      # 데이터셋 재생성만 (Qwen 미호출)
  python eval/eval_qwen_accuracy.py --id D-01  # 단일 케이스 디버그
  python eval/eval_qwen_accuracy.py --cat vital_crisis_solo  # 카테고리별
"""

import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.emergency_score import compute_emergency_score

DATASET_PATH = os.path.join(os.path.dirname(__file__), "qwen_eval_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "qwen_eval_results.json")

SLM_MODEL_PATH = os.getenv(
    "SLM_MODEL",
    os.path.join(os.path.dirname(__file__), "..", "volumes", "models", "qwen_05b"),
)

# ─────────────────────────────────────────────────────────────────────────────
# 100 케이스 원형 정의
# 필드: id, cat, scenario, hr, rr, fall, fall_det, env, env_conf, tx, kw
#   + 선택: vital_conf(0.70), fall_conf(0.75), speech_conf(0.55)
# ─────────────────────────────────────────────────────────────────────────────

_CASE_DEFS = [
    # A: pure_normal (20)
    {"id":"A-01","cat":"pure_normal","scenario":"완전 정상 기본","hr":72,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.95,"tx":"","kw":[]},
    {"id":"A-02","cat":"pure_normal","scenario":"활동중 정상 HR90","hr":90,"rr":18,"fall":0.05,"fall_det":False,"env":"noise","env_conf":0.60,"tx":"","kw":[]},
    {"id":"A-03","cat":"pure_normal","scenario":"취침 정상 HR62","hr":62,"rr":13,"fall":0.01,"fall_det":False,"env":"silence","env_conf":0.95,"tx":"","kw":[]},
    {"id":"A-04","cat":"pure_normal","scenario":"TV 시청 정상","hr":75,"rr":16,"fall":0.04,"fall_det":False,"env":"speech","env_conf":0.80,"tx":"뉴스 나와","kw":[]},
    {"id":"A-05","cat":"pure_normal","scenario":"음악 정상","hr":78,"rr":14,"fall":0.03,"fall_det":False,"env":"music","env_conf":0.70,"tx":"","kw":[]},
    {"id":"A-06","cat":"pure_normal","scenario":"fall 신뢰도 낮음 정상","hr":70,"rr":15,"fall":0.10,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[],"fall_conf":0.40},
    {"id":"A-07","cat":"pure_normal","scenario":"전 신뢰도 낮음 정상","hr":80,"rr":16,"fall":0.05,"fall_det":False,"env":"silence","env_conf":0.50,"tx":"","kw":[],"vital_conf":0.50,"fall_conf":0.50},
    {"id":"A-08","cat":"pure_normal","scenario":"HR95 경계 아래","hr":95,"rr":20,"fall":0.08,"fall_det":False,"env":"silence","env_conf":0.95,"tx":"","kw":[]},
    {"id":"A-09","cat":"pure_normal","scenario":"RR21 경계 아래","hr":72,"rr":21,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.95,"tx":"","kw":[]},
    {"id":"A-10","cat":"pure_normal","scenario":"일반 대화 비응급","hr":75,"rr":15,"fall":0.03,"fall_det":False,"env":"speech","env_conf":0.80,"tx":"배고파요","kw":[]},
    {"id":"A-11","cat":"pure_normal","scenario":"fall 0.15 정상 범주","hr":68,"rr":14,"fall":0.15,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"A-12","cat":"pure_normal","scenario":"HR65 정상 하한","hr":65,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.95,"tx":"","kw":[]},
    {"id":"A-13","cat":"pure_normal","scenario":"RR12 정상 하한","hr":72,"rr":12,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.95,"tx":"","kw":[]},
    {"id":"A-14","cat":"pure_normal","scenario":"noise 낮은 신호","hr":80,"rr":16,"fall":0.06,"fall_det":False,"env":"noise","env_conf":0.30,"tx":"","kw":[]},
    {"id":"A-15","cat":"pure_normal","scenario":"기상 직후 HR98","hr":98,"rr":19,"fall":0.07,"fall_det":False,"env":"silence","env_conf":0.85,"tx":"","kw":[]},
    {"id":"A-16","cat":"pure_normal","scenario":"fall 0.20 경계 정상","hr":72,"rr":15,"fall":0.20,"fall_det":False,"env":"silence","env_conf":0.95,"tx":"","kw":[]},
    {"id":"A-17","cat":"pure_normal","scenario":"speech 일반 대화","hr":72,"rr":15,"fall":0.03,"fall_det":False,"env":"speech","env_conf":0.70,"tx":"오늘 날씨","kw":[]},
    {"id":"A-18","cat":"pure_normal","scenario":"모든 신뢰도 최고","hr":72,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":1.00,"tx":"","kw":[],"vital_conf":1.00,"fall_conf":1.00},
    {"id":"A-19","cat":"pure_normal","scenario":"HR56 warn 경계 바로 위","hr":56,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.95,"tx":"","kw":[]},
    {"id":"A-20","cat":"pure_normal","scenario":"RR11 warn 경계 바로 위","hr":72,"rr":11,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.95,"tx":"","kw":[]},

    # B: fall_only (10)
    {"id":"B-01","cat":"fall_only","scenario":"낙상 위험 50%","hr":72,"rr":15,"fall":0.50,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"B-02","cat":"fall_only","scenario":"낙상 위험 75%","hr":72,"rr":15,"fall":0.75,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"B-03","cat":"fall_only","scenario":"낙상 감지 True","hr":72,"rr":15,"fall":0.90,"fall_det":True,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"B-04","cat":"fall_only","scenario":"낙상 최고 99%","hr":72,"rr":15,"fall":0.99,"fall_det":True,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"B-05","cat":"fall_only","scenario":"낙상 60% 신뢰도 높음","hr":72,"rr":15,"fall":0.60,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[],"fall_conf":0.90},
    {"id":"B-06","cat":"fall_only","scenario":"낙상 감지 신뢰도 낮음","hr":72,"rr":15,"fall":0.85,"fall_det":True,"env":"silence","env_conf":0.90,"tx":"","kw":[],"fall_conf":0.40},
    {"id":"B-07","cat":"fall_only","scenario":"낙상 70% RR 약간 높음","hr":80,"rr":20,"fall":0.70,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"B-08","cat":"fall_only","scenario":"낙상 30% 저위험","hr":72,"rr":15,"fall":0.30,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"B-09","cat":"fall_only","scenario":"낙상 40% 저위험","hr":72,"rr":15,"fall":0.40,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"B-10","cat":"fall_only","scenario":"낙상 감지 noise 배경","hr":78,"rr":16,"fall":0.95,"fall_det":True,"env":"noise","env_conf":0.40,"tx":"","kw":[]},

    # C: vital_warn_only (8)
    {"id":"C-01","cat":"vital_warn_only","scenario":"HR54 경고","hr":54,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"C-02","cat":"vital_warn_only","scenario":"HR108 경고","hr":108,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"C-03","cat":"vital_warn_only","scenario":"RR9 경고","hr":72,"rr":9,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"C-04","cat":"vital_warn_only","scenario":"RR24 경고","hr":72,"rr":24,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"C-05","cat":"vital_warn_only","scenario":"HR50 신뢰도 낮음","hr":50,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[],"vital_conf":0.50},
    {"id":"C-06","cat":"vital_warn_only","scenario":"HR120 fall 소량","hr":120,"rr":18,"fall":0.15,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"C-07","cat":"vital_warn_only","scenario":"RR8 경고","hr":75,"rr":8,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"C-08","cat":"vital_warn_only","scenario":"HR110+RR23 모두 경고","hr":110,"rr":23,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},

    # D: vital_crisis_solo (12)
    {"id":"D-01","cat":"vital_crisis_solo","scenario":"HR33 극한 서맥","hr":33,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"D-02","cat":"vital_crisis_solo","scenario":"HR20 극한 서맥","hr":20,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"D-03","cat":"vital_crisis_solo","scenario":"HR140 극한 빈맥","hr":140,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"D-04","cat":"vital_crisis_solo","scenario":"HR160 극한 빈맥","hr":160,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"D-05","cat":"vital_crisis_solo","scenario":"RR3 무호흡","hr":72,"rr":3,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"D-06","cat":"vital_crisis_solo","scenario":"RR2 심한 무호흡","hr":72,"rr":2,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"D-07","cat":"vital_crisis_solo","scenario":"RR40 빈호흡","hr":72,"rr":40,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"D-08","cat":"vital_crisis_solo","scenario":"HR35 경계(위기)","hr":35,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"D-09","cat":"vital_crisis_solo","scenario":"HR130 경계(위기)","hr":130,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"D-10","cat":"vital_crisis_solo","scenario":"RR4 경계(위기)","hr":72,"rr":4,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"D-11","cat":"vital_crisis_solo","scenario":"HR28+RR5 이중 위기","hr":28,"rr":5,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"D-12","cat":"vital_crisis_solo","scenario":"HR33 vital_bypass 낮은신뢰도","hr":33,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[],"vital_conf":0.50},

    # E: two_domain (12)
    {"id":"E-01","cat":"two_domain","scenario":"fall70+vital warn HR","hr":108,"rr":15,"fall":0.70,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"E-02","cat":"two_domain","scenario":"fall 감지+vital warn HR","hr":110,"rr":15,"fall":0.85,"fall_det":True,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"E-03","cat":"two_domain","scenario":"fall60+alarm","hr":72,"rr":15,"fall":0.60,"fall_det":False,"env":"alarm","env_conf":0.85,"tx":"","kw":[]},
    {"id":"E-04","cat":"two_domain","scenario":"fall 감지+impact","hr":72,"rr":15,"fall":0.90,"fall_det":True,"env":"impact","env_conf":0.80,"tx":"","kw":[]},
    {"id":"E-05","cat":"two_domain","scenario":"vital warn+alarm","hr":115,"rr":15,"fall":0.02,"fall_det":False,"env":"alarm","env_conf":0.88,"tx":"","kw":[]},
    {"id":"E-06","cat":"two_domain","scenario":"vital warn+살려","hr":110,"rr":15,"fall":0.02,"fall_det":False,"env":"speech","env_conf":0.80,"tx":"살려주세요","kw":["살려"]},
    {"id":"E-07","cat":"two_domain","scenario":"fall65+살려","hr":72,"rr":15,"fall":0.65,"fall_det":False,"env":"speech","env_conf":0.80,"tx":"살려","kw":["살려"]},
    {"id":"E-08","cat":"two_domain","scenario":"fall55+RR warn","hr":72,"rr":8,"fall":0.55,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"E-09","cat":"two_domain","scenario":"fall 감지+impact","hr":72,"rr":15,"fall":0.80,"fall_det":True,"env":"impact","env_conf":0.75,"tx":"","kw":[]},
    {"id":"E-10","cat":"two_domain","scenario":"vital warn+noise","hr":108,"rr":15,"fall":0.02,"fall_det":False,"env":"noise","env_conf":0.90,"tx":"","kw":[]},
    {"id":"E-11","cat":"two_domain","scenario":"fall60+도와줘","hr":72,"rr":15,"fall":0.60,"fall_det":False,"env":"speech","env_conf":0.80,"tx":"도와줘","kw":["도와"]},
    {"id":"E-12","cat":"two_domain","scenario":"vital crisis+fall20 (bypass)","hr":33,"rr":15,"fall":0.20,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},

    # F: three_domain (8)
    {"id":"F-01","cat":"three_domain","scenario":"fall+vital warn+alarm","hr":108,"rr":15,"fall":0.80,"fall_det":True,"env":"alarm","env_conf":0.88,"tx":"","kw":[]},
    {"id":"F-02","cat":"three_domain","scenario":"fall+vital warn+살려","hr":110,"rr":15,"fall":0.75,"fall_det":True,"env":"speech","env_conf":0.80,"tx":"살려","kw":["살려"]},
    {"id":"F-03","cat":"three_domain","scenario":"fall+alarm+응급","hr":72,"rr":15,"fall":0.80,"fall_det":True,"env":"alarm","env_conf":0.88,"tx":"응급","kw":["응급"]},
    {"id":"F-04","cat":"three_domain","scenario":"vital crisis+fall+alarm","hr":33,"rr":15,"fall":0.60,"fall_det":False,"env":"alarm","env_conf":0.88,"tx":"","kw":[]},
    {"id":"F-05","cat":"three_domain","scenario":"vital crisis+fall+살려","hr":33,"rr":15,"fall":0.70,"fall_det":True,"env":"speech","env_conf":0.80,"tx":"살려","kw":["살려"]},
    {"id":"F-06","cat":"three_domain","scenario":"vital warn+alarm+위험","hr":115,"rr":15,"fall":0.02,"fall_det":False,"env":"alarm","env_conf":0.90,"tx":"위험해","kw":["위험"]},
    {"id":"F-07","cat":"three_domain","scenario":"fall+RR warn+alarm","hr":72,"rr":8,"fall":0.75,"fall_det":True,"env":"alarm","env_conf":0.85,"tx":"","kw":[]},
    {"id":"F-08","cat":"three_domain","scenario":"fall+impact+119","hr":72,"rr":15,"fall":0.85,"fall_det":True,"env":"impact","env_conf":0.80,"tx":"119","kw":["119"]},

    # G: four_domain (6)
    {"id":"G-01","cat":"four_domain","scenario":"4도메인 표준 — HR33 fall감지 alarm 살려","hr":33,"rr":5,"fall":0.95,"fall_det":True,"env":"alarm","env_conf":0.92,"tx":"살려 응급","kw":["살려","응급"]},
    {"id":"G-02","cat":"four_domain","scenario":"4도메인 HR 빈맥","hr":145,"rr":28,"fall":0.88,"fall_det":True,"env":"alarm","env_conf":0.88,"tx":"살려","kw":["살려"]},
    {"id":"G-03","cat":"four_domain","scenario":"4도메인 fall 최고","hr":33,"rr":5,"fall":0.99,"fall_det":True,"env":"alarm","env_conf":0.95,"tx":"위험","kw":["위험"]},
    {"id":"G-04","cat":"four_domain","scenario":"4도메인 신뢰도 낮음","hr":33,"rr":5,"fall":0.90,"fall_det":True,"env":"alarm","env_conf":0.80,"tx":"살려","kw":["살려"],"vital_conf":0.60,"fall_conf":0.65},
    {"id":"G-05","cat":"four_domain","scenario":"4도메인 화재","hr":140,"rr":35,"fall":0.92,"fall_det":True,"env":"alarm","env_conf":0.90,"tx":"불 화재","kw":["불","화재"]},
    {"id":"G-06","cat":"four_domain","scenario":"4도메인 무호흡+빈맥","hr":150,"rr":2,"fall":0.95,"fall_det":True,"env":"alarm","env_conf":0.92,"tx":"응급 119","kw":["응급","119"]},

    # H: keyword_fall_bonus (8)
    {"id":"H-01","cat":"keyword_fall","scenario":"살려+fall30 보너스","hr":72,"rr":15,"fall":0.30,"fall_det":False,"env":"speech","env_conf":0.80,"tx":"살려줘","kw":["살려"]},
    {"id":"H-02","cat":"keyword_fall","scenario":"도와+fall50 보너스","hr":72,"rr":15,"fall":0.50,"fall_det":False,"env":"speech","env_conf":0.80,"tx":"도와줘","kw":["도와"]},
    {"id":"H-03","cat":"keyword_fall","scenario":"응급+fall80 보너스","hr":72,"rr":15,"fall":0.80,"fall_det":True,"env":"speech","env_conf":0.80,"tx":"응급","kw":["응급"]},
    {"id":"H-04","cat":"keyword_fall","scenario":"키워드 단독 fall20 보너스없음","hr":72,"rr":15,"fall":0.20,"fall_det":False,"env":"speech","env_conf":0.80,"tx":"살려","kw":["살려"]},
    {"id":"H-05","cat":"keyword_fall","scenario":"119+fall60 보너스","hr":72,"rr":15,"fall":0.60,"fall_det":False,"env":"speech","env_conf":0.80,"tx":"119","kw":["119"]},
    {"id":"H-06","cat":"keyword_fall","scenario":"넘어+fall40 보너스","hr":72,"rr":15,"fall":0.40,"fall_det":False,"env":"speech","env_conf":0.80,"tx":"넘어졌어","kw":["넘어"]},
    {"id":"H-07","cat":"keyword_fall","scenario":"아파+fall30 보너스","hr":72,"rr":15,"fall":0.30,"fall_det":False,"env":"speech","env_conf":0.80,"tx":"너무 아파","kw":["아파"]},
    {"id":"H-08","cat":"keyword_fall","scenario":"키워드2+fall 감지","hr":72,"rr":15,"fall":0.90,"fall_det":True,"env":"speech","env_conf":0.80,"tx":"살려 응급","kw":["살려","응급"]},

    # I: alarm_sound (8)
    {"id":"I-01","cat":"alarm_sound","scenario":"alarm 단독 기타 정상","hr":72,"rr":15,"fall":0.02,"fall_det":False,"env":"alarm","env_conf":0.90,"tx":"","kw":[]},
    {"id":"I-02","cat":"alarm_sound","scenario":"alarm+fall 소량","hr":72,"rr":15,"fall":0.20,"fall_det":False,"env":"alarm","env_conf":0.85,"tx":"","kw":[]},
    {"id":"I-03","cat":"alarm_sound","scenario":"alarm 낮은 신뢰도","hr":72,"rr":15,"fall":0.02,"fall_det":False,"env":"alarm","env_conf":0.40,"tx":"","kw":[]},
    {"id":"I-04","cat":"alarm_sound","scenario":"impact+fall55","hr":72,"rr":15,"fall":0.55,"fall_det":False,"env":"impact","env_conf":0.80,"tx":"","kw":[]},
    {"id":"I-05","cat":"alarm_sound","scenario":"alarm+vital warn","hr":115,"rr":15,"fall":0.02,"fall_det":False,"env":"alarm","env_conf":0.88,"tx":"","kw":[]},
    {"id":"I-06","cat":"alarm_sound","scenario":"alarm+speech 일반","hr":72,"rr":15,"fall":0.02,"fall_det":False,"env":"alarm","env_conf":0.85,"tx":"안녕","kw":[]},
    {"id":"I-07","cat":"alarm_sound","scenario":"impact 단독","hr":72,"rr":15,"fall":0.02,"fall_det":False,"env":"impact","env_conf":0.85,"tx":"","kw":[]},
    {"id":"I-08","cat":"alarm_sound","scenario":"alarm+fall 감지","hr":72,"rr":15,"fall":0.92,"fall_det":True,"env":"alarm","env_conf":0.90,"tx":"","kw":[]},

    # J: edge/boundary (8)
    {"id":"J-01","cat":"edge","scenario":"HR36 위기 경계 바로 위(warn)","hr":36,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"J-02","cat":"edge","scenario":"HR34 위기 범위(bypass)","hr":34,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"J-03","cat":"edge","scenario":"RR5 위기 범위(bypass)","hr":72,"rr":5,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"J-04","cat":"edge","scenario":"RR36 위기 범위(bypass)","hr":72,"rr":36,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"J-05","cat":"edge","scenario":"HR100 정상 상한 정확히","hr":100,"rr":15,"fall":0.02,"fall_det":False,"env":"silence","env_conf":0.90,"tx":"","kw":[]},
    {"id":"J-06","cat":"edge","scenario":"전 신뢰도 0.5 중립","hr":33,"rr":5,"fall":0.95,"fall_det":True,"env":"alarm","env_conf":0.50,"tx":"살려","kw":["살려"],"vital_conf":0.50,"fall_conf":0.50,"speech_conf":0.50},
    {"id":"J-07","cat":"edge","scenario":"fall50+alarm 경계 복합","hr":72,"rr":15,"fall":0.50,"fall_det":False,"env":"alarm","env_conf":0.88,"tx":"","kw":[]},
    {"id":"J-08","cat":"edge","scenario":"vital crisis+alarm 복합 위기","hr":33,"rr":4,"fall":0.02,"fall_det":False,"env":"alarm","env_conf":0.90,"tx":"","kw":[]},
]

assert len(_CASE_DEFS) == 100, f"케이스 수 오류: {len(_CASE_DEFS)}"


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _build_expert(d: dict) -> dict:
    vc = d.get("vital_conf", 0.70)
    fc = d.get("fall_conf",  0.75)
    sc = d.get("speech_conf", 0.55)
    ec = d.get("env_conf",   0.80)
    kw = d.get("kw", [])
    tx = d.get("tx", "")
    return {
        "fall": {
            "fall_score":       float(d["fall"]),
            "fall_detected":    bool(d["fall_det"]),
            "infer_confidence": float(fc),
        },
        "vital": {
            "heart_rate":       float(d["hr"]),
            "breathing_rate":   float(d["rr"]),
            "infer_confidence": float(vc),
        },
        "env_sound": {
            "label":              d["env"],
            "env_sound_label":    d["env"],
            "confidence":         float(d["env_conf"]),
            "env_sound_confidence": float(d["env_conf"]),
            "infer_confidence":   float(ec),
        },
        "speech_ko": {
            "transcript_ko":  tx,
            "speech_detected": bool(tx),
            "stt_confidence":  float(sc) if tx else 0.0,
            "keywords":        kw,
            "infer_confidence": float(sc),
        },
    }


def _gt_level(emg_score: float) -> str:
    if emg_score >= 0.85:
        return "critical"
    if emg_score >= 0.60:
        return "warning"
    return "normal"


# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋 생성
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset() -> list[dict]:
    cases = []
    for d in _CASE_DEFS:
        er = _build_expert(d)
        score, bd = compute_emergency_score(er)
        gt = _gt_level(score)
        cases.append({
            "id":             d["id"],
            "category":       d["cat"],
            "scenario":       d["scenario"],
            "expert_results": er,
            "ground_truth": {
                "emg_score":    round(score, 4),
                "risk_level":   gt,
                "vital_bypass": bd.get("vital_bypass", False),
                "keyword_fall_bonus": bd.get("keyword_fall_bonus", False),
                "breakdown":    bd,
            },
        })
    return cases


def save_dataset(cases: list[dict]):
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump({"version": "1.0", "total": len(cases), "cases": cases},
                  f, ensure_ascii=False, indent=2)
    print(f"[GEN] 데이터셋 저장: {DATASET_PATH}  ({len(cases)}건)")


def load_dataset() -> list[dict]:
    if not os.path.exists(DATASET_PATH):
        print("[GEN] 데이터셋 없음 — 자동 생성합니다.")
        cases = generate_dataset()
        save_dataset(cases)
        return cases
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


# ─────────────────────────────────────────────────────────────────────────────
# Qwen 평가
# ─────────────────────────────────────────────────────────────────────────────

def _load_qwen():
    p = SLM_MODEL_PATH
    if not os.path.exists(p):
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "volumes", "models", "qwen_05b"),
            r"C:\qwen_llmops\volumes\models\qwen_05b",
        ]
        for c in candidates:
            if os.path.exists(c):
                p = c
                break
        else:
            print(f"[ERR] Qwen 모델 경로 없음: {SLM_MODEL_PATH}")
            return None
    try:
        from inference.qwen_05b import QwenLogic
        q = QwenLogic(p)
        print(f"[LOAD] Qwen: {p}")
        return q
    except Exception as e:
        print(f"[ERR] Qwen 로드 실패: {e}")
        return None


def _level_to_int(level: str) -> int:
    return {"normal": 0, "warning": 1, "critical": 2}.get(level, -1)


def run_evaluation(cases: list[dict], qwen, filter_id=None, filter_cat=None):
    if filter_id:
        cases = [c for c in cases if c["id"] == filter_id]
    if filter_cat:
        cases = [c for c in cases if c["category"] == filter_cat]

    results = []
    per_cat: dict[str, list] = {}

    for i, case in enumerate(cases):
        gt   = case["ground_truth"]["risk_level"]
        emg  = case["ground_truth"]["emg_score"]

        t0 = time.perf_counter()
        qr = qwen.evaluate(case["expert_results"])
        elapsed = (time.perf_counter() - t0) * 1000

        pred = qr.get("risk_level", "unknown")
        pred_score = qr.get("risk_score", 0.0)
        reason = qr.get("qwen_reason", "")
        raw = qr.get("qwen_response", "")

        exact  = (pred == gt)
        gt_int = _level_to_int(gt)
        pd_int = _level_to_int(pred)
        adj    = abs(gt_int - pd_int) <= 1
        safe_fail = (gt == "critical" and pred == "normal")

        rec = {
            "id":         case["id"],
            "category":   case["category"],
            "scenario":   case["scenario"],
            "gt_level":   gt,
            "gt_emg":     emg,
            "pred_level": pred,
            "pred_score": round(pred_score, 3),
            "exact":      exact,
            "adjacent":   adj,
            "safe_fail":  safe_fail,
            "reason":     reason,
            "qwen_raw":   (raw or "")[:200],
            "infer_ms":   round(elapsed, 1),
        }
        results.append(rec)
        per_cat.setdefault(case["category"], []).append(rec)

        status = "O" if exact else ("~" if adj else "X")
        sf_mark = " [SAFE_FAIL!]" if safe_fail else ""
        print(f"  [{status}] {case['id']} gt={gt:8s} pred={pred:8s} "
              f"emg={emg:.3f} score={pred_score:.2f} {elapsed:6.0f}ms{sf_mark}")

    return results, per_cat


# ─────────────────────────────────────────────────────────────────────────────
# 리포트 출력
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: list[dict], per_cat: dict):
    n = len(results)
    exact   = sum(1 for r in results if r["exact"])
    adj     = sum(1 for r in results if r["adjacent"])
    sf      = sum(1 for r in results if r["safe_fail"])
    avg_ms  = sum(r["infer_ms"] for r in results) / n if n else 0

    gt_crit = [r for r in results if r["gt_level"] == "critical"]
    crit_recall = sum(1 for r in gt_crit if r["pred_level"] in ("warning","critical")) / len(gt_crit) * 100 if gt_crit else 0

    gt_norm = [r for r in results if r["gt_level"] == "normal"]
    fn_rate = sum(1 for r in gt_norm if r["pred_level"] != "normal") / len(gt_norm) * 100 if gt_norm else 0

    w = 60
    print("\n" + "=" * w)
    print(f"  Qwen 정확도 평가 결과  ({n}건)")
    print("=" * w)
    print(f"  Exact match   : {exact:3d}/{n}  ({exact/n*100:5.1f}%)")
    print(f"  Adjacent(+-1) : {adj:3d}/{n}  ({adj/n*100:5.1f}%)")
    print(f"  Safe fail     : {sf:3d}/{n}  (GT=critical, pred=normal)")
    print(f"  Critical recall: {crit_recall:5.1f}%  (GT critical -> warning or critical)")
    print(f"  Normal FP rate : {fn_rate:5.1f}%  (GT normal -> 비normal 예측)")
    print(f"  Avg infer ms  : {avg_ms:6.1f}ms")
    print()
    print(f"  {'카테고리':<22} {'건수':>4}  {'Exact':>6}  {'Adj':>6}")
    print(f"  {'-'*22}  {'-'*4}  {'-'*6}  {'-'*6}")
    for cat, recs in sorted(per_cat.items()):
        c_ex = sum(1 for r in recs if r["exact"])
        c_adj = sum(1 for r in recs if r["adjacent"])
        print(f"  {cat:<22} {len(recs):>4}  {c_ex:>3}/{len(recs)}  {c_adj:>3}/{len(recs)}")
    print("=" * w)

    wrong = [r for r in results if not r["exact"]]
    if wrong:
        print(f"\n  오답 케이스 ({len(wrong)}건):")
        for r in wrong:
            sf = " [SAFE_FAIL]" if r["safe_fail"] else ""
            print(f"    {r['id']} {r['scenario'][:30]}")
            print(f"      GT={r['gt_level']} PRED={r['pred_level']} emg={r['gt_emg']:.3f}{sf}")
            print(f"      reason: {r['reason']}")


def save_results(results: list[dict]):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    out = {
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total":        len(results),
        "exact":        sum(1 for r in results if r["exact"]),
        "adjacent":     sum(1 for r in results if r["adjacent"]),
        "safe_fail":    sum(1 for r in results if r["safe_fail"]),
        "results":      results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {RESULTS_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen",  action="store_true", help="데이터셋 재생성만 (Qwen 미호출)")
    ap.add_argument("--id",   default=None, help="단일 케이스 ID (예: D-01)")
    ap.add_argument("--cat",  default=None, help="카테고리 필터 (예: vital_crisis_solo)")
    args = ap.parse_args()

    if args.gen:
        cases = generate_dataset()
        save_dataset(cases)
        _print_dataset_summary(cases)
        return

    cases = load_dataset()
    print(f"[LOAD] 데이터셋: {len(cases)}건")

    qwen = _load_qwen()
    if qwen is None:
        print("[ERR] Qwen 모델을 로드할 수 없어 평가를 종료합니다.")
        sys.exit(1)

    print(f"\n추론 시작 (첫 호출 JIT warmup ~6s 포함)...\n")
    results, per_cat = run_evaluation(cases, qwen, filter_id=args.id, filter_cat=args.cat)
    print_report(results, per_cat)
    if not args.id and not args.cat:
        save_results(results)


def _print_dataset_summary(cases: list[dict]):
    from collections import Counter
    cats = Counter(c["category"] for c in cases)
    gts  = Counter(c["ground_truth"]["risk_level"] for c in cases)
    bps  = sum(1 for c in cases if c["ground_truth"]["vital_bypass"])
    print("\n  [데이터셋 요약]")
    for cat, n in sorted(cats.items()):
        print(f"    {cat:<22} {n}건")
    print(f"\n  GT 분포: normal={gts['normal']} warning={gts['warning']} critical={gts['critical']}")
    print(f"  vital_bypass 케이스: {bps}건")


if __name__ == "__main__":
    main()
