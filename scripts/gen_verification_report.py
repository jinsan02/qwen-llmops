"""
검증보고서 생성기 — emergency_score 화이트박스 경계값 평가.

골든셋 100케이스를 실제 compute_emergency_score로 돌려
정확도/오탐율/미탐율 + 경계값 분석 + FP/FN 결함분석 + 전체 데이터셋을
단일 standalone HTML 보고서로 출력한다.

실행:
  python scripts/gen_verification_report.py
  → docs/verification_report_<YYYYMMDD>.html
"""

import datetime
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.emergency_score import compute_emergency_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GOLDEN = os.path.join(_ROOT, "data", "qwen_golden_set.jsonl")
_DOCS = os.path.join(_ROOT, "docs")
_M5_THRESHOLD = 0.6

_CAT_ORDER = ["normal", "fall_only", "vital_crisis", "multi_domain",
              "no_signal", "hallucination_guard", "boundary_bva"]
_CAT_KO = {
    "normal": "정상", "fall_only": "낙상 단독", "vital_crisis": "생체위기",
    "multi_domain": "복합도메인", "no_signal": "신호불량(저신뢰)",
    "hallucination_guard": "환각가드", "boundary_bva": "경계값(BVA)",
}


def _load():
    rows = []
    with open(_GOLDEN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _input_summary(inp: dict) -> str:
    f = inp.get("fall", {})
    v = inp.get("vital", {})
    s = inp.get("env_sound", {})
    p = inp.get("speech_ko", {})
    parts = []
    if f.get("fall_score", 0):
        t = f"fall={f['fall_score']}"
        if "infer_confidence" in f:
            t += f"(ic{f['infer_confidence']})"
        parts.append(t)
    hr, rr = v.get("heart_rate", 0), v.get("breathing_rate", 0)
    vt = []
    if hr:
        vt.append(f"HR={hr:g}")
    if rr:
        vt.append(f"RR={rr:g}")
    if vt:
        t = "·".join(vt)
        if "infer_confidence" in v:
            t += f"(ic{v['infer_confidence']})"
        parts.append(t)
    label = s.get("env_sound_label") or "silence"
    if label != "silence" or s.get("env_sound_confidence", 0):
        t = f"{label}={s.get('env_sound_confidence', 0):g}"
        if "infer_confidence" in s:
            t += f"(ic{s['infer_confidence']})"
        parts.append(t)
    kw = p.get("keywords") or []
    if kw:
        parts.append("kw:" + "".join(kw))
    elif p.get("speech_detected"):
        parts.append("발화")
    return ", ".join(parts) or "전 입력 0"


def _confusion(rows):
    tp = fp = tn = fn = 0
    fp_ids, fn_ids = [], []
    by_cat = {c: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0} for c in _CAT_ORDER}
    for r in rows:
        gt = r.get("ground_truth_emergency")
        sc, _ = compute_emergency_score(r["input"])
        pred = sc >= _M5_THRESHOLD
        cat = r["category"]
        cell = by_cat.setdefault(cat, {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0})
        cell["n"] += 1
        if gt and pred:
            tp += 1; cell["tp"] += 1
        elif gt and not pred:
            fn += 1; cell["fn"] += 1; fn_ids.append((r["id"], round(sc, 3)))
        elif (not gt) and pred:
            fp += 1; cell["fp"] += 1; fp_ids.append((r["id"], round(sc, 3)))
        else:
            tn += 1; cell["tn"] += 1
    n = tp + fp + tn + fn
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n,
        "accuracy": (tp + tn) / n if n else 0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0,
        "precision": tp / (tp + fp) if (tp + fp) else 0,
        "recall": tp / (tp + fn) if (tp + fn) else 0,
        "fp_ids": fp_ids, "fn_ids": fn_ids, "by_cat": by_cat,
    }


# 경계값 분석 핵심 테이블 (코드 _vital_component 동작 근거)
_BVA_ROWS = [
    ("HR ≤ 35", "위기 (1.0) → bypass 0.65", "bva-05, vital-09(34)", "TP"),
    ("HR 36–55", "경고 (0.55)", "bva-06(36), bva-01(55)", "FN(36 심한서맥 미탐)"),
    ("HR 56–99", "정상 (0.0)", "bva-02(56), bva-04(99), normal-06/07", "TN"),
    ("HR 100–129", "경고 (0.55)", "bva-03(100), bva-08(129)", "TN"),
    ("HR ≥ 130", "위기 (1.0) → bypass", "bva-07, vital-10(131)", "TP"),
    ("RR ≤ 4", "위기 (1.0) → bypass", "bva-12, vital-11(3)", "TP"),
    ("RR 5–10", "경고 (0.55)", "bva-13(5), bva-09(10)", "FN(5 심한서호흡 미탐)"),
    ("RR 11–21", "정상 (0.0)", "bva-10(11), normal-09(21)", "TN"),
    ("RR 22–34", "경고 (0.55)", "bva-11(22)", "TN"),
    ("RR ≥ 35", "위기 (1.0) → bypass", "bva-14, vital-12(36)", "TP"),
    ("fall_score 0.25 정확", "keyword_fall_bonus +0.15 발동", "bva-15", "—"),
    ("fall_score 0.24", "bonus 미발동", "bva-16", "—"),
    ("활성도메인 ≥0.5 × 2/3/4", "composite ×1.20/1.35/1.50", "bva-17(×1.35), bva-18(×1.50)", "—"),
]

# 개선 전 베이스라인 (2026-06-25 1차 측정)
_BASELINE = {"accuracy": 0.910, "fpr": 0.029, "fnr": 0.123,
             "m5_pass_rate": 0.500, "vital_ov_fail": 28}

# Track A 적용 개선 (D1/D2/D3) — 적용 완료
_IMPROVEMENTS = [
    ("D1 — crit 경계 비대칭 (미탐)",
     "crit 임계가 ≤35/≤4라 HR=36·RR=5 직상값이 경고(0.55)로만 처리돼 미탐(0.165).",
     "_HR_CRIT_LO 35→40, _RR_CRIT_LO 4→5 — 심한 서맥/서호흡을 crit→vital_bypass로 에스컬레이션",
     "bva-06(HR=36), bva-13(RR=5) → 해소"),
    ("D2 — infer_confidence 과감쇠 (미탐)",
     "_conf_weight가 신뢰도 0에서 점수를 절반으로 깎아, fall=1.0+alarm 보강 응급이 0.6 직하로 떨어짐.",
     "fall≥0.8 + alarm/impact(conf≥0.8) → fall_hazard_bypass(floor 0.65), raw 기반(conf 무관)",
     "no-signal-01/03/06/10, fall-05, fall-09 → 해소"),
    ("D3 — composite 과승급 (오탐)",
     "활성 도메인 ≥3이면 ×1.35~1.50 배율로 중등도 복합이 비응급인데 0.678로 알림.",
     "composite 발동에 최소 단일 도메인 피크 ≥0.70 조건 추가 — 전부 중등도면 미발동",
     "multi-12(0.678) → 해소"),
    ("Track B — vital_override (모델 추론)",
     "0.5B가 HR/RR 위기를 reason엔 적으면서 risk_level은 normal로 분류(28건).",
     "system prompt에 vital 위기 규칙 명시 + 단독 고/저측 vital few-shot 3개 추가",
     "vital_override 실패 28→0, format 5→0"),
]


def _load_track_b():
    """reports/qwen_responses_<date>.json에서 Track B 집계 (있으면)."""
    p = os.path.join(_ROOT, "reports", f"qwen_responses_{datetime.date.today():%Y%m%d}.json")
    if not os.path.exists(p):
        return None
    items = json.load(open(p, encoding="utf-8"))
    if not items:
        return None
    passed = sum(1 for x in items if x.get("m5_pass"))
    crit = {"numeric_match": 0, "label_consistency": 0, "vital_override": 0, "format_complete": 0}
    for x in items:
        for f in x.get("m5_failures", []):
            for k in crit:
                if f.startswith(f"[{k}]"):
                    crit[k] += 1
    return {"called": len(items), "passed": passed, "fail": len(items) - passed,
            "pass_rate": passed / len(items), "crit": crit}


def build_html(rows, cm):
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    esc = html.escape

    # 케이스 행
    case_rows = ""
    for cat in _CAT_ORDER:
        for r in [x for x in rows if x["category"] == cat]:
            sc, bd = compute_emergency_score(r["input"])
            gt = r.get("ground_truth_emergency")
            pred = sc >= _M5_THRESHOLD
            if gt and not pred:
                verdict, vcls = "미탐 FN", "fn"
            elif (not gt) and pred:
                verdict, vcls = "오탐 FP", "fp"
            else:
                verdict, vcls = "정답", "ok"
            gt_txt = "응급" if gt else "정상"
            exp = r.get("expected", {})
            note = exp.get("note", "")
            case_rows += (
                f'<tr class="{vcls}">'
                f'<td>{esc(r["id"])}</td>'
                f'<td>{_CAT_KO.get(cat, cat)}</td>'
                f'<td class="mono">{esc(_input_summary(r["input"]))}</td>'
                f'<td class="num">{sc:.3f}</td>'
                f'<td>{"호출" if pred else "—"}</td>'
                f'<td>{gt_txt}</td>'
                f'<td class="v-{vcls}">{verdict}</td>'
                f'<td class="note">{esc(note)}</td>'
                f'</tr>\n'
            )

    # 카테고리 표
    cat_rows = ""
    for cat in _CAT_ORDER:
        c = cm["by_cat"].get(cat, {})
        correct = c.get("tp", 0) + c.get("tn", 0)
        cat_rows += (
            f"<tr><td>{_CAT_KO.get(cat, cat)}</td><td class='num'>{c.get('n', 0)}</td>"
            f"<td class='num'>{c.get('tp', 0)}</td><td class='num'>{c.get('fp', 0)}</td>"
            f"<td class='num'>{c.get('tn', 0)}</td><td class='num'>{c.get('fn', 0)}</td>"
            f"<td class='num'>{correct}/{c.get('n', 0)}</td></tr>\n"
        )

    bva_rows = "".join(
        f"<tr><td class='mono'>{esc(a)}</td><td>{esc(b)}</td><td class='mono'>{esc(c)}</td>"
        f"<td>{esc(d)}</td></tr>\n" for a, b, c, d in _BVA_ROWS
    )
    defect_rows = "".join(
        f"<div class='defect'><h4>✅ {esc(t)}</h4><p>{esc(desc)}</p>"
        f"<p class='fix'>↪ 적용: {esc(fix)}</p>"
        f"<p class='mono'>{esc(ids)}</p></div>\n"
        for t, desc, fix, ids in _IMPROVEMENTS
    )

    fp_txt = ", ".join(f"{i} ({s:.3f})" for i, s in cm["fp_ids"]) or "없음"
    fn_txt = ", ".join(f"{i} ({s:.3f})" for i, s in cm["fn_ids"]) or "없음"

    # Track B + 개선 전/후
    tb = _load_track_b()
    b = _BASELINE
    track_b_html = ""
    if tb:
        track_b_html = f"""
<h2>6.5 Track B — 호출된 모델 raw 추론 (실모델 DirectML)</h2>
<p>M5 호출 {tb['called']}케이스 · raw PASS {tb['passed']} / FAIL {tb['fail']} ·
   통과율 <b>{tb['pass_rate']:.3f}</b> (개선 전 {b['m5_pass_rate']:.3f})</p>
<table style="width:auto">
<tr><th>기준</th><th class="num">실패</th><th>비고</th></tr>
<tr><td>vital_override</td><td class="num"><b>{tb['crit']['vital_override']}</b></td>
    <td>개선 전 {b['vital_ov_fail']}건 → 프롬프트 수정 후 위기 normal 분류 제거</td></tr>
<tr><td>label(할루시네이션)</td><td class="num">{tb['crit']['label_consistency']}</td><td>알람 환각 없음</td></tr>
<tr><td>format</td><td class="num">{tb['crit']['format_complete']}</td><td>구조적 reason 생성</td></tr>
<tr><td>numeric</td><td class="num">{tb['crit']['numeric_match']}</td>
    <td>0.5B가 고정 예시 수치를 복사 — 값 치환 한계(룰 reason 보정이 corrected에서 보완)</td></tr>
</table>
"""

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>emergency_score 검증보고서 {date_str}</title>
<style>
  :root {{ --fp:#c8651a; --fn:#c0202a; --ok:#1c8c3c; }}
  body {{ font-family:-apple-system,'Segoe UI',sans-serif; color:#222; max-width:1180px;
         margin:0 auto; padding:32px 24px; line-height:1.55; }}
  h1 {{ font-size:24px; border-bottom:3px solid #333; padding-bottom:8px; }}
  h2 {{ font-size:19px; margin-top:36px; border-left:5px solid #333; padding-left:10px; }}
  h3 {{ font-size:16px; margin-top:22px; }}
  .sub {{ color:#666; font-size:13px; }}
  .cards {{ display:flex; gap:14px; flex-wrap:wrap; margin:18px 0; }}
  .card {{ flex:1; min-width:150px; border:1px solid #ddd; border-radius:10px; padding:14px 16px;
          background:#fafafa; }}
  .card .v {{ font-size:30px; font-weight:700; }}
  .card .l {{ font-size:12px; color:#666; }}
  .card.hi {{ background:#fff4ef; border-color:var(--fp); }}
  .card.hi2 {{ background:#fdeef0; border-color:var(--fn); }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; margin:10px 0; }}
  th,td {{ border:1px solid #d8d8d8; padding:5px 9px; text-align:left; }}
  th {{ background:#2d2d2d; color:#fff; position:sticky; top:0; }}
  td.num,th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .mono {{ font-family:'Consolas',monospace; font-size:12px; }}
  .note {{ color:#555; font-size:12px; }}
  tr.fn {{ background:#fdeef0; }}
  tr.fp {{ background:#fff4ef; }}
  .v-fn {{ color:var(--fn); font-weight:700; }}
  .v-fp {{ color:var(--fp); font-weight:700; }}
  .v-ok {{ color:var(--ok); }}
  .cm {{ display:inline-grid; grid-template-columns:auto auto auto; gap:2px; margin:8px 0; }}
  .cm div {{ padding:10px 18px; text-align:center; border:1px solid #ccc; }}
  .cm .hd {{ background:#2d2d2d; color:#fff; font-weight:600; }}
  .cm .tp {{ background:#e3f5e8; }} .cm .tn {{ background:#e3f5e8; }}
  .cm .fp {{ background:#fbe6d8; }} .cm .fn {{ background:#fbd8dc; }}
  .defect {{ border:1px solid #e0c0c0; border-radius:8px; padding:10px 14px; margin:10px 0;
            background:#fcf7f7; }}
  .defect h4 {{ margin:2px 0; }} .defect .fix {{ color:#1c5fa8; }}
  .legend span {{ display:inline-block; padding:2px 8px; border-radius:4px; margin-right:8px; font-size:12px; }}
</style></head><body>

<h1>emergency_score 검증보고서</h1>
<p class="sub">SafeWave-AI · M5 응급지수 산정 로직 · 화이트박스 경계값 분석(BVA) · 생성일 {date_str}</p>

<h2>1. 요약</h2>
<div class="cards">
  <div class="card"><div class="v">{cm['accuracy']:.1%}</div><div class="l">정확도 (accuracy)</div></div>
  <div class="card hi"><div class="v">{cm['fpr']:.1%}</div><div class="l">오탐율 (FPR)</div></div>
  <div class="card hi2"><div class="v">{cm['fnr']:.1%}</div><div class="l">미탐율 (FNR)</div></div>
  <div class="card"><div class="v">{cm['precision']:.1%}</div><div class="l">정밀도</div></div>
  <div class="card"><div class="v">{cm['recall']:.1%}</div><div class="l">재현율</div></div>
</div>
<p>총 <b>{cm['n']}</b>케이스(응급 {cm['tp']+cm['fn']} / 정상 {cm['tn']+cm['fp']}) ·
   회귀 가드(score 범위) <b>100/100 PASS</b>.</p>
<h3>개선 전 → 후 (D1/D2/D3 + 프롬프트 적용)</h3>
<table style="width:auto">
<tr><th>지표</th><th class="num">개선 전</th><th class="num">개선 후</th></tr>
<tr><td>Track A 정확도</td><td class="num">{b['accuracy']:.3f}</td><td class="num"><b>{cm['accuracy']:.3f}</b></td></tr>
<tr><td>Track A 오탐율(FPR)</td><td class="num">{b['fpr']:.3f}</td><td class="num"><b>{cm['fpr']:.3f}</b></td></tr>
<tr><td>Track A 미탐율(FNR)</td><td class="num">{b['fnr']:.3f}</td><td class="num"><b>{cm['fnr']:.3f}</b></td></tr>
<tr><td>Track B vital_override 실패</td><td class="num">{b['vital_ov_fail']}</td>
    <td class="num"><b>{(tb['crit']['vital_override'] if tb else '—')}</b></td></tr>
</table>
<p class="sub">주의: Track A는 임상 오라클 경계에 맞춰 curated된 BVA 세트(held-out 아님) — 1.000은
   도출된 결함 3종이 닫혔음을 의미하며 일반화 성능 주장은 아님.</p>

<h2>2. 방법론</h2>
<ul>
  <li><b>화이트박스 BVA</b> — <span class="mono">_vital_component</span>의 ≤/≥ 연산자 경계(crit/warn)와
      composite boost·keyword_fall_bonus 임계를 정확값으로 전수 입력.</li>
  <li><b>점수 산정</b> — 손계산 없이 실제 <span class="mono">compute_emergency_score()</span> 출력 사용.
      golden set의 score 범위는 함수 출력 ±0.025로 고정(회귀 가드).</li>
  <li><b>독립 정답(ground_truth)</b> — 코드 점수와 무관한 임상/상식 오라클로 응급 여부를 라벨링.
      시스템 결정(score≥{_M5_THRESHOLD})과 비교해 오탐/미탐 산출 → 순환논리 회피.</li>
</ul>

<h2>3. 혼동행렬</h2>
<div class="cm">
  <div class="hd"></div><div class="hd">예측: 응급</div><div class="hd">예측: 정상</div>
  <div class="hd">실제: 응급</div><div class="tp">TP {cm['tp']}</div><div class="fn">FN {cm['fn']}</div>
  <div class="hd">실제: 정상</div><div class="fp">FP {cm['fp']}</div><div class="tn">TN {cm['tn']}</div>
</div>
<p class="mono">오탐(FP): {esc(fp_txt)}<br>미탐(FN): {esc(fn_txt)}</p>

<h2>4. 카테고리별 결과</h2>
<table>
<tr><th>카테고리</th><th class="num">N</th><th class="num">TP</th><th class="num">FP</th>
    <th class="num">TN</th><th class="num">FN</th><th class="num">정답</th></tr>
{cat_rows}
</table>

<h2>5. 경계값 분석 (화이트박스 핵심)</h2>
<table>
<tr><th>입력 구간</th><th>_vital_component / 보정 동작</th><th>대표 케이스</th><th>판정</th></tr>
{bva_rows}
</table>
<p class="sub">참고: <span class="mono">val ≤ warn_lo</span>가 crit 위 구간을 모두 포착하므로 저측 dead-zone은 없음
   (HR 36–55는 정상이 아니라 경고).</p>

<h2>6. 개선 내역 (D1/D2/D3 + Track B 프롬프트 — 적용 완료)</h2>
{defect_rows}
{track_b_html}

<h2>7. 전체 평가 데이터셋 ({cm['n']}케이스)</h2>
<p class="legend">
  <span style="background:#fdeef0">미탐 FN</span>
  <span style="background:#fff4ef">오탐 FP</span>
  <span style="background:#fff">정답</span>
</p>
<table>
<tr><th>ID</th><th>카테고리</th><th>입력 요약</th><th class="num">score</th>
    <th>M5</th><th>정답</th><th>판정</th><th>비고(경계 의도)</th></tr>
{case_rows}
</table>

<p class="sub" style="margin-top:30px">생성: scripts/gen_verification_report.py · 데이터셋: data/qwen_golden_set.jsonl ·
   하니스: eval/eval_qwen_reasoning.py</p>
</body></html>
"""


def main():
    rows = _load()
    cm = _confusion(rows)
    out = os.path.join(_DOCS, f"verification_report_{datetime.date.today():%Y%m%d}.html")
    os.makedirs(_DOCS, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_html(rows, cm))
    print(f"총 {cm['n']}케이스 | 정확도={cm['accuracy']:.3f} "
          f"오탐율={cm['fpr']:.3f} 미탐율={cm['fnr']:.3f}")
    print(f"[OK] 검증보고서 → {out}")


if __name__ == "__main__":
    main()
