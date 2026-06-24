"""
Qwen 추론 품질 평가 하니스 (Task A / D)

실행:
  python eval/eval_qwen_reasoning.py --golden data/qwen_golden_set.jsonl --report docs/
  python eval/eval_qwen_reasoning.py --mock   # 모델 없이 score 검증만
"""

import argparse
import json
import os
import re
import sys
import datetime
from html import escape as html_escape

# Windows cp949 콘솔에서 한글/특수문자 출력 깨짐 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.emergency_score import compute_emergency_score

_M5_THRESHOLD = 0.6
_SCORE_ATOL   = 0.02


# ── 4가지 룰베이스 채점 함수 ───────────────────────────────────────────────

def _score_numeric_match(reason: str, exp: dict) -> tuple[bool, str]:
    """reason에서 숫자를 추출해 HR/RR 값과 ±5% 비교."""
    numbers = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", reason)]
    for key, field in (("numeric_hr", "heart_rate"), ("numeric_rr", "breathing_rate")):
        target = exp.get(key)
        if target is None:
            continue
        lo, hi = target * 0.95, target * 1.05
        if not any(lo <= n <= hi for n in numbers):
            return False, f"{key}={target} 미언급 (추출된 숫자: {numbers})"
    return True, ""


def _score_label_consistency(reason: str, exp: dict) -> tuple[bool, str]:
    """env_label ≠ alarm/impact 시 reason에 '알람' 포함 금지."""
    if not exp.get("hallucination_guard", False):
        return True, ""
    if "알람" in reason:
        return False, f"hallucination_guard 활성(label={exp['env_label']})인데 '알람' 포함"
    return True, ""


def _score_vital_override(reason: str, risk_level: str, exp: dict) -> tuple[bool, str]:
    """vital_crisis 케이스에서 'normal'/'정상' 미포함 확인."""
    if not exp.get("vital_override", False):
        return True, ""
    if risk_level == "normal":
        return False, f"vital_override 케이스인데 risk_level=normal"
    if "normal" in reason.lower() or "정상" in reason:
        return False, f"vital_override 케이스인데 reason에 'normal'/'정상' 포함"
    return True, ""


def _score_format_complete(reason: str) -> tuple[bool, str]:
    """최소 길이 + 자연어 문장 또는 구조적 compact 포맷 여부 확인.

    Qwen-0.5B는 '낙상위험+심박이상(hr=50)' 같은 compact tag 포맷을 출력한다.
    이는 정보를 충분히 전달하는 유효한 포맷이므로 구두점 없어도 PASS.
    단, '정상'(2자) 같이 의미 없는 단어 하나만 나오는 경우는 FAIL.
    """
    s = reason.strip()
    if len(s) < 4:
        return False, f"reason 너무 짧음 (len={len(s)}): {s!r}"
    # 자연어 문장형: 구두점으로 끝남
    if re.search(r"[.!?。]$", s):
        return True, ""
    # compact 구조형: 한글 2자 이상 + (수치 OR + 연결 OR 괄호 포함)
    if re.search(r"[가-힣]{2,}", s) and re.search(r"\d|\+|\(", s):
        return True, ""
    return False, f"미완결/비구조적: {s!r}"


# ── raw 모델 응답 파서 (가드레일 보정 이전) ─────────────────────────────────

def _parse_raw_response(raw: str) -> tuple[str, str, float]:
    """모델 raw 응답 문자열에서 risk_level / reason / risk_score를 추출.

    evaluate()의 vital_override·알람제거 후처리가 적용되기 전의 원본을 본다.
    JSON 파싱 실패 시 정규식으로 best-effort 복구.
    """
    if not raw:
        return "", "", -1.0
    s = raw.strip()
    try:
        obj = json.loads(s)
        return (str(obj.get("risk_level", "")),
                str(obj.get("reason", "")),
                float(obj.get("risk_score", -1.0)))
    except Exception:
        pass
    lvl = re.search(r'"risk_level"\s*:\s*"([^"]*)"', s)
    rsn = re.search(r'"reason"\s*:\s*"([^"]*)"', s)
    sco = re.search(r'"risk_score"\s*:\s*([0-9.]+)', s)
    return (lvl.group(1) if lvl else "",
            rsn.group(1) if rsn else "",
            float(sco.group(1)) if sco else -1.0)


# ── 케이스 평가 ───────────────────────────────────────────────────────────

def _evaluate_case(case: dict, qwen_logic=None) -> dict:
    inp  = case["input"]
    exp  = case["expected"]
    cid  = case["id"]

    score, breakdown = compute_emergency_score(inp)
    m5_called_actual = score >= _M5_THRESHOLD

    results = {
        "id": cid,
        "category": case["category"],
        "score": round(score, 4),
        "breakdown": breakdown,
        "criteria": {},
        "pass": True,            # Track A: 응급지수 알고리즘 (score 범위 + m5_called)
        "failures": [],
        "m5_pass": None,         # Track B: 호출된 모델 raw 추론 (None=대상아님/미실행)
        "m5_failures": [],
        # 정확도/오탐율 측정용: 독립 정답(ground_truth) vs 시스템 결정(m5_called)
        "ground_truth": case.get("ground_truth_emergency"),
        "predicted": m5_called_actual,
    }

    # ── 점수 범위 검증 ────────────────────────────────────────────────────
    score_min = exp.get("score_min")
    score_max = exp.get("score_max")
    if score_min is not None and score < score_min - _SCORE_ATOL:
        results["pass"] = False
        results["failures"].append(f"score={score:.4f} < min={score_min}")
    if score_max is not None and score > score_max + _SCORE_ATOL:
        results["pass"] = False
        results["failures"].append(f"score={score:.4f} > max={score_max}")

    # m5_called 기대값 일치 확인
    if exp.get("m5_called") is not None and m5_called_actual != exp["m5_called"]:
        results["pass"] = False
        results["failures"].append(
            f"m5_called={m5_called_actual} (expected={exp['m5_called']}, score={score:.4f})"
        )

    # ── M5 추론 필요 없는 케이스 → Track A(점수)만, Track B 대상 아님 ────────
    if not m5_called_actual:
        for name in ("numeric_match", "label_consistency", "vital_override", "format_complete"):
            results["criteria"][name] = {"pass": None, "msg": "M5 미호출 — 대상 아님"}
        return results

    # ── Track B: 모델 미로드면 raw 추론 평가 불가 (Track A는 영향 없음) ──────
    if qwen_logic is None:
        for name in ("numeric_match", "label_consistency", "vital_override", "format_complete"):
            results["criteria"][name] = {"pass": None, "msg": "mock 모드 — 모델 미실행"}
        return results

    # ── M5 추론 실행 ─────────────────────────────────────────────────────
    try:
        eval_result = qwen_logic.evaluate(inp)
    except Exception as exc:
        results["m5_pass"] = False
        results["m5_failures"].append(f"QwenLogic.evaluate 오류: {exc}")
        return results

    # raw(가드레일 이전) vs corrected(후처리) 둘 다 보존
    raw_resp = eval_result.get("qwen_response") or ""
    raw_level, raw_reason, raw_score = _parse_raw_response(raw_resp)
    results["raw_response"]       = raw_resp
    results["raw_risk_level"]     = raw_level
    results["raw_reason"]         = raw_reason
    results["raw_risk_score"]     = raw_score
    results["corrected_reason"]   = eval_result.get("qwen_reason") or ""
    results["corrected_level"]    = eval_result.get("risk_level", "")
    results["vital_override_hit"] = bool(eval_result.get("vital_override"))
    results["qwen_infer_ms"]      = eval_result.get("qwen_infer_ms")
    results["slm_mode"]           = eval_result.get("slm_mode")

    # ── Track B 채점: 가드레일 보정 이전 raw 출력 기준 ──────────────────────
    # (corrected가 아니라 raw를 채점해야 프롬프트/chat_template 효과가 측정됨)
    results["m5_pass"] = True
    scorers = (
        ("numeric_match",      lambda: _score_numeric_match(raw_reason, exp)),
        ("label_consistency",  lambda: _score_label_consistency(raw_reason, exp)),
        ("vital_override",     lambda: _score_vital_override(raw_reason, raw_level, exp)),
        ("format_complete",    lambda: _score_format_complete(raw_reason)),
    )
    for name, fn in scorers:
        ok, msg = fn()
        results["criteria"][name] = {"pass": ok, "msg": msg}
        if not ok:
            results["m5_pass"] = False
            results["m5_failures"].append(f"[{name}] {msg}")

    return results


# ── 콘솔 리포트 ───────────────────────────────────────────────────────────

_CATEGORIES = ["normal", "fall_only", "vital_crisis", "multi_domain",
               "no_signal", "hallucination_guard", "boundary_bva"]


def _confusion(all_results: list[dict]) -> dict:
    """ground_truth(실제 응급) vs predicted(시스템 m5 호출)로 혼동행렬·지표 계산."""
    tp = fp = tn = fn = 0
    skipped = 0
    fp_ids, fn_ids = [], []
    for r in all_results:
        gt = r.get("ground_truth")
        if gt is None:          # 라벨 없는 케이스는 정확도 산정에서 제외
            skipped += 1
            continue
        pred = r.get("predicted", False)
        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
            fn_ids.append((r["id"], r["score"]))
        elif (not gt) and pred:
            fp += 1
            fp_ids.append((r["id"], r["score"]))
        else:
            tn += 1
    n = tp + fp + tn + fn
    acc  = (tp + tn) / n if n else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) else 0.0     # 오탐율
    fnr  = fn / (fn + tp) if (fn + tp) else 0.0     # 미탐율
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n, "skipped": skipped,
        "accuracy": acc, "fpr": fpr, "fnr": fnr, "precision": prec, "recall": rec,
        "fp_ids": fp_ids, "fn_ids": fn_ids,
    }


def _print_metrics(cm: dict) -> None:
    print()
    print("=" * 45)
    print("  정확도 / 오탐율 / 미탐율 (화이트박스 평가)")
    print("=" * 45)
    print(f"  라벨 케이스 N={cm['n']}  (라벨없음 skip={cm['skipped']})")
    print(f"  TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}")
    print(f"  정확도(accuracy)   = {cm['accuracy']:.3f}")
    print(f"  오탐율(FPR)        = {cm['fpr']:.3f}   FP/(FP+TN)")
    print(f"  미탐율(FNR)        = {cm['fnr']:.3f}   FN/(FN+TP)")
    print(f"  정밀도(precision)  = {cm['precision']:.3f}")
    print(f"  재현율(recall)     = {cm['recall']:.3f}")
    if cm["fp_ids"]:
        print(f"  오탐(FP): " + ", ".join(f"{i}({s:.3f})" for i, s in cm["fp_ids"]))
    if cm["fn_ids"]:
        print(f"  미탐(FN): " + ", ".join(f"{i}({s:.3f})" for i, s in cm["fn_ids"]))
    print()


def _m5_summary(all_results: list[dict]) -> dict:
    """Track B: M5 호출된 케이스의 raw 추론 채점 집계."""
    called = [r for r in all_results if r.get("m5_pass") is not None]
    passed = [r for r in called if r["m5_pass"]]
    crit_fail = {"numeric_match": 0, "label_consistency": 0, "vital_override": 0, "format_complete": 0}
    for r in called:
        for name, c in r.get("criteria", {}).items():
            if c.get("pass") is False:
                crit_fail[name] = crit_fail.get(name, 0) + 1
    return {
        "called": len(called), "passed": len(passed),
        "fail": len(called) - len(passed),
        "pass_rate": len(passed) / len(called) if called else 0.0,
        "crit_fail": crit_fail,
    }


def _print_m5_report(all_results: list[dict]) -> None:
    s = _m5_summary(all_results)
    if not s["called"]:
        return
    print("=" * 45)
    print("  Track B — 호출된 모델 raw 추론 평가")
    print("=" * 45)
    print(f"  M5 호출 {s['called']}케이스 · raw PASS {s['passed']} / FAIL {s['fail']} "
          f"(통과율 {s['pass_rate']:.3f})")
    print(f"  기준별 실패: numeric={s['crit_fail']['numeric_match']} "
          f"label={s['crit_fail']['label_consistency']} "
          f"vital_ov={s['crit_fail']['vital_override']} "
          f"format={s['crit_fail']['format_complete']}")
    for r in all_results:
        if r.get("m5_pass") is False:
            print(f"  FAIL [{r['id']}] raw_level={r.get('raw_risk_level','')!r} "
                  f"raw_reason={ (r.get('raw_reason','') or '')[:60]!r}")
            for f in r["m5_failures"]:
                print(f"        {f}")
    print()


def _dump_raw_responses(all_results: list[dict], report_dir: str) -> str:
    """Track B 데이터셋: 케이스별 raw 응답 전문을 JSON으로 저장."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"qwen_responses_{datetime.date.today():%Y%m%d}.json")
    items = []
    for r in all_results:
        if r.get("m5_pass") is None:
            continue
        items.append({
            "id": r["id"], "category": r["category"], "score": r["score"],
            "raw_response": r.get("raw_response", ""),
            "raw_risk_level": r.get("raw_risk_level", ""),
            "raw_reason": r.get("raw_reason", ""),
            "raw_risk_score": r.get("raw_risk_score"),
            "corrected_reason": r.get("corrected_reason", ""),
            "corrected_level": r.get("corrected_level", ""),
            "vital_override_hit": r.get("vital_override_hit", False),
            "qwen_infer_ms": r.get("qwen_infer_ms"),
            "m5_pass": r.get("m5_pass"),
            "m5_failures": r.get("m5_failures", []),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return path


def _print_report(all_results: list[dict]) -> None:
    by_cat: dict[str, list] = {c: [] for c in _CATEGORIES}
    for r in all_results:
        by_cat.setdefault(r["category"], []).append(r)

    total_pass = total_fail = 0
    print()
    print(f"{'카테고리':<22} {'PASS':>5} {'FAIL':>5} {'케이스':>6}")
    print("-" * 45)
    for cat in _CATEGORIES:
        cases = by_cat.get(cat, [])
        p = sum(1 for c in cases if c["pass"])
        f = len(cases) - p
        total_pass += p
        total_fail += f
        status = "  OK" if f == 0 else "FAIL"
        print(f"  {cat:<20} {p:>5} {f:>5} {len(cases):>6}  {status}")
    print("-" * 45)
    print(f"  {'합계':<20} {total_pass:>5} {total_fail:>5} {total_pass+total_fail:>6}  "
          f"{'OK' if total_fail == 0 else f'FAIL {total_fail}개'}")
    print()

    for r in all_results:
        if not r["pass"]:
            print(f"  FAIL  [{r['id']}]  score={r['score']:.4f}")
            for f in r["failures"]:
                print(f"         {f}")
            if r.get("raw_reason"):
                print(f"         raw_reason: {r['raw_reason'][:120]!r}")
            print()


# ── HTML 리포트 ───────────────────────────────────────────────────────────

def _write_html(all_results: list[dict], report_dir: str, cm: dict = None) -> str:
    date_str = datetime.date.today().strftime("%Y%m%d")
    path = os.path.join(report_dir, f"qwen_eval_report_{date_str}.html")

    by_cat: dict[str, list] = {c: [] for c in _CATEGORIES}
    for r in all_results:
        by_cat.setdefault(r["category"], []).append(r)

    rows_html = ""
    for cat in _CATEGORIES:
        cases = by_cat.get(cat, [])
        for r in cases:
            status_cell = '<td style="color:green">PASS</td>' if r["pass"] else '<td style="color:red">FAIL</td>'
            # Track B (모델 raw 추론) 상태
            if r.get("m5_pass") is None:
                m5_cell = "<td>—</td>"
            elif r["m5_pass"]:
                m5_cell = '<td style="color:green">raw PASS</td>'
            else:
                m5_cell = '<td style="color:red" title="{}">raw FAIL</td>'.format(
                    html_escape("; ".join(r.get("m5_failures", []))))
            criteria_cells = ""
            for cname in ("numeric_match", "label_consistency", "vital_override", "format_complete"):
                c = r.get("criteria", {}).get(cname, {})
                if c.get("pass") is None:
                    criteria_cells += "<td>—</td>"
                elif c.get("pass"):
                    criteria_cells += '<td style="color:green">✓</td>'
                else:
                    criteria_cells += '<td style="color:red" title="{}">✗</td>'.format(
                        html_escape(c.get("msg", "")))
            reason_html = html_escape((r.get("raw_reason") or "")[:100])
            gt = r.get("ground_truth")
            pred = r.get("predicted")
            gt_cell = "—" if gt is None else ("응급" if gt else "정상")
            # 오탐/미탐 강조
            verdict = ""
            if gt is not None:
                if gt and not pred:
                    verdict = '<td style="color:#c00;font-weight:bold">미탐 FN</td>'
                elif (not gt) and pred:
                    verdict = '<td style="color:#c60;font-weight:bold">오탐 FP</td>'
                else:
                    verdict = '<td style="color:#090">정답</td>'
            else:
                verdict = "<td>—</td>"
            rows_html += (
                f"<tr>{status_cell}"
                f"<td>{r['id']}</td><td>{r['category']}</td>"
                f"<td>{r['score']:.4f}</td>"
                f"<td>{gt_cell}</td>{verdict}"
                f"{m5_cell}"
                f"{criteria_cells}"
                f"<td>{reason_html}</td></tr>\n"
            )

    total_pass = sum(1 for r in all_results if r["pass"])
    total      = len(all_results)
    summary    = f"{total_pass}/{total} PASS"

    # 정확도/오탐율 패널
    metrics_html = ""
    if cm and cm.get("n"):
        metrics_html = f"""
<h3>정확도 / 오탐율 / 미탐율 (화이트박스 평가)</h3>
<table style="width:auto">
<tr><th>지표</th><th>값</th><th>정의</th></tr>
<tr><td>정확도 (accuracy)</td><td><b>{cm['accuracy']:.3f}</b></td><td>(TP+TN)/N</td></tr>
<tr><td>오탐율 (FPR)</td><td><b>{cm['fpr']:.3f}</b></td><td>FP/(FP+TN) — 정상을 응급으로 오판</td></tr>
<tr><td>미탐율 (FNR)</td><td><b>{cm['fnr']:.3f}</b></td><td>FN/(FN+TP) — 응급을 놓침</td></tr>
<tr><td>정밀도 (precision)</td><td>{cm['precision']:.3f}</td><td>TP/(TP+FP)</td></tr>
<tr><td>재현율 (recall)</td><td>{cm['recall']:.3f}</td><td>TP/(TP+FN)</td></tr>
</table>
<p>혼동행렬: TP={cm['tp']} · FP={cm['fp']} · TN={cm['tn']} · FN={cm['fn']} · N={cm['n']} (라벨없음 skip={cm['skipped']})</p>
<p><b>오탐(FP)</b>: {', '.join(f"{i}({s:.3f})" for i, s in cm['fp_ids']) or '없음'}</p>
<p><b>미탐(FN)</b>: {', '.join(f"{i}({s:.3f})" for i, s in cm['fn_ids']) or '없음'}</p>
"""

    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>Qwen Eval {date_str}</title>
<style>
  body{{font-family:monospace;font-size:13px;padding:20px}}
  h2{{margin-bottom:8px}}
  table{{border-collapse:collapse;width:100%}}
  th,td{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
  th{{background:#f0f0f0}}
  tr:nth-child(even){{background:#fafafa}}
</style>
</head><body>
<h2>Qwen Reasoning Eval — {date_str}</h2>
<p><b>{summary}</b></p>
{metrics_html}
<h3>케이스별 상세</h3>
<p class="sub">결과=Track A(알고리즘 score) · 모델raw=Track B(가드레일 이전 모델 추론) · ✓✗는 raw 기준 채점</p>
<table>
<tr>
  <th>결과</th><th>ID</th><th>카테고리</th><th>score</th>
  <th>정답(GT)</th><th>판정</th>
  <th>모델raw</th>
  <th>numeric</th><th>label</th><th>vital_ov</th><th>format</th>
  <th>raw_reason (앞100자)</th>
</tr>
{rows_html}
</table>
</body></html>
"""
    os.makedirs(report_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ── 진입점 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen 추론 품질 평가")
    parser.add_argument("--golden", default="data/qwen_golden_set.jsonl")
    parser.add_argument("--report", default="docs/")
    parser.add_argument("--model",  default=None, help="Qwen ONNX 모델 경로 (미입력 시 env SLM_MODEL 참조)")
    parser.add_argument("--mock",   action="store_true", help="모델 없이 score 검증만")
    parser.add_argument("--gpu",    action="store_true", help="GPU 추론(ORT_USE_GPU=1, DirectML→CUDA→CPU)")
    parser.add_argument("--impl",   default="05b", choices=["05b", "15b"],
                        help="추론 구현 선택: 05b=qwen_05b / 15b=qwen_15b (Qwen2.5-1.5B)")
    args = parser.parse_args()

    # GPU 설정: 모델 로드 전에 env 주입 (utils.get_ort_providers가 참조)
    if args.gpu:
        os.environ["ORT_USE_GPU"] = "1"
        print("[INFO] GPU 추론 활성화 (ORT_USE_GPU=1): DirectML → CUDA → CPU fallback")

    # 골든셋 로드
    if not os.path.exists(args.golden):
        print(f"[ERR] 골든셋 파일 없음: {args.golden}")
        sys.exit(1)
    cases = []
    with open(args.golden, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    print(f"[INFO] 골든셋 {len(cases)}케이스 로드 완료")

    # Qwen 모델 로드 (mock 아닐 때)
    qwen_logic = None
    if not args.mock:
        _default_model = "volumes/models/qwen_15b" if args.impl == "15b" else "volumes/models/qwen_05b"
        model_path = args.model or os.getenv("SLM_MODEL", _default_model)
        if os.path.exists(model_path):
            if args.impl == "15b":
                from inference.qwen_15b import QwenLogic
            else:
                from inference.qwen_05b import QwenLogic
            qwen_logic = QwenLogic(model_path)
            print(f"[INFO] 모델 로드: {model_path} (impl={args.impl})")
        else:
            print(f"[WARN] 모델 경로 없음({model_path}), mock 모드로 전환")

    # 케이스별 평가
    all_results = [_evaluate_case(c, qwen_logic) for c in cases]

    # 콘솔 출력
    _print_report(all_results)

    # Track A: 정확도/오탐율/미탐율 (ground_truth 라벨 기반)
    cm = _confusion(all_results)
    _print_metrics(cm)

    # Track B: 호출된 모델 raw 추론 평가 + raw 응답 덤프
    _print_m5_report(all_results)
    if any(r.get("m5_pass") is not None for r in all_results):
        json_path = _dump_raw_responses(all_results, args.report)
        print(f"[INFO] raw 응답 데이터셋: {json_path}")

    # HTML 리포트 저장
    html_path = _write_html(all_results, args.report, cm)
    print(f"[INFO] HTML 리포트: {html_path}")

    total_fail = sum(1 for r in all_results if not r["pass"])
    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
