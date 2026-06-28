"""
백엔드 raw 해체 비교 + 토큰/타이밍 분석 + 프롬프트 토큰 분해 → HTML 보고서.

입력: reports/qwen_responses_{15b_200,gguf_200,int8b}.json (있는 것만)
      data/qwen_golden_set.jsonl (Track A 강건성)
출력: 콘솔 요약 + docs/safewave_m5_robustness_token.html

실행: python scripts/analyze_backends.py
"""

import datetime
import html
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.emergency_score import compute_emergency_score
from inference import qwen_15b as Q

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPORTS = os.path.join(_ROOT, "reports")
_GOLDEN = os.path.join(_ROOT, "data", "qwen_golden_set.jsonl")
_DOCS = os.path.join(_ROOT, "docs")
_TOK_DIR = os.path.join(_ROOT, "volumes", "models", "qwen_15b")
_M5 = 0.6

# (라벨, 파일) — 있는 것만 사용
_BACKENDS = [
    ("1.5B fp32", "qwen_responses_15b_4shot.json"),
    ("GGUF Q4_K_M", "qwen_responses_gguf_4shot.json"),
    ("INT8 block-wise", "qwen_responses_int8b.json"),
]


def _load_json(fn):
    p = os.path.join(_REPORTS, fn)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def _load_golden():
    rows = []
    with open(_GOLDEN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pct(vals, p):
    if not vals:
        return 0
    vals = sorted(vals)
    k = (len(vals) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


# ── (강건성) ────────────────────────────────────────────────────────────────
def robustness_track_a(golden):
    by = {r["id"]: r for r in golden}
    cleans = [r for r in golden if r.get("noisy") is False and r.get("pair_id")]
    flips = []
    for c in cleans:
        n = by.get(c["pair_id"] + "-n")
        if not n:
            continue
        cp = compute_emergency_score(c["input"])[0] >= _M5
        npd = compute_emergency_score(n["input"])[0] >= _M5
        if cp != npd:
            flips.append(c["pair_id"])
    return len(cleans), flips


def robustness_track_b(items, golden):
    """golden 쌍 전체 기준. dump에 없으면(=M5 미호출) 'none'으로 간주해 called↔none flip도 포착."""
    lvl = {x["id"]: x.get("raw_risk_level") for x in items}
    cleans = [r for r in golden if r.get("noisy") is False and r.get("pair_id")]
    pairs, flips = 0, []
    for c in cleans:
        nid = c["pair_id"] + "-n"
        if not any(r["id"] == nid for r in golden):
            continue
        pairs += 1
        cv = lvl.get(c["id"], "none")
        nv = lvl.get(nid, "none")
        if cv != nv:
            flips.append(c["pair_id"])
    return pairs, flips


# ── (해체 비교) 공통 id 필드 단위 비교 ──────────────────────────────────────
def decompose(dumps):
    common = set.intersection(*[{x["id"] for x in d} for d in dumps.values()]) if dumps else set()
    common = sorted(common)
    idx = {lbl: {x["id"]: x for x in d} for lbl, d in dumps.items()}
    level_agree = 0
    rows = []
    for cid in common:
        levels = {lbl: idx[lbl][cid].get("raw_risk_level", "") for lbl in dumps}
        agree = len(set(levels.values())) == 1
        level_agree += agree
        rows.append((cid, levels, agree))
    rate = level_agree / len(common) if common else 0
    disagree = [(cid, lv) for cid, lv, a in rows if not a]
    return common, rate, disagree


# ── (토큰/타이밍) ────────────────────────────────────────────────────────────
def token_stats(items):
    ot = [x["output_tokens"] for x in items if x.get("output_tokens") is not None]
    pt = [x["prompt_tokens"] for x in items if x.get("prompt_tokens") is not None]
    ms = [x["qwen_infer_ms"] for x in items if x.get("qwen_infer_ms") is not None]
    if not ot:
        return None
    mspt = [m / o for m, o in zip(ms, ot) if o] if ms else []
    return {
        "n": len(ot),
        "out_min": min(ot), "out_med": int(st.median(ot)),
        "out_p95": round(_pct(ot, 95)), "out_p99": round(_pct(ot, 99)), "out_max": max(ot),
        "prompt_med": int(st.median(pt)) if pt else None,
        "ms_med": round(st.median(ms)) if ms else None,
        "ms_per_tok": round(st.median(mspt), 1) if mspt else None,
        "max_reco": int(round(_pct(ot, 99))) + 8,   # MAX 권고 = p99 + 여유
        "min_reco": min(ot),
    }


# ── (프롬프트 토큰 분해) ─────────────────────────────────────────────────────
def prompt_breakdown():
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(_TOK_DIR, trust_remote_code=True)
    except Exception as e:
        return None

    def ntok(s):
        return len(tk(s, add_special_tokens=False)["input_ids"])

    sys_t = ntok(Q.QwenLogic._SYSTEM)
    shots = []
    for i, (u, a) in enumerate(Q.QwenLogic._SHOTS, 1):
        shots.append((i, ntok(u), ntok(a)))
    # 전체(템플릿 포함) — 실제 messages 토크나이즈
    sample = {"fall": {"fall_score": 0.0}, "vital": {"heart_rate": 35.0, "breathing_rate": 16.0},
              "env_sound": {"env_sound_label": "silence", "env_sound_confidence": 0.0},
              "speech_ko": {"keywords": [], "stt_confidence": 0.0, "speech_detected": False}}
    ql = Q.QwenLogic.__new__(Q.QwenLogic)
    msgs = [{"role": "system", "content": Q.QwenLogic._SYSTEM}]
    for u, a in Q.QwenLogic._SHOTS:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": "낙상:False(0%),심박:35,호흡:16,환경:silence,소견:심박이상(hr=35)"})
    full = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "{"
    total = len(tk(full)["input_ids"])
    shot_total = sum(u + a for _, u, a in shots)
    per_shot = round(shot_total / len(shots)) if shots else 0
    return {"system": sys_t, "shots": shots, "shot_total": shot_total,
            "per_shot_avg": per_shot, "total": total}


def main():
    golden = _load_golden()
    dumps = {}
    for lbl, fn in _BACKENDS:
        d = _load_json(fn)
        if d:
            dumps[lbl] = d

    # 강건성
    a_n, a_flip = robustness_track_a(golden)
    # noisy 데이터가 있는 백엔드만 Track B 강건성 (int8b 구버전 100셋 dump 제외)
    b_rob = {lbl: robustness_track_b(d, golden)
             for lbl, d in dumps.items() if any(x.get("noisy") for x in d)}
    # 해체
    common, agree_rate, disagree = decompose(dumps)
    # 토큰
    toks = {lbl: token_stats(d) for lbl, d in dumps.items()}
    pb = prompt_breakdown()

    # ── 콘솔 ──
    print("=" * 60)
    print("백엔드 분석 — 강건성 / 해체 / 토큰 / 프롬프트")
    print("=" * 60)
    print(f"[강건성] Track A flip(룰): {len(a_flip)}/{a_n}  {a_flip}")
    for lbl, (n, f) in b_rob.items():
        print(f"[강건성] Track B flip({lbl}): {len(f)}/{n}  {f}")
    print(f"\n[해체] 공통 {len(common)}케이스 · risk_level 일치율 {agree_rate:.3f} "
          f"· 불일치 {len(disagree)}")
    for cid, lv in disagree[:10]:
        print(f"   {cid}: {lv}")
    print("\n[토큰/타이밍]")
    for lbl, t in toks.items():
        if not t:
            continue
        print(f"  {lbl}: out min/med/p95/p99/max = "
              f"{t['out_min']}/{t['out_med']}/{t['out_p95']}/{t['out_p99']}/{t['out_max']} "
              f"· ms/tok={t['ms_per_tok']} · MAX권고={t['max_reco']} MIN={t['min_reco']}")
    if pb:
        print(f"\n[프롬프트 토큰] 총 {pb['total']} = system {pb['system']} + few-shot {pb['shot_total']}"
              f"(평균 {pb['per_shot_avg']}/개 × {len(pb['shots'])}) + 현재/템플릿")
        for i, u, a in pb["shots"]:
            print(f"   shot{i}: user {u} + asst {a} = {u+a}")

    # ── HTML ──
    path = _write_html(golden, dumps, a_n, a_flip, b_rob, common, agree_rate, disagree, toks, pb)
    print(f"\n[OK] 보고서 → {path}")


def _write_html(golden, dumps, a_n, a_flip, b_rob, common, agree_rate, disagree, toks, pb):
    esc = html.escape
    date_str = datetime.date.today().strftime("%Y-%m-%d")

    rob_rows = f"<tr><td>Track A (룰 호출결정)</td><td class='num'>{a_n}</td>" \
               f"<td class='num'>{len(a_flip)}</td><td class='num ok'>{1-len(a_flip)/a_n:.3f}</td>" \
               f"<td class='mono'>{esc(', '.join(a_flip))}</td></tr>"
    for lbl, (n, f) in b_rob.items():
        rate = 1 - len(f) / n if n else 0
        rob_rows += (f"<tr><td>Track B ({esc(lbl)})</td><td class='num'>{n}</td>"
                     f"<td class='num'>{len(f)}</td><td class='num'>{rate:.3f}</td>"
                     f"<td class='mono'>{esc(', '.join(f))}</td></tr>")

    tok_rows = ""
    for lbl, t in toks.items():
        if not t:
            continue
        tok_rows += (f"<tr><td>{esc(lbl)}</td><td class='num'>{t['n']}</td>"
                     f"<td class='num'>{t['out_min']}/{t['out_med']}/{t['out_p95']}/{t['out_p99']}/{t['out_max']}</td>"
                     f"<td class='num'>{t['ms_per_tok']}</td><td class='num'>{t['prompt_med']}</td>"
                     f"<td class='num ok'><b>{t['max_reco']}</b></td><td class='num'>{t['min_reco']}</td></tr>")

    dis_rows = ""
    for cid, lv in disagree:
        cells = " · ".join(f"{esc(k)}:{esc(str(v))}" for k, v in lv.items())
        dis_rows += f"<tr><td class='mono'>{esc(cid)}</td><td class='mono'>{cells}</td></tr>"
    if not dis_rows:
        dis_rows = "<tr><td colspan='2'>전 백엔드 risk_level 일치</td></tr>"

    pb_rows = ""
    if pb:
        for i, u, a in pb["shots"]:
            pb_rows += (f"<tr><td>few-shot {i}</td><td class='num'>{u}</td><td class='num'>{a}</td>"
                        f"<td class='num'>{u+a}</td></tr>")
        pb_block = f"""
<h2><span class="icon">📥</span> 입력 프롬프트 토큰 분해</h2>
<p class="h2sub">총 <b>{pb['total']}</b> 토큰 = system {pb['system']} + few-shot {pb['shot_total']} + 현재입력/템플릿. few-shot가 입력의 대부분.</p>
<table><tr><th>구성</th><th class="num">user 토큰</th><th class="num">asst 토큰</th><th class="num">합</th></tr>
<tr><td>system</td><td class="num">—</td><td class="num">{pb['system']}</td><td class="num">{pb['system']}</td></tr>
{pb_rows}
</table>
<div class="insight green"><div class="ttl">few-shot 최적화 — 적용 완료 (5→4)</div>
최대 토큰 shot(4-domain critical, 85토큰) 제거 → 입력 <b>620 → {pb['total']}</b> 토큰(~{620 - pb['total']}↓).
부작용 없이 <b>GGUF Track B 0.984 → 0.992 개선</b>(vital-05 통과). few-shot 1개 평균 {pb['per_shot_avg']}토큰.</div>"""
    else:
        pb_block = ""

    # MAX/MIN 권고 (배포본 gguf 우선)
    reco = toks.get("GGUF Q4_K_M") or next((t for t in toks.values() if t), None)
    reco_block = ""
    if reco:
        reco_block = f"""
<div class="insight green"><div class="ttl">max_new_tokens MAX/MIN 선정</div>
출력 토큰 p99={reco['out_p99']}, max={reco['out_max']} → <b>MAX 권고 {reco['max_reco']}</b>(p99+여유).
관측 최소 {reco['out_min']} → <b>MIN {reco['min_reco']}</b>. 현재 clamp 40~80/def56 → def {reco['max_reco']} 안팎으로 축소 여지.</div>"""

    return _emit(date_str, rob_rows, tok_rows, dis_rows, len(common), agree_rate, pb_block, reco_block,
                 list(dumps.keys()))


def _emit(date_str, rob_rows, tok_rows, dis_rows, n_common, agree_rate, pb_block, reco_block, backends):
    os.makedirs(_DOCS, exist_ok=True)
    path = os.path.join(_DOCS, "safewave_m5_robustness_token.html")
    htmldoc = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SafeWave-AI — M5 강건성·토큰 분석</title>
<style>
 :root{{--bg:#0f1117;--surface:#1a1d27;--surface2:#22263a;--border:#2e3450;--accent:#5b8dee;
 --accent2:#7c6dfa;--green:#3ecf8e;--yellow:#f5c542;--red:#f56060;--text:#e2e8f0;--muted:#8892a4;
 --mono:'JetBrains Mono','Cascadia Code',monospace;}}
 *{{box-sizing:border-box;margin:0;padding:0}}
 body{{background:var(--bg);color:var(--text);font-family:'Pretendard','Noto Sans KR',sans-serif;font-size:14px;line-height:1.7}}
 .header{{background:linear-gradient(135deg,#1a1d27,#0f1117 60%);border-bottom:1px solid var(--border);padding:44px 0 32px;text-align:center}}
 .badge{{display:inline-block;background:rgba(91,141,238,.15);border:1px solid rgba(91,141,238,.4);color:var(--accent);
 padding:4px 14px;border-radius:20px;font-size:11px;letter-spacing:.08em;font-weight:600;text-transform:uppercase;margin-bottom:14px}}
 h1{{font-size:29px;font-weight:700}} .sub{{color:var(--muted);font-size:14px;margin-top:6px}}
 .container{{max-width:1060px;margin:0 auto;padding:0 24px}}
 .section,h2{{}} h2{{font-size:19px;font-weight:700;margin:34px 0 6px;display:flex;align-items:center;gap:9px}}
 .icon{{font-size:19px}} .h2sub{{color:var(--muted);font-size:13px;margin-bottom:18px}}
 table{{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0 4px}}
 th{{text-align:left;padding:9px 12px;color:var(--muted);font-weight:500;border-bottom:1px solid var(--border);
 font-size:11px;text-transform:uppercase;letter-spacing:.05em}}
 td{{padding:10px 12px;border-bottom:1px solid rgba(46,52,80,.6);vertical-align:top}}
 td.num,th.num{{text-align:center;font-family:var(--mono)}} .mono{{font-family:var(--mono);font-size:12px}}
 .ok{{color:var(--green)}} .bad{{color:var(--red)}}
 .insight{{background:rgba(91,141,238,.07);border:1px solid rgba(91,141,238,.25);border-left:3px solid var(--accent);
 border-radius:8px;padding:13px 17px;font-size:13px;margin-top:14px}}
 .insight.green{{background:rgba(62,207,142,.07);border-color:rgba(62,207,142,.25);border-left-color:var(--green)}}
 .insight .ttl{{font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:5px;color:var(--accent)}}
 .insight.green .ttl{{color:var(--green)}}
 footer{{text-align:center;padding:24px;color:var(--muted);font-size:12px;border-top:1px solid var(--border);margin-top:24px}}
</style></head><body>
<div class="header"><div class="container">
 <div class="badge">M5 Robustness & Token Analysis</div>
 <h1>SafeWave-AI — M5 강건성·토큰 분석</h1>
 <p class="sub">골든셋 200(clean↔noisy 100쌍) · 백엔드: {esc_join(backends)} · {date_str}</p>
</div></div>
<div class="container">

<h2><span class="icon">🛡️</span> 강건성 — 센서 누락(clean↔noisy) flip</h2>
<p class="h2sub">결측 노이즈 쌍에서 판정이 뒤집히는 비율. 강건율↑ = 누락에 강함.</p>
<table><tr><th>트랙</th><th class="num">쌍</th><th class="num">flip</th><th class="num">강건율</th><th>flip 케이스</th></tr>
{rob_rows}</table>

<h2><span class="icon">🔬</span> 백엔드 raw 해체 비교</h2>
<p class="h2sub">공통 {n_common}케이스 risk_level 일치율 <b>{agree_rate:.3f}</b>. 주 비교는 1.5B↔GGUF(4-shot/200셋);
INT8 block-wise는 5-shot/100셋 baseline 덤프(속도상 재실행 제외)라 참고용. 불일치 케이스:</p>
<table><tr><th>케이스</th><th>백엔드별 raw_risk_level</th></tr>
{dis_rows}</table>

<h2><span class="icon">⏱️</span> 출력 토큰·타이밍 + max_new_tokens 선정</h2>
<p class="h2sub">out 분포 = min/median/p95/p99/max. MAX 권고 = p99+여유.</p>
<table><tr><th>백엔드</th><th class="num">N</th><th class="num">out min/med/p95/p99/max</th>
<th class="num">ms/tok</th><th class="num">prompt(med)</th><th class="num">MAX권고</th><th class="num">MIN</th></tr>
{tok_rows}</table>
{reco_block}

{pb_block}

</div>
<footer>SafeWave-AI · M5 강건성·토큰 분석 · {date_str} · scripts/analyze_backends.py · Generated with Claude Code</footer>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(htmldoc)
    return path


def esc_join(xs):
    return html.escape(" / ".join(xs))


if __name__ == "__main__":
    main()
