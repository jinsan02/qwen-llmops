"""
출력 필드 순서 A/B 재채점 — answer-first(A) vs reason-first(B) raw dump를 골든셋과 id로 조인해 채점한다.

원본 dump는 reports/(gitignore)에만 있다. 모델을 다시 돌리지 않고 저장된 raw reason·등급만 다시 채점한다.
채점 함수는 하니스(eval/eval_qwen_reasoning.py)의 것을 그대로 쓴다.

- strict   : 4기준 모두 통과. numeric_match는 기대 위기값 ±5% 인용만 인정(grounded 크레딧 없음)
- 기준별 실패는 케이스마다 독립으로 센다(다중 라벨).
- 놓친 위기 수치: A에서 strict numeric_match 실패한 케이스 중 B에서 통과한 건수

  python scripts/rescore_ab.py
  python scripts/rescore_ab.py --a reports/ab_A_current.json --b reports/ab_B_reasonfirst.json
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from eval.eval_qwen_reasoning import (
    _score_format_complete, _score_label_consistency,
    _score_numeric_match, _score_vital_override,
)

_CRITS = ("numeric_match", "label_consistency", "vital_override", "format_complete")


def _load_golden(path: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["id"]] = d
    return out


def _score(dump_path: str, golden: dict) -> dict:
    """id → 4기준 strict 판정(dict[str,bool])."""
    with open(dump_path, encoding="utf-8") as f:
        items = json.load(f)
    res = {}
    for it in items:
        gold = golden.get(it["id"])
        if gold is None:
            continue
        exp = gold["expected"]
        reason = it.get("raw_reason", "") or ""
        level = (it.get("raw_risk_level", "") or "").lower()
        res[it["id"]] = {
            "numeric_match": _score_numeric_match(reason, exp, None, "")[0],   # inp=None → strict
            "label_consistency": _score_label_consistency(reason, exp)[0],
            "vital_override": _score_vital_override(reason, level, exp)[0],
            "format_complete": _score_format_complete(reason)[0],
        }
    return res


def main():
    ap = argparse.ArgumentParser(description="출력 필드 순서 A/B 재채점(strict)")
    ap.add_argument("--a", default=os.path.join(ROOT, "reports", "ab_A_current.json"), help="answer-first dump")
    ap.add_argument("--b", default=os.path.join(ROOT, "reports", "ab_B_reasonfirst.json"), help="reason-first dump")
    ap.add_argument("--golden", default=os.path.join(ROOT, "data", "qwen_golden_set.jsonl"))
    args = ap.parse_args()

    golden = _load_golden(args.golden)
    A, B = _score(args.a, golden), _score(args.b, golden)
    ids = sorted(set(A) & set(B))

    for name, D in (("A answer-first", A), ("B reason-first", B)):
        passed = sum(1 for i in ids if all(D[i].values()))
        fails = {c: sum(1 for i in ids if not D[i][c]) for c in _CRITS}
        print(f"{name}: strict {passed}/{len(ids)} = {passed / len(ids):.3f}  기준별 실패 {fails}")

    missed = [i for i in ids if not A[i]["numeric_match"]]
    caught = [i for i in missed if B[i]["numeric_match"]]
    print(f"놓친 위기 수치: A에서 strict 수치 인용 실패 {len(missed)}건 중 B가 인용 {len(caught)}건 {caught}")
    print(f"format 실패: A {sum(1 for i in ids if not A[i]['format_complete'])} → "
          f"B {sum(1 for i in ids if not B[i]['format_complete'])}")


if __name__ == "__main__":
    main()
