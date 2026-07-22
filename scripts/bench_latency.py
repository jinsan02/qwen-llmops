"""
M5 추론 latency 벤치 — prefill/decode 분리 + 코어 제한 + RPi5 투영.

노트북 x86에 성능 제한(스레드 수)을 걸어 RPi5(4×Cortex-A76)와 유사 환경을 구성하고,
입력 토큰당(prefill)·출력 토큰당(decode) 시간을 측정한다. published RPi5 계수로 환산 투영.

주의: x86↔ARM은 ISA·메모리 대역폭이 달라 절대값은 다르다. 본 스크립트는
  (a) 코어 수 제한 하의 실측 + (b) 보정계수 투영을 제공(상대 비교·견적용).

실행:
  python scripts/bench_latency.py --threads 4 --iters 5
  python scripts/bench_latency.py --sweep 1,2,4,8
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MODEL = "volumes/models/qwen_15b_gguf_q5/qwen2.5-1.5b-instruct-q5_k_m.gguf"
_TOK = "volumes/models/qwen_15b"

# published RPi5(8GB) 1.5B Q-quant 계수 (x86 대비 느린 배수). 측정 환경에 맞게 --rpi5-* 조정.
_RPI5_DECODE_FACTOR = 3.6   # 노트북 decode ~36 tok/s → RPi5 ~10 tok/s
_RPI5_PREFILL_FACTOR = 2.4  # 노트북 prefill ~159 tok/s → RPi5 ~66 tok/s


def _build_prompt():
    from transformers import AutoTokenizer
    from inference.qwen_15b import QwenLogic
    tok = AutoTokenizer.from_pretrained(_TOK, trust_remote_code=True)
    q = QwenLogic.__new__(QwenLogic)
    inp = {"fall": {"fall_score": 0.0}, "vital": {"heart_rate": 105, "breathing_rate": 36},
           "env_sound": {"env_sound_label": "silence"}, "speech_ko": {"keywords": []}}
    ts = [{"m": i - 29, "hr": int(79 + (105 - 79) * i / 29), "rr": int(18 + (36 - 18) * i / 29)} for i in range(30)]
    msgs = q._build_messages(inp, time_series=ts)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "{"
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    return prompt, n_prompt


def _bench(threads, iters, prompt, n_prompt):
    from llama_cpp import Llama
    llm = Llama(model_path=_MODEL, n_ctx=2048, n_threads=threads, verbose=False)
    stop = ["<|im_end|>", "<|endoftext|>"]
    # 워밍업
    llm.reset(); llm(prompt, max_tokens=8, temperature=0.0, top_k=1, stop=stop)

    t1s, tNs, outN = [], [], []
    N = 40
    for _ in range(iters):
        # 전체 prefill 강제(KV 리셋) + 1토큰 → prefill + 1 decode
        llm.reset()
        t0 = time.perf_counter()
        llm(prompt, max_tokens=1, temperature=0.0, top_k=1, stop=stop)
        t1s.append(time.perf_counter() - t0)
        # 전체 prefill + N decode
        llm.reset()
        t0 = time.perf_counter()
        o = llm(prompt, max_tokens=N, temperature=0.0, top_k=1, stop=stop)
        tNs.append(time.perf_counter() - t0)
        outN.append(max(1, o["usage"]["completion_tokens"]))

    def med(a):
        a = sorted(a); return a[len(a) // 2]
    t1, tN, nout = med(t1s), med(tNs), med(outN)
    decode_ms = (tN - t1) * 1000.0 / max(1, nout - 1)     # 출력 토큰당
    prefill_total_ms = (t1 * 1000.0) - decode_ms          # prefill = t1 - 1 decode
    prefill_ms = prefill_total_ms / n_prompt              # 입력 토큰당
    return {"prefill_ms_tok": prefill_ms, "decode_ms_tok": decode_ms,
            "prefill_total_ms": prefill_total_ms, "decode_tps": 1000.0 / decode_ms,
            "prefill_tps": 1000.0 / prefill_ms, "out_tokens": nout,
            "total_ms": prefill_total_ms + decode_ms * nout}


def _report(threads, r, n_prompt, df, pf):
    print(f"\n=== n_threads={threads} (입력 {n_prompt}토큰, 출력 {r['out_tokens']}토큰) ===")
    print(f"  [측정 x86] prefill {r['prefill_ms_tok']:.2f} ms/입력토큰 ({r['prefill_tps']:.0f} tok/s) · "
          f"decode {r['decode_ms_tok']:.2f} ms/출력토큰 ({r['decode_tps']:.1f} tok/s)")
    print(f"             위기케이스 총지연 ≈ {r['total_ms']/1000:.2f}s "
          f"(prefill {r['prefill_total_ms']/1000:.2f}s + decode {r['decode_ms_tok']*r['out_tokens']/1000:.2f}s)")
    rpi_decode = r["decode_ms_tok"] * df
    rpi_prefill_total = r["prefill_total_ms"] * pf
    rpi_total = rpi_prefill_total + rpi_decode * r["out_tokens"]
    print(f"  [RPi5 투영] ×decode{df}/prefill{pf} → decode {rpi_decode:.1f} ms/tok ({1000/rpi_decode:.1f} tok/s) · "
          f"총지연 ≈ {rpi_total/1000:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sweep", default=None, help="콤마 구분 스레드 수 (예: 1,2,4,8)")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--rpi5-decode", type=float, default=_RPI5_DECODE_FACTOR)
    ap.add_argument("--rpi5-prefill", type=float, default=_RPI5_PREFILL_FACTOR)
    args = ap.parse_args()

    prompt, n_prompt = _build_prompt()
    print(f"[INFO] 프롬프트 {n_prompt}토큰, iters={args.iters}, RPi5 계수 decode×{args.rpi5_decode}/prefill×{args.rpi5_prefill}")
    threads_list = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.threads]
    for th in threads_list:
        r = _bench(th, args.iters, prompt, n_prompt)
        _report(th, r, n_prompt, args.rpi5_decode, args.rpi5_prefill)


if __name__ == "__main__":
    main()
