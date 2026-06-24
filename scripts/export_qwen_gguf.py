"""
M5 GGUF 변환 — Qwen2.5-1.5B → Q4_K_M (llama.cpp).

대전제 변경: M5는 llama.cpp(GGUF)를 추론 백엔드로 허용. ai-qwen 컨테이너 격리로
M1~M4 ONNX 파이프라인은 영향 없음.

두 가지 경로:
  download : HF 공식 prebuilt Q4_K_M 내려받기 (의존성 최소, 권장 — 검증에 사용)
  convert  : 로컬 HF 가중치 → f16 GGUF → Q4_K_M 풀 변환 (재현/커스텀 가중치용)

실행:
  # 공식 prebuilt (가장 단순)
  python scripts/export_qwen_gguf.py download \
      --repo Qwen/Qwen2.5-1.5B-Instruct-GGUF \
      --file qwen2.5-1.5b-instruct-q4_k_m.gguf \
      --out volumes/models/qwen_15b_gguf

  # 풀 변환 (llama.cpp 필요: LLAMA_CPP_DIR에 convert_hf_to_gguf.py + llama-quantize)
  LLAMA_CPP_DIR=/path/to/llama.cpp \
  python scripts/export_qwen_gguf.py convert \
      --hf Qwen/Qwen2.5-1.5B-Instruct \
      --out volumes/models/qwen_15b_gguf \
      --qtype Q4_K_M
"""

import argparse
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download(repo: str, fname: str, out_dir: str) -> str:
    from huggingface_hub import hf_hub_download
    os.makedirs(out_dir, exist_ok=True)
    print(f"[download] {repo} :: {fname}")
    p = hf_hub_download(repo, fname, local_dir=out_dir)
    print(f"[OK] {p} ({os.path.getsize(p) // (1024 * 1024)} MB)")
    return p


def convert(hf_model: str, out_dir: str, qtype: str) -> str:
    """HF 가중치 → f16 GGUF → 양자화(qtype). llama.cpp 도구 필요."""
    llama_dir = os.getenv("LLAMA_CPP_DIR")
    if not llama_dir or not os.path.isdir(llama_dir):
        sys.exit("[ERR] LLAMA_CPP_DIR 미설정. llama.cpp 레포(convert_hf_to_gguf.py + "
                 "빌드된 llama-quantize) 경로를 지정하세요.")
    convert_py = os.path.join(llama_dir, "convert_hf_to_gguf.py")
    quantize_bin = next((os.path.join(llama_dir, b) for b in
                         ("build/bin/llama-quantize", "build/bin/llama-quantize.exe",
                          "llama-quantize", "llama-quantize.exe")
                         if os.path.exists(os.path.join(llama_dir, b))), None)
    if not os.path.exists(convert_py):
        sys.exit(f"[ERR] convert_hf_to_gguf.py 없음: {convert_py}")
    if not quantize_bin:
        sys.exit("[ERR] llama-quantize 바이너리 없음 (llama.cpp 빌드 필요)")

    os.makedirs(out_dir, exist_ok=True)
    f16 = os.path.join(out_dir, "qwen2.5-1.5b-f16.gguf")
    q_out = os.path.join(out_dir, f"qwen2.5-1.5b-{qtype.lower()}.gguf")

    # 1) HF → f16 GGUF
    print(f"[convert] {hf_model} → {f16}")
    subprocess.run([sys.executable, convert_py, hf_model,
                    "--outfile", f16, "--outtype", "f16"], check=True)
    # 2) f16 → Q4_K_M 등
    print(f"[quantize] {f16} → {q_out} ({qtype})")
    subprocess.run([quantize_bin, f16, q_out, qtype], check=True)

    print(f"[OK] {q_out} ({os.path.getsize(q_out) // (1024 * 1024)} MB)")
    print("\n검증:")
    print(f"  python eval/eval_qwen_reasoning.py --impl gguf --model {out_dir} "
          f"--tokenizer volumes/models/qwen_15b --golden data/qwen_golden_set.jsonl")
    return q_out


def main():
    ap = argparse.ArgumentParser(description="Qwen2.5-1.5B → GGUF Q4_K_M")
    sub = ap.add_subparsers(dest="method", required=True)

    d = sub.add_parser("download", help="HF 공식 prebuilt GGUF 다운로드")
    d.add_argument("--repo", default="Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    d.add_argument("--file", default="qwen2.5-1.5b-instruct-q4_k_m.gguf")
    d.add_argument("--out", default=os.path.join(_ROOT, "volumes/models/qwen_15b_gguf"))

    c = sub.add_parser("convert", help="로컬 HF 가중치 풀 변환 (llama.cpp 필요)")
    c.add_argument("--hf", default="Qwen/Qwen2.5-1.5B-Instruct")
    c.add_argument("--out", default=os.path.join(_ROOT, "volumes/models/qwen_15b_gguf"))
    c.add_argument("--qtype", default="Q4_K_M")

    args = ap.parse_args()
    if args.method == "download":
        download(args.repo, args.file, args.out)
    else:
        convert(args.hf, args.out, args.qtype)


if __name__ == "__main__":
    main()
