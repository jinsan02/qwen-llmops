"""
Qwen ONNX block-wise INT8 양자화 (호환 기법).

배경: optimum dynamic INT8(per-tensor/per-channel)은 Qwen2.5-1.5B 생성을 붕괴시킴
(raw 통과율 0.000, 출력층 logit 손상으로 토큰 반복). 해결책:

  MatMulNBitsQuantizer(bits=8, block_size=32) — GGUF k-quant와 동형의 블록 단위
  가중치 양자화. 블록별 scale로 오차를 국소화해 품질 보존. 추가로 출력층 /lm_head/MatMul을
  제외해 logit 손상을 차단.

요구: onnxruntime >= 1.20 (MatMulNBitsQuantizer)

실행:
  python scripts/quantize_qwen_int8_blockwise.py \
    --model volumes/models/qwen_15b \
    --output volumes/models/qwen_15b_int8b \
    --bits 8 --block-size 32
"""

import argparse
import os
import shutil
import sys


def quantize(model_path: str, output_path: str, bits: int, block_size: int) -> None:
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    src_onnx = os.path.join(model_path, "model.onnx")
    if not os.path.exists(src_onnx):
        sys.exit(f"[ERR] model.onnx 없음: {src_onnx}")
    os.makedirs(output_path, exist_ok=True)

    print(f"[INFO] 로드(외부 데이터 포함): {src_onnx}")
    model = onnx.load(src_onnx, load_external_data=True)

    # 출력층(lm_head)은 제외 — 양자화 시 logit 손상으로 생성 붕괴
    exclude = ["/lm_head/MatMul"]
    print(f"[INFO] block-wise INT{bits} (block_size={block_size}), 제외: {exclude}")
    quant = MatMulNBitsQuantizer(
        model,
        bits=bits,
        block_size=block_size,
        is_symmetric=True,
        nodes_to_exclude=exclude,
    )
    quant.process()

    out_onnx = os.path.join(output_path, "model.onnx")
    quant.model.save_model_to_file(out_onnx, use_external_data_format=True)
    print(f"[OK] 저장: {out_onnx}")

    # 토크나이저/설정 파일 복사 (추론에 필요)
    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
               "merges.txt", "added_tokens.json", "special_tokens_map.json",
               "chat_template.jinja", "generation_config.json"):
        sp = os.path.join(model_path, fn)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(output_path, fn))

    _report_size(src_onnx, output_path)


def _report_size(src_onnx: str, dst_dir: str) -> None:
    def _onnx_total(*paths):
        total = 0
        for p in paths:
            if os.path.isfile(p):
                total += os.path.getsize(p)
            elif os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for f in files:
                        if f.endswith(".onnx") or f.endswith(".onnx_data") or f.endswith(".onnx.data"):
                            total += os.path.getsize(os.path.join(root, f))
        return total / (1024 * 1024)

    src_dir = os.path.dirname(src_onnx)
    src_mb = _onnx_total(src_dir)
    dst_mb = _onnx_total(dst_dir)
    ratio = (1 - dst_mb / src_mb) * 100 if src_mb else 0
    print(f"[INFO] 크기: fp32={src_mb:.0f}MB → int8b={dst_mb:.0f}MB ({ratio:.0f}% 감소)")
    print("\n검증:")
    print(f"  python eval/eval_qwen_reasoning.py --impl 15b --model {dst_dir} "
          f"--golden data/qwen_golden_set.jsonl")


def main():
    ap = argparse.ArgumentParser(description="Qwen ONNX block-wise INT8 (MatMulNBits)")
    ap.add_argument("--model", default="volumes/models/qwen_15b")
    ap.add_argument("--output", default="volumes/models/qwen_15b_int8b")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=32)
    args = ap.parse_args()
    quantize(args.model, args.output, args.bits, args.block_size)


if __name__ == "__main__":
    main()
