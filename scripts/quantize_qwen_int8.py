"""
Qwen-0.5B ONNX int8 dynamic quantization 스켈레톤 (Task C)

실행 전 Task A 골든셋으로 fp32 기준선 측정 후 이 스크립트를 실행,
eval_qwen_reasoning.py --model <quantized_path> 으로 품질 비교.

요구: optimum >= 1.17, onnxruntime >= 1.20

실행:
  python scripts/quantize_qwen_int8.py \
    --model volumes/models/qwen_05b \
    --output volumes/models/qwen_05b_int8
"""

import argparse
import os
import sys


def quantize(model_path: str, output_path: str) -> None:
    try:
        from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError:
        print("[ERR] optimum[onnxruntime] 미설치. pip install 'optimum[onnxruntime]'")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"[ERR] 모델 경로 없음: {model_path}")
        sys.exit(1)

    print(f"[INFO] 양자화 소스: {model_path}")
    print(f"[INFO] 출력 경로:   {output_path}")
    os.makedirs(output_path, exist_ok=True)

    # fp32 모델 로드
    model = ORTModelForCausalLM.from_pretrained(model_path, export=False)

    # int8 dynamic quantization 설정
    # avx512_vnni: RPi5(ARM64)는 지원 안 함 → arm64 or avx2 사용
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

    quantizer = ORTQuantizer.from_pretrained(model)
    quantizer.quantize(
        save_dir=output_path,
        quantization_config=qconfig,
    )

    print(f"[OK] 양자화 완료: {output_path}")
    _compare_size(model_path, output_path)


def _compare_size(src: str, dst: str) -> None:
    def _dir_mb(path: str) -> float:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".onnx"):
                    total += os.path.getsize(os.path.join(root, f))
        return total / (1024 * 1024)

    src_mb = _dir_mb(src)
    dst_mb = _dir_mb(dst)
    ratio  = (1 - dst_mb / src_mb) * 100 if src_mb > 0 else 0
    print(f"[INFO] 모델 크기: fp32={src_mb:.1f}MB → int8={dst_mb:.1f}MB ({ratio:.1f}% 감소)")
    print()
    print("다음 단계:")
    print(f"  python eval/eval_qwen_reasoning.py --model {dst} --golden data/qwen_golden_set.jsonl")
    print(f"  python scripts/benchmark_rpi5_qwen.py  --model {dst}")


def main():
    parser = argparse.ArgumentParser(description="Qwen ONNX int8 dynamic quantization")
    parser.add_argument("--model",  default="volumes/models/qwen_05b")
    parser.add_argument("--output", default="volumes/models/qwen_05b_int8")
    args = parser.parse_args()
    quantize(args.model, args.output)


if __name__ == "__main__":
    main()
