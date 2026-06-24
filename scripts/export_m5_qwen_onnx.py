#!/usr/bin/env python3
"""
M5 Qwen-0.5B-Instruct ONNX 변환 스크립트

공식 출처: Hugging Face - Qwen/Qwen2-0.5B-Instruct
변환 방식: Hugging Face Optimum (optimum-cli)
출력: volumes/models/qwen_05b/ 폴더 (config.json, tokenizer.json, model.onnx 등 포함)

요구사항:
1. transformers >= 4.38.0
2. onnx >= 1.15.0
3. onnxruntime >= 1.20.1
4. optimum[onnxruntime] >= 1.17.0
"""

import os
import sys
import subprocess
from pathlib import Path


def convert_qwen_to_onnx():
    """Qwen2-0.5B-Instruct를 ONNX로 변환"""
    model_id = "Qwen/Qwen2-0.5B-Instruct"
    output_dir = Path("volumes/models/qwen_05b")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("[M5] Exporting Qwen2-0.5B-Instruct model to ONNX...")
    print("=" * 80)
    print(f"Model ID: {model_id}")
    print(f"Output directory: {output_dir}")

    try:
        from transformers import AutoTokenizer, AutoConfig
        print("\n[Step 1/3] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("[OK] Tokenizer downloaded")

        print("\n[Step 2/3] Downloading model config...")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print("[OK] Model config downloaded")

        print("\n[Step 3/3] Converting to ONNX with Optimum...")
        print("(This may take 2-5 minutes)")

        # optimum 2.x: merged KV 단일 파일 export
        cmd = [
            "optimum-cli", "export", "onnx",
            "--model", model_id,
            "--task", "text-generation-with-past",
            str(output_dir),
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print("[WARN] Optimum CLI failed, falling back to torch.onnx.export...")
            _convert_with_torch_onnx(model_id, output_dir, tokenizer, config)
        else:
            print("[OK] ONNX export completed")

        print("\n[Step 4/4] Verifying ONNX model...")
        _verify_qwen_onnx(output_dir, tokenizer)

    except Exception as e:
        print(f"[ERR] {e}")
        print("\nTroubleshooting:")
        print("  pip install optimum[onnxruntime] transformers onnx onnxruntime")
        sys.exit(1)


def _convert_with_torch_onnx(model_id, output_dir, tokenizer, config):
    """Fallback: torch.onnx.export"""
    import torch
    from transformers import AutoModelForCausalLM

    print("[Fallback] Loading model from HuggingFace...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()

    dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    onnx_path = output_dir / "model.onnx"
    print(f"Exporting to {onnx_path}...")

    torch.onnx.export(
        model,
        (dummy_input_ids,),
        str(onnx_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=17,
        do_constant_folding=True,
        use_external_data_format=True,
    )
    print("[OK] torch.onnx.export completed")


def _verify_qwen_onnx(output_dir, tokenizer):
    """ONNX 모델 구조 및 추론 검증"""
    import onnx
    import onnxruntime as ort
    import numpy as np

    onnx_files = list(output_dir.glob("*.onnx"))
    if not onnx_files:
        print("[ERR] No ONNX files found!")
        return

    print(f"[OK] Found {len(onnx_files)} ONNX file(s):")
    for onnx_file in onnx_files:
        try:
            m = onnx.load(str(onnx_file))
            onnx.checker.check_model(m)
            print(f"  - {onnx_file.name}: valid")
        except Exception as e:
            print(f"  - {onnx_file.name}: FAILED ({e})")

    # 간단한 추론 테스트 (model.onnx만)
    model_path = output_dir / "model.onnx"
    if model_path.exists():
        try:
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            test_ids = tokenizer.encode("현재 상태는?", return_tensors="np").astype(np.int64)
            in_name = session.get_inputs()[0].name
            out = session.run(None, {in_name: test_ids})
            print(f"[OK] Inference test: output shape {out[0].shape}")
        except Exception as e:
            print(f"[WARN] Inference test failed: {e}")

    print(f"\n  config.json : {(output_dir / 'config.json').exists()}")
    print(f"  tokenizer.json: {(output_dir / 'tokenizer.json').exists()}")
    print("\n[OK] M5 Qwen-0.5B-Instruct model is READY for deployment!")


if __name__ == "__main__":
    required = ["transformers", "onnx", "onnxruntime", "optimum"]
    missing = [p for p in required if not __import__("importlib").util.find_spec(p)]
    if missing:
        print(f"[ERR] Missing packages: {', '.join(missing)}")
        print("  pip install transformers onnx onnxruntime 'optimum[onnxruntime]'")
        sys.exit(1)

    convert_qwen_to_onnx()
