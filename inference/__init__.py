from inference.emergency_score import compute_emergency_score

__all__ = ["compute_emergency_score", "QwenLogic"]


def __getattr__(name):
    # QwenLogic 지연 로드 — emergency_score(순수 numpy)를 onnxruntime 없이 import 가능하게.
    # (하위호환: `from inference import QwenLogic`는 그대로 동작)
    if name == "QwenLogic":
        from inference.qwen_05b import QwenLogic
        return QwenLogic
    raise AttributeError(f"module 'inference' has no attribute {name!r}")
