"""
service/qwen_service.py
M5 Qwen-0.5B 전용 서비스 루프.

ai-experts 컨테이너가 ai:result에 slm_needed=True로 기록한 항목을 소비하고
Qwen 추론 결과를 ai:emergency에 XADD한다.
"""

import json
import logging
import os
import time

import redis as _redis

# 추론 백엔드 선택: SLM_BACKEND = 05b | 15b | gguf
# (대전제 변경: gguf는 llama.cpp 백엔드. M5는 ai-qwen 컨테이너 격리라 M1~M4 ONNX 무관)
_SLM_BACKEND = os.getenv("SLM_BACKEND", "05b").lower()
if _SLM_BACKEND == "gguf":
    from inference.qwen_gguf import QwenLogic
elif _SLM_BACKEND == "15b":
    from inference.qwen_15b import QwenLogic
else:
    from inference.qwen_05b import QwenLogic
from inference.utils import (
    stream_id_ts_ms as _stream_id_ts_ms,
    json_loads as _json_loads,
    build_context_window,
)


RESULT_STREAM    = "ai:result"
EMERGENCY_STREAM = "ai:emergency"
EMERGENCY_STREAM_MAXLEN = int(os.getenv("EMERGENCY_STREAM_MAXLEN", "3600"))
SLM_MIN_INTERVAL_MS     = int(os.getenv("SLM_MIN_INTERVAL_MS", "5000"))
CONTEXT_WINDOW_MINUTES  = int(os.getenv("CONTEXT_WINDOW_MINUTES", "10"))
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")
SLM_MODEL  = os.getenv("SLM_MODEL",  "qwen_05b")
# gguf 백엔드용 chat_template/tokenizer 폴더 (models/ 하위)
SLM_TOKENIZER = os.getenv("SLM_TOKENIZER", "qwen_15b")

LOGGER = logging.getLogger("qwen_llmops.qwen_svc")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False


def _log(level: int, event: str, **fields):
    payload = {"service": "ai-qwen", "event": event, "ts_ms": int(time.time() * 1000)}
    payload.update(fields)
    LOGGER.log(level, json.dumps(payload, ensure_ascii=False))


def _connect_redis() -> _redis.Redis:
    host = os.getenv("REDIS_HOST", "db")
    port = int(os.getenv("REDIS_PORT", 6379))
    while True:
        try:
            r = _redis.Redis(host=host, port=port, decode_responses=False,
                             socket_connect_timeout=3)
            r.ping()
            _log(logging.INFO, "redis_connected", host=host, port=port)
            return r
        except _redis.exceptions.ConnectionError as exc:
            _log(logging.WARNING, "redis_not_ready", error=str(exc), retry_in_sec=2)
            time.sleep(2)


def _build_context_window(r: _redis.Redis, ts_ms: int) -> dict:
    return build_context_window(r, ts_ms, EMERGENCY_STREAM, CONTEXT_WINDOW_MINUTES)


def _warmup_qwen(qwen: QwenLogic):
    """ONNX Runtime 내부 버퍼/스레드 사전 초기화."""
    dummy = {
        "fall":      {"fall_score": 0.0, "infer_confidence": 0.5},
        "vital":     {"heart_rate": 72.0, "breathing_rate": 16.0, "infer_confidence": 0.5},
        "env_sound": {"label": "silence", "confidence": 0.0, "infer_confidence": 0.0},
        "speech_ko": {"keywords": [], "stt_confidence": 0.0, "speech_detected": False, "infer_confidence": 0.0},
    }
    try:
        qwen.evaluate(dummy)
        _log(logging.INFO, "qwen_warmup_completed")
    except Exception as exc:
        _log(logging.WARNING, "qwen_warmup_failed", error=str(exc))


def _write_emergency(r: _redis.Redis, snapshot: dict, fused: dict):
    entry = {
        "ts_ms":                  snapshot.get("ts_ms"),
        "node_id":                snapshot.get("node_id", 0),
        "risk_score":             snapshot.get("risk_score", 0.0),
        "risk_level":             fused.get("risk_level", snapshot.get("risk_level", "warning")),
        "emergency":              fused.get("emergency", snapshot.get("emergency", False)),
        "qwen_reason":            fused.get("qwen_reason"),
        "slm_invoked":            True,
        "is_outlier":             bool(fused.get("is_outlier", False)),
        "correlated_with_history": bool(fused.get("correlated_with_history", False)),
        "slm_mode":               fused.get("slm_mode"),
        "summary":                fused.get("summary") or fused.get("qwen_reason", ""),
        "emergency_breakdown":    snapshot.get("emergency_breakdown"),
    }
    r.xadd(
        EMERGENCY_STREAM,
        {"data": json.dumps(entry, ensure_ascii=False)},
        maxlen=EMERGENCY_STREAM_MAXLEN,
        approximate=True,
    )


def run():
    r = _connect_redis()
    if _SLM_BACKEND == "gguf":
        qwen = QwenLogic(os.path.join(MODEL_PATH, SLM_MODEL),
                         tokenizer_dir=os.path.join(MODEL_PATH, SLM_TOKENIZER))
    else:
        qwen = QwenLogic(os.path.join(MODEL_PATH, SLM_MODEL))
    qwen.redis_client = r
    _warmup_qwen(qwen)

    # 서비스 기동 시점 이후 항목만 소비 (백로그 무시)
    latest = r.xrevrange(RESULT_STREAM, count=1)
    last_id = latest[0][0] if latest else b"0-0"

    last_invoked_ms: int = 0
    _log(logging.INFO, "qwen_service_started", stream=RESULT_STREAM)

    while True:
        try:
            entries = r.xread({RESULT_STREAM: last_id}, count=5, block=2000)
            if not entries:
                continue

            for _, messages in entries:
                for msg_id, fields in messages:
                    last_id = msg_id

                    if fields.get(b"slm_needed") != b"True":
                        continue

                    now_ms = int(time.time() * 1000)
                    if now_ms - last_invoked_ms < SLM_MIN_INTERVAL_MS:
                        continue

                    snapshot = _json_loads(fields.get(b"data", b""))
                    expert_results = snapshot.get("experts")
                    if not expert_results:
                        continue

                    ts_ms = int(snapshot.get("ts_ms", now_ms))
                    context_window = _build_context_window(r, ts_ms)
                    try:
                        fused = qwen.evaluate(expert_results, context_window=context_window)
                        _write_emergency(r, snapshot, fused)
                        last_invoked_ms = now_ms
                        _log(logging.INFO, "qwen_invoked",
                             node_id=snapshot.get("node_id", 0),
                             risk_score=snapshot.get("risk_score", 0.0),
                             risk_level=fused.get("risk_level", "?"),
                             qwen_infer_ms=fused.get("qwen_infer_ms"))
                    except Exception as exc:
                        _log(logging.ERROR, "qwen_failed", error=str(exc))

        except _redis.exceptions.ConnectionError as exc:
            _log(logging.WARNING, "redis_reconnecting", error=str(exc))
            time.sleep(2)
            r = _connect_redis()
            qwen.redis_client = r

        except Exception as exc:
            _log(logging.ERROR, "qwen_service_error", error=str(exc))


if __name__ == "__main__":
    run()
