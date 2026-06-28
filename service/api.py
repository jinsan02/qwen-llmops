"""
M5 서빙 API (FastAPI) — Ops 단계 ①.

M1~M4 전문가 출력(JSON)을 받아 응급지수(룰) 게이트 → 임계 초과 시 M5(Qwen) 추론까지
단일 HTTP 엔드포인트로 노출한다. ESP32/Redis 없이 합성/실데이터로 단독 호출 가능.

백엔드: SLM_BACKEND={05b|15b|gguf} (기본 gguf 배포본). qwen_service.py와 동일 선택 규칙.

실행:
  SLM_BACKEND=gguf SLM_MODEL=qwen_15b_gguf MODEL_PATH=volumes/models \
  uvicorn service.api:app --host 0.0.0.0 --port 8000
  → POST /evaluate  ·  GET /health  ·  GET /metrics  ·  /docs(스키마)
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from inference.emergency_score import compute_emergency_score

_BACKEND = os.getenv("SLM_BACKEND", "gguf").lower()
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "volumes", "models"))
SLM_MODEL = os.getenv("SLM_MODEL", "qwen_15b_gguf")
SLM_TOKENIZER = os.getenv("SLM_TOKENIZER", "qwen_15b")
M5_THRESHOLD = float(os.getenv("M5_THRESHOLD", "0.6"))

_LOG = logging.getLogger("qwen_llmops.api")
if not _LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# 간이 모니터링 카운터 (Ops ③ 기반)
_METRICS = {"requests": 0, "m5_called": 0, "errors": 0,
            "level_normal": 0, "level_warning": 0, "level_critical": 0,
            "latency_ms_sum": 0.0}
_state = {"qwen": None}


def _make_qwen():
    path = os.path.join(MODEL_PATH, SLM_MODEL)
    if _BACKEND == "gguf":
        from inference.qwen_gguf import QwenLogic
        return QwenLogic(path, tokenizer_dir=os.path.join(MODEL_PATH, SLM_TOKENIZER))
    if _BACKEND == "15b":
        from inference.qwen_15b import QwenLogic
        return QwenLogic(path)
    from inference.qwen_05b import QwenLogic
    return QwenLogic(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        q = _make_qwen()
        q._ensure_model_loaded()
        _state["qwen"] = q
        _LOG.info("model_loaded backend=%s model=%s", _BACKEND, SLM_MODEL)
    except Exception as e:
        _LOG.error("model_load_failed error=%s", e)
        _state["qwen"] = None
    yield
    _state["qwen"] = None


app = FastAPI(title="SafeWave-AI M5 (Qwen SLM)", version="1.0", lifespan=lifespan)


# ── 스키마 ───────────────────────────────────────────────────────────────────
class ExpertResults(BaseModel):
    """M1~M4 전문가 출력. 누락 도메인은 빈 dict로 안전 처리됨."""
    fall: dict = Field(default_factory=dict, description="M1 낙상: {fall_score, infer_confidence}")
    vital: dict = Field(default_factory=dict, description="M2 생체: {heart_rate, breathing_rate, infer_confidence}")
    env_sound: dict = Field(default_factory=dict, description="M3 환경음: {env_sound_label, env_sound_confidence}")
    speech_ko: dict = Field(default_factory=dict, description="M4 STT: {keywords, stt_confidence, speech_detected}")


class EvalResponse(BaseModel):
    emergency_score: float
    m5_called: bool
    risk_level: str | None = None
    risk_score: float | None = None
    reason: str | None = None
    breakdown: dict
    qwen_infer_ms: float | None = None
    output_tokens: int | None = None
    latency_ms: float


# ── 엔드포인트 ───────────────────────────────────────────────────────────────
@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: ExpertResults):
    t0 = time.perf_counter()
    _METRICS["requests"] += 1
    expert = req.model_dump()
    score, breakdown = compute_emergency_score(expert)
    m5 = score >= M5_THRESHOLD

    resp = {"emergency_score": round(float(score), 4), "m5_called": m5, "breakdown": breakdown,
            "risk_level": None, "risk_score": None, "reason": None,
            "qwen_infer_ms": None, "output_tokens": None}

    if m5 and _state["qwen"] is not None:
        try:
            r = _state["qwen"].evaluate(expert)
            resp.update({
                "risk_level": r.get("risk_level"),
                "risk_score": r.get("risk_score"),
                "reason": r.get("qwen_reason") or r.get("qwen_response"),
                "qwen_infer_ms": r.get("qwen_infer_ms"),
                "output_tokens": r.get("output_tokens"),
            })
            _METRICS["m5_called"] += 1
        except Exception as e:
            _METRICS["errors"] += 1
            _LOG.error("m5_eval_failed error=%s", e)
    elif not m5:
        resp["risk_level"] = "normal"
        resp["risk_score"] = round(float(score), 4)
        resp["reason"] = "M5 미호출 (응급지수 임계 미만)"

    lvl = resp["risk_level"]
    if lvl in ("normal", "warning", "critical"):
        _METRICS[f"level_{lvl}"] += 1
    latency = (time.perf_counter() - t0) * 1000.0
    resp["latency_ms"] = round(latency, 2)
    _METRICS["latency_ms_sum"] += latency
    _LOG.info("evaluate score=%.3f m5=%s level=%s latency=%.0fms",
              score, m5, lvl, latency)
    return resp


@app.get("/health")
def health():
    q = _state["qwen"]
    loaded = q is not None and getattr(q, "session", None) is not None
    return {"status": "ok" if loaded else "degraded", "backend": _BACKEND,
            "model": SLM_MODEL, "model_loaded": loaded, "m5_threshold": M5_THRESHOLD}


@app.get("/metrics")
def metrics():
    n = max(1, _METRICS["requests"])
    return {**_METRICS, "avg_latency_ms": round(_METRICS["latency_ms_sum"] / n, 2),
            "m5_call_rate": round(_METRICS["m5_called"] / n, 3)}
