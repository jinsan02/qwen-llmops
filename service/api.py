"""
M5 서빙 API (FastAPI) — Ops 단계 ①(서빙) + ②(모니터링) + ③(피드백/거버넌스).

M1~M4 전문가 출력(JSON)을 받아 응급지수(룰) 게이트 → 임계 초과 시 M5(Qwen) 추론까지
단일 HTTP 엔드포인트로 노출한다. ESP32/Redis 없이 합성/실데이터로 단독 호출 가능.

백엔드: SLM_BACKEND={05b|15b|gguf} (기본 gguf 배포본 Q5_K_M). qwen_service.py와 동일 선택 규칙.

엔드포인트:
  POST /evaluate   — 응급지수 게이트 → M5
  POST /feedback   — 보호자 피드백(false_alarm|missed_alert|confirm) → Redis(다음 추론에 반영)
  GET  /health     — 모델 로드 상태 + 백엔드/버전/핑거프린트
  GET  /metrics    — Prometheus 텍스트(text/plain) 노출
  GET  /metrics.json — 동일 카운터의 JSON 뷰
  GET  /docs       — 스키마

실행:
  SLM_BACKEND=gguf SLM_MODEL=qwen_15b_gguf_q5 MODEL_PATH=volumes/models \
  uvicorn service.api:app --host 0.0.0.0 --port 8000
"""

import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from inference.emergency_score import compute_emergency_score

_BACKEND = os.getenv("SLM_BACKEND", "gguf").lower()
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "volumes", "models"))
SLM_MODEL = os.getenv("SLM_MODEL", "qwen_15b_gguf_q5")
SLM_TOKENIZER = os.getenv("SLM_TOKENIZER", "qwen_15b")
M5_THRESHOLD = float(os.getenv("M5_THRESHOLD", "0.6"))

# Ops ③ 피드백/메트릭 지속화용 Redis(선택). 미설정/미가용 시 인메모리만(graceful).
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
FEEDBACK_KEY = os.getenv("MQTT_FEEDBACK_REDIS_KEY", "mqtt:feedback:last")
METRICS_SNAPSHOT_KEY = "m5:metrics:snapshot"
_TTL = 3600  # 운영 Redis 키 TTL ≤ 3600s (프로젝트 제약)

_LOG = logging.getLogger("qwen_llmops.api")
if not _LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# 간이 모니터링 카운터 (Ops ②)
_METRICS = {"requests": 0, "m5_called": 0, "errors": 0,
            "level_normal": 0, "level_warning": 0, "level_critical": 0,
            "feedback_false_alarm": 0, "feedback_missed_alert": 0, "feedback_confirm": 0,
            "latency_ms_sum": 0.0}
_state = {"qwen": None, "redis": None, "model_sha": "unknown"}


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


def _model_fingerprint() -> str:
    """모델 파일(name+size+mtime)의 12자 핑거프린트 — 거버넌스(모델 스왑 감지)용.
    대용량 파일 전체 해시는 기동 지연이 커서 메타 기반 빠른 지문 사용."""
    path = os.path.join(MODEL_PATH, SLM_MODEL)
    target = path
    try:
        if os.path.isdir(path):
            ggufs = [f for f in os.listdir(path) if f.endswith(".gguf")]
            if ggufs:
                target = os.path.join(path, sorted(ggufs)[0])
            else:
                onnx = os.path.join(path, "model.onnx")
                target = onnx if os.path.exists(onnx) else path
        st = os.stat(target)
        raw = f"{os.path.basename(target)}:{st.st_size}:{int(st.st_mtime)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
    except Exception:
        return "unknown"


def _connect_redis():
    if not REDIS_HOST:
        return None
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True,
                        socket_connect_timeout=2)
        r.ping()
        _LOG.info("redis_connected host=%s", REDIS_HOST)
        return r
    except Exception as e:
        _LOG.warning("redis_unavailable error=%s — 인메모리 모드", e)
        return None


def _restore_metrics(r):
    if r is None:
        return
    try:
        snap = r.hgetall(METRICS_SNAPSHOT_KEY)
        for k, v in snap.items():
            if k in _METRICS:
                _METRICS[k] = float(v) if k == "latency_ms_sum" else int(v)
        if snap:
            _LOG.info("metrics_restored from redis")
    except Exception as e:
        _LOG.warning("metrics_restore_failed error=%s", e)


def _persist_metrics():
    r = _state["redis"]
    if r is None:
        return
    try:
        r.hset(METRICS_SNAPSHOT_KEY, mapping={k: str(v) for k, v in _METRICS.items()})
        r.expire(METRICS_SNAPSHOT_KEY, _TTL)  # 1h 롤링(프로젝트 TTL 제약)
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["redis"] = _connect_redis()
    _restore_metrics(_state["redis"])
    _state["model_sha"] = _model_fingerprint()
    try:
        q = _make_qwen()
        q._ensure_model_loaded()
        if _state["redis"] is not None:
            q.redis_client = _state["redis"]   # 피드백 루프: evaluate가 mqtt:feedback:last 반영
        _state["qwen"] = q
        _LOG.info("model_loaded backend=%s model=%s sha=%s", _BACKEND, SLM_MODEL, _state["model_sha"])
    except Exception as e:
        _LOG.error("model_load_failed error=%s", e)
        _state["qwen"] = None
    yield
    _persist_metrics()
    _state["qwen"] = None


app = FastAPI(title="SafeWave-AI M5 (Qwen SLM)", version="1.1", lifespan=lifespan)


# ── 스키마 ───────────────────────────────────────────────────────────────────
class ExpertResults(BaseModel):
    """M1~M4 전문가 출력. 누락 도메인은 빈 dict로 안전 처리됨."""
    fall: dict = Field(default_factory=dict, description="M1 낙상: {fall_score, infer_confidence}")
    vital: dict = Field(default_factory=dict, description="M2 생체: {heart_rate, breathing_rate, infer_confidence}")
    env_sound: dict = Field(default_factory=dict, description="M3 환경음: {env_sound_label, env_sound_confidence}")
    speech_ko: dict = Field(default_factory=dict, description="M4 STT: {keywords, stt_confidence, speech_detected}")
    time_series: list | None = Field(default=None, description="분당 vital 시계열(≤60행) — 있으면 게이트·M5에 반영")


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


class Feedback(BaseModel):
    feedback: str = Field(description="false_alarm | missed_alert | confirm")
    delta: float | None = Field(default=None, description="직접 보정값(미지정 시 feedback에서 유도)")
    case_id: str | None = None


# ── 엔드포인트 ───────────────────────────────────────────────────────────────
@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: ExpertResults):
    t0 = time.perf_counter()
    _METRICS["requests"] += 1
    payload = req.model_dump()
    ts = payload.pop("time_series", None)
    expert = payload
    score, breakdown = compute_emergency_score(expert, time_series=ts)
    m5 = score >= M5_THRESHOLD

    resp = {"emergency_score": round(float(score), 4), "m5_called": m5, "breakdown": breakdown,
            "risk_level": None, "risk_score": None, "reason": None,
            "qwen_infer_ms": None, "output_tokens": None}

    if m5 and _state["qwen"] is not None:
        try:
            r = _state["qwen"].evaluate(expert, time_series=ts)
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
    _persist_metrics()
    _LOG.info("evaluate score=%.3f m5=%s level=%s latency=%.0fms", score, m5, lvl, latency)
    return resp


@app.post("/feedback")
def feedback(fb: Feedback):
    """보호자 피드백 → Redis(mqtt:feedback:last, TTL≤3600). 다음 evaluate가 risk_score에 반영."""
    kind = fb.feedback.strip().lower()
    if kind not in ("false_alarm", "missed_alert", "confirm"):
        return {"ok": False, "error": "feedback must be false_alarm|missed_alert|confirm"}
    delta = fb.delta
    if delta is None:
        delta = {"false_alarm": -0.08, "missed_alert": 0.08, "confirm": 0.0}[kind]
    _METRICS[f"feedback_{kind}"] = _METRICS.get(f"feedback_{kind}", 0) + 1

    r = _state["redis"]
    persisted = False
    if r is not None:
        try:
            import json
            r.set(FEEDBACK_KEY, json.dumps({"feedback": kind, "delta": delta,
                                            "case_id": fb.case_id, "ts_ms": int(time.time() * 1000)},
                                           ensure_ascii=False), ex=_TTL)
            persisted = True
        except Exception as e:
            _LOG.error("feedback_persist_failed error=%s", e)
    _persist_metrics()
    _LOG.info("feedback kind=%s delta=%+.2f persisted=%s", kind, delta, persisted)
    return {"ok": True, "feedback": kind, "delta": delta, "persisted_to_redis": persisted}


@app.get("/health")
def health():
    q = _state["qwen"]
    loaded = q is not None and getattr(q, "session", None) is not None
    return {"status": "ok" if loaded else "degraded", "backend": _BACKEND,
            "model": SLM_MODEL, "model_loaded": loaded, "m5_threshold": M5_THRESHOLD,
            "model_version": SLM_MODEL, "model_sha": _state["model_sha"],
            "redis": _state["redis"] is not None}


@app.get("/metrics.json")
def metrics_json():
    n = max(1, _METRICS["requests"])
    return {**_METRICS, "avg_latency_ms": round(_METRICS["latency_ms_sum"] / n, 2),
            "m5_call_rate": round(_METRICS["m5_called"] / n, 3)}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Prometheus exposition 텍스트(text/plain; version=0.0.4) — 외부 스택 없이 스크랩 가능."""
    n = max(1, _METRICS["requests"])
    labels = f'backend="{_BACKEND}",model="{SLM_MODEL}"'
    lines = [
        "# HELP m5_requests_total Total /evaluate requests",
        "# TYPE m5_requests_total counter",
        f'm5_requests_total{{{labels}}} {_METRICS["requests"]}',
        "# HELP m5_called_total Requests that invoked the M5 model",
        "# TYPE m5_called_total counter",
        f'm5_called_total{{{labels}}} {_METRICS["m5_called"]}',
        "# HELP m5_errors_total M5 inference errors",
        "# TYPE m5_errors_total counter",
        f'm5_errors_total{{{labels}}} {_METRICS["errors"]}',
        "# HELP m5_level_total Risk level outcomes",
        "# TYPE m5_level_total counter",
        f'm5_level_total{{{labels},level="normal"}} {_METRICS["level_normal"]}',
        f'm5_level_total{{{labels},level="warning"}} {_METRICS["level_warning"]}',
        f'm5_level_total{{{labels},level="critical"}} {_METRICS["level_critical"]}',
        "# HELP m5_feedback_total Caregiver feedback events",
        "# TYPE m5_feedback_total counter",
        f'm5_feedback_total{{{labels},type="false_alarm"}} {_METRICS["feedback_false_alarm"]}',
        f'm5_feedback_total{{{labels},type="missed_alert"}} {_METRICS["feedback_missed_alert"]}',
        f'm5_feedback_total{{{labels},type="confirm"}} {_METRICS["feedback_confirm"]}',
        "# HELP m5_avg_latency_ms Average /evaluate latency",
        "# TYPE m5_avg_latency_ms gauge",
        f'm5_avg_latency_ms{{{labels}}} {_METRICS["latency_ms_sum"] / n:.2f}',
        "# HELP m5_call_rate Fraction of requests invoking M5",
        "# TYPE m5_call_rate gauge",
        f'm5_call_rate{{{labels}}} {_METRICS["m5_called"] / n:.3f}',
        "# HELP m5_model_loaded Model load status (1=ok)",
        "# TYPE m5_model_loaded gauge",
        f'm5_model_loaded{{{labels}}} {1 if (_state["qwen"] is not None and getattr(_state["qwen"], "session", None) is not None) else 0}',
    ]
    return "\n".join(lines) + "\n"
