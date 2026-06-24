import json
import os
import time
from typing import Any

import numpy as np
import onnxruntime as ort


def get_ort_providers():
	# ORT_USE_GPU=1: DML(Windows/DX12) → CUDA → CPU 순서로 시도.
	# 사용 불가 EP는 ORT가 자동으로 건너뜀. RPi5(둘 다 없음)는 CPU fallback.
	if os.getenv("ORT_USE_GPU", "0") == "1":
		return [
			"DmlExecutionProvider",
			("CUDAExecutionProvider", {"do_copy_in_default_stream": True}),
			"CPUExecutionProvider",
		]
	return ["CPUExecutionProvider"]


def get_session_opts() -> ort.SessionOptions:
	opts = ort.SessionOptions()
	opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
	if os.getenv("ORT_USE_GPU", "0") == "1":
		opts.log_severity_level = 3
	return opts


def stream_id_ts_ms(stream_id: Any) -> int:
	if isinstance(stream_id, bytes):
		stream_id = stream_id.decode("utf-8", errors="ignore")
	try:
		return int(str(stream_id).split("-")[0])
	except Exception:
		return int(time.time() * 1000)


def safe_float(value: Any, default: float = 0.0) -> float:
	try:
		if isinstance(value, dict):
			for key in ("value", "score", "mean", "avg"):
				if key in value:
					return float(value[key])
			return default
		if isinstance(value, (list, tuple)):
			if not value:
				return default
			return float(value[0])
		return float(value)
	except Exception:
		return default


def json_loads(raw: Any) -> dict:
	if not raw:
		return {}
	if isinstance(raw, bytes):
		raw = raw.decode("utf-8", errors="ignore")
	try:
		return json.loads(raw)
	except Exception:
		return {}


def build_context_window(r: Any, ts_ms: int, emergency_stream: str, window_minutes: int) -> dict:
	since_ms = ts_ms - (window_minutes * 60 * 1000)
	warning_count = critical_count = 0
	recent_events: list[str] = []
	try:
		entries = r.xrevrange(emergency_stream, count=128)
	except Exception:
		entries = []
	for msg_id, fields in entries:
		if stream_id_ts_ms(msg_id) < since_ms:
			break
		payload = json_loads(fields.get(b"data", b""))
		level = payload.get("risk_level", "normal")
		if level == "critical":
			critical_count += 1
		elif level == "warning":
			warning_count += 1
		summary = payload.get("summary")
		if summary and len(recent_events) < 5:
			recent_events.append(summary)
	parts = []
	if critical_count:
		parts.append(f"critical {critical_count}")
	if warning_count:
		parts.append(f"warning {warning_count}")
	if recent_events:
		parts.append("recent=" + " | ".join(recent_events))
	return {
		"window_minutes":        window_minutes,
		"recent_warning_count":  warning_count,
		"recent_critical_count": critical_count,
		"recent_events":         recent_events,
		"text":                  "; ".join(parts),
	}
