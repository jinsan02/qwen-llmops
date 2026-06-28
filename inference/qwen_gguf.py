"""
M5: GGUF(llama.cpp) 추론 백엔드 — qwen_15b.QwenLogic을 상속해 생성부만 교체.

대전제 변경: 이 모듈은 ONNX Runtime이 아니라 llama-cpp-python을 쓴다(프로젝트 예외).
M5는 ai-qwen 컨테이너로 격리돼 있어 M1~M4 ONNX 파이프라인은 영향받지 않는다.

- 메시지 구성(_build_messages/_SYSTEM/_SHOTS)·후처리(vital_override·알람제거·parse)는
  qwen_15b와 100% 동일 → 백엔드(GGUF Q4_K_M 등)만의 품질 차이를 공정 비교.
- 생성: llama.cpp로 chat 프롬프트 렌더 후 JSON prefix forcing + 첫 완결 JSON early-stop.

실행:
  from inference.qwen_gguf import QwenLogic
  q = QwenLogic("volumes/models/qwen_15b_gguf/qwen2.5-1.5b-q4_k_m.gguf",
                tokenizer_dir="volumes/models/qwen_15b")
"""

import logging
import os
import time

from inference.qwen_15b import QwenLogic as _QwenLogic15B

_LOGGER = logging.getLogger("qwen_llmops.qwen_gguf")


class QwenLogic(_QwenLogic15B):
    def __init__(self, model_path, tokenizer_dir=None):
        """
        Args:
            model_path: .gguf 파일 경로 (또는 .gguf를 담은 폴더)
            tokenizer_dir: chat_template/tokenizer 폴더 (기본: qwen_15b ONNX 폴더 재사용).
                           env QWEN_GGUF_TOKENIZER로도 지정 가능.
        """
        # 부모 __init__의 ONNX 로딩을 타지 않도록 필요한 속성만 직접 세팅
        self.model_path = model_path
        self.session = None
        self.tokenizer = None
        # 기본 64: 출력 토큰 분석 p99≈56(cap)에서 truncation 발생 → MAX=p99+여유=64 적용
        self.max_new_tokens = int(os.getenv("QWEN_MAX_NEW_TOKENS", "64"))
        self.max_new_tokens = max(40, min(80, self.max_new_tokens))
        self.n_ctx = int(os.getenv("QWEN_GGUF_N_CTX", "2048"))
        self.n_threads = int(os.getenv("QWEN_GGUF_THREADS", "0")) or None  # 0 → llama 기본
        self.hourly_window_ms = int(os.getenv("SLM_HOURLY_WINDOW_MS", "3600000"))
        self.hourly_result_scan_limit = int(os.getenv("SLM_HOURLY_RESULT_SCAN_LIMIT", "1800"))
        self.hourly_emergency_scan_limit = int(os.getenv("SLM_HOURLY_EMERGENCY_SCAN_LIMIT", "600"))
        self.hourly_speech_sample_limit = int(os.getenv("SLM_HOURLY_SPEECH_SAMPLE_LIMIT", "8"))
        self.hourly_event_sample_limit = int(os.getenv("SLM_HOURLY_EVENT_SAMPLE_LIMIT", "8"))
        self.hourly_cache_ms = int(os.getenv("SLM_HOURLY_CACHE_MS", "10000"))
        self.redis_client = None
        self._hourly_cache_at_ms = 0
        self._hourly_cache_data = None
        self._stop_ids = None
        self._last_prompt_tokens = None
        self._last_output_tokens = None
        self.feedback_topic_key = os.getenv("MQTT_FEEDBACK_REDIS_KEY", "mqtt:feedback:last")

        # gguf 파일 해석
        if os.path.isdir(model_path):
            ggufs = [f for f in os.listdir(model_path) if f.endswith(".gguf")]
            if not ggufs:
                raise FileNotFoundError(f"gguf 파일 없음: {model_path}")
            self._gguf_file = os.path.join(model_path, sorted(ggufs)[0])
        else:
            self._gguf_file = model_path

        self._model_dir = tokenizer_dir or os.getenv("QWEN_GGUF_TOKENIZER") \
            or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "volumes", "models", "qwen_15b")
        self._llama = None
        self._load_attempted = False

    def _ensure_model_loaded(self):
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from llama_cpp import Llama
            self._llama = Llama(
                model_path=self._gguf_file,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                logits_all=False,
                verbose=False,
            )
            self.session = self._llama  # evaluate()의 truthiness 게이트 통과용
            _LOGGER.info("qwen_gguf_loaded path=%s", self._gguf_file)
        except Exception as e:
            _LOGGER.error("qwen_gguf_load_failed error=%s", e)
            self._llama = None
            self.session = None

        # chat_template 적용용 토크나이저(transformers) — 메시지 렌더 + json_complete 디코드
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self._model_dir, trust_remote_code=True)
        except Exception as e:
            _LOGGER.warning("qwen_gguf_tokenizer_failed error=%s", e)
            self.tokenizer = None

    def _evaluate_with_qwen(self, messages):
        """qwen_15b와 동일한 프롬프트(chat_template + '{' 강제) → llama.cpp 생성 → 첫 완결 JSON."""
        if self._llama is None or self.tokenizer is None:
            return None
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted += "{"  # JSON prefix forcing (qwen_15b와 동일)

            out = self._llama(
                formatted,
                max_tokens=self.max_new_tokens,
                temperature=0.0,            # 결정적(greedy) — ONNX argmax와 정합
                top_k=1,
                stop=["<|im_end|>", "<|endoftext|>", "\n\n"],
                echo=False,
            )
            usage = out.get("usage") or {}
            self._last_prompt_tokens = usage.get("prompt_tokens")
            self._last_output_tokens = usage.get("completion_tokens")
            text = out["choices"][0]["text"]
            if not text:
                return None
            raw = "{" + text.lstrip("{")
            # 첫 완결 JSON만 남김 (강제 '{' 기준 depth 1에서 시작)
            depth, end = 0, None
            for idx, ch in enumerate(raw):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx + 1
                        break
            return raw[:end] if end else raw
        except Exception as e:
            _LOGGER.error("qwen_gguf_infer_failed error=%s", e)
            return None
