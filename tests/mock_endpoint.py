"""A local fake LLM endpoint, used by the offline test suite.

Speaks all four supported wire protocols (OpenAI chat-completions, OpenAI
Responses, Anthropic Messages, Gemini generateContent) with deterministic
outputs and deliberately realistic failure modes, so the whole probe battery can
be exercised without a network, a key or a cent of spend.

Behaviour is configurable through MockServer(**options):

    require_session_header  gateway rejects everything without a routing header
    reasoning_model         burns the output budget on hidden thinking when the
                            budget is below reasoning_min_budget (unless thinking
                            is pinned off via the reasoning/thinking parameter)
    omit_usage              returns 200 but never reports token usage
    max_output_cap          reject output budgets above this
    max_context_tokens      reject prompts above this (context-ceiling behaviour)
"""

import json
import threading
import typing
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

SERVER_HEADER = "mock-endpoint/1.0"
MODELS = ["mock-alpha", "mock-beta", "mock-gamma"]
UNKNOWN_PARAMS = ("repetition_penalty", "min_tokens", "ignore_eos")
# top-level keys a strict chat-completions endpoint accepts
CHAT_KEYS = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "n",
        "stream",
        "stream_options",
        "stop",
        "logprobs",
        "top_logprobs",
        "seed",
        "tools",
        "tool_choice",
        "response_format",
        "user",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "parallel_tool_calls",
        "reasoning_effort",
        "metadata",
        "store",
        "service_tier",
        "modalities",
        "prediction",
    }
)


def crc(text):
    return zlib.crc32(text.encode("utf-8"))


def count_tokens(text):
    """Deterministic stand-in for a real tokenizer."""
    return len(text) // 3 + 5


def answer_for(text):
    if "alpha" in text:
        return "alpha"
    return "ans-%02d" % (crc(text) % 100)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    options: typing.ClassVar[dict] = {}

    def log_message(self, *args):  # keep the test output clean
        pass

    # ------------------------------------------------------------- plumbing
    def _send(self, code, payload, extra_headers=None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("x-mock-server", SERVER_HEADER)
        self.send_header("x-request-id", "mock-%08x" % crc(str(self.path)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _openai_error(self, code, etype, message, param=None, code_str=None):
        self._send(
            code, {"error": {"message": message, "type": etype, "param": param, "code": code_str}}
        )

    def _anthropic_error(self, code, etype, message):
        self._send(code, {"type": "error", "error": {"type": etype, "message": message}})

    def _gemini_error(self, code, message, status="INVALID_ARGUMENT"):
        self._send(code, {"error": {"code": code, "message": message, "status": status}})

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            return {}

    def _session_ok(self):
        if self.options.get("require_session_header") and not self.headers.get("x-mock-session"):
            self._send(
                400,
                {
                    "type": "error",
                    "error": {
                        "type": "MissingSessionID",
                        "message": "Request is missing x-mock-session; cannot be routed.",
                    },
                },
            )
            return False
        return True

    def _context_exceeded(self, text, style="openai"):
        """Optional context ceiling - mirrors a gateway refusing an oversized prompt."""
        cap = self.options.get("max_context_tokens")
        if not cap:
            return False
        tokens = count_tokens(text)
        if tokens <= cap:
            return False
        message = (
            "This model's maximum context length is %d tokens. However, your messages "
            "resulted in %d tokens." % (cap, tokens)
        )
        if style == "anthropic":
            self._anthropic_error(400, "invalid_request_error", message)
        elif style == "gemini":
            self._gemini_error(400, message)
        else:
            self._openai_error(400, "invalid_request_error", message, param="messages")
        return True

    def _thinking_disabled(self, body):
        """Honour a pinned-off thinking setting so a pinned run gets real output."""
        reasoning = body.get("reasoning") or {}
        if isinstance(reasoning, dict) and reasoning.get("effort") == "minimal":
            return True
        thinking = body.get("thinking") or {}
        if isinstance(thinking, dict) and thinking.get("type") == "disabled":
            return True
        config = (body.get("generationConfig") or {}).get("thinkingConfig") or {}
        return config.get("thinkingBudget") == 0

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        path = urlparse(self.path).path
        if not self._session_ok():
            return
        if path == "/v1/models":
            self._send(
                200,
                {
                    "object": "list",
                    "data": [{"id": mid, "object": "model", "owned_by": "mock"} for mid in MODELS],
                },
            )
        elif path == "/v1/health":
            self._send(200, {"status": "ok"})
        else:
            self._openai_error(404, "invalid_request_error", "Not found: %s" % path)

    # ----------------------------------------------------------------- POST
    def do_POST(self):
        if not self._session_ok():
            return
        path = urlparse(self.path).path
        body = self._read_body()
        if path == "/v1/chat/completions":
            self._chat_completions(body)
        elif path == "/v1/responses":
            self._responses(body)
        elif path == "/v1/messages":
            self._messages(body)
        elif ":generateContent" in path:
            self._generate_content(path, body)
        else:
            self._openai_error(404, "invalid_request_error", "Unknown path: %s" % path)

    # ------------------------------------------------------- protocol shapes
    def _check_model(self, model, style="openai"):
        if model not in MODELS:
            message = "The model `%s` does not exist" % model
            if style == "anthropic":
                self._anthropic_error(404, "not_found_error", message)
            elif style == "gemini":
                self._gemini_error(404, message, "NOT_FOUND")
            else:
                self._openai_error(
                    404, "invalid_request_error", message, param="model", code_str="model_not_found"
                )
            return False
        return True

    def _budget_error(self, value, style):
        cap = self.options.get("max_output_cap", 65536)
        if value is None:
            message = "Missing required parameter: `max_tokens`."
            param = "max_tokens"
        elif value < 16:
            message = "`max_tokens` The number must be `>= 16`."
            param = "max_tokens"
        elif value > cap:
            message = "`max_tokens` is too large: maximum is %d." % cap
            param = "max_tokens"
        else:
            return None
        if style == "anthropic":
            self._anthropic_error(400, "invalid_request_error", message)
        elif style == "gemini":
            self._gemini_error(400, message)
        else:
            self._openai_error(400, "invalid_request_error", message, param=param)
        return True

    def _chat_completions(self, body):
        if not self._check_model(body.get("model")):
            return
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._openai_error(
                400,
                "invalid_request_error",
                "Invalid value: 'messages' must be a non-empty array.",
                param="messages",
            )
            return
        for message in messages:
            role = (message or {}).get("role")
            if role not in ("system", "developer", "user", "assistant", "tool"):
                self._openai_error(
                    400,
                    "invalid_request_error",
                    "Invalid value: '%s' is not one of ['system','developer','user',"
                    "'assistant','tool'] - 'messages[0].role'" % role,
                    param="messages[0].role",
                )
                return
        text = self._text_of_messages(messages)
        if self._context_exceeded(text):
            return
        budget = body.get("max_tokens")
        if self._budget_error(budget, "openai"):
            return
        temperature = body.get("temperature")
        if temperature is not None and not 0 <= temperature <= 2:
            self._openai_error(
                400,
                "invalid_request_error",
                "`temperature` must be between 0 and 2",
                param="temperature",
            )
            return
        for name in UNKNOWN_PARAMS:
            if name in body:
                self._openai_error(
                    400,
                    "invalid_request_error",
                    "Unrecognized request argument supplied: %s" % name,
                    param=name,
                )
                return
        if "seed" in body:
            self._openai_error(
                400, "invalid_request_error", "unknown parameter `seed`", param="seed"
            )
            return
        if body.get("n", 1) > 1:
            self._openai_error(400, "invalid_request_error", "unknown parameter `n`", param="n")
            return
        if "response_format" in body:
            self._openai_error(
                400,
                "invalid_request_error",
                "unknown parameter `response_format`",
                param="response_format",
            )
            return
        unknown = sorted(set(body) - CHAT_KEYS)
        if unknown:
            self._openai_error(
                400,
                "invalid_request_error",
                "Unrecognized request argument supplied: %s" % unknown[0],
                param=unknown[0],
            )
            return
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer_for(text)},
                    "finish_reason": "stop",
                }
            ],
        }
        if not self.options.get("omit_usage"):
            prompt_tokens = count_tokens(text)
            payload["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 3,
                "total_tokens": prompt_tokens + 3,
            }
        self._send(200, payload)

    def _responses(self, body):
        if not self._check_model(body.get("model")):
            return
        text = body.get("input") or ""
        if self._context_exceeded(text):
            return
        budget = body.get("max_output_tokens")
        if self._budget_error(budget, "openai"):
            return
        for name in UNKNOWN_PARAMS + ("seed", "n", "logprobs", "stop"):
            if name in body:
                self._openai_error(
                    400, "invalid_request_error", "unknown parameter `%s`" % name, param=name
                )
                return
        if "top_logprobs" in body:
            self._openai_error(
                400,
                "invalid_request_error",
                '`top_logprobs` requires `include` to contain "message.output_text.logprobs"',
                param="top_logprobs",
            )
            return
        text = body.get("input") or ""
        prompt_tokens = count_tokens(text)
        reasoning_floor = self.options.get("reasoning_min_budget", 256)
        cut_off = (
            self.options.get("reasoning_model")
            and budget < reasoning_floor
            and not self._thinking_disabled(body)
        )
        if cut_off:
            payload = {
                "id": "resp-mock",
                "object": "response",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs-mock",
                        "summary": [],
                        "encrypted_content": "...",
                    }
                ],
                "usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": budget,
                    "output_tokens_details": {"reasoning_tokens": max(1, budget - 3)},
                },
            }
            self._send(200, payload)
            return
        payload = {
            "id": "resp-mock",
            "object": "response",
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs-mock", "summary": [], "encrypted_content": "..."},
                {
                    "type": "message",
                    "id": "msg-mock",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": answer_for(text), "annotations": []}
                    ],
                },
            ],
        }
        if not self.options.get("omit_usage"):
            payload["usage"] = {
                "input_tokens": prompt_tokens,
                "output_tokens": 8,
                "output_tokens_details": {"reasoning_tokens": 4},
            }
        self._send(200, payload)

    def _messages(self, body):
        if not self._check_model(body.get("model"), "anthropic"):
            return
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._anthropic_error(
                400, "invalid_request_error", "messages: must be a non-empty array"
            )
            return
        for message in messages:
            if (message or {}).get("role") not in ("user", "assistant"):
                self._anthropic_error(
                    400,
                    "invalid_request_error",
                    "messages.0.role: Input should be 'user' or 'assistant'",
                )
                return
        text = self._text_of_messages(messages)
        if self._context_exceeded(text, "anthropic"):
            return
        if self._budget_error(body.get("max_tokens"), "anthropic"):
            return
        payload = {
            "id": "msg-mock",
            "type": "message",
            "role": "assistant",
            "model": body.get("model"),
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": answer_for(text)}],
        }
        if not self.options.get("omit_usage"):
            prompt_tokens = count_tokens(text)
            payload["usage"] = {"input_tokens": prompt_tokens, "output_tokens": 4}
        self._send(200, payload)

    def _generate_content(self, path, body):
        model = path.split("/models/")[-1].split(":")[0]
        if not self._check_model(model, "gemini"):
            return
        text = ""
        for part in (body.get("contents") or [{}])[0].get("parts", []):
            text += part.get("text", "")
        if self._context_exceeded(text, "gemini"):
            return
        config = body.get("generationConfig") or {}
        if self._budget_error(config.get("maxOutputTokens"), "gemini"):
            return
        payload = {
            "candidates": [
                {
                    "content": {"parts": [{"text": answer_for(text)}], "role": "model"},
                    "finishReason": "STOP",
                }
            ]
        }
        if not self.options.get("omit_usage"):
            prompt_tokens = count_tokens(text)
            payload["usageMetadata"] = {
                "promptTokenCount": prompt_tokens,
                "candidatesTokenCount": 4,
                "totalTokenCount": prompt_tokens + 4,
            }
        self._send(200, payload)

    @staticmethod
    def _text_of_messages(messages):
        parts = []
        for message in messages or []:
            content = message.get("content")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            parts.append(str(content))
        return "\n".join(parts)


class MockServer:
    """Runs the fake endpoint on an ephemeral localhost port."""

    def __init__(self, **options):
        handler = type("BoundHandler", (_Handler,), {"options": dict(options)})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        return "http://127.0.0.1:%d/v1" % self.port

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
