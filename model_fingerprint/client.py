"""HTTP client with one request builder per wire protocol."""

import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request

from .probes import MIN_TOKENS

SSL_CONTEXT = ssl.create_default_context()
USER_AGENT = "model-fingerprint/0.1"


def detect_api(base_url, model=None):
    """Best guess at the wire protocol from the URL alone."""
    url = (base_url or "").lower().rstrip("/")
    if "anthropic" in url:
        return "anthropic"
    if "generativelanguage.googleapis.com" in url or "aiplatform.googleapis.com" in url:
        return "gemini"
    if url.endswith("/responses"):
        return "responses"
    return "openai"


def load_env_file(path):
    """Minimal dotenv parse: KEY=VALUE, ignoring comments and a leading 'export '."""
    out = {}
    if not path:
        return out
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.replace("export ", "").strip()
                value = value.strip().strip("'\"")
                if key and value:
                    out[key] = value
    except OSError as exc:
        print("warning: cannot read env file %s (%s)" % (path, exc), file=sys.stderr)
    return out


def messages_to_text(messages):
    """Flatten chat messages into a single input string (Responses/Gemini style)."""
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        parts.append(str(content))
    return "\n".join(parts)


class Client:
    """Thin transport: builds protocol-correct payloads and records every call."""

    def __init__(
        self,
        base_url,
        api_key=None,
        model=None,
        api="openai",
        timeout=90,
        extra_headers=None,
        reasoning_effort=None,
    ):
        self.base = (base_url or "").rstrip("/")
        self.key = api_key or ""
        self.model = model
        self.api = api
        self.timeout = timeout
        self.extra = dict(extra_headers or {})
        self.reasoning_effort = reasoning_effort
        self.log = []

    # ------------------------------------------------------------- transport
    def request(self, method, url, body=None, extra_headers=None):
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        headers.update(self.extra)
        if self.key:
            if self.api == "anthropic":
                headers["x-api-key"] = self.key
                headers["anthropic-version"] = "2023-06-01"
            else:
                headers["Authorization"] = "Bearer " + self.key
        if extra_headers:
            headers.update(extra_headers)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        start = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=SSL_CONTEXT) as resp:
                raw = resp.read().decode("utf-8", "replace")
                status, hdrs = resp.status, dict(resp.headers)
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read().decode("utf-8", "replace")
                status, hdrs = exc.code, dict(exc.headers)
            finally:
                exc.close()  # HTTPError holds a socket; not closing it leaks a ResourceWarning
        except Exception as exc:  # noqa: BLE001 - a transport failure is data too
            out = {
                "status": None,
                "headers": {},
                "text": "",
                "json": None,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "ms": round((time.time() - start) * 1000),
            }
            self.log.append({"url": url, "status": None, "ms": out["ms"], "error": out["error"]})
            return out
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        out = {
            "status": status,
            "headers": hdrs,
            "text": raw,
            "json": parsed,
            "ms": round((time.time() - start) * 1000),
        }
        self.log.append({"url": url, "status": status, "ms": out["ms"]})
        return out

    # -------------------------------------------------------- payload builders
    def apply_reasoning(self, body):
        """Thinking effort must be identical on target and reference, or the
        statistical layer compares a different model configuration."""
        if not self.reasoning_effort:
            return body
        effort = self.reasoning_effort
        if self.api == "responses":
            body["reasoning"] = {"effort": effort}
        elif self.api == "anthropic":
            # extended thinking requires temperature=1, so only enable when asked
            body["thinking"] = (
                {"type": "disabled"}
                if effort == "minimal"
                else {"type": "enabled", "budget_tokens": 1024}
            )
        elif self.api == "gemini":
            config = body.setdefault("generationConfig", {})
            config["thinkingConfig"] = {"thinkingBudget": 0 if effort == "minimal" else 1024}
        else:
            body["reasoning_effort"] = effort
        return body

    def chat(self, messages, max_tokens=8, temperature=None, extra=None):
        budget = max(MIN_TOKENS, int(max_tokens))
        if self.api == "anthropic":
            body = {"model": self.model, "max_tokens": budget, "messages": messages}
            if temperature is not None:
                body["temperature"] = temperature
            if extra:
                body.update(extra)
            return self.request("POST", self.base + "/messages", self.apply_reasoning(body))
        if self.api == "gemini":
            url = "%s/models/%s:generateContent?key=%s" % (self.base, self.model, self.key)
            config = {"maxOutputTokens": budget}
            if temperature is not None:
                config["temperature"] = temperature
            body = {
                "contents": [{"role": "user", "parts": [{"text": messages_to_text(messages)}]}],
                "generationConfig": config,
            }
            if extra:
                body.update(extra)
            return self.request(
                "POST",
                url,
                self.apply_reasoning(body),
                extra_headers={"Content-Type": "application/json"},
            )
        if self.api == "responses":
            body = {
                "model": self.model,
                "input": messages_to_text(messages),
                "max_output_tokens": budget,
            }
            if temperature is not None:
                body["temperature"] = temperature
            if extra:
                body.update(extra)
            return self.request("POST", self.base + "/responses", self.apply_reasoning(body))
        body = {"model": self.model, "messages": messages, "max_tokens": budget}
        if temperature is not None:
            body["temperature"] = temperature
        if extra:
            body.update(extra)
        return self.request("POST", self.base + "/chat/completions", self.apply_reasoning(body))

    def probe_body(self):
        """Base payload for the error probes (mutated in place, sent raw).

        The budget is deliberately >= MIN_TOKENS: a strict endpoint rejects tiny
        output budgets *before* validating anything else, which would mask the
        parameter this probe is actually testing.
        """
        if self.api == "anthropic":
            return {
                "model": self.model,
                "max_tokens": MIN_TOKENS,
                "messages": [{"role": "user", "content": "hi"}],
            }
        if self.api == "gemini":
            return {
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "generationConfig": {"maxOutputTokens": MIN_TOKENS},
            }
        if self.api == "responses":
            return {"model": self.model, "input": "hi", "max_output_tokens": MIN_TOKENS}
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": MIN_TOKENS,
        }

    def send_raw(self, body, path=None):
        """Send a hand-built payload - used by the error probes, no reasoning injection."""
        if self.api == "gemini":
            url = "%s/models/%s:generateContent?key=%s" % (self.base, self.model, self.key)
            return self.request("POST", url, body)
        default_path = {"anthropic": "/messages", "responses": "/responses"}.get(
            self.api, "/chat/completions"
        )
        return self.request("POST", self.base + (path or default_path), body)
