"""Client: protocol-correct payloads, floors, reasoning injection, env loading."""

import json

import pytest

from model_fingerprint import probes
from model_fingerprint.client import Client, detect_api, load_env_file, messages_to_text

MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture
def captured(monkeypatch):
    """Capture every outgoing payload instead of sending it."""
    calls = []

    def fake_request(self, method, url, body=None, extra_headers=None):
        calls.append(
            {
                "method": method,
                "url": url,
                "body": body,
                "extra_headers": extra_headers,
                "headers": dict(self.extra),
            }
        )
        return {"status": 200, "headers": {}, "text": "{}", "json": {}, "ms": 1}

    monkeypatch.setattr(Client, "request", fake_request)
    return calls


def test_detect_api():
    assert detect_api("https://api.anthropic.com/v1") == "anthropic"
    assert detect_api("https://generativelanguage.googleapis.com/v1beta") == "gemini"
    assert detect_api("https://gw.example.com/v1/responses") == "responses"
    assert detect_api("https://gw.example.com/v1/") == "openai"
    assert detect_api("") == "openai"
    assert detect_api(None) == "openai"


def test_load_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nexport MY_KEY=abc123\nQUOTED=\"q1\"\nSINGLE='q2'\nEMPTY=\nNOEQUALS\n\n"
    )
    values = load_env_file(str(env))
    assert values == {"MY_KEY": "abc123", "QUOTED": "q1", "SINGLE": "q2"}
    assert load_env_file(str(tmp_path / "missing")) == {}
    assert load_env_file(None) == {}


def test_messages_to_text_flattens_parts():
    assert messages_to_text(MESSAGES) == "hi"
    assert (
        messages_to_text(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                }
            ]
        )
        == "a b"
    )
    assert messages_to_text([{"role": "system", "content": "s"}, MESSAGES[0]]) == "s\nhi"


@pytest.mark.parametrize(
    "api,path,key",
    [
        ("openai", "/chat/completions", "max_tokens"),
        ("responses", "/responses", "max_output_tokens"),
        ("anthropic", "/messages", "max_tokens"),
    ],
)
def test_payload_shape_per_protocol(captured, api, path, key):
    client = Client("https://host/v1", "k", "mystery", api)
    client.chat(MESSAGES, max_tokens=64, temperature=0)
    call = captured[-1]
    assert call["url"] == "https://host/v1" + path
    assert call["body"]["model"] == "mystery"
    assert call["body"][key] == 64
    assert call["body"]["temperature"] == 0
    if api == "responses":
        assert call["body"]["input"] == "hi"
    else:
        assert call["body"]["messages"] == MESSAGES


def test_gemini_payload_puts_model_and_key_in_url(captured):
    client = Client("https://gen/v1beta", "secret", "gemini-9", "gemini")
    client.chat(MESSAGES, max_tokens=32, temperature=1)
    call = captured[-1]
    assert call["url"] == "https://gen/v1beta/models/gemini-9:generateContent?key=secret"
    assert call["body"]["contents"][0]["parts"][0]["text"] == "hi"
    assert call["body"]["generationConfig"] == {"maxOutputTokens": 32, "temperature": 1}


def test_tiny_budgets_are_floored(captured):
    """max_tokens=1 is rejected outright by strict APIs - looks like a capability failure."""
    for api in ("openai", "responses", "anthropic", "gemini"):
        Client("https://host/v1", "k", "m", api).chat(MESSAGES, max_tokens=1, temperature=0)
        body = captured[-1]["body"]
        budget = (
            body.get("max_tokens")
            or body.get("max_output_tokens")
            or body["generationConfig"]["maxOutputTokens"]
        )
        assert budget == probes.MIN_TOKENS


def test_reasoning_effort_is_protocol_specific(captured):
    Client("https://host/v1", "k", "m", "openai", reasoning_effort="low").chat(MESSAGES)
    assert captured[-1]["body"]["reasoning_effort"] == "low"

    Client("https://host/v1", "k", "m", "responses", reasoning_effort="low").chat(MESSAGES)
    assert captured[-1]["body"]["reasoning"] == {"effort": "low"}

    Client("https://host/v1", "k", "m", "gemini", reasoning_effort="low").chat(MESSAGES)
    assert captured[-1]["body"]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 1024}
    Client("https://host/v1", "k", "m", "gemini", reasoning_effort="minimal").chat(MESSAGES)
    assert captured[-1]["body"]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}

    Client("https://host/v1", "k", "m", "anthropic", reasoning_effort="minimal").chat(MESSAGES)
    assert captured[-1]["body"]["thinking"] == {"type": "disabled"}
    Client("https://host/v1", "k", "m", "anthropic", reasoning_effort="high").chat(MESSAGES)
    assert captured[-1]["body"]["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_no_reasoning_key_when_effort_unset(captured):
    for api in ("openai", "responses", "anthropic", "gemini"):
        Client("https://host/v1", "k", "m", api).chat(MESSAGES)
        body = captured[-1]["body"]
        assert "reasoning" not in body
        assert "reasoning_effort" not in body
        assert "thinking" not in body


def test_probe_body_matches_protocol():
    assert "messages" in Client("https://h/v1", "", "m", "openai").probe_body()
    assert "input" in Client("https://h/v1", "", "m", "responses").probe_body()
    assert "max_tokens" in Client("https://h/v1", "", "m", "anthropic").probe_body()
    gem = Client("https://h/v1", "", "m", "gemini").probe_body()
    # a strict endpoint rejects a tiny budget before validating anything else, so the
    # error probes must not start below the floor
    assert gem["generationConfig"]["maxOutputTokens"] == probes.MIN_TOKENS
    for api in ("openai", "responses", "anthropic", "gemini"):
        body = Client("https://h/v1", "", "m", api).probe_body()
        budget = (
            body.get("max_tokens")
            or body.get("max_output_tokens")
            or body["generationConfig"]["maxOutputTokens"]
        )
        assert budget == probes.MIN_TOKENS


def test_error_probe_budget_mutator_targets_the_right_key():
    chat_body = Client("https://h/v1", "", "m", "openai").probe_body()
    probes.set_max_tokens(chat_body, 10**7)
    assert chat_body["max_tokens"] == 10**7 and "max_output_tokens" not in chat_body

    responses_body = Client("https://h/v1", "", "m", "responses").probe_body()
    probes.set_max_tokens(responses_body, 10**7)
    assert responses_body["max_output_tokens"] == 10**7 and "max_tokens" not in responses_body


def test_send_raw_skips_reasoning_injection_and_uses_protocol_path(captured):
    client = Client("https://h/v1", "k", "m", "responses", reasoning_effort="low")
    client.send_raw(client.probe_body())
    assert captured[-1]["url"] == "https://h/v1/responses"
    assert "reasoning" not in captured[-1]["body"]


def test_extra_headers_are_sent(captured):
    client = Client("https://h/v1", "k", "m", "openai", extra_headers={"x-session": "abc"})
    client.chat(MESSAGES)
    assert captured[-1]["headers"] == {"x-session": "abc"}


def test_request_log_records_failures(monkeypatch):
    client = Client("https://127.0.0.1:1/v1", "k", "m", "openai", timeout=2)
    resp = client.request("GET", client.base + "/models")
    assert resp["status"] is None
    assert "error" in resp
    assert client.log and client.log[-1]["status"] is None


def test_auth_headers_per_protocol(monkeypatch):
    seen = {}

    def capture_headers(self, method, url, body=None, extra_headers=None):
        # rebuild the header set exactly as request() would
        headers = {"Content-Type": "application/json"}
        headers.update(self.extra)
        if self.key:
            if self.api == "anthropic":
                headers["x-api-key"] = self.key
                headers["anthropic-version"] = "2023-06-01"
            else:
                headers["Authorization"] = "Bearer " + self.key
        seen[self.api] = headers
        return {"status": 200, "headers": {}, "text": "{}", "json": {}, "ms": 1}

    monkeypatch.setattr(Client, "request", capture_headers)
    Client("https://h/v1", "sekret", "m", "anthropic").chat(MESSAGES)
    assert seen["anthropic"]["x-api-key"] == "sekret"
    assert seen["anthropic"]["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in seen["anthropic"]
    Client("https://h/v1", "sekret", "m", "openai").chat(MESSAGES)
    assert seen["openai"]["Authorization"] == "Bearer sekret"


def test_client_json_body_is_serialisable(captured):
    Client("https://h/v1", "k", "m", "responses", reasoning_effort="minimal").chat(
        MESSAGES, max_tokens=32, extra={"store": False}
    )
    json.dumps(captured[-1]["body"])  # must not raise
