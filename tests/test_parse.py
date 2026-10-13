"""Parsing layer: every protocol's text, usage, status and error shapes."""

from model_fingerprint import parse


def test_norm_answer_collapses_variants():
    assert parse.norm_answer(" Blue. ") == "blue"
    assert parse.norm_answer("blue\n\nActually, teal.") == "blue"
    assert parse.norm_answer("Yes, absolutely") == "yes"
    assert parse.norm_answer("no, because bread") == "no"
    assert parse.norm_answer("**42**") == "42"
    assert parse.norm_answer("") == "<empty>"
    assert parse.norm_answer("   \n ") == "<empty>"
    assert parse.norm_answer("x" * 50) == "x" * 24


def test_usage_of_across_protocols():
    assert parse.usage_of({"json": {"usage": {"prompt_tokens": 11}}}) == 11
    assert parse.usage_of({"json": {"usage": {"input_tokens": 12}}}) == 12
    assert parse.usage_of({"json": {"usageMetadata": {"promptTokenCount": 13}}}) == 13
    assert parse.usage_of({"json": {"usage": {}}}) is None
    assert parse.usage_of({"json": None}) is None
    assert parse.usage_of({}) is None
    # a non-integer must not be mistaken for a measurement
    assert parse.usage_of({"json": {"usage": {"prompt_tokens": "11"}}}) is None


def test_reasoning_tokens_across_protocols():
    assert (
        parse.reasoning_tokens(
            {"json": {"usage": {"output_tokens_details": {"reasoning_tokens": 328}}}}
        )
        == 328
    )
    assert (
        parse.reasoning_tokens(
            {"json": {"usage": {"completion_tokens_details": {"reasoning_tokens": 7}}}}
        )
        == 7
    )
    assert parse.reasoning_tokens({"json": {"usageMetadata": {"thoughtsTokenCount": 9}}}) == 9
    assert parse.reasoning_tokens({"json": {"usage": {}}}) is None


def test_response_status_per_protocol():
    assert parse.response_status(
        {"status": 200, "json": {"choices": [{"finish_reason": "stop"}]}}
    ) == ("stop", None)
    assert parse.response_status(
        {
            "status": 200,
            "json": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        }
    ) == ("incomplete", "max_output_tokens")
    assert parse.response_status({"status": 200, "json": {"stop_reason": "end_turn"}}) == (
        "end_turn",
        None,
    )
    assert parse.response_status(
        {"status": 200, "json": {"candidates": [{"finishReason": "STOP"}]}}
    ) == ("STOP", None)
    assert parse.response_status({"status": 400, "json": {"error": {}}}) == (None, None)
    assert parse.response_status({"status": None, "json": None}) == (None, None)


def test_text_of_per_protocol():
    assert parse.text_of({"json": {"choices": [{"message": {"content": "42"}}]}}) == "42"
    assert parse.text_of({"json": {"choices": [{"text": "legacy"}]}}) == "legacy"
    assert parse.text_of({"json": {"content": [{"type": "text", "text": "claude"}]}}) == "claude"
    assert (
        parse.text_of({"json": {"candidates": [{"content": {"parts": [{"text": "gem"}]}}]}})
        == "gem"
    )
    assert parse.text_of({"json": {"output_text": "shortcut"}}) == "shortcut"
    assert (
        parse.text_of(
            {
                "json": {
                    "output": [
                        {"type": "reasoning", "summary": []},
                        {"type": "message", "content": [{"type": "output_text", "text": "42"}]},
                    ]
                }
            }
        )
        == "42"
    )
    # an incomplete reasoning-only response yields no text, not a crash
    assert (
        parse.text_of({"json": {"status": "incomplete", "output": [{"type": "reasoning"}]}}) == ""
    )
    assert parse.text_of({"json": None}) == ""
    assert parse.text_of({}) == ""


def test_resp_keys_and_error_text():
    assert parse.resp_keys({"json": {"b": 1, "a": 2}}) == ["a", "b"]
    assert parse.resp_keys({"json": None}) == []
    assert parse.err_text({"json": {"error": {"message": "x"}}, "text": ""}) == ""
    assert parse.err_text({"text": "  plain body  "}) == "plain body"
    assert parse.err_text({"error": "TypeError: boom", "text": "ignored"}) == "TypeError: boom"


def test_err_signature_finds_vendor_codes():
    vendor = {
        "text": '{"error":{"message":"Upstream request failed: [1210] The temperature '
        'parameter is illegal."}}',
        "json": {"error": {"message": "Upstream request failed: [1210] ..."}},
    }
    sig = parse.err_signature(vendor)
    assert any("[1210]" in part for part in sig)

    typed = {
        "text": "",
        "json": {
            "error": {"type": "invalid_request_error", "param": "max_output_tokens", "code": None}
        },
    }
    sig = parse.err_signature(typed)
    assert "type=invalid_request_error" in sig
    assert "param=max_output_tokens" in sig
    assert all("code=" not in part for part in sig)

    leaking = {"text": "litellm.BadRequestError: Upstream HTTP 400, max_tokens is too large"}
    assert any("Error" in part for part in parse.err_signature(leaking))
    assert parse.err_signature({"text": "everything is fine"}) == []


def test_hdr_dna_drops_noise_and_truncates():
    headers = {
        "Date": "today",
        "Content-Length": "12",
        "Connection": "keep-alive",
        "Server": "cloudflare",
        "x-log-id": "y" * 400,
    }
    dna = parse.hdr_dna(headers)
    assert set(dna) == {"server", "x-log-id"}
    assert len(dna["x-log-id"]) == 120
