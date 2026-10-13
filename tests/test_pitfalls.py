"""The traps that make a fingerprint report lie. Each one is a regression test."""

from model_fingerprint import layers
from model_fingerprint.client import Client


def make_client(mock, api="responses", **kwargs):
    return Client(mock.base_url, "test-key", "mock-alpha", api, **kwargs)


def test_reasoning_model_burning_the_budget_is_uninformative(mock_factory):
    """A cut-off answer is a missing measurement, not a distribution."""
    mock = mock_factory(reasoning_model=True, reasoning_min_budget=1024)
    out = layers.behaviour(make_client(mock), samples=2, cells={"fav_number"}, budget=512)
    cell = out["cells"]["fav_number"]
    assert cell["informative"] is False
    assert cell["dist"] == {"<empty>": 2}
    assert any("cut off (max_output_tokens)" in note for note in cell["notes"])
    assert cell["reasoning_tokens"] and cell["reasoning_tokens"][0] > 0


def test_empty_output_is_not_determinism(mock_factory):
    mock = mock_factory(reasoning_model=True, reasoning_min_budget=1024)
    out = layers.capabilities(make_client(mock))
    greedy = out["greedy_deterministic"]
    assert greedy["same_text"] is True  # trivially "equal" - both empty
    assert greedy["informative"] is False  # ...and that is why it is flagged
    assert "do not read as determinism" in greedy["note"]
    assert greedy["incomplete"] == "max_output_tokens"


def test_larger_budget_or_pinned_effort_restores_informative_cells(mock_factory):
    generous = mock_factory(reasoning_model=True, reasoning_min_budget=1024)
    out = layers.behaviour(make_client(generous), samples=1, cells={"fav_number"}, budget=2048)
    assert out["cells"]["fav_number"]["informative"] is True

    pinned = mock_factory(reasoning_model=True, reasoning_min_budget=1024)
    client = make_client(pinned, reasoning_effort="minimal")
    out = layers.capabilities(client, generous=256)
    assert out["greedy_deterministic"]["informative"] is True


def test_missing_usage_is_uninformative_not_a_mismatch(mock_factory):
    mock = mock_factory(omit_usage=True)
    out = layers.tokenizer(make_client(mock, "openai"), quick=True)
    assert out["probes_usable"] == 0
    assert out["probes_attempted"] == 10
    assert all(v["norm"] is None for k, v in out["probes"].items() if k != "__baseline_a")


def test_required_routing_header_breaks_everything_until_passed(mock_factory):
    mock = mock_factory(require_session_header=True)
    blocked = layers.transport(make_client(mock, "openai"))
    assert blocked["/models"]["status"] == 400
    assert "MissingSessionID" in blocked["/models"]["body_head"]

    client = make_client(mock, "openai", extra_headers={"x-mock-session": "fp-1"})
    allowed = layers.transport(client)
    assert allowed["/models"]["status"] == 200
    assert allowed["/models"]["model_count"] == 3


def test_protocol_mismatch_is_reported_as_an_error_not_a_capability(mock):
    """Chat-only parameters against a Responses endpoint are never sent at all."""
    out = layers.capabilities(make_client(mock, "responses"))
    assert "logprobs" not in out  # chat-completions-only probe, skipped by design
    assert "seed" not in out
    assert "engine_params" not in out
    # the responses-native equivalents are the ones that carry information
    assert out["json_schema"]["accepted"] is True
    assert out["top_logprobs"]["accepted"] is False
    assert "include" in out["top_logprobs"]["error"]


def test_chat_protocol_probes_the_chat_only_parameters(mock):
    out = layers.capabilities(make_client(mock, "openai"))
    for name in ("logprobs", "seed", "n_multiple", "engine_params"):
        assert name in out
    assert "top_logprobs" not in out


def test_raw_counts_differ_from_normalised_when_a_template_is_hidden(mock):
    out = layers.tokenizer(make_client(mock, "openai"), quick=True)
    baseline = out["baseline_a"]
    assert baseline > 0
    for name, entry in out["probes"].items():
        if name == "__baseline_a":
            continue
        assert entry["raw"] != entry["norm"] or entry["raw"] == baseline


def test_repeat_stability_detects_a_drifting_endpoint(mock_factory):
    mock = mock_factory()
    out = layers.tokenizer(make_client(mock, "openai"), quick=True)
    assert out["repeat_stable"] is True
    assert len(set(out["repeat_3x"])) == 1


def test_context_probe_reports_the_refusal_text(mock_factory):
    mock = mock_factory(max_context_tokens=20000)
    client = make_client(mock, "openai")
    out = layers.context(client, sizes=(131072,))
    assert out[0]["status"] == 400
    assert "maximum context length is 20000" in out[0]["error"]
