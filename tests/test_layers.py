"""Layers, exercised end to end against the local mock endpoint (no network)."""

import pytest

from model_fingerprint import layers
from model_fingerprint.client import Client

PROTOCOLS = ("openai", "responses", "anthropic", "gemini")


def make_client(mock, api="openai", model="mock-alpha", **kwargs):
    return Client(mock.base_url, "test-key", model, api, **kwargs)


@pytest.mark.parametrize("api", PROTOCOLS)
def test_transport_reports_models_and_headers(mock, api):
    client = make_client(mock, api)
    out = layers.transport(client)
    models = out["/models"]
    assert models["status"] == 200
    assert models["model_count"] == 3
    assert "mock-alpha" in models["model_ids"]
    assert "id" in models["model_fields"]
    assert models["headers"]["x-mock-server"] == "mock-endpoint/1.0"
    # a path the endpoint does not implement is recorded, not raised
    assert out["/model/info"]["status"] == 404
    assert out["/health"]["status"] == 200


@pytest.mark.parametrize("api", PROTOCOLS)
def test_tokenizer_vector_is_complete_and_stable(mock, api):
    out = layers.tokenizer(make_client(mock, api), quick=True)
    assert out["baseline_a"] == 5  # count_tokens("a") in the mock
    assert out["probes_usable"] == out["probes_attempted"] > 0
    assert out["repeat_stable"] is True
    assert len(set(out["repeat_3x"])) == 1
    for name, entry in out["probes"].items():
        if name == "__baseline_a":
            continue
        assert entry["norm"] == entry["raw"] - out["baseline_a"]
    assert out["probes"]["english_common"]["norm"] > out["probes"]["spaces_run"]["norm"]


def test_tokenizer_reports_unusable_when_usage_is_missing(mock_factory):
    mock = mock_factory(omit_usage=True)
    out = layers.tokenizer(make_client(mock), quick=True)
    assert out["baseline_a"] is None
    assert out["probes_usable"] == 0
    assert out["probes_attempted"] > 0
    assert all(v["norm"] is None for k, v in out["probes"].items() if k != "__baseline_a")


def test_errors_layer_captures_signatures(mock):
    out = layers.errors(make_client(mock), quick=True)
    assert out["bad_model"]["status"] == 404
    assert "type=invalid_request_error" in out["bad_model"]["signature"]
    assert "model_not_found" in out["bad_model"]["error_text"]
    assert out["engine_params"]["status"] == 400
    assert "repetition_penalty" in out["engine_params"]["error_text"]
    assert out["temperature_range"]["status"] == 400
    assert out["huge_max_tokens"]["status"] == 400
    assert out["huge_max_tokens"]["signature"]  # "maximum is 65536" style evidence
    assert out["huge_max_tokens"]["headers"]  # failing calls keep their headers


def test_errors_layer_full_set_marks_missing_ones(mock):
    out = layers.errors(make_client(mock), quick=False)
    assert len(out) == 9
    assert out["bad_role"]["status"] == 400
    assert "messages[0].role" in out["bad_role"]["error_text"]
    assert out["empty_messages"]["status"] == 400
    assert out["negative_max_tokens"]["status"] == 400
    assert out["unknown_top_param"]["status"] == 400
    assert out["parts_content"]["status"] == 200  # valid on this endpoint, recorded as such


def test_capabilities_chat_protocol(mock):
    out = layers.capabilities(make_client(mock, "openai"))
    assert out["tools"]["accepted"] is True
    assert out["logprobs"]["accepted"] is True
    assert out["stop_seq"]["accepted"] is True
    assert out["developer_role"]["accepted"] is True
    assert out["json_object"]["accepted"] is False
    assert out["seed"]["accepted"] is False
    assert out["n_multiple"]["accepted"] is False
    assert out["engine_params"]["accepted"] is False
    assert out["greedy_deterministic"]["same_text"] is True
    assert out["greedy_deterministic"]["informative"] is True
    assert out["greedy_deterministic"]["text"] == "alpha"
    assert out["self_id"]["text"]  # soft evidence is captured, whatever it says
    assert out["greedy_deterministic"]["finish"] == "stop"


def test_capabilities_responses_protocol(mock):
    out = layers.capabilities(make_client(mock, "responses"))
    assert out["tools"]["accepted"] is True
    assert out["json_object"]["accepted"] is True
    assert out["json_schema"]["accepted"] is True
    assert out["parallel_tool_calls"]["accepted"] is True
    assert out["store_false"]["accepted"] is True
    assert out["top_logprobs"]["accepted"] is False
    assert "include" in out["top_logprobs"]["error"]
    assert out["greedy_deterministic"]["reasoning_tokens"] == 4
    assert out["greedy_deterministic"]["finish"] == "completed"


def test_capabilities_anthropic_and_gemini(mock):
    anthropic = layers.capabilities(make_client(mock, "anthropic"))
    assert anthropic["greedy_deterministic"]["text"] == "alpha"
    assert anthropic["greedy_deterministic"]["finish"] == "end_turn"
    gemini = layers.capabilities(make_client(mock, "gemini"))
    assert gemini["greedy_deterministic"]["text"] == "alpha"
    assert gemini["greedy_deterministic"]["finish"] == "STOP"


def test_limits_walks_until_rejection(mock):
    out = layers.limits(make_client(mock))
    assert out["max_output_tokens"] == 65536
    assert out["first_rejection"]["attempted"] == 131072
    assert out["first_rejection"]["status"] == 400
    assert "maximum is 65536" in out["first_rejection"]["error"]
    assert out["ladder"]


def test_limits_when_everything_is_accepted(mock_factory):
    mock = mock_factory(max_output_cap=10**9)
    out = layers.limits(make_client(mock))
    assert out["max_output_tokens"] == 262144
    assert out["first_rejection"] is None


@pytest.mark.parametrize("api", PROTOCOLS)
def test_behaviour_cells_are_informative(mock, api):
    out = layers.behaviour(make_client(mock, api), samples=2, cells={"fav_number", "letter"})
    assert out["samples_per_cell"] == 2
    assert set(out["cells"]) == {"fav_number", "letter"}
    for cell in out["cells"].values():
        assert cell["informative"] is True
        assert cell["n"] == 2
        assert cell["notes"] == []
        assert cell["mode"] != "<empty>"
        assert sum(cell["dist"].values()) == 2


def test_behaviour_entropy_reflects_spread(mock):
    out = layers.behaviour(make_client(mock), samples=3, cells={"fav_number"})
    cell = out["cells"]["fav_number"]
    # one deterministic answer -> zero entropy
    assert cell["entropy_bits"] == 0.0


def test_behaviour_cell_subset_and_defaults(mock):
    out = layers.behaviour(make_client(mock), samples=1, cells=None)
    assert len(out["cells"]) == 12
    out = layers.behaviour(make_client(mock), samples=1, cells={"hotdog"})
    assert set(out["cells"]) == {"hotdog"}


def test_context_probe_stops_at_first_failure(mock_factory):
    mock = mock_factory(max_context_tokens=20000)
    client = Client(mock.base_url, "test-key", "mock-alpha", "openai")
    out = layers.context(client, sizes=(8192, 131072))
    assert len(out) == 2
    assert out[0]["status"] == 200
    assert out[1]["status"] == 400
    assert "maximum context length is 20000" in out[1]["error"]
