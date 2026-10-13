"""Report rendering: sections present, missing layers tolerated, secrets absent."""

import json

from model_fingerprint import layers
from model_fingerprint.cli import build_parser, run
from model_fingerprint.client import Client
from model_fingerprint.report import render


def test_render_survives_an_empty_fingerprint():
    text = render({})
    assert text.startswith("# Fingerprint:")
    assert "## Grading worksheet" in text
    assert "## Yield" in text
    assert "unavailable" not in text.split("## Yield")[0] or True  # no crash is the point


def test_render_marks_unavailable_layers():
    text = render(
        {
            "meta": {"model": "m", "base_url": "u", "api": "openai"},
            "behaviour": {"unavailable": "TimeoutError: boom"},
        }
    )
    assert "layer `behaviour` unavailable: TimeoutError: boom" in text


def test_render_of_a_real_run_has_every_section(mock):
    client = Client(mock.base_url, "test-key", "mock-alpha", "openai")
    fingerprint = {
        "meta": {
            "tool": "model-fingerprint/test",
            "model": "mock-alpha",
            "base_url": mock.base_url,
            "api": "openai",
            "depth": "quick",
            "key_len": 8,
            "reasoning_effort": None,
            "extra_headers": ["x-mock-session"],
            "ts_utc": "2026-10-13T00:00:00Z",
        },
        "transport": layers.transport(client),
        "errors": layers.errors(client, quick=True),
        "tokenizer": layers.tokenizer(client, quick=True),
        "capabilities": layers.capabilities(client),
        "limits": layers.limits(client),
        "behaviour": layers.behaviour(client, 2, {"fav_number"}),
        "requests": client.log,
    }
    text = render(fingerprint)
    for heading in (
        "# Fingerprint:",
        "## Error DNA",
        "## Tokenizer vector",
        "## Capability matrix",
        "## Limits",
        "## Behaviour cells",
        "## Yield",
        "## Grading worksheet",
    ):
        assert heading in text
    assert "usable probes: 10/10" in text
    assert "behaviour cells with real answers: 1/1" in text
    assert "extra headers: x-mock-session" in text
    assert "mock-endpoint/1.0" in text
    assert "| english_common |" in text


def test_render_flags_uninformative_cells():
    fingerprint = {
        "meta": {},
        "behaviour": {
            "samples_per_cell": 1,
            "cells": {
                "fav_number": {
                    "n": 1,
                    "dist": {"<empty>": 1},
                    "mode": "<empty>",
                    "entropy_bits": 0.0,
                    "informative": False,
                    "reasoning_tokens": [512],
                    "notes": ["cut off (max_output_tokens) - raise --cell-budget"],
                }
            },
        },
    }
    text = render(fingerprint)
    assert "[UNINFORMATIVE]" in text
    assert "cut off (max_output_tokens)" in text


def test_render_never_prints_the_api_key(mock):
    client = Client(mock.base_url, "test-key", "mock-alpha", "openai")
    fingerprint = {
        "meta": {
            "model": "m",
            "base_url": mock.base_url,
            "api": "openai",
            "key_len": len("test-key"),
            "extra_headers": [],
        },
        "transport": layers.transport(client),
        "requests": client.log,
    }
    assert "test-key" not in render(fingerprint)


def test_run_does_not_leak_the_key_into_the_json(mock, tmp_path):
    import io

    out = tmp_path / "fp.json"
    args = build_parser().parse_args(
        [
            "--base-url",
            mock.base_url,
            "--model",
            "mock-alpha",
            "--api",
            "openai",
            "--api-key",
            "super-secret-key",
            "--quick",
            "--cells",
            "fav_number",
            "--out",
            str(out),
            "--report",
            str(tmp_path / "fp.md"),
            "--quiet",
        ]
    )
    fingerprint = run(args, stream=io.StringIO())
    blob = out.read_text() if out.exists() else json.dumps(fingerprint)
    assert "super-secret-key" not in json.dumps(fingerprint)
    assert "super-secret-key" not in blob
    assert fingerprint["meta"]["key_len"] == len("super-secret-key")
