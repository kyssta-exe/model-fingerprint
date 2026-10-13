"""The six probe layers. Each returns plain JSON-serialisable evidence and
reports its own gaps explicitly instead of guessing."""

import math
import time
from collections import Counter

from . import parse
from .probes import (
    BEHAVIOUR_CELLS,
    ERROR_PROBES,
    QUICK_ERRORS,
    QUICK_PROBES,
    SELF_ID_PROMPT,
    TOKEN_PROBES,
)

REPEAT_PROBE_TEXT = "Test test test test."


def transport(client):
    """API surface: model listing, headers, latency, plus a few common health paths."""
    out = {}
    for path in ("/models", "/model/info", "/health"):
        resp = client.request("GET", client.base + path)
        entry = {
            "status": resp["status"],
            "ms": resp["ms"],
            "headers": parse.hdr_dna(resp["headers"]),
            "body_head": (resp["text"] or "")[:400],
        }
        payload = resp.get("json")
        if (
            path == "/models"
            and isinstance(payload, dict)
            and isinstance(payload.get("data"), list)
        ):
            models = [m for m in payload["data"] if isinstance(m, dict)]
            entry["model_count"] = len(models)
            entry["model_ids"] = [m.get("id") for m in models[:40]]
            fields = set()
            for m in models[:20]:
                fields.update(m.keys())
            entry["model_fields"] = sorted(fields)
        out[path] = entry
    return out


def errors(client, quick=False):
    """Forced failures - upstream codes and wording leak through most gateways."""
    out = {}
    wanted = QUICK_ERRORS if quick else {name for name, _, _ in ERROR_PROBES}
    for name, note, mutate in ERROR_PROBES:
        if name not in wanted:
            continue
        body = client.probe_body()
        try:
            mutate(body)
        except Exception as exc:  # noqa: BLE001 - a broken mutator is reported, not hidden
            out[name] = {"note": note, "mutator_error": "%s: %s" % (type(exc).__name__, exc)}
            continue
        resp = client.send_raw(body)
        out[name] = {
            "note": note,
            "status": resp["status"],
            "ms": resp["ms"],
            "signature": parse.err_signature(resp),
            "error_text": parse.err_text(resp),
            "headers": (parse.hdr_dna(resp["headers"]) if resp["status"] not in (200, 201) else {}),
        }
    return out


def tokenizer(client, quick=False):
    """Shift-invariant prompt-token counts: count(text) - count('a').

    The subtraction cancels hidden system prompts and chat templates, so two
    hosts wrapping the same tokenizer match where raw counts never do.
    """
    wanted = QUICK_PROBES if quick else {name for name, _ in TOKEN_PROBES}
    probes = {}
    baseline_resp = client.chat([{"role": "user", "content": "a"}], max_tokens=1, temperature=0)
    baseline = parse.usage_of(baseline_resp)
    probes["__baseline_a"] = {
        "raw": baseline,
        "status": baseline_resp["status"],
        "resp_keys": parse.resp_keys(baseline_resp),
    }
    for name, text in TOKEN_PROBES:
        if name not in wanted:
            continue
        resp = client.chat([{"role": "user", "content": text}], max_tokens=1, temperature=0)
        raw = parse.usage_of(resp)
        probes[name] = {
            "raw": raw,
            "status": resp["status"],
            "norm": (raw - baseline) if (raw is not None and baseline is not None) else None,
            "text_len": len(text),
        }
        time.sleep(0.05)
    repeats = []
    for _ in range(3):
        resp = client.chat(
            [{"role": "user", "content": REPEAT_PROBE_TEXT}], max_tokens=1, temperature=0
        )
        repeats.append(parse.usage_of(resp))
        time.sleep(0.05)
    usable = sum(
        1
        for key, value in probes.items()
        if key != "__baseline_a" and value.get("norm") is not None
    )
    return {
        "baseline_a": baseline,
        "probes": probes,
        "repeat_3x": repeats,
        "repeat_stable": len(set(repeats)) == 1,
        "probes_usable": usable,
        "probes_attempted": max(0, len(probes) - 1),
    }


def capabilities(client, generous=256):
    """Capability matrix, with protocol-appropriate payload shapes.

    Sending the chat-completions shape to a Responses-style API (or the reverse)
    produces a rejection that has nothing to do with model capability - the probe
    must match the wire protocol or it measures nothing.
    """
    out = {}

    def record(name, resp, ok_fn=None):
        accepted = resp["status"] in (200, 201)
        if accepted and ok_fn:
            accepted = ok_fn(resp.get("json") or {})
        status, incomplete = parse.response_status(resp)
        out[name] = {
            "status": resp["status"],
            "accepted": bool(accepted),
            "finish": status,
            "incomplete": incomplete,
            "reasoning_tokens": parse.reasoning_tokens(resp),
            "resp_keys": parse.resp_keys(resp),
            "error": None if resp["status"] in (200, 201) else parse.err_text(resp, 200),
        }

    simple = [{"role": "user", "content": "hi"}]
    if client.api == "responses":
        record(
            "tools",
            client.chat(
                simple,
                max_tokens=generous,
                extra={
                    "tools": [
                        {
                            "type": "function",
                            "name": "now",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ]
                },
            ),
        )
        record(
            "json_object",
            client.chat(
                simple, max_tokens=generous, extra={"text": {"format": {"type": "json_object"}}}
            ),
        )
        record(
            "json_schema",
            client.chat(
                simple,
                max_tokens=generous,
                extra={
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "x",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {"a": {"type": "integer"}},
                                "required": ["a"],
                                "additionalProperties": False,
                            },
                        }
                    }
                },
            ),
        )
        record("top_logprobs", client.chat(simple, max_tokens=generous, extra={"top_logprobs": 3}))
        record(
            "parallel_tool_calls",
            client.chat(simple, max_tokens=generous, extra={"parallel_tool_calls": False}),
        )
        record("store_false", client.chat(simple, max_tokens=generous, extra={"store": False}))
        record(
            "include_reasoning",
            client.chat(
                simple, max_tokens=generous, extra={"include": ["reasoning.encrypted_content"]}
            ),
        )
    else:
        record(
            "tools",
            client.chat(
                simple,
                max_tokens=generous,
                extra={
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "now",
                                "description": "time",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ]
                },
            ),
        )
        record(
            "json_object",
            client.chat(
                simple, max_tokens=generous, extra={"response_format": {"type": "json_object"}}
            ),
        )
        record(
            "json_schema",
            client.chat(
                simple,
                max_tokens=generous,
                extra={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "x",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {"a": {"type": "integer"}},
                                "required": ["a"],
                                "additionalProperties": False,
                            },
                        },
                    }
                },
            ),
        )
        record(
            "logprobs",
            client.chat(simple, max_tokens=generous, extra={"logprobs": True, "top_logprobs": 3}),
        )
        record("seed", client.chat(simple, max_tokens=generous, extra={"seed": 42}))
        record("n_multiple", client.chat(simple, max_tokens=generous, extra={"n": 3}))
        record(
            "stop_seq",
            client.chat(
                [{"role": "user", "content": "count 1 2 3"}],
                max_tokens=generous,
                extra={"stop": ["3"]},
            ),
        )
        record(
            "developer_role",
            client.chat(
                [{"role": "developer", "content": "be terse"}, *simple], max_tokens=generous
            ),
        )
        record(
            "engine_params",
            client.chat(
                simple,
                max_tokens=generous,
                extra={"repetition_penalty": 1.2, "min_tokens": 2, "ignore_eos": False},
            ),
        )

    first = client.chat(
        [{"role": "user", "content": "Say the word: alpha"}], max_tokens=generous, temperature=0
    )
    second = client.chat(
        [{"role": "user", "content": "Say the word: alpha"}], max_tokens=generous, temperature=0
    )
    text_a, text_b = parse.text_of(first), parse.text_of(second)
    informative = bool(text_a.strip()) and first["status"] in (200, 201)
    status, incomplete = parse.response_status(first)
    out["greedy_deterministic"] = {
        "same_text": text_a == text_b,
        "informative": informative,
        "note": None
        if informative
        else (
            "uninformative: empty/failed response (finish=%s incomplete=%s) - "
            "do not read as determinism" % (status, incomplete)
        ),
        "text": text_a[:120],
        "finish": status,
        "incomplete": incomplete,
        "reasoning_tokens": parse.reasoning_tokens(first),
        "resp_keys": parse.resp_keys(first),
        "status": first["status"],
    }

    self_id = client.chat(
        [{"role": "user", "content": SELF_ID_PROMPT}], max_tokens=max(generous, 1024), temperature=0
    )
    out["self_id"] = {
        "text": parse.text_of(self_id)[:300],
        "status": self_id["status"],
        "finish": parse.response_status(self_id)[0],
        "reasoning_tokens": parse.reasoning_tokens(self_id),
    }
    return out


def limits(client, ladder=(4096, 8192, 16384, 32768, 65536, 131072, 262144)):
    """Output-token ceiling by walking up until the endpoint refuses."""
    out = {"max_output_tokens": None, "first_rejection": None, "ladder": list(ladder)}
    for candidate in ladder:
        resp = client.chat([{"role": "user", "content": "hi"}], max_tokens=candidate, temperature=0)
        if resp["status"] in (200, 201):
            out["max_output_tokens"] = candidate
        else:
            out["first_rejection"] = {
                "attempted": candidate,
                "status": resp["status"],
                "error": parse.err_text(resp, 300),
                "signature": parse.err_signature(resp),
            }
            break
    return out


def context(client, sizes=(8192, 32768, 131072, 262144, 1048576)):
    """Context ceiling by escalating filler until the endpoint refuses (costly)."""
    out = []
    for approx_tokens in sizes:
        filler = "pad " * approx_tokens
        resp = client.chat(
            [{"role": "user", "content": filler + "\nReply with the single word OK."}],
            max_tokens=2,
            temperature=0,
        )
        out.append(
            {
                "approx_tokens": approx_tokens,
                "status": resp["status"],
                "error": None if resp["status"] in (200, 201) else parse.err_text(resp, 300),
                "signature": parse.err_signature(resp),
                "text": parse.text_of(resp)[:40],
            }
        )
        if resp["status"] not in (200, 201):
            break
        time.sleep(0.2)
    return out


def behaviour(client, samples, cells=None, budget=512):
    """Short-answer distributions. A cell the endpoint cut off mid-thought is
    recorded as uninformative - reasoning models do exactly that on a small budget."""
    out = {"samples_per_cell": samples, "output_budget": budget, "cells": {}}
    for cell_id, prompt in BEHAVIOUR_CELLS:
        if cells and cell_id not in cells:
            continue
        answers, notes, reason_counts = [], [], []
        for _ in range(samples):
            resp = client.chat(
                [{"role": "user", "content": prompt}], max_tokens=budget, temperature=1.0
            )
            text = parse.text_of(resp)
            _, incomplete = parse.response_status(resp)
            spent = parse.reasoning_tokens(resp)
            if spent is not None:
                reason_counts.append(spent)
            if not text.strip() and incomplete:
                notes.append("cut off (%s) - raise --cell-budget" % incomplete)
            answers.append(parse.norm_answer(text))
            time.sleep(0.05)
        dist = dict(Counter(answers).most_common())
        entropy = 0.0
        if answers:
            entropy = -sum(
                (count / len(answers)) * math.log2(count / len(answers)) for count in dist.values()
            )
        out["cells"][cell_id] = {
            "n": len(answers),
            "dist": dist,
            "mode": answers[0] if answers else None,
            "entropy_bits": round(max(0.0, entropy), 3),
            "informative": any(a != "<empty>" for a in answers),
            "reasoning_tokens": reason_counts,
            "notes": sorted(set(notes)),
        }
    return out
