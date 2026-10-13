"""Human-readable report rendered from a fingerprint dict."""

import json

HTTP_OK = 200


def _line_list(items):
    return "\n".join(items)


def render(fp):
    """Render a fingerprint dict as markdown. Never raises on missing layers."""
    out = []
    meta = fp.get("meta") or {}
    out.append("# Fingerprint: `%s` @ %s" % (meta.get("model"), meta.get("base_url")))
    out.append("")
    out.append(
        "- protocol: `%s`  depth: %s  key_len: %s  reasoning_effort: %s  %s"
        % (
            meta.get("api"),
            meta.get("depth"),
            meta.get("key_len"),
            meta.get("reasoning_effort"),
            meta.get("ts_utc"),
        )
    )
    if meta.get("extra_headers"):
        out.append("- extra headers: %s" % ", ".join(meta["extra_headers"]))

    transport = fp.get("transport") or {}
    models = transport.get("/models") if isinstance(transport, dict) else None
    if isinstance(models, dict):
        ids = models.get("model_ids") or []
        out.append(
            "- `/models`: HTTP %s, %s models%s"
            % (
                models.get("status"),
                models.get("model_count", "?"),
                (" e.g. " + ", ".join(ids[:6])) if ids else "",
            )
        )
        headers = models.get("headers") or {}
        out.append(
            "- response headers: %s"
            % (", ".join("%s: %s" % (k, v) for k, v in sorted(headers.items())) or "none/expired")
        )

    out.append("")
    out.append("## Error DNA (serving-stack evidence)")
    error_layer = fp.get("errors") or {}
    if not error_layer:
        out.append("- unavailable")
    for name in sorted(error_layer):
        entry = error_layer[name]
        out.append(
            "- `%s` (%s): HTTP %s %s"
            % (name, entry.get("note"), entry.get("status"), "|".join(entry.get("signature") or []))
        )
        if entry.get("error_text"):
            out.append("  - %s" % entry["error_text"].replace("\n", " ")[:260])
        if entry.get("mutator_error"):
            out.append("  - mutator error: %s" % entry["mutator_error"])

    tokenizer = fp.get("tokenizer") or {}
    out.append("")
    out.append(
        "## Tokenizer vector (baseline `a` = %s, normalised = raw - baseline)"
        % (tokenizer.get("baseline_a"))
    )
    out.append("")
    out.append("| probe | raw | norm |")
    out.append("|---|---|---|")
    probes = tokenizer.get("probes") or {}
    for name in sorted(probes):
        if name == "__baseline_a":
            continue
        entry = probes[name]
        out.append("| %s | %s | %s |" % (name, entry.get("raw"), entry.get("norm")))
    out.append("")
    out.append(
        "- usable probes: %s/%s; repeat 3x = %s (stable: %s)"
        % (
            tokenizer.get("probes_usable"),
            tokenizer.get("probes_attempted"),
            tokenizer.get("repeat_3x"),
            tokenizer.get("repeat_stable"),
        )
    )

    cap = fp.get("capabilities") or {}
    out.append("")
    out.append("## Capability matrix")
    cap_keys = [k for k in cap if k not in ("greedy_deterministic", "self_id")]
    order = [
        "tools",
        "json_object",
        "json_schema",
        "top_logprobs",
        "logprobs",
        "seed",
        "n_multiple",
        "stop_seq",
        "developer_role",
        "engine_params",
        "parallel_tool_calls",
        "store_false",
        "include_reasoning",
    ]
    for name in sorted(cap_keys, key=lambda x: (order.index(x) if x in order else 99, x)):
        entry = cap.get(name) or {}
        spent = entry.get("reasoning_tokens")
        out.append(
            "- %s: %s (HTTP %s%s)%s%s"
            % (
                name,
                "accepted" if entry.get("accepted") else "rejected",
                entry.get("status"),
                ", reasoning_tokens=%s" % spent if spent is not None else "",
                " finish=%s" % entry.get("finish") if entry.get("finish") else "",
                (" - " + entry["error"].replace("\n", " ")[:160])
                if entry.get("error") and not entry.get("accepted")
                else "",
            )
        )
    greedy = cap.get("greedy_deterministic") or {}
    if greedy:
        out.append(
            "- greedy determinism (same prompt twice): %s (informative=%s)%s"
            % (
                greedy.get("same_text"),
                greedy.get("informative"),
                (" - " + greedy["note"]) if greedy.get("note") else "",
            )
        )
        out.append("- response keys: `%s`" % ", ".join(greedy.get("resp_keys") or []))
    self_id = cap.get("self_id") or {}
    if self_id:
        out.append(
            "- self-id (SOFT evidence, never decisive): %s"
            % ((self_id.get("text") or "n/a").replace("\n", " ")[:220])
        )

    limits = fp.get("limits") or {}
    out.append("")
    out.append("## Limits")
    out.append("- highest accepted max_tokens: %s" % limits.get("max_output_tokens"))
    rejection = limits.get("first_rejection")
    if rejection:
        out.append(
            "- first rejection at %s: HTTP %s %s"
            % (
                rejection.get("attempted"),
                rejection.get("status"),
                (rejection.get("error") or "").replace("\n", " ")[:220],
            )
        )
    if fp.get("context"):
        out.append("- context probe: %s" % json.dumps(fp["context"])[:400])

    behaviour = fp.get("behaviour") or {}
    out.append("")
    out.append("## Behaviour cells (n=%s/cell, temperature=1)" % behaviour.get("samples_per_cell"))
    for cell_id, entry in (behaviour.get("cells") or {}).items():
        top = ", ".join(
            "%s x%d" % (answer, count) for answer, count in list(entry.get("dist", {}).items())[:6]
        )
        suffix = "" if entry.get("informative") else "  [UNINFORMATIVE]"
        out.append("- `%s` H=%.2f: %s%s" % (cell_id, entry.get("entropy_bits", 0.0), top, suffix))
        for note in entry.get("notes") or []:
            out.append("  - %s" % note)

    out.append("")
    out.append("## Yield (how much of this report is actually informative)")
    requests = fp.get("requests") or []
    ok = sum(1 for r in requests if r.get("status") == HTTP_OK)
    cells = behaviour.get("cells") or {}
    cells_ok = sum(1 for c in cells.values() if c.get("informative"))
    out.append(
        "- requests: %d (%d HTTP 200, %d other - error probes are expected to fail)"
        % (len(requests), ok, len(requests) - ok)
    )
    out.append(
        "- tokenizer probes usable: %s/%s"
        % (tokenizer.get("probes_usable"), tokenizer.get("probes_attempted"))
    )
    out.append("- behaviour cells with real answers: %s/%s" % (cells_ok, len(cells)))
    for layer in (
        "transport",
        "errors",
        "tokenizer",
        "capabilities",
        "limits",
        "context",
        "behaviour",
    ):
        entry = fp.get(layer)
        if isinstance(entry, dict) and entry.get("unavailable"):
            out.append("- layer `%s` unavailable: %s" % (layer, entry["unavailable"]))

    out.append("")
    out.append("## Grading worksheet")
    out.append("- hard match? (headers/error codes/tokenizer vector identical to reference) ______")
    out.append("- statistical match? (meanJSD vs reference; thresholds 0.25 / 0.35) ______")
    out.append("- uninformative probes (429/timeout/no usage) ______ of %s" % len(requests))
    out.append(
        "- verdict: CONFIRMED_SAME | LIKELY | UNCERTAIN | UNKNOWN  "
        "(re-test on another date before acting)"
    )
    return _line_list(out) + "\n"
