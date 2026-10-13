---
name: model-fingerprint
description: Use when identifying the real model behind an API/proxy.
license: MIT
version: 0.1.0
---

# Model Fingerprint

Identify which model a black-box OpenAI / OpenAI-Responses / Anthropic / Gemini compatible
endpoint is actually serving: anonymous or renamed model ids, reseller substitution, silent
quantization, gateway-side prompt injection. Works with just a key — no weights, no logits, no
privileged access.

It produces **layered evidence with a confidence grade, never proof**. Report it that way.

Only probe endpoints and keys you own or have explicit written permission to test.

## When to use

- A model name is anonymous, renamed or obviously synthetic and you want to know what is under it
- A reseller or gateway claims `gpt-5` / `claude-opus` and you want to test the claim
- Quality changed on the same model name (silent swap, quantization, new upstream host)
- You need a provider record: context ceiling, output cap, which parameters are accepted
- Do **not** use for benchmark or quality scoring of a known model, or for prompt engineering

## Setup

The tool is a zero-dependency Python package (3.8+, stdlib only). Pick one:

```bash
# from a clone (gives you the `model-fingerprint` command)
git clone https://github.com/kyssta-exe/model-fingerprint.git
cd model-fingerprint && python -m pip install -e .

# or run in place, no install
python -m model_fingerprint --help
```

If this skill was installed by `install.sh --copy`, the package sits next to this file:
`python -m model_fingerprint` works from the skill directory (`PYTHONPATH=.` if needed).

The API key comes from an environment variable or a dotenv file. Never inline it on a command
line and never paste it into a report:

```bash
export TARGET_API_KEY=...                 # or: --env-file ~/.hermes/.env
```

## Run the tests

```bash
python -m pip install pytest
python -m pytest -q          # 80+ tests, fully offline: no network, no key, no spend
```

The suite drives `tests/mock_endpoint.py`, a local fake endpoint speaking all four protocols with
realistic failure modes (routing-header enforcement, output caps, context caps, reasoning models
that eat the budget, missing usage). If a change makes a probe lie, this suite is what catches it —
run it before and after touching anything in `model_fingerprint/`.

## Fingerprint an endpoint

```bash
# smallest run (~40 requests) on an OpenAI-compatible gateway
model-fingerprint --base-url https://gw.example.com/v1 --model mystery-model \
    --api-key-env TARGET_API_KEY --quick --out fp.json --report fp.md

# gateway needing a routing header + a Responses-style API + dotenv key
model-fingerprint --base-url https://gw.example.com/v1 --model mystery-model \
    --api responses --header 'x-session: fp-1' --reasoning-effort minimal \
    --api-key-env TARGET_API_KEY --env-file ~/.hermes/.env \
    --out fp.json --report fp.md
```

Flags that change the answer:

| Flag | Why it matters |
|---|---|
| `--api auto\|openai\|responses\|anthropic\|gemini` | Probe payloads must match the wire protocol, or every rejection measures your own payload |
| `--header 'Name: Value'` | Repeatable. A gateway that wants a session header 400s everything without it |
| `--reasoning-effort minimal\|low\|medium\|high` | Pin thinking effort. It must match the reference run or the distribution layer compares two configurations |
| `--cell-budget N` | Output budget per behaviour cell (default 512). Reasoning models spend it all thinking and return nothing visible |
| `--quick` / default / `--full` | ~40 / ~90 / ~250 requests. Ask before spending on `--full` or `--context-probe` |
| `--cells a,b,c` | Probe a subset of behaviour cells |
| `--api-key-env NAME --env-file PATH` | Keeps the key out of shell history and out of the report |

## Procedure

1. **Pin the target.** Record base_url, the model id *exactly* as the vendor writes it, and key
   length (never the key). *Done when* the id in the report is byte-identical to the vendor's.
2. **Check for an aggregator shortcut first.** If the host is OpenRouter, every response carries a
   generation id; `GET https://openrouter.ai/api/v1/generation?id=<id>` returns the real upstream
   model and provider. One request instead of a scan.
3. **Run the battery** (`--quick` first). *Done when* `fp.json` exists with every layer either
   populated or explicitly marked unavailable — the tool never silently skips a layer.
4. **Read the report** (`fp.md`). Layers: `transport` (models list, headers), `errors` (nine forced
   failures — upstream codes and wording leak through gateways), `tokenizer` (prompt-token counts
   normalised as `count(text) - count("a")`), `capabilities`, `limits`, `behaviour`.
5. **Differential run.** Fingerprint a reference endpoint you trust with the same flags, then compare
   the normalised tokenizer vectors and capability matrices. Match/mismatch per probe, plus the
   count of uninformative probes, is the deliverable — not a vibe.
6. **Grade and write up.** One verdict, layer-separated evidence (serving layer vs model layer),
   the uninformative count, and a re-test date. *Done when* every claim cites a probe result or is
   labelled soft evidence.

## Grading

| Grade | Requires |
|---|---|
| `CONFIRMED_SAME` | normalised tokenizer vector exact-match + consistent error/header layer + distribution match, nothing contradicting |
| `LIKELY_SAME` | hard layers match, statistical layer unavailable |
| `UNCERTAIN` | layers disagree, or a key layer uninformative (429 / no usage) |
| `LIKELY_DIFFERENT` / `CONFIRMED_DIFFERENT` | hard evidence points at another family or stack |
| `UNKNOWN` | open set: no reference match — keep the vector, it is now your reference |

Statistical layer thresholds (published calibration): match ≤ 0.25 < uncertain ≤ 0.35 < mismatch.
Run a split-half check on the target's own samples before trusting any cross-comparison.

## Pitfalls

1. **Token counts are not lineage.** They fingerprint a shared *tokenization stack*. A 24-pair
   holdout study found only 0.50 sensitivity for model-*family* attribution
   ([arXiv:2608.29930](https://arxiv.org/abs/2608.29930)). Say "same tokenizer stack".
2. **Self-identification is the weakest signal.** A persona, system prompt or router rewrites it.
   Never lead with it.
3. **Raw counts drift across hosts.** Hidden system prompts and chat templates add a constant
   offset. Only the normalised vector is comparable.
4. **Uninformative ≠ negative.** A 429, a timeout, a cut-off answer or a missing `usage` block is a
   missing measurement. State the count; never fold it into a score.
5. **A cut-off answer is not a distribution.** `status: incomplete`, `reason: max_output_tokens` and
   `reasoning_tokens` ≈ budget means the model never finished thinking. Raise `--cell-budget` or pin
   `--reasoning-effort minimal`.
6. **Empty output is not determinism.** Two empty responses compare equal and would "confirm"
   greedy determinism; the tool marks those `informative: false`.
7. **Error probes need a valid budget.** Send `max_tokens=1` and a strict endpoint rejects on the
   budget before validating the parameter you meant to test — you then fingerprint your own payload.
8. **Different models share defaults.** Two unrelated models can both answer `42`, `blue`,
   `Canberra`. The behaviour layer only raises confidence *after* the hard layers agree, and
   disagreement there is decisive on its own.
9. **Quantization changes behaviour, not the tokenizer.** Expect family-level hard matches with
   model-level distribution mismatch, and say exactly that.
10. **A gateway can rewrite error text.** Vendor codes leaking through (`[1210]`-style, Python
    exception names, upstream field names) are strong evidence; their absence proves nothing.
11. **Don't hammer.** A full scan is 100–400 requests. Respect 429s, back off, and never merge data
    collected while rate limited into a fingerprint.
12. **Re-test before acting.** Fingerprints drift with routing and model updates.

## Verification

- Self-consistency: one probe repeated three times yields the identical normalised count and mode
  (the report shows `repeat_3x`).
- Round-trip: fingerprint a known endpoint, then fingerprint it again as the "unknown" target — the
  comparison must come back as a match before you trust it on a real unknown.
- Completeness: every layer present or explicitly marked unavailable, with the uninformative-probe
  count stated in the report's `## Yield` section.
- `python -m pytest -q` passes after any change to the probe code.

## References

- `docs/interpretation.md` — serving-stack signature table (vLLM, SGLang, LiteLLM, OpenRouter,
  Azure, Bedrock, Groq, Ollama, llama.cpp), worked example, reporting rules
- Related projects: [llmfd](https://github.com/wofeiwo/llmfd),
  [modelprint](https://github.com/unclecode/modelprint),
  [llm-fingerprint-detector](https://github.com/ToseaAI/llm-fingerprint-detector)
