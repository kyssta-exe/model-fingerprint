# model-fingerprint

**Is that API really serving the model it claims?**

Black-box attribution for any OpenAI / OpenAI-Responses / Anthropic / Gemini compatible
endpoint. No weights, no logits, no privileged access — just a key and a few dozen cheap
requests. It catches anonymous model names, renamed models, reseller substitution, silent
quantization and gateway-side prompt injection.

It produces **layered evidence with a confidence grade, never proof**. Report it that way.

- Zero dependencies — Python standard library only (3.8+), so it runs on a bare VPS
- Four wire protocols handled natively, including the strict Responses-style APIs that reject
  chat-completions payloads wholesale
- Reports its own gaps: a rate-limited, cut-off or usage-less probe is marked *uninformative*,
  never silently folded into a score

Authorisation: only probe endpoints and keys you own or have written permission to test.

## Install

```bash
git clone https://github.com/kyssta-exe/model-fingerprint.git
cd model-fingerprint
python -m pip install -e .          # or just run it in place: python -m model_fingerprint
```

## Use

```bash
# smallest run (~40 requests): anonymous model on an OpenAI-compatible gateway
model-fingerprint --base-url https://gw.example.com/v1 --model mystery-model \
    --api-key-env TARGET_API_KEY --quick --out fp.json

# gateway that needs a routing header, Responses-style API, key from a dotenv file
model-fingerprint --base-url https://gw.example.com/v1 --model mystery-model \
    --api responses --header 'x-session: fp-1' --reasoning-effort minimal \
    --api-key-env TARGET_API_KEY --env-file ~/.hermes/.env \
    --out fp.json --report fp.md
```

Run `python -m model_fingerprint --help` for every flag. The key is read from the environment
(or a dotenv file) and is never printed, written to the output or sent anywhere else.

### Flags worth knowing

| Flag | Why |
|---|---|
| `--api auto\|openai\|responses\|anthropic\|gemini` | Probe payloads must match the wire protocol or the rejections mean nothing |
| `--header 'Name: Value'` | Repeatable; gateways that demand a session/routing header 400 everything without it |
| `--reasoning-effort minimal\|low\|medium\|high` | Pin thinking effort; it must match the reference run or the distribution layer compares two configurations |
| `--cell-budget N` | Output budget per behaviour cell (default 512); reasoning models spend it all thinking and return nothing |
| `--quick` / default / `--full` | ~40 / ~90 / ~250 requests |
| `--context-probe` | Escalating filler to find the real context ceiling (costly) |
| `--cells a,b,c` | Probe a subset of behaviour cells |
| `--api-key-env NAME --env-file PATH` | Keep the key out of shell history and out of the report |

## What it collects

| Layer | Evidence | Nature |
|---|---|---|
| `transport` | `/models` listing, response headers, latency, which paths exist | hard |
| `errors` | nine forced failures: upstream codes, error wording, accepted-vs-rejected params | hard |
| `tokenizer` | prompt-token counts per probe text, normalised `count(text) - count("a")` | hard |
| `capabilities` | tools, json schema, logprobs, seed, `n`, stop, developer role, engine-specific params | hard |
| `limits` | output-token ceiling by walking up until refusal | hard |
| `context` | context ceiling by escalating filler (opt-in) | hard |
| `behaviour` | short-answer distributions per cell + greedy determinism | statistical |

Normalisation matters: hidden system prompts and chat templates add a constant offset, so two
hosts wrapping the *same* tokenizer match on the normalised vector while raw counts never do.

## Grading

| Grade | Requires |
|---|---|
| `CONFIRMED_SAME` | normalised tokenizer vector exact-match + consistent error/header layer + distribution match, nothing contradicting |
| `LIKELY_SAME` | hard layers match, statistical layer unavailable |
| `UNCERTAIN` | layers disagree, or a key layer uninformative (429 / no usage) |
| `LIKELY_DIFFERENT` / `CONFIRMED_DIFFERENT` | hard evidence points at another family or stack |
| `UNKNOWN` | open set: no reference match — keep the vector, it is now your reference |

Thresholds for the statistical layer, from the published calibration: match ≤ 0.25 < uncertain
≤ 0.35 < mismatch. Always run a split-half check on the target's own samples first.

## Read this before trusting a result

`docs/interpretation.md` covers serving-stack signatures (vLLM, SGLang, LiteLLM, OpenRouter,
Azure, Bedrock, Groq, Ollama, llama.cpp), the aggregator shortcut that can identify an upstream
model in one request, and a worked example. The short version:

1. **Token counts are not lineage.** They fingerprint a shared *tokenization stack*; a 24-pair
   holdout study found only 0.50 sensitivity for model-*family* attribution
   ([arXiv:2608.29930](https://arxiv.org/abs/2608.29930)).
2. **Self-identification is the weakest signal** — a persona or system prompt rewrites it.
3. **Uninformative ≠ negative.** Missing usage, a 429 or a timeout is a missing measurement.
4. **Different models share defaults.** Two unrelated models can both answer `42`, `blue`,
   `Canberra`; the behaviour layer only raises confidence *after* the hard layers agree, and
   disagreement there is decisive on its own.
5. **Quantization changes behaviour, not the tokenizer** — expect family-level matches with
   model-level distribution mismatch.

## Install as an agent skill

`SKILL.md` follows the [Agent Skills](https://agentskills.io/) open standard, so the same file
loads in Hermes, Claude Code, Codex CLI and anything else that reads `SKILL.md`:

```bash
./install.sh              # detect installed agents and install the skill for each
./install.sh --list       # show what would be installed, change nothing
./install.sh --project    # install into ./.claude/skills (commit it with your repo)
./install.sh --copy       # standalone copy instead of a symlink
./install.sh --force      # replace an existing directory (refused otherwise)
./install.sh --dry-run    # print the actions, write nothing
```

Default mode symlinks this checkout into the agent's skills directory, so `git pull` keeps the
skill current; `--copy` writes a self-contained directory (`SKILL.md` + the package + docs) for
hosts without the clone. Manual equivalent — put the skill directory here:

| Agent | Path |
|---|---|
| Hermes | `~/.hermes/skills/model-fingerprint/SKILL.md` |
| Claude Code (personal) | `~/.claude/skills/model-fingerprint/SKILL.md` |
| Claude Code (project) | `.claude/skills/model-fingerprint/SKILL.md` |
| Codex CLI | `~/.codex/skills/model-fingerprint/SKILL.md` |
| opencode | `~/.config/opencode/skill/model-fingerprint/SKILL.md` |

The skill teaches an agent the whole loop: set up, run the offline test suite, fingerprint an
endpoint, grade the evidence, and the traps that produce false results. It is drift-checked —
`tests/test_skill.py` fails the build if the skill documents a flag the CLI does not implement.

## Tests

```bash
python -m pip install pytest && python -m pytest -q
```

The suite runs entirely offline against `tests/mock_endpoint.py`, a local fake endpoint that
speaks all four protocols with deterministic outputs and realistic failure modes (routing-header
enforcement, output caps, context caps, reasoning models that eat the budget, missing usage).
Nothing in the suite touches the network or needs a key. It also covers the agent skill and the
installer: frontmatter rules, flag drift, and that `install.sh` is idempotent and never clobbers
an unrelated directory.

## Related work

- [llmfd](https://github.com/wofeiwo/llmfd) — infra + distribution + fusion engines
- [modelprint](https://github.com/unclecode/modelprint) — infrastructure probe design
- [llm-fingerprint-detector](https://github.com/ToseaAI/llm-fingerprint-detector) — single-token
  distribution fingerprinting, from *One Token Is Enough* ([arXiv:2607.10252](https://arxiv.org/abs/2607.10252))
- [LLM-Fingerprinter](https://github.com/litemars/LLM-Fingerprinter) — behavioural profiling

## License

MIT — see [LICENSE](LICENSE).
