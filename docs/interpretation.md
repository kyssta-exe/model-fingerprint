# Interpreting Fingerprint Evidence

Read this before grading. Layers are near-orthogonal: **headers/errors/tokenizer are
serving- and weight-layer evidence (hard)**, distributions are statistical, self-report is soft.

## 1. Serving-stack signatures (from headers, endpoints and error wording)

| Observation | Likely stack | Confidence |
|---|---|---|
| `GET /health` 200, `GET /v1/model/info` or `/model_group/info` present, error text embeds `litellm.BadRequestError` / upstream exception class | LiteLLM proxy | high |
| `x-groq-*` headers, `system_fingerprint: fp_...` | Groq | high |
| `x-ms-*`, `apim-request-id`, error JSON with `innererror`, `api-version` query param required | Azure OpenAI | high |
| `x-amzn-*` headers, bare `{"message": ...}` errors | AWS Bedrock | high |
| `x-goog-*` headers; usage in `usageMetadata.promptTokenCount`; `candidates[].finishReason` | Google Vertex / Gemini | high |
| response `id` starts `gen-`; `x-ratelimit-*`; replay works | OpenRouter (see §2 — the fastest possible win) | high |
| `engine_params` probe (`repetition_penalty`, `min_tokens`, `ignore_eos`) **accepted**; `logprobs`+`top_logprobs` support; `seed` accepted | vLLM / SGLang self-hosted | medium-high |
| `/props` or `/health` from llama.cpp, params like `n_predict`, `grammar`, `cache_prompt` accepted | llama.cpp server | medium-high |
| `/api/tags` + `/api/chat` respond; model ids carry `:tag` suffixes (`llama3:8b`) | Ollama | high |
| Error JSON `{"error": {"message", "type", "param", "code"}}` and *only* OpenAI-shaped params accepted | OpenAI direct or a faithful clone — the clone case is why tokenizer/dist layers exist | medium |
| Gateway error text carries vendor codes, e.g. Zhipu `[1210] illegal temperature` | upstream vendor (Zhipu/GLM family) behind a gateway | high, when seen |

Absence of any of these proves nothing: a deliberate gateway strips headers and rewrites errors.
Record `headers-hidden` / `error-shape-generic` explicitly instead of guessing.

## 2. Aggregator shortcut (do this first when the host looks like a router)

OpenRouter returns a generation id on every completion. Joining it to
`GET https://openrouter.ai/api/v1/generation?id=<id>` returns the real upstream **model and
provider name**. One request, near-certain attribution — always try it before a full scan.

## 3. Tokenizer layer

- Use **shift-invariant** counts: `count(text) - count("a")`. Hidden system prompts and chat
  templates add a constant per request, so raw counts differ between two hosts wrapping the
  *same* tokenizer while normalised vectors match exactly.
- Exact-match on the normalised vector ⇒ **same tokenization stack** (same base model family and
  chat template). This is the strongest cheap signal.
- It is *not* lineage: a 24-pair holdout study (arXiv:2608.29930) found only 0.50 sensitivity for
  family attribution, with same-family pairs (Qwen 3.8, DeepSeek V4 variants) falling below the
  frozen threshold. Say "same tokenizer stack", and let the other layers carry the rest.
- Vectors are only comparable if every probe was sent with the same role/template. A cell that
  timed out or reported no `usage` is **uninformative**, not a mismatch.

## 4. Statistical layer (single-token distributions)

- Cells are short-answer prompts at temperature 1; you compare answer distributions, not text.
- Jensen–Shannon divergence, calibrated in the literature: **match ≤ 0.25 < uncertain ≤ 0.35 <
  mismatch**; same-model-across-vendors median ≈ 0.227, different-model median ≈ 0.463.
- Always run a **split-half check** on the target's own samples first. If split-half JSD is
  already high, the endpoint is too noisy (multi-host routing, quantization, or rate limiting)
  and no cross-comparison is valid.
- Different models can share a preference (both loving `42` / `blue`) — the llmfd write-up
  measured meanJSD 0.09 between two genuinely different models on the random-number cell, where
  the tokenizer layer correctly disagreed. Never let this layer alone overturn hard evidence.
- More cells and more samples both help: 8 cells ≈ 10.6% EER, 40 cells ≈ 7.3%.

## 5. Soft layer (use only as corroboration)

- Self-identification, style, refusal phrasing, emoji defaults, markdown habits.
- A system prompt, persona wrapper or router rewrites all of it. Report as soft evidence.
- Knowledge-cutoff probing: ask about dated events in ascending order and bisect the ceiling.
  **Verify each fact with `web_search` before using it** — a wrong fact silently moves the
  cutoff — and disable/note any tool-use or search capability first, or the model will simply
  look the answer up.

## 6. Confidence grades

| Grade | Requires |
|---|---|
| `CONFIRMED_SAME` | normalised tokenizer vector exact-match + error/header layer consistent + distribution match, no contradicting layer |
| `LIKELY_SAME` | hard layer matches, statistical layer unavailable |
| `UNCERTAIN` | layers disagree, or a key layer uninformative (429/no usage) |
| `LIKELY_DIFFERENT` / `CONFIRMED_DIFFERENT` | hard evidence points at another family/stack |
| `UNKNOWN` | open set: no reference match. Keep the vector — it is now your reference |

Quantization, fine-tunes and adapters keep the tokenizer intact but shift behaviour: expect
family-level hard matches with model-level distribution mismatch, and report exactly that.

## 7. Worked example — why the behaviour layer alone is not enough

Verified run against a gateway (OpenCode Go, 2026-10-13), target `muse-spark-1.3-contributor`
(Responses protocol) vs reference `deepseek-v4-flash` (chat-completions, same host, same key):

| Layer | Result |
|---|---|
| Behaviour cells | **Nearly identical** — both answered `42`, `blue`, `蓝色`, `heads`, `dog`, `canberra`, `no`, `19`. Only `letter` (m vs a), colour ordering, emoji and the random cell differed. |
| Tokenizer vector | baseline `a` = 8 vs 31; 6/10 normalised probes matched, 4 mismatched (`accented`, `cjk_sentence`, `english_rare`, `pathological`) |
| Capabilities | target accepted `text.format` json_schema, flat `tools`, `parallel_tool_calls`, `store`, `include`; reference accepted `logprobs`, `seed`, `n`, `stop`, `developer_role`, `engine_params` — near-disjoint |
| Serving layer | same host: identical `bad_model` error (`Upstream request failed: Model is unavailable`), Cloudflare headers |

Verdict: **different model stacks behind one gateway**. Had the run stopped at the behaviour
layer, it would have reported a match — the defaults two unrelated models share (42, blue,
Canberra) are exactly the false positive the tokenizer and capability layers exist to catch.
Practical rule: agreement in the behaviour layer raises confidence only *after* the hard layers
agree; disagreement there is decisive on its own.

Also note the `baseline_a` gap (8 vs 31): the hidden instructions/template on the Responses
endpoint are ~23 tokens heavier. That constant offset is why normalised counts are used — the
raw counts (8 vs 31) would have made every probe look like a mismatch.

## 8. Reporting rules

- Split the writeup: **serving layer** (who deployed it) vs **model layer** (whose weights).
  Same weights move between hosts and only the serving layer changes.
- State the count of unusable probes and any rate limiting encountered.
- Never state a vendor as fact from soft evidence. "Consistent with the GLM family" ≠ "is GLM".
- Recommend a re-test on a different date/hour before anyone acts on the finding.
