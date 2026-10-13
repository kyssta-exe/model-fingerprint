"""Model fingerprinting - identify the real model behind an API, proxy or reseller.

Layered black-box attribution for OpenAI / OpenAI-Responses / Anthropic / Gemini
compatible endpoints: serving-stack DNA (headers, error wording, accepted
parameters), shift-invariant tokenizer counts, capability matrix, output/context
ceilings and short-answer behaviour distributions.

The tool reports evidence with a confidence grade, never proof.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
