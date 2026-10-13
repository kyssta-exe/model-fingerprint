"""Response parsing for every supported wire protocol.

One place that knows where each protocol hides its text, token usage, reasoning
tokens and finish status, so the layers never guess.
"""

import re

_ERROR_SIGNATURE_PATTERNS = (
    r"\[\d{3,5}\]",
    r"\b[A-Za-z]*(?:Error|Exception)\b",
    r'Upstream[^"]{0,40}',
    r'(?:context|maximum|max)[^.,;"]{0,60}',
)

_HEADER_NOISE = frozenset({"date", "content-length", "connection", "keep-alive", "vary"})


def norm_answer(text):
    """Collapse a model answer to a comparable single token."""
    t = (text or "").strip().split("\n")[0].strip()
    t = t.strip("'\"`* .!?\u3002\uff01")
    low = t.lower()
    if not low:
        return "<empty>"
    if low.startswith("yes"):
        return "yes"
    if low.startswith("no"):
        return "no"
    return low[:24]


def usage_of(resp):
    """Prompt-token count wherever this protocol puts it, or None if unreported."""
    j = resp.get("json") or {}
    u = j.get("usage") or {}
    for key in ("prompt_tokens", "input_tokens", "promptTokenCount"):
        if isinstance(u.get(key), int):
            return u[key]
    um = j.get("usageMetadata") or {}
    if isinstance(um.get("promptTokenCount"), int):
        return um["promptTokenCount"]
    return None


def reasoning_tokens(resp):
    """Reasoning/thinking tokens spent, when the stack reports them."""
    u = (resp.get("json") or {}).get("usage") or {}
    for key in ("completion_tokens_details", "output_tokens_details"):
        details = u.get(key) or {}
        if isinstance(details.get("reasoning_tokens"), int):
            return details["reasoning_tokens"]
    um = (resp.get("json") or {}).get("usageMetadata") or {}
    if isinstance(um.get("thoughtsTokenCount"), int):
        return um["thoughtsTokenCount"]
    return None


def response_status(resp):
    """(finish status, incomplete reason) - a budget-capped answer is not an answer."""
    if resp.get("status") not in (200, 201):
        return None, None
    j = resp.get("json") or {}
    incomplete = j.get("incomplete_details") or {}
    if isinstance(incomplete, dict):
        incomplete = incomplete.get("reason")
    status = j.get("status")
    if not status and isinstance(j.get("choices"), list) and j["choices"]:
        status = j["choices"][0].get("finish_reason")
    if not status:
        status = j.get("stop_reason")  # anthropic
    if not status and isinstance(j.get("candidates"), list) and j["candidates"]:
        status = j["candidates"][0].get("finishReason")  # gemini
    return status, incomplete


def text_of(resp):
    """The visible completion text, whatever shape the protocol uses."""
    j = resp.get("json") or {}
    try:
        if isinstance(j.get("choices"), list) and j["choices"]:
            choice = j["choices"][0]
            message = choice.get("message") or {}
            if isinstance(message.get("content"), str):
                return message["content"]
            return choice.get("text") or ""
        if isinstance(j.get("content"), list):  # anthropic
            return "".join(p.get("text", "") for p in j["content"] if isinstance(p, dict))
        if isinstance(j.get("candidates"), list) and j["candidates"]:  # gemini
            parts = j["candidates"][0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if isinstance(j.get("output_text"), str):  # responses (convenience)
            return j["output_text"]
        if isinstance(j.get("output"), list):  # responses (native)
            for item in j["output"]:
                for part in item.get("content") or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        return part["text"]
    except (KeyError, IndexError, AttributeError, TypeError):
        return ""
    return ""


def resp_keys(resp):
    j = resp.get("json")
    return sorted(j.keys()) if isinstance(j, dict) else []


def err_text(resp, limit=500):
    if resp.get("error"):
        return resp["error"]
    return (resp.get("text") or "").strip()[:limit]


def err_signature(resp):
    """How this stack words an error - the fingerprint inside the failure."""
    sig = []
    j = resp.get("json")
    if isinstance(j, dict):
        err = j.get("error")
        if isinstance(err, dict):
            for key in ("type", "code", "param"):
                if err.get(key) not in (None, ""):
                    sig.append("%s=%s" % (key, err[key]))
        elif isinstance(err, str):
            sig.append("error=str")
        if not isinstance(err, dict):
            for key in ("detail", "message", "type"):
                if key in j:
                    sig.append("top:%s" % key)
    text = (resp.get("text") or "")[:800]
    for pattern in _ERROR_SIGNATURE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            sig.append(m.group(0)[:70])
    return sig


def hdr_dna(headers):
    """Response headers minus the noisy ones - the serving stack's DNA."""
    out = {}
    for key, value in headers.items():
        low = key.lower()
        if low in _HEADER_NOISE:
            continue
        out[low] = value[:120]
    return out
