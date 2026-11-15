"""How much grounding a Studio draft may send, priced in the model's window.

`gather` cuts a selection to a character budget; this sets that budget from the
model that will read it, so a draft is sized to its window instead of to a flat
cap that only fit the machine the cap was written on. On a local 8k window the
flat 24,000 characters came out as a request larger than the window itself, and
the draft failed whole (issue #2421). The accounting mirrors
modules/chat/budget.py: the fixed parts are priced first, the answer's share is
held back, and what is left — estimated at the same characters-per-token — is
the grounding's.

An unknown window is an absent fact, not a narrow one
(modules/llm/providers/protocols.py). The fallback stays conservative: fewer
characters than the old flat cap, so a model this app cannot size takes no more
of its window than a small local one would. When the request still overflows —
an endpoint smaller than it reported, or the chat template's own overhead — the
failure is deterministic, and the job's failure path reads it through
`is_context_overflow` to give the user the fix instead of a retry loop.
"""

import asyncio

from modules.chat.budget import CHARS_PER_TOKEN
from modules.llm.catalog.remote.manifest.loader import remote_lookup
from modules.llm.catalog.remote.rows import CUSTOM
from modules.llm.resolution import ResolvedGeneration

__all__ = [
    "OUTPUT_RESERVE_TOKENS",
    "PROMPT_RESERVE_TOKENS",
    "SAFETY_TOKENS",
    "UNKNOWN_WINDOW_CHARS",
    "grounding_budget_chars",
    "is_context_overflow",
    "overflow_reason",
    "window_tokens",
]

# What the fixed prompt parts may spend together, at the largest tier this app
# loads: the system prompt (a frontier prompt is ~1,600 characters), the header
# wrapped around each source, and the instructions some pipelines append after
# the sources.
PROMPT_RESERVE_TOKENS = 1_024

# The reply. A draft's answer is the artifact itself — a document, a brief —
# and run_model sends no max_tokens, so this is the room it may use.
OUTPUT_RESERVE_TOKENS = 2_048

# Safety margin: the chat template's roles and separators, tokenizer drift on
# dense scripts, and the estimate itself.
SAFETY_TOKENS = 512

_FIXED_TOKENS = PROMPT_RESERVE_TOKENS + OUTPUT_RESERVE_TOKENS + SAFETY_TOKENS

# For a window this module cannot see. Half the old flat cap: grounding enough
# to be useful, small enough that the whole request fits the default window an
# OpenAI-shaped endpoint typically opens with.
UNKNOWN_WINDOW_CHARS = 12_000

# What an overflow says, across the endpoints this app talks to. Matched
# loosely on purpose: a shape not listed here falls through to the generic
# failure path, while one that is listed is certain.
_OVERFLOW_MARKERS = (
    # OpenAI-compatible endpoints, e.g. Ollama: the request outgrew the window.
    "exceeds the available context size",
    "exceed_context_size_error",
    "context length exceeded",
    "maximum context length",
    "too many tokens",
    "reduce the length",
    # llama.cpp: one request past the unified cache ends every request on it.
    "context size has been exceeded",
    "failed to find free space in the kv cache",
    "failed to find a memory slot",
)


def window_tokens(model: ResolvedGeneration) -> int | None:
    """The model's context window, from the endpoint or the catalog, or None.

    None is "unknown", never "narrow": a budget read from it falls back rather
    than squeezing a wide endpoint.
    """
    reported = _endpoint_window(model)
    if reported is not None:
        return reported
    return _catalog_window(model)


def _endpoint_window(model: ResolvedGeneration) -> int | None:
    """What the generator itself reports. An endpoint that will not say, or
    that cannot be reached to ask, leaves the window unknown."""
    try:
        return asyncio.run(model.generator.context_tokens(model.selection.name))
    except Exception:
        return None


def _catalog_window(model: ResolvedGeneration) -> int | None:
    """What the bundled remote manifest records, for a listed catalog model."""
    connection = model.selection.connection
    if connection is None or connection.catalog_provider == CUSTOM:
        return None
    found = remote_lookup().classify(
        model.selection.name, provider=connection.catalog_provider
    )
    return found.supports.context_window if found.supports else None


def grounding_budget_chars(model: ResolvedGeneration) -> int:
    """The grounding share, in characters, for `gather` to cut a selection to.

    Floors at zero: a window narrower than the fixed parts plus the answer
    reserve leaves no grounding at all, which is correct — not a request that
    cannot fit.
    """
    window = window_tokens(model)
    if window is None:
        return UNKNOWN_WINDOW_CHARS
    return max(0, window - _FIXED_TOKENS) * CHARS_PER_TOKEN


def is_context_overflow(failure: Exception) -> bool:
    """Whether a job failure is the model's window being exceeded.

    The message is matched wherever it may hide: an HTTP error carries it in
    the response body, a streamed error in its own text, and either may wrap
    the one that named it.
    """
    lowered = _spilled(failure).lower()
    return any(marker in lowered for marker in _OVERFLOW_MARKERS)


def overflow_reason() -> str:
    """The plain-language line the failed artifact's tooltip reads."""
    return (
        "The selected sources were still too long for this model's context "
        "window after trimming. Use fewer or shorter sources, or pick a model "
        "with a larger context window."
    )


def _spilled(failure: Exception) -> str:
    parts = [str(failure)]
    response = getattr(failure, "response", None)
    if response is not None:
        parts.append(getattr(response, "text", "") or "")
    # getattr, not an attribute read: a wrapped error (a proxy or a mock) can
    # lack __cause__ entirely, and this runs on the job's failure path, where
    # raising would break the handler itself.
    cause = getattr(failure, "__cause__", None)
    if cause is not None:
        parts.append(_spilled(cause))
    return "\n".join(parts)
