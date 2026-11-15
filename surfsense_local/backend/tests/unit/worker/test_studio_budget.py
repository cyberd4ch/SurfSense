"""How a Studio draft's grounding is priced into the model's context window."""

from types import SimpleNamespace

import pytest

from modules.chat.budget import CHARS_PER_TOKEN
from modules.llm.catalog.remote.manifest.loader import load_remote_manifest
from modules.llm.resolution import ResolvedGeneration
from worker.studio.shared import budget

pytestmark = pytest.mark.unit


class _Window:
    """A generator that reports the window a real endpoint would, or fails."""

    def __init__(self, window: int | None, error: Exception | None = None) -> None:
        self.window = window
        self.error = error

    async def context_tokens(self, model: str) -> int | None:
        if self.error is not None:
            raise self.error
        return self.window


def _generation(
    window: int | None,
    error: Exception | None = None,
    catalog_provider: str | None = None,
) -> ResolvedGeneration:
    selection = SimpleNamespace(
        name="m",
        connection=None
        if catalog_provider is None
        else SimpleNamespace(catalog_provider=catalog_provider),
    )
    return ResolvedGeneration(selection, _Window(window, error))


def test_a_known_window_prices_the_grounding_into_what_is_left() -> None:
    """The fixed parts and the answer are held back first; the rest, estimated
    at the shared characters-per-token, is what gather may send."""
    left = (
        16_384
        - budget.PROMPT_RESERVE_TOKENS
        - budget.OUTPUT_RESERVE_TOKENS
        - budget.SAFETY_TOKENS
    )
    assert budget.grounding_budget_chars(_generation(16_384)) == left * CHARS_PER_TOKEN


def test_a_narrow_window_sends_less_than_the_old_flat_cap() -> None:
    """The reported case: an 8k window overflowed at the flat 24,000 chars.
    Priced, the same window leaves room for the prompts and the answer."""
    assert (
        budget.grounding_budget_chars(_generation(8_192))
        < budget.UNKNOWN_WINDOW_CHARS * 2
    )


def test_a_window_narrower_than_the_fixed_parts_leaves_no_grounding() -> None:
    """Zero, not negative: the request still fits, with nothing but the
    prompts in it, rather than corrupting a slice with a negative budget."""
    assert budget.grounding_budget_chars(_generation(1_024)) == 0


def test_an_unknown_window_falls_back_conservatively() -> None:
    """A remote endpoint that reports no window is not a narrow one; it is an
    absent fact. The fallback sends fewer characters than the flat cap did, so
    the request it cannot size still fits the smallest window it plausibly has."""
    assert budget.grounding_budget_chars(_generation(None)) == (
        budget.UNKNOWN_WINDOW_CHARS
    )
    assert budget.UNKNOWN_WINDOW_CHARS < 24_000


def test_an_endpoint_that_cannot_be_asked_is_an_unknown_window() -> None:
    """A runtime not up yet, or an endpoint in a bad minute, must not fail the
    draft before the model is even called: the ask failing is "unknown"."""
    assert (
        budget.grounding_budget_chars(_generation(None, error=RuntimeError("not up")))
        == budget.UNKNOWN_WINDOW_CHARS
    )


def _catalog_model() -> tuple[str, str, int]:
    for provider_id, provider in load_remote_manifest().providers.items():
        for model_id, model in provider.models.items():
            if model.context is not None:
                return provider_id, model_id, model.context
    pytest.skip("the manifest carries no context window to read")


def test_the_catalog_sizes_a_window_the_endpoint_will_not_report() -> None:
    """The OpenAI shape carries no window, but the manifest this app ships
    often does; a listed model is budgeted from its record."""
    provider, model_id, context = _catalog_model()
    model = _generation(None, catalog_provider=provider)
    model.selection.name = model_id

    assert budget.window_tokens(model) == context


def test_a_custom_endpoint_is_an_unknown_window() -> None:
    """A connection's own listing, outside the manifest, says nothing here."""
    model = _generation(None, catalog_provider="custom")

    assert budget.window_tokens(model) is None


def test_a_model_the_manifest_does_not_list_is_an_unknown_window() -> None:
    """A catalog provider is claimed but the manifest has no row: the window
    is unknown, and the fallback applies rather than a guessed one."""
    model = _generation(None, catalog_provider="openai")

    assert budget.window_tokens(model) is None


def test_an_overflow_message_is_read_wherever_it_hides() -> None:
    """The reporter's Ollama phrasing, llama.cpp's cache phrasing, and the
    same words inside an HTTP body or a wrapped cause all read as overflow."""
    ollama = RuntimeError(
        "request (8545 tokens) exceeds the available context size (8192 tokens)"
    )
    llamacpp = RuntimeError("Context size has been exceeded.")

    class _Body:
        text = '{"error": {"type": "exceed_context_size_error"}}'

    class _Wrapped:
        response = _Body()

    assert budget.is_context_overflow(ollama)
    assert budget.is_context_overflow(llamacpp)
    assert budget.is_context_overflow(_Wrapped())


def test_a_wrapped_overflow_is_still_found() -> None:
    """The message one hop down the cause chain reads the same as on the
    surface: a wrapped endpoint error is still an overflow."""
    cause = RuntimeError("exceeds the available context size")
    wrapped = RuntimeError("the model failed")
    wrapped.__cause__ = cause

    assert budget.is_context_overflow(wrapped)


def test_any_other_failure_is_not_an_overflow() -> None:
    """A failure without an overflow marker stays retryable."""
    assert not budget.is_context_overflow(RuntimeError("the model refused"))


def test_the_overflow_line_names_the_fix() -> None:
    """The tooltip must tell the user what to change, not that it failed."""
    reason = budget.overflow_reason()

    assert "context window" in reason
    assert "fewer or shorter sources" in reason
