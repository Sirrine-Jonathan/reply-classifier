"""Provider-agnostic LLM client.

The service never talks to a vendor SDK directly. Anything that wants to
classify with an LLM goes through the `LLMClient` protocol below, so swapping
providers (Anthropic, OpenAI, AWS Bedrock, a local model, ...) is a one-class
change in this file.

Weekend TODO (Jonathan): implement a real client — e.g. `AnthropicClient` —
that satisfies the protocol, reads its key/model from the environment
(see .env.example), and returns the model's raw text completion. The
classifier in app/classifier.py already knows how to parse it.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal text-completion interface the classifier depends on."""

    def complete(self, prompt: str) -> str:
        """Return the model's text response to `prompt`."""
        ...


class StubLLMClient:
    """Placeholder client used until a real provider is wired in.

    It deliberately raises instead of guessing: running the service with
    CLASSIFIER=llm before a provider exists should fail loudly, not silently
    return junk classifications.
    """

    def complete(self, prompt: str) -> str:
        raise NotImplementedError(
            "No LLM provider configured. Implement a real LLMClient in "
            "app/llm_client.py (see the module docstring) or run with "
            "CLASSIFIER=heuristic."
        )


def get_llm_client() -> LLMClient:
    """Build the configured LLM client.

    Weekend TODO: branch on os.environ["LLM_PROVIDER"] here and construct the
    real client. For now everything falls through to the stub.
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider in ("", "stub"):
        return StubLLMClient()
    raise NotImplementedError(
        f"LLM_PROVIDER={provider!r} is not implemented yet. Add it in "
        "app/llm_client.py::get_llm_client()."
    )
