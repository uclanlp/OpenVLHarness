"""Shared harness prompt wrapper for the generation benchmarks.

⛔ THIS CHANGES INFERENCE BEHAVIOUR AND BREAKS COMPARABILITY WITH BARE-PROMPT RUNS.
It is a deliberate harness-protocol decision, recorded here rather than left implicit
in five separate ``build_prompt`` methods.

Why it exists
-------------
KnowGen / WISE / T2I-FactualBench / FactIP / SearchGen are all text-to-image tasks whose
judges score a produced IMAGE; a sample with no generated image is scored 0. Given only
the bare benchmark prompt, an agent built for understanding tasks frequently answers in
text and scores 0 for a harness reason rather than a modelling one -- that is the
framework-is-understanding-centric gap behind the early WISE 23.1% / KnowGen 19.4%
numbers.

Why it is shared
----------------
KnowGen and WISE each grew their own copy of this wrapper while FactIP, SearchGen and
T2I-FactualBench kept bare prompts, so three of the five were measured under a different
protocol than the other two -- an inconsistency that is invisible in the result tables.
One function, one decision.

Comparability
-------------
Numbers produced with the wrapper are NOT comparable to numbers produced without it.
Set ``MVTK_GEN_BARE_PROMPT=1`` to send the bare benchmark prompt instead, which is what
reproducing pre-wrapper results requires.
"""

import os

__all__ = ["wrap_generation_prompt", "generation_prompt_mode"]

_INSTRUCTION = (
    "Text-to-image generation task. The description below is an implicit, "
    "knowledge-grounded prompt: infer what must appear using world knowledge "
    "(search for facts or visual references when helpful), then GENERATE an "
    "image that realizes it. You MUST produce an image using the image "
    "generation tool as your final result; do not answer with text only."
)


def generation_prompt_mode() -> str:
    """``"bare"`` when MVTK_GEN_BARE_PROMPT=1, else ``"wrapped"``."""
    return "bare" if os.environ.get("MVTK_GEN_BARE_PROMPT") == "1" else "wrapped"


def wrap_generation_prompt(prompt: str) -> str:
    """Frame ``prompt`` as an image-generation instruction.

    Returns the prompt unchanged under ``MVTK_GEN_BARE_PROMPT=1``.
    """
    prompt = str(prompt)
    if generation_prompt_mode() == "bare":
        return prompt
    return f"{_INSTRUCTION}\n\nPrompt to realize: {prompt}"
