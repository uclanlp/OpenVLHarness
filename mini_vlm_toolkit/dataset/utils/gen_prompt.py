"""Shared prompt framing for the generation benchmarks.

KnowGen / WISE / T2I-FactualBench / FactIP / SearchGen are text-to-image tasks: the
judge scores a produced IMAGE, and a sample without one scores 0. Telling the agent to
produce an image is therefore part of stating the task correctly, not an aid — a bare
benchmark caption never says "generate this", so a bare-prompt run partly measures
whether the agent guesses the task, which is not what any of these benchmarks are for.
Our agent comes from search/understanding work and answers in text readily, so it was
losing points for the framing rather than for the generation. That is what the early
WISE 23.1% / KnowGen 19.4% numbers were largely made of.

KnowGen and WISE were fixed on 2026-07-15; FactIP, SearchGen and T2I-FactualBench kept
bare prompts, so five benchmarks ran under two protocols and their scores sat in the
same table as if comparable. Those three were UNDER-measured by a harness defect; this
module finishes the 07-15 change and gives all five one implementation.

``MVTK_GEN_BARE_PROMPT=1`` restores the bare prompt. It exists to reproduce pre-fix
numbers or to match a paper's protocol exactly, not because the bare setting is the
neutral one.

Carry-forward, agreed on PR #186:
  1. Published paper baselines feed bare prompts. Any table that puts our numbers next
     to theirs must footnote the wrapper, or be produced with MVTK_GEN_BARE_PROMPT=1.
  2. These pilot numbers predate the fix on their benchmark and are UNDER-MEASURED --
     re-run under the wrapped protocol before they appear anywhere:
         FactIP 5.67 / 0.552        (2026-07-23, bare)
         SearchGen 44.71 / 44.93    (2026-07-23, bare)
         T2I-FactualBench 0.524     (2026-07-24, bare)
         WISE 23.1%, KnowGen 19.4%  (early July, before the 07-15 fix)
     KnowGen 0.32 and WISE 0.648 (2026-07-23/24) already ran wrapped and stand.
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
    """State the task: produce an image realizing ``prompt``.

    Returns the prompt unchanged under ``MVTK_GEN_BARE_PROMPT=1``.
    """
    prompt = str(prompt)
    if generation_prompt_mode() == "bare":
        return prompt
    return f"{_INSTRUCTION}\n\nPrompt to realize: {prompt}"
