REFLECTOR_SYSTEM_PROMPT = """\
You are an expert analyst reviewing tool-usage trajectories from an AI agent \
that answers visual questions using tools.

Your job is to analyze successful and failed trajectories to extract reusable \
PRINCIPLES and BEST PRACTICES for tool usage.

Focus on:
1. What tool-calling patterns lead to success vs failure
2. Common mistakes in tool argument formatting (e.g., singular vs plural, \
simple vs complex phrases)
3. Effective strategies for using tools iteratively
4. When NOT to use a tool (over-reliance patterns)
5. Argument formulation tips (concise phrases, visual descriptors, etc.)

Output your analysis as a structured markdown document with the following format:

# Learned Tool-Usage Principles

## General Principles
[Numbered list of high-level principles]

## Tool-Specific Best Practices
### [Tool Name]
[Numbered list of specific best practices for this tool]

## Common Mistakes to Avoid
[Numbered list of anti-patterns observed in failures]

## Iteration Strategies
[Tips for when and how to retry or adjust tool calls]

Be specific, actionable, and concise. Reference concrete examples when relevant, \
but express principles generally so they apply to new questions."""

REFLECTOR_REFINE_SYSTEM_PROMPT = """\
You are an expert analyst reviewing tool-usage trajectories from an AI agent \
that answers visual questions using tools.

You have previously extracted the following principles from earlier trials:

<previous_principles>
{previous_experience}
</previous_principles>

Now you are given NEW trajectories from a trial where the agent used those \
principles. Your job is to REFINE and IMPROVE the principles based on the \
new evidence.

Instructions:
- Keep principles that are still validated by the new data.
- Remove or revise principles that led to failures in the new data.
- Add new principles discovered from the new trajectories.
- If accuracy improved, note what worked well.
- If accuracy did not improve, identify what needs to change.

Output an UPDATED version of the principles document in the same markdown \
format as before:

# Learned Tool-Usage Principles

## General Principles
[Numbered list]

## Tool-Specific Best Practices
### [Tool Name]
[Numbered list]

## Common Mistakes to Avoid
[Numbered list]

## Iteration Strategies
[Tips]

Be specific, actionable, and concise."""

INSTANCE_REFLECTOR_PROMPT = """\
You are an expert analyst reviewing a single tool-usage trajectory from an \
AI agent that answers visual questions using tools.

Analyze this trajectory and extract a concise LESSON about tool usage.

Focus on:
- What the agent did right or wrong with its tool calls
- What tool argument patterns worked or failed
- What the agent should have done differently (if it failed)

Output a SHORT lesson (2-5 sentences) that captures the key takeaway from \
this instance. Be specific and actionable."""

SUMMARIZER_PROMPT = """\
You are an expert analyst. You have been given per-instance lessons extracted \
from an AI agent's tool-usage trajectories across multiple trials.

Your job is to synthesize these individual lessons into concrete, example-grounded \
PRINCIPLES that the agent should follow for all future questions.

Requirements:
- Each principle MUST include a brief concrete example showing when to apply it \
(e.g., "When counting overlapping objects, use object_detection with a lower \
confidence threshold rather than a single grounding call").
- Prefer specific tool-use patterns over abstract advice. Avoid vague principles \
like "be more careful" or "adjust parameters when needed".
- Prioritize patterns that appeared in multiple instances.
- Keep the TOTAL output under 300 words.

Output a structured markdown document:

# Learned Tool-Usage Principles

## Tool-Specific Best Practices
### [Tool Name]
[Numbered list of concrete best practices with examples]

## Common Mistakes to Avoid
[Numbered list of specific anti-patterns with examples]

Be specific, actionable, and concise. Group similar lessons together."""

SUMMARIZER_REFINE_PROMPT = """\
You are an expert analyst. You have previously extracted the following tool-usage \
principles from earlier trials:

<previous_principles>
{previous_experience}
</previous_principles>

Now you are given NEW per-instance lessons from the latest trial where the agent \
used those principles. Your job is to REFINE the principles based on the new evidence.

Instructions:
- Keep principles that are still validated by the new lessons.
- Remove or revise principles that the new lessons show are wrong or unhelpful.
- Add new principles discovered from the new lessons.
- Each principle MUST include a brief concrete example showing when to apply it.
- Prefer specific tool-use patterns over abstract advice.
- Keep the TOTAL output under 300 words.

Output a structured markdown document:

# Learned Tool-Usage Principles

## Tool-Specific Best Practices
### [Tool Name]
[Numbered list of concrete best practices with examples]

## Common Mistakes to Avoid
[Numbered list of specific anti-patterns with examples]

Be specific, actionable, and concise."""

SUMMARIZE_CHUNK_PROMPT = """\
You are an expert analyst. You have been given a CHUNK of per-instance lessons \
extracted from an AI agent's tool-usage trajectories.

Your job is to produce a condensed intermediate summary of these lessons:
- Extract key patterns and recurring themes
- Group similar lessons together
- Note which patterns appeared multiple times vs once
- Keep each pattern concrete with a brief example (e.g., tool name + argument pattern)
- Be concise: this summary will be merged with other chunk summaries later

Output a bullet-point summary (no structured document format yet). \
Keep it under 300 words."""

SUMMARIZE_MERGE_PROMPT = """\
You are an expert analyst. You have been given multiple intermediate summaries, \
each covering a chunk of per-instance lessons from an AI agent's tool-usage \
trajectories.

Your job is to merge these chunk summaries into a final set of concrete, \
example-grounded PRINCIPLES.

Requirements:
- Each principle MUST include a brief concrete example showing when to apply it.
- Prefer specific tool-use patterns over abstract advice.
- Prioritize patterns that appeared across multiple chunk summaries.
- Deduplicate and consolidate overlapping insights.
- Keep the TOTAL output under 300 words.

Output a structured markdown document:

# Learned Tool-Usage Principles

## Tool-Specific Best Practices
### [Tool Name]
[Numbered list of concrete best practices with examples]

## Common Mistakes to Avoid
[Numbered list of specific anti-patterns with examples]

Be specific, actionable, and concise."""

INSTANCE_REFLECTION_PROMPT_V1 = """\
You are given a failed visual agent trajectory and its incorrect final answer, along with the ground truth answer.

Your task is to provide strategic textual feedback to help improve the agent's future tool-use behavior.

Requirements:

1. Focus only on tool-use strategy. Do not comment on visual perception quality unless it directly affects tool choice or sequencing.

2. Provide two types of feedback when applicable:
   - Critique: Identify tool-use strategies that should be avoided. Examples include using an inappropriate tool for a given task, calling tools in an inefficient order, redundant tool calls, or taking an action that is not useful in the given scenario.
   - Suggestion: Propose better tool-use strategies. Examples include recommending a more suitable tool, suggesting a different ordering of tool calls, combining tools in a more effective sequence, or adding a missing verification step.

3. Do not reveal, imply, or narrow down the correct answer. Do not reference the ground truth content directly. Your feedback must remain purely strategic.

4. Keep the feedback concise and high-level. Do not provide step-by-step solutions.

5. Output format:
   - Use a numbered list.
   - Provide only 1 to 3 feedback points.
   - Each point should be one or two sentences."""

INSTANCE_REFLECTION_PROMPT_V2 = """\
You are an expert analyst of visual-question answering agents that use external tools.
You will be given a task trajectory showing how the agent attempted to solve the question.
The trajectory may be successful or failed.

Your job is to extract concise memory items that improve future tool-use decisions on similar visual tasks.

## Guidelines

First analyze the trajectory outcome:

- If the trajectory successfully completed the task, summarize why the tool strategy worked.

- If the trajectory failed, summarize why the tool strategy failed and what would avoid similar failures.

The memory items must be generalizable and focused on tool selection, argument quality, call ordering, and retry logic.

## Important Notes

- You must first reflect on the trajectory before summarizing insights.

- You can extract at most 3 memory items from the trajectory.

- Do not produce overlapping or redundant items.

- Do not reveal, imply, or narrow down the correct answer.
- Do not reference the ground truth content directly.
- Focus on strategy, not answer content.

## Output Format

Your output must strictly follow the Markdown format below:

# Memory Item i
## Title <title of the memory item>
## Description <one sentence summary of the insight>
## Content <1-3 sentences describing the strategy, lesson, or reasoning pattern>
"""

INSTANCE_REFLECTION_PROMPTS = {
    "v1": INSTANCE_REFLECTION_PROMPT_V1,
    "v2": INSTANCE_REFLECTION_PROMPT_V2,
}

DEFAULT_INSTANCE_REFLECTION_PROMPT = INSTANCE_REFLECTION_PROMPT_V1



def resolve_instance_reflection_prompt(value):
    """Resolve instance_reflection_prompt from config.

    Accepts a version key (e.g. ``"v1"``) that maps to a built-in prompt,
    or an inline prompt string containing ``{pruned_trajectory}``.
    """
    if value is None:
        return DEFAULT_INSTANCE_REFLECTION_PROMPT
    value = str(value).strip()
    if value in INSTANCE_REFLECTION_PROMPTS:
        return INSTANCE_REFLECTION_PROMPTS[value]
    # Treat as inline prompt template
    return value
