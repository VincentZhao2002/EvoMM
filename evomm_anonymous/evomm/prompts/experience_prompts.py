"""Prompt templates for the self-evolution experience loop."""

EXPERIENCE_EXTRACT_PROMPT = """
You are an experience analyst. Given the research trace for a question-answering task, distill actionable lessons that can help answer similar questions in the future.

QUESTION:
{question}

CATEGORY: {category}

RESEARCH TRACE SUMMARY:
- Search iterations performed: {search_rounds}
- Tools used: {tools_used}
- Queries issued: {queries_issued}
- Integrated result (abbreviated):
{integrated_result}

INSTRUCTIONS:
1. Identify which queries were most effective at retrieving useful evidence.
2. Summarise the single most important finding in one sentence ("key_insight").
3. Briefly assess whether the integrated result is likely complete and accurate ("self_assessment").
4. Extract 2-4 short, actionable lessons that would help a future agent answer a similar question faster or more accurately. Focus on:
   - Which retrieval strategy worked best (keyword vs vector vs page_index).
   - What type of information was hardest to find.
   - Any pitfalls to avoid (e.g. wrong entity disambiguation, missing time context).

THINKING STEP
- Before producing the output, reason about effectiveness inside <think>...</think>.
- After </think>, output ONLY the JSON object.

OUTPUT JSON SPEC:
Return ONE JSON object with EXACTLY these keys:
- "effective_queries": array of strings (the most useful queries from the trace)
- "key_insight": string
- "self_assessment": string
- "lessons": array of strings (2-4 items)

All keys MUST appear.
After the <think> section, return ONLY the JSON object.
"""


def format_experience_context(experiences) -> str:
    """Format a list of Experience objects into a prompt-injectable text block.

    This is injected into system_prompts so the planning and integration
    stages can benefit from past experience without changing their core
    prompt templates.
    """
    if not experiences:
        return ""

    lines = ["PAST EXPERIENCE (from similar questions you answered before):"]
    for i, exp in enumerate(experiences, 1):
        lessons_str = "; ".join(exp.lessons) if exp.lessons else "N/A"
        queries_str = ", ".join(exp.effective_queries[:3]) if exp.effective_queries else "N/A"
        lines.append(
            f"{i}. Question: \"{exp.question}\" (category={exp.category})\n"
            f"   Key insight: {exp.key_insight}\n"
            f"   Effective queries: {queries_str}\n"
            f"   Lessons: {lessons_str}"
        )
    lines.append(
        "\nUse these experiences to guide your strategy. "
        "Reuse effective query patterns when applicable. "
        "Avoid pitfalls mentioned in the lessons."
    )
    return "\n".join(lines)
