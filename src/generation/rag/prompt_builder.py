"""Prompt construction for grounded estimate generation (Session 9 + 11).

The system prompt encodes the grounding policy: every task must cite the
specific chunk_id(s) it derives from, include verbatim evidence from the source,
and set grounded=False when no source supports the task. Fabricated ids are
forbidden, and when the context cannot support an estimate the model must say
so via ``confidence="insufficient"`` rather than guess.
"""

from __future__ import annotations

from src.generation.rag.schemas import EstimationQuery


def build_system_prompt() -> str:
    """Return the grounding system prompt for the generator."""
    return (
        "You are a senior software-delivery estimator. Produce a detailed cost "
        "estimate in engineer-days for the project described by the user, grounded "
        "in historical budgets supplied as <source> blocks.\n"
        "\n"
        "Structure the estimate as functional MODULES, each decomposed into the "
        "concrete engineering TASKS needed to deliver it:\n"
        "- Organise the work into functional blocks (e.g. Authentication & Access, "
        "Payments & Billing, Core Domain, Data & Integrations, Frontend/UX, "
        "Infrastructure & DevOps, Security & Compliance, QA & Testing, Project "
        "Management). Use the modules that fit THIS project; omit the ones that "
        "do not apply.\n"
        "- Within each module, break the work into granular tasks. Give each task a "
        "short `description` and its own `engineer_days`. Aim for a thorough "
        "breakdown — typically 4-8 modules with several tasks each — rather than a "
        "few coarse line items.\n"
        "- A historical <source> component usually maps to a module or a small group "
        "of tasks; decompose it into the finer tasks a delivery team would actually "
        "plan, distributing its engineer-days across them.\n"
        "\n"
        "CITATION RULES (critical):\n"
        "1. Base every estimate ONLY on the <source> blocks provided. Do not rely on "
        "outside knowledge for the numbers.\n"
        "2. For each task, fill the `sources` array with one or more SourceCitation "
        "objects. Each SourceCitation must include:\n"
        "   - `chunk_id`: the `id` attribute of the <source> element you are citing\n"
        "   - `document_id`: the `budget_id` attribute of the same <source> element\n"
        "   - `evidence`: a VERBATIM span or figure from that source that directly "
        "supports the task (copy the exact text, do not paraphrase)\n"
        "   - `relevance`: 'primary' if the source is the main basis, 'supporting' "
        "if it provides additional context, 'tangential' if loosely related\n"
        "   - `used_for`: a brief description of what this source contributed\n"
        "3. Set `grounded=true` ONLY when at least one source directly supports the "
        "task with verbatim evidence. Set `grounded=false` when no source supports "
        "the task (leave `sources` empty in that case).\n"
        "4. NEVER invent chunk_ids. If you cannot find a source that directly supports "
        "a task, set `grounded=false` and move that scope to the `assumptions` array "
        "instead of fabricating a citation.\n"
        "5. total_engineer_days must equal the sum of all tasks across all modules.\n"
        "6. If the provided context is insufficient to estimate responsibly, set "
        'confidence="insufficient", leave total_engineer_days and duration_weeks '
        "null, leave modules empty, and explain what is missing in "
        "insufficient_context_explanation.\n"
        "7. Otherwise set confidence to high/medium/low based on how well the sources "
        "match the project, and explain your derivation in `reasoning`."
    )


def _brief(structured_query: EstimationQuery) -> str:
    """Render the structured project brief shared by both user-message builders."""
    lines = [
        f"Function: {structured_query.function}",
        f"Technologies: {', '.join(structured_query.technologies) or 'n/a'}",
        f"Sector: {structured_query.sector or 'n/a'}",
        f"Scale: {structured_query.scale}",
        f"Country: {structured_query.country or 'n/a'}",
        f"Regulations: {', '.join(structured_query.regulations) or 'n/a'}",
        f"Constraints: {', '.join(structured_query.constraints) or 'n/a'}",
    ]
    return "\n".join(lines)


def build_user_message(context_block: str, structured_query: EstimationQuery) -> str:
    """Assemble the user turn: the structured brief plus the retrieved sources."""
    return (
        "<project_brief>\n"
        f"{_brief(structured_query)}\n"
        "</project_brief>\n"
        "\n"
        "<sources>\n"
        f"{context_block}\n"
        "</sources>\n"
        "\n"
        "Produce the grounded estimate now, citing source ids for every "
        "quantitative claim."
    )


# ---------------------------------------------------------------------------
# Session 10 — structure-only generation WITHOUT retrieval.
#
# The wizard generates the module→task structure as a FREE decomposition of the
# brief (no <sources>, no citations): grounding the *structure* in a handful of
# retrieved budgets impoverished the tree. Retrieval re-enters later, per task,
# only to derive the hours (see app/generation/rag/task_hours.py).
# ---------------------------------------------------------------------------


def build_structure_system_prompt() -> str:
    """System prompt for the ungrounded structure-only decomposition."""
    return (
        "You are a senior software-delivery architect. Decompose the project "
        "described by the user into the functional MODULES and the concrete "
        "engineering TASKS needed to deliver it. This is a STRUCTURE-ONLY step: "
        "you do NOT estimate hours and you do NOT have historical sources — rely "
        "on your own engineering judgement about what the project entails.\n"
        "\n"
        "- Organise the work into functional blocks (e.g. Authentication & Access, "
        "Payments & Billing, Core Domain, Data & Integrations, Frontend/UX, "
        "Infrastructure & DevOps, Security & Compliance, QA & Testing, Project "
        "Management). Use the modules that fit THIS project; add sector-specific "
        "ones when the brief calls for them; omit the ones that do not apply.\n"
        "- Within each module, break the work into granular tasks with a short "
        "`description`. Be thorough — typically 5-9 modules with several tasks "
        "each — so a delivery team could plan from it.\n"
        "\n"
        "Rules:\n"
        "1. Leave `engineer_days` null for every task and `total_engineer_days` "
        "null — hours are derived in a later step.\n"
        "2. Leave `sources` empty: there is no historical context here. Do not "
        "invent citations.\n"
        "3. Use `assumptions` for scope you are inferring beyond the brief.\n"
        "4. If the brief is too vague to scope responsibly, set "
        'confidence="insufficient", leave modules empty and explain what is '
        "missing in insufficient_context_explanation; otherwise set confidence to "
        "high/medium/low based on how well-specified the brief is and explain your "
        "reasoning in `reasoning`."
    )


def build_structure_user_message(structured_query: EstimationQuery) -> str:
    """User turn for structure-only generation: just the brief, no sources."""
    return (
        "<project_brief>\n"
        f"{_brief(structured_query)}\n"
        "</project_brief>\n"
        "\n"
        "Decompose this project into modules and tasks now. No hours, no sources."
    )
