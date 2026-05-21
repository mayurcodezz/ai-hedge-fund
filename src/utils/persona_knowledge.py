"""persona_knowledge — read each ATLAS persona's knowledge.md and extract
actionable sections for injection into the persona's system_prompt.

Phase 3X (2026-05-21). Per mayur's directive: "make sure each persona is
actually having in-depth knowledge, real detailed knowledge about their
personas, so their context is actually really rich and accessible when
they're making a sound judgment."

Knowledge files live at: `~/Mriga/edge/funds/01-atlas/knowledge/{persona}.md`

Strategy:
- Each .md has a YAML frontmatter + biographical + framework sections
- We don't inject everything (token cost). We inject only the DECISION-RELEVANT
  sections: "## ATLAS lens", "## Signal patterns", "## Citation requirement"
- Loader is cached — read once at module load, served from memory thereafter
- ~500-800 tokens per persona × 8 personas × 2 rounds = ~12k extra tokens per
  fund_01 run = ~$0.05 added cost. Total still under $0.50/run budget.

Usage:
    from src.utils.persona_knowledge import load_persona_knowledge

    snippet = load_persona_knowledge("nassim_taleb_agent")
    # Returns the actionable sections, ready to append to system_prompt
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

KNOWLEDGE_DIR = Path.home() / "Mriga" / "edge" / "funds" / "01-atlas" / "knowledge"

# Persona id (agent name in code) -> filename in knowledge/
PERSONA_FILENAMES = {
    "nassim_taleb_agent": "nassim_taleb.md",
    "mark_spitznagel_agent": "mark_spitznagel.md",
    "sheldon_natenberg_agent": "sheldon_natenberg.md",
    "euan_sinclair_agent": "euan_sinclair.md",
    "tony_saliba_agent": "tony_saliba.md",
    "lawrence_mcmillan_agent": "lawrence_mcmillan.md",
    "pr_sundar_agent": "pr_sundar.md",
    "subasish_pani_agent": "subasish_pani.md",
}

# Sections we extract from each .md (case-insensitive prefix match)
# These are the DECISION-RELEVANT chunks — biographical / books / quotes
# are great for the file but waste tokens at runtime
ACTIONABLE_SECTION_HEADERS = [
    "## ATLAS lens",
    "## Signal patterns",
    "## What",  # catches "What Taleb would signal" / "What Spitznagel examines"
    "## Citation requirement",
    "## Voice fingerprints",
    "## Famous principles",
]


def _extract_sections(md_text: str, headers: list[str]) -> str:
    """Pull sections matching any of `headers` from the markdown text.

    Returns concatenated section text (header + body until next ## or EOF).
    """
    lines = md_text.split("\n")
    chunks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Is this line a heading we want?
        is_match = any(line.startswith(h) for h in headers)
        if is_match:
            section_start = i
            i += 1
            # Collect until next ## heading at same level
            while i < len(lines) and not (lines[i].startswith("## ") and not lines[i].startswith("### ")):
                i += 1
            chunks.append("\n".join(lines[section_start:i]))
        else:
            i += 1
    return "\n\n".join(chunks).strip()


@lru_cache(maxsize=16)
def load_persona_knowledge(persona_agent_id: str) -> Optional[str]:
    """Load the actionable sections of a persona's knowledge.md.

    Cached — read once per process. Returns None if file missing.

    Args:
        persona_agent_id: e.g., "nassim_taleb_agent" / "pr_sundar_agent"

    Returns:
        Concatenated text of ATLAS lens + Signal patterns + Voice + Citation
        sections, ready to append to system_prompt. Or None if file not found.
    """
    filename = PERSONA_FILENAMES.get(persona_agent_id)
    if not filename:
        return None

    path = KNOWLEDGE_DIR / filename
    if not path.exists():
        return None

    try:
        text = path.read_text()
    except Exception:
        return None

    # Strip YAML frontmatter (--- ... ---)
    if text.startswith("---"):
        end_marker = text.find("\n---", 4)
        if end_marker > 0:
            text = text[end_marker + 4:]

    sections = _extract_sections(text, ACTIONABLE_SECTION_HEADERS)
    if not sections:
        return None

    # Wrap with delimiter so it's clear in the prompt
    return (
        "\n\n--- KNOWLEDGE BASE (cite specific data from OptionsContext per "
        "Citation requirement) ---\n\n"
        + sections
        + "\n\n--- END KNOWLEDGE BASE ---"
    )


def has_knowledge(persona_agent_id: str) -> bool:
    """Quick check whether a persona has a knowledge file."""
    filename = PERSONA_FILENAMES.get(persona_agent_id)
    if not filename:
        return False
    return (KNOWLEDGE_DIR / filename).exists()
