"""Shared prompt builder for LLM evaluation across all backends.

Builds evaluation prompts based on the test/LLM_validator.py style
(yes/no/maybe + supporting sentences) with equivalent names and
optional entity info from node_dict.
"""
from typing import List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluation_agent import TripleData


def get_matching_rules() -> str:
    """Return shared matching rules for entity and relationship matching."""
    return (
        "**MATCHING RULES**:\n"
        "1. **Entity Matching**: Start with equivalent names from the provided lists above\n"
        "   - You MAY use your knowledge to recognize common abbreviations and variants "
        "of the equivalent names (e.g., 'AAA' from 'AAA gene/protein')\n"
        "   - BUT any alternative name you use MUST be clearly related to names in the provided list\n"
        "2. **Relationship Matching**: Match by semantic meaning:\n"
        "   - 'inhibits' = 'suppresses' = 'reduces' = 'blocks' = 'decreases activity/expression/abundance'\n"
        "   - 'activates' = 'stimulates' = 'promotes' = 'increases' = 'upregulates' = 'enhances' "
        "= 'increases activity/expression/abundance'\n"
        "   - 'expression', 'abundance', and 'levels' are interchangeable\n\n"
        "**CRITICAL REQUIREMENTS**:\n"
        "• Relationship MUST be between the SUBJECT and OBJECT from the triple\n"
        "• Do NOT confuse relationships involving other entities\n"
        "• Supporting sentences MUST EXPLICITLY MENTION BOTH SUBJECT and OBJECT (or equivalent names)\n\n"
        "**CRITICAL LOGIC RULES**:\n"
        "1. **Correlation ≠ Causation**: 'A correlates with B' does NOT support 'A causes B'\n"
        "2. **No Transitive Reasoning**: If X→A and X→B, does NOT mean A→B (must be direct)\n"
        "3. **Opposite = No**: Inverse relationship → answer 'no'\n"
        "4. **Both Entities Required**: Supporting sentences must mention BOTH entities "
        "(names from the equivalent names list or via common abbreviations)\n\n"
    )


def _extract_triple_info(triple: Union[List[str], 'TripleData']) -> dict:
    """Extract all triple information into a flat dictionary."""
    if hasattr(triple, 'subject'):
        return {
            'subject': triple.subject,
            'predicate': triple.predicate,
            'object': triple.object,
            'subject_names': getattr(triple, 'subject_names', None) or [triple.subject],
            'object_names': getattr(triple, 'object_names', None) or [triple.object],
            'subject_info': getattr(triple, 'subject_info', None),
            'object_info': getattr(triple, 'object_info', None),
            'qualified_predicate': getattr(triple, 'qualified_predicate', None),
            'qualified_object_aspect': getattr(triple, 'qualified_object_aspect', None),
            'qualified_object_direction': getattr(triple, 'qualified_object_direction', None),
            'has_qualifiers': getattr(triple, 'has_qualifiers', lambda: False)(),
        }
    else:
        return {
            'subject': triple[0],
            'predicate': triple[1],
            'object': triple[2],
            'subject_names': [triple[0]],
            'object_names': [triple[2]],
            'subject_info': None,
            'object_info': None,
            'qualified_predicate': None,
            'qualified_object_aspect': None,
            'qualified_object_direction': None,
            'has_qualifiers': False,
        }


def _build_triple_description(info: dict) -> str:
    """Build the triple description string, with qualifiers if present."""
    if info['has_qualifiers']:
        parts = []
        if info['qualified_object_direction']:
            parts.append(info['qualified_object_direction'])
        if info['qualified_object_aspect']:
            parts.append(info['qualified_object_aspect'])
        qualified_desc = " ".join(parts)
        return (
            f"'{info['subject']}' {info['qualified_predicate']} "
            f"{qualified_desc} of '{info['object']}'"
        )
    return f"'{info['subject']}' {info['predicate']} '{info['object']}'"


def _build_entity_section(
    entity_type: str,
    name: str,
    entity_info: Optional[dict],
    equiv_names: List[str],
) -> str:
    """Build the entity info + equivalent names section."""
    section = f"**{entity_type.upper()}**: {name}\n"

    if entity_info:
        if entity_info.get('name'):
            section += f"  Name: {entity_info['name']}\n"
        if entity_info.get('category'):
            section += f"  Category: {entity_info['category']}\n"
        if entity_info.get('description'):
            section += f"  Description: {entity_info['description']}\n"

    label = entity_type.upper()
    section += (
        f"\n**{label} EQUIVALENT NAMES** "
        f"(check for ANY of these + common abbreviations in the abstract):\n"
    )
    for i, n in enumerate(equiv_names, 1):
        section += f"  {i}. {n}\n"

    return section


def _build_qualifier_section(info: dict) -> str:
    """Build qualifier guidance if qualifiers are present."""
    if not info['has_qualifiers']:
        return ""

    qp = info['qualified_predicate']
    qd = info['qualified_object_direction']
    qa = info['qualified_object_aspect']

    section = (
        f"\n**QUALIFIERS TO CHECK**:\n"
        f"- Predicate: {qp}\n"
        f"- Direction: {qd or 'any'}\n"
        f"- Aspect: {qa or 'any'}\n\n"
        f"**QUALIFIER MATCHING**:\n"
    )

    if qd:
        if qd == 'increased':
            examples = 'increases/activates/upregulates'
        elif qd == 'decreased':
            examples = 'decreases/inhibits/downregulates'
        else:
            examples = qd
        section += (
            f"• Direction '{qd}': Abstract must show {qd} effect "
            f"(e.g., {examples})\n"
        )

    if qa == 'activity_or_abundance':
        section += (
            "• Aspect 'activity_or_abundance': Abstract can mention "
            "activity OR abundance OR both\n"
        )

    section += (
        "• Match semantic meaning: 'inhibitor' = 'causes decreased activity', "
        "'upregulates expression' = 'causes increased abundance'\n\n"
    )

    return section


def _get_instructions() -> str:
    """Return the evaluation instructions."""
    return (
        "**INSTRUCTIONS**:\n"
        "- Determine if the abstract provides evidence for this triple.\n"
        '- Use "yes" if the relation is explicitly supported.\n'
        '- Use "no" if the relation is not mentioned or contradicted.\n'
        '- Use "maybe" if the evidence is indirect, ambiguous, or suggestive.\n'
        '- If "yes", return one or more exact supporting sentences from the abstract.\n'
        "  Multiple sentences are allowed if they together support the triple.\n"
        '- If "no" or "maybe", return an empty list for "sentences".\n'
        "- For subject_mentioned / object_mentioned: set to true if the entity appears\n"
        "  ANYWHERE in the abstract (using equivalent names or common abbreviations).\n"
        "  Entity mention is INDEPENDENT from whether the triple is supported.\n"
        '- For "reasoning": briefly explain why you chose yes/no/maybe.\n\n'
    )


def _get_output_format() -> str:
    """Return the JSON output format specification."""
    return (
        "**OUTPUT** (JSON only, no other text):\n"
        "{\n"
        '  "support": "yes" | "no" | "maybe",\n'
        '  "sentences": ["exact sentence from abstract", ...],\n'
        '  "subject_mentioned": true/false,\n'
        '  "object_mentioned": true/false,\n'
        '  "reasoning": "brief explanation of your judgment"\n'
        "}\n"
    )


def build_evaluation_prompt(
    triple: Union[List[str], 'TripleData'],
    abstract: str,
) -> str:
    """Build the complete evaluation prompt.

    Args:
        triple: TripleData object or [subject, predicate, object] list
        abstract: Abstract text to evaluate

    Returns:
        Complete prompt string
    """
    info = _extract_triple_info(triple)

    prompt = (
        "Please analyze whether the provided abstract supports the following triple.\n"
        "Carefully consider the subject, object, and predicate details.\n\n"
        f"**TRIPLE**: {_build_triple_description(info)}\n\n"
        f"{_build_entity_section('Subject', info['subject'], info['subject_info'], info['subject_names'])}\n"
        f"{_build_entity_section('Object', info['object'], info['object_info'], info['object_names'])}\n"
        f"{_build_qualifier_section(info)}"
        f"**ABSTRACT**:\n{abstract}\n\n"
        f"{get_matching_rules()}"
        f"{_get_instructions()}"
        f"{_get_output_format()}"
    )

    return prompt
