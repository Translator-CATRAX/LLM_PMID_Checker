"""LLM agent for evaluating triples against PMID abstracts."""
import logging
from typing import List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TripleEvaluation:
    """Result of triple evaluation against an abstract."""
    pmid: str
    support: str = "no"  # "yes", "no", "maybe"
    supporting_sentences: List[str] = field(default_factory=list)
    reasoning: str = ""
    subject_mentioned: bool = False
    object_mentioned: bool = False

    @property
    def is_supported(self) -> bool:
        """Backward-compatible property."""
        return self.support == "yes"


@dataclass
class TripleData:
    """Input triple data for evaluation."""
    subject: str
    predicate: str
    object: str
    subject_names: List[str] = None
    object_names: List[str] = None
    subject_info: Optional[dict] = None
    object_info: Optional[dict] = None
    qualified_predicate: Optional[str] = None
    qualified_object_aspect: Optional[str] = None
    qualified_object_direction: Optional[str] = None

    def __post_init__(self):
        if self.subject_names is None:
            self.subject_names = [self.subject]
        if self.object_names is None:
            self.object_names = [self.object]
        self._validate_qualifiers()

    def _validate_qualifiers(self):
        has_any_qualifier = any([
            self.qualified_predicate,
            self.qualified_object_aspect,
            self.qualified_object_direction,
        ])
        if has_any_qualifier:
            if not self.qualified_predicate:
                raise ValueError(
                    "qualified_predicate is required when using qualifiers"
                )
            if not self.qualified_object_aspect and not self.qualified_object_direction:
                raise ValueError(
                    "At least one of qualified_object_aspect or "
                    "qualified_object_direction must be provided when using qualifiers"
                )

    def has_qualifiers(self) -> bool:
        return bool(self.qualified_predicate)

    def to_string(self) -> str:
        if self.has_qualifiers():
            parts = []
            if self.qualified_object_direction:
                parts.append(self.qualified_object_direction)
            if self.qualified_object_aspect:
                parts.append(self.qualified_object_aspect)
            qualified_desc = " ".join(parts)
            return (
                f"'{self.subject}' {self.qualified_predicate} "
                f"{qualified_desc} of '{self.object}'"
            )
        return f"'{self.subject}' {self.predicate} '{self.object}'"


class EvaluationAgent:
    """Agent for evaluating whether abstracts support research triples."""

    def __init__(self, llm_client, round2_client=None):
        """Initialize the evaluation agent.

        Args:
            llm_client: LLM client for Round 1 evaluation
            round2_client: Optional LLM client for Round 2 re-evaluation
        """
        self.llm_client = llm_client
        self.round2_client = round2_client
        logger.info("Evaluation agent initialized")

    async def evaluate_triple_against_abstract(
        self,
        triple: TripleData,
        abstract: str,
        pmid: str,
        title: str = "",
        use_round2: bool = False,
    ) -> TripleEvaluation:
        """Evaluate whether an abstract supports a research triple.

        Round 1: Primary evaluation using llm_client.
        Round 2 (optional): Independent re-evaluation of "yes"/"maybe" results
                            using round2_client with the same prompt.

        Args:
            triple: The research triple to evaluate
            abstract: The abstract text to analyze
            pmid: PubMed ID for the abstract
            title: Article title (optional)
            use_round2: Whether to run Round 2 re-evaluation

        Returns:
            TripleEvaluation result
        """
        from .sentence_mapper import verify_sentences

        try:
            # Round 1: Primary evaluation
            result = await self.llm_client.evaluate_triple_support(
                triple=triple,
                abstract=abstract,
            )

            evaluation = self._parse_result(result, pmid)

            # Anti-hallucination: verify supporting sentences
            if evaluation.supporting_sentences:
                verified, unverified = verify_sentences(
                    evaluation.supporting_sentences, abstract
                )
                if unverified:
                    logger.warning(
                        f"PMID {pmid}: {len(unverified)} unverified sentences removed"
                    )
                evaluation.supporting_sentences = verified

                if not verified and evaluation.support == "yes":
                    evaluation.support = "maybe"
                    evaluation.reasoning += (
                        " [Auto-corrected: all supporting sentences failed verification]"
                    )

            # Round 2: Independent re-evaluation for yes/maybe results
            if (
                use_round2
                and self.round2_client
                and evaluation.support in ("yes", "maybe")
            ):
                logger.info(
                    f"Running Round 2 re-evaluation for PMID {pmid} "
                    f"(Round 1: {evaluation.support})"
                )
                try:
                    r2_result = await self.round2_client.evaluate_triple_support(
                        triple=triple,
                        abstract=abstract,
                    )
                    r2_eval = self._parse_result(r2_result, pmid)

                    # Verify Round 2 sentences too
                    if r2_eval.supporting_sentences:
                        r2_verified, _ = verify_sentences(
                            r2_eval.supporting_sentences, abstract
                        )
                        r2_eval.supporting_sentences = r2_verified

                    # Round 2 takes precedence
                    r2_eval.reasoning = (
                        f"[Round2] {r2_eval.reasoning} "
                        f"[Round1 was: {evaluation.support}]"
                    )
                    evaluation = r2_eval

                except Exception as e:
                    logger.error(f"Round 2 failed for PMID {pmid}: {e}")
                    evaluation.reasoning += f" [Round 2 failed: {str(e)}]"

            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating PMID {pmid}: {e}")
            return TripleEvaluation(
                pmid=pmid,
                support="no",
                reasoning=f"Evaluation failed: {str(e)}",
            )

    def _parse_result(self, result: dict, pmid: str) -> TripleEvaluation:
        """Parse LLM result dict into a TripleEvaluation.

        Handles both the new format (support/sentences) and legacy format
        (is_supported/evidence_category/supporting_sentence).
        """
        # New format: support + sentences
        support = result.get("support", "").lower().strip()

        if support not in ("yes", "no", "maybe"):
            # Legacy fallback
            if result.get("is_supported"):
                support = "yes"
            elif result.get("evidence_category") == "opposite_assertion":
                support = "no"
            elif result.get("evidence_category") in (
                "missing_qualifier", "wrong_qualifier"
            ):
                support = "maybe"
            else:
                support = "no"

        # Handle sentences field (new) vs supporting_sentence (legacy)
        sentences = result.get("sentences", [])
        if not sentences:
            legacy_sent = result.get("supporting_sentence")
            if legacy_sent and isinstance(legacy_sent, str) and legacy_sent.strip():
                sentences = [legacy_sent.strip()]

        if isinstance(sentences, str):
            sentences = [sentences] if sentences.strip() else []

        return TripleEvaluation(
            pmid=pmid,
            support=support,
            supporting_sentences=sentences,
            reasoning=result.get("reasoning", ""),
            subject_mentioned=result.get("subject_mentioned", False),
            object_mentioned=result.get("object_mentioned", False),
        )
