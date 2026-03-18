"""Main triple evaluation system orchestrating PMID extraction and LLM evaluation."""
import logging
import asyncio
from typing import List, Dict, Any
from .pmid_extractor import PMIDExtractor
from .evaluation_agent import EvaluationAgent, TripleData, TripleEvaluation
from .llm_factory import create_llm_client
from .config import settings

logger = logging.getLogger(__name__)


class TripleEvaluationResult:
    """Container for the complete evaluation result."""

    def __init__(self, triple: TripleData, evaluations: List[TripleEvaluation]):
        self.triple = triple
        self.evaluations = evaluations

    def format_output(self, verbose: bool = False) -> str:
        """Format the evaluation results for display."""
        lines = []
        for ev in self.evaluations:
            support_text = ev.support.capitalize()
            subject_mentioned = "Yes" if ev.subject_mentioned else "No"
            object_mentioned = "Yes" if ev.object_mentioned else "No"

            main_line = (
                f"PMID:{ev.pmid}, {support_text}, "
                f"Subject:{subject_mentioned}, Object:{object_mentioned}"
            )

            if ev.supporting_sentences:
                sents = " | ".join(ev.supporting_sentences)
                main_line += f", [{sents}]"

            lines.append(main_line)

            if verbose:
                lines.append(f"  Support: {ev.support}")
                lines.append(f"  Subject Mentioned: {subject_mentioned}")
                lines.append(f"  Object Mentioned: {object_mentioned}")
                if ev.supporting_sentences:
                    lines.append("  Supporting Sentences:")
                    for s in ev.supporting_sentences:
                        lines.append(f"    - {s}")
                lines.append(f"  Reasoning: {ev.reasoning}")
                lines.append("")

        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation results."""
        total = len(self.evaluations)
        yes_count = sum(1 for ev in self.evaluations if ev.support == "yes")
        maybe_count = sum(1 for ev in self.evaluations if ev.support == "maybe")
        no_count = sum(1 for ev in self.evaluations if ev.support == "no")

        yes_pct = (yes_count / total * 100) if total > 0 else 0.0
        maybe_pct = (maybe_count / total * 100) if total > 0 else 0.0
        no_pct = (no_count / total * 100) if total > 0 else 0.0

        return {
            "total_pmids": total,
            "yes_count": yes_count,
            "maybe_count": maybe_count,
            "no_count": no_count,
            "yes_percentage": round(yes_pct, 1),
            "maybe_percentage": round(maybe_pct, 1),
            "no_percentage": round(no_pct, 1),
            # Backward-compatible aliases
            "supported_pmids": yes_count,
            "unsupported_pmids": no_count + maybe_count,
            "supported_percentage": round(yes_pct, 1),
            "unsupported_percentage": round((no_pct + maybe_pct), 1),
        }


class TripleEvaluatorSystem:
    """Main system for evaluating triples against PMID abstracts using LLMs."""

    def __init__(self, llm_provider, round2_model=None):
        """Initialize the triple evaluation system.

        Args:
            llm_provider: LLM provider for Round 1 (e.g., 'gpt-oss-20b-vllm', model name).
            round2_model: Optional model for Round 2 re-evaluation. If None, Round 2 is skipped.
        """
        self.pmid_extractor = PMIDExtractor(
            api_key=settings.ncbi_api_key,
            email=settings.ncbi_email,
        )

        llm_client = create_llm_client(llm_provider)
        round2_client = create_llm_client(round2_model) if round2_model else None

        self.evaluation_agent = EvaluationAgent(
            llm_client=llm_client,
            round2_client=round2_client,
        )
        self.round2_model = round2_model
        self.use_round2 = round2_model is not None

    async def evaluate_triple_with_names(
        self,
        subject: str,
        predicate: str,
        object_: str,
        subject_names: List[str] = None,
        object_names: List[str] = None,
        pmids: List[str] = None,
        subject_info: dict = None,
        object_info: dict = None,
    ) -> TripleEvaluationResult:
        """Evaluate a research triple against a list of PMIDs.

        Args:
            subject: Subject entity name
            predicate: Predicate/relation
            object_: Object entity name
            subject_names: Equivalent names for the subject
            object_names: Equivalent names for the object
            pmids: List of PubMed identifiers
            subject_info: Optional dict with name, category, description from node_dict
            object_info: Optional dict with name, category, description from node_dict

        Returns:
            TripleEvaluationResult with evaluation for each PMID
        """
        logger.info(
            f"Evaluating triple ['{subject}' {predicate} '{object_}'] "
            f"against {len(pmids)} PMIDs"
        )

        triple = TripleData(
            subject=subject,
            predicate=predicate,
            object=object_,
            subject_names=subject_names or [subject],
            object_names=object_names or [object_],
            subject_info=subject_info,
            object_info=object_info,
        )

        logger.info(f"Subject equivalent names: {triple.subject_names}")
        logger.info(f"Object equivalent names: {triple.object_names}")
        if triple.subject_info:
            logger.info(f"Subject info: {triple.subject_info.get('name', 'N/A')}")
        if triple.object_info:
            logger.info(f"Object info: {triple.object_info.get('name', 'N/A')}")

        # Step 1: Extract abstracts from PMIDs
        logger.info("Extracting abstracts from PMIDs...")
        abstract_data = self.pmid_extractor.extract_abstracts(pmids)

        # Step 2: Separate valid and invalid PMIDs
        valid_abstracts = []
        evaluations = []

        for pmid in pmids:
            data = abstract_data.get(pmid)
            if not data:
                evaluations.append(TripleEvaluation(
                    pmid=pmid, support="no",
                    reasoning="PMID not found in results",
                ))
                continue
            if data.error:
                logger.warning(f"Error for PMID {pmid}: {data.error}")
                evaluations.append(TripleEvaluation(
                    pmid=pmid, support="no",
                    reasoning=f"Error: {data.error}",
                ))
                continue
            if not data.abstract.strip():
                evaluations.append(TripleEvaluation(
                    pmid=pmid, support="no",
                    reasoning="No abstract available",
                ))
                continue
            valid_abstracts.append((pmid, data.title, data.abstract))

        # Step 3: Evaluate valid abstracts concurrently
        if valid_abstracts:
            logger.info(
                f"Evaluating {len(valid_abstracts)} valid abstracts "
                f"(max {settings.max_concurrent_requests} concurrent)..."
            )
            semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

            async def evaluate_with_semaphore(pmid, title, abstract):
                async with semaphore:
                    try:
                        evaluation = await self.evaluation_agent.evaluate_triple_against_abstract(
                            triple=triple,
                            abstract=abstract,
                            pmid=pmid,
                            title=title,
                            use_round2=self.use_round2,
                        )
                        evaluation = self._validate_evaluation_logic(evaluation, pmid)
                        return evaluation
                    except Exception as e:
                        logger.error(f"Failed to evaluate PMID {pmid}: {e}")
                        return TripleEvaluation(
                            pmid=pmid, support="no",
                            reasoning=f"Evaluation failed: {str(e)}",
                        )

            tasks = [
                evaluate_with_semaphore(pmid, title, abstract)
                for pmid, title, abstract in valid_abstracts
            ]
            batch_evaluations = await asyncio.gather(*tasks)
            evaluations.extend(batch_evaluations)

        # Sort by original PMID order
        pmid_order = {pmid: idx for idx, pmid in enumerate(pmids)}
        evaluations.sort(key=lambda x: pmid_order.get(x.pmid, float('inf')))

        return TripleEvaluationResult(triple=triple, evaluations=evaluations)

    def _validate_evaluation_logic(
        self,
        evaluation: TripleEvaluation,
        pmid: str,
    ) -> TripleEvaluation:
        """Apply validation rules to ensure logical consistency.

        Rules:
        1. "yes" without supporting sentences -> downgrade to "maybe"
        2. "no"/"maybe" should not have supporting sentences
        """
        if evaluation.support == "yes" and not evaluation.supporting_sentences:
            evaluation.support = "maybe"
            evaluation.reasoning += (
                " [Auto-corrected: 'yes' without supporting sentences -> 'maybe']"
            )

        if evaluation.support in ("no", "maybe") and evaluation.supporting_sentences:
            evaluation.supporting_sentences = []
            evaluation.reasoning += (
                " [Auto-corrected: cleared sentences for non-'yes' result]"
            )

        return evaluation
