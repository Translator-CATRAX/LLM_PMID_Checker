"""Anti-hallucination sentence mapper.

Verifies that LLM-extracted sentences actually exist in the source abstract.
Ported from test/utils.py with adaptations for the src/ pipeline.
"""
import re
import logging
import unicodedata
from typing import List, Optional, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def normalize_sentence(sentence: str) -> str:
    """Normalize a sentence for comparison.

    Handles Unicode normalization, whitespace, hyphens, quotes, and punctuation.
    """
    sentence = sentence.strip()
    sentence = unicodedata.normalize('NFKD', sentence)
    # Various hyphens/dashes → standard hyphen
    sentence = re.sub(r'[\u2010-\u2015]', '-', sentence)
    # Various quotes → standard
    sentence = re.sub(r'[\u2018\u2019\u201a]', "'", sentence)
    sentence = re.sub(r'[\u201c\u201d\u201e]', '"', sentence)
    sentence = sentence.replace('\\/', '/')
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'\s+([.,;:!?])', r'\1', sentence)
    return sentence


def is_exact_match(sent1: str, sent2: str) -> bool:
    """Check if two sentences match after normalization (case-insensitive)."""
    norm1 = normalize_sentence(sent1)
    norm2 = normalize_sentence(sent2)

    if norm1.lower() == norm2.lower():
        return True

    words1 = re.findall(r'\b\w+\b', norm1.lower())
    words2 = re.findall(r'\b\w+\b', norm2.lower())
    return words1 == words2


def string_similarity(str1: str, str2: str) -> float:
    """Calculate similarity ratio between two strings (0-1)."""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def find_matching_sentence_index(
    llm_sent: str,
    sentences_list: List[str],
) -> Optional[int]:
    """Find the index of a matching sentence using multiple strategies.

    Strategies tried in order:
    1. Exact match (normalized, case-insensitive)
    2. Substring match (LLM sentence contained in original)
    3. Fuzzy match (>= 95% similarity)
    """
    norm_llm = normalize_sentence(llm_sent)

    for idx, orig_sent in enumerate(sentences_list):
        norm_orig = normalize_sentence(orig_sent)

        if is_exact_match(llm_sent, orig_sent):
            return idx

        llm_for_matching = norm_llm.lower()
        if llm_for_matching.endswith('.'):
            llm_for_matching = llm_for_matching[:-1].strip()

        if llm_for_matching in norm_orig.lower():
            return idx

        if string_similarity(llm_for_matching, norm_orig) >= 0.95:
            return idx

        if llm_for_matching == norm_orig.lower().rstrip('.'):
            return idx

    return None


def find_sentence_with_ellipsis(
    ellipsis_sent: str,
    sentences_list: List[str],
) -> Optional[List[int]]:
    """Find sentence(s) that match an ellipsis pattern.

    Handles:
    - '...' at end: find sentence starting with the given text
    - '...' at beginning: find sentence ending with the given text
    - '...' in middle: find sentences containing all parts in order
    """
    if '...' not in ellipsis_sent:
        return None

    norm_ellipsis = normalize_sentence(ellipsis_sent)

    # Ellipsis at the end only
    if norm_ellipsis.endswith('...') and norm_ellipsis.count('...') == 1:
        prefix = normalize_sentence(norm_ellipsis[:-3]).strip().rstrip('.')
        if prefix:
            for idx, sentence in enumerate(sentences_list):
                norm_sent = normalize_sentence(sentence)
                if norm_sent.lower().startswith(prefix.lower()):
                    return [idx]
        return None

    # Ellipsis at the beginning only
    if norm_ellipsis.startswith('...') and norm_ellipsis.count('...') == 1:
        suffix = normalize_sentence(norm_ellipsis[3:]).strip()
        if suffix:
            for idx, sentence in enumerate(sentences_list):
                norm_sent = normalize_sentence(sentence)
                if norm_sent.lower().endswith(suffix.lower()):
                    return [idx]
        return None

    # Ellipsis in the middle: split and check parts appear in order
    parts = [normalize_sentence(p).strip() for p in ellipsis_sent.split('...')]
    parts = [p.strip('.').strip() for p in parts if p.strip()]

    if len(parts) < 2:
        return None

    # Try single sentence containing all parts
    for idx, sentence in enumerate(sentences_list):
        norm_sent = normalize_sentence(sentence).lower()
        current_pos = 0
        all_found = True
        for part in parts:
            pos = norm_sent.find(part.lower().strip('.,;:'), current_pos)
            if pos == -1:
                all_found = False
                break
            current_pos = pos + len(part)
        if all_found:
            return [idx]

    # Try multiple sentences
    matched_indices = []
    for part in parts:
        part_lower = part.lower().strip('.,;:')
        found = False
        for idx, sentence in enumerate(sentences_list):
            norm_sent = normalize_sentence(sentence).lower()
            if part_lower in norm_sent:
                matched_indices.append(idx)
                found = True
                break
        if not found:
            return None

    if matched_indices != sorted(matched_indices) or len(matched_indices) < 2:
        return None

    return list(range(matched_indices[0], matched_indices[-1] + 1))


def verify_sentences(
    llm_sentences: List[str],
    abstract_text: str,
) -> Tuple[List[str], List[str]]:
    """Verify LLM-extracted sentences actually exist in the abstract.

    Uses normalized comparison and fuzzy matching to account for
    minor differences in tokenization or formatting.

    Args:
        llm_sentences: Sentences extracted by the LLM
        abstract_text: Full abstract text

    Returns:
        Tuple of (verified_sentences, unverified_sentences)
    """
    if not llm_sentences:
        return [], []

    # Sentence-tokenize the abstract for index-based matching
    try:
        from nltk.tokenize import sent_tokenize
        abstract_sentences = sent_tokenize(abstract_text)
    except Exception:
        abstract_sentences = [s.strip() for s in abstract_text.split('.') if s.strip()]

    verified = []
    unverified = []

    for llm_sent in llm_sentences:
        if not llm_sent or not llm_sent.strip():
            continue

        # Strategy 1: Index-based match against sentence-tokenized abstract
        idx = find_matching_sentence_index(llm_sent, abstract_sentences)
        if idx is not None:
            verified.append(llm_sent)
            continue

        # Strategy 2: Substring match against full abstract
        norm_llm = normalize_sentence(llm_sent)
        norm_abstract = normalize_sentence(abstract_text)
        if norm_llm.lower() in norm_abstract.lower():
            verified.append(llm_sent)
            continue

        # Strategy 3: Ellipsis handling
        if '...' in llm_sent:
            result = find_sentence_with_ellipsis(llm_sent, abstract_sentences)
            if result is not None:
                verified.append(llm_sent)
                continue

        unverified.append(llm_sent)

    if unverified:
        logger.warning(
            f"Sentence verification: {len(verified)} verified, "
            f"{len(unverified)} unverified (potential hallucination)"
        )
        for sent in unverified:
            logger.debug(f"  Unverified: {sent[:100]}...")

    return verified, unverified
