"""PMID abstract extraction using easy-entrez with local caching."""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from easy_entrez import EntrezAPI
from src.pmid_cache import PMIDCache

logger = logging.getLogger(__name__)


def _normalize_pmid(raw: str) -> str:
    """Strip 'PMID:' prefix to get the bare numeric ID expected by NCBI."""
    raw = raw.strip()
    if raw.upper().startswith("PMID:"):
        return raw[5:]
    return raw

@dataclass
class AbstractData:
    """Container for PMID abstract data."""
    pmid: str
    title: str
    abstract: str
    error: Optional[str] = None

class PMIDExtractor:
    """Extracts abstracts from PubMed using easy-entrez with local caching."""
    
    def __init__(self, api_key, email, use_cache: bool = True):
        """Initialize the PMID extractor.
        
        Args:
            api_key: NCBI E-utilities API key for higher rate limits
            email: Email for NCBI requests (required for higher volumes)
            use_cache: If True, use local cache before querying NCBI (default: True)
        """
        # Initialize EntrezAPI
        self.entrez_api = EntrezAPI(
            tool="llm_pmid_checker",
            email=email,
            api_key=api_key,
            return_type='xml'
        )
        
        # Initialize cache if enabled
        self.use_cache = use_cache
        self.cache = PMIDCache() if use_cache else None
        
    def extract_abstracts(self, pmids: List[str]) -> Dict[str, AbstractData]:
        """Extract abstracts for a list of PMIDs, using cache when available.
        
        Accepts PMIDs with or without the ``PMID:`` prefix.  Internally all
        identifiers are normalized to bare numeric strings for cache lookups
        and NCBI API calls.  The returned dictionary is keyed by the
        **original** PMID strings the caller passed in.
        
        Args:
            pmids: List of PubMed identifiers (e.g. "2843522" or "PMID:2843522")
            
        Returns:
            Dictionary mapping the original PMID string to AbstractData
        """
        results: Dict[str, AbstractData] = {}
        
        if not pmids:
            return results

        bare_to_orig: Dict[str, str] = {}
        for orig in pmids:
            bare_to_orig[_normalize_pmid(orig)] = orig
        bare_ids = list(bare_to_orig.keys())
        
        # Try to get cached results first
        bare_to_fetch = bare_ids
        if self.use_cache and self.cache:
            cached_results = self.cache.get_many(bare_ids)
            
            for bare, cached_abstract in cached_results.items():
                orig = bare_to_orig[bare]
                results[orig] = AbstractData(
                    pmid=bare,
                    title=cached_abstract.title,
                    abstract=cached_abstract.abstract,
                    error=None
                )
            
            bare_to_fetch = [b for b in bare_ids if b not in cached_results]
            
            if cached_results:
                logger.info(f"Found {len(cached_results)}/{len(pmids)} PMIDs in cache")
            
            if not bare_to_fetch:
                logger.info("All PMIDs found in cache, no NCBI query needed")
                return results
        
        # Fetch remaining PMIDs from NCBI
        logger.info(f"Fetching {len(bare_to_fetch)} PMIDs from NCBI E-utilities")
        
        fetched_bare: Dict[str, AbstractData] = {}
        try:
            response = self.entrez_api.fetch(
                bare_to_fetch,
                max_results=len(bare_to_fetch),
                database='pubmed'
            )
            
            if response.data:
                import xml.etree.ElementTree as ET
                root = response.data if hasattr(response.data, 'tag') else ET.fromstring(str(response.data))
                
                articles = root.findall('.//PubmedArticle')
                
                for article in articles:
                    try:
                        pmid_elem = article.find('.//PMID')
                        bare = pmid_elem.text.strip() if pmid_elem is not None else ""
                        
                        if not bare:
                            continue
                        
                        title_elem = article.find('.//ArticleTitle')
                        title = ''.join(title_elem.itertext()).strip() if title_elem is not None else ""
                        
                        abstract_elems = article.findall('.//AbstractText')
                        abstract_parts = []
                        for elem in abstract_elems:
                            text = ''.join(elem.itertext()).strip()
                            if text:
                                label = elem.get('Label', '')
                                if label:
                                    abstract_parts.append(f"{label}: {text}")
                                else:
                                    abstract_parts.append(text)
                        
                        abstract = ' '.join(abstract_parts) if abstract_parts else ""
                        
                        abstract_data = AbstractData(
                            pmid=bare,
                            title=title,
                            abstract=abstract,
                            error=None if abstract else "No abstract available"
                        )
                        
                        fetched_bare[bare] = abstract_data
                        
                        if self.use_cache and self.cache:
                            self.cache.put(
                                pmid=bare,
                                title=title,
                                abstract=abstract,
                                fetch_date=datetime.utcnow().isoformat(),
                                error=abstract_data.error
                            )
                        
                    except Exception as e:
                        logger.error(f"Error parsing article data: {e}")
                        continue
            
            # Handle PMIDs that weren't found in the response
            for bare in bare_to_fetch:
                if bare not in fetched_bare:
                    abstract_data = AbstractData(
                        pmid=bare,
                        title="",
                        abstract="",
                        error="PMID not found or could not be retrieved"
                    )
                    fetched_bare[bare] = abstract_data
                    
                    if self.use_cache and self.cache:
                        self.cache.put(
                            pmid=bare,
                            title="",
                            abstract="",
                            fetch_date=datetime.utcnow().isoformat(),
                            error=abstract_data.error
                        )
                    
        except Exception as e:
            logger.error(f"Failed to extract abstracts: {e}")
            for bare in bare_to_fetch:
                if bare not in fetched_bare:
                    abstract_data = AbstractData(
                        pmid=bare,
                        title="",
                        abstract="",
                        error=f"Extraction failed: {str(e)}"
                    )
                    fetched_bare[bare] = abstract_data
                    
                    if self.use_cache and self.cache:
                        self.cache.put(
                            pmid=bare,
                            title="",
                            abstract="",
                            fetch_date=datetime.utcnow().isoformat(),
                            error=abstract_data.error
                        )

        # Map fetched results back to original PMID strings
        for bare, data in fetched_bare.items():
            orig = bare_to_orig.get(bare, bare)
            results[orig] = data
        
        return results
    