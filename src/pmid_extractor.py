"""PMID abstract extraction using easy-entrez with local caching."""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from easy_entrez import EntrezAPI
from src.pmid_cache import PMIDCache

logger = logging.getLogger(__name__)

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
        
        Args:
            pmids: List of PubMed identifiers
            
        Returns:
            Dictionary mapping PMID to AbstractData
        """
        results = {}
        
        if not pmids:
            return results
        
        # Try to get cached results first
        pmids_to_fetch = pmids
        if self.use_cache and self.cache:
            cached_results = self.cache.get_many(pmids)
            
            # Add cached results to output
            for pmid, cached_abstract in cached_results.items():
                results[pmid] = AbstractData(
                    pmid=cached_abstract.pmid,
                    title=cached_abstract.title,
                    abstract=cached_abstract.abstract,
                    error=None
                )
            
            # Filter out already cached PMIDs
            pmids_to_fetch = [pmid for pmid in pmids if pmid not in cached_results]
            
            if cached_results:
                logger.info(f"Found {len(cached_results)}/{len(pmids)} PMIDs in cache")
            
            if not pmids_to_fetch:
                logger.info("All PMIDs found in cache, no NCBI query needed")
                return results
        
        # Fetch remaining PMIDs from NCBI
        logger.info(f"Fetching {len(pmids_to_fetch)} PMIDs from NCBI E-utilities")
        
        try:
            # Fetch articles from PubMed
            response = self.entrez_api.fetch(
                pmids_to_fetch,
                max_results=len(pmids_to_fetch),
                database='pubmed'
            )
            
            # Parse the XML response data
            if response.data:
                import xml.etree.ElementTree as ET
                root = response.data if hasattr(response.data, 'tag') else ET.fromstring(str(response.data))
                
                # Find all PubmedArticle elements
                articles = root.findall('.//PubmedArticle')
                
                for article in articles:
                    try:
                        # Extract PMID
                        pmid_elem = article.find('.//PMID')
                        pmid = pmid_elem.text if pmid_elem is not None else ""
                        
                        if not pmid:
                            continue
                        
                        # Extract title - use itertext() to handle nested formatting tags
                        title_elem = article.find('.//ArticleTitle')
                        title = ''.join(title_elem.itertext()).strip() if title_elem is not None else ""
                        
                        # Extract abstract - combine all AbstractText elements
                        abstract_elems = article.findall('.//AbstractText')
                        abstract_parts = []
                        for elem in abstract_elems:
                            text = ''.join(elem.itertext()).strip()
                            if text:
                                # Check if there's a label attribute
                                label = elem.get('Label', '')
                                if label:
                                    abstract_parts.append(f"{label}: {text}")
                                else:
                                    abstract_parts.append(text)
                        
                        abstract = ' '.join(abstract_parts) if abstract_parts else ""
                        
                        # Create AbstractData object
                        abstract_data = AbstractData(
                            pmid=pmid,
                            title=title,
                            abstract=abstract,
                            error=None if abstract else "No abstract available"
                        )
                        
                        results[pmid] = abstract_data
                        
                        # Cache the result if caching is enabled
                        if self.use_cache and self.cache:
                            self.cache.put(
                                pmid=pmid,
                                title=title,
                                abstract=abstract,
                                fetch_date=datetime.utcnow().isoformat(),
                                error=abstract_data.error
                            )
                        
                    except Exception as e:
                        logger.error(f"Error parsing article data: {e}")
                        continue
            
            # Handle PMIDs that weren't found in the response
            for pmid in pmids_to_fetch:
                if pmid not in results:
                    abstract_data = AbstractData(
                        pmid=pmid,
                        title="",
                        abstract="",
                        error="PMID not found or could not be retrieved"
                    )
                    results[pmid] = abstract_data
                    
                    # Cache the error result if caching is enabled
                    if self.use_cache and self.cache:
                        self.cache.put(
                            pmid=pmid,
                            title="",
                            abstract="",
                            fetch_date=datetime.utcnow().isoformat(),
                            error=abstract_data.error
                        )
                    
        except Exception as e:
            logger.error(f"Failed to extract abstracts: {e}")
            for pmid in pmids_to_fetch:
                if pmid not in results:
                    abstract_data = AbstractData(
                        pmid=pmid,
                        title="",
                        abstract="",
                        error=f"Extraction failed: {str(e)}"
                    )
                    results[pmid] = abstract_data
                    
                    # Cache the error result if caching is enabled
                    if self.use_cache and self.cache:
                        self.cache.put(
                            pmid=pmid,
                            title="",
                            abstract="",
                            fetch_date=datetime.utcnow().isoformat(),
                            error=abstract_data.error
                        )
        
        return results
    