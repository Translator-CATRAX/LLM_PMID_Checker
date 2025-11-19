"""
HGNC REST API Client for Gene/Protein Name Resolution

This module provides a safe way to query the HGNC (HUGO Gene Nomenclature Committee)
REST API to find gene and protein names while filtering out non-gene entities like
diseases, phenotypes, and pseudogenes.

Documentation: https://www.genenames.org/help/rest/
"""

import requests
from typing import Dict, List, Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class HGNCClient:
    """Client for querying HGNC REST API with automatic filtering for genes/proteins."""
    
    BASE_URL = "https://rest.genenames.org"
    
    # Locus types that represent actual genes with protein products
    VALID_GENE_TYPES = [
        "gene with protein product",
        "RNA, micro",
        "RNA, ribosomal",
        "RNA, transfer",
        "RNA, long non-coding",
        "RNA, small nuclear",
        "RNA, small nucleolar",
    ]
    
    # Locus groups to include (primary: protein-coding genes)
    VALID_LOCUS_GROUPS = [
        "protein-coding gene",
        "non-coding RNA",
    ]
    
    # Status values to include
    VALID_STATUS = [
        "Approved",
    ]
    
    def __init__(self, rate_limit_delay: float = 0.1, timeout: int = 10):
        """
        Initialize HGNC client.
        
        Args:
            rate_limit_delay: Delay between requests in seconds (default: 0.1s = 10 req/s)
            timeout: Request timeout in seconds (default: 10)
        """
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'Accept': 'application/json'})
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting (max 10 requests per second)."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self._last_request_time = time.time()
    
    def _make_request(self, endpoint: str) -> Optional[Dict]:
        """
        Make a request to HGNC API with rate limiting.
        
        Args:
            endpoint: API endpoint (e.g., "/search/symbol/BRAF")
            
        Returns:
            JSON response as dictionary, or None if request fails
        """
        self._rate_limit()
        
        try:
            response = self.session.get(f"{self.BASE_URL}{endpoint}", timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                logger.error("Rate limit exceeded (403 error). Please reduce request frequency.")
                return None
            else:
                logger.warning(f"HGNC API returned status {response.status_code} for {endpoint}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def search_gene(
        self, 
        query: str, 
        only_protein_coding: bool = True,
        only_approved: bool = True,
        max_results: int = 10
    ) -> List[Dict]:
        """
        Search for genes/proteins by name or symbol.
        
        This method automatically filters out:
        - Phenotype-only entries (disease names)
        - Pseudogenes
        - Withdrawn entries
        - Non-protein-coding genes (if only_protein_coding=True)
        
        Args:
            query: Search term (gene name, symbol, or alias)
            only_protein_coding: If True, only return protein-coding genes (default: True)
            only_approved: If True, only return approved genes (default: True)
            max_results: Maximum number of results to return (default: 10)
            
        Returns:
            List of dictionaries with keys: symbol, hgnc_id, score
        """
        # Pre-process query: replace "/" with space to avoid API errors
        # The HGNC API returns 400 errors when "/" is in the query
        sanitized_query = query.replace('/', ' ')
        
        # Build filtered search query
        search_terms = [sanitized_query]
        
        if only_protein_coding:
            search_terms.append('locus_group:"protein-coding gene"')
        else:
            # Exclude phenotype-only entries
            search_terms.append('NOT locus_group:phenotype')
        
        if only_approved:
            search_terms.append('status:Approved')
        
        search_query = " AND ".join(search_terms)
        # URL encode the query
        encoded_query = requests.utils.quote(search_query)
        
        endpoint = f"/search/{encoded_query}"
        result = self._make_request(endpoint)
        
        if result and 'response' in result:
            docs = result['response']['docs'][:max_results]
            logger.info(f"Found {result['response']['numFound']} total results, returning top {len(docs)}")
            return docs
        
        return []
    
    def fetch_gene_details(self, hgnc_id: str) -> Optional[Dict]:
        """
        Fetch detailed information for a gene by HGNC ID.
        
        Args:
            hgnc_id: HGNC ID (e.g., "HGNC:11406")
            
        Returns:
            Dictionary with all available gene information, or None if not found
        """
        endpoint = f"/fetch/hgnc_id/{hgnc_id}"
        result = self._make_request(endpoint)
        
        if result and 'response' in result and result['response']['docs']:
            return result['response']['docs'][0]
        
        return None
    
    def fetch_by_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Fetch detailed information for a gene by its symbol.
        
        Args:
            symbol: Gene symbol (e.g., "BRAF", "STK3")
            
        Returns:
            Dictionary with all available gene information, or None if not found
        """
        endpoint = f"/fetch/symbol/{symbol}"
        result = self._make_request(endpoint)
        
        if result and 'response' in result and result['response']['docs']:
            return result['response']['docs'][0]
        
        return None
    
    def _normalize_for_matching(self, text: str) -> str:
        """
        Normalize text for matching by handling common variations.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        normalized = text.lower().strip()
        # Normalize slashes and hyphens (serine/threonine vs serine threonine)
        normalized = normalized.replace('/', ' ').replace('-', ' ')
        # Normalize multiple spaces to single space
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def is_valid_gene(self, name: str, require_exact_match: bool = True) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a given name is a valid gene/protein (not a disease, drug, etc.).
        
        Args:
            name: Name to check
            require_exact_match: If True, require that the name closely matches the gene symbol,
                               name, or aliases (not just appearing in metadata). This prevents
                               disease/drug names from matching genes they're associated with.
            
        Returns:
            Tuple of (is_valid, gene_info)
            - is_valid: True if the name represents a real protein-coding gene
            - gene_info: Dictionary with gene details if found, None otherwise
        """
        # Search for multiple results to find best match
        results = self.search_gene(name, only_protein_coding=True, only_approved=True, max_results=20)
        
        if not results:
            return False, None
        
        query_lower = name.lower().strip()
        query_normalized = self._normalize_for_matching(name)
        
        # Check each result for exact match
        for result in results:
            # Fetch full details to verify
            hgnc_id = result['hgnc_id']
            gene_details = self.fetch_gene_details(hgnc_id)
            
            if not gene_details:
                continue
            
            # Check if it's truly a gene with protein product
            locus_type = gene_details.get('locus_type', '')
            locus_group = gene_details.get('locus_group', '')
            status = gene_details.get('status', '')
            
            is_valid_type = (
                locus_group == "protein-coding gene" and
                status == "Approved" and
                locus_type == "gene with protein product"
            )
            
            if not is_valid_type:
                continue
            
            # If require_exact_match, verify the query matches gene symbol or name closely
            if require_exact_match:
                # Check exact match against symbol
                symbol = gene_details.get('symbol', '').lower()
                if query_lower == symbol:
                    return True, gene_details
                
                # Check exact match against official name (with normalization)
                official_name = gene_details.get('name', '').lower()
                official_name_normalized = self._normalize_for_matching(gene_details.get('name', ''))
                if query_lower == official_name or query_normalized == official_name_normalized:
                    return True, gene_details
                
                # Check against alias symbols
                alias_symbols = gene_details.get('alias_symbol', [])
                if isinstance(alias_symbols, list):
                    for alias in alias_symbols:
                        alias_lower = alias.lower()
                        alias_normalized = self._normalize_for_matching(alias)
                        if query_lower == alias_lower or query_normalized == alias_normalized:
                            return True, gene_details
                
                # Check against alias names
                alias_names = gene_details.get('alias_name', [])
                if isinstance(alias_names, list):
                    for alias in alias_names:
                        alias_lower = alias.lower()
                        alias_normalized = self._normalize_for_matching(alias)
                        if query_lower == alias_lower or query_normalized == alias_normalized:
                            return True, gene_details
                
                # Check against previous symbols
                prev_symbols = gene_details.get('prev_symbol', [])
                if isinstance(prev_symbols, list):
                    for prev in prev_symbols:
                        prev_lower = prev.lower()
                        prev_normalized = self._normalize_for_matching(prev)
                        if query_lower == prev_lower or query_normalized == prev_normalized:
                            return True, gene_details
            else:
                # If not requiring exact match, return first valid gene
                return True, gene_details
        
        # No exact match found in any results
        if require_exact_match and results:
            top_symbol = results[0].get('symbol', 'unknown')
            logger.info(f"'{name}' matches genes but not as an exact symbol/name match (top result: {top_symbol})")
        
        return False, None
    
    def get_equivalent_names(self, name: str) -> Optional[Dict[str, any]]:
        """
        Get all equivalent names (symbols and aliases) for a gene/protein.
        
        Args:
            name: Gene name or symbol to search for
            
        Returns:
            Dictionary containing:
            - symbol: Official gene symbol
            - name: Full gene name
            - alias_symbols: List of alternative symbols
            - alias_names: List of alternative names
            - prev_symbols: List of previous symbols
            - prev_names: List of previous names
            - hgnc_id: HGNC identifier
            - entrez_id: NCBI Entrez Gene ID
            - ensembl_gene_id: Ensembl ID
            - uniprot_ids: UniProt IDs
            Returns None if not a valid gene
        """
        is_valid, gene_info = self.is_valid_gene(name)
        
        if not is_valid or not gene_info:
            logger.warning(f"'{name}' is not a valid gene/protein name")
            return None
        
        return {
            'symbol': gene_info.get('symbol', ''),
            'name': gene_info.get('name', ''),
            'alias_symbols': gene_info.get('alias_symbol', []),
            'alias_names': gene_info.get('alias_name', []),
            'prev_symbols': gene_info.get('prev_symbol', []),
            'prev_names': gene_info.get('prev_name', []),
            'hgnc_id': gene_info.get('hgnc_id', ''),
            'entrez_id': gene_info.get('entrez_id', ''),
            'ensembl_gene_id': gene_info.get('ensembl_gene_id', ''),
            'uniprot_ids': gene_info.get('uniprot_ids', []),
            'locus_type': gene_info.get('locus_type', ''),
            'locus_group': gene_info.get('locus_group', ''),
        }


def main():
    """Example usage of HGNCClient."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    client = HGNCClient()
    
    print("=" * 80)
    print("HGNC API Client - Example Usage")
    print("=" * 80)
    
    # Test 1: Search for serine/threonine kinase 3
    print("\n1. Searching for 'serine threonine kinase 3'...")
    results = client.search_gene("serine threonine kinase 3", max_results=5)
    print(f"   Found {len(results)} results:")
    for i, gene in enumerate(results, 1):
        print(f"   {i}. {gene['symbol']} ({gene['hgnc_id']}) - score: {gene['score']:.2f}")
    
    # Test 2: Get equivalent names for STK3
    print("\n2. Getting equivalent names for 'STK3'...")
    equiv_names = client.get_equivalent_names("STK3")
    if equiv_names:
        print(f"   Official Symbol: {equiv_names['symbol']}")
        print(f"   Official Name: {equiv_names['name']}")
        print(f"   Alias Symbols: {equiv_names['alias_symbols']}")
        print(f"   HGNC ID: {equiv_names['hgnc_id']}")
        print(f"   UniProt IDs: {equiv_names['uniprot_ids']}")
    
    # Test 3: Check if "diabetes" is a valid gene (should fail)
    print("\n3. Checking if 'diabetes' is a valid gene...")
    is_valid, _ = client.is_valid_gene("diabetes")
    print(f"   Result: {'Valid gene' if is_valid else 'NOT a valid gene (likely a disease name)'}")
    
    # Test 4: Check if "aspirin" is a valid gene (should fail)
    print("\n4. Checking if 'aspirin' is a valid gene...")
    is_valid, _ = client.is_valid_gene("aspirin")
    print(f"   Result: {'Valid gene' if is_valid else 'NOT a valid gene (likely a drug name)'}")
    
    # Test 5: Check if "BRAF" is a valid gene (should pass)
    print("\n5. Checking if 'BRAF' is a valid gene...")
    is_valid, gene_info = client.is_valid_gene("BRAF")
    print(f"   Result: {'Valid gene' if is_valid else 'NOT a valid gene'}")
    if is_valid and gene_info:
        print(f"   Name: {gene_info['name']}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

