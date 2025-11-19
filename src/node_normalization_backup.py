import requests
import json
import re
from typing import List, Dict, Optional
from src.umls_client import UMLSClient
from src.hgnc_client import HGNCClient

class NodeNormalizationClient:
    """Client for interacting with the Node Normalization API, ARAX TRAPI API, and HGNC API."""

    def __init__(
        self,
        nn_base_url: str = "https://nodenormalization-sri.renci.org",
        arax_base_url: str = "https://arax.transltr.io/api/arax/v1.4",
        use_umls: bool = True,
        use_hgnc: bool = True,
        timeout: int = 10
    ):
        self.nn_base_url = nn_base_url.rstrip('/')
        self.arax_base_url = arax_base_url.rstrip('/')
        self.nn_session = requests.Session()
        self.arax_session = requests.Session()
        self.use_umls = use_umls
        self.use_hgnc = use_hgnc
        self.timeout = timeout
        
        # Initialize UMLS client if enabled
        self.umls_client = None
        if use_umls:
            try:
                self.umls_client = UMLSClient(timeout=timeout)
            except Exception as e:
                print(f"Warning: Could not initialize UMLS client: {e}", flush=True)
                self.umls_client = None
        
        # Initialize HGNC client if enabled
        self.hgnc_client = None
        if use_hgnc:
            try:
                self.hgnc_client = HGNCClient(timeout=timeout)
            except Exception as e:
                print(f"Warning: Could not initialize HGNC client: {e}", flush=True)
                self.hgnc_client = None
    
    def _get_normalized_curie(self, name: str) -> Optional[str]:
        """
        Get normalized CURIE from the ARAX TRAPI API.
        
        Args:
            name: Name to normalize
            
        Returns:
            Normalized CURIE or None on error
        """
        endpoint = f"{self.arax_base_url}/entity"
        
        # Sanitize name by removing commas (ARAX API doesn't handle them well)
        sanitized_name = self._preprocess_name(name)

        try:
            response = self.arax_session.get(
                endpoint,
                params={'q': [sanitized_name]},
                headers={'Accept': 'application/json'},
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            print("Request timed out", flush=True)
            return None
        except requests.exceptions.ConnectionError:
            print("Connection error occurred", flush=True)
            return None
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}", flush=True)
            print(f"Response content: {response.text}", flush=True)
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error occurred: {e}", flush=True)
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}", flush=True)
            return None
    
    def get_normalized_node_info(
        self, 
        curie: str = None,
        name: str = None,
        conflate: bool = True,
        drug_chemical_conflate: bool = False,
        description: bool = True
    ) -> Optional[Dict]:
        """
        Get normalized nodes from the API.
        
        Args:
            curie: CURIE to normalize
            name: Name to normalize
            conflate: Whether to conflate nodes
            drug_chemical_conflate: Whether to conflate drugs and chemicals
            description: Whether to include descriptions
            
        Returns:
            Dictionary containing normalized node information or None on error
        """
        
        if not curie and not name:
            print("No CURIE or name provided")
            return None
        
        if name:
            # Preprocess name for API call
            cleaned_name = self._preprocess_name(name)
            response_data = self._get_normalized_curie(cleaned_name)
            
            if response_data and cleaned_name in response_data:
                res = response_data[cleaned_name]
                if res:
                    curie = res.get('id')['identifier']
                else:
                    print("No CURIE found", flush=True)
                    return None
            else:
                print(f"No CURIE found for '{name}' (cleaned: '{cleaned_name}')", flush=True)
                return None
        
        endpoint = f"{self.nn_base_url}/get_normalized_nodes"
        
        params = {
            'curie': curie,
            'conflate': str(conflate).lower(),
            'drug_chemical_conflate': str(drug_chemical_conflate).lower(),
            'description': str(description).lower()
        }
        
        try:
            response = self.nn_session.get(
                endpoint,
                params=params,
                headers={'Accept': 'application/json'},
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()[curie]
            
        except requests.exceptions.Timeout:
            print("Request timed out", flush=True)
            return None
        except requests.exceptions.ConnectionError:
            print("Connection error occurred", flush=True)
            return None
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}", flush=True)
            print(f"Response content: {response.text}", flush=True)
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error occurred: {e}", flush=True)
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}", flush=True)
            return None

    def _preprocess_name(self, name: str) -> str:
        """
        Preprocess a name by removing problematic characters.
        
        Args:
            name: Original name
            
        Returns:
            Cleaned name with commas, square brackets, and slashes removed
        """
        if not name:
            return name
        
        cleaned = re.sub(r'\s*\[.*?\]', '', name)
        # Remove commas and slashes, normalize whitespace
        cleaned = cleaned.replace(', ', ' ').replace('/ ', ' ')
        # Normalize multiple spaces to single space
        cleaned = ' '.join(cleaned.split())
        return cleaned
    
    def get_equivalent_names(self, curie: str = None, name: str = None, **kwargs) -> Optional[List[str]]:
        """
        Get equivalent names for a single CURIE or name.
        
        When name is provided: 
        1. Checks if it's a gene using HGNC first
        2. If gene: gets HGNC equivalent names (symbol, aliases, etc.)
        3. Then searches UMLS and node normalization
        4. Combines all results with HGNC names prioritized
        
        When CURIE is provided: gets node normalization results first, then searches UMLS and HGNC.
        
        Args:
            curie: CURIE to normalize
            name: Name to search
            **kwargs: Additional arguments for node normalization
            
        Returns:
            List of equivalent names, or None if no results found
        """
        nn_names = set()  # Names from node normalization
        umls_names = []  # Names from UMLS (ordered)
        hgnc_names = []  # Names from HGNC (ordered: symbol, name, aliases)
        is_gene = False
        
        # Preprocess name if provided
        original_name = name
        if name:
            cleaned_name = self._preprocess_name(name)
            if cleaned_name != name:
                print(f"Preprocessed name: '{name}' → '{cleaned_name}'", flush=True)
        else:
            cleaned_name = name
        
        # If name is provided (name-based query)
        if name:
            # First, check if it's a gene using HGNC (try both original and cleaned)
            if self.use_hgnc and self.hgnc_client:
                try:
                    # Try cleaned name first
                    is_valid_gene, gene_info = self.hgnc_client.is_valid_gene(cleaned_name)
                    # If not found and names differ, try original
                    if not is_valid_gene and cleaned_name != original_name:
                        is_valid_gene, gene_info = self.hgnc_client.is_valid_gene(original_name)
                    
                    if is_valid_gene and gene_info:
                        is_gene = True
                        # Get HGNC equivalent names in priority order
                        symbol = gene_info.get('symbol', '')
                        official_name = gene_info.get('name', '')
                        alias_symbols = gene_info.get('alias_symbol', [])
                        alias_names = gene_info.get('alias_name', [])
                        prev_symbols = gene_info.get('prev_symbol', [])
                        
                        # Add in order: symbol, official name, alias symbols, alias names, prev symbols
                        if symbol:
                            hgnc_names.append(symbol.lower())
                        if official_name:
                            hgnc_names.append(official_name.lower())
                        if isinstance(alias_symbols, list):
                            hgnc_names.extend([a.lower() for a in alias_symbols if a])
                        if isinstance(alias_names, list):
                            hgnc_names.extend([a.lower() for a in alias_names if a])
                        if isinstance(prev_symbols, list):
                            hgnc_names.extend([p.lower() for p in prev_symbols if p])
                        
                        print(f"HGNC: '{cleaned_name}' is a gene ({symbol}), found {len(hgnc_names)} equivalent names", flush=True)
                except Exception as e:
                    print(f"Error checking HGNC for '{cleaned_name}': {e}", flush=True)
            
            # Then, get UMLS results (top 5) - try cleaned name first
            if self.use_umls and self.umls_client:
                try:
                    umls_names = self.umls_client.get_term_names(cleaned_name, max_results=5)
                    # If no results and names differ, try original
                    if not umls_names and cleaned_name != original_name:
                        umls_names = self.umls_client.get_term_names(original_name, max_results=5)
                    print(f"UMLS found {len(umls_names)} names for '{cleaned_name}'", flush=True)
                except Exception as e:
                    print(f"Error searching UMLS for '{cleaned_name}': {e}", flush=True)
            
            # Then get node normalization results - try cleaned name first
            result = self.get_normalized_node_info(curie=None, name=cleaned_name, **kwargs)
            # If no results and names differ, try original
            if not result and cleaned_name != original_name:
                result = self.get_normalized_node_info(curie=None, name=original_name, **kwargs)
            if result:
                eq_ids = result.get('equivalent_identifiers', [])
                nn_names = {x['label'].lower() for x in eq_ids if x.get('label')}
        
        # If CURIE is provided (CURIE-based query)
        elif curie:
            # First get node normalization results
            result = self.get_normalized_node_info(curie=curie, name=None, **kwargs)
            if result:
                eq_ids = result.get('equivalent_identifiers', [])
                nn_names = {x['label'].lower() for x in eq_ids if x.get('label')}
                
                # Get the primary name for further searches
                primary_name = list(nn_names)[0] if nn_names else None
                
                # Then search UMLS using the primary name
                if self.use_umls and self.umls_client and primary_name:
                    try:
                        umls_names = self.umls_client.get_term_names(primary_name, max_results=5)
                        print(f"UMLS found {len(umls_names)} names for '{primary_name}'", flush=True)
                    except Exception as e:
                        print(f"Error searching UMLS for '{primary_name}': {e}", flush=True)
                
                # Also check if it's a gene in HGNC
                if self.use_hgnc and self.hgnc_client and primary_name:
                    try:
                        is_valid_gene, gene_info = self.hgnc_client.is_valid_gene(primary_name)
                        if is_valid_gene and gene_info:
                            is_gene = True
                            symbol = gene_info.get('symbol', '')
                            official_name = gene_info.get('name', '')
                            alias_symbols = gene_info.get('alias_symbol', [])
                            alias_names = gene_info.get('alias_name', [])
                            prev_symbols = gene_info.get('prev_symbol', [])
                            
                            if symbol:
                                hgnc_names.append(symbol.lower())
                            if official_name:
                                hgnc_names.append(official_name.lower())
                            if isinstance(alias_symbols, list):
                                hgnc_names.extend([a.lower() for a in alias_symbols if a])
                            if isinstance(alias_names, list):
                                hgnc_names.extend([a.lower() for a in alias_names if a])
                            if isinstance(prev_symbols, list):
                                hgnc_names.extend([p.lower() for p in prev_symbols if p])
                            
                            print(f"HGNC: '{primary_name}' is a gene ({symbol}), found {len(hgnc_names)} equivalent names", flush=True)
                    except Exception as e:
                        print(f"Error checking HGNC for '{primary_name}': {e}", flush=True)
        else:
            print("No CURIE or name provided", flush=True)
            return None
        
        # Combine results with priority: HGNC names (if gene) > node normalization > UMLS
        final_names = []
        seen = set()
        
        # Add HGNC names first (if this is a gene)
        if is_gene and hgnc_names:
            for hgnc_name in hgnc_names:
                if hgnc_name not in seen:
                    final_names.append(hgnc_name)
                    seen.add(hgnc_name)

        # Add node normalization names
        for nn_name in nn_names:
            if nn_name not in seen:
                final_names.append(nn_name)
                seen.add(nn_name)

        # Add UMLS names (they're already ordered by relevance)
        for umls_name in umls_names:
            if umls_name not in seen:
                final_names.append(umls_name)
                seen.add(umls_name)
        
        # If no results from any source, return None or the original input
        if not final_names:
            print("No equivalent names found from HGNC, UMLS, or node normalization", flush=True)
            # Return the original input as fallback
            if name:
                return [name.lower()]
            elif curie:
                return [curie]
            return None
        
        # Extract base names from gene/protein names (e.g., 'smad7' from 'smad7 gene')
        # This helps LLM match 'Smad7' in abstract to 'smad7 gene' in equivalent names
        base_names_added = []
        for name in list(final_names):
            lower_name = name.lower()
            if any(suffix in lower_name for suffix in [' gene', ' protein']):
                # Extract first word as base name
                base = lower_name.split()[0]
                if len(base) > 2 and base not in seen:  # Avoid single letters
                    final_names.append(base)
                    seen.add(base)
                    base_names_added.append(base)
        
        # Add source information in output message
        sources = []
        if is_gene and hgnc_names:
            sources.append(f"HGNC ({len(hgnc_names)})")
        if nn_names:
            sources.append(f"NodeNorm ({len(nn_names)})")
        if umls_names:
            sources.append(f"UMLS ({len(umls_names)})")
        if base_names_added:
            sources.append(f"BaseNames ({len(base_names_added)})")
        print(f"Combined {len(final_names)} unique equivalent names from: {', '.join(sources)}", flush=True)
        
        return final_names