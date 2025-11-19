"""UMLS API client for searching medical terms."""
import requests
import os
from typing import List, Dict


class UMLSClient:
    """Client for interacting with the UMLS Terminology Services (UTS) API."""

    def __init__(self, api_key: str = None, version: str = "current", timeout: int = 10):
        """
        Initialize UMLS client.
        
        Args:
            api_key: UMLS API key. If not provided, will load from UMLS_API_KEY env variable
            version: UMLS version (default: "current")
            timeout: Request timeout in seconds (default: 10)
        """
        self.api_key = api_key or os.getenv("UMLS_API_KEY")
        if not self.api_key:
            raise ValueError("UMLS API key not provided and UMLS_API_KEY not found in environment")
        
        self.version = version
        self.timeout = timeout
        self.base_url = "https://uts-ws.nlm.nih.gov"
        self.search_endpoint = f"{self.base_url}/rest/search/{self.version}"
        self.session = requests.Session()

    def search_term(
        self, 
        search_string: str, 
        max_results: int = 5,
        page_size: int = 25
    ) -> List[Dict[str, str]]:
        """
        Search UMLS for a term and return top results.
        
        Args:
            search_string: The search term to look up
            max_results: Maximum number of results to return (default: 5)
            page_size: Number of results per page (default: 25)
            
        Returns:
            List of dictionaries containing CUI information with keys:
            - ui: Unique identifier (CUI)
            - uri: Resource URI
            - name: Concept name
            - rootSource: Source vocabulary
        """
        results = []
        page = 0
        
        try:
            while len(results) < max_results:
                page += 1
                query_params = {
                    'string': search_string,
                    'apiKey': self.api_key,
                    'pageNumber': page,
                    'pageSize': page_size
                }
                
                response = self.session.get(
                    self.search_endpoint,
                    params=query_params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                items = data.get('result', {}).get('results', [])
                
                if not items:
                    break
                
                for item in items:
                    if len(results) >= max_results:
                        break
                    
                    results.append({
                        'ui': item.get('ui', ''),
                        'uri': item.get('uri', ''),
                        'name': item.get('name', ''),
                        'rootSource': item.get('rootSource', '')
                    })
                
                # If we got fewer results than page_size, we've reached the end
                if len(items) < page_size:
                    break
                    
        except requests.exceptions.Timeout:
            print(f"UMLS API request timed out for search term: {search_string}", flush=True)
        except requests.exceptions.HTTPError as e:
            print(f"UMLS API HTTP error: {e}", flush=True)
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}", flush=True)
        except requests.exceptions.RequestException as e:
            print(f"UMLS API request error: {e}", flush=True)
        except Exception as e:
            print(f"Unexpected error during UMLS search: {e}", flush=True)
        
        return results

    def get_term_names(
        self, 
        search_string: str, 
        max_results: int = 5
    ) -> List[str]:
        """
        Search UMLS and return just the concept names.
        
        Args:
            search_string: The search term to look up
            max_results: Maximum number of names to return (default: 5)
            
        Returns:
            List of concept names (strings), lowercased and deduplicated
        """
        results = self.search_term(search_string, max_results)
        
        # Extract names, lowercase them, and deduplicate while preserving order
        names = []
        seen = set()
        
        for item in results:
            name = item.get('name', '').lower().strip()
            if name and name not in seen:
                names.append(name)
                seen.add(name)
        
        return names

