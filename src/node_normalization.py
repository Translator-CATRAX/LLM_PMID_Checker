import re
from typing import List, Optional
from src.arax_nn.node_synonymizer import NodeSynonymizer

class NodeNormalizationClient:
    """Client for interacting with the ARAX local node normalization service."""

    def __init__(self):
        self.arax_synonymizer = NodeSynonymizer()
    
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
        
        Simplified version that only returns the preprocessed name without external API calls.
        
        Args:
            curie: CURIE to normalize
            name: Name to search
            **kwargs: Additional arguments for node normalization
            
        Returns:
            List containing the preprocessed name(s)
        """
        if not name and not curie:
            print("No CURIE or name provided", flush=True)
            return None
        
        # If name is provided, preprocess and return it
        if name:
            sanitized_name = self._preprocess_name(name)
            response = self.arax_synonymizer.get_normalizer_results(entities=sanitized_name)[sanitized_name]
            if response:
                return list(set([x['label'].lower() for x in response.get('nodes') if x.get('label')]))
            else:
                return None
        
        # If CURIE is provided, just return it
        elif curie:
            response = self.arax_synonymizer.get_normalizer_results(entities=curie)[curie]
            if response:
                try:
                    return list(set([x['label'].lower() for x in response.get('nodes') if x.get('label')]))
                except Exception as e:
                    print(f"{response}", flush=True)
                    return None
            else:
                return None