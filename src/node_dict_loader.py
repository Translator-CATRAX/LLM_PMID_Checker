"""Load entity info from KG2 nodes JSONL.gz for enriching prompts.

Streams through the compressed file once, extracting only the nodes
whose id or equivalent_curies match the requested CURIEs.
"""
import gzip
import json
import logging
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class NodeDictLoader:
    """Loads node info (name, category, description) from a KG2 nodes file.

    Supports:
      - .jsonl.gz  (KG2 raw format, streamed with target-CURIE filtering)
      - .json / .json.gz  (pre-built {curie: {name, category, description}} dict)
    """

    def __init__(self):
        self._dict: Dict[str, dict] = {}

    @classmethod
    def from_file(
        cls,
        path: str,
        target_curies: Optional[Set[str]] = None,
    ) -> "NodeDictLoader":
        """Load node info from a file.

        Args:
            path: Path to a .jsonl.gz (KG2 nodes) or .json/.json.gz (pre-built dict)
            target_curies: If provided, only load nodes matching these CURIEs.
                           Strongly recommended for .jsonl.gz files (6M+ nodes).

        Returns:
            NodeDictLoader instance with loaded data
        """
        loader = cls()

        if path.endswith('.jsonl.gz'):
            loader._load_jsonl_gz(path, target_curies)
        elif path.endswith('.json.gz'):
            loader._load_json_gz(path, target_curies)
        elif path.endswith('.json'):
            loader._load_json(path, target_curies)
        else:
            raise ValueError(
                f"Unsupported file format: {path}. "
                f"Expected .jsonl.gz, .json.gz, or .json"
            )

        logger.info(f"Loaded {len(loader._dict)} node entries from {path}")
        return loader

    def get_node_info(self, curie: str) -> Optional[dict]:
        """Look up node info by CURIE.

        Returns:
            Dict with keys: name, category, description (or None if not found)
        """
        return self._dict.get(curie)

    def get_all_names(self, curie: str) -> Optional[list]:
        """Get all_names for a CURIE (useful for supplementing equivalent names)."""
        info = self._dict.get(curie)
        if info:
            return info.get('all_names')
        return None

    def __len__(self) -> int:
        return len(self._dict)

    def __contains__(self, curie: str) -> bool:
        return curie in self._dict

    def _load_jsonl_gz(
        self,
        path: str,
        target_curies: Optional[Set[str]],
    ):
        """Stream through KG2 nodes JSONL.gz, keeping only matching entries."""
        if not target_curies:
            logger.warning(
                "Loading JSONL.gz without target_curies -- "
                "this will scan all nodes but only store unique ids. "
                "Consider providing target_curies for efficiency."
            )

        target = set(target_curies) if target_curies else None
        found_count = 0
        scanned = 0

        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                scanned += 1
                if scanned % 1_000_000 == 0:
                    logger.info(
                        f"  Scanned {scanned:,} nodes, "
                        f"found {found_count} matches..."
                    )

                node = json.loads(line)
                node_id = node.get('id', '')
                equiv = node.get('equivalent_curies', [])

                # Determine if this node matches any target CURIE
                if target is not None:
                    matching_curies = target.intersection(equiv)
                    matching_curies.update(
                        {node_id} if node_id in target else set()
                    )
                    if not matching_curies:
                        continue
                else:
                    matching_curies = {node_id}

                info = {
                    'name': node.get('name', ''),
                    'category': node.get('category', ''),
                    'description': node.get('description', ''),
                    'all_names': node.get('all_names', []),
                }

                # Map every matching CURIE to this node's info
                for curie in matching_curies:
                    self._dict[curie] = info
                found_count += 1

                # Early exit if all targets found
                if target is not None and target.issubset(self._dict.keys()):
                    logger.info(
                        f"All {len(target)} target CURIEs found "
                        f"after scanning {scanned:,} nodes"
                    )
                    break

        logger.info(
            f"Scanned {scanned:,} nodes total, "
            f"loaded {len(self._dict)} CURIE mappings"
        )

    def _load_json(self, path: str, target_curies: Optional[Set[str]]):
        """Load from a pre-built JSON dict file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self._apply_dict(data, target_curies)

    def _load_json_gz(self, path: str, target_curies: Optional[Set[str]]):
        """Load from a gzipped pre-built JSON dict file."""
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        self._apply_dict(data, target_curies)

    def _apply_dict(self, data: dict, target_curies: Optional[Set[str]]):
        """Apply a pre-built dict, optionally filtering to target CURIEs."""
        if target_curies:
            for curie in target_curies:
                if curie in data:
                    self._dict[curie] = data[curie]
        else:
            self._dict = data
