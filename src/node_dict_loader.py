"""Load entity info from KG2 or KGX nodes JSONL for enriching prompts.

Streams through the file once, extracting only the nodes whose id or
equivalent CURIEs match the requested set.

Supports both KG2 (``equivalent_curies``, ``all_names``) and KGX
(``equivalent_identifiers``) node schemas.  For KGX, call
:meth:`merge_all_names_tsv` after loading to supplement ``all_names``
from a ``curie_all_names.tsv`` produced by
``scripts/extract_curie_names.py``.
"""
import csv
import gzip
import json
import logging
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class NodeDictLoader:
    """Loads node info (name, category, description) from a nodes file.

    Supports:
      - .jsonl / .jsonl.gz  (KG2 or KGX node format, streamed with target-CURIE filtering)
      - .json / .json.gz    (pre-built {curie: {name, category, description}} dict)
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
            path: Path to a .jsonl/.jsonl.gz (KG2/KGX nodes) or
                  .json/.json.gz (pre-built dict)
            target_curies: If provided, only load nodes matching these CURIEs.
                           Strongly recommended for large node files.

        Returns:
            NodeDictLoader instance with loaded data
        """
        loader = cls()

        if path.endswith('.jsonl.gz'):
            loader._load_jsonl(path, target_curies, compressed=True)
        elif path.endswith('.jsonl'):
            loader._load_jsonl(path, target_curies, compressed=False)
        elif path.endswith('.json.gz'):
            loader._load_json_gz(path, target_curies)
        elif path.endswith('.json'):
            loader._load_json(path, target_curies)
        else:
            raise ValueError(
                f"Unsupported file format: {path}. "
                f"Expected .jsonl, .jsonl.gz, .json.gz, or .json"
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

    def merge_all_names_tsv(self, tsv_path: str) -> None:
        """Merge ``all_names`` from a TSV produced by ``extract_curie_names.py``.

        Expects columns ``curie_id`` and ``all_names`` (pipe-delimited).
        Only updates entries already present in ``_dict``; names loaded here
        **replace** the fallback ``[name]`` list.
        """
        updated = 0
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                curie = row['curie_id']
                if curie not in self._dict:
                    continue
                raw = row.get('all_names', '')
                if raw:
                    names = [n for n in raw.split('|') if n]
                    if names:
                        self._dict[curie]['all_names'] = names
                        updated += 1
        logger.info(
            f"Merged all_names for {updated:,} CURIEs from {tsv_path}"
        )

    def __len__(self) -> int:
        return len(self._dict)

    def __contains__(self, curie: str) -> bool:
        return curie in self._dict

    @staticmethod
    def _get_equiv_curies(node: dict) -> list:
        """Return equivalent CURIEs regardless of schema flavour."""
        return (
            node.get('equivalent_curies')
            or node.get('equivalent_identifiers')
            or []
        )

    @staticmethod
    def _normalize_category(raw) -> str:
        """Accept both a plain string and a list of strings."""
        if isinstance(raw, list):
            return raw[0] if raw else ''
        return raw or ''

    def _load_jsonl(
        self,
        path: str,
        target_curies: Optional[Set[str]],
        *,
        compressed: bool = False,
    ):
        """Stream through a JSONL node file (KG2 or KGX), keeping matches."""
        if not target_curies:
            logger.warning(
                "Loading JSONL without target_curies -- "
                "this will scan all nodes but only store unique ids. "
                "Consider providing target_curies for efficiency."
            )

        target = set(target_curies) if target_curies else None
        found_count = 0
        scanned = 0

        opener = (
            gzip.open(path, 'rt', encoding='utf-8')
            if compressed
            else open(path, 'r', encoding='utf-8')
        )

        with opener as f:
            for line in f:
                scanned += 1
                if scanned % 1_000_000 == 0:
                    logger.info(
                        f"  Scanned {scanned:,} nodes, "
                        f"found {found_count} matches..."
                    )

                node = json.loads(line)
                node_id = node.get('id', '')
                equiv = self._get_equiv_curies(node)

                if target is not None:
                    matching_curies = target.intersection(equiv)
                    if node_id in target:
                        matching_curies.add(node_id)
                    if not matching_curies:
                        continue
                else:
                    matching_curies = {node_id}

                name = node.get('name', '')
                info = {
                    'name': name,
                    'category': self._normalize_category(node.get('category', '')),
                    'description': node.get('description', ''),
                    'all_names': node.get('all_names') or ([name] if name else []),
                }

                for curie in matching_curies:
                    self._dict[curie] = info
                found_count += 1

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
