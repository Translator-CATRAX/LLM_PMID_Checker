#!/usr/bin/env python3
"""Extract predicates and descriptions from biolink-model.yaml into a TSV file."""

import argparse
import csv
from pathlib import Path

import yaml


def find_predicate_descendants(slots: dict, root: str = "related to") -> set[str]:
    """Find all slots that descend from `root` via `is_a` chains."""
    descendants = {root}
    changed = True
    while changed:
        changed = False
        for name, props in slots.items():
            if name in descendants:
                continue
            if not isinstance(props, dict):
                continue
            is_a = props.get("is_a")
            if is_a and is_a in descendants:
                descendants.add(name)
                changed = True
    return descendants


def format_predicate_name(name: str) -> str:
    return "biolink:" + name.replace(" ", "_")


def collapse_description(desc: str) -> str:
    """Collapse multi-line YAML description into a single line."""
    if not desc:
        return ""
    return " ".join(desc.split())


def main():
    parser = argparse.ArgumentParser(
        description="Extract predicates and descriptions from biolink-model.yaml into a TSV file.",
    )
    parser.add_argument("--input", required=True,
                        help="Path to biolink-model.yaml")
    parser.add_argument("--output", required=True,
                        help="Output TSV path (columns: predicate, description)")
    args = parser.parse_args()

    yaml_path = Path(args.input)
    output_path = Path(args.output)

    with open(yaml_path) as f:
        model = yaml.safe_load(f)

    slots = model.get("slots", {})
    predicates = find_predicate_descendants(slots)

    rows = []
    for name in sorted(predicates):
        props = slots[name]
        desc = props.get("description", "") if isinstance(props, dict) else ""
        rows.append((format_predicate_name(name), collapse_description(desc)))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["predicate", "description"])
        writer.writerows(rows)

    print(f"Extracted {len(rows)} predicates to {output_path}")


if __name__ == "__main__":
    main()
