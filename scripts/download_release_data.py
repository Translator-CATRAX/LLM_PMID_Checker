#!/usr/bin/env python3
"""
Download release assets from GitHub, reassemble (if split) and extract them.

Usage:
    python scripts/download_release_data.py [--output-dir OUTPUT_DIR] [--tag TAG]

Example:
    python scripts/download_release_data.py --output-dir results --tag tmkp-v1.0
    python scripts/download_release_data.py --output-dir results --tag semmeddb-v1.0
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

REPO = "RTXteam/LLM_PMID_Checker"
API_URL = f"https://api.github.com/repos/{REPO}/releases"

RELEASE_CONFIG = {
    "tmkp-v1.0": {
        "archive_name": "TMKP_Sentences_Evaluation_v1.0.tar.gz",
        "description": "TMKP KGX sentence-level evaluation (gpt-oss-120b)",
    },
    "semmeddb-v1.0": {
        "archive_name": "LLM_Pmid_Evaluation_SemMedDB_with_names_v1.0.tar.gz",
        "description": "SemMedDB KGX PMID-level evaluation (gpt-oss-120b)",
    },
}

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB read chunks


def get_release_assets(tag: str) -> list:
    """Fetch the asset list for a given release tag."""
    url = f"{API_URL}/tags/{tag}"
    req = Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urlopen(req) as resp:
            import json
            data = json.loads(resp.read())
    except HTTPError as e:
        sys.exit(f"Failed to fetch release '{tag}': HTTP {e.code} - {e.reason}")
    except URLError as e:
        sys.exit(f"Network error fetching release '{tag}': {e.reason}")

    assets = data.get("assets", [])
    if not assets:
        sys.exit(f"No assets found for release '{tag}'.")
    return assets


def download_file(url: str, dest: Path, name: str, size=None) -> None:
    """Download a single file with progress reporting."""
    req = Request(url, headers={"Accept": "application/octet-stream"})
    downloaded = 0
    with urlopen(req) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if size:
                pct = downloaded / size * 100
                print(
                    f"\r  Downloading {name}: "
                    f"{downloaded / 1e6:.1f} MB / {size / 1e6:.1f} MB "
                    f"({pct:.1f}%)",
                    end="", flush=True,
                )
            else:
                print(
                    f"\r  Downloading {name}: {downloaded / 1e6:.1f} MB",
                    end="", flush=True,
                )
    print()


def reassemble(parts: list, output: Path) -> None:
    """Concatenate split parts into the original archive."""
    print(f"Reassembling {len(parts)} parts into {output.name} ...")
    with open(output, "wb") as out:
        for part in parts:
            print(f"  Appending {part.name} ({part.stat().st_size / 1e6:.1f} MB)")
            with open(part, "rb") as inp:
                while True:
                    chunk = inp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
    print(f"  Done - {output.name} ({output.stat().st_size / 1e6:.1f} MB)")


def extract(archive: Path, output_dir: Path) -> None:
    """Extract the tar.gz archive."""
    print(f"Extracting {archive.name} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=output_dir)
    print(f"  Extracted to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Download LLM evaluation data from GitHub release."
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Directory to save downloaded files (default: current directory)",
    )
    parser.add_argument(
        "--tag", "-t",
        default="tmkp-v1.0",
        help="Release tag to download (default: tmkp-v1.0). Options: tmkp-v1.0, semmeddb-v1.0",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip extraction after download/reassembly",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RELEASE_CONFIG.get(args.tag)
    if config:
        archive_name = config["archive_name"]
        print(f"Release: {config['description']}")
    else:
        archive_name = None
        print(f"Unknown tag '{args.tag}', will attempt auto-detection.")

    print(f"Fetching release '{args.tag}' from {REPO} ...")
    assets = get_release_assets(args.tag)

    if archive_name:
        part_assets = sorted(
            [a for a in assets if a["name"].startswith(archive_name + ".part_")],
            key=lambda a: a["name"],
        )
        single_asset = next(
            (a for a in assets if a["name"] == archive_name),
            None,
        )
    else:
        part_assets = sorted(
            [a for a in assets if ".part_" in a["name"]],
            key=lambda a: a["name"],
        )
        single_asset = next(
            (a for a in assets if a["name"].endswith(".tar.gz") and ".part_" not in a["name"]),
            None,
        )

    if part_assets:
        print(f"Found {len(part_assets)} split parts to download:")
        for a in part_assets:
            print(f"  {a['name']}  ({a['size'] / 1e6:.1f} MB)")

        part_paths = []
        for asset in part_assets:
            dest = output_dir / asset["name"]
            if dest.exists() and dest.stat().st_size == asset["size"]:
                print(f"  {asset['name']} already downloaded, skipping.")
            else:
                download_file(
                    asset["browser_download_url"],
                    dest,
                    asset["name"],
                    asset["size"],
                )
            part_paths.append(dest)

        target_name = archive_name or part_assets[0]["name"].rsplit(".part_", 1)[0]
        archive_path = output_dir / target_name
        reassemble(part_paths, archive_path)

        if not args.no_extract:
            extract(archive_path, output_dir)

        print("Cleaning up split parts ...")
        for p in part_paths:
            p.unlink()
        print("  Done.")

    elif single_asset:
        print(f"Found single archive: {single_asset['name']} ({single_asset['size'] / 1e6:.1f} MB)")
        archive_path = output_dir / single_asset["name"]

        if archive_path.exists() and archive_path.stat().st_size == single_asset["size"]:
            print(f"  {single_asset['name']} already downloaded, skipping.")
        else:
            download_file(
                single_asset["browser_download_url"],
                archive_path,
                single_asset["name"],
                single_asset["size"],
            )

        if not args.no_extract:
            extract(archive_path, output_dir)
    else:
        sys.exit(f"No downloadable archive found in release '{args.tag}'.")

    extracted = [f for f in output_dir.iterdir() if f.suffix == ".parquet"]
    if extracted:
        print(f"\nExtracted Parquet files:")
        for f in sorted(extracted):
            print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")

    print(f"\nAll done! Files are in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
