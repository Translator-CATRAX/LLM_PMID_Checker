#!/usr/bin/env python3
"""
Download split release assets from GitHub, reassemble and extract them.

Usage:
    python scripts/download_release_data.py [--output-dir OUTPUT_DIR] [--tag TAG]

Example:
    python scripts/download_release_data.py --output-dir data/release
    python scripts/download_release_data.py --output-dir . --tag v1.0
"""

import argparse
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

REPO = "RTXteam/LLM_PMID_Checker"
API_URL = f"https://api.github.com/repos/{REPO}/releases"
ARCHIVE_NAME = "LLM_Pmid_Evaluation_SemMedDB_with_names_v1.0.tar.gz"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB read chunks


def get_release_assets(tag: str) -> list[dict]:
    """Fetch the asset list for a given release tag."""
    url = f"{API_URL}/tags/{tag}"
    req = Request(url, headers={"Accept": "application/vnd.github+json"})
    try:
        with urlopen(req) as resp:
            import json
            data = json.loads(resp.read())
    except HTTPError as e:
        sys.exit(f"Failed to fetch release '{tag}': HTTP {e.code} — {e.reason}")
    except URLError as e:
        sys.exit(f"Network error fetching release '{tag}': {e.reason}")

    assets = data.get("assets", [])
    if not assets:
        sys.exit(f"No assets found for release '{tag}'.")
    return assets


def download_file(url: str, dest: Path, name: str, size: int | None = None) -> None:
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
                print(f"\r  Downloading {name}: {downloaded / 1e9:.2f} GB / {size / 1e9:.2f} GB ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\r  Downloading {name}: {downloaded / 1e9:.2f} GB", end="", flush=True)
    print()


def reassemble(parts: list[Path], output: Path) -> None:
    """Concatenate split parts into the original archive."""
    print(f"Reassembling {len(parts)} parts into {output.name} ...")
    with open(output, "wb") as out:
        for part in parts:
            print(f"  Appending {part.name} ({part.stat().st_size / 1e9:.2f} GB)")
            with open(part, "rb") as inp:
                while True:
                    chunk = inp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
    print(f"  Done — {output.name} ({output.stat().st_size / 1e9:.2f} GB)")


def extract(archive: Path, output_dir: Path) -> None:
    """Extract the tar.gz archive."""
    print(f"Extracting {archive.name} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=output_dir)
    print(f"  Extracted to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Download and reassemble LLM PMID evaluation data from GitHub release."
    )
    parser.add_argument(
        "--output-dir", "-o", default=".",
        help="Directory to save downloaded files (default: current directory)",
    )
    parser.add_argument(
        "--tag", "-t", default="v1.0",
        help="Release tag to download (default: v1.0)",
    )
    parser.add_argument(
        "--no-extract", action="store_true",
        help="Skip extraction after reassembly",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching release '{args.tag}' from {REPO} ...")
    assets = get_release_assets(args.tag)

    part_assets = sorted(
        [a for a in assets if a["name"].startswith(ARCHIVE_NAME + ".part_")],
        key=lambda a: a["name"],
    )

    if not part_assets:
        sys.exit(f"No split parts found matching '{ARCHIVE_NAME}.part_*' in release '{args.tag}'.")

    print(f"Found {len(part_assets)} parts to download:")
    for a in part_assets:
        print(f"  {a['name']}  ({a['size'] / 1e9:.2f} GB)")

    part_paths = []
    for asset in part_assets:
        dest = output_dir / asset["name"]
        if dest.exists() and dest.stat().st_size == asset["size"]:
            print(f"  {asset['name']} already downloaded, skipping.")
        else:
            download_file(
                asset["browser_download_url"], dest, asset["name"], asset["size"]
            )
        part_paths.append(dest)

    archive_path = output_dir / ARCHIVE_NAME
    reassemble(part_paths, archive_path)

    if not args.no_extract:
        extract(archive_path, output_dir)
        extracted = [f for f in output_dir.iterdir() if f.suffix == ".parquet"]
        print(f"\nExtracted files:")
        for f in sorted(extracted):
            print(f"  {f.name}  ({f.stat().st_size / 1e9:.2f} GB)")

    print("Cleaning up split parts ...")
    for p in part_paths:
        p.unlink()
    print("  Done.")

    print(f"\nAll done! Files are in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
