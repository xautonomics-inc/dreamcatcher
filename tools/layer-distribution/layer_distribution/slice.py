"""Command-line interface for GGUF layer slicing.

Usage:
    python3 -m layer_distribution.slice <gguf-or-glob> <out_dir> [--rate RATE] [--force] [--resume]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .slicer import slice_model


def main(argv: Sequence[str] | None = None) -> int:
    """Run layer slicer CLI."""
    parser = argparse.ArgumentParser(
        prog="layer_distribution.slice",
        description="Slice a GGUF model into a per-layer library with whole-file hash scope.",
    )
    parser.add_argument(
        "model_glob",
        help="Source GGUF file or glob pattern (multi-file sources supported)",
    )
    parser.add_argument(
        "out_dir",
        help="Output directory (the layer library destination)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help="I/O rate limit in MiB/s for writing and hashing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing non-empty library directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted slice, reusing valid completed files",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip content hashes in manifest",
    )
    parser.add_argument(
        "--source-label",
        type=str,
        default=None,
        help="Optional sanitized label for source glob in manifest provenance",
    )

    args = parser.parse_args(argv)

    try:
        manifest = slice_model(
            args.model_glob,
            args.out_dir,
            rate=args.rate,
            force=args.force,
            resume=args.resume,
            no_hash=args.no_hash,
            source_label=args.source_label,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as err:
        sys.stderr.write(f"error: {err}\n")
        return 1

    files = manifest.get("files", [])
    total_bytes = sum(int(f.get("n_bytes_data", 0)) for f in files)
    print(
        f"[lib] DONE {len(files)} files, {total_bytes / 1e9:.2f} GB tensor data -> {args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
