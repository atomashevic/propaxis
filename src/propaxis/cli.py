"""Small CLI for model setup and scoring a CSV with a saved instrument."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import Instrument, download_models


def main(argv=None):
    parser = argparse.ArgumentParser(description="Proposition-based text measurement")
    sub = parser.add_subparsers(dest="command", required=True)
    models = sub.add_parser("download", help="Download models for subsequent local use")
    models.add_argument("--encoder", choices=["e5-small-v2", "e5-large-v2"], default="e5-small-v2")
    models.add_argument("--claim-check", action="store_true")
    score = sub.add_parser("score", help="Score a CSV using saved axes and scaling parameters")
    score.add_argument("instrument", type=Path)
    score.add_argument("input", type=Path)
    score.add_argument("--text-column", default="text")
    score.add_argument("--id-column")
    score.add_argument("--output", required=True, type=Path)
    score.add_argument("--device", default="cpu")
    score.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            print(json.dumps(download_models(args.encoder, claim_check=args.claim_check), indent=2))
            return
        with args.input.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        instrument = Instrument.load(
            args.instrument, device=args.device, local_files_only=args.offline
        )
        result = instrument.score(
            [row[args.text_column] for row in rows],
            ids=None if not args.id_column else [row[args.id_column] for row in rows],
        )
        result.to_json(args.output)
        print(f"Scored {len(result.scores)} documents. Wrote {args.output}")
    except (ValueError, KeyError, OSError, ImportError) as error:
        parser.exit(2, f"propaxis: {error}\n")


if __name__ == "__main__":
    main()
