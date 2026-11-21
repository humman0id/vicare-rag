#!/usr/bin/env python3
"""Sync notebooks/ViCare_ai4life_submission.ipynb with Google Drive Colab file."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEFAULT_FILE_ID = "1oobFwEBHSVCTt2S8qAH-UNI63OraRCHN"
DEFAULT_NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "ViCare_ai4life_submission.ipynb"
COLAB_MIME = "application/vnd.google.colaboratory"


def main() -> None:
    parser = argparse.ArgumentParser(description="Push/pull ViCare_ai4life_submission.ipynb on Google Drive")
    parser.add_argument("direction", choices=["push", "pull"], help="push local -> Drive or pull Drive -> local")
    parser.add_argument("--file-id", default=DEFAULT_FILE_ID)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    args = parser.parse_args()

    if args.direction == "push":
        cmd = [
            "gdrive",
            "mutate",
            "update",
            args.file_id,
            str(args.notebook),
            "--mime-type",
            COLAB_MIME,
        ]
    else:
        cmd = [
            "gdrive",
            "readonly",
            "download",
            args.file_id,
            "--out",
            str(args.notebook),
        ]

    subprocess.run(cmd, check=True)
    print(f"Synced ({args.direction}) {args.notebook} <-> Drive ID {args.file_id}")


if __name__ == "__main__":
    main()
