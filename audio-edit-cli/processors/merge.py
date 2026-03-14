#!/usr/bin/env python3
"""
Audio Merging Processor - Command line tool for merging multiple audio files.

Usage:
    python -m audio_edit_cli.merge --inputs "file1.wav,file2.wav" --output merged.wav
    python -m audio_edit_cli.merge -i "intro.wav,vocals.wav,outro.wav" -o final.wav --crossfade 0.5
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge multiple audio files")
    parser.add_argument(
        "-i", "--inputs", required=True, help="Input audio files (comma-separated)"
    )
    parser.add_argument("-o", "--output", required=True, help="Output audio file")
    parser.add_argument(
        "-c", "--crossfade", type=float, default=0, help="Crossfade duration in seconds"
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Normalize output audio"
    )

    args = parser.parse_args()

    input_files = [f.strip() for f in args.inputs.split(",")]

    for f in input_files:
        if not Path(f).exists():
            print(f"Error: Input file not found: {f}")
            sys.exit(1)

    print(f"Processing: {input_files}")
    print(f"Merging {len(input_files)} files...")

    if len(input_files) == 1:
        import shutil

        shutil.copy(input_files[0], args.output)
        print(f"Success: Output saved to {args.output}")
        return

    cmd = ["ffmpeg", "-y"]

    for f in input_files:
        cmd.extend(["-i", f])

    filter_parts = []
    for i in range(len(input_files)):
        filter_parts.append(f"[{i}:a]")

    if args.crossfade > 0 and len(input_files) > 1:
        filter_complex = ""
        for i in range(len(input_files)):
            filter_complex += f"[{i}:a]"
        filter_complex += f"acrossfade=d={args.crossfade}[out]"
    else:
        filter_complex = (
            "".join(filter_parts) + f"concat=n={len(input_files)}:v=0:a=1[out]"
        )

    cmd.extend(["-filter_complex", filter_complex])

    if args.normalize:
        cmd.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])

    cmd.extend(["-map", "[out]", args.output])

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: ffmpeg failed with return code {result.returncode}")
        print(result.stderr)
        sys.exit(1)

    print(f"Success: Output saved to {args.output}")


if __name__ == "__main__":
    main()
