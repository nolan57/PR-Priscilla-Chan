#!/usr/bin/env python3
"""
Audio Trimming Processor - Command line tool for trimming audio files.

Usage:
    python -m audio_edit_cli.trimmer --input audio.wav --output trimmed.wav --start 0.5 --end 10.0
    python -m audio_edit_cli.trimmer -i input.wav -o output.wav -s 0 -e 30 --fade-in 0.01 --fade-out 0.01 --normalize
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Trim audio file")
    parser.add_argument("-i", "--input", required=True, help="Input audio file")
    parser.add_argument("-o", "--output", required=True, help="Output audio file")
    parser.add_argument(
        "-s",
        "--start",
        type=float,
        default=0,
        help="Start time in seconds (default: 0)",
    )
    parser.add_argument(
        "-e", "--end", type=float, default=None, help="End time in seconds"
    )
    parser.add_argument(
        "--fade-in", type=float, default=0, help="Fade in duration in seconds"
    )
    parser.add_argument(
        "--fade-out", type=float, default=0, help="Fade out duration in seconds"
    )
    parser.add_argument("--normalize", action="store_true", help="Normalize audio")
    parser.add_argument(
        "--padding", type=float, default=0, help="Add padding to start/end in seconds"
    )
    parser.add_argument(
        "--curve",
        default="lin",
        choices=["lin", "exp", "log", "scurve"],
        help="Fade curve type",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if args.end is not None and args.end <= args.start:
        print(f"Error: End time must be greater than start time")
        sys.exit(1)

    print(f"Processing: {args.input}")
    print(
        f"Trim: {args.start}s -> {args.end}s"
        if args.end
        else f"Trim from {args.start}s to end"
    )

    cmd = ["ffmpeg", "-y", "-i", args.input]

    filters = []

    trim_start = max(0, args.start - args.padding) if args.padding > 0 else args.start
    trim_end = args.end

    if trim_start > 0 or trim_end is not None:
        if trim_end is not None:
            duration = trim_end - trim_start
            cmd.extend(["-ss", str(trim_start), "-t", str(duration)])
        else:
            cmd.extend(["-ss", str(trim_start)])

    filter_str = ""

    duration = trim_end - trim_start if trim_end is not None else None

    actual_start = args.start - trim_start if args.padding > 0 else 0
    if args.fade_in > 0 and actual_start >= 0:
        filter_str = f"afade=t=in:st={actual_start}:d={args.fade_in}:curve={args.curve}"

    if args.fade_out > 0:
        if filter_str:
            filter_str += ","
        fade_out_start = duration - args.fade_out if duration else 0
        filter_str += (
            f"afade=t=out:st={fade_out_start}:d={args.fade_out}:curve={args.curve}"
        )

    if args.normalize:
        if filter_str:
            filter_str += ","
        filter_str += "loudnorm=I=-16:TP=-1.5:LRA=11"

    if filter_str:
        cmd.extend(["-af", filter_str])

    cmd.append(args.output)

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: ffmpeg failed with return code {result.returncode}")
        print(result.stderr)
        sys.exit(1)

    print(f"Success: Output saved to {args.output}")


if __name__ == "__main__":
    main()
