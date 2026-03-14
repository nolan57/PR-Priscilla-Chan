#!/usr/bin/env python3
"""
Audio Muting Processor - Command line tool for muting regions in audio files.

Usage:
    python -m audio_edit_cli.muter --input audio.wav --output muted.wav --regions "1.5,3.0;5.0,6.0"
    python -m audio_edit_cli.muter -i input.wav -o output.wav -r "0,2" --method harmonic_residual

Methods:
    original:     Simple fade in/out silence
    harmonic_residual: Preserve high-frequency harmonics (recommended for singing)
    adaptive_ducking: Adaptive compression method
    noise_replacement: Environmental noise replacement
    spectral_subtraction: Spectral subtraction method
    pink_noise_blend: Pink noise blending
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_regions(regions_str: str) -> list[tuple[float, float]]:
    regions = []
    for part in regions_str.split(";"):
        part = part.strip()
        if "," in part:
            start, end = part.split(",")
            regions.append((float(start.strip()), float(end.strip())))
    return regions


def build_ffmpeg_chain(
    input_path: str,
    output_path: str,
    regions: list[tuple[float, float]],
    method: str = "original",
    fade_duration: float = 0.004,
):
    cmd = ["ffmpeg", "-y", "-i", input_path]

    filters = []

    for i, (start, end) in enumerate(regions):
        duration = end - start
        if duration <= 0:
            continue

        if method == "original":
            fade = min(fade_duration, duration / 2)
            seg_filter = f"[0:a]atrim=start={start}:end={end},afade=t=out:st=0:d={fade},afade=t=in:st={duration - fade}:d={fade}[seg{i}]"
        elif method == "harmonic_residual":
            fade = min(0.025, duration / 4)
            highpass = f"highpass=f=3000:poles=2"
            seg_filter = (
                f"[0:a]atrim=start={start}:end={end},"
                f"highpass=f=3000:poles=2,"
                f"volume=0.01,"
                f"highpass=f=3000:poles=2,"
                f"volume=0.3[high{i}];"
                f"[0:a]atrim=start={start}:end={end},volume=0.01[full{i}];"
                f"[high{i}][full{i}]amix=inputs=2:weights='0.3 1':duration=longest,"
                f"volume=0.08,"
                f"afade=t=in:st=0:d={fade},"
                f"afade=t=out:st={duration - fade}:d={fade}[seg{i}]"
            )
        elif method == "adaptive_ducking":
            fade = min(0.02, duration / 4)
            seg_filter = (
                f"[0:a]atrim=start={start}:end={end},"
                f"compand=attacks=0.001:decays=0.1:0=-60dB:-60dB:-100dB:-100dB:-100dB:-100dB:0=-60dB:0=-60dB:gain=-60,"
                f"volume=0.001,"
                f"afade=t=in:st=0:d={fade},"
                f"afade=t=out:st={duration - fade}:d={fade}[seg{i}]"
            )
        elif method == "noise_replacement":
            fade = min(0.015, duration / 3)
            seg_filter = (
                f"[0:a]atrim=start={start}:end={end},"
                f"volume=0,"
                f"afftfilt=real='hypot(re,im)*exp(0)'[seg{i}]"
            )
        elif method == "spectral_subtraction":
            fade = min(0.01, duration / 5)
            seg_filter = (
                f"[0:a]atrim=start={start}:end={end},"
                f"afftfilt=real='hypot(re,im)*exp(-0.5)':imag='hypot(re,im)*exp(-0.5)*0.2',"
                f"volume=0.3,"
                f"afade=t=in:st=0:d={fade},"
                f"afade=t=out:st={duration - fade}:d={fade}[seg{i}]"
            )
        elif method == "pink_noise_blend":
            fade = min(0.02, duration / 4)
            seg_filter = (
                f"[0:a]atrim=start={start}:end={end},"
                f"volume=0,"
                f"aformat=sample_fmts=fltp,"
                f"anoisered=sample_rate=44100:noiseamount=0.5,"
                f"volume=0.05,"
                f"afade=t=in:st=0:d={fade},"
                f"afade=t=out:st={duration - fade}:d={fade}[seg{i}]"
            )
        else:
            seg_filter = f"[0:a]atrim=start={start}:end={end},volume=0[seg{i}]"

        filters.append(seg_filter)

    if not filters:
        print("No valid regions to process")
        return None

    filter_complex = ";".join(filters)

    inputs_str = "".join([f"[seg{i}]" for i in range(len(regions))])
    if len(regions) == 1:
        final_filter = f"{filter_complex};{inputs_str}concat=n=1:v=0:a=1[out]"
    else:
        final_filter = (
            f"{filter_complex};{inputs_str}concat=n={len(regions)}:v=0:a=1[out]"
        )

    cmd.extend(["-filter_complex", final_filter, "-map", "[out]", output_path])

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Mute specified regions in audio file")
    parser.add_argument("-i", "--input", required=True, help="Input audio file")
    parser.add_argument("-o", "--output", required=True, help="Output audio file")
    parser.add_argument(
        "-r",
        "--regions",
        required=True,
        help="Regions to mute (format: 'start,end;start,end')",
    )
    parser.add_argument(
        "-m",
        "--method",
        default="original",
        choices=[
            "original",
            "harmonic_residual",
            "adaptive_ducking",
            "noise_replacement",
            "spectral_subtraction",
            "pink_noise_blend",
        ],
        help="Muting method (default: original)",
    )
    parser.add_argument(
        "-f",
        "--fade",
        type=float,
        default=0.004,
        help="Fade duration in seconds (default: 0.004)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    regions = parse_regions(args.regions)
    if not regions:
        print("Error: No valid regions specified")
        sys.exit(1)

    print(f"Processing: {args.input}")
    print(f"Regions to mute: {regions}")
    print(f"Method: {args.method}")

    cmd = build_ffmpeg_chain(args.input, args.output, regions, args.method, args.fade)

    if cmd is None:
        sys.exit(1)

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: ffmpeg failed with return code {result.returncode}")
        print(result.stderr)
        sys.exit(1)

    print(f"Success: Output saved to {args.output}")


if __name__ == "__main__":
    main()
