#!/usr/bin/env python3
"""
Audio Edit CLI - Command line interface for audio processing tools.

Usage:
    python -m audio_edit_cli.muter --input audio.wav --output muted.wav --regions 1.5,3.0
    python -m audio_edit_cli.trimmer --input audio.wav --output trimmed.wav --start 0.5 --end 10.0
    python -m audio_edit_cli.harmony --input audio.wav --output clean.wav --ref-region 0.5,1.5
"""

import argparse
import sys
from pathlib import Path

SAMPLE_RATE = 44100


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Available commands: muter, trimmer, harmony, merge")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "muter":
        from processors.muter import main as muter_main

        muter_main()
    elif command == "trimmer":
        from processors.trimmer import main as trimmer_main

        trimmer_main()
    elif command == "harmony":
        from processors.harmony import main as harmony_main

        harmony_main()
    elif command == "merge":
        from processors.merge import main as merge_main

        merge_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: muter, trimmer, harmony, merge")
        sys.exit(1)


if __name__ == "__main__":
    main()
