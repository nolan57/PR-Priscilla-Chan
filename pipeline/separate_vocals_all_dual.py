#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust two-stage separation with:
- Full recursive .wav discovery (including subdirs)
- Per-directory processing to ensure UVR handles all files
- Detailed progress and error reporting
- Output validation
"""

import sys
import argparse
import os
from pathlib import Path
import shutil
from typing import List, Tuple

# Add the ultimatevocalremovergui directory to the path so we can import its modules
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "ultimatevocalremovergui"))

# Import the CLI class from UVR
from cli import CLI

DEBUG = False


def is_valid_wav(file_path: Path) -> bool:
    """Basic check: file exists, ends with .wav (case-insensitive), non-zero size."""
    try:
        return (
                file_path.is_file()
                and file_path.suffix.lower() == ".wav"
                and file_path.stat().st_size > 44  # at least header size
        )
    except Exception:
        return False


def collect_all_wav_dirs(root_dir: Path) -> List[Path]:
    """
    Find all directories that contain at least one valid .wav file.
    Returns list of directories (including root if it has .wav).
    """
    wav_dirs = set()
    for wav_file in root_dir.rglob("*.wav"):
        if is_valid_wav(wav_file):
            wav_dirs.add(wav_file.parent)
    return sorted(wav_dirs)


def run_separation_on_dir(input_dir: Path, output_dir: Path, model_name: str, process_method: str, sample_rate: int) -> Tuple[bool, str]:
    """Run UVR on a single directory (non-recursive) by directly using the CLI class."""
    try:
        # Get all .wav files in the input directory
        wav_files = list(input_dir.glob("*.wav"))

        if not wav_files:
            return False, f"No WAV files found in {input_dir}"

        # Create CLI instance
        cli = CLI()

        # Temporarily modify sys.argv to simulate command line arguments
        original_argv = sys.argv[:]

        try:
            # Prepare arguments for the CLI
            args_list = [sys.argv[0]]  # Script name
            args_list.extend([str(f) for f in wav_files])  # Input files
            args_list.extend(["-o", str(output_dir)])  # Output directory
            args_list.extend(["-m", process_method])  # Processing method
            args_list.extend(["-t", model_name])  # Model name
            args_list.extend(["--primary-only"])  # Only save primary stem (vocals)

            if DEBUG:
                args_list.append("--verbose")

            sys.argv = args_list

            # Parse and configure the CLI
            parsed_args = cli.parse_args()
            cli.configure(parsed_args)

            # Run the processing
            result_code = cli.process()

            # Restore original argv
            sys.argv = original_argv

            if result_code:
                return True, ""
            else:
                return False, "CLI processing failed"

        except Exception as e:
            # Restore original argv in case of exception
            sys.argv = original_argv
            raise e

    except Exception as e:
        return False, f"Exception: {e}"


def process_stage2_recursive(input_root: Path, output_root: Path, model2: str, method2: str, sr: int) -> bool:
    """Process all .wav files recursively by directory."""
    wav_dirs = collect_all_wav_dirs(input_root)

    if not wav_dirs:
        print(f"[WARNING] No valid .wav files found in '{input_root}' or its subdirectories.")
        return False

    print(f"[INFO] Found {len(wav_dirs)} directory(ies) containing .wav files:")
    for d in wav_dirs:
        rel = d.relative_to(input_root)
        wav_count = len([f for f in d.iterdir() if is_valid_wav(f)])
        print(f"  - {rel} ({wav_count} file(s))")

    success_count = 0
    for i, src_dir in enumerate(wav_dirs, 1):
        rel_path = src_dir.relative_to(input_root)
        dst_dir = output_root / rel_path
        print(f"\n[PROGRESS] Processing directory {i}/{len(wav_dirs)}: {rel_path}")

        # Ensure output dir exists
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Run UVR on this directory
        success, err_msg = run_separation_on_dir(src_dir, dst_dir, model2, method2, sr)
        if not success:
            print(f"[ERROR] Failed to process {src_dir}:\n{err_msg}", file=sys.stderr)
            continue

        # Verify output
        output_wavs = list(dst_dir.glob("*_(Vocals).wav"))
        if output_wavs:
            success_count += 1
            print(f"  → Success: {len(output_wavs)} vocal file(s) saved to {dst_dir}")
        else:
            print(f"  ⚠️ Warning: No output files generated in {dst_dir} (model may have skipped)")

    if success_count == 0:
        print("[ERROR] All directories failed to produce output.", file=sys.stderr)
        return False

    print(f"\n[SUMMARY] Stage 2 completed: {success_count}/{len(wav_dirs)} directories succeeded.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Two-stage vocal separation with full recursive support.")
    parser.add_argument("input_dir", help="Input directory (will scan all subdirs for .wav)")
    parser.add_argument("output_dir", help="Output directory (structure preserved)")
    parser.add_argument("--mode", choices=["full", "stage1", "stage2"], default="full")
    parser.add_argument("--model1", default="MDX23C Model")
    parser.add_argument("--method1", choices=["mdx", "vr", "demucs"], default="mdx",
                        help="Processing method for stage 1 (default: mdx)")
    parser.add_argument("--model2", default="DeEcho Aggressive")
    parser.add_argument("--method2", choices=["mdx", "vr", "demucs"], default="vr",
                        help="Processing method for stage 2 (default: vr)")
    parser.add_argument("--sr", type=int, default=44100)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[FATAL] Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # === Stage 1 only ===
    if args.mode == "stage1":
        print("=== STAGE 1 ONLY ===")
        if not process_stage2_recursive(input_dir, output_dir, args.model1, args.method1, args.sr):
            sys.exit(1)
        print("[SUCCESS] Stage 1 completed.")
        return

    # === Stage 2 only ===
    if args.mode == "stage2":
        print("=== STAGE 2 ONLY ===")
        if not process_stage2_recursive(input_dir, output_dir, args.model2, args.method2, args.sr):
            sys.exit(1)
        print("[SUCCESS] Stage 2 completed.")
        return

    # === Full mode ===
    if args.mode == "full":
        print("=== FULL MODE: Stage1 → Stage2 ===")
        temp_stage1 = output_dir / "_temp_stage1"
        temp_stage1.mkdir(parents=True, exist_ok=True)

        print("\n>>> Running Stage 1...")
        if not process_stage2_recursive(input_dir, temp_stage1, args.model1, args.method1, args.sr):
            shutil.rmtree(temp_stage1, ignore_errors=True)
            sys.exit(1)

        print("\n>>> Running Stage 2...")
        if not process_stage2_recursive(temp_stage1, output_dir, args.model2, args.method2, args.sr):
            shutil.rmtree(temp_stage1, ignore_errors=True)
            sys.exit(1)

        shutil.rmtree(temp_stage1, ignore_errors=True)
        print("[SUCCESS] Full pipeline completed.")
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[FATAL] Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)