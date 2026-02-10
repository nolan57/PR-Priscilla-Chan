#!/usr/bin/env python3
# scripts/separate_vocals_skip_existing.py
import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Toggleable debug mode (set in main via --debug)
DEBUG = True
# Default subprocess timeout (seconds)
TIMEOUT = 1800

def separate_one_folder(args):
    input_folder, output_folder, model, sr, timeout = args
    if DEBUG:
        print(f"[DEBUG] Separating vocals from {input_folder} to {output_folder}")

    # Get the absolute path to the UVR script to avoid working directory issues
    # Resolve UVR script relative to this script's project root to avoid CWD issues
    project_root = Path(__file__).resolve().parents[1]
    uvr_script = str(project_root / "ultimatevocalremovergui" / "separate.py")

    cmd = [
        sys.executable, uvr_script,
        "--input_folder", str(input_folder),
        "--output_folder", str(output_folder),
        "--model", model,
        "--vocals_only", "true",
        "--sample_rate", str(sr)
    ]

    # Add debug flag to subprocess command if enabled
    if DEBUG:
        cmd.append("--debug")

    try:
        if DEBUG:
            # Print command and stream child output to parent (avoid pipe buffer deadlocks)
            print(f"[SUBPROC CMD] {' '.join(cmd)}")
            res = subprocess.run(cmd, check=False, stdout=None, stderr=None, timeout=timeout)
            # When streaming to parent, subprocess.run returns the returncode only
            print(f"[SUBPROC] returncode={res}")
            success = (res == 0)
        else:
            # Normal, quiet mode - capture output and enforce timeout
            res = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, timeout=timeout)
            success = (res.returncode == 0)
            # Only log errors
            if not success:
                print(f"[ERROR] Subprocess failed for {input_folder} (returncode={res.returncode})")
                stderr = res.stderr.decode('utf-8')[:500] if res.stderr else ""
                if stderr:
                    print(f"[ERROR] Stderr: {stderr}...")
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Subprocess timed out for {input_folder} after 30 minutes")
        return False
    except Exception as e:
        print(f"[EXCEPTION] separate_one_folder failed for {input_folder}: {e}")
        return False

    return success

def main():
    parser = argparse.ArgumentParser(description="Separate vocals from audio files, skipping existing outputs.")
    # Accept both legacy and canonical names for compatibility with GUI and CLI
    parser.add_argument("--input", "--input_dir", dest="input_dir", type=Path, required=True, help="Input directory containing WAV files.")
    parser.add_argument("--output", "--output_dir", dest="output_dir", type=Path, required=True, help="Output directory for separated vocals.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use for separation.")
    parser.add_argument("--sr", "--sample_rate", dest="sample_rate", type=int, default=44100, help="Sample rate for output audio.")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output for subprocesses")
    parser.add_argument("--timeout", type=int, default=3600, help="Subprocess timeout in seconds (default: 3600)")

    args = parser.parse_args()
    if args.debug:
        print(f"separate vocals args:{args}")

    # Set module-level DEBUG flag so worker processes follow same debug mode
    global DEBUG
    DEBUG = bool(args.debug)
    # Set module-level TIMEOUT
    global TIMEOUT
    TIMEOUT = int(args.timeout)

    input_dir = args.input_dir
    output_dir = args.output_dir
    model = args.model
    sr = args.sample_rate

    # Find all unique subdirectories that contain WAV files
    wav_dirs = set()
    for wav in input_dir.glob("**/*.wav"):
        wav_dirs.add(wav.parent)

    # Map each input directory to its corresponding output directory
    # preserving the directory structure, but only if there are files to process
    dir_mapping = {}
    for wav_dir in wav_dirs:
        rel_path = wav_dir.relative_to(input_dir)
        output_subdir = output_dir / rel_path
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Check if any WAV file in this directory doesn't exist in output
        wav_files = list(wav_dir.glob("*.wav"))
        need_processing = False
        for wav_file in wav_files:
            # Calculate the expected output file paths (both with and without suffix)
            # When vocals_only is true, UVR typically outputs files named like "filename_(Vocals).wav"
            output_file_path_no_suffix = output_subdir / wav_file.name
            output_file_path_with_suffix = output_subdir / f"{wav_file.stem}_(Vocals).wav"

            # Check if either file exists (with priority to the suffixed version)
            if not output_file_path_with_suffix.exists() and not output_file_path_no_suffix.exists():
                need_processing = True
                break

        if need_processing:
            # Create a temporary directory with only the files that need processing
            temp_input_dir = Path(f"/tmp/uvt_{wav_dir.name}_temp")  # Use system temp directory
            temp_input_dir.mkdir(exist_ok=True)

            # Copy only files that don't exist in output to the temp directory
            for wav_file in wav_files:
                # Check both possible output file names
                output_file_path_no_suffix = output_subdir / wav_file.name
                output_file_path_with_suffix = output_subdir / f"{wav_file.stem}_(Vocals).wav"

                # If neither output file exists, we need to process this input file
                if not output_file_path_with_suffix.exists() and not output_file_path_no_suffix.exists():
                    # Copy file to temp directory
                    temp_file_path = temp_input_dir / wav_file.name
                    shutil.copy2(wav_file, temp_file_path)

            # Map temp input directory to the actual output directory
            dir_mapping[temp_input_dir] = output_subdir
        elif DEBUG:
            print(f"[DEBUG] Skipping directory {wav_dir} - all files already exist in {output_subdir}")

    # Create tasks for each unique directory
    tasks = [(inp_dir, out_dir, model, sr, TIMEOUT)
             for inp_dir, out_dir in dir_mapping.items()]

    if DEBUG:
        print(f"[DEBUG] Directories to process: {tasks}")

    # If there are no directories with WAV files to process, treat it as an error so the GUI can surface it
    if len(tasks) == 0:
        print(f"[ERROR] No WAV files found in input directory or all files already exist in output: {input_dir}")
        sys.exit(1)

    # Limit workers to 1 for GPU processing to avoid resource competition
    # GPU models can't be effectively parallelized across processes
    max_workers = min(args.workers, 1) if "gpu" in model.lower() or "cuda" in model.lower() else args.workers
    if max_workers != args.workers:
        print(f"[INFO] Limiting workers to {max_workers} for GPU processing to avoid resource competition")

    # Process tasks sequentially with a delay between each to avoid resource spikes
    results = []
    print("Separating vocals...")
    print(f"To process: {len(tasks)} directories")
    i = 0
    for task in tqdm(tasks, total=len(tasks)):
        input_dir_task = task[0]
        output_dir_task = task[1]
        print(f"[INFO] [{i+1}/{len(tasks)}] Processing {input_dir_task} -> {output_dir_task}")
        result = separate_one_folder(task)
        results.append(result)
        # Add a short delay between tasks to allow GPU memory to clear
        import time
        time.sleep(1)
        i += 1

        # Clean up the temporary directory after processing
        shutil.rmtree(input_dir_task)

    print(f"✅ Vocal separation: {sum(results)}/{len(tasks)} directories")

if __name__ == "__main__":
    main()
