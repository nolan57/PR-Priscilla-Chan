# scripts/separate_vocals.py
import argparse
import subprocess
import sys
import os
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
    parser = argparse.ArgumentParser(description="Separate vocals from audio files.")
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
    # preserving the directory structure
    dir_mapping = {}
    for wav_dir in wav_dirs:
        rel_path = wav_dir.relative_to(input_dir)
        output_subdir = output_dir / rel_path
        output_subdir.mkdir(parents=True, exist_ok=True)
        dir_mapping[wav_dir] = output_subdir

    # Create tasks for each unique directory
    tasks = [(inp_dir, out_dir, model, sr, TIMEOUT) 
             for inp_dir, out_dir in dir_mapping.items()]

    if DEBUG:
        print(f"[DEBUG] Directories to process: {tasks}")

    # If there are no directories with WAV files to process, treat it as an error so the GUI can surface it
    if len(tasks) == 0:
        print(f"[ERROR] No WAV files found in input directory: {input_dir}")
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
        print(f"[INFO] [{++i}/{len(tasks)}] Processing {task[0]} -> {task[1]} /{len(tasks)}")
        print(f"[INFO] Processing directory {task[0]} -> {task[1]}")
        result = separate_one_folder(task)
        results.append(result)
        # Add a short delay between tasks to allow GPU memory to clear
        import time
        time.sleep(1)

    print(f"✅ Vocal separation: {sum(results)}/{len(tasks)} directories")

if __name__ == "__main__":
    main()