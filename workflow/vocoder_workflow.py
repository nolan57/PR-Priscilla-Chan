#!/usr/bin/env python3
"""
Vocoder Workflow - Complete pipeline for vocoder training data preparation.

Workflow:
    Raw Audio -> Separation (UVR5) [vocoder] -> Cleaning [sola] -> Slicing [vocoder] -> Quality Filter [vocoder] -> Preprocess (npz) [vocoder]

Usage:
    python -m workflow vocoder --input-dir ./raw_songs --output-dir ./data
    python -m workflow vocoder -i ./raw -o ./data --skip-existing
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Optional
import shutil

PCS_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PCS_ROOT))

from workflow.state import StateTracker
from workflow.env_config import get_default_config
from workflow.uvr_model_config import resolve_model_name


# Load environment configuration
_env_config = get_default_config()

# Environment mapping (from env_config.toml)
ENV_UVR = _env_config.get_env("uvr")
ENV_SOLA = _env_config.get_env("sola")

DATASET_TOOLS_CLI = str(PCS_ROOT / "SingingVocoders" / "utils" / "dataset-tools-cli")


def run_in_env(env: str, cmd: list, timeout: int = 3600) -> tuple[bool, str]:
    """Run command in specified conda environment."""
    full_cmd = ["conda", "run", "-n", env, *cmd]
    try:
        result = subprocess.run(
            full_cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
    except Exception as e:
        return False, str(e)


def separate_vocal(
    input_path: Path, output_dir: Path, model: str, sr: int = 44100
) -> tuple[bool, str]:
    """Separate vocal from mixed audio using UVR5 (uvr env).
    
    Args:
        input_path: Path to input audio file
        output_dir: Output directory for separated vocals
        model: Model alias (e.g., 'MDX23C-8KFFT') or file name
        sr: Sample rate for output audio
        
    Returns:
        (success, message) tuple
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve model alias to actual file name
    resolved_model = resolve_model_name(model)

    cmd = [
        sys.executable,
        str(PCS_ROOT / "pipeline" / "separate_vocals_all.py"),
        "--input",
        str(input_path.parent),
        "--output",
        str(output_dir),
        "--model",
        resolved_model,
        "--sample_rate",
        str(sr),
    ]

    return run_in_env(ENV_UVR, cmd)


def clean_audio(
    input_path: Path, output_path: Path, method: str = "harmonic_residual"
) -> tuple[bool, str]:
    """Clean audio using CLI tools (sola env)."""
    if method == "none":
        shutil.copy(input_path, output_path)
        return True, "Skipped cleaning"

    cmd = [
        sys.executable,
        "-m",
        "audio_edit_cli.muter",
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-r",
        "0,0",
        "-m",
        method,
    ]

    return run_in_env(ENV_SOLA, cmd)


def slice_audio(
    input_path: Path,
    output_dir: Path,
    threshold: int = -40,
    min_length: int = 5000,
    min_interval: int = 300,
    hop_size: int = 10,
    max_sil_kept: int = 500,
) -> tuple[bool, str]:
    """
    Slice audio based on silence detection using dataset-tools-cli (vocoder env).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        DATASET_TOOLS_CLI,
        "slice-audio",
        str(input_path),
        "-o",
        str(output_dir),
        "-t",
        str(threshold),
        "-l",
        str(min_length),
        "-i",
        str(min_interval),
        "-s",
        str(hop_size),
        "-m",
        str(max_sil_kept),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            sliced_files = list(output_dir.glob("*.wav"))
            return True, f"Sliced into {len(sliced_files)} files"
        return False, result.stderr
    except Exception as e:
        return False, str(e)


def validate_audio_quality(
    input_dir: Path,
    output_dir: Path,
    rejected_dir: Optional[Path] = None,
    device: str = "cpu",
) -> tuple[bool, str]:
    """
    Validate sliced audio quality using validate_slice_auido.py (vocoder env).

    Filters audio based on:
    - voice_ratio: minimum human voice content
    - low_noise_ratio: maximum low frequency noise
    - high_pass_ok: presence of high frequency harmonics
    - is_multi_speaker: detects multiple speakers
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if rejected_dir:
        rejected_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PCS_ROOT / "SingingVocoders" / "utils" / "validate_slice_auido.py"),
        str(input_dir),
        "--html",
        str(output_dir / "filter_report.html"),
        "--svg",
        str(output_dir / "filter_report.svg"),
        "--list",
        str(output_dir / "clean_files.txt"),
        "--device",
        device,
    ]

    if rejected_dir:
        cmd.extend(["--rejected-dir", str(rejected_dir)])

    return run_in_env(ENV_UVR, cmd)


def preprocess_vocoder(wav_dir: Path, npz_dir: Path, config: Path) -> tuple[bool, str]:
    """Preprocess audio to npz for vocoder training (vocoder env)."""
    npz_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PCS_ROOT / "SingingVocoders" / "process.py"),
        "--config",
        str(config),
    ]

    return run_in_env(ENV_UVR, cmd)


def find_vocal_file(separated_dir: Path, original_name: str) -> Optional[Path]:
    """Find the separated vocal file."""
    stem = Path(original_name).stem
    for ext in [".wav", ".WAV"]:
        f = separated_dir / f"{stem}_(Vocals){ext}"
        if f.exists():
            return f
        f = separated_dir / f"{stem}{ext}"
        if f.exists():
            return f
    return None


def process_single_file(args: tuple) -> dict:
    """Process a single file through the vocoder pipeline."""
    (
        input_file,
        output_dir,
        model,
        config,
        skip_clean,
        skip_slicing,
        skip_validation,
        slice_params,
        validation_params,
        tracker,
        workflow_name,
    ) = args

    result = {
        "file": str(input_file),
        "success": False,
        "stages": {},
    }

    separated_dir = output_dir / "separated"
    cleaned_dir = output_dir / "cleaned"
    sliced_dir = output_dir / "sliced"
    validated_dir = output_dir / "validated"
    rejected_dir = output_dir / "rejected"
    npz_dir = output_dir / "npz"

    try:
        sep_path = find_vocal_file(separated_dir, input_file.name)

        if sep_path is None:
            success, msg = separate_vocal(input_file, separated_dir, model)
            result["stages"]["separation"] = msg
            if not success:
                tracker.update_file(str(input_file), "separated", "failed", msg)
                tracker.increment_failed(workflow_name)
                return result
            sep_path = find_vocal_file(separated_dir, input_file.name)

        if sep_path is None:
            raise FileNotFoundError(f"Separated vocal not found for {input_file.name}")

        tracker.update_file(str(input_file), "separated", "completed")

        cleaned_path = cleaned_dir / sep_path.name
        if skip_clean:
            shutil.copy(sep_path, cleaned_path)
            result["stages"]["cleaning"] = "skipped"
        else:
            success, msg = clean_audio(sep_path, cleaned_path)
            result["stages"]["cleaning"] = msg
            if not success:
                tracker.update_file(str(input_file), "cleaned", "failed", msg)
                tracker.increment_failed(workflow_name)
                return result

        tracker.update_file(str(input_file), "cleaned", "completed")

        if skip_slicing:
            result["stages"]["slicing"] = "skipped"
        else:
            success, msg = slice_audio(cleaned_path, sliced_dir, **slice_params)
            result["stages"]["slicing"] = msg
            if not success:
                tracker.update_file(str(input_file), "sliced", "failed", msg)
                tracker.increment_failed(workflow_name)
                return result

        tracker.update_file(str(input_file), "sliced", "completed")

        if skip_validation:
            result["stages"]["validation"] = "skipped"
        else:
            success, msg = validate_audio_quality(
                sliced_dir,
                validated_dir,
                rejected_dir,
                validation_params.get("device", "cpu"),
            )
            result["stages"]["validation"] = msg
            if not success:
                tracker.update_file(str(input_file), "validated", "failed", msg)
                tracker.increment_failed(workflow_name)
                return result

        tracker.update_file(str(input_file), "validated", "completed")

        result["success"] = True
        tracker.increment_completed(workflow_name)

    except Exception as e:
        result["error"] = str(e)
        tracker.increment_failed(workflow_name)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Vocoder training data preparation workflow"
    )
    parser.add_argument(
        "-i", "--input-dir", required=True, help="Input directory with raw audio files"
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, help="Output directory for processed data"
    )
    parser.add_argument("-m", "--model", default="UVR5_1", help="UVR5 model to use")
    parser.add_argument("-c", "--config", help="Vocoder config file")
    parser.add_argument(
        "--skip-cleaning", action="store_true", help="Skip audio cleaning step"
    )
    parser.add_argument(
        "--skip-slicing", action="store_true", help="Skip audio slicing step"
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip quality validation step"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip already processed files"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument("--reset", action="store_true", help="Reset workflow state")

    parser.add_argument(
        "--slice-threshold",
        type=int,
        default=-40,
        help="Silence detection threshold (dB)",
    )
    parser.add_argument(
        "--slice-min-length", type=int, default=5000, help="Minimum slice length (ms)"
    )
    parser.add_argument(
        "--slice-min-interval",
        type=int,
        default=300,
        help="Minimum interval between slices (ms)",
    )
    parser.add_argument("--slice-hop-size", type=int, default=10, help="Hop size (ms)")
    parser.add_argument(
        "--slice-max-sil-kept",
        type=int,
        default=500,
        help="Maximum silence to keep (ms)",
    )

    parser.add_argument(
        "--validation-device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for validation",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = list(input_dir.glob("**/*.wav")) + list(input_dir.glob("**/*.flac"))
    print(f"Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        print("No audio files found")
        sys.exit(1)

    tracker = StateTracker(str(output_dir / "workflow_state.db"))
    workflow_name = f"vocoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.reset:
        tracker.reset_workflow(workflow_name)
        print("Workflow state reset")

    tracker.init_workflow(workflow_name, len(audio_files))

    for f in audio_files:
        tracker.update_file(str(f), "raw", "pending")

    slice_params = {
        "threshold": args.slice_threshold,
        "min_length": args.slice_min_length,
        "min_interval": args.slice_min_interval,
        "hop_size": args.slice_hop_size,
        "max_sil_kept": args.slice_max_sil_kept,
    }

    validation_params = {
        "device": args.validation_device,
    }

    tasks = [
        (
            f,
            output_dir,
            args.model,
            args.config,
            args.skip_cleaning,
            args.skip_slicing,
            args.skip_validation,
            slice_params,
            validation_params,
            tracker,
            workflow_name,
        )
        for f in audio_files
    ]

    print(f"Processing {len(tasks)} files...")
    print(
        f"Using environments: separation={ENV_UVR}, cleaning={ENV_SOLA}, slicing={ENV_UVR}, validation={ENV_UVR}, preprocess={ENV_UVR}"
    )
    print(
        f"Slicing params: threshold={args.slice_threshold}dB, min_length={args.slice_min_length}ms"
    )

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_single_file, task) for task in tasks]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    stats = tracker.get_stats(workflow_name)

    print(f"\n{'=' * 50}")
    print(f"Workflow completed!")
    print(f"Total: {stats['total']}")
    print(f"Completed: {stats['completed']}")
    print(f"Failed: {stats['failed']}")
    print(f"{'=' * 50}")

    manifest = {
        "workflow": "vocoder",
        "created": datetime.now().isoformat(),
        "stats": stats,
        "output_dir": str(output_dir),
        "environments": {
            "separation": ENV_UVR,
            "cleaning": ENV_SOLA,
            "slicing": ENV_UVR,
            "validation": ENV_UVR,
            "preprocess": ENV_UVR,
        },
        "slicing_params": slice_params,
        "validation_params": validation_params,
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
