#!/usr/bin/env python3
"""
DiffSinger Workflow - Complete pipeline for DiffSinger training data preparation.

Workflow:
    Raw Audio -> Separation [vocoder] -> Cleaning [sola] -> F0 Extract [diffsinger] -> MFA [mfa] -> NPZ [diffsinger]

Usage:
    python -m workflow diffsinger --input-dir ./raw_songs --output-dir ./data
    python -m workflow diffsinger -i ./raw -o ./data --skip-existing
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

PCS_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PCS_ROOT))

from workflow.state import StateTracker
from workflow.env_config import get_default_config
from workflow.uvr_model_config import resolve_model_name


# Load environment configuration
_env_config = get_default_config()

# Environment mapping (from env_config.toml)
ENV_VOCODER = _env_config.get_env("vocoder")
ENV_DIFFSINGER = _env_config.get_env("diffsinger")
ENV_SOLA = _env_config.get_env("sola")
ENV_MFA = _env_config.get_env("mfa")


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
    """Separate vocal from mixed audio using UVR5 (vocoder env).
    
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

    return run_in_env(ENV_VOCODER, cmd)


def clean_audio(
    input_path: Path, output_path: Path, method: str = "harmonic_residual"
) -> tuple[bool, str]:
    """Clean audio using CLI tools (sola env)."""
    if method == "none":
        import shutil

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


def extract_f0(
    wav_path: Path, output_dir: Path, save_vad: bool = False
) -> tuple[bool, str]:
    """Extract high-precision F0 (diffsinger env)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PCS_ROOT / "pipeline" / "extract_perfect_f0.py"),
        str(wav_path),
        "--output_dir",
        str(output_dir),
    ]

    if save_vad:
        cmd.append("--save_vad")

    return run_in_env(ENV_DIFFSINGER, cmd)


def run_mfa(
    wav_dir: Path, text_dir: Path, output_dir: Path, dict_path: Optional[str] = None
) -> tuple[bool, str]:
    """Run Montreal Forced Aligner (mfa env)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mfa_align",
        str(wav_dir),
        dict_path or "english_us_arpa",
        str(output_dir),
        "--clean",
    ]

    return run_in_env(ENV_MFA, cmd)


def build_npz(
    wav_dir: Path, textgrid_dir: Path, f0_dir: Path, output_dir: Path
) -> tuple[bool, str]:
    """Build NPZ files for DiffSinger (diffsinger env)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PCS_ROOT / "pipeline" / "build_npzs.py"),
        "--wav_dir",
        str(wav_dir),
        "--textgrid_dir",
        str(textgrid_dir),
        "--output_dir",
        str(output_dir),
        "--use_external_f0",
    ]

    if f0_dir and f0_dir.exists():
        cmd.extend(["--f0_dir", str(f0_dir)])

    return run_in_env(ENV_DIFFSINGER, cmd)


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
    """Process a single file through the DiffSinger pipeline."""
    input_file, output_dir, model, skip_clean, skip_mfa, tracker, workflow_name = args

    result = {
        "file": str(input_file),
        "success": False,
        "stages": {},
    }

    separated_dir = output_dir / "separated"
    cleaned_dir = output_dir / "cleaned"
    f0_dir = output_dir / "f0"
    aligned_dir = output_dir / "aligned"
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
            import shutil

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

        f0_path = f0_dir / f"{cleaned_path.stem}.f0.npy"
        if not f0_path.exists():
            success, msg = extract_f0(cleaned_path, f0_dir)
            result["stages"]["f0_extraction"] = msg
            if not success:
                tracker.update_file(str(input_file), "f0_extracted", "failed", msg)
                tracker.increment_failed(workflow_name)
                return result

        tracker.update_file(str(input_file), "f0_extracted", "completed")

        if not skip_mfa:
            textgrid_path = aligned_dir / f"{cleaned_path.stem}.TextGrid"
            if not textgrid_path.exists():
                success, msg = run_mfa(cleaned_dir, output_dir / "texts", aligned_dir)
                result["stages"]["alignment"] = msg
                if not success:
                    tracker.update_file(str(input_file), "aligned", "failed", msg)
                    tracker.increment_failed(workflow_name)
                    return result

        tracker.update_file(str(input_file), "aligned", "completed")

        success, msg = build_npz(cleaned_dir, aligned_dir, f0_dir, npz_dir)
        result["stages"]["npz_building"] = msg

        if success:
            tracker.update_file(str(input_file), "completed", "completed")
            tracker.increment_completed(workflow_name)
            result["success"] = True
        else:
            tracker.update_file(str(input_file), "npz_ready", "failed", msg)
            tracker.increment_failed(workflow_name)

    except Exception as e:
        result["error"] = str(e)
        tracker.increment_failed(workflow_name)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="DiffSinger training data preparation workflow"
    )
    parser.add_argument(
        "-i", "--input-dir", required=True, help="Input directory with raw audio files"
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, help="Output directory for processed data"
    )
    parser.add_argument("-m", "--model", default="UVR5_1", help="UVR5 model to use")
    parser.add_argument(
        "--skip-cleaning", action="store_true", help="Skip audio cleaning step"
    )
    parser.add_argument(
        "--skip-mfa", action="store_true", help="Skip MFA alignment step"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip already processed files"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument("--reset", action="store_true", help="Reset workflow state")

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
    workflow_name = f"diffsinger_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.reset:
        tracker.reset_workflow(workflow_name)
        print("Workflow state reset")

    tracker.init_workflow(workflow_name, len(audio_files))

    for f in audio_files:
        tracker.update_file(str(f), "raw", "pending")

    tasks = [
        (
            f,
            output_dir,
            args.model,
            args.skip_cleaning,
            args.skip_mfa,
            tracker,
            workflow_name,
        )
        for f in audio_files
    ]

    print(f"Processing {len(tasks)} files...")
    print(
        f"Using environments: separation={ENV_VOCODER}, cleaning={ENV_SOLA}, f0={ENV_DIFFSINGER}, mfa={ENV_MFA}, npz={ENV_DIFFSINGER}"
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
        "workflow": "diffsinger",
        "created": datetime.now().isoformat(),
        "stats": stats,
        "output_dir": str(output_dir),
        "npz_dir": str(output_dir / "npz"),
        "environments": {
            "separation": ENV_VOCODER,
            "cleaning": ENV_SOLA,
            "f0_extraction": ENV_DIFFSINGER,
            "mfa_alignment": ENV_MFA,
            "npz_build": ENV_DIFFSINGER,
        },
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
