#!/usr/bin/env python3
"""
PCS MCP Server - Model Context Protocol server for audio processing automation.

This server exposes PCS audio processing tools and workflows as MCP tools,
allowing AI assistants to orchestrate audio processing pipelines.

Usage:
    # Run directly
    python -m mcp_server.server

    # Or use with uv
    uv run mcp_server.server
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Error: mcp package not installed. Install with: pip install mcp")
    sys.exit(1)

from ty.models import (
    SepVocalsInput,
    MuteAudioInput,
    TrimAudioInput,
    RemoveHarmonyInput,
    MergeAudioInput,
    SliceAudioInput,
    ExtractF0Input,
    BuildNpzInput,
    VocoderWorkflowInput,
    DiffSingerWorkflowInput,
    WorkflowStatusInput,
    ToolResult,
)

PCS_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PCS_ROOT))

# Import environment configuration
from workflow.env_config import get_default_config
from workflow.uvr_model_config import resolve_model_name

_env_config = get_default_config()

# Environment mapping (from env_config.toml)
ENV_UVR = _env_config.get_env("uvr")
ENV_SOLA = _env_config.get_env("sola")
ENV_DIFFSINGER = _env_config.get_env("diffsinger")
ENV_MFA = _env_config.get_env("mfa")
ENV_VOCODER = _env_config.get_env("vocoder")


def run_in_env(env_name: str, cmd: list, timeout: int = 3600) -> tuple[bool, str]:
    """Run command in specified conda environment."""
    full_cmd = ["conda", "run", "-n", env_name, *cmd]
    try:
        result = subprocess.run(
            full_cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, result.stderr
    except Exception as e:
        return False, str(e)


mcp = FastMCP("PCS Audio Pipeline")

# ============================================
# Audio Processing Tools
# ============================================


@mcp.tool()
def separate_vocals(req: SepVocalsInput) -> ToolResult:
    """Separate vocals from mixed audio using UVR5.
    
    Supports model aliases (e.g., 'MDX23C-8KFFT') which are resolved to actual
    model file names before passing to the separation script.
    """
    req.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Resolve model alias to actual file name
    resolved_model = resolve_model_name(req.model)

    cmd = [
        sys.executable,
        str(PCS_ROOT / "pipeline" / "separate_vocals_all.py"),
        "--input",
        str(req.input_dir),
        "--output",
        str(req.output_dir),
        "--model",
        resolved_model,
        "--sample_rate",
        str(req.sample_rate),
    ]

    success, msg = run_in_env(ENV_UVR, cmd, timeout=3600)
    if success:
        return ToolResult(success=True, message=f"Separation completed: {msg}")
    return ToolResult(success=False, message=f"Separation failed: {msg}")


@mcp.tool()
def mute_audio(req: MuteAudioInput) -> ToolResult:
    """Mute specified regions in audio file."""
    req.output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "audio_edit_cli.muter",
        "-i",
        str(req.input_path),
        "-o",
        str(req.output_path),
        "-r",
        req.regions,
        "-m",
        req.method,
    ]

    success, msg = run_in_env(ENV_SOLA, cmd, timeout=600)
    if success:
        return ToolResult(success=True, message=f"Muting completed: {msg}")
    return ToolResult(success=False, message=f"Muting failed: {msg}")


@mcp.tool()
def trim_audio(req: TrimAudioInput) -> ToolResult:
    """Trim audio file to specified time range."""
    req.output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "audio_edit_cli.trimmer",
        "-i",
        str(req.input_path),
        "-o",
        str(req.output_path),
        "-s",
        str(req.start),
    ]

    if req.end is not None:
        cmd.extend(["-e", str(req.end)])
    if req.fade_in > 0:
        cmd.extend(["--fade-in", str(req.fade_in)])
    if req.fade_out > 0:
        cmd.extend(["--fade-out", str(req.fade_out)])
    if req.normalize:
        cmd.append("--normalize")

    success, msg = run_in_env(ENV_SOLA, cmd, timeout=300)
    if success:
        return ToolResult(success=True, message=f"Trimming completed: {msg}")
    return ToolResult(success=False, message=f"Trimming failed: {msg}")


@mcp.tool()
def remove_harmony(req: RemoveHarmonyInput) -> ToolResult:
    """Remove harmony from audio using spectral masking."""
    req.output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "audio_edit_cli.harmony",
        "-i",
        str(req.input_path),
        "-o",
        str(req.output_path),
        "-r",
        req.ref_region,
        "-t",
        str(req.threshold),
        "-s",
        str(req.sensitivity),
    ]

    success, msg = run_in_env(ENV_SOLA, cmd, timeout=600)
    if success:
        return ToolResult(success=True, message=f"Harmony removal completed: {msg}")
    return ToolResult(success=False, message=f"Harmony removal failed: {msg}")


@mcp.tool()
def merge_audio(req: MergeAudioInput) -> ToolResult:
    """Merge multiple audio files."""
    req.output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "audio_edit_cli.merge",
        "-i",
        req.inputs,
        "-o",
        str(req.output_path),
    ]

    if req.crossfade > 0:
        cmd.extend(["-c", str(req.crossfade)])
    if req.normalize:
        cmd.append("--normalize")

    success, msg = run_in_env(ENV_SOLA, cmd, timeout=600)
    if success:
        return ToolResult(success=True, message=f"Merge completed: {msg}")
    return ToolResult(success=False, message=f"Merge failed: {msg}")


@mcp.tool()
def slice_audio(req: SliceAudioInput) -> ToolResult:
    """
    Slice audio based on silence detection using dataset-tools-cli.

    Splits audio files into smaller segments based on silence detection threshold.
    """
    req.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_tools_cli = str(
        PCS_ROOT / "SingingVocoders" / "utils" / "dataset-tools-cli"
    )

    cmd = [
        dataset_tools_cli,
        "slice-audio",
        str(req.input_path),
        "-o",
        str(req.output_dir),
        "-t",
        str(req.threshold),
        "-l",
        str(req.min_length),
        "-i",
        str(req.min_interval),
        "-s",
        str(req.hop_size),
        "-m",
        str(req.max_sil_kept),
    ]

    success, msg = run_in_env(ENV_VOCODER, cmd, timeout=600)
    if success:
        sliced_files = list(req.output_dir.glob("*.wav"))
        return ToolResult(
            success=True,
            message=f"Sliced into {len(sliced_files)} files: {msg}",
            data={
                "sliced_count": len(sliced_files),
                "output_dir": str(req.output_dir),
            },
        )
    return ToolResult(success=False, message=f"Slicing failed: {msg}")


# ============================================
# Pipeline Tools
# ============================================


@mcp.tool()
def extract_f0(req: ExtractF0Input) -> ToolResult:
    """Extract high-precision F0 from audio using CREPE+Harvest+Parselmouth fusion."""
    output_dir = req.output_dir or req.wav_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PCS_ROOT / "pipeline" / "extract_perfect_f0.py"),
        str(req.wav_path),
        "--output_dir",
        str(output_dir),
    ]

    if req.save_vad:
        cmd.append("--save_vad")

    success, msg = run_in_env(ENV_DIFFSINGER, cmd, timeout=1200)
    if success:
        f0_path = output_dir / f"{req.wav_path.stem}.f0.npy"
        return ToolResult(
            success=True,
            message=f"F0 extraction completed: {msg}",
            data={"f0_path": str(f0_path)},
        )
    return ToolResult(success=False, message=f"F0 extraction failed: {msg}")


@mcp.tool()
def build_diffsinger_npz(req: BuildNpzInput) -> ToolResult:
    """Build DiffSinger training NPZ files from WAV + TextGrid + F0."""
    req.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PCS_ROOT / "pipeline" / "build_npzs.py"),
        "--wav_dir",
        str(req.wav_dir),
        "--textgrid_dir",
        str(req.textgrid_dir),
        "--output_dir",
        str(req.output_dir),
    ]

    if req.use_external_f0 and req.f0_dir:
        cmd.extend(["--f0_dir", str(req.f0_dir), "--use_external_f0"])

    success, msg = run_in_env(ENV_DIFFSINGER, cmd, timeout=3600)
    if success:
        npz_count = len(list(req.output_dir.glob("*.npz")))
        return ToolResult(
            success=True,
            message=f"NPZ build completed: {msg}",
            data={"npz_count": npz_count, "output_dir": str(req.output_dir)},
        )
    return ToolResult(success=False, message=f"NPZ build failed: {msg}")


# ============================================
# Workflow Tools (Skills)
# ============================================


@mcp.tool()
def run_vocoder_workflow(req: VocoderWorkflowInput) -> ToolResult:
    """
    [Skill] Complete vocoder training data preparation workflow.

    Process: Separation -> Cleaning -> Preprocess

    Supports breakpoint resume - already processed files are skipped.
    """
    req.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "workflow.vocoder_workflow",
        "-i",
        str(req.input_dir),
        "-o",
        str(req.output_dir),
        "-m",
        req.model,
        "--workers",
        str(req.workers),
    ]

    if req.skip_cleaning:
        cmd.append("--skip-cleaning")
    if req.skip_existing:
        cmd.append("--skip-existing")
    if req.reset:
        cmd.append("--reset")
    if req.config:
        cmd.extend(["-c", str(req.config)])

    # Run workflow module in base environment (it will manage sub-process environments)
    success, msg = run_in_env("base", cmd, timeout=86400)
    if success:
        return ToolResult(
            success=True,
            message=f"Vocoder workflow completed: {msg}",
            data={"output_dir": str(req.output_dir)},
        )
    return ToolResult(success=False, message=f"Workflow failed: {msg}")


@mcp.tool()
def run_diffsinger_workflow(req: DiffSingerWorkflowInput) -> ToolResult:
    """
    [Skill] Complete DiffSinger training data preparation workflow.

    Process: Separation -> Cleaning -> F0 Extract -> MFA Alignment -> NPZ Build

    Supports breakpoint resume - already processed files are skipped.
    """
    req.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "workflow.diffsinger_workflow",
        "-i",
        str(req.input_dir),
        "-o",
        str(req.output_dir),
        "-m",
        req.model,
        "--workers",
        str(req.workers),
    ]

    if req.skip_cleaning:
        cmd.append("--skip-cleaning")
    if req.skip_mfa:
        cmd.append("--skip-mfa")
    if req.skip_existing:
        cmd.append("--skip-existing")
    if req.reset:
        cmd.append("--reset")

    # Run workflow module in base environment (it will manage sub-process environments)
    success, msg = run_in_env("base", cmd, timeout=86400)
    if success:
        return ToolResult(
            success=True,
            message=f"DiffSinger workflow completed: {msg}",
            data={"output_dir": str(req.output_dir)},
        )
    return ToolResult(success=False, message=f"Workflow failed: {msg}")


@mcp.tool()
def get_workflow_status(req: WorkflowStatusInput) -> ToolResult:
    """Get workflow processing status from SQLite database."""
    try:
        import sqlite3

        conn = sqlite3.connect(req.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT workflow_name, total_files, completed, failed, started_at
            FROM workflow_stats ORDER BY id DESC LIMIT 5
        """)

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "workflow": row[0],
                    "total": row[1],
                    "completed": row[2],
                    "failed": row[3],
                    "started_at": row[4],
                }
            )

        conn.close()

        return ToolResult(
            success=True,
            message="Workflow status retrieved",
            data={"workflows": results},
        )
    except Exception as e:
        return ToolResult(success=False, message=f"Error: {str(e)}")


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    print("Starting PCS MCP Server...")
    mcp.run()
