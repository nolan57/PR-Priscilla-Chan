#!/usr/bin/env python3
"""
Pydantic types for PCS MCP Server.
"""

from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path


class SepVocalsInput(BaseModel):
    input_dir: Path = Field(description="Input directory containing audio files")
    output_dir: Path = Field(description="Output directory for separated vocals")
    model: str = Field(default="UVR5_1", description="UVR5 model name")
    sample_rate: int = Field(default=44100, description="Sample rate for output")


class MuteAudioInput(BaseModel):
    input_path: Path = Field(description="Input audio file")
    output_path: Path = Field(description="Output audio file")
    regions: str = Field(description="Regions to mute (format: 'start,end;start,end')")
    method: str = Field(
        default="original",
        description="Muting method: original, harmonic_residual, adaptive_ducking, noise_replacement, spectral_subtraction, pink_noise_blend",
    )


class TrimAudioInput(BaseModel):
    input_path: Path = Field(description="Input audio file")
    output_path: Path = Field(description="Output audio file")
    start: float = Field(default=0, description="Start time in seconds")
    end: Optional[float] = Field(default=None, description="End time in seconds")
    fade_in: float = Field(default=0, description="Fade in duration in seconds")
    fade_out: float = Field(default=0, description="Fade out duration in seconds")
    normalize: bool = Field(default=False, description="Normalize audio")


class RemoveHarmonyInput(BaseModel):
    input_path: Path = Field(description="Input audio file")
    output_path: Path = Field(description="Output audio file")
    ref_region: str = Field(
        description="Reference region for harmony (format: 'start,end')"
    )
    threshold: float = Field(default=0.5, description="Threshold for similarity")
    sensitivity: float = Field(default=10, description="Sensitivity")


class MergeAudioInput(BaseModel):
    inputs: str = Field(description="Input audio files (comma-separated)")
    output_path: Path = Field(description="Output audio file")
    crossfade: float = Field(default=0, description="Crossfade duration in seconds")
    normalize: bool = Field(default=False, description="Normalize output")


class SliceAudioInput(BaseModel):
    input_path: Path = Field(description="Input audio file to slice")
    output_dir: Path = Field(description="Output directory for sliced files")
    threshold: int = Field(default=-40, description="Silence detection threshold (dB)")
    min_length: int = Field(default=5000, description="Minimum slice length (ms)")
    min_interval: int = Field(
        default=300, description="Minimum interval between slices (ms)"
    )
    hop_size: int = Field(default=10, description="Hop size (ms)")
    max_sil_kept: int = Field(default=500, description="Maximum silence to keep (ms)")


class ExtractF0Input(BaseModel):
    wav_path: Path = Field(description="Input WAV file")
    output_dir: Optional[Path] = Field(
        default=None, description="Output directory for F0 file"
    )
    save_vad: bool = Field(default=False, description="Save VAD mask")


class BuildNpzInput(BaseModel):
    wav_dir: Path = Field(description="Directory containing WAV files")
    textgrid_dir: Path = Field(description="Directory containing TextGrid files")
    f0_dir: Optional[Path] = Field(
        default=None, description="Directory containing F0 files"
    )
    output_dir: Path = Field(description="Output directory for NPZ files")
    use_external_f0: bool = Field(
        default=True, description="Use external F0 if available"
    )


class VocoderWorkflowInput(BaseModel):
    input_dir: Path = Field(description="Input directory with raw audio files")
    output_dir: Path = Field(description="Output directory for processed data")
    model: str = Field(default="UVR5_1", description="UVR5 model to use")
    config: Optional[Path] = Field(default=None, description="Vocoder config file")
    skip_cleaning: bool = Field(default=False, description="Skip audio cleaning step")
    skip_slicing: bool = Field(default=False, description="Skip audio slicing step")
    skip_existing: bool = Field(
        default=False, description="Skip already processed files"
    )
    workers: int = Field(default=4, description="Number of parallel workers")
    reset: bool = Field(default=False, description="Reset workflow state")
    slice_threshold: int = Field(
        default=-40, description="Silence detection threshold (dB)"
    )
    slice_min_length: int = Field(default=5000, description="Minimum slice length (ms)")
    slice_min_interval: int = Field(
        default=300, description="Minimum interval between slices (ms)"
    )


class DiffSingerWorkflowInput(BaseModel):
    input_dir: Path = Field(description="Input directory with raw audio files")
    output_dir: Path = Field(description="Output directory for processed data")
    model: str = Field(default="UVR5_1", description="UVR5 model to use")
    skip_cleaning: bool = Field(default=False, description="Skip audio cleaning step")
    skip_mfa: bool = Field(default=False, description="Skip MFA alignment step")
    skip_existing: bool = Field(
        default=False, description="Skip already processed files"
    )
    workers: int = Field(default=4, description="Number of parallel workers")
    reset: bool = Field(default=False, description="Reset workflow state")


class WorkflowStatusInput(BaseModel):
    db_path: str = Field(
        default="workflow_state.db", description="Path to workflow state database"
    )


class ToolResult(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None
