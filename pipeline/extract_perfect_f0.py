#!/usr/bin/env python3
"""
Robust F0 extractor for AI-separated vocal tracks (e.g., from Demucs/UVR5).
Uses Silero VAD to detect true voiced regions, then fuses:
  - CREPE (robust, with confidence)
  - Harvest (high-precision)
  - get_pitch_parselmouth (consistent with DiffSinger pipeline)
Outputs f0.npy aligned with mel spectrogram frames.
"""

import os
import argparse
import sys
from pathlib import Path

import numpy as np
import librosa
import pyworld as pw
import crepe
import torch

diffsinger_dir = Path(__file__).parent.parent / "DiffSinger"
os.environ['PYTHONPATH'] = str(diffsinger_dir)
sys.path.insert(0, str(diffsinger_dir))

# Import get_pitch_parselmouth from binarizer_utils for F0 extraction
from utils.binarizer_utils import get_pitch_parselmouth

# ------------------ CONFIGURATION ------------------
SAMPLE_RATE = 44100
HOP_LENGTH = 512  # MUST match your mel extraction script!
FRAME_PERIOD = HOP_LENGTH / SAMPLE_RATE * 1000  # in ms

# VAD & Post-processing
VAD_THRESHOLD = 0.5      # Silero VAD speech probability threshold
MIN_VOICED_DURATION = 3  # min voiced segment length (frames)

# CREPE
CREPE_CONFIDENCE_THRESHOLD = 0.7

# --------------------------------------------------

def load_silero_vad():
    """Load Silero VAD model."""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    get_speech_timestamps, _, read_audio, *_, collect_chunks = utils
    return model, get_speech_timestamps

def extract_vad_mask(y, sr, hop_length, vad_model, get_speech_timestamps):
    """
    Extract VAD mask using Silero VAD (which requires 16kHz).
    Input y can be at any sample rate (e.g., 44100 Hz).
    Output mask is aligned to original hop_length and sr.
    """
    # Silero VAD only supports 8k/16k/etc. → resample to 16kHz for VAD
    TARGET_VAD_SR = 16000
    if sr != TARGET_VAD_SR:
        y_vad = librosa.resample(y, orig_sr=sr, target_sr=TARGET_VAD_SR)
        sr_vad = TARGET_VAD_SR
    else:
        y_vad = y
        sr_vad = sr

    # Run VAD on 16kHz audio
    speech_timestamps = get_speech_timestamps(
        torch.tensor(y_vad).float(),
        vad_model,
        sampling_rate=sr_vad,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        window_size_samples=512  # fixed for 16kHz
    )

    # Convert timestamps back to original sample rate's frame indices
    total_frames = len(y) // hop_length + 1
    vad_mask = np.zeros(total_frames, dtype=bool)

    for seg in speech_timestamps:
        # seg times are in samples @ 16kHz → convert to seconds → to original frames
        start_sec = seg['start'] / sr_vad
        end_sec = seg['end'] / sr_vad
        start_frame = int(start_sec * sr / hop_length)
        end_frame = int(end_sec * sr / hop_length) + 1
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)
        vad_mask[start_frame:end_frame] = True

    # Remove short voiced segments
    if MIN_VOICED_DURATION > 1:
        from scipy.ndimage import binary_opening
        vad_mask = binary_opening(vad_mask, structure=np.ones(MIN_VOICED_DURATION))

    return vad_mask

def resample_f0(f0, time_orig, time_target):
    """Resample F0 while preserving unvoiced (0) regions."""
    nonzero = f0 > 0
    if np.sum(nonzero) == 0:
        return np.zeros_like(time_target)
    f0_interp = np.interp(time_target, time_orig[nonzero], f0[nonzero], left=0, right=0)
    voiced_interp = np.interp(time_target, time_orig, nonzero.astype(float))
    f0_resampled = np.where(voiced_interp > 0.5, f0_interp, 0.0)
    return f0_resampled

def main(wav_path, output_dir=None, save_vad=False):
    # Load audio
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    y = y.astype(np.float64)  # Ensure double precision for pyworld
    assert sr == SAMPLE_RATE, f"Expected sample rate {SAMPLE_RATE}, got {sr}"

    total_frames = len(y) // HOP_LENGTH + 1
    target_times = np.arange(total_frames) * HOP_LENGTH / sr

    # --- Step 1: VAD ---
    print("Running Silero VAD...")
    vad_model, get_speech_timestamps = load_silero_vad()
    vad_mask = extract_vad_mask(y, sr, HOP_LENGTH, vad_model, get_speech_timestamps)

    # --- Step 2: Extract F0 candidates ---
    print("Extracting F0 with CREPE...")
    # CREPE predict returns (time, frequency, confidence, activation)
    _, f0_crepe, confidence, _ = crepe.predict(
        y,
        sr,
        model_capacity='full',
        viterbi=False,
        step_size=int(HOP_LENGTH / sr * 1000)
        # device='cuda' if torch.cuda.is_available() else 'cpu',
        # return_periodicity=False
    )
    t_crepe = np.arange(len(f0_crepe)) * (HOP_LENGTH / sr)
    f0_crepe = np.nan_to_num(f0_crepe, nan=0.0)
    confidence = np.nan_to_num(confidence, nan=0.0)

    print("Extracting F0 with Harvest...")
    f0_harvest, t_harvest = pw.harvest(y, sr, frame_period=FRAME_PERIOD)
    f0_harvest = pw.stonemask(y, f0_harvest, t_harvest, sr)

    print("Extracting F0 with get_pitch_parselmouth...")
    # Calculate the expected length in frames
    length = len(y) // HOP_LENGTH + 1
    f0_parselmouth, uv = get_pitch_parselmouth(
        y, sr, length,
        hop_size=HOP_LENGTH,
        f0_min=40,  # Standard minimum F0
        f0_max=16000,  # Standard maximum F0
        interp_uv=False  # Don't interpolate unvoiced regions initially
    )
    # Convert to the same time axis as other methods
    t_parselmouth = np.arange(len(f0_parselmouth)) * (HOP_LENGTH / sr)

    # --- Step 3: Resample all to target time axis ---
    f0_crepe_rs = resample_f0(f0_crepe, t_crepe, target_times)
    conf_crepe_rs = np.interp(target_times, t_crepe, confidence, left=0, right=0)
    f0_harvest_rs = resample_f0(f0_harvest, t_harvest, target_times)
    f0_parselmouth_rs = resample_f0(f0_parselmouth, t_parselmouth, target_times)

    # --- Step 4: Fuse with priority: CREPE > Harvest > get_pitch_parselmouth (only in VAD regions) ---
    f0_final = np.zeros(total_frames, dtype=np.float32)

    # Priority 1: CREPE (high confidence)
    crepe_good = vad_mask & (conf_crepe_rs >= CREPE_CONFIDENCE_THRESHOLD)
    f0_final[crepe_good] = f0_crepe_rs[crepe_good]

    # Priority 2: Harvest (if CREPE not confident)
    harvest_good = vad_mask & (~crepe_good) & (f0_harvest_rs > 0)
    f0_final[harvest_good] = f0_harvest_rs[harvest_good]

    # Priority 3: get_pitch_parselmouth (last resort)
    parselmouth_good = vad_mask & (~crepe_good) & (~harvest_good) & (f0_parselmouth_rs > 0)
    f0_final[parselmouth_good] = f0_parselmouth_rs[parselmouth_good]

    # --- Step 5: Post-process: median filter in voiced regions ---
    f0_smooth = f0_final.copy()
    for i in range(len(f0_final)):
        if vad_mask[i]:
            start = max(0, i - 2)
            end = min(len(f0_final), i + 3)
            f0_smooth[i] = np.median(f0_final[start:end])
    f0_final = f0_smooth

    # --- Step 6: Ensure exact frame count ---
    if len(f0_final) > total_frames:
        f0_final = f0_final[:total_frames]
    elif len(f0_final) < total_frames:
        f0_final = np.pad(f0_final, (0, total_frames - len(f0_final)), constant_values=0)

    # --- Save outputs ---
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    output_dir = output_dir or os.path.dirname(wav_path) or "."
    f0_path = os.path.join(output_dir, f"{base_name}.f0.npy")
    np.save(f0_path, f0_final.astype(np.float32))
    print(f"✅ Saved F0 to {f0_path}")

    if save_vad:
        vad_path = os.path.join(output_dir, f"{base_name}.vad.npy")
        np.save(vad_path, vad_mask)
        print(f"✅ Saved VAD mask to {vad_path}")

    return f0_final, vad_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract robust F0 for AI-separated vocals.")
    parser.add_argument("wav_path", type=str, help="Path to input WAV file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--save_vad", action="store_true", help="Also save VAD mask (.vad.npy)")
    args = parser.parse_args()

    main(args.wav_path, args.output_dir, args.save_vad)