#!/usr/bin/env python3
"""
Harmony Removal Processor - Command line tool for removing harmony from audio files.

Usage:
    python -m audio_edit_cli.harmony --input audio.wav --output clean.wav --ref-region 0.5,1.5
    python -m audio_edit_cli.harmony -i input.wav -o output.wav -r "0.5,1.5" --threshold 0.5 --sensitivity 10 --smoothing 10
"""

import argparse
import numpy as np
import soundfile as sf
from scipy import signal
import sys
from pathlib import Path


SAMPLE_RATE = 44100


def extract_reference_spectrum(Zxx, time_bins, ref_regions):
    """Extract average spectrum from reference regions."""
    spectra = []

    for start_time, end_time in ref_regions:
        start_idx = np.argmin(np.abs(time_bins - start_time))
        end_idx = np.argmin(np.abs(time_bins - end_time))

        if end_idx > start_idx:
            region_spec = np.abs(Zxx[:, start_idx:end_idx])
            avg_spec = np.mean(region_spec, axis=1)
            spectra.append(avg_spec)

    if spectra:
        return np.mean(spectra, axis=0)
    return None


def compute_soft_mask(Zxx, ref_spectrum, threshold=0.5, sensitivity=10, smoothing=10):
    """Compute soft mask based on reference spectrum similarity."""
    magnitude = np.abs(Zxx)

    normalized_ref = ref_spectrum / (np.max(ref_spectrum) + 1e-8)
    normalized_mag = magnitude / (np.max(magnitude) + 1e-8)

    similarity = normalized_mag * normalized_ref[:, np.newaxis]

    mask = np.exp(sensitivity * (similarity - threshold))
    mask = np.clip(mask, 0, 1)

    if smoothing > 0:
        from scipy.ndimage import uniform_filter1d

        mask = uniform_filter1d(mask, size=smoothing, axis=0)

    return mask


def remove_harmony(
    input_path, output_path, ref_regions, threshold=0.5, sensitivity=10, smoothing=10
):
    """Remove harmony from audio using spectral masking."""
    print(f"Loading audio: {input_path}")
    waveform, sr = sf.read(input_path)

    if sr != SAMPLE_RATE:
        print(f"Warning: Sample rate {sr} != {SAMPLE_RATE}, resampling...")
        from scipy import signal

        num_samples = int(len(waveform) * SAMPLE_RATE / sr)
        waveform = signal.resample(waveform, num_samples)
        sr = SAMPLE_RATE

    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)

    print(f"Computing STFT...")
    n_fft = 2048
    hop_length = 512

    f, t, Zxx = signal.stft(waveform, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    print(f"Extracting reference spectrum from regions: {ref_regions}")
    ref_spectrum = extract_reference_spectrum(Zxx, t, ref_regions)

    if ref_spectrum is None:
        print("Error: No valid reference regions found")
        sys.exit(1)

    print(f"Generating spectral mask...")
    mask = compute_soft_mask(Zxx, ref_spectrum, threshold, sensitivity, smoothing)

    print(f"Applying mask...")
    enhanced_Zxx = Zxx * mask

    print(f"Reconstructing audio...")
    _, enhanced_audio = signal.istft(
        enhanced_Zxx, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length
    )

    enhanced_audio = enhanced_audio[: len(waveform)]

    max_val = np.max(np.abs(enhanced_audio))
    if max_val > 0:
        enhanced_audio = enhanced_audio * (0.95 / max_val)

    print(f"Saving output: {output_path}")
    sf.write(output_path, enhanced_audio, sr)

    print(f"Success: Output saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove harmony from audio using spectral masking"
    )
    parser.add_argument("-i", "--input", required=True, help="Input audio file")
    parser.add_argument("-o", "--output", required=True, help="Output audio file")
    parser.add_argument(
        "-r",
        "--ref-region",
        required=True,
        help="Reference region for harmony (format: 'start,end')",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for similarity (default: 0.5)",
    )
    parser.add_argument(
        "-s", "--sensitivity", type=float, default=10, help="Sensitivity (default: 10)"
    )
    parser.add_argument(
        "-m",
        "--smoothing",
        type=int,
        default=10,
        help="Smoothing window size (default: 10)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    ref_region = args.ref_region.split(",")
    if len(ref_region) != 2:
        print(f"Error: Invalid reference region format: {args.ref_region}")
        sys.exit(1)

    ref_regions = [(float(ref_region[0]), float(ref_region[1]))]

    print(f"Processing: {args.input}")
    print(f"Reference region: {ref_regions[0]}")
    print(
        f"Threshold: {args.threshold}, Sensitivity: {args.sensitivity}, Smoothing: {args.smoothing}"
    )

    remove_harmony(
        args.input,
        args.output,
        ref_regions,
        args.threshold,
        args.sensitivity,
        args.smoothing,
    )


if __name__ == "__main__":
    main()
