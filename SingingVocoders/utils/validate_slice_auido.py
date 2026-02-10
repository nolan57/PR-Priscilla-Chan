#!/usr/bin/env python3
import os
import torch
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def mel_frequencies(n_mels=128, fmin=0.0, fmax=8000.0):
    """Reproduce librosa.mel_frequencies in PyTorch"""
    mels = torch.linspace(0, n_mels - 1, n_mels)
    freqs = 700.0 * (10**(mels / 2595.0) - 1)
    # Scale to [fmin, fmax]
    freqs = freqs - freqs[0]  # start from 0
    freqs = freqs * (fmax - fmin) / (freqs[-1] + 1e-6) + fmin
    return freqs


import torchaudio.functional as F

import torchaudio.functional as F_audio  # Avoid conflict with torch.nn.functional

def is_clean_voice_torch(audio_path, sr=22050, device="cpu", **thresholds):
    try:
        y, orig_sr = torchaudio.load(str(audio_path))
    except Exception as e:
        raise RuntimeError(f"Load failed: {e}")

    if y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)
    if orig_sr != sr:
        resampler = T.Resample(orig_sr, sr).to(device)
        y = resampler(y)
    y = y.squeeze(0).to(device)

    # === VAD via RMS ===
    frame_length = 2048
    hop_length = 512
    pad_len = (frame_length - hop_length) // 2
    y_pad = torch.nn.functional.pad(y, (pad_len, pad_len), mode='constant')
    frames = y_pad.unfold(0, frame_length, hop_length)
    rms = torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-9)
    rms_db = 20 * torch.log10(rms)
    rms_db = rms_db - rms_db.max()
    voice_frames = rms_db > -thresholds.get("top_db", 30)
    voice_ratio = voice_frames.float().mean().item()

    # === Low noise check (<100 Hz) ===
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        win_length=int(0.05 * sr),
        hop_length=hop_length,
        f_min=50,
        f_max=sr // 2,
        n_mels=80,
        window_fn=torch.hann_window,
        power=2.0
    ).to(device)
    mel = mel_transform(y.unsqueeze(0))
    log_mel = T.AmplitudeToDB(stype='power', top_db=None)(mel).squeeze(0)
    freqs_mel = mel_frequencies(n_mels=80, fmin=50, fmax=sr // 2)
    low_bin = torch.where(freqs_mel < 100)[0]
    low_idx = low_bin[-1].item() if len(low_bin) > 0 else 0
    avg_mel = log_mel.mean(dim=1)
    total_energy = avg_mel.abs().sum()
    low_energy = avg_mel[:low_idx + 1].abs().sum()
    low_ratio = (low_energy / (total_energy + 1e-6)).item()

    # === ORIGINAL: High-band harmonic check using F0 ===
    high_pass_ok = True
    median_f0 = None
    if y.shape[0] > sr // 2:  # >0.5s
        try:
            f0, _ = F_audio.pitch.pYIN(
                y.cpu(),
                sample_rate=sr,
                frame_time=hop_length / sr,
                freq_low=60,
                freq_high=800
            )
            voiced = f0 > 0
            if voiced.any():
                median_f0 = torch.median(f0[voiced]).item()
                max_harm = int(12000 / median_f0) if median_f0 > 0 else 1
                harmonics = [median_f0 * h for h in range(1, max_harm + 1)]
                harmonics_in_band = [h for h in harmonics if 7000 <= h <= 11000]

                if len(harmonics_in_band) == 0:
                    # Fallback: use energy ratio with relaxed threshold
                    spec = torch.stft(
                        y, n_fft=2048, hop_length=hop_length,
                        win_length=int(0.05*sr),
                        window=torch.hann_window(int(0.05*sr), device=device),
                        return_complex=True
                    )
                    mag = torch.abs(spec)
                    freqs = torch.linspace(0, sr//2, mag.shape[0], device=device)
                    high_mask = (freqs >= 7000) & (freqs <= 11000)
                    high_energy = mag[high_mask].mean()
                    total_energy_stft = mag.mean()
                    high_ratio = (high_energy / (total_energy_stft + 1e-6)).item()
                    high_pass_ok = high_ratio <= thresholds.get("max_high_ratio_fallback", 0.5)
                else:
                    high_pass_ok = True
            else:
                high_pass_ok = False
        except Exception:
            high_pass_ok = voice_ratio > 0.7

    # === NEW: Multi-speaker detection (using same F0 if available) ===
    is_multi_speaker = False
    try:
        if median_f0 is not None and y.shape[0] > int(0.5 * sr):
            # Reuse f0 from above if possible, or recompute
            f0_local = f0 if 'f0' in locals() else None
            if f0_local is None:
                f0_local, _ = F_audio.pitch.pYIN(
                    y.cpu(), sample_rate=sr,
                    frame_time=hop_length / sr,
                    freq_low=60, freq_high=800
                )
            voiced = f0_local > 0
            if voiced.any():
                f0_voiced = f0_local[voiced]
                f0_std = torch.std(f0_voiced).item()
                f0_mean = torch.mean(f0_voiced).item()
                f0_cv = f0_std / (f0_mean + 1e-6)

                # Spectral flatness in 2–8 kHz
                n_fft = 2048
                win_len = int(0.05 * sr)
                spec = torch.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_len,
                                  window=torch.hann_window(win_len, device=device),
                                  return_complex=True)
                mag = torch.abs(spec)
                freqs = torch.linspace(0, sr//2, mag.shape[0], device=device)
                mid_high_mask = (freqs >= 2000) & (freqs <= 8000)
                mid_high_mag = mag[mid_high_mask]
                if mid_high_mag.numel() > 0:
                    eps = 1e-9
                    geo = torch.exp(torch.mean(torch.log(mid_high_mag + eps), dim=0))
                    arith = torch.mean(mid_high_mag, dim=0)
                    flatness = torch.mean(geo / (arith + eps)).item()
                else:
                    flatness = 0.5

                # Heuristic: unstable pitch + flat spectrum → likely multi-speaker
                if f0_cv > 0.35 and flatness > 0.6:
                    is_multi_speaker = True
    except Exception:
        is_multi_speaker = False

    # === Final Decision ===
    min_voice = thresholds.get("min_voice_ratio", 0.3)
    max_low = thresholds.get("max_low_noise_ratio", 0.25)

    is_clean = (
        not is_multi_speaker and
        voice_ratio >= min_voice and
        low_ratio <= max_low and
        high_pass_ok
    )

    metrics = {
        "voice_ratio": voice_ratio,
        "low_noise_ratio": low_ratio,
        "high_pass_ok": high_pass_ok,
        "is_multi_speaker": is_multi_speaker,
        "reasons": []
    }

    if is_multi_speaker:
        metrics["reasons"].append("multi_speaker")
    if voice_ratio < min_voice:
        metrics["reasons"].append("low_voice_ratio")
    if low_ratio > max_low:
        metrics["reasons"].append("high_low_noise")
    if not high_pass_ok:
        metrics["reasons"].append("high_band_failed")

    # print(f"{audio_path} : clean={is_clean}, reasons={metrics['reasons'] or 'none'}")

    return is_clean, metrics


import shutil  # <-- Added import

def filter_and_report(input_dir, output_html="filter_report.html", output_svg="filter_report.svg", output_list="clean_files.txt", device="cpu", rejected_dir=None):
    input_dir = Path(input_dir)
    all_wavs = sorted(input_dir.rglob("*.wav"))
    if not all_wavs:
        print("⚠️ No .wav files found!")
        return

    results = []
    thresholds = {
        "top_db": 30,
        "min_voice_ratio": 0.3,
        "max_low_noise_ratio": 0.25,
        "max_high_ratio_fallback": 0.5,
    }

    print(f"🎵 Processing {len(all_wavs)} files on {device}...")
    for p in tqdm(all_wavs, desc="Analyzing"):
        try:
            is_clean, metrics = is_clean_voice_torch(p, device=device, **thresholds)
            results.append({
                "path": str(p),
                "name": p.stem,
                "is_clean": is_clean,
                "voice": metrics["voice_ratio"],
                "low": metrics["low_noise_ratio"],
                "high": metrics["high_pass_ok"],
                "multi_speaker": metrics["is_multi_speaker"],
                "reason": metrics["reasons"] or "none",
            })
        except Exception as e:
            print(f"❌ Skip {p.name}: {e}")

    if not results:
        print("No valid files processed.")
        return

    # Sort by name
    results.sort(key=lambda x: x["name"])
    clean_files = [r["path"] for r in results if r["is_clean"]]
    rejected_files = [r["path"] for r in results if not r["is_clean"]]

    print(f"✅ Found {len(clean_files)} clean files out of {len(results)} total files")
    with open(output_list, "w") as f:
        f.write("\n".join(clean_files))
    print(f"📄 Clean file list saved to: {output_list}")

    # === Added: Process rejected files ===
    if rejected_dir and rejected_files:
        rejected_dir = Path(rejected_dir)
        rejected_dir.mkdir(parents=True, exist_ok=True)

        print(f"🗑️ Moving {len(rejected_files)} rejected files to: {rejected_dir}")
        moved_paths = []
        for src_path in tqdm(rejected_files, desc="Moving rejected"):
            src = Path(src_path)
            dst = rejected_dir / src.name
            try:
                shutil.move(str(src), str(dst))
                moved_paths.append(str(dst))
            except Exception as e:
                print(f"⚠️ Failed to move {src}: {e}")

        # Save rejected list to rejected_dir
        rejected_list_file = rejected_dir / "rejected_files.txt"
        with open(rejected_list_file, "w") as f:
            f.write("\n".join(moved_paths))
        print(f"📄 Rejected file list saved to: {rejected_list_file}")

    # === Plotly Report (unchanged) ===
    names = [r["name"] for r in results]
    voice_vals = [r["voice"] for r in results]
    low_vals = [-r["low"] for r in results]
    high_vals = [-int(r["high"]) for r in results]  # bool -> int for plotting
    colors = ["#2ca02c" if r["is_clean"] else "#d62728" for r in results]

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(f"Audio Quality Report (Total: {len(results)}, Clean: {len(clean_files)})",)
    )
    fig.add_trace(go.Bar(y=names, x=voice_vals, orientation='h', name='Voice Ratio', marker_color='steelblue'))
    fig.add_trace(go.Bar(y=names, x=low_vals, orientation='h', name='Low Noise Ratio', marker_color='orange'))
    fig.add_trace(go.Bar(y=names, x=high_vals, orientation='h', name='High Pass OK', marker_color='purple'))

    # Color background per row
    shapes = []
    for i, color in enumerate(colors):
        shapes.append(dict(
            type="rect", xref="paper", yref="y",
            x0=0, x1=1, y0=i - 0.45, y1=i + 0.45,
            fillcolor=color, opacity=0.1, line_width=0
        ))
    fig.update_layout(shapes=shapes)
    fig.update_layout(
        barmode='overlay',
        height=max(400, len(names) * 25),
        title="Green = Keep, Red = Reject",
        xaxis_title="Metric Value (positive=voice, negative=noise)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Save
    fig.write_html(output_html)
    fig.write_image(output_svg, width=1200, height=max(400, len(names)*25))
    print(f"\n📊 Report saved:")
    print(f" HTML: {output_html}")
    print(f" SVG: {output_svg}")
    print(f" Clean list: {output_list}")
    print(f" Clean rate: {len(clean_files)/len(results):.1%}")


# ===== CLI =====
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter noisy vocals using pure PyTorch/torchaudio")
    parser.add_argument("input_dir", help="Root directory of .wav files")
    parser.add_argument("--html", default="filter_report.html", help="Output HTML report")
    parser.add_argument("--svg", default="filter_report.svg", help="Output SVG image")
    parser.add_argument("--list", default="clean_files.txt", help="Clean file list")
    parser.add_argument("--rejected-dir", default=None, help="Directory to move rejected files (optional)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    args = parser.parse_args()

    filter_and_report(
        input_dir=args.input_dir,
        output_html=args.html,
        output_svg=args.svg,
        output_list=args.list,
        device=args.device,
        rejected_dir=args.rejected_dir  # <-- Pass new parameter
    )