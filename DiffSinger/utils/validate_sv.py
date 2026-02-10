#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SVS Data Validation (Alignment Directory Structure) → Output CSV + HTML + PNG
"""

import os
import argparse
import numpy as np
import librosa
import textgrid
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import json

# Configuration
SAMPLE_RATE = 22050
HOP_LENGTH = 256
F0_VOICED_THRESHOLD = 50.0
VAD_IOU_WARN_THRESHOLD = 0.6
SILENCE_PHONES = {"sil", "sp", "spn", "", "br", "<UNK>"}

# Global statistics
issue_counter = {
    "missing_files": 0,
    "duration_mismatch": 0,
    "low_vad_f0_iou": 0,
    "low_vad_tg_iou": 0,
    "low_f0_tg_iou": 0,
    "large_vad_only_region": 0,
    "load_error": 0,
}


def get_audio_info(wav_path):
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    duration = len(y) / sr
    n_frames = int(np.ceil(len(y) / HOP_LENGTH))
    return duration, n_frames


def load_and_validate_vad(vad_path, expected_frames):
    vad = np.load(vad_path)
    if vad.ndim != 1:
        raise ValueError("VAD must be 1D")
    if vad.dtype == bool:
        pass
    elif np.issubdtype(vad.dtype, np.number):
        vad = vad.astype(bool)
    else:
        raise ValueError(f"Unsupported VAD dtype: {vad.dtype}")
    if len(vad) > expected_frames:
        return vad[:expected_frames]
    elif len(vad) < expected_frames:
        return np.pad(vad, (0, expected_frames - len(vad)), constant_values=False)
    return vad


def load_and_validate_f0(f0_path, expected_frames):
    f0 = np.load(f0_path)
    if f0.ndim != 1:
        raise ValueError("F0 must be 1D")
    if len(f0) > expected_frames:
        return f0[:expected_frames]
    elif len(f0) < expected_frames:
        return np.pad(f0, (0, expected_frames - len(f0)), constant_values=0.0)
    return f0


def get_textgrid_non_silence_vad(tg_path, total_duration, n_frames):
    tg = textgrid.TextGrid.fromFile(tg_path)
    intervals = []
    for tier in tg:
        name = tier.name.lower()
        if "phone" in name or name in ["phones", "phonemes"]:
            for interval in tier:
                if interval.mark not in SILENCE_PHONES:
                    intervals.append((interval.minTime, interval.maxTime))
            break
    else:
        for tier in tg:
            if "word" in name or name in ["words"]:
                for interval in tier:
                    if interval.mark.strip() not in SILENCE_PHONES:
                        intervals.append((interval.minTime, interval.maxTime))
                break

    frame_times = np.linspace(0, total_duration, n_frames, endpoint=False)
    vad_tg = np.zeros(n_frames, dtype=bool)
    for start, end in intervals:
        idx = np.where((frame_times >= start) & (frame_times < end))[0]
        if len(idx) > 0:
            vad_tg[idx] = True
    return vad_tg


def compute_iou(mask1, mask2):
    overlap = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(overlap / union) if union > 0 else 0.0


def check_sample(wav_path, tg_path, f0_path, vad_path, time_tol=0.15):
    global issue_counter
    result = {
        "rel_path": "",
        "status": "valid",
        "issues": [],
        "dur_wav": 0.0,
        "dur_tg": 0.0,
        "iou_vad_f0": 0.0,
        "iou_vad_tg": 0.0,
        "iou_f0_tg": 0.0,
        "vad_only_ratio": 0.0,
        "n_frames": 0,
    }

    try:
        dur_wav, n_frames = get_audio_info(wav_path)
        dur_tg = textgrid.TextGrid.fromFile(tg_path).maxTime
        result.update({"dur_wav": dur_wav, "dur_tg": dur_tg, "n_frames": n_frames})
    except Exception as e:
        result["status"] = "error"
        result["issues"].append(f"Load error: {e}")
        issue_counter["load_error"] += 1
        return result

    if abs(dur_wav - dur_tg) > time_tol:
        msg = f"Audio-TG duration mismatch: {dur_wav:.2f}s vs {dur_tg:.2f}s"
        result["issues"].append(msg)
        result["status"] = "warning"
        issue_counter["duration_mismatch"] += 1

    try:
        f0 = load_and_validate_f0(f0_path, n_frames)
        vad_user = load_and_validate_vad(vad_path, n_frames)
    except Exception as e:
        result["status"] = "error"
        result["issues"].append(f"Feature load failed: {e}")
        issue_counter["load_error"] += 1
        return result

    vad_from_f0 = f0 > F0_VOICED_THRESHOLD

    try:
        vad_tg = get_textgrid_non_silence_vad(tg_path, dur_wav, n_frames)
    except Exception as e:
        result["issues"].append(f"TextGrid parsing failed: {e}")
        result["status"] = "warning"
        return result

    iou_vad_f0 = compute_iou(vad_user, vad_from_f0)
    iou_vad_tg = compute_iou(vad_user, vad_tg)
    iou_f0_tg = compute_iou(vad_from_f0, vad_tg)

    result.update({
        "iou_vad_f0": iou_vad_f0,
        "iou_vad_tg": iou_vad_tg,
        "iou_f0_tg": iou_f0_tg,
    })

    if iou_vad_f0 < VAD_IOU_WARN_THRESHOLD:
        result["issues"].append(f"VAD vs F0-VAD IoU={iou_vad_f0:.2f}")
        result["status"] = "warning"
        issue_counter["low_vad_f0_iou"] += 1

    if iou_vad_tg < VAD_IOU_WARN_THRESHOLD:
        result["issues"].append(f"VAD vs TG-VAD IoU={iou_vad_tg:.2f}")
        result["status"] = "warning"
        issue_counter["low_vad_tg_iou"] += 1

    if iou_f0_tg < VAD_IOU_WARN_THRESHOLD:
        result["issues"].append(f"F0-VAD vs TG-VAD IoU={iou_f0_tg:.2f}")
        result["status"] = "warning"
        issue_counter["low_f0_tg_iou"] += 1

    vad_no_f0 = np.logical_and(vad_user, ~vad_from_f0)
    vad_only_ratio = vad_no_f0.sum() / n_frames if n_frames > 0 else 0.0
    result["vad_only_ratio"] = vad_only_ratio

    if vad_only_ratio > 0.05:
        result["issues"].append(f"Large VAD-only region ({vad_only_ratio:.1%})")
        result["status"] = "warning"
        issue_counter["large_vad_only_region"] += 1

    return result


def generate_html_report(df, stats, output_path):
    # Template
    template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>SVS Data Validation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .valid { background-color: #d4edda; }
            .warning { background-color: #fff3cd; }
            .error { background-color: #f8d7da; }
            .summary { background-color: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>SVS Data Validation Report</h1>

        <div class="summary">
            <h2>📊 Summary</h2>
            <p><strong>Total samples:</strong> {{ stats.total }}</p>
            <p><strong>Valid:</strong> {{ stats.valid }} ({{ "%.1f"|format(stats.valid_ratio*100) }}%)</p>
            <p><strong>Warning:</strong> {{ stats.warning }} ({{ "%.1f"|format(stats.warning_ratio*100) }}%)</p>
            <p><strong>Error (missing/corrupt):</strong> {{ stats.error }} ({{ "%.1f"|format(stats.error_ratio*100) }}%)</p>
            <p><strong>Config:</strong> SR={{ stats.sr }}, Hop={{ stats.hop }}, F0_Thresh={{ stats.f0_thresh }}</p>
        </div>

        <h2>📈 Issue Distribution</h2>
        <img src="consistency_stats.png" alt="Issue Stats" width="600">

        <h2>📋 Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Relative Path</th>
                    <th>Status</th>
                    <th>Issues</th>
                    <th>VAD-F0 IoU</th>
                    <th>VAD-TG IoU</th>
                    <th>F0-TG IoU</th>
                    <th>VAD-only Ratio</th>
                </tr>
            </thead>
            <tbody>
            {% for _, row in df.iterrows() %}
            <tr class="{{ row.status }}">
                <td>{{ row.rel_path }}</td>
                <td>{{ row.status|upper }}</td>
                <td>{{ row.issues or '' }}</td>
                <td>{{ "%.2f"|format(row.iou_vad_f0) if row.iou_vad_f0 else 'N/A' }}</td>
                <td>{{ "%.2f"|format(row.iou_vad_tg) if row.iou_vad_tg else 'N/A' }}</td>
                <td>{{ "%.2f"|format(row.iou_f0_tg) if row.iou_f0_tg else 'N/A' }}</td>
                <td>{{ "%.1f%%"|format(row.vad_only_ratio*100) if row.vad_only_ratio else '0.0%' }}</td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    """
    template = Template(template_str)
    html_out = template.render(df=df, stats=stats)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)


def plot_issue_stats(issue_counter, output_path):
    labels = []
    counts = []
    for k, v in issue_counter.items():
        if v > 0:
            # Readable labels
            label_map = {
                "missing_files": "Missing Files",
                "duration_mismatch": "Duration Mismatch",
                "low_vad_f0_iou": "Low VAD-F0 IoU",
                "low_vad_tg_iou": "Low VAD-TG IoU",
                "low_f0_tg_iou": "Low F0-TG IoU",
                "large_vad_only_region": "Large VAD-only Region",
                "load_error": "Load Error",
            }
            labels.append(label_map.get(k, k))
            counts.append(v)

    if not counts:
        # No issues, draw an "all passed" chart
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "✅ All Samples Valid!", ha='center', va='center', fontsize=16)
        plt.axis('off')
    else:
        plt.figure(figsize=(10, 6))
        sns.set(style="whitegrid")
        ax = sns.barplot(x=counts, y=labels, palette="Blues_d")
        ax.set_xlabel("Count")
        ax.set_title("Issue Distribution Across Dataset")
        for i, v in enumerate(counts):
            ax.text(v + 0.5, i, str(v), color='black', va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main(wav_dir, tg_dir, f0_dir, output_prefix):
    wav_root = Path(wav_dir)
    tg_root = Path(tg_dir)
    f0_root = Path(f0_dir)

    wav_files = sorted(wav_root.rglob("*.wav"))
    records = []

    print(f"🔍 Processing {len(wav_files)} samples...")

    for i, wav_path in enumerate(wav_files, 1):
        rel_path = wav_path.relative_to(wav_root)
        stem_rel = rel_path.with_suffix("")
        tg_path = tg_root / (stem_rel.with_suffix(".TextGrid"))
        f0_path = f0_root / (stem_rel.with_suffix(".f0.npy"))
        vad_path = f0_root / (stem_rel.with_suffix(".vad.npy"))

        missing = []
        if not tg_path.exists():
            missing.append("TextGrid")
        if not f0_path.exists():
            missing.append("f0.npy")
        if not vad_path.exists():
            missing.append("vad.npy")

        if missing:
            issue_counter["missing_files"] += 1
            records.append({
                "rel_path": str(rel_path),
                "status": "error",
                "issues": f"Missing: {', '.join(missing)}",
                "dur_wav": None,
                "dur_tg": None,
                "iou_vad_f0": None,
                "iou_vad_tg": None,
                "iou_f0_tg": None,
                "vad_only_ratio": None,
                "n_frames": None,
            })
            continue

        result = check_sample(str(wav_path), str(tg_path), str(f0_path), str(vad_path))
        result["rel_path"] = str(rel_path)
        records.append(result)

        if len(wav_files) <= 30 or i % 30 == 0:
            status_emoji = {"valid": "✅", "warning": "⚠️ ", "error": "❌"}.get(result["status"], "?")
            print(f"[{i}/{len(wav_files)}] {status_emoji} {rel_path}")

    # Convert to DataFrame
    df = pd.DataFrame(records)
    df.fillna("", inplace=True)

    # Statistics
    total = len(df)
    valid = len(df[df["status"] == "valid"])
    warning = len(df[df["status"] == "warning"])
    error = len(df[df["status"] == "error"])

    stats = {
        "total": total,
        "valid": valid,
        "valid_ratio": valid / total if total > 0 else 0,
        "warning": warning,
        "warning_ratio": warning / total if total > 0 else 0,
        "error": error,
        "error_ratio": error / total if total > 0 else 0,
        "sr": SAMPLE_RATE,
        "hop": HOP_LENGTH,
        "f0_thresh": F0_VOICED_THRESHOLD,
    }

    # Save CSV
    csv_path = f"{output_prefix}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"✅ CSV saved to: {csv_path}")

    # Save PNG
    png_path = f"{output_prefix}_stats.png"
    plot_issue_stats(issue_counter, png_path)
    print(f"📊 PNG saved to: {png_path}")

    # Save HTML
    html_path = f"{output_prefix}.html"
    generate_html_report(df, stats, html_path)
    print(f"🌐 HTML report saved to: {html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate SVS data → CSV + HTML + PNG")
    parser.add_argument("--wav_dir", required=True, help="WAV root directory")
    parser.add_argument("--tg_dir", required=True, help="TextGrid root directory")
    parser.add_argument("--f0_dir", required=True, help="F0/VAD root directory")
    parser.add_argument("--output", default="validation_report",
                        help="Output prefix (e.g., 'my_report' → my_report.csv/html/png)")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate")
    parser.add_argument("--hop", type=int, default=256, help="Hop length")
    parser.add_argument("--f0-thresh", type=float, default=50.0, help="F0 voiced threshold")

    args = parser.parse_args()

    # Update global variables
    globals()['SAMPLE_RATE'] = args.sr
    globals()['HOP_LENGTH'] = args.hop
    globals()['F0_VOICED_THRESHOLD'] = args.f0_thresh

    main(args.wav_dir, args.tg_dir, args.f0_dir, args.output)