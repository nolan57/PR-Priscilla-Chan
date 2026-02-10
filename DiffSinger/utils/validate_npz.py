#!/usr/bin/env python3
"""[Improved version] DiffSinger .npz data accuracy validation tool (includes vocoder closed-loop reconstruction)
Features:
1. Visualize F0 curve comparison (original vs .npz)
2. Overlay phoneme boundaries (from TextGrid)
3. Reconstruct audio using NSF-HiFiGAN (via vocoder_utils.py)
4. Compare reconstructed audio with original dry vocal mel spectrograms
5. Generate HTML report (with audio player)
6. [New] Diagnostics and fixes specifically for F0 length misalignment issues

Input:
- Original dry vocal WAV
- Corresponding .TextGrid file
- DiffSinger training .npz file

Output:
- validation_report.html (interactive report)
- plots/ (detailed visualizations)
"""

import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.interpolate import interp1d
import base64
from textgrid import TextGrid

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))
from utils.hparams import set_hparams, hparams

# ====== Vocoder support (imported from independent module) ======
try:
    from utils.vocoder_utils import load_vocoder

    VOCODER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ vocoder_utils.py not found or failed to load: {e}")
    VOCODER_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Vocoder initialization failed: {e}")
    VOCODER_AVAILABLE = False
# ========================================

# Global parameters
SAMPLE_RATE = 44100
HOP_LENGTH = 512  # 10ms
N_FFT = 2048
N_MELS = 128


def load_npz_data(npz_path):
    """Load core data from DiffSinger .npz file"""
    data = np.load(npz_path, allow_pickle=True)
    return {
        'f0': data['f0'],
        'mel': data['mel'],
        'phones': data['ph_seq'] if 'ph_seq' in data else None,
        'phone_durs': data['ph_dur'] if 'ph_dur' in data else None,
        'sr': SAMPLE_RATE,
        'hop_length': HOP_LENGTH
    }


def extract_ground_truth_f0(wav_path, f0_dir=None):
    """
    Load external .f0.npy as ground truth (aligned with your training data).
    If not found, fall back to parselmouth.
    """
    base_name = Path(wav_path).stem
    if f0_dir:
        f0_path = Path(f0_dir) / f"{base_name}.f0.npy"
        print(f"🔍 Searching for external F0: {f0_path}")
        if f0_path.exists():
            print(f"✅ Using external F0 as GT: {f0_path}")
            f0_gt = np.load(str(f0_path))
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            return f0_gt, y

    # Fallback to parselmouth (more consistent with the rest of the pipeline)
    print("⚠️ Falling back to parselmouth for GT F0")
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    # Calculate the expected length in frames
    length = int(len(y) / HOP_LENGTH) + 1

    # Use get_pitch_parselmouth from binarizer_utils
    from utils.binarizer_utils import get_pitch_parselmouth
    f0, uv = get_pitch_parselmouth(
        y, sr, length,
        hop_size=HOP_LENGTH,
        f0_min=hparams['fmin'],  # Match the fmin from the original librosa.pyin call
        f0_max=hparams['fmax'],  # Match the fmax from the original librosa.pyin call
        interp_uv=False  # Keep the same behavior as the original
    )

    return f0, y


def load_textgrid_phones(textgrid_path):
    """Load phoneme sequence and boundaries from TextGrid"""
    tg = TextGrid.fromFile(textgrid_path)
    phones_tier = tg.getFirst("phones")
    phones = []
    boundaries = [0.0]
    for interval in phones_tier:
        if interval.mark != "":
            phones.append(interval.mark)
            boundaries.append(interval.maxTime)
    return phones, np.array(boundaries)


def align_f0_with_mel(f0, mel_shape, method='interpolation'):
    """
    Align F0 to the length of mel spectrogram
    Args:
        f0: Input F0 sequence
        mel_shape: Target mel spectrogram shape (n_mels, T)
        method: Alignment method ('interpolation' or 'padding')
    """
    target_len = mel_shape[1]  # Time steps of mel spectrogram
    current_len = len(f0)

    if current_len == target_len:
        return f0
    
    if method == 'interpolation':
        # Use interpolation for alignment
        x_old = np.linspace(0, 1, current_len)
        x_new = np.linspace(0, 1, target_len)
        f0_aligned = np.interp(x_new, x_old, f0)
    elif method == 'padding':
        # Use padding or truncation for alignment
        if current_len < target_len:
            # Padding
            f0_aligned = np.pad(f0, (0, target_len - current_len), mode='edge')
        else:
            # Truncation
            f0_aligned = f0[:target_len]
    
    return f0_aligned


# === Modification 1: Add boundaries parameter to function signature ===
def plot_f0_comparison(gt_f0, npz_f0, wav_path, boundaries, output_path):
    """Plot F0 comparison chart (with phoneme boundaries)"""
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    t_audio = np.arange(len(y)) / sr
    # Interpolate to audio time axis
    t_gt = np.arange(len(gt_f0)) * HOP_LENGTH / sr
    t_npz = np.arange(len(npz_f0)) * HOP_LENGTH / sr
    gt_f0_interp = interp1d(t_gt, gt_f0, kind='linear', fill_value=0, bounds_error=False)(t_audio)
    npz_f0_interp = interp1d(t_npz, npz_f0, kind='linear', fill_value=0, bounds_error=False)(t_audio)

    plt.figure(figsize=(20, 6))
    plt.plot(t_audio, gt_f0_interp, 'b-', alpha=0.7, linewidth=1.5, label='Ground Truth (pyin)')
    plt.plot(t_audio, npz_f0_interp, 'r--', alpha=0.8, linewidth=1.5, label='From .npz')

    # === Modification 2: Directly use passed boundaries to draw phoneme boundaries ===
    # Add phoneme boundaries
    if boundaries is not None:
        for b in boundaries:
            plt.axvline(x=b, color='g', linestyle=':', alpha=0.5)

    plt.ylim(0, 1200)
    plt.xlabel('Time (s)')
    plt.ylabel('F0 (Hz)')
    plt.title('F0 Curve Validation: Ground Truth vs .npz')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# === [New] Modify plot_mel_spectrogram_comparison function to include error_mel ===
def plot_mel_spectrogram_comparison(gt_mel, npz_mel, rec_mel=None, error_mel=None, output_path=None):
    """Plot mel spectrogram comparison (supports reconstructed spectrum and error spectrum)"""
    n_rows = 4 if error_mel is not None else (3 if rec_mel is not None else 2)
    fig, axes = plt.subplots(n_rows, 1, figsize=(20, 4 * n_rows), sharex=True)

    if n_rows == 2:
        axes = [axes[0], axes[1], None, None]
    elif n_rows == 3:
        axes = [axes[0], axes[1], axes[2], None]

    # Ground Truth Mel
    im1 = axes[0].imshow(gt_mel, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title('Ground Truth Mel Spectrogram')
    axes[0].set_ylabel('Mel Frequency')

    # From .npz Mel
    im2 = axes[1].imshow(npz_mel, aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title('From .npz Mel Spectrogram')
    axes[1].set_ylabel('Mel Frequency')

    # Reconstructed Mel (if available)
    if rec_mel is not None:
        im3 = axes[2].imshow(rec_mel, aspect='auto', origin='lower', cmap='magma')
        axes[2].set_title('Reconstructed Mel Spectrogram')
        axes[2].set_ylabel('Mel Frequency')

    # Error Mel (if available)
    if error_mel is not None:
        im4 = axes[3].imshow(error_mel, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-20, vmax=20)
        axes[3].set_title('Error Mel (Rec - GT)')
        axes[3].set_ylabel('Mel Frequency')
        axes[3].set_xlabel('Frames')
        fig.colorbar(im4, ax=axes[3], shrink=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def calculate_similarity_metrics(gt_f0, npz_f0, gt_mel, npz_mel, rec_mel=None):
    """Calculate objective similarity metrics"""
    # F0 similarity (voiced segments only)
    # Align lengths
    min_len_f0 = min(len(gt_f0), len(npz_f0))
    gt_f0_aligned = gt_f0[:min_len_f0]
    npz_f0_aligned = npz_f0[:min_len_f0]

    voiced_mask = (gt_f0_aligned > 0) & (npz_f0_aligned > 0)
    if np.sum(voiced_mask) > 0:
        f0_corr = np.corrcoef(gt_f0_aligned[voiced_mask], npz_f0_aligned[voiced_mask])[0, 1]
        f0_mae = np.mean(np.abs(gt_f0_aligned[voiced_mask] - npz_f0_aligned[voiced_mask]))
    else:
        f0_corr = 0.0
        f0_mae = float('inf')

    # Mel spectrogram similarity - need to handle different shapes
    # Ensure both spectrograms have the same shape for comparison
    min_time_steps = min(gt_mel.shape[1], npz_mel.shape[1])
    min_mels = min(gt_mel.shape[0], npz_mel.shape[0])

    # Trim both spectrograms to the same dimensions
    gt_mel_trimmed = gt_mel[:min_mels, :min_time_steps]
    npz_mel_trimmed = npz_mel[:min_mels, :min_time_steps]

    mel_rmse = np.sqrt(np.mean((gt_mel_trimmed - npz_mel_trimmed) ** 2))

    metrics = {
        'f0_correlation': float(f0_corr),
        'f0_mae_hz': float(f0_mae),
        'mel_rmse': float(mel_rmse)
    }

    # Reconstructed spectrum metrics
    if rec_mel is not None:
        # Important: Align reconstructed spectrum dimensions with ground truth
        min_time_steps_rec = min(gt_mel.shape[1], rec_mel.shape[1])
        min_mels_rec = min(gt_mel.shape[0], rec_mel.shape[0])

        gt_mel_rec_trimmed = gt_mel[:min_mels_rec, :min_time_steps_rec]
        rec_mel_trimmed = rec_mel[:min_mels_rec, :min_time_steps_rec]

        # Calculate RMSE of reconstructed mel spectrogram
        rec_mel_rmse = np.sqrt(np.mean((gt_mel_rec_trimmed - rec_mel_trimmed) ** 2))
        
        # Add additional metrics to evaluate reconstruction quality
        # Calculate spectral distortion rate (SDR-like measure)
        signal_power = np.mean(gt_mel_rec_trimmed ** 2)
        noise_power = np.mean((gt_mel_rec_trimmed - rec_mel_trimmed) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))  # Add small constant to prevent division by zero
        
        metrics['rec_mel_rmse'] = float(rec_mel_rmse)
        metrics['rec_snr'] = float(snr)

    return metrics


def audio_to_base64(wav_path):
    """Convert audio file to base64 (for HTML embedding)"""
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()
    return base64.b64encode(audio_bytes).decode('utf-8')


def array_to_wav_base64(wav_array, sr=SAMPLE_RATE):
    """Convert numpy audio array to base64 WAV"""
    import io
    from scipy.io.wavfile import write
    buffer = io.BytesIO()
    # Ensure audio is within [-1, 1] range
    wav_int16 = (np.clip(wav_array, -1.0, 1.0) * 32767).astype(np.int16)
    write(buffer, sr, wav_int16)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def generate_html_report(report_data, output_path):
    """Generate interactive HTML report"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DiffSinger .npz Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            audio {{ width: 100%; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>DiffSinger .npz Validation Report</h1>
        <h2>File: {filename}</h2>
        <div class="metric">
            <h3>F0 Correlation: {f0_corr:.3f}</h3>
            <h3>F0 MAE: {f0_mae:.2f} Hz</h3>
            <h3>Mel RMSE (.npz): {mel_rmse:.4f}</h3>
            {rec_mel_rmse_line}
            {rec_snr_line}
        </div>
        <h3>Original Dry Vocal</h3>
        <audio controls>
            <source src="data:audio/wav;base64,{orig_audio}" type="audio/wav">
        </audio>
        {reconstructed_audio_block}
        <h3>F0 Comparison</h3>
        <img src="{f0_plot}" alt="F0 Comparison">
        <h3>Mel Spectrogram Comparison</h3>
        <img src="{mel_plot}" alt="Mel Comparison">
    </body>
    </html>
    """

    # Read image as base64
    def img_to_base64(img_path):
        with open(img_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode('utf-8')

    # Reconstructed audio block
    rec_audio_block = ""
    rec_mel_rmse_line = ""
    rec_snr_line = ""
    if report_data.get('rec_audio_b64'):
        rec_audio_block = f"""
        <h3>Reconstructed Audio (from .npz)</h3>
        <audio controls>
            <source src="data:audio/wav;base64,{report_data['rec_audio_b64']}" type="audio/wav">
        </audio>
        """
        rec_mel_rmse_line = f"<h3>Reconstructed Mel RMSE: {report_data['metrics']['rec_mel_rmse']:.4f}</h3>"
        rec_snr_line = f"<h3>Reconstruction SNR: {report_data['metrics']['rec_snr']:.2f} dB</h3>"

    html_content = html_template.format(
        filename=report_data['filename'],
        f0_corr=report_data['metrics']['f0_correlation'],
        f0_mae=report_data['metrics']['f0_mae_hz'],
        mel_rmse=report_data['metrics']['mel_rmse'],
        rec_mel_rmse_line=rec_mel_rmse_line,
        rec_snr_line=rec_snr_line,
        orig_audio=report_data['orig_audio_b64'],
        reconstructed_audio_block=rec_audio_block,
        f0_plot=img_to_base64(report_data['f0_plot']),
        mel_plot=img_to_base64(report_data['mel_plot'])
    )

    with open(output_path, 'w') as f:
        f.write(html_content)


def main():
    out_path = os.path.join(root_dir, "data", "validate")
    parser = argparse.ArgumentParser(description="Validate DiffSinger .npz data against original dry vocal.")
    parser.add_argument("--wav", required=True, help="Path to original dry vocal WAV")
    parser.add_argument("--textgrid", required=True, help="Path to corresponding TextGrid file")
    parser.add_argument("--npz", required=True, help="Path to DiffSinger .npz file")
    parser.add_argument("--config", type=str, default="configs/acoustic_singer_config.yaml",
                        help="Path to the configuration file for the acoustic model (default: configs/acoustic_singer_config.yaml)")
    parser.add_argument("--f0_dir", type=str, help="Directory containing .f0.npy files for GT F0")

    # Vocoder parameters (only required when module is available)
    if VOCODER_AVAILABLE:
        parser.add_argument("--vocoder_class", type=str, help="Specify vocoder class (optional)")
        parser.add_argument("--vocoder_ckpt", type=str, help="Specify vocoder checkpoint path (optional)")
    parser.add_argument("--output_dir", default=out_path, help="Output directory for report")

    args = parser.parse_args()

    # Load hparams with the specified config
    config_path = os.path.join(root_dir, args.config)
    if not os.path.exists(config_path):
        print(f"⚠️ Config file not found: {config_path}")
        config_path = os.path.join(root_dir, "configs/variance_singer_config.yaml")  # fallback
        print(f"⚠️ Using fallback config: {config_path}")
    set_hparams(config=config_path)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 1. Load data
    print("Loading .npz data...")
    npz_data = load_npz_data(args.npz)
    print("Extracting ground truth F0...")
    gt_f0, y = extract_ground_truth_f0(args.wav, f0_dir=getattr(args, 'f0_dir', None))

    # === Modification 3: Load phones and boundaries, and use it immediately ===
    print("Loading TextGrid phones...")
    phones, boundaries = load_textgrid_phones(args.textgrid)

    # 2. Reconstruct Ground Truth Mel
    print("Reconstructing ground truth Mel...")
    # === Modification: Use get_mel_torch function, consistent with val_nsf_hifigan.py ===
    from utils.binarizer_utils import get_mel_torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt_mel = get_mel_torch(
        y, SAMPLE_RATE,
        num_mel_bins=hparams['audio_num_mel_bins'],
        hop_size=hparams['hop_size'],
        win_size=hparams['win_size'],
        fft_size=hparams['fft_size'],
        fmin=hparams['fmin'],
        fmax=hparams['fmax'],
        device=device
    )

    # [New] Align F0 from npz to mel spectrogram length, solving length misalignment issues
    print(f"Original F0 length: {len(npz_data['f0'])}, Mel spectrogram time steps: {npz_data['mel'].shape[1]}")
    npz_f0_aligned = align_f0_with_mel(npz_data['f0'], npz_data['mel'].shape)
    print(f"Aligned F0 length: {len(npz_f0_aligned)}, Mel spectrogram time steps: {npz_data['mel'].shape[1]}")

    # 3. Reconstruct audio (if vocoder is available)
    rec_wav = None
    rec_mel = None
    if VOCODER_AVAILABLE:
        print("Reconstructing audio from .npz using DiffSinger vocoder...")
        try:
            # Use the same config that was loaded for the main script
            vocoder = load_vocoder(
                # :vocoder_class=args.vocoder_class,
                vocoder_ckpt=args.vocoder_ckpt,
                config_path=args.config  # Pass the config path to ensure correct parameters
            )
            
            # Use aligned F0 for reconstruction
            rec_wav = vocoder.reconstruct(npz_data['mel'], npz_f0_aligned)

            # Calculate mel spectrogram of reconstructed audio
            # === Modification: Use get_mel_torch function, consistent with val_nsf_hifigan.py ===
            rec_mel = get_mel_torch(
                rec_wav, SAMPLE_RATE,
                num_mel_bins=hparams['audio_num_mel_bins'],
                hop_size=hparams['hop_size'],
                win_size=hparams['win_size'],
                fft_size=hparams['fft_size'],
                fmin=hparams['fmin'],
                fmax=hparams['fmax'],
                device=device
            )

            # === [New] Calculate error_mel ===
            # Align time dimension
            min_T = min(gt_mel.shape[1], rec_mel.shape[1])
            gt_mel_aligned = gt_mel[:, :min_T]
            rec_mel_aligned = rec_mel[:, :min_T]
            error_mel = rec_mel_aligned - gt_mel_aligned

        except FileNotFoundError as e:
            print(f"❌ Vocoder checkpoint file not found: {e}")
            print("⚠️ Please ensure vocoder checkpoint exists at correct path")
            rec_wav = None
            rec_mel = None
            error_mel = None
        except Exception as e:
            print(f"❌ Reconstruction failed: {e}")
            import traceback
            traceback.print_exc()
            rec_wav = None
            rec_mel = None
            error_mel = None
    else:
        print("⏭️ Skipping reconstruction step (vocoder unavailable)")
        error_mel = None

    # 4. Draw comparison charts
    f0_plot_path = plots_dir / "f0_comparison.png"
    # === Modification 4: Call plot_f0_comparison with boundaries ===
    plot_f0_comparison(gt_f0, npz_f0_aligned, args.wav, boundaries, str(f0_plot_path))

    mel_plot_path = plots_dir / "mel_comparison.png"
    # === [New] Pass error_mel when calling ===
    plot_mel_spectrogram_comparison(gt_mel, npz_data['mel'], rec_mel, error_mel, str(mel_plot_path))

    # 5. Calculate similarity metrics
    metrics = calculate_similarity_metrics(gt_f0, npz_f0_aligned, gt_mel, npz_data['mel'], rec_mel)

    # 6. Generate HTML report
    report_data = {
        'filename': Path(args.npz).name,
        'metrics': metrics,
        'orig_audio_b64': audio_to_base64(args.wav),
        'f0_plot': str(f0_plot_path),
        'mel_plot': str(mel_plot_path)
    }
    if rec_wav is not None:
        report_data['rec_audio_b64'] = array_to_wav_base64(rec_wav)

    report_path = output_dir / "validation_report.html"
    generate_html_report(report_data, str(report_path))

    # 7. Print summary
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE!")
    print(f"F0 Correlation: {metrics['f0_correlation']:.3f} (ideal value > 0.95)")
    print(f"F0 MAE: {metrics['f0_mae_hz']:.2f} Hz (ideal value < 20)")
    print(f"Mel RMSE (.npz): {metrics['mel_rmse']:.4f}")
    if 'rec_mel_rmse' in metrics:
        print(f"Reconstructed Mel RMSE: {metrics['rec_mel_rmse']:.4f} (ideal value < 0.3)")
        print(f"Reconstruction SNR: {metrics['rec_snr']:.2f} dB")
        print("⚠️  High Reconstructed Mel RMSE explanation:")
        print("   - Vocoder doesn't fully reproduce GT, possibly due to f0 length misalignment causing phase shifts")
        print("   - Check F0 and mel frame alignment")
        print("   - Verify if vocoder model is well trained")
    else:
        print("(Reconstruction not performed, no reconstruction metrics)")
    print(f"\nReport saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    # Temporarily import torch (to avoid early errors in non-GPU environments)
    try:
        import torch
    except ImportError:
        if VOCODER_AVAILABLE:
            raise ImportError("Vocoder requires PyTorch, but it's not installed.")
    main()