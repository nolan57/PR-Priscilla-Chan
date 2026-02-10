#!/usr/bin/env python3
"""
全面诊断 .npz 文件中的 mel 和 f0 是否适合 NSF-HiFiGAN 声码器。
"""

import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt

# ------------------ 配置 ------------------
SAMPLE_RATE = 44100
HOP_LENGTH = 512
N_MELS = 128  # DiffSinger 默认
FMIN = 40
FMAX = 16000

# NSF-HiFiGAN 期望的配置（来自 DiffSinger 官方预处理）


def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    required_keys = ['mel', 'f0']
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {npz_path}")
    return data['mel'], data['f0']


def validate_shapes(mel, f0):
    print("🔍 [1/5] 验证形状...")
    print(f"  Mel shape: {mel.shape}")
    print(f"  F0 shape: {f0.shape}")

    if len(mel.shape) != 2:
        raise ValueError("Mel 应为 2D (T, n_mels)")
    if len(f0.shape) != 1:
        raise ValueError("F0 应为 1D (T,)")

    T_mel = mel.shape[0]
    T_f0 = len(f0)
    if T_mel != T_f0:
        raise ValueError(f"Mel ({T_mel}) 和 F0 ({T_f0}) 帧数不一致！")

    if mel.shape[1] != N_MELS:
        print(f"  ⚠️ 警告：n_mels={mel.shape[1]}，但预期为 {N_MELS}（可能兼容，但需注意）")
    print("  ✅ 形状验证通过\n")


def validate_mel_range(mel):
    print("🔍 [2/5] 验证 Mel 范围（应为 log-mel）...")
    mel_min = mel.min()
    mel_max = mel.max()
    print(f"  Mel min: {mel_min:.3f}")
    print(f"  Mel max: {mel_max:.3f}")

    # 检查是否为 log-mel（值应 ≤ 0）
    if mel_max > 0.0:
        raise ValueError(f"Mel 最大值过高 ({mel_max:.3f})！log-mel 值应 ≤ 0。您可能忘了取对数。")
    
    # 检查最小值是否合理（避免数值问题）
    if mel_min < -30.0:
        print("  ⚠️ 警告：Mel 最小值过低 —— 可能存在数值问题")
    else:
        print("  ✅ Mel 范围正常（log-mel）")
    print()


def validate_f0_range(f0):
    print("🔍 [3/5] 验证 F0 范围（应为 Hz）...")
    f0_voiced = f0[f0 > 0]
    if len(f0_voiced) == 0:
        print("  ⚠️ 警告：F0 全为 0（无声段）")
        return

    f0_min = f0_voiced.min()
    f0_max = f0_voiced.max()
    print(f"  F0 voiced range: {f0_min:.1f} ～ {f0_max:.1f} Hz")

    if f0_max > FMAX * 1.2:
        raise ValueError(f"F0 最大值过高 ({f0_max:.1f} Hz)！人类歌声通常 ≤ 1600 Hz。您可能单位错误（如 MIDI）。")
    if f0_min < FMIN * 0.8:
        print("  ⚠️ 警告：F0 最小值偏低（可能包含低频噪声）")

    if FMIN <= f0_max <= FMAX:
        print("  ✅ F0 范围合理")
    else:
        print("  ❌ F0 范围可疑！")
    print()


def validate_energy_alignment(mel, f0, wav_path=None):
    print("🔍 [4/5] 验证能量与 F0 对齐...")
    energy = np.linalg.norm(mel, axis=1)
    voiced = f0 > 0

    # 统计：高能量区域是否 voiced
    high_energy = energy > np.percentile(energy, 70)
    energy_voiced_ratio = np.mean(voiced[high_energy])
    print(f"  高能量帧中 voiced 比例: {energy_voiced_ratio:.1%}")

    if energy_voiced_ratio < 0.8:
        print("  ⚠️ 警告：高能量区域很多 unvoiced —— 可能 VAD 或 F0 提取有问题")
    else:
        print("  ✅ 能量与 F0 对齐良好")
    print()

    # 可选：与原始音频对比
    if wav_path:
        print("🔍 [5/5] 与原始音频对比（可选）...")
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        expected_frames = len(y) // HOP_LENGTH + 1
        if len(f0) != expected_frames:
            print(f"  ⚠️ 警告：F0 帧数 ({len(f0)}) ≠ 音频预期帧数 ({expected_frames})")
        else:
            print("  ✅ F0 帧数与音频长度匹配")
        print()


def plot_diagnostics(mel, f0, output_plot=None):
    print("📊 生成诊断图...")
    T = len(f0)
    time_sec = np.arange(T) * HOP_LENGTH / SAMPLE_RATE

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Mel
    im = axes[0].imshow(mel.T, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_ylabel('Mel Channel')
    axes[0].set_title('Mel Spectrogram (from .npz)')
    plt.colorbar(im, ax=axes[0])

    # F0
    axes[1].plot(time_sec, f0, color='red')
    axes[1].set_ylabel('F0 (Hz)')
    axes[1].set_title('F0 Contour')

    # Energy
    energy = np.linalg.norm(mel, axis=1)
    axes[2].plot(time_sec, energy, label='Mel Norm', color='blue')
    axes[2].plot(time_sec, voiced := (f0 > 0).astype(float) * energy.max() * 0.9,
                 label='Voiced (F0>0)', color='green', alpha=0.7)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Energy')
    axes[2].legend()
    axes[2].set_title('Energy vs Voiced Regions')

    plt.tight_layout()
    if output_plot:
        plt.savefig(output_plot, dpi=150)
        print(f"  保存诊断图至: {output_plot}")
    else:
        plt.show()
    print()


def main(npz_path, wav_path=None, plot_path=None):
    print(f"🧪 诊断 .npz 文件: {npz_path}\n")

    try:
        mel, f0 = load_npz(npz_path)
        mel = mel.T
        print(f"✅ 成功加载 mel ({mel.shape}) 和 f0 ({f0.shape})\n")

        validate_shapes(mel, f0)
        validate_mel_range(mel)
        validate_f0_range(f0)
        validate_energy_alignment(mel, f0, wav_path)

        plot_diagnostics(mel, f0, plot_path)

        print("🎉 所有检查完成！如果无报错，数据应可被 NSF-HiFiGAN 正确使用。")

    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="诊断 .npz 中的 mel/f0 是否适合 NSF-HiFiGAN")
    parser.add_argument("npz_path", type=str, help=".npz 文件路径")
    parser.add_argument("--wav", type=str, default=None, help="原始 WAV 路径（用于帧数验证）")
    parser.add_argument("--plot", type=str, default=None, help="保存诊断图的路径（如 diag.png）")
    args = parser.parse_args()

    main(args.npz_path, args.wav, args.plot)