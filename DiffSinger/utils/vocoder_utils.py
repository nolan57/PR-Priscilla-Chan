# vocoder_utils.py
"""
DiffSinger native vocoder toolkit
Encapsulates the standard process for loading and using vocoders from the DiffSinger project
"""

import os
import sys
from pathlib import Path
import torch

from utils.hparams import hparams


class DiffSingerVocoder:
    def __init__(self, vocoder_ckpt=None, vocoder_class=None, config_path="configs/acoustic_singer_config.yaml"):
        """
        Initialize DiffSinger vocoder

        Args:
            vocoder_ckpt (str): Vocoder checkpoint path
            vocoder_class (str): Vocoder class name, e.g. 'NsfHifiGAN'
            config_path (str): Configuration file path (default: acoustic_singer_config.yaml)
        """
        import os
        from pathlib import Path

        # Determine the root directory (DiffSinger/)
        root_dir = Path(__file__).parent.parent.resolve()

        # Construct full config path
        full_config_path = os.path.join(root_dir, config_path)

        # Load the acoustic configuration if it exists
        if os.path.exists(full_config_path):
            from utils.hparams import set_hparams
            set_hparams(config=full_config_path)

        # Override parameters
        if vocoder_class:
            hparams['vocoder'] = vocoder_class
        if vocoder_ckpt:
            hparams['vocoder_ckpt'] = vocoder_ckpt

        # Ensure the vocoder checkpoint path is resolved relative to the project root if it's a relative path
        if hparams.get('vocoder_ckpt'):
            import os
            from pathlib import Path
            ckpt_path = hparams['vocoder_ckpt']
            if not os.path.isabs(ckpt_path):
                # Resolve relative path from the project root (DiffSinger directory)
                project_root = Path(__file__).parent.parent.resolve()
                full_ckpt_path = project_root / ckpt_path
                hparams['vocoder_ckpt'] = str(full_ckpt_path)

        # Load the actual vocoder (not the acoustic model)
        from modules.vocoders.registry import get_vocoder_cls
        vocoder_cls = get_vocoder_cls(hparams)
        self.vocoder = vocoder_cls()
        self.device = self.vocoder.device

    def reconstruct(self, mel, f0):
        """
        Reconstruct audio

        Args:
            mel (np.ndarray): Mel spectrogram, shape [n_mel, T] (T, n_mel)
            f0 (np.ndarray): Fundamental frequency, shape [T]

        Returns:
            np.ndarray: Reconstructed audio waveform, shape [T_audio,]
        """
        # The vocoder expects mel spectrogram with shape [T, n_mel] and f0 with shape [T]
        # Convert numpy arrays to the expected format
        wav = self.vocoder.spec2wav(mel.T, f0=f0)  # Transpose mel from [n_mel, T] to [T, n_mel]
        return wav


# Convenience function
def load_vocoder(vocoder_ckpt=None, vocoder_class=None, config_path="configs/acoustic_singer_config.yaml"):
    """Convenience loading function, parameters same as DiffSingerVocoder.__init__"""
    return DiffSingerVocoder(vocoder_ckpt=vocoder_ckpt, vocoder_class=vocoder_class, config_path=config_path)