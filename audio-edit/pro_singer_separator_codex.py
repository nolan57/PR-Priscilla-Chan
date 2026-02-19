import sys
import os
import warnings
import tempfile
import subprocess
from datetime import datetime
import numpy as np
import soundfile as sf
from pathlib import Path
import torch
import librosa
import gc
from scipy.ndimage import gaussian_filter1d, binary_opening, binary_closing
import shutil  # For file operations

# --- SUPPRESS WARNINGS ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", message=".*Unknown device for graph fuser.*")
warnings.filterwarnings("ignore", message=".*mps.*fallback.*")


# --- DEVICE DETECTION ---
def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"


DEVICE = get_best_device()
print(f"Using device: {DEVICE}")

# --- LOCAL PATH DISCOVERY ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = SCRIPT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Model configuration simplified for local-only loading


def _find_uvr_root():
    """Locate a local UVR checkout inside this project."""
    default = PROJECT_ROOT / "ultimatevocalremovergui"
    if (default / "lib_v5").exists():
        return default

    for path in PROJECT_ROOT.rglob("ultimatevocalremovergui"):
        if path.is_dir() and (path / "lib_v5").exists():
            return path
    return None


UVR_ROOT = _find_uvr_root()
if UVR_ROOT is not None and str(UVR_ROOT) not in sys.path:
    sys.path.insert(0, str(UVR_ROOT))

# --- OFFLINE MODE SETUP ---
import huggingface_hub
import torchaudio

# Fix torchaudio compatibility issues
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
else:
    try:
        backends = torchaudio.list_audio_backends()
    except (AttributeError, TypeError, RuntimeError):
        def dummy_list_audio_backends():
            return ["soundfile"]


        torchaudio.list_audio_backends = dummy_list_audio_backends

_old_hf_hub_download = huggingface_hub.hf_hub_download


def _patched_hf_hub_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token", None)
    kwargs["local_files_only"] = True
    return _old_hf_hub_download(*args, **kwargs)


huggingface_hub.hf_hub_download = _patched_hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- PYQT6 IMPORTS ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QMessageBox, QGroupBox, QSlider,
    QScrollArea, QStatusBar, QComboBox, QTextEdit,
    QSplitter, QSpinBox, QTabWidget, QCheckBox, QLineEdit, QStyle,
    QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QColor

# --- UVR COMPONENT INTEGRATION ---
UVR_AVAILABLE = False
UVR_MODULES = {}

try:
    # Try to import UVR modules individually
    try:
        from lib_v5.spec_utils import (
            wave_to_spectrogram, spectrogram_to_wave,
            normalize, merge_artifacts, reduce_vocal_aggressively
        )

        UVR_MODULES['spec_utils'] = True
    except ImportError:
        UVR_MODULES['spec_utils'] = False

    try:
        from lib_v5.vr_network.nets import CascadedASPPNet, determine_model_capacity

        UVR_MODULES['vr_network'] = True
    except ImportError:
        UVR_MODULES['vr_network'] = False

    try:
        from lib_v5.vr_network.model_param_init import ModelParameters

        UVR_MODULES['model_params'] = True
    except ImportError:
        UVR_MODULES['model_params'] = False


        class ModelParameters:
            def __init__(self, config_path='', device='cpu'):
                self.param = {
                    'n_fft': 2048, 'hl': 512, 'nn_architecture': 123821,
                    'model_state_dict': {}, 'mid_side': False, 'mid_side_b2': False,
                    'stereo_w': False, 'stereo_n': False, 'reverse': False
                }

    try:
        from lib_v5.mdxnet import ConvTDFNet

        UVR_MODULES['mdxnet'] = True
    except ImportError:
        UVR_MODULES['mdxnet'] = False

    try:
        from core.models import ModelData

        UVR_MODULES['models'] = True
    except ImportError:
        UVR_MODULES['models'] = False


        class ModelData:
            def __init__(self, model_name="fallback", config=None):
                self.config = config or {}
                self.model_path = ""
                self.process_method = 'demucs'

    try:
        from core.config import ConfigManager

        UVR_MODULES['config'] = True
    except ImportError:
        UVR_MODULES['config'] = False


        class ConfigManager:
            def __init__(self):
                self.config = {'device_set': 'cuda' if torch.cuda.is_available() else 'cpu'}

    UVR_AVAILABLE = any([UVR_MODULES['spec_utils'], UVR_MODULES['vr_network'], UVR_MODULES['mdxnet']])

except Exception as e:
    print(f"⚠ UVR components import failed: {e}")
    UVR_AVAILABLE = False

# --- HYBRID VOICE SEPARATION IMPORTS ---
try:
    if UVR_ROOT is not None and str(UVR_ROOT) not in sys.path:
        sys.path.insert(0, str(UVR_ROOT))
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from speechbrain.inference import SpeakerRecognition

    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

SAMPLE_RATE = 44100


def clear_gpu_cache():
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


# Download workers removed - only local model loading supported


class SeparationWorker(QThread):
    finished = pyqtSignal(object, str, object)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, separator, mixed_audio_path, reference_segments):
        super().__init__()
        self.separator = separator
        self.mixed_audio_path = mixed_audio_path
        self.reference_segments = reference_segments
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True
        clear_gpu_cache()

    def run(self):
        try:
            if self._is_cancelled: return
            result, mask = self.separator.separate_target_voice(
                self.mixed_audio_path,
                self.reference_segments,
                progress_callback=self.progress.emit,
            )
            if not self._is_cancelled:
                clear_gpu_cache()
                self.finished.emit(result, "Separation completed successfully", mask)
        except Exception as e:
            if not self._is_cancelled:
                clear_gpu_cache()
                self.error.emit(str(e))


# All download workers removed - only local model loading supported


# --- SEPARATOR CLASSES ---

class UVRSeparator:
    def __init__(self, device=DEVICE):
        self.device = device
        self.model = None
        self.model_params = None
        self.config_manager = ConfigManager() if UVR_MODULES.get('config', False) else None

    def load_vr_model(self, model_path):
        """Load VR (Vocal Remover) model using UVR's architecture"""
        if not UVR_MODULES.get('vr_network', False):
            print("VR network not available")
            return False

        try:
            # --- FIX: Removed self.device argument ---
            # Real UVR ModelParameters only takes the model path
            self.model_params = ModelParameters(model_path)

            # Determine model capacity and create architecture
            nn_architecture = self.model_params.param['nn_architecture']
            n_fft_bins = self.model_params.param['n_fft'] // 2 + 1

            if UVR_MODULES.get('vr_network', False):
                model_capacity_data = determine_model_capacity(n_fft_bins, nn_architecture)

                # Create cascaded ASPP network (UVR's main architecture)
                self.model = CascadedASPPNet(
                    n_fft=self.model_params.param['n_fft'],
                    model_capacity_data=model_capacity_data,
                    nn_architecture=nn_architecture
                )

                # Load weights
                self.model.load_state_dict(self.model_params.param['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()

                print(f"✓ VR model loaded: {Path(model_path).name}")
                return True
            else:
                print("VR network components not available")
                return False

        except Exception as e:
            print(f"Failed to load VR model: {e}")
            return False

    def load_mdx_model(self, model_path):
        """Load MDX-Net model using UVR's architecture"""
        if not UVR_MODULES.get('mdxnet', False):
            print("MDX-Net not available")
            return False

        try:
            # Load MDX model configuration
            if UVR_MODULES.get('models', False):
                model_data = ModelData(model_name="MDX",
                                       config=self.config_manager.config if self.config_manager else {})
            else:
                model_data = ModelData(model_name="MDX")
            model_data.model_path = model_path
            model_data.process_method = 'mdx'

            # Create MDX network
            if UVR_MODULES.get('mdxnet', False):
                self.model = ConvTDFNet(
                    target_name='vocals',
                    lr=0.001,
                    optimizer='adamw',
                    dim_c=2,
                    dim_f=2048,
                    dim_t=256,
                    n_fft=2048,
                    hop_length=512,
                    overlap=0.25,
                    num_blocks=12,
                    l=3,
                    g=48,
                    k=3,
                    bn=64,
                    bias=True
                )

                # Load weights
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)

                self.model.to(self.device)
                self.model.eval()

                print(f"✓ MDX model loaded: {Path(model_path).name}")
                return True
            else:
                print("MDX-Net components not available")
                return False

        except Exception as e:
            print(f"Failed to load MDX model: {e}")
            return False

    def separate_vocals(self, audio_path, progress_callback=None):
        """Separate vocals using UVR's proven pipeline"""
        if not self.model:
            raise ValueError("No UVR model loaded")

        if progress_callback:
            progress_callback("Loading audio...")

        # Load and preprocess audio using UVR methods
        waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=False)

        if waveform.ndim == 1:
            waveform = np.asfortranarray([waveform, waveform])

        # Normalize using UVR method if available
        if UVR_MODULES.get('spec_utils', False):
            waveform = normalize(waveform, is_normalize=True)
        else:
            # Fallback normalization
            max_amp = np.max(np.abs(waveform))
            if max_amp > 1.0:
                waveform = waveform / max_amp

        if progress_callback:
            progress_callback("Converting to spectrogram...")

        # Convert to spectrogram using UVR's method if available
        if self.model_params and UVR_MODULES.get('spec_utils', False):
            spec = wave_to_spectrogram(
                wave=waveform,
                hop_length=self.model_params.param['hl'],
                n_fft=self.model_params.param['n_fft'],
                mp=self.model_params,
                band=1,
                is_v51_model=True
            )
        else:
            # Fallback spectrogram conversion
            spec_left = librosa.stft(waveform[0], n_fft=2048, hop_length=512)
            spec_right = librosa.stft(waveform[1], n_fft=2048, hop_length=512)
            spec = np.asfortranarray([spec_left, spec_right])

        if progress_callback:
            progress_callback("Running inference...")

        # Run inference
        with torch.no_grad():
            spec_tensor = torch.from_numpy(spec).to(self.device)

            if hasattr(self.model, 'predict_mask'):
                # VR model
                mask = self.model.predict_mask(spec_tensor.unsqueeze(0))
                mask = mask.squeeze(0).cpu().numpy()
            else:
                # MDX model or other
                mask = self.model(spec_tensor.unsqueeze(0))
                if isinstance(mask, tuple):
                    mask = mask[0]
                mask = mask.squeeze(0).cpu().numpy()

        if progress_callback:
            progress_callback("Applying mask and converting back...")

        # Apply mask to extract vocals
        vocal_spec = spec * mask

        # Convert back to waveform using UVR's method if available
        if self.model_params and UVR_MODULES.get('spec_utils', False):
            vocal_waveform = spectrogram_to_wave(
                vocal_spec,
                hop_length=self.model_params.param['hl'],
                mp=self.model_params,
                band=1,
                is_v51_model=True
            )
        else:
            # Fallback conversion
            vocal_left = librosa.istft(vocal_spec[0], hop_length=512)
            vocal_right = librosa.istft(vocal_spec[1], hop_length=512)
            vocal_waveform = np.asfortranarray([vocal_left, vocal_right])

        # Return mono for compatibility
        return np.mean(vocal_waveform, axis=0)

class HybridVoiceSeparator:
    def __init__(self, device=DEVICE):
        self.device = device
        self.uvr_separator = UVRSeparator(device)
        self.demucs_model = None
        self.speaker_model = None
        self.whisper_model = None
        self.embedding_sample_rate = 16000
        self.similarity_threshold = 0.72
        self._load_fallback_models()

    def _load_fallback_models(self):
        # Fallback Demucs
        if DEMUCS_AVAILABLE:
            try:
                self.demucs_model = get_model("htdemucs_ft", repo=None)
            except:
                pass

        # Speaker Recognition
        if SPEECHBRAIN_AVAILABLE:
            try:
                candidate_dirs = [
                    SCRIPT_DIR / "models" / "embedding_model",
                    PROJECT_ROOT / "models" / "embedding_model",
                ]
                for model_dir in candidate_dirs:
                    if model_dir.exists():
                        self.speaker_model = SpeakerRecognition.from_hparams(
                            source=str(model_dir),
                            savedir=str(model_dir),
                            run_opts={"device": self.device},
                        )
                        break
            except Exception as e:
                print(f"Speaker model warning: {e}")

    def set_similarity_threshold(self, threshold):
        self.similarity_threshold = float(np.clip(threshold, 0.5, 0.95))

    def auto_load_best_model(self):
        """Automatically find and load the best available model for separation."""
        models_dir = Path("./models")
        if not models_dir.exists():
            models_dir = Path(__file__).parent / "models"
            
        # Priority order for model types
        model_patterns = [
            "*.onnx",  # UVR models
            "*.pth",   # PyTorch models
            "*.pt",    # PyTorch models
            "*.ckpt",  # Checkpoint files
        ]
        
        for pattern in model_patterns:
            for model_file in models_dir.rglob(pattern):
                # Skip if it's clearly not a UVR model
                if any(skip in model_file.name.lower() for skip in ['whisper', 'demucs', 'speaker']):
                    continue
                    
                # Try to load the model
                success = False
                if "MDX" in model_file.name or "mdx" in model_file.name:
                    success = self.uvr_separator.load_mdx_model(str(model_file))
                    if not success:  # Fallback try VR
                        success = self.uvr_separator.load_vr_model(str(model_file))
                else:
                    success = self.uvr_separator.load_vr_model(str(model_file))
                
                if success:
                    print(f"✓ Auto-loaded model: {model_file.name}")
                    return True
                    
        return False

    def separate_target_voice(self, mixed_audio_path, reference_segments, progress_callback=None):
        if not reference_segments:
            raise ValueError("Please add at least one reference segment before separation.")

        if progress_callback: progress_callback("Step 1: Loading models...")

        # Auto-load best available model if none loaded
        if self.uvr_separator.model is None:
            if not self.auto_load_best_model():
                if progress_callback: progress_callback("No UVR models found, trying fallback methods...")

        if progress_callback: progress_callback("Step 2: Vocal Separation...")

        # 1. Separation with automatically loaded model
        if self.uvr_separator.model is not None:
            clean_vocals = self.uvr_separator.separate_vocals(mixed_audio_path, progress_callback)
        elif self.demucs_model:
            waveform, sr = librosa.load(mixed_audio_path, sr=SAMPLE_RATE, mono=True)
            clean_vocals = self._demucs_separation(waveform, sr)
        else:
            waveform, sr = librosa.load(mixed_audio_path, sr=SAMPLE_RATE, mono=True)
            clean_vocals = self._spectral_subtraction(waveform)

        if progress_callback: progress_callback("Step 3: Analysis...")

        # 2. Speaker Verification & Masking
        target_profile = self._extract_target_profile(reference_segments)
        speaker_mask = self._create_enhanced_speaker_mask(clean_vocals, target_profile, progress_callback)

        # 3. Post Processing
        isolated_vocal = self._uvr_post_processing(clean_vocals, speaker_mask)

        return isolated_vocal, speaker_mask

    def _demucs_separation(self, waveform, sr):
        if self.demucs_model:
            tensor_waveform = torch.tensor(waveform).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                sources = apply_model(self.demucs_model, tensor_waveform.to(self.device))
                vocals = sources[0][3].cpu().numpy()
            return vocals.flatten()
        return waveform

    def _spectral_subtraction(self, audio):
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        noise_floor = np.mean(magnitude[:, :10], axis=1, keepdims=True)
        enhanced_mag = np.maximum(magnitude - noise_floor * 1.5, 0)
        return librosa.istft(enhanced_mag * np.exp(1j * np.angle(stft)), hop_length=512)

    def _resample_for_embedding(self, audio):
        if len(audio) == 0:
            return np.zeros(0, dtype=np.float32)
        if SAMPLE_RATE == self.embedding_sample_rate:
            return audio.astype(np.float32)
        return librosa.resample(
            audio.astype(np.float32),
            orig_sr=SAMPLE_RATE,
            target_sr=self.embedding_sample_rate,
        ).astype(np.float32)

    def _extract_speaker_embedding(self, audio):
        if self.speaker_model is None:
            return None

        audio = self._resample_for_embedding(audio)
        if len(audio) < int(0.35 * self.embedding_sample_rate):
            return None

        # Trim near-silence to improve embedding stability.
        audio, _ = librosa.effects.trim(audio, top_db=35)
        if len(audio) < int(0.25 * self.embedding_sample_rate):
            return None

        peak = np.max(np.abs(audio)) + 1e-8
        audio = np.clip(audio / peak, -1.0, 1.0)
        wav = torch.tensor(audio, dtype=torch.float32, device=self.device).unsqueeze(0)

        try:
            with torch.no_grad():
                embedding = self.speaker_model.encode_batch(wav).squeeze().detach().cpu().numpy()
            return embedding / (np.linalg.norm(embedding) + 1e-8)
        except Exception:
            return None

    def _extract_mfcc_profile(self, audio):
        if len(audio) < int(0.2 * SAMPLE_RATE):
            return None
        mfcc = librosa.feature.mfcc(y=audio.astype(np.float32), sr=SAMPLE_RATE, n_mfcc=20)
        if mfcc.size == 0:
            return None
        profile = np.concatenate(
            [np.mean(mfcc, axis=1), np.std(mfcc, axis=1)],
            axis=0,
        )
        return profile / (np.linalg.norm(profile) + 1e-8)

    def _extract_target_profile(self, reference_segments):
        embeddings = []
        mfcc_profiles = []

        for segment in reference_segments:
            emb = self._extract_speaker_embedding(segment)
            if emb is not None:
                embeddings.append(emb)

            mfcc_profile = self._extract_mfcc_profile(segment)
            if mfcc_profile is not None:
                mfcc_profiles.append(mfcc_profile)

        profile = {"mode": "none", "reference_count": len(reference_segments)}
        if embeddings:
            ref_matrix = np.stack(embeddings, axis=0)
            profile["mode"] = "speechbrain"
            profile["embeddings"] = ref_matrix
            profile["centroid"] = np.mean(ref_matrix, axis=0)
            profile["centroid"] /= np.linalg.norm(profile["centroid"]) + 1e-8

        if mfcc_profiles:
            mfcc_matrix = np.stack(mfcc_profiles, axis=0)
            profile["mfcc_centroid"] = np.mean(mfcc_matrix, axis=0)
            profile["mfcc_centroid"] /= np.linalg.norm(profile["mfcc_centroid"]) + 1e-8

            if profile["mode"] == "none":
                profile["mode"] = "mfcc"

        if profile["mode"] == "none":
            raise ValueError("Could not extract usable target profile from reference segments.")

        return profile

    def _create_enhanced_speaker_mask(self, vocals, target_profile, progress_callback=None):
        frame_length = int(1.0 * SAMPLE_RATE)
        hop_length = int(0.25 * SAMPLE_RATE)
        if len(vocals) < frame_length:
            return np.ones_like(vocals, dtype=np.float32)

        starts = list(range(0, len(vocals) - frame_length + 1, hop_length))
        if starts[-1] != len(vocals) - frame_length:
            starts.append(len(vocals) - frame_length)

        scores = np.zeros(len(starts), dtype=np.float32)
        energies = np.zeros(len(starts), dtype=np.float32)
        window = np.hanning(frame_length).astype(np.float32)

        for idx, start in enumerate(starts):
            if progress_callback and idx % max(1, len(starts) // 20) == 0:
                progress_callback(f"Step 3: Speaker matching {idx}/{len(starts)}")

            frame = vocals[start:start + frame_length].astype(np.float32)
            energies[idx] = np.sqrt(np.mean(np.square(frame)) + 1e-10)

            if target_profile["mode"] == "speechbrain":
                emb = self._extract_speaker_embedding(frame)
                if emb is None:
                    scores[idx] = 0.0
                    continue
                ref_embeddings = target_profile["embeddings"]
                similarity = np.max(ref_embeddings @ emb)
                scores[idx] = float(np.clip(similarity, -1.0, 1.0))
            else:
                mfcc_profile = self._extract_mfcc_profile(frame)
                if mfcc_profile is None:
                    scores[idx] = 0.0
                    continue
                similarity = float(np.dot(mfcc_profile, target_profile["mfcc_centroid"]))
                scores[idx] = float(np.clip(similarity, -1.0, 1.0))

        # Suppress low-energy frames before thresholding.
        energy_gate = np.percentile(energies, 20)
        voiced = energies >= max(energy_gate * 0.6, 1e-4)
        scores = np.where(voiced, scores, -1.0)

        # Convert similarity to a soft probability mask.
        slope = 14.0 if target_profile["mode"] == "speechbrain" else 9.0
        soft_scores = 1.0 / (1.0 + np.exp(-slope * (scores - self.similarity_threshold)))

        mask = np.zeros(len(vocals), dtype=np.float32)
        weight = np.zeros(len(vocals), dtype=np.float32)
        for idx, start in enumerate(starts):
            end = start + frame_length
            frame_mask = soft_scores[idx] * window
            mask[start:end] += frame_mask
            weight[start:end] += window

        mask = mask / (weight + 1e-8)
        hard_mask = mask > 0.5
        hard_mask = binary_opening(hard_mask, structure=np.ones(int(0.05 * SAMPLE_RATE)))
        hard_mask = binary_closing(hard_mask, structure=np.ones(int(0.08 * SAMPLE_RATE)))
        mask = np.where(hard_mask, np.maximum(mask, 0.65), mask * 0.2)
        mask = gaussian_filter1d(mask.astype(np.float32), sigma=3)
        return np.clip(mask, 0.0, 1.0)

    def _uvr_post_processing(self, vocals, mask):
        isolated = vocals * mask
        if UVR_MODULES.get('spec_utils', False):
            isolated = normalize(isolated, is_normalize=True)
        else:
            max_amp = np.max(np.abs(isolated))
            if max_amp > 1.0: isolated /= max_amp
        return isolated


# --- GUI WIDGETS ---

class WaveformDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_waveform = None
        self.processed_waveform = None
        self.duration = 0.0
        self.playback_position = -1.0
        self.setMinimumHeight(300)

    def set_waveforms(self, original, processed, duration):
        self.original_waveform = original
        self.processed_waveform = processed
        self.duration = duration
        self.update()

    def paintEvent(self, a0):
        painter = QPainter(self)
        # Use default widget background instead of black
        painter.fillRect(a0.rect(), self.palette().color(self.backgroundRole()))
        width, height = self.width(), self.height()
        mid = height // 2

        if self.original_waveform is not None:
            painter.setPen(QColor(0, 255, 255))
            self._draw_wave(painter, self.original_waveform, 0, mid)

        if self.processed_waveform is not None:
            painter.setPen(QColor(255, 100, 100))
            self._draw_wave(painter, self.processed_waveform, mid, height)

        if self.playback_position >= 0:
            px = int((self.playback_position / (self.duration or 1)) * width)
            painter.setPen(QColor(255, 255, 0))
            painter.drawLine(px, 0, px, height)

    def _draw_wave(self, painter, wave, y1, y2):
        h = y2 - y1
        mid = y1 + h // 2
        step = max(1, len(wave) // (self.width() * 2))
        sub = wave[::step]
        scale = (h / 2.2) / (np.max(np.abs(sub)) + 1e-6)

        for i in range(len(sub) - 1):
            if i >= self.width(): break
            painter.drawLine(i, mid - int(sub[i] * scale), i + 1, mid - int(sub[i + 1] * scale))


class ProSingerSeparatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        print("DEBUG: Starting initialization...")
        self.setWindowTitle("Professional Singer Voice Separator - Automatic Pipeline")
        self.resize(1600, 1200)

        print("DEBUG: Creating separator...")
        self.separator = HybridVoiceSeparator()
        self.mixed_audio_path = None
        self.reference_segments = []
        self.original_waveform = None
        self.processed_waveform = None
        self.latest_mask = None
        self.playback_process = None
        self.uvr_model_widgets = []
        self.other_model_widgets = []
        # model_status removed - using main status bar now

        print("DEBUG: Setting up timer...")
        self.cursor_timer = QTimer()
        self.cursor_timer.timeout.connect(self.update_cursor)

        print("DEBUG: Initializing UI...")
        self.init_ui()
        print("DEBUG: UI initialized!")
        # Pipeline status removed

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        self.main_tab = QWidget()
        tabs.addTab(self.main_tab, "Voice Separation")
        
        self.setup_main_tab(self.main_tab)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def setup_main_tab(self, parent):
        print("DEBUG: setup_main_tab started")
        layout = QVBoxLayout(parent)
        print("DEBUG: Main layout created")
        
        # Create horizontal splitter for main layout
        print("DEBUG: Creating main splitter")
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter)
        print("DEBUG: Main splitter created")
        
        # Left panel - Model Management (full height)
        print("DEBUG: Creating left panel for models")
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Model Management section (full left panel)
        print("DEBUG: Creating models group")
        models_group = QGroupBox("Model Management")
        models_layout = QVBoxLayout()
        
        # UVR Models section
        print("DEBUG: Creating UVR models section")
        uvr_label = QLabel("<b>UVR Models</b>")
        models_layout.addWidget(uvr_label)
        
        # UVR Models list
        self.uvr_model_widgets = []
        uvr_models = [
            {'name': 'Demucs_Models', 'description': 'Facebook Demucs source separation models'},
            {'name': 'MDX_Net_Models', 'description': 'MDX-Net neural network models'},
            {'name': 'VR_Models', 'description': 'Vocal Remover architecture models'}
        ]
        
        for model_info in uvr_models:
            model_widget = self.create_simple_model_widget(model_info, 'uvr')
            models_layout.addWidget(model_widget)
            self.uvr_model_widgets.append(model_widget)
        
        # Other Models section
        print("DEBUG: Creating other models section")
        other_label = QLabel("<b>Other Model Types</b>")
        models_layout.addWidget(other_label)
        
        # Other models list
        self.other_model_widgets = []
        other_models = [
            {'name': 'Whisper', 'description': 'OpenAI Whisper speech recognition models'},
            {'name': 'SpeechBrain', 'description': 'SpeechBrain speaker embedding models'},
            {'name': 'PyAnnote', 'description': 'PyAnnote speaker diarization models'}
        ]
        
        for model_info in other_models:
            model_widget = self.create_simple_model_widget(model_info, 'other')
            models_layout.addWidget(model_widget)
            self.other_model_widgets.append(model_widget)
        
        models_layout.addStretch()
        print("DEBUG: Models layout completed")
        
        models_group.setLayout(models_layout)
        left_layout.addWidget(models_group)
        print("DEBUG: Models group completed")
        
        # Input Files section (moved to left panel)
        print("DEBUG: Creating input group")
        input_group = QGroupBox("Input Files")
        i_layout = QVBoxLayout()
        
        h1 = QHBoxLayout()
        self.mixed_label = QLabel("None")
        btn_mix = QPushButton("Load Mixed")
        btn_mix.clicked.connect(self.load_mixed_vocals)
        h1.addWidget(self.mixed_label)
        h1.addWidget(btn_mix)
        
        h2 = QHBoxLayout()
        self.ref_label = QLabel("Refs: 0")
        btn_ref = QPushButton("Add Ref")
        btn_ref.clicked.connect(self.add_reference_segment)
        btn_clr = QPushButton("Clear")
        btn_clr.clicked.connect(lambda: [self.reference_segments.clear(), self.ref_label.setText("Refs: 0")])
        h2.addWidget(self.ref_label)
        h2.addWidget(btn_ref)
        h2.addWidget(btn_clr)
        
        i_layout.addLayout(h1)
        i_layout.addLayout(h2)
        input_group.setLayout(i_layout)
        left_layout.addWidget(input_group)
        print("DEBUG: Input group completed")
        
        # Processing section (moved to left panel)
        print("DEBUG: Creating processing group")
        proc_group = QGroupBox("Processing")
        p_layout = QVBoxLayout()
        self.threshold_label = QLabel("Similarity Threshold: 0.72")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 95)
        self.threshold_slider.setValue(72)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        self.export_mask_checkbox = QCheckBox("Export mask diagnostic plot")
        self.export_mask_checkbox.setChecked(False)
        p_layout.addWidget(self.threshold_label)
        p_layout.addWidget(self.threshold_slider)
        p_layout.addWidget(self.export_mask_checkbox)
        self.process_btn = QPushButton("Separate Target Voice")
        self.process_btn.clicked.connect(self.start_separation)
        p_layout.addWidget(self.process_btn)
        proc_group.setLayout(p_layout)
        left_layout.addWidget(proc_group)
        print("DEBUG: Processing group completed")
        
        # Output section (moved to left panel)
        print("DEBUG: Creating output group")
        out_group = QGroupBox("Output")
        o_layout = QVBoxLayout()
        self.save_btn = QPushButton("Save Vocal")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        o_layout.addWidget(self.save_btn)
        out_group.setLayout(o_layout)
        left_layout.addWidget(out_group)
        print("DEBUG: Output group completed")
        
        left_layout.addStretch()
        main_splitter.addWidget(left_panel)
        print("DEBUG: Left panel completed")
        
        # Right panel - Waveform and Playback only
        print("DEBUG: Creating right panel")
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        
        # Waveform display
        print("DEBUG: Creating waveform widget")
        self.waveform = WaveformDisplayWidget()
        scroll = QScrollArea()
        scroll.setWidget(self.waveform)
        scroll.setWidgetResizable(True)
        right_layout.addWidget(scroll, stretch=1)
        print("DEBUG: Waveform widget completed")
        
        # Playback controls
        print("DEBUG: Creating playback group")
        play_group = QGroupBox("Playback")
        pl_layout = QHBoxLayout()
        self.btn_play_orig = QPushButton("Play Orig")
        self.btn_play_proc = QPushButton("Play Iso")
        self.btn_stop = QPushButton("Stop")
        self.btn_play_orig.clicked.connect(self.play_original)
        self.btn_play_proc.clicked.connect(self.play_processed)
        self.btn_stop.clicked.connect(self.stop_playback)
        pl_layout.addWidget(self.btn_play_orig)
        pl_layout.addWidget(self.btn_play_proc)
        pl_layout.addWidget(self.btn_stop)
        play_group.setLayout(pl_layout)
        right_layout.addWidget(play_group)
        print("DEBUG: Playback group completed")
        
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([400, 1200])
        print("DEBUG: Right panel completed")
        print("DEBUG: setup_main_tab completed")

    # setup_settings_tab method removed - model management moved to main tab

    def create_simple_model_widget(self, model_info, model_type):
        """Create a simple widget for a model item with label and select button"""
        print(f"DEBUG: Creating model widget: {model_info['name']}")
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 2, 0, 2)
        
        # Model name
        name_label = QLabel(f"<b>{model_info['name']}</b>")
        layout.addWidget(name_label)
        
        layout.addStretch()
        
        # Selected model display label (initially empty)
        selected_label = QLabel("")
        selected_label.hide()  # Hide initially until a model is selected
        layout.addWidget(selected_label)
        
        # Select button
        btn_select = QPushButton("Select")
        
        # Connect button signals
        if model_type == 'uvr':
            btn_select.clicked.connect(lambda: self.select_uvr_model(model_info))
        else:
            btn_select.clicked.connect(lambda: self.select_other_model(model_info))
        
        layout.addWidget(btn_select)
        
        # Store references
        widget.model_info = model_info
        widget.model_type = model_type
        widget.btn_select = btn_select
        widget.name_label = name_label
        widget.selected_label = selected_label
        
        print(f"DEBUG: Model widget created: {model_info['name']}")
        return widget

    def update_model_widget(self, model_info, model_name):
        """Update the model widget to show the selected model"""
        print(f"DEBUG: Updating model widget for {model_info['name']} with {model_name}")
        
        # Update UVR model widgets
        for widget in self.uvr_model_widgets:
            if widget.model_info['name'] == model_info['name']:
                widget.selected_label.setText(f"({model_name})")
                widget.selected_label.show()
                widget.btn_select.setText("Change")
                print(f"DEBUG: Updated UVR widget: {model_info['name']}")
                return
        
        # Update other model widgets
        for widget in self.other_model_widgets:
            if widget.model_info['name'] == model_info['name']:
                widget.selected_label.setText(f"({model_name})")
                widget.selected_label.show()
                widget.btn_select.setText("Change")
                print(f"DEBUG: Updated other widget: {model_info['name']}")
                return
        
        print(f"DEBUG: No widget found for {model_info['name']}")

    # --- MODEL MANAGEMENT LOGIC ---

    def select_uvr_model(self, model_info):
        """Select a UVR model from local directory"""
        target_dir = PROJECT_ROOT / f"ultimatevocalremovergui/models/{model_info['name']}/"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        file_filter = "Model Files (*.onnx *.pth *.pt *.ckpt)"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            f"Select {model_info['name']} Model File",
            str(target_dir),
            file_filter
        )
        
        if file_path:
            self.validate_and_copy_model(file_path, target_dir, model_info)

    def select_other_model(self, model_info):
        """Select a non-UVR model from local directory"""
        target_dir = SCRIPT_DIR / f"models/{model_info['name'].lower()}/"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        file_filter = "Model Files (*.onnx *.pth *.pt *.bin *.ckpt *.yaml)"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            f"Select {model_info['name']} Model File",
            str(target_dir),
            file_filter
        )
        
        if file_path:
            self.validate_and_copy_model(file_path, target_dir, model_info)

    def validate_and_copy_model(self, file_path, target_dir, model_info):
        """Validate and copy model to target directory"""
        file_path = Path(file_path)
        target_path = target_dir / file_path.name
        
        try:
            # Validate file exists
            if not file_path.exists():
                QMessageBox.warning(self, "Error", f"File does not exist: {file_path}")
                return
            
            # Copy file if not already there
            if target_path != file_path:
                import shutil
                shutil.copy2(file_path, target_path)
                
                # Update UI to show selected model
                self.update_model_widget(model_info, file_path.name)
                
                status_msg = f"{model_info['name']} model loaded: {file_path.name}"
                self.status_bar.showMessage(status_msg)
                
                QMessageBox.information(self, "Success", 
                    f"{model_info['name']} model loaded:\n{file_path.name}\n\nCopied to:\n{target_path}")
            else:
                # Update UI to show selected model
                self.update_model_widget(model_info, file_path.name)
                
                status_msg = f"{model_info['name']} model located: {file_path.name}"
                self.status_bar.showMessage(status_msg)
                
        except Exception as e:
            error_msg = f"Failed to load {model_info['name']}: {e}"
            self.status_bar.showMessage(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def scan_local_models(self):
        """Scan all model directories and show status in status bar"""
        total_models = 0
        model_dirs = []
        
        # UVR model directories
        uvr_dirs = ["Demucs_Models", "MDX_Net_Models", "VR_Models"]
        for dir_name in uvr_dirs:
            target_dir = PROJECT_ROOT / f"ultimatevocalremovergui/models/{dir_name}/"
            if target_dir.exists():
                model_dirs.append(target_dir)
        
        # Other model directories
        other_dirs = ["whisper", "speechbrain", "pyannote"]
        for dir_name in other_dirs:
            target_dir = SCRIPT_DIR / f"models/{dir_name}/"
            if target_dir.exists():
                model_dirs.append(target_dir)
        
        # Count files
        extensions = ['*.onnx', '*.pth', '*.tflite', '*.pt', '*.bin', '*.ckpt', '*.yaml']
        for model_dir in model_dirs:
            for ext in extensions:
                total_models += len(list(model_dir.glob(ext)))
        
        # Show status in main window status bar instead of removed pipeline status section
        status_message = f"Found {total_models} model files in {len(model_dirs)} directories"
        self.status_bar.showMessage(status_message)
        
        # Also show initial processing info in status bar
        if UVR_AVAILABLE or DEMUCS_AVAILABLE or WHISPER_AVAILABLE or SPEECHBRAIN_AVAILABLE:
            self.status_bar.showMessage(f"{status_message} | Ready for voice processing")
        else:
            self.status_bar.showMessage(f"{status_message} | Limited processing capabilities")

    # Download functionality removed - only local model loading supported

    # --- EXISTING FUNCTIONALITY ---

    def load_mixed_vocals(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Audio", "", "Audio (*.wav *.mp3 *.flac)")
        if path:
            self.mixed_audio_path = path
            self.mixed_label.setText(Path(path).name)
            wf, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            self.original_waveform = wf
            self.waveform.set_waveforms(wf, None, len(wf) / SAMPLE_RATE)
            self.save_btn.setEnabled(False)

    def add_reference_segment(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ref Audio", "", "Audio (*.wav *.mp3 *.flac)")
        if path:
            seg, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            self.reference_segments.append(seg)
            self.ref_label.setText(f"Refs: {len(self.reference_segments)}")

    def on_threshold_changed(self, value):
        self.threshold_label.setText(f"Similarity Threshold: {value / 100:.2f}")

    def start_separation(self):
        if not self.mixed_audio_path:
            QMessageBox.warning(self, "Missing Audio", "Please load mixed vocals first.")
            return
        if not self.reference_segments:
            QMessageBox.warning(self, "Missing Reference", "Please add at least one reference segment.")
            return
        
        self.process_btn.setEnabled(False)
        # Progress indicator removed
        
        # Show processing status in main status bar
        self.status_bar.showMessage("Starting voice separation...")
        self.separator.set_similarity_threshold(self.threshold_slider.value() / 100.0)
        
        self.worker = SeparationWorker(self.separator, self.mixed_audio_path, self.reference_segments)
        self.worker.progress.connect(self.status_bar.showMessage)
        self.worker.finished.connect(self.on_sep_finished)
        self.worker.error.connect(self.on_sep_error)
        self.worker.start()

    def on_sep_error(self, error):
        self.process_btn.setEnabled(True)
        error_msg = f"Voice separation failed: {error}"
        self.status_bar.showMessage(error_msg)
        QMessageBox.critical(self, "Error", error_msg)

    def on_sep_finished(self, result, msg, mask):
        self.process_btn.setEnabled(True)
        
        # Show completion status in main status bar
        success_msg = "Voice separation completed successfully"
        self.status_bar.showMessage(success_msg)
        
        self.processed_waveform = result
        self.latest_mask = mask
        self.waveform.set_waveforms(self.original_waveform, result, len(result) / SAMPLE_RATE)
        self.save_btn.setEnabled(True)
        if self.export_mask_checkbox.isChecked() and self.mixed_audio_path and mask is not None:
            plot_path = self.export_mask_diagnostic_plot(self.original_waveform, result, mask, self.mixed_audio_path)
            if plot_path is not None:
                self.status_bar.showMessage(f"Mask diagnostic plot exported: {plot_path}")
            else:
                self.status_bar.showMessage("Mask diagnostic export skipped (matplotlib unavailable or failed).")
        QMessageBox.information(self, "Done", msg)

    def export_mask_diagnostic_plot(self, original, isolated, mask, source_audio_path):
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(
                self,
                "Export Skipped",
                "matplotlib is not available. Install matplotlib to export mask diagnostic plots.",
            )
            return None

        try:
            length = min(len(original), len(isolated), len(mask))
            if length == 0:
                return None

            timeline = np.arange(length, dtype=np.float32) / SAMPLE_RATE
            source = Path(source_audio_path)
            out_name = f"{source.stem}_mask_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            out_path = source.parent / out_name

            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

            axes[0].plot(timeline, original[:length], color="#35b7d9", linewidth=0.8)
            axes[0].set_title("Original Waveform")
            axes[0].set_ylabel("Amplitude")
            axes[0].grid(alpha=0.2)

            axes[1].plot(timeline, mask[:length], color="#f39c12", linewidth=1.0)
            axes[1].set_title("Target Speaker Mask")
            axes[1].set_ylabel("Mask")
            axes[1].set_ylim(-0.05, 1.05)
            axes[1].grid(alpha=0.2)

            axes[2].plot(timeline, isolated[:length], color="#e74c3c", linewidth=0.8, label="Isolated")
            axes[2].plot(
                timeline,
                original[:length] * np.clip(mask[:length], 0.0, 1.0),
                color="#2ecc71",
                linewidth=0.7,
                alpha=0.8,
                label="Original x Mask",
            )
            axes[2].set_title("Isolated Result vs Original x Mask")
            axes[2].set_ylabel("Amplitude")
            axes[2].set_xlabel("Time (s)")
            axes[2].grid(alpha=0.2)
            axes[2].legend(loc="upper right")

            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return str(out_path)
        except Exception as e:
            QMessageBox.warning(self, "Export Failed", f"Failed to export mask diagnostic plot:\n{e}")
            return None

    def save_result(self):
        if self.processed_waveform is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Save", "", "WAV (*.wav)")
        if path:
            sf.write(path, self.processed_waveform, SAMPLE_RATE)

    def play_original(self):
        if self.mixed_audio_path:
            self._start_play(self.mixed_audio_path)

    def play_processed(self):
        if self.processed_waveform is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, self.processed_waveform, SAMPLE_RATE)
                self._start_play(f.name)

    def _start_play(self, path):
        self.stop_playback()
        self.playback_process = subprocess.Popen(["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", path])
        self.playback_start = np.datetime64("now")
        self.cursor_timer.start(100)

    def stop_playback(self):
        if self.playback_process:
            self.playback_process.terminate()
            self.playback_process = None
        self.cursor_timer.stop()
        self.waveform.playback_position = -1
        self.waveform.update()

    def update_cursor(self):
        if self.playback_process and self.playback_process.poll() is None:
            el = (np.datetime64("now") - self.playback_start) / np.timedelta64(1, 's')
            self.waveform.playback_position = el
            self.waveform.update()
        else:
            self.stop_playback()

    def closeEvent(self, e):
        self.stop_playback()
        clear_gpu_cache()
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = ProSingerSeparatorApp()
    w.show()
    sys.exit(app.exec())
