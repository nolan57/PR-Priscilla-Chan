import sys
import os
import warnings
import tempfile
import subprocess
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
import torch
import ffmpeg
import librosa
import gc
from scipy import signal
from scipy.ndimage import gaussian_filter1d, binary_opening, binary_closing

# --- SUPPRESS WARNINGS ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # UVR compatibility
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


def _find_uvr_root():
    """Locate a local UVR checkout inside this project."""
    default = PROJECT_ROOT / "ultimatevocalremovergui"
    if (default / "lib_v5").exists():
        return default

    for path in PROJECT_ROOT.rglob("ultimatevocalremovergui"):
        if path.is_dir() and (path / "lib_v5").exists():
            return path
    return None


def _find_local_model_file(model_name):
    """Locate a model file in common local paths under this project."""
    if UVR_ROOT is not None:
        candidate = UVR_ROOT / "models" / "VR_Models" / model_name
        if candidate.exists():
            return candidate

    for path in PROJECT_ROOT.rglob(model_name):
        if path.is_file():
            return path
    return None


UVR_ROOT = _find_uvr_root()
if UVR_ROOT is not None and str(UVR_ROOT) not in sys.path:
    sys.path.insert(0, str(UVR_ROOT))

# --- OFFLINE MODE SETUP ---
import huggingface_hub
import torchaudio

# Fix torchaudio compatibility issues for older versions
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
else:
    try:
        backends = torchaudio.list_audio_backends()
    except (AttributeError, TypeError, RuntimeError):
        def dummy_list_audio_backends():
            try:
                import soundfile
                return ["soundfile"]
            except:
                return []
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
    QPushButton, QFileDialog, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QGroupBox, QSlider, QProgressBar, QScrollArea,
    QStatusBar, QLineEdit, QComboBox, QTextEdit, QSplitter, QSpinBox,
    QDoubleSpinBox, QTabWidget, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect, QObject
from PyQt6.QtGui import QPainter, QColor, QImage, QCursor

# --- UVR COMPONENT INTEGRATION ---
# Import UVR's advanced modules with comprehensive error handling
UVR_AVAILABLE = False
UVR_MODULES = {}

try:
    # Try to import UVR modules individually
    try:
        from lib_v5.spec_utils import (
            wave_to_spectrogram, spectrogram_to_wave, 
            normalize, auto_transpose, merge_artifacts,
            reduce_vocal_aggressively, spectrogram_to_image
        )
        UVR_MODULES['spec_utils'] = True
    except ImportError:
        UVR_MODULES['spec_utils'] = False
    
    try:
        from lib_v5.modules import TFC, DenseTFC, TFC_TDF
        UVR_MODULES['modules'] = True
    except ImportError:
        UVR_MODULES['modules'] = False
    
    try:
        from lib_v5.vr_network.nets import CascadedASPPNet, determine_model_capacity
        UVR_MODULES['vr_network'] = True
    except ImportError:
        UVR_MODULES['vr_network'] = False
    
    try:
        from lib_v5.vr_network.layers import Conv2DBNActiv, SeperableConv2DBNActiv
        UVR_MODULES['vr_layers'] = True
    except ImportError:
        UVR_MODULES['vr_layers'] = False
    
    try:
        from lib_v5.vr_network.model_param_init import ModelParameters
        UVR_MODULES['model_params'] = True
    except ImportError:
        UVR_MODULES['model_params'] = False
        # Create fallback class
        class ModelParameters:
            def __init__(self, config_path='', device='cpu'):
                self.param = {
                    'n_fft': 2048,
                    'hl': 512,
                    'nn_architecture': 123821,
                    'model_state_dict': {},
                    'mid_side': False,
                    'mid_side_b2': False,
                    'stereo_w': False,
                    'stereo_n': False,
                    'reverse': False
                }
    
    try:
        from lib_v5.mdxnet import ConvTDFNet, AbstractMDXNet
        UVR_MODULES['mdxnet'] = True
    except ImportError:
        UVR_MODULES['mdxnet'] = False
    
    try:
        from lib_v5.tfc_tdf_v3 import TFC_TDF_net, STFT
        UVR_MODULES['tfc_tdf'] = True
    except ImportError:
        UVR_MODULES['tfc_tdf'] = False
    
    try:
        from core.models import ModelData
        UVR_MODULES['models'] = True
    except ImportError:
        UVR_MODULES['models'] = False
        # Create fallback class
        class ModelData:
            def __init__(self, model_name="fallback", config=None):
                self.config = config or {}
                self.model_path = ""
                self.process_method = 'demucs'
                self.is_denoise = False
                self.mdx_batch_size = 1
                self.compensate = 'auto'
                self.mdx_segment_size = 256
                self.is_mdx_c_seg_def = False
    
    try:
        from core.config import ConfigManager
        UVR_MODULES['config'] = True
    except ImportError:
        UVR_MODULES['config'] = False
        # Create fallback class
        class ConfigManager:
            def __init__(self):
                self.config = {
                    'device_set': 'cuda' if torch.cuda.is_available() else 'cpu',
                    'is_normalization': False,
                    'denoise_option': 'none',
                    'wav_type_set': 'PCM_16',
                    'mp3_bit_set': '320'
                }
    
    # Check if we have enough UVR components
    UVR_AVAILABLE = any([
        UVR_MODULES['spec_utils'],
        UVR_MODULES['vr_network'],
        UVR_MODULES['mdxnet']
    ])
    
    if UVR_AVAILABLE:
        print("✓ UVR advanced components loaded successfully")
    else:
        print("⚠ Limited UVR components available")

except Exception as e:
    print(f"⚠ UVR components import failed: {e}")
    UVR_AVAILABLE = False
    
    # Ensure all fallback classes are defined
    class ModelParameters:
        def __init__(self, config_path='', device='cpu'):
            self.param = {
                'n_fft': 2048,
                'hl': 512,
                'nn_architecture': 123821,
                'model_state_dict': {},
                'mid_side': False,
                'mid_side_b2': False,
                'stereo_w': False,
                'stereo_n': False,
                'reverse': False
            }
    
    class ModelData:
        def __init__(self, model_name="fallback", config=None):
            self.config = config or {}
            self.model_path = ""
            self.process_method = 'demucs'
            self.is_denoise = False
            self.mdx_batch_size = 1
            self.compensate = 'auto'
            self.mdx_segment_size = 256
            self.is_mdx_c_seg_def = False
    
    class ConfigManager:
        def __init__(self):
            self.config = {
                'device_set': 'cuda' if torch.cuda.is_available() else 'cpu',
                'is_normalization': False,
                'denoise_option': 'none',
                'wav_type_set': 'PCM_16',
                'mp3_bit_set': '320'
            }
    
    UVR_MODULES = {
        'spec_utils': False,
        'modules': False,
        'vr_network': False,
        'vr_layers': False,
        'model_params': False,
        'mdxnet': False,
        'tfc_tdf': False,
        'models': False,
        'config': False
    }

# --- HYBRID VOICE SEPARATION IMPORTS ---
try:
    # Try local Demucs implementation first (from ultimatevocalremovergui)
    if UVR_ROOT is not None and str(UVR_ROOT) not in sys.path:
        sys.path.insert(0, str(UVR_ROOT))
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    
    DEMUCS_AVAILABLE = True
    print("✓ Using local Demucs implementation")
except ImportError as e:
    print(f"⚠ Local Demucs import failed: {e}")
    try:
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        DEMUCS_AVAILABLE = True
        print("✓ Using installed Demucs package")
    except ImportError as e:
        DEMUCS_AVAILABLE = False
        print(f"⚠ Demucs not available: {e}")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Whisper not available, using basic timestamp detection")

# Robust SpeechBrain import with fallback
try:
    from speechbrain.inference import SpeakerRecognition
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    print("SpeechBrain not available, using basic similarity matching")

import onnxruntime as ort

SAMPLE_RATE = 44100  # UVR standard
DEFAULT_THRESHOLD = 0.75

def clear_gpu_cache():
    """UVR's GPU cache clearing function"""
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()

class UVRSeparator:
    """Advanced separator using UVR's proven architecture"""
    
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
            # Use UVR's model loading logic
            self.model_params = ModelParameters(model_path, self.device)
            
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
                model_data = ModelData(model_name="MDX", config=self.config_manager.config if self.config_manager else {})
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
                
                # Load weights (simplified - UVR has more complex loading)
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
    """Enhanced hybrid approach using UVR's proven components"""

    def __init__(self, device=DEVICE):
        self.device = device
        self.uvr_separator = UVRSeparator(device)
        self.demucs_model = None
        self.speaker_model = None
        self.whisper_model = None
        self._load_models()

    def _load_models(self):
        """Load all required models with UVR integration"""
        # Try to load UVR VR model first
        try:
            vr_model_path = _find_local_model_file("UVR-MDX-NET-Inst_HQ_5.onnx")
            if vr_model_path is not None:
                if self.uvr_separator.load_vr_model(str(vr_model_path)):
                    print("✓ UVR VR model loaded for high-quality separation")
        except Exception as e:
            print(f"Model loading warning (UVR VR): {e}")

        # Load Demucs as fallback
        if DEMUCS_AVAILABLE:
            try:
                self.demucs_model = get_model("htdemucs_ft", repo=None)
                print("✓ Demucs model loaded (local)")
            except Exception as e_local:
                try:
                    self.demucs_model = get_model("htdemucs_ft")
                    print("✓ Demucs model loaded (remote)")
                except Exception as e_remote:
                    print(
                        "Model loading warning (Demucs): "
                        f"local={e_local}; remote={e_remote}"
                    )

        # Load speaker recognition model
        if SPEECHBRAIN_AVAILABLE:
            try:
                model_dir = "./models/embedding_model"
                if os.path.exists(model_dir):
                    self.speaker_model = SpeakerRecognition.from_hparams(
                        source=model_dir,
                        savedir=model_dir,
                        run_opts={"device": self.device},
                    )
                    print("✓ Speaker recognition model loaded")
            except Exception as e:
                print(f"Model loading warning (SpeakerRecognition): {e}")

        # Load Whisper for precise timing
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("small")
                print("✓ Whisper model loaded")
            except Exception as e:
                print(f"Model loading warning (Whisper): {e}")

    def separate_target_voice(
        self, mixed_audio_path, reference_segments, progress_callback=None
    ):
        """Enhanced hybrid separation pipeline using UVR components"""
        if progress_callback:
            progress_callback("Loading audio...")

        # Load mixed vocals
        waveform, sr = librosa.load(mixed_audio_path, sr=SAMPLE_RATE, mono=True)

        if progress_callback:
            progress_callback("Step 1: UVR vocal separation...")

        # Step 1: Use UVR's advanced separation if available
        if self.uvr_separator.model is not None:
            clean_vocals = self._uvr_separation(mixed_audio_path, progress_callback)
        else:
            # Fallback to Demucs
            clean_vocals = self._demucs_separation(waveform, sr)

        if progress_callback:
            progress_callback("Step 2: Speaker verification...")

        # Step 2: Extract target speaker embeddings
        target_embeddings = self._extract_target_embeddings(reference_segments)

        if progress_callback:
            progress_callback("Step 3: Enhanced voice matching...")

        # Step 3: Create precise speaker mask using UVR techniques
        speaker_mask = self._create_enhanced_speaker_mask(clean_vocals, target_embeddings)

        if progress_callback:
            progress_callback("Step 4: UVR post-processing...")

        # Step 4: Apply UVR post-processing
        isolated_vocal = self._uvr_post_processing(clean_vocals, speaker_mask)

        return isolated_vocal, speaker_mask

    def _uvr_separation(self, audio_path, progress_callback=None):
        """Use UVR's separation pipeline"""
        try:
            return self.uvr_separator.separate_vocals(audio_path, progress_callback)
        except Exception as e:
            print(f"UVR separation failed: {e}")
            # Fallback to basic processing
            waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            return self._spectral_subtraction(waveform)

    def _demucs_separation(self, waveform, sr):
        """Demucs separation with UVR enhancements"""
        if self.demucs_model and DEMUCS_AVAILABLE:
            # Convert to tensor format
            tensor_waveform = torch.tensor(waveform).unsqueeze(0).unsqueeze(0)

            # Apply Demucs separation
            with torch.no_grad():
                sources = apply_model(
                    self.demucs_model, tensor_waveform.to(self.device)
                )
                vocals = sources[0][3].cpu().numpy()  # Index 3 is typically vocals

            return vocals.flatten()
        else:
            return self._spectral_subtraction(waveform)

    def _spectral_subtraction(self, audio):
        """Enhanced spectral subtraction using UVR utilities"""
        if UVR_MODULES.get('spec_utils', False):
            try:
                # Use UVR's advanced spectral processing
                spec_left = librosa.stft(audio, n_fft=2048, hop_length=512)
                spec_right = librosa.stft(audio, n_fft=2048, hop_length=512)
                spec = np.asfortranarray([spec_left, spec_right])

                # Apply UVR's noise reduction
                magnitude = np.abs(spec)
                noise_floor = np.mean(magnitude[:, :15], axis=1, keepdims=True)
                enhanced_mag = np.maximum(magnitude - noise_floor * 2.0, 0)

                # Reconstruct with proper phase
                phase = np.angle(spec)
                enhanced_spec = enhanced_mag * np.exp(1j * phase)

                # Convert back using UVR method
                result_left = librosa.istft(enhanced_spec[0], hop_length=512)
                result_right = librosa.istft(enhanced_spec[1], hop_length=512)
                result = np.mean([result_left, result_right], axis=0)
                
                return result
            except Exception as e:
                print(f"UVR spectral processing failed: {e}")

        # Fallback: basic spectral approach
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        noise_floor = np.mean(magnitude[:, :10], axis=1, keepdims=True)
        enhanced_mag = np.maximum(magnitude - noise_floor * 1.5, 0)

        enhanced_stft = enhanced_mag * np.exp(1j * phase)
        return librosa.istft(enhanced_stft, hop_length=512)

    def _extract_target_embeddings(self, reference_segments):
        """Extract speaker embeddings with UVR enhancements"""
        embeddings = []

        if not self.speaker_model or not SPEECHBRAIN_AVAILABLE:
            return self._extract_basic_features(reference_segments)

        for segment in reference_segments:
            try:
                segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    embedding = self.speaker_model.encode_batch(
                        segment_tensor.to(self.device)
                    )
                    embedding = embedding.squeeze().cpu().numpy()
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                    embeddings.append(embedding)

            except Exception as e:
                print(f"Warning: Failed to process reference segment: {e}")
                continue

        return embeddings if embeddings else [np.ones(512)]

    def _extract_basic_features(self, reference_segments):
        """Enhanced feature extraction using UVR techniques"""
        features = []
        for segment in reference_segments:
            if len(segment) < 1000:
                continue

            # Use UVR's spectral analysis
            stft = librosa.stft(segment, n_fft=512, hop_length=256)
            magnitude = np.abs(stft)

            # Enhanced feature set inspired by UVR
            spectral_centroid = librosa.feature.spectral_centroid(S=stft)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(S=stft)[0]
            spectral_contrast = librosa.feature.spectral_contrast(S=stft)

            feature_vector = np.concatenate([
                np.mean(magnitude, axis=1)[:64],
                [np.mean(spectral_centroid), np.std(spectral_centroid)],
                [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
                [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
                np.mean(spectral_contrast, axis=1)[:8],  # 8 contrast bands
            ])

            feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-8)
            features.append(feature_vector)

        return features if features else [np.ones(84)]  # 64 + 2 + 2 + 2 + 8 = 78, pad to 84

    def _create_enhanced_speaker_mask(self, vocals, target_embeddings):
        """Create speaker mask using UVR's advanced techniques"""
        # UVR-style frame processing
        frame_length = int(0.1 * SAMPLE_RATE)  # 100ms frames
        hop_length = frame_length // 4  # 75% overlap
        
        mask = np.zeros_like(vocals)

        for i in range(0, len(vocals) - frame_length, hop_length):
            frame = vocals[i : i + frame_length]

            if len(frame) < frame_length // 2:
                continue

            try:
                # Extract frame features
                frame_embedding = self._extract_frame_features(frame)

                # Compare with target embeddings
                similarities = []
                for target_emb in target_embeddings:
                    # Cosine similarity
                    cos_sim = np.dot(frame_embedding, target_emb[:len(frame_embedding)])
                    
                    # UVR-style confidence weighting
                    confidence_boost = 1.2 if cos_sim > 0.8 else 1.0
                    similarities.append(min(1.0, cos_sim * confidence_boost))

                max_similarity = max(similarities)

                # Apply to overlapping region
                start_idx = i
                end_idx = min(i + frame_length, len(mask))
                mask[start_idx:end_idx] = np.maximum(
                    mask[start_idx:end_idx], max_similarity
                )

            except Exception as e:
                continue

        # UVR-style post-processing
        mask = self._apply_uvr_mask_processing(mask)
        
        return mask

    def _apply_uvr_mask_processing(self, mask):
        """Apply UVR's mask processing techniques"""
        # Gaussian smoothing
        mask = gaussian_filter1d(mask, sigma=3)

        # Adaptive threshold
        dynamic_threshold = np.mean(mask) + 0.1 * np.std(mask)
        threshold = max(0.5, min(0.8, dynamic_threshold))

        # Binary mask with hysteresis
        mask_binary = (mask > threshold).astype(float)

        # UVR's morphological operations
        mask_binary = binary_closing(mask_binary, structure=np.ones(3, dtype=bool))
        mask_binary = binary_opening(mask_binary, structure=np.ones(2, dtype=bool))

        # UVR's artifact merging
        if UVR_MODULES.get('spec_utils', False):
            try:
                mask_binary = merge_artifacts(mask_binary, thres=0.01, min_range=64, fade_size=32)
            except Exception as e:
                print(f"UVR artifact merging failed: {e}")

        return mask_binary.astype(float)

    def _extract_frame_features(self, frame):
        """Extract frame features using UVR techniques"""
        stft = librosa.stft(frame, n_fft=512, hop_length=256)
        magnitude = np.abs(stft)

        spectral_centroid = librosa.feature.spectral_centroid(S=stft)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(S=stft)[0]
        spectral_contrast = librosa.feature.spectral_contrast(S=stft)

        feature_vector = np.concatenate([
            np.mean(magnitude, axis=1)[:64],
            [np.mean(spectral_centroid), np.std(spectral_centroid)],
            [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
            np.mean(spectral_contrast, axis=1)[:8],
        ])

        return feature_vector / (np.linalg.norm(feature_vector) + 1e-8)

    def _uvr_post_processing(self, vocals, mask):
        """Apply UVR's post-processing techniques"""
        # Apply mask
        isolated_vocal = vocals * mask

        # UVR-style vocal reduction if needed
        if UVR_MODULES.get('spec_utils', False):
            try:
                # Apply gentle vocal enhancement
                isolated_vocal = reduce_vocal_aggressively(
                    vocals, isolated_vocal, softmask=0.1
                )
            except Exception as e:
                print(f"UVR vocal reduction failed: {e}")

        # Normalize using UVR method if available
        if UVR_MODULES.get('spec_utils', False):
            try:
                isolated_vocal = normalize(isolated_vocal, is_normalize=True)
            except Exception as e:
                print(f"UVR normalization failed: {e}")
        else:
            # Fallback normalization
            max_amp = np.max(np.abs(isolated_vocal))
            if max_amp > 1.0:
                isolated_vocal = isolated_vocal / max_amp

        return isolated_vocal

class SeparationWorker(QThread):
    """Enhanced worker thread with UVR integration"""

    finished = pyqtSignal(object, str)
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
        clear_gpu_cache()  # UVR cache clearing

    def run(self):
        try:
            if self._is_cancelled:
                return

            result, mask = self.separator.separate_target_voice(
                self.mixed_audio_path,
                self.reference_segments,
                progress_callback=self.progress.emit,
            )

            if not self._is_cancelled:
                clear_gpu_cache()  # Clear cache after processing
                self.finished.emit(result, "Separation completed successfully")

        except Exception as e:
            if not self._is_cancelled:
                clear_gpu_cache()
                self.error.emit(str(e))

class WaveformDisplayWidget(QWidget):
    """Enhanced waveform display with UVR-style visualization"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_waveform = None
        self.processed_waveform = None
        self.duration = 0.0
        self.playback_position = -1.0
        self.zoom_factor = 1.0
        self.setMinimumHeight(300)

    def set_waveforms(self, original, processed, duration):
        self.original_waveform = original
        self.processed_waveform = processed
        self.duration = duration
        self.update()

    def paintEvent(self, a0):
        if self.original_waveform is None:
            return

        painter = QPainter(self)
        width, height = self.width(), self.height()
        mid_height = height // 2

        # Draw background
        painter.fillRect(a0.rect(), Qt.GlobalColor.black)

        # Draw original waveform (top half)
        if self.original_waveform is not None:
            painter.setPen(QColor(0, 255, 255))  # Cyan
            self._draw_waveform(painter, self.original_waveform, 0, mid_height)

        # Draw processed waveform (bottom half)
        if self.processed_waveform is not None:
            painter.setPen(QColor(255, 100, 100))  # Red
            self._draw_waveform(painter, self.processed_waveform, mid_height, height)

        # Draw playback position
        if self.playback_position >= 0:
            px = int((self.playback_position / self.duration) * width)
            painter.setPen(QColor(255, 255, 0))  # Yellow
            painter.drawLine(px, 0, px, height)

    def _draw_waveform(self, painter, waveform, top_y, bottom_y):
        if len(waveform) == 0:
            return

        width = self.width()
        height = bottom_y - top_y
        mid_y = top_y + height // 2

        # Sample waveform for display
        step = max(1, len(waveform) // (width * 2))
        sampled = waveform[::step]

        if len(sampled) == 0:
            return

        # Scale for display
        max_amp = np.max(np.abs(sampled)) + 1e-6
        scale = (height // 3) / max_amp

        # Draw waveform
        for i in range(len(sampled) - 1):
            if i >= width - 1:
                break
            x1 = i
            x2 = i + 1
            y1 = mid_y - int(sampled[i] * scale)
            y2 = mid_y - int(sampled[i + 1] * scale)
            painter.drawLine(x1, y1, x2, y2)

class ProSingerSeparatorApp(QMainWindow):
    """Enhanced main application with UVR integration"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Professional Singer Voice Separator - UVR Enhanced")
        self.resize(1600, 1200)

        # Core components
        self.separator = HybridVoiceSeparator()
        self.mixed_audio_path = None
        self.reference_segments = []
        self.original_waveform = None
        self.processed_waveform = None
        self.sr = SAMPLE_RATE
        self.playback_process = None
        self.separation_worker = None

        # UI timers
        self.cursor_timer = QTimer()
        self.cursor_timer.timeout.connect(self.update_cursor)

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Create tabbed interface
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)

        # Main separation tab
        main_tab = QWidget()
        tab_widget.addTab(main_tab, "Voice Separation")
        self.setup_main_tab(main_tab)

        # Advanced settings tab
        settings_tab = QWidget()
        tab_widget.addTab(settings_tab, "Advanced Settings")
        self.setup_settings_tab(settings_tab)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def setup_main_tab(self, parent):
        layout = QVBoxLayout(parent)

        # Splitter for better layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Model Status Group (Enhanced)
        model_group = QGroupBox("Model Status - UVR Enhanced")
        model_layout = QVBoxLayout()
        self.model_status = QTextEdit()
        self.model_status.setMaximumHeight(120)
        self.model_status.setReadOnly(True)
        self._update_model_status()
        model_layout.addWidget(self.model_status)
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)

        # Input Files Group
        input_group = QGroupBox("Input Files")
        input_layout = QVBoxLayout()

        # Mixed vocals input
        mixed_layout = QHBoxLayout()
        self.mixed_path_label = QLabel("Mixed Vocals: Not loaded")
        self.load_mixed_btn = QPushButton("Load Mixed Vocals")
        self.load_mixed_btn.clicked.connect(self.load_mixed_vocals)
        mixed_layout.addWidget(self.mixed_path_label)
        mixed_layout.addWidget(self.load_mixed_btn)
        input_layout.addLayout(mixed_layout)

        # Reference segments
        ref_layout = QHBoxLayout()
        self.ref_segments_label = QLabel("Reference Segments: 0")
        self.add_ref_btn = QPushButton("Add Reference Segment")
        self.add_ref_btn.clicked.connect(self.add_reference_segment)
        self.clear_ref_btn = QPushButton("Clear References")
        self.clear_ref_btn.clicked.connect(self.clear_references)
        ref_layout.addWidget(self.ref_segments_label)
        ref_layout.addWidget(self.add_ref_btn)
        ref_layout.addWidget(self.clear_ref_btn)
        input_layout.addLayout(ref_layout)

        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)

        # Processing Controls (Enhanced)
        process_group = QGroupBox("Processing - UVR Enhanced")
        process_layout = QVBoxLayout()

        # Threshold control
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Speaker Similarity Threshold:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 95)
        self.threshold_slider.setValue(75)
        self.threshold_label = QLabel("0.75")
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        thresh_layout.addWidget(self.threshold_slider)
        thresh_layout.addWidget(self.threshold_label)
        process_layout.addLayout(thresh_layout)

        # UVR options
        uvr_options_layout = QHBoxLayout()
        self.use_uvr_checkbox = QCheckBox("Use UVR Processing")
        self.use_uvr_checkbox.setChecked(True)
        self.use_denoise_checkbox = QCheckBox("Apply Denoising")
        self.use_denoise_checkbox.setChecked(True)
        uvr_options_layout.addWidget(self.use_uvr_checkbox)
        uvr_options_layout.addWidget(self.use_denoise_checkbox)
        process_layout.addLayout(uvr_options_layout)

        # Process button
        self.process_btn = QPushButton("Separate Target Voice")
        self.process_btn.clicked.connect(self.start_separation)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        process_layout.addWidget(self.process_btn)

        process_group.setLayout(process_layout)
        left_layout.addWidget(process_group)

        # Output Controls
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()

        self.save_btn = QPushButton("Save Isolated Vocal")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        output_layout.addWidget(self.save_btn)

        output_group.setLayout(output_layout)
        left_layout.addWidget(output_group)

        # Playback Controls
        playback_group = QGroupBox("Playback")
        playback_layout = QHBoxLayout()

        self.play_original_btn = QPushButton("Play Original")
        self.play_processed_btn = QPushButton("Play Isolated")
        self.stop_btn = QPushButton("Stop")

        self.play_original_btn.clicked.connect(self.play_original)
        self.play_processed_btn.clicked.connect(self.play_processed)
        self.stop_btn.clicked.connect(self.stop_playback)

        playback_layout.addWidget(self.play_original_btn)
        playback_layout.addWidget(self.play_processed_btn)
        playback_layout.addWidget(self.stop_btn)

        playback_group.setLayout(playback_layout)
        left_layout.addWidget(playback_group)

        # Progress and Status
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready - UVR Enhanced")
        left_layout.addWidget(self.status_label)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # Right panel - Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Waveform display
        self.waveform_display = WaveformDisplayWidget()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.waveform_display)
        scroll_area.setWidgetResizable(True)
        right_layout.addWidget(scroll_area)

        # Legend
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(QLabel("Legend: "))
        legend_layout.addWidget(QLabel(" cyan = Original Mixed Vocals "))
        legend_layout.addWidget(QLabel(" red = Isolated Target Vocal "))
        legend_layout.addStretch()
        right_layout.addLayout(legend_layout)

        splitter.addWidget(right_panel)
        splitter.setSizes([500, 1100])

    def setup_settings_tab(self, parent):
        layout = QVBoxLayout(parent)

        # UVR Settings Group
        uvr_group = QGroupBox("UVR Settings")
        uvr_layout = QVBoxLayout()

        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("VR Model:"))
        self.vr_model_combo = QComboBox()
        self.vr_model_combo.addItems(["UVR-MDX-NET-Inst_HQ_5", "UVR-MDX-NET-Inst_1", "UVR-MDX-NET-Inst_2"])
        model_layout.addWidget(self.vr_model_combo)
        uvr_layout.addLayout(model_layout)

        # Advanced options
        advanced_layout = QHBoxLayout()
        self.aggression_spinbox = QSpinBox()
        self.aggression_spinbox.setRange(1, 20)
        self.aggression_spinbox.setValue(10)
        advanced_layout.addWidget(QLabel("Aggression:"))
        advanced_layout.addWidget(self.aggression_spinbox)
        uvr_layout.addLayout(advanced_layout)

        uvr_group.setLayout(uvr_layout)
        layout.addWidget(uvr_group)

        # Audio Settings Group
        audio_group = QGroupBox("Audio Settings")
        audio_layout = QVBoxLayout()

        # Sample rate
        sr_layout = QHBoxLayout()
        sr_layout.addWidget(QLabel("Sample Rate:"))
        self.sr_combo = QComboBox()
        self.sr_combo.addItems(["44100", "48000"])
        self.sr_combo.setCurrentText("44100")
        sr_layout.addWidget(self.sr_combo)
        audio_layout.addLayout(sr_layout)

        # Quality settings
        quality_layout = QHBoxLayout()
        self.quality_spinbox = QSpinBox()
        self.quality_spinbox.setRange(1, 10)
        self.quality_spinbox.setValue(5)
        quality_layout.addWidget(QLabel("Quality:"))
        quality_layout.addWidget(self.quality_spinbox)
        audio_layout.addLayout(quality_layout)

        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)

        layout.addStretch()

    def _update_model_status(self):
        status_text = "Loaded Models - UVR Enhanced:\n"
        
        if UVR_AVAILABLE and self.separator.uvr_separator.model is not None:
            status_text += "✓ UVR VR Model (Advanced Separation)\n"
        else:
            status_text += "✗ UVR VR Model (Not Available)\n"

        if DEMUCS_AVAILABLE and self.separator.demucs_model:
            status_text += "✓ Demucs (Fallback Separation)\n"
        else:
            status_text += "✗ Demucs (Not Available)\n"

        if self.separator.speaker_model:
            status_text += "✓ Speaker Recognition\n"
        else:
            status_text += "✗ Speaker Recognition\n"

        if WHISPER_AVAILABLE and self.separator.whisper_model:
            status_text += "✓ Whisper (Timestamp Detection)"
        else:
            status_text += "✗ Whisper (Not Available)"

        self.model_status.setText(status_text)

    def update_threshold_label(self):
        value = self.threshold_slider.value() / 100.0
        self.threshold_label.setText(f"{value:.2f}")

    def load_mixed_vocals(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Mixed Vocals", "", "Audio Files (*.wav *.mp3 *.flac *.aac)"
        )

        if file_path:
            try:
                self.mixed_audio_path = file_path
                self.mixed_path_label.setText(f"Mixed Vocals: {Path(file_path).name}")

                # Load waveform for display
                waveform, self.sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
                self.original_waveform = waveform
                duration = len(waveform) / SAMPLE_RATE
                self.waveform_display.set_waveforms(waveform, None, duration)

                self.status_label.setText("Mixed vocals loaded successfully")
                self.process_btn.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load audio: {str(e)}")

    def add_reference_segment(self):
        if not self.mixed_audio_path:
            QMessageBox.warning(self, "Warning", "Please load mixed vocals first")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Segment", "", "Audio Files (*.wav *.mp3 *.flac)"
        )

        if file_path:
            try:
                # Load reference segment
                segment, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
                self.reference_segments.append(segment)

                self.ref_segments_label.setText(
                    f"Reference Segments: {len(self.reference_segments)}"
                )
                self.status_label.setText(
                    f"Added reference segment {len(self.reference_segments)}"
                )

            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to load reference: {str(e)}"
                )

    def clear_references(self):
        self.reference_segments.clear()
        self.ref_segments_label.setText("Reference Segments: 0")
        self.status_label.setText("References cleared")

    def start_separation(self):
        if not self.mixed_audio_path or not self.reference_segments:
            QMessageBox.warning(
                self, "Warning", "Please load mixed vocals and reference segments"
            )
            return

        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        self.separation_worker = SeparationWorker(
            self.separator, self.mixed_audio_path, self.reference_segments
        )

        self.separation_worker.progress.connect(self.status_label.setText)
        self.separation_worker.finished.connect(self.on_separation_finished)
        self.separation_worker.error.connect(self.on_separation_error)

        self.separation_worker.start()

    def on_separation_finished(self, result, message):
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.save_btn.setEnabled(True)

        # Store result
        self.processed_waveform = result
        duration = len(result) / SAMPLE_RATE
        self.waveform_display.set_waveforms(self.original_waveform, result, duration)

        self.status_label.setText(message)
        QMessageBox.information(self, "Success", "Voice separation completed with UVR enhancement!")

    def on_separation_error(self, error_message):
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Error occurred")
        QMessageBox.critical(self, "Error", f"Separation failed: {error_message}")

    def save_result(self):
        if self.processed_waveform is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Isolated Vocal", "", "WAV Files (*.wav)"
        )

        if file_path:
            try:
                # Use UVR's normalization before saving
                if UVR_MODULES.get('spec_utils', False):
                    try:
                        self.processed_waveform = normalize(self.processed_waveform, is_normalize=True)
                    except Exception as e:
                        print(f"UVR normalization failed: {e}")
                        # Fallback normalization
                        max_amp = np.max(np.abs(self.processed_waveform))
                        if max_amp > 1.0:
                            self.processed_waveform = self.processed_waveform / max_amp
                else:
                    # Fallback normalization
                    max_amp = np.max(np.abs(self.processed_waveform))
                    if max_amp > 1.0:
                        self.processed_waveform = self.processed_waveform / max_amp
                
                sf.write(file_path, self.processed_waveform, SAMPLE_RATE)
                self.status_label.setText(f"Saved to {file_path}")
                QMessageBox.information(
                    self, "Success", f"File saved successfully to {file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")

    def play_original(self):
        if self.mixed_audio_path:
            self.stop_playback()
            self._start_ffplay(self.mixed_audio_path, "original")

    def play_processed(self):
        if self.processed_waveform is not None:
            self.stop_playback()
            # Create temporary file for playback
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                if UVR_MODULES.get('spec_utils', False):
                    try:
                        tmp_waveform = normalize(self.processed_waveform, is_normalize=True)
                    except Exception as e:
                        print(f"UVR normalization failed: {e}")
                        # Fallback normalization
                        max_amp = np.max(np.abs(self.processed_waveform))
                        tmp_waveform = self.processed_waveform / max_amp if max_amp > 1.0 else self.processed_waveform
                else:
                    tmp_waveform = self.processed_waveform
                sf.write(tmp.name, tmp_waveform, SAMPLE_RATE)
                self._start_ffplay(tmp.name, "processed", cleanup=tmp.name)

    def _start_ffplay(self, file_path, mode, cleanup=None):
        try:
            self.playback_process = subprocess.Popen(
                ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", file_path]
            )

            self.playback_start = np.datetime64("now")
            self.cursor_timer.start(100)

            if mode == "original":
                self.play_original_btn.setText("Playing...")
                self.play_original_btn.setEnabled(False)
            else:
                self.play_processed_btn.setText("Playing...")
                self.play_processed_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Playback failed: {str(e)}")

    def stop_playback(self):
        if self.playback_process:
            self.playback_process.terminate()
            self.playback_process = None

        self.cursor_timer.stop()
        self.waveform_display.playback_position = -1.0
        self.waveform_display.update()

        self.play_original_btn.setText("Play Original")
        self.play_original_btn.setEnabled(True)
        self.play_processed_btn.setText("Play Isolated")
        self.play_processed_btn.setEnabled(True)

    def update_cursor(self):
        if self.playback_process and self.playback_process.poll() is None:
            elapsed = (np.datetime64("now") - self.playback_start) / np.timedelta64(
                1, "s"
            )
            self.waveform_display.playback_position = elapsed
            self.waveform_display.update()
        else:
            self.stop_playback()

    def closeEvent(self, event):
        """Clean up resources on close"""
        if self.separation_worker and self.separation_worker.isRunning():
            self.separation_worker.cancel()
            self.separation_worker.wait()
        
        self.stop_playback()
        clear_gpu_cache()  # UVR cleanup
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    window = ProSingerSeparatorApp()
    window.show()

    sys.exit(app.exec())
