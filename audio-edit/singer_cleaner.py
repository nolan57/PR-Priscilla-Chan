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

# --- SUPPRESS MPS WARNINGS ---
# Disable MPS graph fuser warnings and optimize performance
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Allow fallback to CPU for unsupported ops
warnings.filterwarnings("ignore", message=".*Unknown device for graph fuser.*")
warnings.filterwarnings("ignore", message=".*mps.*fallback.*")
if torch.backends.mps.is_available():
    # Reduce MPS warning verbosity
    warnings.filterwarnings("ignore", module="torch._inductor.codegen")

# --- DEVICE DETECTION ---
# Check for available devices: CUDA (NVIDIA), MPS (Apple Silicon), or CPU
# Note: MPS doesn't support float64, so we enforce float32 for all tensors
def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"

DEVICE = get_best_device()
print(f"Using device: {DEVICE}")
if DEVICE == "mps":
    print("Note: MPS backend requires float32 tensors (float64 not supported)")

# --- MANDATORY OFFLINE & COMPATIBILITY PATCHES ---
# This block MUST be at the very top to intercept internal library calls.
import huggingface_hub
import torchaudio

# 1. Fix the 'use_auth_token' TypeError and force local-only mode
_old_hf_hub_download = huggingface_hub.hf_hub_download


def _patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token', None)
    # Force local loading for every internal call to bypass the Hub cache mechanism
    kwargs['local_files_only'] = True
    return _old_hf_hub_download(*args, **kwargs)


huggingface_hub.hf_hub_download = _patched_hf_hub_download

# 2. Suppress Windows-specific SpeechBrain and Torchaudio warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="speechbrain.utils.autocast")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain.utils.parameter_transfer")
# Suppress SIP DeprecationWarnings from PyQt6
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*sipPyTypeDict.*")
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# 3. Disable all network telemetry
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# ------------------------------------------------

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QGroupBox, QSlider, QProgressBar, QScrollArea,
    QStatusBar, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect
from PyQt6.QtGui import QPainter, QColor, QImage

from silero_vad import get_speech_timestamps
from speechbrain.inference import SpeakerRecognition
import torch
import onnxruntime as ort

SAMPLE_RATE = 16000
DEFAULT_THRESHOLD = 0.4

# Standardized Professional Styling
BUTTON_STYLE = """
    QPushButton {
        background-color: #34495e; color: white; border-radius: 8px;
        padding: 5px 15px; font-weight: bold; min-height: 35px; border: 1px solid #2c3e50;
    }
    QPushButton:hover { background-color: #2c3e50; }
    QPushButton:disabled { background-color: #95a5a6; color: #bdc3c7; }
"""
# CANCEL_BUTTON_STYLE = BUTTON_STYLE.replace("#34495e", "#c0392b")
# SAVE_BUTTON_STYLE = BUTTON_STYLE.replace("#34495e", "#1B5E20")


class ConfigWorker(QThread):
    """Asynchronous loader to prevent UI freezing during recursive search."""
    finished = pyqtSignal(bool, str, object, object)
    progress = pyqtSignal(str)

    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir

    def _search_hf_cache(self, filename):
        """Search for model files in Hugging Face local cache directories."""
        # Common HF cache locations
        hf_cache_home = os.environ.get('HUGGINGFACE_HUB_CACHE', None)
        if not hf_cache_home:
            hf_cache_home = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')

        if not os.path.exists(hf_cache_home):
            return None

        self.progress.emit(f"Searching HF cache at {hf_cache_home}...")
        cache_path = Path(hf_cache_home)

        # Search recursively in cache
        result = next(cache_path.rglob(filename), None)
        return result

    def run(self):
        try:
            root_path = Path(self.root_dir)
            self.progress.emit("Scanning directory for local weights...")

            # Find required local files in specified directory
            spk_ckpt = next(root_path.rglob("embedding_model.ckpt"), None)
            vad_onnx = next(root_path.rglob("silero_vad.onnx"), None)
            vad_jit = next(root_path.rglob("silero_vad.jit"), None)

            # If not found, search in Hugging Face cache
            if not spk_ckpt:
                self.progress.emit("embedding_model.ckpt not found in specified folder, checking HF cache...")
                spk_ckpt = self._search_hf_cache("embedding_model.ckpt")

            if not (vad_onnx or vad_jit):
                self.progress.emit("Silero VAD not found in specified folder, checking HF cache...")
                if vad_onnx is None:
                    vad_onnx = self._search_hf_cache("silero_vad.onnx")
                if vad_jit is None:
                    vad_jit = self._search_hf_cache("silero_vad.jit")

            if not spk_ckpt:
                self.finished.emit(False, "Error: 'embedding_model.ckpt' not found in specified folder or HF cache.", None, None)
                return
            if not (vad_onnx or vad_jit):
                self.finished.emit(False, "Error: Silero VAD (.onnx/.jit) not found in specified folder or HF cache.", None, None)
                return

            self.progress.emit("Initializing VAD engine...")
            # Prefer JIT model (supports PyTorch/MPS) over ONNX (CPU only)
            if vad_jit:
                vad_model = torch.jit.load(str(vad_jit), map_location=DEVICE)
            elif vad_onnx:
                self.progress.emit("Using ONNX VAD model (CPU only). JIT model recommended for GPU acceleration.")
                vad_model = ort.InferenceSession(str(vad_onnx))
            else:
                self.finished.emit(False, "Error: No VAD model found.", None, None)
                return

            # Anchor the model directory to the recursively found path
            model_dir = str(spk_ckpt.parent)
            self.progress.emit(f"Loading Speaker Recognition from {spk_ckpt.parent.name}... (Device: {DEVICE})")

            # This forces SpeechBrain to load hyperparams.yaml from YOUR folder
            try:
                spk_model = SpeakerRecognition.from_hparams(
                    source=model_dir,
                    savedir=model_dir,
                    run_opts={"device": DEVICE}
                )
            except Exception as model_err:
                # Fallback to CPU if MPS fails
                self.progress.emit(f"Failed to load on {DEVICE}: {str(model_err)}. Retrying with CPU...")
                spk_model = SpeakerRecognition.from_hparams(
                    source=model_dir,
                    savedir=model_dir,
                    run_opts={"device": "cpu"}
                )
            self.finished.emit(True, "Offline Ready.", vad_model, spk_model)
        except Exception as e:
            self.finished.emit(False, f"Internal Error: {str(e)}", None, None)


class AnalysisWorker(QThread):
    finished, progress, error = pyqtSignal(list, list), pyqtSignal(str), pyqtSignal(str)

    def __init__(self, waveform, sr, ref_regions, vad_model, spk_model):
        super().__init__()
        self.waveform, self.sr, self.ref_regions, self.vad_model, self.spk_model = waveform, sr, ref_regions, vad_model, spk_model
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            self.progress.emit("Extracting Speaker Embedding...")
            embeddings = []
            for idx, (s, e) in enumerate(self.ref_regions):
                if self._is_cancelled: self.progress.emit("Extracting Speaker Embedding Cancelled."); return
                ref_seg = self.waveform[int(s * self.sr):int(e * self.sr)]
                if len(ref_seg) < 160: continue
                try:
                    # Use float32 for MPS compatibility
                    ref_tensor = torch.tensor(ref_seg, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    # Ensure model is on same device
                    if hasattr(self.spk_model, 'device') and self.spk_model.device != ref_tensor.device:
                        self.spk_model.device = ref_tensor.device
                    emb = self.spk_model.encode_batch(ref_tensor).squeeze().cpu().numpy()
                    embeddings.append(emb / np.linalg.norm(emb))
                    self.progress.emit(f"Extracting Speaker Embedding... ({idx+1}/{len(self.ref_regions)})")
                except Exception as emb_err:
                    self.progress.emit(f"Error processing segment {idx+1}: {str(emb_err)}")
                    raise
            if not embeddings: raise ValueError("No valid reference regions.")
            ref_emb = np.mean(embeddings, axis=0);
            ref_emb /= np.linalg.norm(ref_emb)

            self.progress.emit("Running Voice Activity Detection...")
            # ONNX models (InferenceSession) only support CPU; JIT models support MPS/CUDA
            if isinstance(self.vad_model, ort.InferenceSession):
                waveform_tensor = torch.tensor(self.waveform, dtype=torch.float32, device='cpu')
            else:
                waveform_tensor = torch.tensor(self.waveform, dtype=torch.float32, device=DEVICE)
            speech_ts = get_speech_timestamps(waveform_tensor, self.vad_model, sampling_rate=self.sr)
            vad_segments = [(ts['start'], ts['end']) for ts in speech_ts]

            self.progress.emit("Comparing Similarities...")
            similarities = []
            for idx, (s, e) in enumerate(vad_segments):
                if self._is_cancelled: return
                try:
                    # Use float32 for MPS compatibility
                    seg_tensor = torch.tensor(self.waveform[s:e], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    seg_emb = self.spk_model.encode_batch(seg_tensor).squeeze().cpu().numpy()
                    similarities.append(np.dot(ref_emb, seg_emb / np.linalg.norm(seg_emb)))
                    if idx % 100 == 0:
                        self.progress.emit(f"Comparing Similarities... ({idx+1}/{len(vad_segments)})")
                except Exception as sim_err:
                    self.progress.emit(f"Error at segment {idx}: {str(sim_err)}")
                    raise
            if not self._is_cancelled: self.finished.emit(vad_segments, similarities)
        except Exception as e:
            self.error.emit(str(e))


class WaveformWidget(QWidget):
    def __init__(self, main_app, parent=None):
        super().__init__(parent)
        self.main_app, self.samples, self.duration, self.regions = main_app, None, 0.0, []
        self.playback_position, self.zoom_factor, self.contrast = -1.0, 1.0, 1.0
        self.spectrogram_img, self.drag_start_pos, self.drag_current_x = None, None, None
        self.interaction_mode = None
        self.active_idx = -1
        self.resize_edge = None  # 'left', 'right', or None
        self.hovered_edge = None
        self.edge_threshold = 10  # Pixel threshold for edge detection
        self.setMouseTracking(True);
        self.setMinimumHeight(400)

    def set_audio(self, samples, duration):
        self.samples, self.duration = samples, duration
        self.regions.clear();
        self.generate_spectrogram();
        self.update()

    def generate_spectrogram(self):
        if self.samples is None: return
        n_fft, hop_length = 1024, 512
        data = self.samples[:2000000] if len(self.samples) > 2000000 else self.samples
        spec = [np.abs(np.fft.rfft(data[i:i + n_fft] * np.hanning(n_fft))) for i in
                range(0, len(data) - n_fft, hop_length)]
        spec = np.log1p(np.array(spec).T * self.contrast)
        if spec.size == 0: return
        norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        img_data = np.zeros((norm.shape[0], norm.shape[1], 4), dtype=np.uint8)
        img_data[..., 0] = (norm * 255).astype(np.uint8)
        img_data[..., 1] = (np.sin(norm * np.pi) * 255).astype(np.uint8)
        img_data[..., 2] = ((1 - norm) * 150).astype(np.uint8)
        img_data[..., 3] = 255
        self.spectrogram_img = QImage(np.ascontiguousarray(img_data).tobytes(), norm.shape[1], norm.shape[0],
                                      QImage.Format.Format_RGBA8888)

    def get_view_width(self):
        return int(self.width() * self.zoom_factor)

    def t_to_x(self, t):
        return int((t / self.duration) * self.get_view_width()) if self.duration > 0 else 0

    def x_to_t(self, x):
        return (x / self.get_view_width()) * self.duration if self.get_view_width() > 0 else 0

    def _detect_hovered_edge(self, x):
        """Detect which edge is being hovered over."""
        for i, (s, e) in enumerate(self.regions):
            xs, xe = self.t_to_x(s), self.t_to_x(e)
            if abs(x - xs) <= self.edge_threshold:
                return 'left', i
            if abs(x - xe) <= self.edge_threshold:
                return 'right', i
        return None, -1

    def mousePressEvent(self, event):
        if self.samples is None: return
        x = event.position().x()
        self.drag_start_pos = x

        # Check if clicking on region edges for resize
        edge, idx = self._detect_hovered_edge(x)
        if edge:
            self.interaction_mode = 'resize'
            self.active_idx = idx
            self.resize_edge = edge
            self.drag_start_region = list(self.regions[idx])
            self.drag_start_time = self.x_to_t(x)
            return

        # Check if clicking inside a region for move
        for i, (s, e) in enumerate(self.regions):
            xs, xe = self.t_to_x(s), self.t_to_x(e)
            if xs < x < xe:
                self.interaction_mode = 'move'
                self.active_idx = i
                self.drag_start_region = list(self.regions[i])
                self.update()
                return

        # Otherwise, draw new region
        self.interaction_mode = 'draw'
        self.active_idx = -1
        self.drag_start_time = self.x_to_t(x)
        self.drag_current_x = x
        self.update()

    def mouseMoveEvent(self, event):
        if self.samples is None: return
        x = event.position().x()

        # Update cursor on hover (when not dragging)
        if self.drag_start_pos is None:
            edge, idx = self._detect_hovered_edge(x)
            if edge == 'left':
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                self.hovered_edge = 'left'
            elif edge == 'right':
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                self.hovered_edge = 'right'
            elif any(self.t_to_x(s) < x < self.t_to_x(e) for s, e in self.regions):
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                self.hovered_edge = None
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
                self.hovered_edge = None
            return

        # Handle dragging operations
        if self.interaction_mode == 'draw':
            self.drag_current_x = x
        elif self.interaction_mode == 'move':
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            dt = self.x_to_t(x) - self.x_to_t(self.drag_start_pos)
            s, e = self.drag_start_region
            new_s = max(0, s + dt)
            new_e = min(self.duration, e + dt)
            # Ensure minimum duration
            if new_e - new_s >= 0.05:
                self.regions[self.active_idx] = (new_s, new_e)
        elif self.interaction_mode == 'resize':
            s, e = self.drag_start_region
            if self.resize_edge == 'left':
                new_s = max(0, min(self.x_to_t(x), e - 0.05))
                self.regions[self.active_idx] = (new_s, e)
            elif self.resize_edge == 'right':
                new_e = min(self.duration, max(self.x_to_t(x), s + 0.05))
                self.regions[self.active_idx] = (s, new_e)
        self.update()

    def mouseReleaseEvent(self, event):
        if self.interaction_mode == 'draw' and self.drag_start_pos is not None:
            s, e = sorted([self.drag_start_time, self.x_to_t(event.position().x())])
            if e - s > 0.05:
                self.regions.append((s, e))
                self.main_app.save_history()
        elif self.interaction_mode in ['move', 'resize']:
            self.main_app.save_history()

        # Reset state
        self.drag_start_pos = None
        self.drag_current_x = None
        self.interaction_mode = None
        self.resize_edge = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def paintEvent(self, event):
        if self.samples is None: return
        painter = QPainter(self);
        vw, h = self.get_view_width(), self.height();
        mid_h = h // 2
        painter.fillRect(event.rect(), Qt.GlobalColor.black)
        if self.spectrogram_img: painter.drawImage(QRect(0, 0, vw, mid_h), self.spectrogram_img)
        painter.setPen(QColor(0, 255, 255))
        step = max(1, len(self.samples) // (vw * 2))
        vals = self.samples[np.arange(0, len(self.samples), step)]
        y_scale = (mid_h // 1.5) / (np.max(np.abs(vals)) + 1e-6)
        for i in range(len(vals) - 1):
            painter.drawLine(int((i * step / len(self.samples)) * vw), int(h * 0.75 - vals[i] * y_scale),
                             int(((i + 1) * step / len(self.samples)) * vw), int(h * 0.75 - vals[i + 1] * y_scale))
        # Draw regions with resize handles
        for i, (s, e) in enumerate(self.regions):
            xs, xe = self.t_to_x(s), self.t_to_x(e)
            painter.setPen(QColor(30, 144, 255, 220));
            painter.setBrush(QColor(30, 144, 255, 60));
            painter.drawRect(xs, 0, xe - xs, h)

            # Draw resize handles on edges
            handle_width = 6
            handle_color = QColor(255, 255, 0, 200) if i == self.active_idx else QColor(255, 200, 0, 180)

            # Left handle
            painter.setPen(handle_color)
            painter.setBrush(handle_color)
            painter.drawRect(int(xs - handle_width // 2), 0, handle_width, h)

            # Right handle
            painter.drawRect(int(xe - handle_width // 2), 0, handle_width, h)
        # Draw dragging preview
        if self.drag_start_pos and self.drag_current_x and self.interaction_mode == 'draw':
            xs, xe = sorted([int(self.drag_start_pos), int(self.drag_current_x)])
            painter.setPen(QColor(46, 204, 113, 220));
            painter.setBrush(QColor(46, 204, 113, 80));
            painter.drawRect(xs, 0, xe - xs, h)
        # Draw playback cursor
        if self.playback_position >= 0:
            px = self.t_to_x(self.playback_position)
            painter.setPen(QColor(255, 0, 0));
            painter.drawLine(px, 0, px, h)


class SpeakerCleanerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Speaker Cleaner (Offline)");
        self.resize(1200, 1000)
        self.input_file, self.waveform, self.sr = None, None, SAMPLE_RATE
        self.history, self.history_idx = [], -1
        self.vad_model, self.spk_model, self.playback_process, self.analysis_worker = None, None, None, None
        self.vad_segments, self.similarities = [], []
        self.vad_model_name = ""
        self.spk_model_name = ""
        self.cursor_timer = QTimer();
        self.cursor_timer.timeout.connect(self.update_cursor)
        self.init_ui()

    def init_ui(self):
        central = QWidget();
        self.setCentralWidget(central);
        layout = QVBoxLayout(central)
        model_group = QGroupBox("Strict Local Model Setup")
        model_layout = QHBoxLayout()
        self.model_path_input = QLineEdit();
        self.model_path_input.setText(os.path.join(os.getcwd(), "models"))
        self.btn_browse = QPushButton("Browse");
        self.btn_browse.clicked.connect(self.browse_models);
        # self.btn_browse.setStyleSheet(BUTTON_STYLE)
        self.btn_configure = QPushButton("Configure");
        self.btn_configure.clicked.connect(self.start_config);
        # self.btn_configure.setStyleSheet(BUTTON_STYLE)
        model_layout.addWidget(self.model_path_input);
        model_layout.addWidget(self.btn_browse);
        model_layout.addWidget(self.btn_configure)
        model_group.setLayout(model_layout);
        layout.addWidget(model_group)

        bl = QHBoxLayout()
        self.load_b, self.play_b, self.stop_b = QPushButton("Load Audio"), QPushButton("Play"), QPushButton("Stop")
        self.analyze_b, self.undo_b, self.clear_b = QPushButton("Analyze"), QPushButton("Undo"), QPushButton(
            "Clear Refs")
        self.save_b = QPushButton("Save Cleaned");
        self.analyze_b.setEnabled(False)
        for b in [self.load_b, self.play_b, self.stop_b, self.analyze_b, self.undo_b, self.clear_b]:
            bl.addWidget(b)
        # self.save_b.setStyleSheet(SAVE_BUTTON_STYLE);
        bl.addWidget(self.save_b);
        layout.addLayout(bl)

        ctrl = QHBoxLayout();
        self.thresh_label = QLabel(f"Similarity: {DEFAULT_THRESHOLD:.2f}")
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal);
        self.thresh_slider.setRange(0, 100);
        self.thresh_slider.setValue(int(DEFAULT_THRESHOLD * 100))
        self.thresh_slider.valueChanged.connect(self.on_thresh_changed)
        self.zoom_s = QSlider(Qt.Orientation.Horizontal);
        self.zoom_s.setRange(10, 100);
        self.zoom_s.setValue(10);
        self.zoom_s.valueChanged.connect(self.update_view)
        self.contrast_s = QSlider(Qt.Orientation.Horizontal);
        self.contrast_s.setRange(1, 100);
        self.contrast_s.setValue(10);
        self.contrast_s.valueChanged.connect(self.update_view)
        for w in [self.thresh_label, self.thresh_slider, QLabel("Zoom:"), self.zoom_s, QLabel("Contrast:"),
                  self.contrast_s]: ctrl.addWidget(w)
        layout.addLayout(ctrl)

        self.scroll = QScrollArea();
        self.wv = WaveformWidget(self);
        self.scroll.setWidget(self.wv);
        self.scroll.setWidgetResizable(True);
        layout.addWidget(self.scroll)
        self.seg_list = QListWidget();
        self.seg_list.setMinimumHeight(150);
        layout.addWidget(self.seg_list)
        self.status = QStatusBar();
        self.setStatusBar(self.status)
        # Progress label in the middle
        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet("font-weight: bold; padding: 2px 10px;")
        self.status.addWidget(self.progress_label, 2)
        # Model info labels (permanent on the right)
        self.vad_model_label = QLabel("VAD: Not Loaded")
        self.spk_model_label = QLabel("Speaker: Not Loaded")
        self.status.addPermanentWidget(self.vad_model_label, 1)
        self.status.addPermanentWidget(self.spk_model_label, 1)
        self.load_b.clicked.connect(self.open_audio);
        self.play_b.clicked.connect(self.play_audio);
        self.stop_b.clicked.connect(self.stop_audio)
        self.analyze_b.clicked.connect(self.handle_analysis);
        self.undo_b.clicked.connect(self.undo);
        self.clear_b.clicked.connect(self.clear_refs);
        self.save_b.clicked.connect(self.save_cleaned)

    def browse_models(self):
        p = QFileDialog.getExistingDirectory(self, "Select Folder");
        self.model_path_input.setText(p if p else self.model_path_input.text())

    def start_config(self):
        self.btn_configure.setEnabled(False);
        self.config_worker = ConfigWorker(self.model_path_input.text())
        self.config_worker.progress.connect(self.progress_label.setText);
        self.config_worker.finished.connect(self.on_config_done);
        self.config_worker.start()

    def on_config_done(self, success, msg, vad, spk):
        self.btn_configure.setEnabled(True)
        if success:
            self.vad_model, self.spk_model = vad, spk; self.analyze_b.setEnabled(True)
            # Extract model names and display in status bar
            vad_type = "ONNX" if isinstance(vad, ort.InferenceSession) else "JIT"
            self.vad_model_name = f"VAD ({vad_type}, Device: {DEVICE})"
            self.spk_model_name = f"Speaker (Device: {DEVICE})"
            self.vad_model_label.setText(self.vad_model_name)
            self.spk_model_label.setText(self.spk_model_name)
            self.progress_label.setText("Configuration Successful.")
        else:
            QMessageBox.critical(self, "Error", msg)
            self.progress_label.setText("Configuration Failed.")

    def handle_analysis(self):
        if self.analysis_worker and self.analysis_worker.isRunning():
            self.analysis_worker.cancel(); self.analyze_b.setText("Analyze") # ; self.analyze_b.setStyleSheet(BUTTON_STYLE)
        else:
            if self.waveform is None or not self.wv.regions: return QMessageBox.warning(self, "Warning",
                                                                                        "Select Reference Regions.")
            self.analyze_b.setText("Cancel");
            # self.analyze_b.setStyleSheet(CANCEL_BUTTON_STYLE)
            self.analysis_worker = AnalysisWorker(self.waveform, SAMPLE_RATE, self.wv.regions, self.vad_model,
                                                  self.spk_model)
            self.analysis_worker.progress.connect(self.progress_label.setText);
            self.analysis_worker.error.connect(lambda msg: QMessageBox.critical(self, "Error", msg))
            self.analysis_worker.finished.connect(self.on_success);
            self.analysis_worker.start()

    def on_success(self, vad, sim):
        self.vad_segments, self.similarities = vad, sim;
        self.update_list();
        self.analyze_b.setText("Analyze");
        # self.analyze_b.setStyleSheet(BUTTON_STYLE)

    def open_audio(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "Audio Files (*.wav *.mp3 *.flac)")
        if p:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp: t_path = tmp.name
            subprocess.run(['ffmpeg', '-i', p, '-ar', '16000', '-ac', '1', '-y', t_path], check=True,
                           capture_output=True)
            self.waveform, _ = sf.read(t_path);
            os.unlink(t_path);
            self.input_file = p
            self.wv.set_audio(self.waveform.astype(np.float32), len(self.waveform) / SAMPLE_RATE);
            self.save_history()

    def update_view(self):
        self.wv.zoom_factor, self.wv.contrast = self.zoom_s.value() / 10, self.contrast_s.value() / 10; self.wv.generate_spectrogram(); self.wv.update()

    def save_history(self):
        self.history = self.history[:self.history_idx + 1] + [
            list(self.wv.regions)]; self.history_idx += 1; self.undo_b.setEnabled(self.history_idx > 0)

    def undo(self):
        if self.history_idx > 0: self.history_idx -= 1; self.wv.regions = list(
            self.history[self.history_idx]); self.wv.update(); self.undo_b.setEnabled(self.history_idx > 0)

    def clear_refs(self):
        self.wv.regions.clear(); self.wv.update(); self.save_history()

    def on_thresh_changed(self):
        self.thresh_label.setText(f"Similarity: {self.thresh_slider.value() / 100:.2f}"); self.update_list()

    def update_list(self):
        self.seg_list.clear();
        thresh = self.thresh_slider.value() / 100
        for i, ((s, e), sim) in enumerate(zip(self.vad_segments, self.similarities)):
            item = QListWidgetItem(f"[{s / SAMPLE_RATE:.1f}s-{e / SAMPLE_RATE:.1f}s] Score: {sim:.3f}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable);
            item.setCheckState(Qt.CheckState.Checked if sim < thresh else Qt.CheckState.Unchecked)
            self.seg_list.addItem(item)

    def play_audio(self):
        if self.input_file: self.stop_audio(); self.playback_process = subprocess.Popen(
            ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet",
             self.input_file]); self.playback_start = np.datetime64('now'); self.cursor_timer.start(50)

    def stop_audio(self):
        if self.playback_process: self.playback_process.terminate(); self.playback_process = None; self.cursor_timer.stop(); self.wv.playback_position = -1.0; self.wv.update()

    def update_cursor(self):
        if self.playback_process and self.playback_process.poll() is None:
            self.wv.playback_position = (np.datetime64('now') - self.playback_start) / np.timedelta64(1,
                                                                                                      's'); self.wv.update()
        else:
            self.stop_audio()

    def save_cleaned(self):
        if self.waveform is None: return
        p, _ = QFileDialog.getSaveFileName(self, "Save Cleaned Audio", "", "WAV (*.wav)")
        if p:
            c = np.copy(self.waveform)
            for i in range(self.seg_list.count()):
                if self.seg_list.item(i).checkState() == Qt.CheckState.Checked: c[
                    self.vad_segments[i][0]:self.vad_segments[i][1]] = 0.0
            sf.write(p, c, SAMPLE_RATE);
            QMessageBox.information(self, "Success", "Saved.")


if __name__ == "__main__":
    app = QApplication(sys.argv);
    window = SpeakerCleanerApp();
    window.show();
    sys.exit(app.exec())