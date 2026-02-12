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
import ffmpeg  # Requires: pip install ffmpeg-python

# --- SUPPRESS MPS WARNINGS ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore", message=".*Unknown device for graph fuser.*")
warnings.filterwarnings("ignore", message=".*mps.*fallback.*")
if torch.backends.mps.is_available():
    warnings.filterwarnings("ignore", module="torch._inductor.codegen")


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

# --- MANDATORY OFFLINE & COMPATIBILITY PATCHES ---
import huggingface_hub
import torchaudio

_old_hf_hub_download = huggingface_hub.hf_hub_download


def _patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token', None)
    kwargs['local_files_only'] = True
    return _old_hf_hub_download(*args, **kwargs)


huggingface_hub.hf_hub_download = _patched_hf_hub_download

warnings.filterwarnings("ignore", category=FutureWarning, module="speechbrain.utils.autocast")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain.utils.parameter_transfer")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*sipPyTypeDict.*")
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# ------------------------------------------------

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QGroupBox, QSlider, QProgressBar, QScrollArea,
    QStatusBar, QLineEdit, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect, QObject
from PyQt6.QtGui import QPainter, QColor, QImage, QCursor

from silero_vad import get_speech_timestamps
from speechbrain.inference import SpeakerRecognition
import onnxruntime as ort

SAMPLE_RATE = 16000
DEFAULT_THRESHOLD = 0.4


class ConfigWorker(QThread):
    finished = pyqtSignal(bool, str, object, object)
    progress = pyqtSignal(str)

    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir

    def run(self):
        try:
            root_path = Path(self.root_dir)
            self.progress.emit("Scanning directory for local weights...")

            spk_ckpt = next(root_path.rglob("embedding_model.ckpt"), None)
            vad_onnx = next(root_path.rglob("silero_vad.onnx"), None)
            vad_jit = next(root_path.rglob("silero_vad.jit"), None)

            if not spk_ckpt:
                self.finished.emit(False, "Error: 'embedding_model.ckpt' not found.", None, None)
                return
            if not (vad_onnx or vad_jit):
                self.finished.emit(False, "Error: Silero VAD not found.", None, None)
                return

            self.progress.emit("Initializing VAD engine...")
            if vad_jit:
                vad_model = torch.jit.load(str(vad_jit), map_location=DEVICE)
            elif vad_onnx:
                vad_model = ort.InferenceSession(str(vad_onnx))
            else:
                self.finished.emit(False, "Error: No VAD model found.", None, None)
                return

            model_dir = str(spk_ckpt.parent)
            self.progress.emit(f"Loading Speaker Recognition... (Device: {DEVICE})")

            try:
                spk_model = SpeakerRecognition.from_hparams(
                    source=model_dir, savedir=model_dir, run_opts={"device": DEVICE}
                )
            except:
                spk_model = SpeakerRecognition.from_hparams(
                    source=model_dir, savedir=model_dir, run_opts={"device": "cpu"}
                )
            self.finished.emit(True, "Offline Ready.", vad_model, spk_model)
        except Exception as e:
            self.finished.emit(False, f"Internal Error: {str(e)}", None, None)


class FFmpegWorker(QObject):
    """
    Worker to handle FFmpeg processing for muting.
    Ported from audio_muter.py to handle complex silence generation.
    """
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_path, regions_to_mute, output_path, muting_method="harmonic_residual"):
        super().__init__()
        self.input_path = input_path
        self.regions = regions_to_mute  # These are the regions to SILENCE
        self.output_path = output_path
        self.muting_method = muting_method
        self._is_running = False

    def create_muted_segment(self, input_file, start, end, total_duration, sample_rate=44100, channels=1):
        duration = end - start

        # --- Harmonic Residual (Recommended for Singing) ---
        if self.muting_method == 'harmonic_residual':
            seg = ffmpeg.input(input_file, ss=start, t=duration)
            seg_high = ffmpeg.filter(seg, 'highpass', f='3000', poles=2)
            seg_full = ffmpeg.input(input_file, ss=start, t=duration)
            seg_full = ffmpeg.filter(seg_full, 'volume', volume='0.01')
            mixed = ffmpeg.filter([seg_high, seg_full], 'amix', inputs=2, weights='0.3 1')
            mixed = ffmpeg.filter(mixed, 'volume', volume='0.08')
            fade = min(0.025, duration / 4)
            mixed = ffmpeg.filter(mixed, 'afade', t='in', st=0, d=fade)
            mixed = ffmpeg.filter(mixed, 'afade', t='out', st=duration - fade, d=fade)
            return mixed

        # --- Adaptive Ducking ---
        elif self.muting_method == 'adaptive_ducking':
            seg = ffmpeg.input(input_file, ss=start, t=duration)
            seg = ffmpeg.filter(seg, 'compand', attacks='0.001', decays='0.1',
                                points='-80/-80|-60/-40|-40/-40|-20/-40|0/-40', volume='0')
            seg = ffmpeg.filter(seg, 'volume', volume='0.03')
            fade = min(0.02, duration / 4)
            seg = ffmpeg.filter(seg, 'afade', t='in', st=0, d=fade)
            seg = ffmpeg.filter(seg, 'afade', t='out', st=duration - fade, d=fade)
            return seg

        # --- Default/Original ---
        else:
            seg = ffmpeg.input(input_file, ss=start, t=duration)
            seg = ffmpeg.filter(seg, "volume", volume=0)
            return seg

    def run(self):
        self._is_running = True
        try:
            if not self.regions:
                # Nothing to mute, copy original
                stream = ffmpeg.input(self.input_path)
                stream = ffmpeg.output(stream, self.output_path, acodec="pcm_s16le")
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
            else:
                probe = ffmpeg.probe(self.input_path)
                audio_stream = next((s for s in probe["streams"] if s["codec_type"] == "audio"), None)
                total_dur = float(audio_stream["duration"])
                sr = int(audio_stream["sample_rate"])
                ch = int(audio_stream.get("channels", 1))

                regions_sorted = sorted(self.regions, key=lambda x: x[0])
                segments = []
                current = 0.0

                for start, end in regions_sorted:
                    # Keep segment before mute region
                    if start > current:
                        keep = ffmpeg.input(self.input_path, ss=current, t=start - current)
                        segments.append(keep)

                    # Create muted segment
                    muted = self.create_muted_segment(self.input_path, start, end, total_dur, sr, ch)
                    segments.append(muted)
                    current = end

                # Final keep segment
                if current < total_dur:
                    keep = ffmpeg.input(self.input_path, ss=current, t=total_dur - current)
                    segments.append(keep)

                if len(segments) == 1:
                    out_stream = segments[0]
                else:
                    out_stream = ffmpeg.concat(*segments, v=0, a=1)

                out = ffmpeg.output(out_stream, self.output_path, acodec="pcm_s16le", ar=sr, ac=ch)
                ffmpeg.run(out, overwrite_output=True, quiet=True)

            if self._is_running:
                self.finished.emit(self.output_path)
        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))
        finally:
            self._is_running = False


class AnalysisWorker(QThread):
    finished, progress, error = pyqtSignal(list, list), pyqtSignal(str), pyqtSignal(str)

    def __init__(self, waveform, sr, ref_regions, vad_model, spk_model):
        super().__init__()
        self.waveform, self.sr, self.ref_regions = waveform, sr, ref_regions
        self.vad_model, self.spk_model = vad_model, spk_model
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            self.progress.emit("Extracting Speaker Embedding...")
            embeddings = []
            min_ref_duration = int(0.5 * self.sr)  # Minimum 0.5 seconds for reliable embedding
            
            for idx, (s, e) in enumerate(self.ref_regions):
                if self._is_cancelled: return
                ref_seg = self.waveform[int(s * self.sr):int(e * self.sr)]
                if len(ref_seg) < min_ref_duration: continue
                try:
                    ref_tensor = torch.tensor(ref_seg, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    if hasattr(self.spk_model, 'device') and self.spk_model.device != ref_tensor.device:
                        self.spk_model.device = ref_tensor.device
                    emb = self.spk_model.encode_batch(ref_tensor).squeeze().cpu().numpy()
                    # Proper normalization
                    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                    embeddings.append(emb_norm)
                except Exception as ex:
                    print(f"Warning: Failed to process reference region {idx}: {ex}")
                    continue

            if not embeddings:
                self.error.emit("No valid reference regions found. Please select longer reference segments (at least 0.5s).")
                return

            # Store multiple reference embeddings for better matching
            self.ref_embeddings = embeddings
            # Also compute average for backward compatibility
            ref_emb = np.mean(embeddings, axis=0)
            ref_emb = ref_emb / (np.linalg.norm(ref_emb) + 1e-8)

            self.progress.emit("Running VAD...")
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
                seg_tensor = torch.tensor(self.waveform[s:e], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                seg_emb = self.spk_model.encode_batch(seg_tensor).squeeze().cpu().numpy()
                # Ensure proper normalization for cosine similarity
                seg_emb_norm = seg_emb / (np.linalg.norm(seg_emb) + 1e-8)
                similarity = np.dot(ref_emb, seg_emb_norm)
                # Clamp to valid range [-1, 1]
                similarity = np.clip(similarity, -1.0, 1.0)
                similarities.append(similarity)

            if not self._is_cancelled: self.finished.emit(vad_segments, similarities)
        except Exception as e:
            self.error.emit(str(e))


class WaveformWidget(QWidget):
    # Signal emitted when a region is moved or resized
    region_changed = pyqtSignal()

    def __init__(self, main_app, parent=None):
        super().__init__(parent)
        self.main_app = main_app
        self.samples = None
        self.duration = 0.0
        self.regions = []  # List of tuples (start, end)

        self.playback_position = -1.0
        self.zoom_factor = 1.0
        self.contrast = 1.0
        self.spectrogram_img = None
        self.drag_start_pos = None
        self.drag_current_x = None
        self.interaction_mode = None
        self.active_idx = -1
        self.resize_edge = None
        self.edge_threshold = 10
        self.setMouseTracking(True)
        self.setMinimumHeight(400)

    def set_audio(self, samples, duration):
        self.samples, self.duration = samples, duration
        self.regions.clear()
        self.generate_spectrogram()
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
        for i, (s, e) in enumerate(self.regions):
            xs, xe = self.t_to_x(s), self.t_to_x(e)
            if abs(x - xs) <= self.edge_threshold: return 'left', i
            if abs(x - xe) <= self.edge_threshold: return 'right', i
        return None, -1

    def mousePressEvent(self, event):
        if self.samples is None: return
        x = event.position().x()
        self.drag_start_pos = x
        edge, idx = self._detect_hovered_edge(x)
        if edge:
            self.interaction_mode, self.active_idx, self.resize_edge = 'resize', idx, edge
            self.drag_start_region = list(self.regions[idx])
            self.drag_start_time = self.x_to_t(x)
            return
        for i, (s, e) in enumerate(self.regions):
            xs, xe = self.t_to_x(s), self.t_to_x(e)
            if xs < x < xe:
                self.interaction_mode, self.active_idx = 'move', i
                self.drag_start_region = list(self.regions[i])
                self.update();
                return
        self.interaction_mode, self.active_idx = 'draw', -1
        self.drag_start_time, self.drag_current_x = self.x_to_t(x), x
        self.update()

    def mouseMoveEvent(self, event):
        if self.samples is None: return
        x = event.position().x()
        if self.drag_start_pos is None:
            edge, _ = self._detect_hovered_edge(x)
            if edge:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif any(self.t_to_x(s) < x < self.t_to_x(e) for s, e in self.regions):
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
            return

        if self.interaction_mode == 'draw':
            self.drag_current_x = x
        elif self.interaction_mode == 'move':
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            dt = self.x_to_t(x) - self.x_to_t(self.drag_start_pos)
            s, e = self.drag_start_region
            new_s, new_e = max(0, s + dt), min(self.duration, e + dt)
            if new_e - new_s >= 0.05: self.regions[self.active_idx] = (new_s, new_e)
            self.region_changed.emit()
        elif self.interaction_mode == 'resize':
            s, e = self.drag_start_region
            if self.resize_edge == 'left':
                self.regions[self.active_idx] = (max(0, min(self.x_to_t(x), e - 0.05)), e)
            elif self.resize_edge == 'right':
                self.regions[self.active_idx] = (s, min(self.duration, max(self.x_to_t(x), s + 0.05)))
            self.region_changed.emit()
        self.update()

    def mouseReleaseEvent(self, event):
        if self.interaction_mode == 'draw' and self.drag_start_pos is not None:
            s, e = sorted([self.drag_start_time, self.x_to_t(event.position().x())])
            if e - s > 0.05:
                self.regions.append((s, e))
                self.region_changed.emit()
        self.drag_start_pos = None;
        self.interaction_mode = None
        self.setCursor(Qt.CursorShape.ArrowCursor);
        self.update()

    def paintEvent(self, event):
        if self.samples is None: return
        painter = QPainter(self)
        vw, h, mid_h = self.get_view_width(), self.height(), self.height() // 2
        painter.fillRect(event.rect(), Qt.GlobalColor.black)

        if self.spectrogram_img: painter.drawImage(QRect(0, 0, vw, mid_h), self.spectrogram_img)

        painter.setPen(QColor(0, 255, 255))
        step = max(1, len(self.samples) // (vw * 2))
        vals = self.samples[np.arange(0, len(self.samples), step)]
        y_scale = (mid_h // 1.5) / (np.max(np.abs(vals)) + 1e-6)
        for i in range(len(vals) - 1):
            painter.drawLine(int((i * step / len(self.samples)) * vw), int(h * 0.75 - vals[i] * y_scale),
                             int(((i + 1) * step / len(self.samples)) * vw), int(h * 0.75 - vals[i + 1] * y_scale))

        for i, (s, e) in enumerate(self.regions):
            xs, xe = self.t_to_x(s), self.t_to_x(e)
            # Set pen for red border
            painter.setPen(QColor(220, 20, 60, 220))  # Crimson red border
            # Set brush for semi-transparent red fill
            painter.setBrush(QColor(255, 99, 71, 100))  # Tomato red fill with transparency
            painter.drawRect(xs, 0, xe - xs, h)
            handle_color = QColor(255, 255, 0, 200) if i == self.active_idx else QColor(255, 200, 0, 180)
            painter.setPen(handle_color);
            painter.setBrush(handle_color)
            painter.drawRect(int(xs - 3), 0, 6, h);
            painter.drawRect(int(xe - 3), 0, 6, h)

        if self.drag_start_pos and self.drag_current_x and self.interaction_mode == 'draw':
            xs, xe = sorted([int(self.drag_start_pos), int(self.drag_current_x)])
            painter.setPen(QColor(46, 204, 113, 220));
            painter.setBrush(QColor(46, 204, 113, 80))
            painter.drawRect(xs, 0, xe - xs, h)

        if self.playback_position >= 0:
            px = self.t_to_x(self.playback_position)
            painter.setPen(QColor(255, 0, 0));
            painter.drawLine(px, 0, px, h)


class SpeakerCleanerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Speaker Cleaner (Offline)")
        self.resize(1200, 1000)
        self.input_file, self.waveform, self.sr = None, None, SAMPLE_RATE
        self.history, self.history_idx = [], -1
        self.vad_model, self.spk_model, self.playback_process, self.analysis_worker = None, None, None, None
        self.ffmpeg_worker, self.ffmpeg_thread = None, None

        self.cursor_timer = QTimer()
        self.cursor_timer.timeout.connect(self.update_cursor)
        self.init_ui()

    def init_ui(self):
        central = QWidget();
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Model Config
        model_group = QGroupBox("Strict Local Model Setup")
        model_layout = QHBoxLayout()
        self.model_path_input = QLineEdit();
        self.model_path_input.setText(os.path.join(os.getcwd(), "models"))
        self.btn_browse = QPushButton("Browse");
        self.btn_browse.clicked.connect(self.browse_models)
        self.btn_configure = QPushButton("Configure");
        self.btn_configure.clicked.connect(self.start_config)
        model_layout.addWidget(self.model_path_input);
        model_layout.addWidget(self.btn_browse);
        model_layout.addWidget(self.btn_configure)
        model_group.setLayout(model_layout);
        layout.addWidget(model_group)

        # Controls
        ctrl_layout = QHBoxLayout()
        self.load_b = QPushButton("Load Audio")
        self.analyze_b = QPushButton("Analyze")
        self.clear_b = QPushButton("Reset Regions")

        self.load_b.clicked.connect(self.open_audio)
        self.analyze_b.clicked.connect(self.handle_analysis);
        self.analyze_b.setEnabled(False)
        self.clear_b.clicked.connect(self.reset_regions)

        ctrl_layout.addWidget(self.load_btn_w());
        ctrl_layout.addWidget(self.analyze_b);
        ctrl_layout.addWidget(self.clear_b)
        layout.addLayout(ctrl_layout)

        # Muting Method Selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Muting Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Harmonic Residual (Recommended-Singing)",
            "Adaptive Compression",
            "Original Silence"
        ])
        method_layout.addWidget(self.method_combo)
        layout.addLayout(method_layout)

        # Threshold
        thresh_layout = QHBoxLayout()
        self.thresh_label = QLabel(f"Similarity Threshold: {DEFAULT_THRESHOLD:.2f}")
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 100);
        self.thresh_slider.setValue(int(DEFAULT_THRESHOLD * 100))
        self.thresh_slider.valueChanged.connect(self.on_thresh_changed)
        thresh_layout.addWidget(self.thresh_label);
        thresh_layout.addWidget(self.thresh_slider)
        layout.addLayout(thresh_layout)

        # Advanced Similarity Options
        advanced_layout = QHBoxLayout()
        self.similarity_mode = QComboBox()
        self.similarity_mode.addItems([
            "Standard Cosine (Default)",
            "Multi-reference Max",
            "Dynamic Threshold"
        ])
        self.similarity_mode.setCurrentIndex(1)  # Default to multi-reference
        advanced_layout.addWidget(QLabel("Similarity Mode:"))
        advanced_layout.addWidget(self.similarity_mode)
        layout.addLayout(advanced_layout)

        # Visualization Controls
        viz_layout = QHBoxLayout()
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 100)
        self.zoom_slider.setValue(10)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(1, 100)
        self.contrast_slider.setValue(10)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        viz_layout.addWidget(QLabel("Zoom:"))
        viz_layout.addWidget(self.zoom_slider)
        viz_layout.addWidget(QLabel("Contrast:"))
        viz_layout.addWidget(self.contrast_slider)
        layout.addLayout(viz_layout)

        # Waveform
        self.scroll = QScrollArea()
        self.wv = WaveformWidget(self)
        self.scroll.setWidget(self.wv);
        self.scroll.setWidgetResizable(True)
        # Connect waveform modification to list update
        self.wv.region_changed.connect(self.update_list_from_waveform)
        layout.addWidget(self.scroll)

        # Playback Controls
        play_layout = QHBoxLayout()
        self.play_orig_b = QPushButton("Play Original")
        self.play_clean_b = QPushButton("Preview Cleaned (Selected Regions)")
        self.stop_b = QPushButton("Stop")

        self.play_orig_b.clicked.connect(self.play_audio)
        self.play_clean_b.clicked.connect(self.play_cleaned_preview)
        self.stop_b.clicked.connect(self.stop_audio)

        play_layout.addWidget(self.play_orig_b);
        play_layout.addWidget(self.play_clean_b);
        play_layout.addWidget(self.stop_b)
        layout.addLayout(play_layout)

        # Results List & Save
        list_layout = QVBoxLayout()
        self.seg_list = QListWidget();
        self.seg_list.setMinimumHeight(150)
        self.seg_list.itemClicked.connect(self.preview_list_item)

        list_btns = QHBoxLayout()
        self.remove_item_b = QPushButton("Remove Selected Region")
        self.remove_item_b.clicked.connect(self.remove_selected_region)
        self.save_b = QPushButton("Save Cleaned Audio (Mute Unselected)")
        # self.save_b.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 5px;")
        self.save_b.clicked.connect(self.save_cleaned)

        list_btns.addWidget(self.remove_item_b);
        list_btns.addWidget(self.save_b)
        list_layout.addWidget(QLabel("Regions to KEEP (Reference + Above Threshold):"))
        list_layout.addWidget(self.seg_list)
        list_layout.addLayout(list_btns)
        layout.addLayout(list_layout)

        # Status
        self.status = QStatusBar();
        self.setStatusBar(self.status)
        self.progress_label = QLabel("Ready");
        self.status.addWidget(self.progress_label)

    def load_btn_w(self):
        return self.load_b  # Helper

    # --- Worker Management ---
    def start_config(self):
        self.btn_configure.setEnabled(False)
        self.config_worker = ConfigWorker(self.model_path_input.text())
        self.config_worker.progress.connect(self.progress_label.setText)
        self.config_worker.finished.connect(self.on_config_done)
        self.config_worker.start()

    def on_config_done(self, success, msg, vad, spk):
        self.btn_configure.setEnabled(True)
        if success:
            self.vad_model, self.spk_model = vad, spk
            self.analyze_b.setEnabled(True)
            self.progress_label.setText("Models Loaded.")
        else:
            QMessageBox.critical(self, "Error", msg)

    # --- Audio & Analysis ---
    def open_audio(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "Audio (*.wav *.mp3 *.flac)")
        if p:
            self.input_file = p
            # Convert to standard WAV for processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp: t_path = tmp.name
            subprocess.run(['ffmpeg', '-i', p, '-ar', str(SAMPLE_RATE), '-ac', '1', '-y', t_path], check=True,
                           capture_output=True)
            self.waveform, _ = sf.read(t_path)
            os.unlink(t_path)
            self.wv.set_audio(self.waveform.astype(np.float32), len(self.waveform) / SAMPLE_RATE)

    def handle_analysis(self):
        if not self.wv.regions: return QMessageBox.warning(self, "Warning",
                                                           "Select at least one reference region on the waveform.")
        self.analysis_worker = AnalysisWorker(self.waveform, SAMPLE_RATE, self.wv.regions, self.vad_model,
                                              self.spk_model)
        self.analysis_worker.progress.connect(self.progress_label.setText)
        self.analysis_worker.finished.connect(self.on_analysis_success)
        self.analysis_worker.start()

    def on_analysis_success(self, vad, sim):
        # Store similarities for display
        self._last_similarities = []
        
        # 1. Gather original reference regions (Keep them)
        keep_regions = list(self.wv.regions)
        # Add reference region scores (assume 1.0 for reference regions)
        self._last_similarities.extend([1.0] * len(self.wv.regions))

        # 2. Add regions above threshold
        thresh = self.thresh_slider.value() / 100
        matched_count = 0
        for (s_sample, e_sample), score in zip(vad, sim):
            if score >= thresh:
                s_sec, e_sec = s_sample / SAMPLE_RATE, e_sample / SAMPLE_RATE
                keep_regions.append((s_sec, e_sec))
                self._last_similarities.append(score)
                matched_count += 1

        # 3. Merge overlapping regions
        keep_regions.sort()
        merged = []
        merged_scores = []
        if keep_regions:
            curr_s, curr_e = keep_regions[0]
            score_indices = list(range(len(keep_regions)))
            curr_scores = [self._last_similarities[0]] if self._last_similarities else []
            
            for i, (s, e) in enumerate(keep_regions[1:], 1):
                if s <= curr_e:
                    curr_e = max(curr_e, e)
                    if i < len(self._last_similarities):
                        curr_scores.append(self._last_similarities[i])
                else:
                    merged.append((curr_s, curr_e))
                    # Use maximum similarity score for merged region
                    merged_scores.append(max(curr_scores) if curr_scores else 0.0)
                    curr_s, curr_e = s, e
                    curr_scores = [self._last_similarities[i]] if i < len(self._last_similarities) else []
            merged.append((curr_s, curr_e))
            merged_scores.append(max(curr_scores) if curr_scores else 0.0)

        # 4. Update Waveform and List
        self.wv.regions = merged
        self._last_similarities = merged_scores
        self.wv.update()
        self.update_list_from_waveform()
        self.progress_label.setText(f"Analysis complete. {len(merged)} regions kept (threshold: {thresh:.2f}). Matched: {matched_count}/{len(vad)}")

    def on_thresh_changed(self):
        self.thresh_label.setText(f"Similarity Threshold: {self.thresh_slider.value() / 100:.2f}")

    def update_zoom(self):
        self.wv.zoom_factor = self.zoom_slider.value() / 10.0
        self.wv.update()

    def update_contrast(self):
        self.wv.contrast = self.contrast_slider.value() / 10.0
        self.wv.generate_spectrogram()
        self.wv.update()

    # --- List & Waveform Sync ---
    def update_list_from_waveform(self):
        self.seg_list.clear()
        # Sort regions by time
        self.wv.regions.sort(key=lambda x: x[0])
        for i, (s, e) in enumerate(self.wv.regions):
            # Display region info with similarity score if available
            if hasattr(self, '_last_similarities') and i < len(self._last_similarities):
                score = self._last_similarities[i]
                duration = e - s
                item_text = f"Region {i + 1}: {s:.2f}s - {e:.2f}s (sim: {score:.3f}, dur: {duration:.2f}s)"
            else:
                item_text = f"Region {i + 1}: {s:.2f}s - {e:.2f}s"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, (s, e))
            # Color code based on similarity score
            if hasattr(self, '_last_similarities') and i < len(self._last_similarities):
                score = self._last_similarities[i]
                if score >= 0.8:
                    item.setBackground(QColor(144, 238, 144, 100))  # Light green for high similarity
                elif score >= 0.6:
                    item.setBackground(QColor(255, 255, 224, 100))  # Light yellow for medium similarity
                else:
                    item.setBackground(QColor(255, 182, 193, 100))  # Light pink for low similarity
            self.seg_list.addItem(item)

    def remove_selected_region(self):
        row = self.seg_list.currentRow()
        if row >= 0:
            del self.wv.regions[row]
            self.wv.update()
            self.update_list_from_waveform()

    def reset_regions(self):
        self.wv.regions.clear()
        self.wv.update()
        self.update_list_from_waveform()

    def preview_list_item(self, item):
        self.stop_audio()
        s, e = item.data(Qt.ItemDataRole.UserRole)
        # Play just this segment using numpy
        start_sample, end_sample = int(s * SAMPLE_RATE), int(e * SAMPLE_RATE)
        if self.waveform is not None:
            sd.play(self.waveform[start_sample:end_sample], SAMPLE_RATE)

    # --- Save & Play Cleaned Logic ---
    def calculate_mute_regions(self):
        """Calculate the gaps between keep regions."""
        keep = sorted(self.wv.regions)
        mute = []
        cursor = 0.0
        for s, e in keep:
            if s > cursor: mute.append((cursor, s))
            cursor = max(cursor, e)

        total_dur = len(self.waveform) / SAMPLE_RATE
        if cursor < total_dur: mute.append((cursor, total_dur))
        return mute

    def get_muting_method(self):
        methods = ['harmonic_residual', 'adaptive_ducking', 'original']
        return methods[self.method_combo.currentIndex()]

    def play_cleaned_preview(self):
        """Generate a temporary file using the FFmpeg worker logic and play it."""
        if not self.input_file or not self.wv.regions: return
        self.stop_audio()

        self.progress_label.setText("Generating preview...")
        # Create temp output file
        fd, temp_out = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        mute_regions = self.calculate_mute_regions()
        method = self.get_muting_method()

        # Use Worker in a thread to avoid freezing
        self.ffmpeg_thread = QThread()
        self.ffmpeg_worker = FFmpegWorker(self.input_file, mute_regions, temp_out, method)
        self.ffmpeg_worker.moveToThread(self.ffmpeg_thread)

        self.ffmpeg_thread.started.connect(self.ffmpeg_worker.run)
        self.ffmpeg_worker.finished.connect(lambda path: self.start_ffplay(path))
        self.ffmpeg_worker.finished.connect(self.ffmpeg_thread.quit)
        self.ffmpeg_worker.error.connect(lambda err: QMessageBox.critical(self, "Error", err))

        self.ffmpeg_thread.start()

    def start_ffplay(self, path):
        self.progress_label.setText("Playing preview...")
        self.playback_process = subprocess.Popen(
            ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", path]
        )
        self.cursor_timer.start(100)
        # Note: Temp file cleanup is tricky with async play, usually done on app exit or next play

    def save_cleaned(self):
        if not self.input_file: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Cleaned Audio", "", "WAV (*.wav)")
        if not path: return

        self.progress_label.setText("Processing audio...")
        self.save_b.setEnabled(False)

        mute_regions = self.calculate_mute_regions()
        method = self.get_muting_method()

        self.ffmpeg_thread = QThread()
        self.ffmpeg_worker = FFmpegWorker(self.input_file, mute_regions, path, method)
        self.ffmpeg_worker.moveToThread(self.ffmpeg_thread)

        self.ffmpeg_thread.started.connect(self.ffmpeg_worker.run)
        self.ffmpeg_worker.finished.connect(
            lambda p: [self.save_b.setEnabled(True), QMessageBox.information(self, "Success", f"Saved to {p}"),
                       self.progress_label.setText("Ready")])
        self.ffmpeg_worker.finished.connect(self.ffmpeg_thread.quit)
        self.ffmpeg_worker.error.connect(
            lambda err: [self.save_b.setEnabled(True), QMessageBox.critical(self, "Error", err)])

        self.ffmpeg_thread.start()

    # --- Standard Playback ---
    def play_audio(self):
        if self.input_file:
            self.stop_audio()
            self.playback_process = subprocess.Popen(
                ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", self.input_file]
            )
            self.playback_start = np.datetime64('now')
            self.cursor_timer.start(50)

    def stop_audio(self):
        if self.playback_process:
            self.playback_process.terminate()
            self.playback_process = None
        sd.stop()
        self.cursor_timer.stop()
        self.wv.playback_position = -1.0;
        self.wv.update()

    def update_cursor(self):
        if self.playback_process and self.playback_process.poll() is None:
            self.wv.playback_position = (np.datetime64('now') - self.playback_start) / np.timedelta64(1, 's')
            self.wv.update()
        else:
            self.stop_audio()

    def browse_models(self):
        p = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.model_path_input.setText(p if p else self.model_path_input.text())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpeakerCleanerApp()
    window.show()
    sys.exit(app.exec())