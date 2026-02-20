import gc
import os
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QSplitter,
)
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter1d

# --- SUPPRESS WARNINGS ---
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", message=".*Unknown device for graph fuser.*")
warnings.filterwarnings("ignore", message=".*mps.*fallback.*")


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


DEVICE = get_best_device()
SAMPLE_RATE = 44100
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# SpeechBrain expects this API in torchaudio; patch older/broken builds.
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
else:
    try:
        torchaudio.list_audio_backends()
    except Exception:
        torchaudio.list_audio_backends = lambda: ["soundfile"]

try:
    from speechbrain.inference import SpeakerRecognition

    SPEECHBRAIN_AVAILABLE = True
except Exception as e:
    print(f"SpeechBrain unavailable: {e}")
    SPEECHBRAIN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def clear_gpu_cache():
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


class SeparationWorker(QThread):
    finished = pyqtSignal(object, str, object)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, separator, vocal_audio_path, reference_segments):
        super().__init__()
        self.separator = separator
        self.vocal_audio_path = vocal_audio_path
        self.reference_segments = reference_segments
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True
        clear_gpu_cache()

    def run(self):
        try:
            if self._is_cancelled:
                return
            result, mask = self.separator.separate_target_voice(
                self.vocal_audio_path,
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


class HybridVoiceSeparator:
    def __init__(self, device=DEVICE):
        self.device = device
        self.speaker_model = None
        self.embedding_sample_rate = 16000
        self.similarity_threshold = 0.72
        self._load_speaker_model()

    def _load_speaker_model(self):
        if not SPEECHBRAIN_AVAILABLE:
            return

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

    def separate_target_voice(self, vocal_audio_path, reference_segments, progress_callback=None):
        if not reference_segments:
            raise ValueError("Please add at least one reference segment before separation.")

        if progress_callback:
            progress_callback("Step 1: Loading mono vocal audio...")

        clean_vocals, _ = librosa.load(vocal_audio_path, sr=SAMPLE_RATE, mono=True)

        if progress_callback:
            progress_callback("Step 2: Building target singer profile...")

        target_profile = self._extract_target_profile(reference_segments)

        if progress_callback:
            progress_callback("Step 3: Frame-level speaker matching...")

        speaker_mask = self._create_enhanced_speaker_mask(clean_vocals, target_profile, progress_callback)
        isolated_vocal = self._post_processing(clean_vocals, speaker_mask)
        return isolated_vocal, speaker_mask

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
        profile = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)], axis=0)
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

        energy_gate = np.percentile(energies, 20)
        voiced = energies >= max(energy_gate * 0.6, 1e-4)
        scores = np.where(voiced, scores, -1.0)

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

    def _post_processing(self, vocals, mask):
        isolated = vocals * mask
        max_amp = np.max(np.abs(isolated))
        if max_amp > 1.0:
            isolated = isolated / max_amp
        return isolated


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

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(event.rect(), self.palette().color(self.backgroundRole()))
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
            if i >= self.width():
                break
            painter.drawLine(i, mid - int(sub[i] * scale), i + 1, mid - int(sub[i + 1] * scale))


class ProSingerSeparatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Singer Separator (Mono Vocal Input)")
        self.resize(1400, 900)

        self.separator = HybridVoiceSeparator()
        self.vocal_audio_path = None
        self.reference_segments = []
        self.original_waveform = None
        self.processed_waveform = None
        self.latest_mask = None
        self.playback_process = None

        self.cursor_timer = QTimer()
        self.cursor_timer.timeout.connect(self.update_cursor)

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        main_tab = QWidget()
        tabs.addTab(main_tab, "Voice Separation")
        self.setup_main_tab(main_tab)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def setup_main_tab(self, parent):
        layout = QVBoxLayout(parent)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        input_group = QGroupBox("Input Files")
        input_layout = QVBoxLayout()

        mix_row = QHBoxLayout()
        self.mixed_label = QLabel("None")
        btn_mix = QPushButton("Load Vocal Mono")
        btn_mix.clicked.connect(self.load_vocal_audio)
        mix_row.addWidget(self.mixed_label)
        mix_row.addWidget(btn_mix)

        ref_row = QHBoxLayout()
        self.ref_label = QLabel("Refs: 0")
        btn_ref = QPushButton("Add Ref")
        btn_ref.clicked.connect(self.add_reference_segment)
        btn_clr = QPushButton("Clear")
        btn_clr.clicked.connect(lambda: [self.reference_segments.clear(), self.ref_label.setText("Refs: 0")])
        ref_row.addWidget(self.ref_label)
        ref_row.addWidget(btn_ref)
        ref_row.addWidget(btn_clr)

        input_layout.addLayout(mix_row)
        input_layout.addLayout(ref_row)
        input_group.setLayout(input_layout)
        left_layout.addWidget(input_group)

        proc_group = QGroupBox("Processing")
        proc_layout = QVBoxLayout()

        self.threshold_label = QLabel("Similarity Threshold: 0.72")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 95)
        self.threshold_slider.setValue(72)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)

        self.export_mask_checkbox = QCheckBox("Export mask diagnostic plot")
        self.export_mask_checkbox.setChecked(False)

        self.process_btn = QPushButton("Separate Target Voice")
        self.process_btn.clicked.connect(self.start_separation)

        proc_layout.addWidget(self.threshold_label)
        proc_layout.addWidget(self.threshold_slider)
        proc_layout.addWidget(self.export_mask_checkbox)
        proc_layout.addWidget(self.process_btn)
        proc_group.setLayout(proc_layout)
        left_layout.addWidget(proc_group)

        out_group = QGroupBox("Output")
        out_layout = QVBoxLayout()
        self.save_btn = QPushButton("Save Vocal")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_result)
        out_layout.addWidget(self.save_btn)
        out_group.setLayout(out_layout)
        left_layout.addWidget(out_group)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.waveform = WaveformDisplayWidget()
        scroll = QScrollArea()
        scroll.setWidget(self.waveform)
        scroll.setWidgetResizable(True)
        right_layout.addWidget(scroll)

        play_group = QGroupBox("Playback")
        play_layout = QHBoxLayout()
        self.btn_play_orig = QPushButton("Play Orig")
        self.btn_play_proc = QPushButton("Play Iso")
        self.btn_stop = QPushButton("Stop")
        self.btn_play_orig.clicked.connect(self.play_original)
        self.btn_play_proc.clicked.connect(self.play_processed)
        self.btn_stop.clicked.connect(self.stop_playback)
        play_layout.addWidget(self.btn_play_orig)
        play_layout.addWidget(self.btn_play_proc)
        play_layout.addWidget(self.btn_stop)
        play_group.setLayout(play_layout)
        right_layout.addWidget(play_group)

        splitter.addWidget(right_panel)
        splitter.setSizes([420, 980])

    def load_vocal_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Vocal Mono", "", "Audio (*.wav *.mp3 *.flac)")
        if path:
            self.vocal_audio_path = path
            self.mixed_label.setText(Path(path).name)
            wf, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            self.original_waveform = wf
            self.waveform.set_waveforms(wf, None, len(wf) / SAMPLE_RATE)
            self.save_btn.setEnabled(False)
            self.latest_mask = None

    def add_reference_segment(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ref Audio", "", "Audio (*.wav *.mp3 *.flac)")
        if path:
            seg, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            self.reference_segments.append(seg)
            self.ref_label.setText(f"Refs: {len(self.reference_segments)}")

    def on_threshold_changed(self, value):
        self.threshold_label.setText(f"Similarity Threshold: {value / 100:.2f}")

    def start_separation(self):
        if not self.vocal_audio_path:
            QMessageBox.warning(self, "Missing Audio", "Please load a mono vocal file first.")
            return

        if not self.reference_segments:
            QMessageBox.warning(self, "Missing Reference", "Please add at least one reference segment.")
            return

        self.process_btn.setEnabled(False)
        self.status_bar.showMessage("Starting target singer separation...")
        self.separator.set_similarity_threshold(self.threshold_slider.value() / 100.0)

        self.worker = SeparationWorker(self.separator, self.vocal_audio_path, self.reference_segments)
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
        self.status_bar.showMessage("Voice separation completed successfully")

        self.processed_waveform = result
        self.latest_mask = mask
        self.waveform.set_waveforms(self.original_waveform, result, len(result) / SAMPLE_RATE)
        self.save_btn.setEnabled(True)

        if self.export_mask_checkbox.isChecked() and self.vocal_audio_path and mask is not None:
            plot_path = self.export_mask_diagnostic_plot(self.original_waveform, result, mask, self.vocal_audio_path)
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
        if self.processed_waveform is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save", "", "WAV (*.wav)")
        if path:
            sf.write(path, self.processed_waveform, SAMPLE_RATE)

    def play_original(self):
        if self.vocal_audio_path:
            self._start_play(self.vocal_audio_path)

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
            elapsed = (np.datetime64("now") - self.playback_start) / np.timedelta64(1, "s")
            self.waveform.playback_position = elapsed
            self.waveform.update()
        else:
            self.stop_playback()

    def closeEvent(self, event):
        self.stop_playback()
        clear_gpu_cache()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ProSingerSeparatorApp()
    window.show()
    sys.exit(app.exec())
