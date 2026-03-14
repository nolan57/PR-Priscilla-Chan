"""
Professional Singer Separator - Enhanced Version (FFmpeg Playback)
Uses ffmpeg for universal cross-platform audio playback

Dependencies:
    pip install -r requirements_ffmpeg.txt
    
System requirement:
    ffmpeg must be installed on your system
    - Linux: sudo apt install ffmpeg
    - macOS: brew install ffmpeg
    - Windows: Download from https://ffmpeg.org/download.html
"""

import os
import sys
import warnings
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

warnings.filterwarnings('ignore')

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import librosa
import soundfile as sf
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add torchaudio compatibility patch
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLineEdit, QLabel, QProgressBar,
    QComboBox, QGroupBox, QTextEdit, QMessageBox, QDialog, QDialogButtonBox,
    QSlider, QSpinBox, QCheckBox, QSplitter, QTabWidget
)
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt, QTimer
from PyQt6.QtGui import QFont


def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class FFmpegPlayer:
    """Simple ffmpeg-based audio player"""
    
    def __init__(self):
        self.process = None
        self.audio_path = None
        self.is_playing = False
        self.is_paused = False
        self.start_time = 0
        self.pause_time = 0
        self.volume = 0.7
        self._position = 0
        self._duration = 0
        
    def load(self, audio_path: str):
        """Load audio file"""
        self.audio_path = audio_path
        self.stop()
        
        # Get duration using ffprobe
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 
                 'format=duration', '-of', 
                 'default=noprint_wrappers=1:nokey=1', audio_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            self._duration = float(result.stdout.strip())
        except:
            self._duration = 0
    
    def play(self):
        """Start or resume playback"""
        if not self.audio_path:
            return
        
        if self.is_paused:
            # Resume from pause (restart from pause position)
            self.is_paused = False
            self.is_playing = True
            self._play_from_position(self.pause_time)
        elif not self.is_playing:
            # Start from beginning
            self.is_playing = True
            self.start_time = time.time()
            self._play_from_position(0)
    
    def _play_from_position(self, position: float):
        """Internal method to play from specific position"""
        if self.process:
            self.process.terminate()
            self.process = None
        
        # Use ffplay (part of ffmpeg) for playback
        # -nodisp: no video display
        # -autoexit: exit when done
        # -ss: start position
        # -volume: volume level (0-100)
        try:
            self.process = subprocess.Popen(
                ['ffplay', '-nodisp', '-autoexit', 
                 '-ss', str(position),
                 '-volume', str(int(self.volume * 100)),
                 self.audio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.pause_time = position
        except FileNotFoundError:
            # ffplay not available, try alternative
            try:
                # Use ffmpeg with audio output
                self.process = subprocess.Popen(
                    ['ffmpeg', '-ss', str(position), 
                     '-i', self.audio_path,
                     '-af', f'volume={self.volume}',
                     '-f', 'wav', '-'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
            except:
                self.is_playing = False
                raise RuntimeError("ffplay/ffmpeg not available for playback")
    
    def pause(self):
        """Pause playback"""
        if self.is_playing and not self.is_paused:
            self.is_paused = True
            self.is_playing = False
            self.pause_time = self.position()
            if self.process:
                self.process.terminate()
                self.process = None
    
    def stop(self):
        """Stop playback"""
        self.is_playing = False
        self.is_paused = False
        self.pause_time = 0
        if self.process:
            self.process.terminate()
            self.process = None
    
    def set_volume(self, volume: float):
        """Set volume (0.0 - 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        # Note: Volume change takes effect on next play
    
    def position(self) -> float:
        """Get current playback position in seconds"""
        if self.is_playing and not self.is_paused:
            return self.pause_time + (time.time() - self.start_time)
        return self.pause_time
    
    def duration(self) -> float:
        """Get total duration in seconds"""
        return self._duration
    
    def is_active(self) -> bool:
        """Check if currently playing"""
        if self.process and self.process.poll() is None:
            return True
        if self.process and self.process.poll() is not None:
            self.is_playing = False
        return self.is_playing


class ModelDownloader(QThread):
    """Thread for downloading models"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, hf_token: Optional[str] = None):
        super().__init__()
        self.hf_token = hf_token
    
    def run(self):
        try:
            self.progress_signal.emit("📥 Starting model download...")
            
            # Try to download pyannote models
            if self.hf_token:
                try:
                    from pyannote.audio import Model
                    from pyannote.audio.pipelines import SpeakerDiarization
                    
                    self.progress_signal.emit("📥 Downloading pyannote/embedding...")
                    Model.from_pretrained(
                        "pyannote/embedding",
                        token=self.hf_token
                    )
                    self.progress_signal.emit("✓ Downloaded pyannote/embedding")
                    
                    self.progress_signal.emit("📥 Downloading pyannote/segmentation-3.0...")
                    Model.from_pretrained(
                        "pyannote/segmentation-3.0",
                        token=self.hf_token
                    )
                    self.progress_signal.emit("✓ Downloaded pyannote/segmentation-3.0")
                    
                    self.progress_signal.emit("📥 Downloading speaker-diarization-3.1...")
                    SpeakerDiarization.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        token=self.hf_token
                    )
                    self.progress_signal.emit("✓ Downloaded speaker-diarization-3.1")
                    
                    self.finished_signal.emit(True, "Pyannote models downloaded successfully!")
                    return
                except Exception as e:
                    self.progress_signal.emit(f"⚠️ Pyannote download failed: {e}")
            
            # Try Resemblyzer as fallback
            try:
                from resemblyzer import VoiceEncoder
                self.progress_signal.emit("📥 Downloading Resemblyzer model...")
                VoiceEncoder()
                self.progress_signal.emit("✓ Downloaded Resemblyzer model")
                self.finished_signal.emit(True, "Resemblyzer model downloaded successfully!")
            except Exception as e:
                self.finished_signal.emit(False, f"Failed to download models: {e}")
                
        except Exception as e:
            self.finished_signal.emit(False, f"Download error: {e}")


class SpeakerEmbedder:
    """Unified interface for speaker embedding extraction"""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and torch.backends.mps.is_available():
            self.device = "mps"
        
        self.model = None
        self.model_type = None
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        self._load_model()
    
    def _load_model(self):
        """Try loading models in order: pyannote > resemblyzer"""
            
        # Try pyannote-audio first
        try:
            from pyannote.audio import Model
            from pyannote.audio.pipelines import SpeakerDiarization
                
            if not self.hf_token:
                print("⚠️  No HuggingFace token found. Pyannote requires authentication.")
                raise ImportError("HuggingFace token required")
                
            print("Loading pyannote-audio (state-of-the-art)..." )
                
            self.embedding_model = Model.from_pretrained(
                "pyannote/embedding",
                token=self.hf_token
            ).to(self.device)
                
            self.diarization = SpeakerDiarization.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token
            ).to(torch.device(self.device))
                
            self.model_type = "pyannote"
            print(f"✓ Loaded pyannote-audio on {self.device}")
            return
                
        except Exception as e:
            print(f"Pyannote not available: {e}")
            # Try alternative approach without speechbrain dependency
            try:
                from pyannote.audio import Pipeline
                print("Trying alternative pyannote pipeline loading...")
                    
                # Use the simpler Pipeline approach
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self.hf_token
                ).to(torch.device(self.device))
                    
                # Load embedding model separately
                self.embedding_model = Model.from_pretrained(
                    "pyannote/embedding",
                    token=self.hf_token
                ).to(self.device)
                    
                self.model_type = "pyannote_alt"
                print(f"✓ Loaded pyannote-audio (alternative) on {self.device}")
                return
                    
            except Exception as alt_e:
                print(f"Alternative pyannote loading failed: {alt_e}")
            
        # Fallback to Resemblyzer
        try:
            from resemblyzer import VoiceEncoder
            print("Loading Resemblyzer (fallback)...")
            self.model = VoiceEncoder(device=self.device)
            self.model_type = "resemblyzer"
            print(f"✓ Loaded Resemblyzer on {self.device}")
            return
        except Exception as e:
            print(f"Resemblyzer not available: {e}")
            
        raise RuntimeError(
            "No speaker recognition model available!\n"
            "Install with: pip install pyannote.audio resemblyzer"
        )
    
    def extract_embedding(self, audio_path: str, start_time: float = 0, end_time: float = None) -> np.ndarray:
        """Extract speaker embedding from audio file or segment"""
        
        if self.model_type == "pyannote":
            return self._extract_pyannote(audio_path, start_time, end_time)
        elif self.model_type == "pyannote_alt":
            return self._extract_pyannote_alt(audio_path, start_time, end_time)
        elif self.model_type == "resemblyzer":
            return self._extract_resemblyzer(audio_path, start_time, end_time)
    
    def _extract_pyannote(self, audio_path: str, start_time: float, end_time: float) -> np.ndarray:
        """Extract embedding using pyannote"""
        from pyannote.audio import Inference
        
        inference = Inference(self.embedding_model, window="whole")
        
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        if end_time is not None:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            y = y[start_sample:end_sample]
        elif start_time > 0:
            start_sample = int(start_time * sr)
            y = y[start_sample:]
        
        if len(y) < 16000:
            y = np.pad(y, (0, 16000 - len(y)), mode='constant')
        
        temp_path = "/tmp/temp_segment.wav"
        sf.write(temp_path, y, sr)
        
        embedding = inference(temp_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return np.array(embedding)
    
    def _extract_pyannote_alt(self, audio_path: str, start_time: float, end_time: float) -> np.ndarray:
        """Extract embedding using alternative pyannote approach"""
        from pyannote.audio import Inference
        
        try:
            inference = Inference(self.embedding_model, window="whole")
            
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            if end_time is not None:
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                y = y[start_sample:end_sample]
            elif start_time > 0:
                start_sample = int(start_time * sr)
                y = y[start_sample:]
            
            if len(y) < 16000:
                y = np.pad(y, (0, 16000 - len(y)), mode='constant')
            
            temp_path = "/tmp/temp_segment.wav"
            sf.write(temp_path, y, sr)
            
            embedding = inference(temp_path)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return np.array(embedding)
        except Exception as e:
            print(f"Alternative embedding extraction failed: {e}")
            # Return zero vector as fallback
            return np.zeros(512)
    
    def _extract_resemblyzer(self, audio_path: str, start_time: float, end_time: float) -> np.ndarray:
        """Extract embedding using Resemblyzer"""
        from resemblyzer import preprocess_wav
        
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        if end_time is not None:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            y = y[start_sample:end_sample]
        elif start_time > 0:
            start_sample = int(start_time * sr)
            y = y[start_sample:]
        
        if len(y) < 16000:
            y = np.pad(y, (0, 16000 - len(y)), mode='constant')
        
        wav = preprocess_wav(y, source_sr=sr)
        embedding = self.model.embed_utterance(wav)
        return embedding
    
    def diarize_speakers(self, audio_path: str, num_speakers: int = None) -> List[Tuple[float, float, int]]:
        """Perform speaker diarization"""
        
        if self.model_type == "pyannote":
            return self._diarize_pyannote(audio_path, num_speakers)
        elif self.model_type == "pyannote_alt":
            return self._diarize_pyannote_alt(audio_path, num_speakers)
        else:
            return self._diarize_resemblyzer(audio_path, num_speakers)
    
    def _diarize_pyannote(self, audio_path: str, num_speakers: int) -> List[Tuple[float, float, int]]:
        """Diarize using pyannote"""
        
        if num_speakers:
            diarization = self.diarization(audio_path, num_speakers=num_speakers)
        else:
            diarization = self.diarization(audio_path)
        
        segments = []
        
        # Handle different pyannote.audio API versions
        if hasattr(diarization, 'itertracks'):
            # Old API
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_id = int(speaker.split("_")[-1]) if "_" in speaker else hash(speaker) % 100
                segments.append((turn.start, turn.end, speaker_id))
        else:
            # New API - pyannote.audio 4.x
            # The diarization result is now a different structure
            try:
                # Try to access segments directly
                for segment in diarization.get_timeline():
                    # Get speaker label for this segment
                    speakers = diarization.get_labels(segment)
                    if speakers:
                        speaker = list(speakers)[0]
                        speaker_id = int(speaker.split("_")[-1]) if "_" in speaker else hash(speaker) % 100
                        segments.append((segment.start, segment.end, speaker_id))
            except Exception as e:
                # Fallback: try to convert to annotation format
                try:
                    from pyannote.core import Annotation
                    if isinstance(diarization, Annotation):
                        for segment, track, speaker in diarization.itertracks(yield_label=True):
                            speaker_id = int(speaker.split("_")[-1]) if "_" in speaker else hash(speaker) % 100
                            segments.append((segment.start, segment.end, speaker_id))
                    else:
                        # Last resort: treat as timeline
                        for segment in diarization:
                            speaker_id = hash(str(segment)) % 100
                            segments.append((segment.start, segment.end, speaker_id))
                except:
                    # If all else fails, return single speaker segment
                    import soundfile as sf
                    info = sf.info(audio_path)
                    segments.append((0, info.duration, 0))
        
        return segments
    
    def _diarize_pyannote_alt(self, audio_path: str, num_speakers: int) -> List[Tuple[float, float, int]]:
        """Diarize using alternative pyannote pipeline"""
        
        try:
            if num_speakers:
                diarization = self.diarization_pipeline(audio_path, num_speakers=num_speakers)
            else:
                diarization = self.diarization_pipeline(audio_path)
            
            segments = []
            
            # Handle the new API structure
            if hasattr(diarization, 'get_timeline'):
                # New API approach
                for segment in diarization.get_timeline():
                    speakers = diarization.get_labels(segment)
                    if speakers:
                        speaker = list(speakers)[0]
                        speaker_id = int(speaker.split("_")[-1]) if "_" in speaker else hash(speaker) % 100
                        segments.append((segment.start, segment.end, speaker_id))
            else:
                # Fallback for unknown structure
                import soundfile as sf
                info = sf.info(audio_path)
                segments.append((0, info.duration, 0))
            
            return segments
            
        except Exception as e:
            print(f"Alternative diarization failed: {e}")
            # Fallback to single speaker
            import soundfile as sf
            info = sf.info(audio_path)
            return [(0, info.duration, 0)]
    
    def _diarize_resemblyzer(self, audio_path: str, num_speakers: int) -> List[Tuple[float, float, int]]:
        """Diarize using Resemblyzer"""
        from sklearn.cluster import AgglomerativeClustering
        
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        window_duration = 1.5
        hop_duration = 0.75
        window_samples = int(window_duration * sr)
        hop_samples = int(hop_duration * sr)
        
        embeddings = []
        timestamps = []
        
        for i in range(0, len(y) - window_samples + 1, hop_samples):
            segment = y[i:i + window_samples]
            
            from resemblyzer import preprocess_wav
            wav = preprocess_wav(segment, source_sr=sr)
            emb = self.model.embed_utterance(wav)
            
            embeddings.append(emb)
            timestamps.append(i / sr)
        
        if not embeddings:
            return [(0, len(y) / sr, 0)]
        
        embeddings = np.array(embeddings)
        
        n_speakers = num_speakers if num_speakers else min(3, len(embeddings))
        clustering = AgglomerativeClustering(n_clusters=n_speakers, metric='cosine', linkage='average')
        labels = clustering.fit_predict(embeddings)
        
        segments = []
        current_speaker = labels[0]
        seg_start = timestamps[0]
        
        for i in range(1, len(labels)):
            if labels[i] != current_speaker:
                segments.append((seg_start, timestamps[i], int(current_speaker)))
                current_speaker = labels[i]
                seg_start = timestamps[i]
        
        segments.append((seg_start, len(y) / sr, int(current_speaker)))
        
        return segments


class AudioVisualizer(QWidget):
    """Waveform and spectrogram visualization with ffmpeg playback"""
    
    def __init__(self):
        super().__init__()
        self.audio_path = None
        self.y = None
        self.sr = None
        self.segments = []
        self.speaker_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        # FFmpeg player
        self.player = FFmpegPlayer()
        self.has_ffmpeg = check_ffmpeg()
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Playback controls
        if self.has_ffmpeg:
            controls_layout = QHBoxLayout()
            
            self.play_btn = QPushButton("▶ Play")
            self.play_btn.clicked.connect(self.toggle_playback)
            self.play_btn.setEnabled(False)
            
            self.stop_btn = QPushButton("⏹ Stop")
            self.stop_btn.clicked.connect(self.stop_playback)
            self.stop_btn.setEnabled(False)
            
            self.position_label = QLabel("00:00 / 00:00")
            
            self.volume_slider = QSlider(Qt.Orientation.Horizontal)
            self.volume_slider.setRange(0, 100)
            self.volume_slider.setValue(70)
            self.volume_slider.valueChanged.connect(self.set_volume)
            self.volume_slider.setMaximumWidth(150)
            
            controls_layout.addWidget(self.play_btn)
            controls_layout.addWidget(self.stop_btn)
            controls_layout.addWidget(self.position_label)
            controls_layout.addStretch()
            controls_layout.addWidget(QLabel("Volume:"))
            controls_layout.addWidget(self.volume_slider)
            
            layout.addLayout(controls_layout)
            
            # Position update timer
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_position)
        else:
            info_label = QLabel("ℹ️  FFmpeg not found - audio playback disabled. Install ffmpeg for playback.")
            info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
            layout.addWidget(info_label)
        
        self.setLayout(layout)
    
    def load_audio(self, audio_path: str):
        """Load audio file for visualization"""
        self.audio_path = audio_path
        
        try:
            # Load audio
            self.y, self.sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Enable playback if ffmpeg available
            if self.has_ffmpeg:
                self.player.load(audio_path)
                self.play_btn.setEnabled(True)
                self.stop_btn.setEnabled(True)
            
            # Initial visualization
            self.update_visualization()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio: {e}")
    
    def set_segments(self, segments: List[Tuple[float, float, int]]):
        """Set speaker segments for visualization"""
        self.segments = segments
        self.update_visualization()
    
    def update_visualization(self):
        """Update waveform and spectrogram display"""
        if self.y is None:
            return
        
        self.figure.clear()
        
        # Create subplots
        ax_wave = self.figure.add_subplot(2, 1, 1)
        ax_spec = self.figure.add_subplot(2, 1, 2)
        
        # Time axis
        duration = len(self.y) / self.sr
        time = np.linspace(0, duration, len(self.y))
        
        # Waveform
        ax_wave.plot(time, self.y, linewidth=0.5, color='steelblue', alpha=0.7)
        ax_wave.set_ylabel('Amplitude')
        ax_wave.set_title('Waveform with Speaker Segments')
        ax_wave.set_xlim(0, duration)
        ax_wave.grid(True, alpha=0.3)
        
        # Add segment annotations to waveform
        y_min, y_max = ax_wave.get_ylim()
        for start, end, speaker_id in self.segments:
            color = self.speaker_colors[speaker_id % len(self.speaker_colors)]
            rect = Rectangle((start, y_min), end - start, y_max - y_min,
                           alpha=0.3, facecolor=color, edgecolor='none')
            ax_wave.add_patch(rect)
        
        # Create legend
        if self.segments:
            unique_speakers = sorted(set(s[2] for s in self.segments))
            legend_elements = [
                Rectangle((0, 0), 1, 1, fc=self.speaker_colors[spk % len(self.speaker_colors)], alpha=0.3)
                for spk in unique_speakers
            ]
            legend_labels = [f'Speaker {spk + 1}' for spk in unique_speakers]
            ax_wave.legend(legend_elements, legend_labels, loc='upper right')
        
        # Spectrogram
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(self.y, n_fft=2048, hop_length=512)),
            ref=np.max
        )
        
        img = ax_spec.imshow(
            D,
            aspect='auto',
            origin='lower',
            extent=[0, duration, 0, self.sr / 2000],
            cmap='viridis',
            interpolation='bilinear'
        )
        
        ax_spec.set_ylabel('Frequency (kHz)')
        ax_spec.set_xlabel('Time (seconds)')
        ax_spec.set_title('Spectrogram with Speaker Segments')
        
        # Add segment annotations
        freq_max = self.sr / 2000
        for start, end, speaker_id in self.segments:
            color = self.speaker_colors[speaker_id % len(self.speaker_colors)]
            rect = Rectangle((start, 0), end - start, freq_max,
                           alpha=0.2, facecolor=color, edgecolor=color, linewidth=2)
            ax_spec.add_patch(rect)
        
        # Colorbar
        self.figure.colorbar(img, ax=ax_spec, format='%+2.0f dB')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def toggle_playback(self):
        """Toggle play/pause"""
        if not self.has_ffmpeg:
            return
        
        if self.player.is_active():
            self.player.pause()
            self.play_btn.setText("▶ Play")
            self.timer.stop()
        else:
            self.player.play()
            self.play_btn.setText("⏸ Pause")
            self.timer.start(100)  # Update every 100ms
    
    def stop_playback(self):
        """Stop playback"""
        if not self.has_ffmpeg:
            return
        
        self.player.stop()
        self.play_btn.setText("▶ Play")
        self.timer.stop()
        self.position_label.setText("00:00 / 00:00")
    
    def set_volume(self, value):
        """Set playback volume"""
        if not self.has_ffmpeg:
            return
        
        self.player.set_volume(value / 100.0)
    
    def update_position(self):
        """Update position label"""
        if not self.has_ffmpeg:
            return
        
        duration = self.player.duration()
        if duration > 0:
            position = self.player.position()
            self.position_label.setText(
                f"{self.format_time(position)} / {self.format_time(duration)}"
            )
            
            # Stop timer if playback finished
            if not self.player.is_active() and position >= duration:
                self.timer.stop()
                self.play_btn.setText("▶ Play")
    
    def format_time(self, seconds):
        """Format seconds as MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"


class AudioProcessor(QObject):
    """Main audio processing engine"""
    
    def __init__(self, hf_token: Optional[str] = None):
        super().__init__()
        self.embedder = SpeakerEmbedder(hf_token)
    
    def match_with_reference(self, 
                            vocal_path: str, 
                            reference_path: str,
                            similarity_threshold: float = 0.7) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, int]]]:
        """Match target singer segments using reference audio"""
        
        print(f"Extracting reference embedding from: {reference_path}")
        reference_embedding = self.embedder.extract_embedding(reference_path)
        
        print(f"Analyzing vocal track: {vocal_path}")
        
        y, sr = librosa.load(vocal_path, sr=16000, mono=True)
        total_duration = len(y) / sr
        
        window_duration = 1.5
        hop_duration = 0.5
        window_samples = int(window_duration * sr)
        hop_samples = int(hop_duration * sr)
        
        similarities = []
        timestamps = []
        
        print("Computing frame-by-frame similarity...")
        for i in range(0, len(y) - window_samples + 1, hop_samples):
            try:
                embedding = self.embedder.extract_embedding(
                    vocal_path,
                    start_time=i / sr,
                    end_time=(i + window_samples) / sr
                )
                
                sim = cosine_similarity([reference_embedding], [embedding])[0][0]
                similarities.append(sim)
                timestamps.append(i / sr)
                
            except Exception as e:
                similarities.append(0.0)
                timestamps.append(i / sr)
        
        if not similarities:
            raise RuntimeError("No valid frames processed")
        
        similarities = np.array(similarities)
        decisions = similarities > similarity_threshold
        
        if len(decisions) >= 7:
            decisions = median_filter(decisions.astype(float), size=7) > 0.5
        
        print(f"Similarity range: {similarities.min():.3f} - {similarities.max():.3f}")
        print(f"Matched {decisions.sum()}/{len(decisions)} frames")
        
        # Convert to segments
        segments = []
        in_segment = False
        seg_start = 0
        
        for i, match in enumerate(decisions):
            if match and not in_segment:
                seg_start = timestamps[i]
                in_segment = True
            elif not match and in_segment:
                segments.append((seg_start, timestamps[i]))
                in_segment = False
        
        if in_segment:
            segments.append((seg_start, total_duration))
        
        # Merge nearby segments
        merged_segments = []
        if segments:
            current = segments[0]
            for next_seg in segments[1:]:
                if next_seg[0] - current[1] < 0.5:
                    current = (current[0], next_seg[1])
                else:
                    merged_segments.append(current)
                    current = next_seg
            merged_segments.append(current)
        
        all_segments = [(start, end, 0) for start, end in merged_segments]
        
        print(f"Found {len(merged_segments)} matching segments")
        return merged_segments, all_segments
    
    def extract_speaker_by_clustering(self,
                                     vocal_path: str,
                                     speaker_index: int = 0,
                                     num_speakers: int = None) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, int]]]:
        """Extract speaker using clustering"""
        
        print(f"Performing speaker diarization on: {vocal_path}")
        
        all_segments = self.embedder.diarize_speakers(vocal_path, num_speakers)
        
        if not all_segments:
            y, sr = librosa.load(vocal_path, sr=16000, mono=True)
            return [(0, len(y) / sr)], [(0, len(y) / sr, 0)]
        
        # Group by speaker
        speaker_groups = {}
        for start, end, speaker_id in all_segments:
            if speaker_id not in speaker_groups:
                speaker_groups[speaker_id] = []
            speaker_groups[speaker_id].append((start, end))
        
        # Sort by duration
        sorted_speakers = sorted(
            speaker_groups.items(),
            key=lambda x: sum(e - s for s, e in x[1]),
            reverse=True
        )
        
        print(f"Detected {len(sorted_speakers)} speakers")
        for i, (spk_id, segments) in enumerate(sorted_speakers):
            duration = sum(e - s for s, e in segments)
            print(f"  Speaker {i+1}: {duration:.1f}s")
        
        target_index = min(speaker_index, len(sorted_speakers) - 1)
        target_segments = sorted_speakers[target_index][1]
        
        print(f"Extracting speaker {target_index + 1}")
        return target_segments, all_segments
    
    def process_file(self,
                    input_path: str,
                    output_path: str,
                    reference_path: Optional[str] = None,
                    speaker_index: int = 0,
                    num_speakers: int = None,
                    similarity_threshold: float = 0.7) -> Tuple[List[Tuple[float, float, int]], float]:
        """Process a single vocal track"""
        
        y_full, sr_full = librosa.load(input_path, sr=None, mono=True)
        result = np.zeros_like(y_full)
        
        try:
            if reference_path and Path(reference_path).exists():
                print("\n=== Reference-based matching ===")
                segments, all_segments = self.match_with_reference(input_path, reference_path, similarity_threshold)
            else:
                print("\n=== Clustering-based extraction ===")
                segments, all_segments = self.extract_speaker_by_clustering(input_path, speaker_index, num_speakers)
            
            print(f"\nExtracting {len(segments)} segments...")
            for start, end in segments:
                start_sample = int(start * sr_full)
                end_sample = int(end * sr_full)
                start_sample = max(0, min(start_sample, len(result)))
                end_sample = max(0, min(end_sample, len(result)))
                
                if end_sample > start_sample:
                    result[start_sample:end_sample] = y_full[start_sample:end_sample]
            
            output_energy = np.sum(np.abs(result))
            input_energy = np.sum(np.abs(y_full))
            retention_ratio = output_energy / input_energy if input_energy > 0 else 0
            
            print(f"\nOutput: {retention_ratio*100:.1f}% retained")
            
            sf.write(output_path, result, sr_full)
            print(f"✓ Saved: {output_path}\n")
            
            return all_segments, retention_ratio
            
        except Exception as e:
            print(f"❌ Error: {e}")
            raise


class VisualizerDialog(QDialog):
    """Dialog for audio visualization"""
    
    def __init__(self, audio_path: str, segments: List[Tuple[float, float, int]], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Audio Visualization - {Path(audio_path).name}")
        self.setGeometry(100, 100, 1200, 800)
        
        layout = QVBoxLayout()
        
        self.visualizer = AudioVisualizer()
        self.visualizer.load_audio(audio_path)
        self.visualizer.set_segments(segments)
        
        layout.addWidget(self.visualizer)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)


class TokenDialog(QDialog):
    """Dialog for entering HuggingFace token"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HuggingFace Token Required")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        info = QLabel(
            "Pyannote-audio requires a HuggingFace token.\n\n"
            "1. Get token: https://huggingface.co/settings/tokens\n"
            "2. Accept agreements:\n"
            "   - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "   - https://huggingface.co/pyannote/segmentation-3.0\n\n"
            "Enter token (or Cancel for Resemblyzer):"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("hf_...")
        layout.addWidget(self.token_input)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def get_token(self):
        return self.token_input.text().strip()


class WorkerThread(QThread):
    """Background processing thread"""
    
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str, list, float)
    
    def __init__(self, processor, input_files, output_dir, reference_path, 
                 speaker_index, num_speakers, similarity_threshold):
        super().__init__()
        self.processor = processor
        self.input_files = input_files
        self.output_dir = output_dir
        self.reference_path = reference_path
        self.speaker_index = speaker_index
        self.num_speakers = num_speakers
        self.similarity_threshold = similarity_threshold
        self.all_segments = []
        self.retention_ratio = 0.0
    
    def run(self):
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total = len(self.input_files)
        for i, input_file in enumerate(self.input_files):
            self.log_signal.emit(f"[{i+1}/{total}] Processing: {Path(input_file).name}")
            
            try:
                output_path = output_dir / f"{Path(input_file).stem}_target_singer.wav"
                segments, retention = self.processor.process_file(
                    input_file,
                    str(output_path),
                    self.reference_path,
                    self.speaker_index,
                    self.num_speakers,
                    self.similarity_threshold
                )
                self.all_segments = segments
                self.retention_ratio = retention
                self.log_signal.emit(f"✓ Completed: {output_path.name} ({retention*100:.1f}%)")
            except Exception as e:
                self.log_signal.emit(f"❌ Error: {str(e)}")
            
            self.progress_signal.emit(int(100 * (i + 1) / total))
        
        self.finished_signal.emit(str(output_dir), self.all_segments, self.retention_ratio)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Professional Singer Separator - Enhanced (FFmpeg)")
        self.setGeometry(100, 100, 1000, 800)
        
        self.input_files = []
        self.output_dir = ""
        self.processor = None
        self.current_segments = []
        
        self.init_ui()
        self.init_processor()
    
    def init_processor(self):
        """Initialize audio processor"""
        try:
            hf_token = os.getenv("HF_TOKEN")
            
            if not hf_token:
                dialog = TokenDialog(self)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    hf_token = dialog.get_token()
                    if hf_token:
                        os.environ["HF_TOKEN"] = hf_token
            
            self.processor = AudioProcessor(hf_token)
            self.log_text.append("✓ Models loaded successfully")
            self.log_text.append(f"  Using: {self.processor.embedder.model_type}")
            self.log_text.append(f"  Device: {self.processor.embedder.device}")
            
            if check_ffmpeg():
                self.log_text.append("✓ FFmpeg detected - audio playback enabled")
            else:
                self.log_text.append("ℹ️  FFmpeg not found - playback disabled")
                self.log_text.append("   Install ffmpeg for audio playback")
            
            self.log_text.append("")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to initialize:\n{str(e)}\n\n"
                "Install: pip install -r requirements_ffmpeg.txt"
            )
            sys.exit(1)
    
    def init_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout()
        
        # Title
        title = QLabel("🎤 Professional Singer Separator - Enhanced")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        subtitle = QLabel("Deep learning • Visualization • FFmpeg playback")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(subtitle)
        
        # Model management
        model_group = QGroupBox("Model Management")
        model_layout = QHBoxLayout()
        
        self.download_btn = QPushButton("↓Pre-download Models")
        self.download_btn.clicked.connect(self.download_models)
        model_layout.addWidget(self.download_btn)
        
        model_layout.addStretch()
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # Input files
        file_group = QGroupBox("Input: Pre-Separated Vocal Tracks")
        file_layout = QHBoxLayout()
        self.file_line = QLineEdit()
        self.file_line.setReadOnly(True)
        add_btn = QPushButton("Add Files...")
        add_btn.clicked.connect(self.add_files)
        file_layout.addWidget(self.file_line)
        file_layout.addWidget(add_btn)
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)
        
        # Output directory
        out_group = QGroupBox("Output Directory")
        out_layout = QHBoxLayout()
        self.out_line = QLineEdit()
        self.out_line.setReadOnly(True)
        out_btn = QPushButton("Select...")
        out_btn.clicked.connect(self.select_output_dir)
        out_layout.addWidget(self.out_line)
        out_layout.addWidget(out_btn)
        out_group.setLayout(out_layout)
        main_layout.addWidget(out_group)
        
        # Reference audio
        ref_group = QGroupBox("Target Singer Reference (Optional)")
        ref_layout = QVBoxLayout()
        ref_info = QLabel("3-10 second clean sample of target singer")
        ref_info.setStyleSheet("color: gray; font-style: italic;")
        ref_layout.addWidget(ref_info)
        ref_file_layout = QHBoxLayout()
        self.ref_line = QLineEdit()
        ref_btn = QPushButton("Select...")
        ref_btn.clicked.connect(self.select_reference)
        ref_clear = QPushButton("Clear")
        ref_clear.clicked.connect(lambda: self.ref_line.clear())
        ref_file_layout.addWidget(self.ref_line)
        ref_file_layout.addWidget(ref_btn)
        ref_file_layout.addWidget(ref_clear)
        ref_layout.addLayout(ref_file_layout)
        ref_group.setLayout(ref_layout)
        main_layout.addWidget(ref_group)
        
        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        
        # Threshold slider
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Similarity Threshold:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 90)
        self.threshold_slider.setValue(70)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_value_label = QLabel("0.70")
        self.threshold_value_label.setMinimumWidth(40)
        threshold_layout.addWidget(self.threshold_value_label)
        settings_layout.addLayout(threshold_layout)
        
        threshold_info = QLabel("Lower: more inclusive • Higher: more precise")
        threshold_info.setStyleSheet("color: gray; font-size: 10px;")
        settings_layout.addWidget(threshold_info)
        
        # Speaker selection
        speaker_layout = QHBoxLayout()
        speaker_layout.addWidget(QLabel("If no reference:"))
        self.speaker_combo = QComboBox()
        self.speaker_combo.addItems([
            "Speaker 1 (Most speech)",
            "Speaker 2",
            "Speaker 3"
        ])
        speaker_layout.addWidget(self.speaker_combo)
        speaker_layout.addStretch()
        settings_layout.addLayout(speaker_layout)
        
        # Number of speakers
        num_speakers_layout = QHBoxLayout()
        num_speakers_layout.addWidget(QLabel("Number of speakers:"))
        self.num_speakers_combo = QComboBox()
        self.num_speakers_combo.addItems(["Auto-detect", "2", "3", "4", "5"])
        num_speakers_layout.addWidget(self.num_speakers_combo)
        num_speakers_layout.addStretch()
        settings_layout.addLayout(num_speakers_layout)
        
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # Log
        log_label = QLabel("Processing Log:")
        main_layout.addWidget(log_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        main_layout.addWidget(self.log_text)
        
        # Progress
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)
        
        # Buttons row - Start and Visualize buttons in same row
        buttons_layout = QHBoxLayout()
        
        # Start button
        self.start_btn = QPushButton("▶ Start Processing")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.start_btn.setMinimumHeight(50)
        buttons_layout.addWidget(self.start_btn)
        
        # Visualize button
        self.visualize_btn = QPushButton("Visualize Last Processed File")
        self.visualize_btn.clicked.connect(self.show_visualization)
        self.visualize_btn.setEnabled(False)
        self.visualize_btn.setMinimumHeight(50)
        buttons_layout.addWidget(self.visualize_btn)
        
        buttons_layout.addStretch()  # Push buttons to the left
        main_layout.addLayout(buttons_layout)
        
        central.setLayout(main_layout)
        self.setCentralWidget(central)
    
    def update_threshold_label(self, value):
        threshold = value / 100.0
        self.threshold_value_label.setText(f"{threshold:.2f}")
    
    def download_models(self):
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            dialog = TokenDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                hf_token = dialog.get_token()
                if hf_token:
                    os.environ["HF_TOKEN"] = hf_token
        
        self.download_btn.setEnabled(False)
        self.log_text.clear()
        
        self.downloader = ModelDownloader(hf_token)
        self.downloader.progress_signal.connect(self.log_text.append)
        self.downloader.finished_signal.connect(self.download_finished)
        self.downloader.start()
    
    def download_finished(self, success, message):
        self.download_btn.setEnabled(True)
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Failed", message)
    
    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Vocal Tracks", "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a)"
        )
        if files:
            self.input_files = files
            self.file_line.setText(f"{len(files)} file(s)")
    
    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output")
        if dir_path:
            self.output_dir = dir_path
            self.out_line.setText(dir_path)
    
    def select_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference", "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a)"
        )
        if path:
            self.ref_line.setText(path)
    
    def start_processing(self):
        if not self.input_files:
            QMessageBox.warning(self, "Warning", "Add input files.")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Select output directory.")
            return
        
        reference_path = self.ref_line.text().strip() or None
        speaker_index = self.speaker_combo.currentIndex()
        
        num_speakers_text = self.num_speakers_combo.currentText()
        num_speakers = None if num_speakers_text == "Auto-detect" else int(num_speakers_text)
        
        similarity_threshold = self.threshold_slider.value() / 100.0
        
        self.start_btn.setEnabled(False)
        self.log_text.clear()
        self.progress_bar.setValue(0)
        
        self.last_input_file = self.input_files[-1]
        
        self.worker = WorkerThread(
            self.processor,
            self.input_files.copy(),
            self.output_dir,
            reference_path,
            speaker_index,
            num_speakers,
            similarity_threshold
        )
        self.worker.log_signal.connect(self.log_message)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self.processing_finished)
        self.worker.start()
    
    def log_message(self, msg):
        self.log_text.append(msg)
    
    def processing_finished(self, output_dir, segments, retention_ratio):
        self.start_btn.setEnabled(True)
        self.current_segments = segments
        self.visualize_btn.setEnabled(True)
        
        QMessageBox.information(
            self, "Complete",
            f"Processing finished!\n\n"
            f"Output: {output_dir}\n"
            f"Retained: {retention_ratio*100:.1f}%\n\n"
            f"Click 'Visualize' to see results."
        )
    
    def show_visualization(self):
        if not self.current_segments or not hasattr(self, 'last_input_file'):
            QMessageBox.warning(self, "Warning", "No file to visualize.")
            return
        
        dialog = VisualizerDialog(self.last_input_file, self.current_segments, self)
        dialog.exec()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
