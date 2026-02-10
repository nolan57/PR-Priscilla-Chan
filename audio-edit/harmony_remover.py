import sys
import os
import tempfile
import subprocess
import numpy as np
import soundfile as sf
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QGroupBox, QSlider, QScrollArea,
    QStatusBar, QLineEdit
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect
from PyQt6.QtGui import QPainter, QColor, QImage

from scipy import signal

SAMPLE_RATE = 44100
DEFAULT_THRESHOLD = 0.5
DEFAULT_SENSITIVITY = 10.0

# Standardized Professional Styling
BUTTON_STYLE = """
    QPushButton {
        background-color: #34495e; color: white; border-radius: 8px;
        padding: 5px 15px; font-weight: bold; min-height: 35px; border: 1px solid #2c3e50;
    }
    QPushButton:hover { background-color: #2c3e50; }
    QPushButton:disabled { background-color: #95a5a6; color: #bdc3c7; }
"""


class WaveformWidget(QWidget):
    """Waveform visualization with region selection for reference samples."""
    
    def __init__(self, main_app, parent=None):
        super().__init__(parent)
        self.main_app = main_app
        self.samples = None
        self.duration = 0.0
        self.regions = []  # List of (start_time, end_time) tuples
        self.playback_position = -1.0
        self.zoom_factor = 1.0
        self.contrast = 1.0
        self.spectrogram_img = None
        
        # Interaction state
        self.drag_start_pos = None
        self.drag_current_x = None
        self.interaction_mode = None  # 'draw', 'move', 'resize'
        self.active_idx = -1
        self.resize_edge = None  # 'left', 'right', or None
        self.hovered_edge = None
        self.edge_threshold = 10  # Pixel threshold for edge detection
        self.drag_start_region = None
        self.drag_start_time = None
        
        self.setMouseTracking(True)
        self.setMinimumHeight(400)

    def set_audio(self, samples, duration):
        """Set audio data and generate spectrogram."""
        self.samples = samples
        self.duration = duration
        self.regions.clear()
        self.generate_spectrogram()
        self.update()

    def generate_spectrogram(self):
        """Generate spectrogram image for visualization."""
        if self.samples is None:
            return
        
        n_fft, hop_length = 2048, 512
        # Limit data size for performance
        data = self.samples[:2000000] if len(self.samples) > 2000000 else self.samples
        
        # Compute STFT
        f, t, Zxx = signal.stft(data, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=n_fft - hop_length)
        spec = np.abs(Zxx)
        spec = np.log1p(spec * self.contrast)
        
        if spec.size == 0:
            return
        
        # Normalize and create color image
        norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        img_data = np.zeros((norm.shape[0], norm.shape[1], 4), dtype=np.uint8)
        img_data[..., 0] = (norm * 255).astype(np.uint8)  # R
        img_data[..., 1] = (np.sin(norm * np.pi) * 255).astype(np.uint8)  # G
        img_data[..., 2] = ((1 - norm) * 150).astype(np.uint8)  # B
        img_data[..., 3] = 255  # Alpha
        
        self.spectrogram_img = QImage(
            np.ascontiguousarray(img_data).tobytes(),
            norm.shape[1], norm.shape[0],
            QImage.Format.Format_RGBA8888
        )

    def get_view_width(self):
        """Get the width of the view area accounting for zoom."""
        return int(self.width() * self.zoom_factor)

    def t_to_x(self, t):
        """Convert time (seconds) to x coordinate."""
        if self.duration <= 0:
            return 0
        return int((t / self.duration) * self.get_view_width())

    def x_to_t(self, x):
        """Convert x coordinate to time (seconds)."""
        if self.get_view_width() <= 0:
            return 0
        return (x / self.get_view_width()) * self.duration

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
        """Handle mouse press for region creation/editing."""
        if self.samples is None:
            return
        
        x = int(event.position().x())
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
        """Handle mouse move for region manipulation."""
        if self.samples is None:
            return
        
        x = int(event.position().x())

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
        """Handle mouse release to finalize region operations."""
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
        """Paint waveform, spectrogram, regions, and playback cursor."""
        if self.samples is None:
            return
        
        painter = QPainter(self)
        vw = self.get_view_width()
        h = self.height()
        mid_h = h // 2
        
        # Black background
        painter.fillRect(event.rect(), Qt.GlobalColor.black)
        
        # Draw spectrogram in top half
        if self.spectrogram_img:
            painter.drawImage(QRect(0, 0, vw, mid_h), self.spectrogram_img)
        
        # Draw waveform in bottom half
        painter.setPen(QColor(0, 255, 255))
        step = max(1, len(self.samples) // (vw * 2))
        vals = self.samples[np.arange(0, len(self.samples), step)]
        y_scale = (mid_h // 1.5) / (np.max(np.abs(vals)) + 1e-6)
        
        for i in range(len(vals) - 1):
            x1 = int((i * step / len(self.samples)) * vw)
            x2 = int(((i + 1) * step / len(self.samples)) * vw)
            y1 = int(h * 0.75 - vals[i] * y_scale)
            y2 = int(h * 0.75 - vals[i + 1] * y_scale)
            painter.drawLine(x1, y1, x2, y2)
        
        # Draw regions with resize handles
        for i, (s, e) in enumerate(self.regions):
            xs, xe = self.t_to_x(s), self.t_to_x(e)
            painter.setPen(QColor(30, 144, 255, 220))
            painter.setBrush(QColor(30, 144, 255, 60))
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
            painter.setPen(QColor(46, 204, 113, 220))
            painter.setBrush(QColor(46, 204, 113, 80))
            painter.drawRect(xs, 0, xe - xs, h)
        
        # Draw playback cursor
        if self.playback_position >= 0:
            px = self.t_to_x(self.playback_position)
            painter.setPen(QColor(255, 0, 0))
            painter.drawLine(px, 0, px, h)


class HarmonyRemovalWorker(QThread):
    """Worker thread for harmony removal processing using spectral masking."""
    
    finished = pyqtSignal(np.ndarray)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, waveform, sr, ref_regions, threshold, sensitivity, smoothing):
        super().__init__()
        self.waveform = waveform
        self.sr = sr
        self.ref_regions = ref_regions
        self.threshold = threshold
        self.sensitivity = sensitivity
        self.smoothing = smoothing
        self._is_cancelled = False

    def cancel(self):
        """Request cancellation of the processing."""
        self._is_cancelled = True

    def run(self):
        """Execute harmony removal algorithm."""
        try:
            # STFT parameters
            n_fft = 2048
            hop_length = 512
            
            self.progress.emit("Computing STFT...")
            
            # Compute STFT of the full audio
            f, t, Zxx = signal.stft(
                self.waveform, 
                fs=self.sr, 
                nperseg=n_fft, 
                noverlap=n_fft - hop_length
            )
            
            if self._is_cancelled:
                self.progress.emit("Processing cancelled.")
                return
            
            # Extract reference spectrum from selected regions
            self.progress.emit("Extracting reference spectrum...")
            ref_spectrum = self._extract_reference_spectrum(Zxx, t)
            
            if self._is_cancelled:
                self.progress.emit("Processing cancelled.")
                return
            
            if ref_spectrum is None:
                self.error.emit("No valid reference regions found.")
                return
            
            # Compute similarity and generate soft mask
            self.progress.emit("Generating spectral mask...")
            mask = self._compute_soft_mask(Zxx, ref_spectrum)
            
            if self._is_cancelled:
                self.progress.emit("Processing cancelled.")
                return
            
            # Apply mask to spectrogram
            self.progress.emit("Applying mask...")
            enhanced_Zxx = Zxx * mask
            
            # Inverse STFT to get time-domain audio
            self.progress.emit("Reconstructing audio...")
            _, enhanced_audio = signal.istft(
                enhanced_Zxx, 
                fs=self.sr, 
                nperseg=n_fft, 
                noverlap=n_fft - hop_length
            )
            
            # Trim to original length
            enhanced_audio = enhanced_audio[:len(self.waveform)]
            
            if not self._is_cancelled:
                self.finished.emit(enhanced_audio)
                
        except Exception as e:
            self.error.emit(f"Processing error: {str(e)}")

    def _extract_reference_spectrum(self, Zxx, time_bins):
        """Extract average spectrum from reference regions."""
        spectra = []
        
        for start_time, end_time in self.ref_regions:
            # Find time bin indices
            start_idx = np.argmin(np.abs(time_bins - start_time))
            end_idx = np.argmin(np.abs(time_bins - end_time))
            
            if end_idx > start_idx:
                # Get magnitude spectrum for this region
                region_spec = np.abs(Zxx[:, start_idx:end_idx])
                # Average over time
                avg_spec = np.mean(region_spec, axis=1)
                spectra.append(avg_spec)
        
        if not spectra:
            return None
        
        # Average across all reference regions
        ref_spectrum = np.mean(spectra, axis=0)
        
        # Normalize
        ref_spectrum = ref_spectrum / (np.linalg.norm(ref_spectrum) + 1e-8)
        
        return ref_spectrum

    def _compute_soft_mask(self, Zxx, ref_spectrum):
        """Compute soft mask based on spectral similarity."""
        n_freqs, n_frames = Zxx.shape
        mask = np.zeros(n_frames)
        
        # Process in chunks to show progress
        chunk_size = 100
        n_chunks = (n_frames + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            if self._is_cancelled:
                return None
            
            start_frame = chunk_idx * chunk_size
            end_frame = min(start_frame + chunk_size, n_frames)
            
            for i in range(start_frame, end_frame):
                # Get current frame magnitude spectrum
                frame_spec = np.abs(Zxx[:, i])
                frame_spec = frame_spec / (np.linalg.norm(frame_spec) + 1e-8)
                
                # Compute cosine similarity
                similarity = np.dot(ref_spectrum, frame_spec)
                
                # Convert to mask value using sigmoid
                # Higher similarity = higher mask value (keep)
                mask[i] = 1.0 / (1.0 + np.exp(-self.sensitivity * (similarity - self.threshold)))
            
            if chunk_idx % 10 == 0:
                self.progress.emit(f"Generating mask... ({chunk_idx + 1}/{n_chunks})")
        
        # Apply temporal smoothing
        if self.smoothing > 0:
            self.progress.emit("Smoothing mask...")
            window_size = int(self.smoothing * self.sr / 512)  # Convert seconds to frames
            if window_size > 1:
                mask = np.convolve(mask, np.ones(window_size) / window_size, mode='same')
        
        # Expand mask to match spectrogram dimensions
        mask = mask[np.newaxis, :]
        
        return mask


class HarmonyRemoverApp(QMainWindow):
    """Main application window for harmony removal."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Harmony Remover - Spectral Voice Extraction")
        self.resize(1200, 1000)
        
        # Audio data
        self.input_file = None
        self.waveform = None
        self.sr = SAMPLE_RATE
        self.processed_waveform = None
        
        # History for undo
        self.history = []
        self.history_idx = -1
        
        # Playback
        self.playback_process = None
        self.playback_start = None
        self.cursor_timer = QTimer()
        self.cursor_timer.timeout.connect(self.update_cursor)
        
        # Worker
        self.processing_worker = None
        
        self.init_ui()

    def init_ui(self):
        """Initialize user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Info group
        info_group = QGroupBox("Application Info")
        info_layout = QHBoxLayout()
        info_label = QLabel("Load a mono vocal track and select reference regions of the target singer.")
        info_label.setStyleSheet("color: #7f8c8d;")
        info_layout.addWidget(info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Control buttons
        bl = QHBoxLayout()
        self.load_b = QPushButton("Load Audio")
        self.play_b = QPushButton("Play Original")
        self.play_proc_b = QPushButton("Play Processed")
        self.stop_b = QPushButton("Stop")
        self.process_b = QPushButton("Process")
        self.undo_b = QPushButton("Undo")
        self.clear_b = QPushButton("Clear Refs")
        self.save_b = QPushButton("Save Result")
        
        self.process_b.setEnabled(False)
        self.play_proc_b.setEnabled(False)
        self.save_b.setEnabled(False)
        
        for b in [self.load_b, self.play_b, self.play_proc_b, self.stop_b, 
                  self.process_b, self.undo_b, self.clear_b]:
            bl.addWidget(b)
        bl.addWidget(self.save_b)
        layout.addLayout(bl)
        
        # Parameter controls
        ctrl = QHBoxLayout()
        
        # Threshold slider
        self.thresh_label = QLabel(f"Threshold: {DEFAULT_THRESHOLD:.2f}")
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(int(DEFAULT_THRESHOLD * 100))
        self.thresh_slider.valueChanged.connect(self.on_thresh_changed)
        
        # Sensitivity slider
        self.sens_label = QLabel(f"Sensitivity: {DEFAULT_SENSITIVITY:.1f}")
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(1, 500)
        self.sens_slider.setValue(int(DEFAULT_SENSITIVITY * 10))
        self.sens_slider.valueChanged.connect(self.on_sens_changed)
        
        # Smoothing slider
        self.smooth_label = QLabel(f"Smoothing: 0.05s")
        self.smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.smooth_slider.setRange(0, 100)
        self.smooth_slider.setValue(5)
        self.smooth_slider.valueChanged.connect(self.on_smooth_changed)
        
        # Zoom slider
        self.zoom_s = QSlider(Qt.Orientation.Horizontal)
        self.zoom_s.setRange(10, 100)
        self.zoom_s.setValue(10)
        self.zoom_s.valueChanged.connect(self.update_view)
        
        # Contrast slider
        self.contrast_s = QSlider(Qt.Orientation.Horizontal)
        self.contrast_s.setRange(1, 100)
        self.contrast_s.setValue(10)
        self.contrast_s.valueChanged.connect(self.update_view)
        
        for w in [self.thresh_label, self.thresh_slider, 
                  self.sens_label, self.sens_slider,
                  self.smooth_label, self.smooth_slider,
                  QLabel("Zoom:"), self.zoom_s, 
                  QLabel("Contrast:"), self.contrast_s]:
            ctrl.addWidget(w)
        layout.addLayout(ctrl)
        
        # Waveform display
        self.scroll = QScrollArea()
        self.wv = WaveformWidget(self)
        self.scroll.setWidget(self.wv)
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll)
        
        # Instructions
        instr_label = QLabel(
            "Instructions: 1) Load audio  2) Draw reference regions on waveform (clean target vocals)  "
            "3) Adjust parameters  4) Click Process  5) Save result"
        )
        instr_label.setStyleSheet("color: #3498db; padding: 5px;")
        layout.addWidget(instr_label)
        
        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        
        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet("font-weight: bold; padding: 2px 10px;")
        self.status.addWidget(self.progress_label, 2)
        
        # Connect signals
        self.load_b.clicked.connect(self.open_audio)
        self.play_b.clicked.connect(self.play_audio)
        self.play_proc_b.clicked.connect(self.play_processed)
        self.stop_b.clicked.connect(self.stop_audio)
        self.process_b.clicked.connect(self.handle_processing)
        self.undo_b.clicked.connect(self.undo)
        self.clear_b.clicked.connect(self.clear_refs)
        self.save_b.clicked.connect(self.save_result)

    def on_thresh_changed(self):
        """Handle threshold slider change."""
        value = self.thresh_slider.value() / 100
        self.thresh_label.setText(f"Threshold: {value:.2f}")

    def on_sens_changed(self):
        """Handle sensitivity slider change."""
        value = self.sens_slider.value() / 10
        self.sens_label.setText(f"Sensitivity: {value:.1f}")

    def on_smooth_changed(self):
        """Handle smoothing slider change."""
        value = self.smooth_slider.value() / 100
        self.smooth_label.setText(f"Smoothing: {value:.2f}s")

    def update_view(self):
        """Update waveform view parameters."""
        self.wv.zoom_factor = self.zoom_s.value() / 10
        self.wv.contrast = self.contrast_s.value() / 10
        self.wv.generate_spectrogram()
        self.wv.update()

    def open_audio(self):
        """Open and load audio file."""
        p, _ = QFileDialog.getOpenFileName(
            self, "Open Audio", "", "Audio Files (*.wav *.mp3 *.flac)"
        )
        if p:
            try:
                # Convert to WAV if needed
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    t_path = tmp.name
                
                subprocess.run(
                    ['ffmpeg', '-i', p, '-ar', str(SAMPLE_RATE), '-ac', '1', '-y', t_path],
                    check=True, capture_output=True
                )
                
                # Load audio
                self.waveform, loaded_sr = sf.read(t_path)
                os.unlink(t_path)
                
                # Ensure mono
                if len(self.waveform.shape) > 1:
                    self.waveform = np.mean(self.waveform, axis=1)
                
                # Ensure float32
                self.waveform = self.waveform.astype(np.float32)
                
                self.input_file = p
                self.processed_waveform = None
                
                # Update UI
                duration = len(self.waveform) / SAMPLE_RATE
                self.wv.set_audio(self.waveform, duration)
                self.save_history()
                
                self.process_b.setEnabled(True)
                self.play_proc_b.setEnabled(False)
                self.save_b.setEnabled(False)
                
                self.progress_label.setText(f"Loaded: {os.path.basename(p)} ({duration:.1f}s)")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load audio: {str(e)}")

    def handle_processing(self):
        """Start or cancel harmony removal processing."""
        if self.processing_worker and self.processing_worker.isRunning():
            self.processing_worker.cancel()
            self.process_b.setText("Process")
        else:
            if self.waveform is None or not self.wv.regions:
                QMessageBox.warning(self, "Warning", "Please select reference regions first.")
                return
            
            self.process_b.setText("Cancel")
            
            # Get parameters
            threshold = self.thresh_slider.value() / 100
            sensitivity = self.sens_slider.value() / 10
            smoothing = self.smooth_slider.value() / 100
            
            # Create and start worker
            self.processing_worker = HarmonyRemovalWorker(
                self.waveform, SAMPLE_RATE, self.wv.regions,
                threshold, sensitivity, smoothing
            )
            self.processing_worker.progress.connect(self.progress_label.setText)
            self.processing_worker.error.connect(lambda msg: QMessageBox.critical(self, "Error", msg))
            self.processing_worker.finished.connect(self.on_processing_done)
            self.processing_worker.start()

    def on_processing_done(self, processed_audio):
        """Handle completion of processing."""
        self.processed_waveform = processed_audio
        self.process_b.setText("Process")
        self.play_proc_b.setEnabled(True)
        self.save_b.setEnabled(True)
        self.progress_label.setText("Processing complete.")

    def play_audio(self):
        """Play original audio."""
        if self.input_file:
            self.stop_audio()
            self.playback_process = subprocess.Popen(
                ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", self.input_file]
            )
            self.playback_start = np.datetime64('now')
            self.cursor_timer.start(50)

    def play_processed(self):
        """Play processed audio."""
        if self.processed_waveform is not None:
            self.stop_audio()
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name
            
            sf.write(temp_path, self.processed_waveform, SAMPLE_RATE)
            
            self.playback_process = subprocess.Popen(
                ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", temp_path]
            )
            self.playback_start = np.datetime64('now')
            self.temp_playback_file = temp_path
            self.cursor_timer.start(50)

    def stop_audio(self):
        """Stop playback."""
        if self.playback_process:
            self.playback_process.terminate()
            self.playback_process = None
        
        self.cursor_timer.stop()
        self.wv.playback_position = -1.0
        self.wv.update()
        
        # Clean up temp file if exists
        if hasattr(self, 'temp_playback_file') and os.path.exists(self.temp_playback_file):
            os.unlink(self.temp_playback_file)
            delattr(self, 'temp_playback_file')

    def update_cursor(self):
        """Update playback cursor position."""
        if self.playback_process and self.playback_process.poll() is None:
            elapsed = (np.datetime64('now') - self.playback_start) / np.timedelta64(1, 's')
            self.wv.playback_position = elapsed
            self.wv.update()
        else:
            self.stop_audio()

    def save_history(self):
        """Save current state for undo."""
        self.history = self.history[:self.history_idx + 1] + [list(self.wv.regions)]
        self.history_idx += 1
        self.undo_b.setEnabled(self.history_idx > 0)

    def undo(self):
        """Undo last action."""
        if self.history_idx > 0:
            self.history_idx -= 1
            self.wv.regions = list(self.history[self.history_idx])
            self.wv.update()
            self.undo_b.setEnabled(self.history_idx > 0)

    def clear_refs(self):
        """Clear all reference regions."""
        self.wv.regions.clear()
        self.wv.update()
        self.save_history()

    def save_result(self):
        """Save processed audio to file."""
        if self.processed_waveform is None:
            return
        
        p, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Audio", "", "WAV (*.wav)"
        )
        if p:
            try:
                sf.write(p, self.processed_waveform, SAMPLE_RATE)
                QMessageBox.information(self, "Success", f"Saved to:\n{p}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HarmonyRemoverApp()
    window.show()
    sys.exit(app.exec())
