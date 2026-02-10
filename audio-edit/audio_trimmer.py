import sys
import os
import subprocess
import ffmpeg
import numpy as np
import signal
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QStatusBar,
    QDoubleSpinBox, QComboBox, QCheckBox, QFrame, QProgressBar, QSlider, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal as Signal, QThread, QObject, QTimer, QRect
from PyQt6.QtGui import QPainter, QColor, QImage


class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.samples = None
        self.sample_rate = None
        self.duration = 0.0
        self.regions = []
        self.playback_position = -1.0
        self.zoom_factor = 1.0
        self.contrast = 1.0
        self.spectrogram_img = None
        self._spec_data = None

        self.active_region_idx = -1
        self.interaction_mode = None
        self.drag_start_pos = None
        self.drag_start_region = None

        self.setMouseTracking(True)
        self.setMinimumHeight(450)

    def set_audio(self, samples, sample_rate, duration):
        self.samples = samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.regions.clear()
        self.generate_spectrogram()
        self.update()

    def generate_spectrogram(self):
        """Generates a high-contrast colorful spectrogram."""
        if self.samples is None: return
        n_fft, hop_length = 1024, 512
        data = self.samples[:2000000] if len(self.samples) > 2000000 else self.samples
        window = np.hanning(n_fft)
        spec = []
        for i in range(0, len(data) - n_fft, hop_length):
            segment = data[i:i + n_fft] * window
            fft = np.abs(np.fft.rfft(segment))
            spec.append(fft)
        spec = np.array(spec).T
        spec = np.log1p(spec * self.contrast)
        if spec.size == 0: return

        norm_spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        h, w = norm_spec.shape

        # Colorful Viridis-like Mapping
        img_data = np.zeros((h, w, 4), dtype=np.uint8)
        img_data[..., 0] = (norm_spec * 255).astype(np.uint8)  # Red
        img_data[..., 1] = (np.sin(norm_spec * np.pi) * 255).astype(np.uint8)  # Green
        img_data[..., 2] = ((1 - norm_spec) * 150).astype(np.uint8)  # Blue
        img_data[..., 3] = 255  # Alpha

        self._spec_data = np.ascontiguousarray(img_data).tobytes()
        self.spectrogram_img = QImage(self._spec_data, w, h, QImage.Format.Format_RGBA8888)

    def get_view_width(self):
        return int(self.width() * self.zoom_factor)

    def t_to_x(self, t):
        if self.duration <= 0: return 0
        return int((t / self.duration) * self.get_view_width())

    def x_to_t(self, x):
        vw = self.get_view_width()
        return (x / vw) * self.duration if vw > 0 else 0

    def get_region_at(self, x):
        handle_size = 12
        for i, (s, e) in enumerate(self.regions):
            xs, xe = self.t_to_x(s), self.t_to_x(e)
            if abs(x - xs) < handle_size: return i, 'resize_left'
            if abs(x - xe) < handle_size: return i, 'resize_right'
            if xs < x < xe: return i, 'move'
        return -1, 'draw'

    def mousePressEvent(self, event):
        if self.samples is None: return
        x = event.position().x()
        self.active_region_idx, self.interaction_mode = self.get_region_at(x)
        self.drag_start_pos = x
        if self.interaction_mode == 'draw':
            self.drag_start_time = self.x_to_t(x)
            self.current_draw_x = x
        else:
            self.drag_start_region = list(self.regions[self.active_region_idx])
        self.update()

    def mouseMoveEvent(self, event):
        if self.samples is None: return
        x = event.position().x()
        if self.drag_start_pos is None:
            _, mode = self.get_region_at(x)
            self.setCursor(Qt.CursorShape.SizeAllCursor if mode == 'move' else
                           Qt.CursorShape.SizeHorCursor if 'resize' in mode else
                           Qt.CursorShape.CrossCursor)
            return

        dt = self.x_to_t(x) - self.x_to_t(self.drag_start_pos)
        if self.interaction_mode == 'move':
            s, e = self.drag_start_region
            self.regions[self.active_region_idx] = [max(0, s + dt), min(self.duration, e + dt)]
        elif 'resize' in self.interaction_mode:
            s, e = self.drag_start_region
            if self.interaction_mode == 'resize_left':
                self.regions[self.active_region_idx][0] = min(e - 0.02, max(0, s + dt))
            else:
                self.regions[self.active_region_idx][1] = max(s + 0.02, min(self.duration, e + dt))
        elif self.interaction_mode == 'draw':
            self.current_draw_x = x
        self.update()

    def mouseReleaseEvent(self, event):
        if self.interaction_mode == 'draw' and self.drag_start_pos is not None:
            et = self.x_to_t(event.position().x())
            s, e = sorted([self.drag_start_time, et])
            if e - s > 0.02: self.regions.append([s, e])
        self.drag_start_pos = None
        self.update()

    def paintEvent(self, event):
        if self.samples is None: return
        painter = QPainter(self)
        vw, h = self.get_view_width(), self.height()
        mid_h = h // 2
        painter.fillRect(event.rect(), Qt.GlobalColor.black)

        if self.spectrogram_img:
            painter.drawImage(QRect(0, 0, vw, mid_h), self.spectrogram_img)

        painter.setPen(QColor(0, 255, 255))
        num_points = min(len(self.samples), vw * 2)
        step = max(1, len(self.samples) // num_points)
        indices = np.arange(0, len(self.samples), step)
        vals = self.samples[indices]
        y_scale = (mid_h // 1.5) / (np.max(np.abs(vals)) + 1e-6)
        wf_center = mid_h + (mid_h // 2)
        for i in range(len(vals) - 1):
            x1 = int((indices[i] / len(self.samples)) * vw)
            x2 = int((indices[i + 1] / len(self.samples)) * vw)
            painter.drawLine(x1, int(wf_center - vals[i] * y_scale), x2, int(wf_center - vals[i + 1] * y_scale))

        for s, e in self.regions:
            xs, xe = self.t_to_x(s), self.t_to_x(e)
            painter.setPen(QColor(255, 0, 0, 220));
            painter.setBrush(QColor(255, 0, 0, 60))
            painter.drawRect(xs, 0, xe - xs, h)
            painter.fillRect(xs, 0, 4, h, QColor(255, 0, 0));
            painter.fillRect(xe - 4, 0, 4, h, QColor(255, 0, 0))
        if self.playback_position >= 0:
            px = self.t_to_x(self.playback_position)
            painter.setPen(QColor(255, 255, 255));
            painter.drawLine(px, 0, px, h)


class FFmpegWorker(QObject):
    finished, error = Signal(str), Signal(str)

    def __init__(self, input_path, regions, output_path, cfg):
        super().__init__()
        self.input_path, self.regions, self.output_path, self.cfg = input_path, regions, output_path, cfg

    def run(self):
        try:
            probe = ffmpeg.probe(self.input_path)
            audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
            sr = int(audio["sample_rate"])
            processed = []
            for i, (s, e) in enumerate(self.regions):
                ext_s, ext_e = max(0, s - self.cfg['padding']), min(float(audio['duration']), e + self.cfg['padding'])
                dur = ext_e - ext_s
                seg = ffmpeg.input(self.input_path, ss=ext_s, t=dur).filter('aresample', sr)
                f_in = self.cfg['padding'] if i == 0 else 0
                seg = seg.filter('afade', t='in', st=f_in, d=self.cfg['fade'])
                f_out = (dur - self.cfg['padding'] - self.cfg['fade']) if i == len(self.regions) - 1 else (
                            dur - self.cfg['fade'])
                seg = seg.filter('afade', t='out', st=max(0, f_out), d=self.cfg['fade'])
                processed.append(seg)
            joined = processed[0]
            if len(processed) > 1:
                for k in range(1, len(processed)):
                    joined = ffmpeg.filter([joined, processed[k]], 'acrossfade', d=self.cfg['fade'],
                                           c1=self.cfg['curve'], c2=self.cfg['curve'])
            if self.cfg['normalize']: joined = joined.filter('loudnorm', I=-16, TP=-1.5, LRA=11)
            ffmpeg.output(joined, self.output_path, acodec="pcm_s16le", ar=sr).run(overwrite_output=True, quiet=True)
            self.finished.emit(self.output_path)
        except Exception as e:
            self.error.emit(str(e))


class AudioTrimmerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Vocoder Trimmer")
        self.resize(1200, 950)
        self.input_file = None
        self.playback_process = None
        self.preview_timer = QTimer()
        self.cursor_timer = QTimer()
        self.cursor_timer.timeout.connect(self.update_cursor)
        self.init_ui()

    def init_ui(self):
        central = QWidget();
        self.setCentralWidget(central);
        layout = QVBoxLayout(central)

        # Merge Settings Panel
        f = QFrame();
        f.setFrameStyle(QFrame.Shape.StyledPanel);
        fl = QHBoxLayout(f)
        self.fade_s, self.pad_s = QDoubleSpinBox(), QDoubleSpinBox()
        self.fade_s.setValue(0.05);
        self.pad_s.setValue(0.1)
        self.curve_c = QComboBox();
        self.curve_c.addItems(["tri", "qsin", "exp"])
        self.norm_check = QCheckBox("LUFS Norm");
        self.norm_check.setChecked(True)
        for w in [QLabel("Fade:"), self.fade_s, QLabel("Pad:"), self.pad_s, self.curve_c,
                  self.norm_check]: fl.addWidget(w)
        layout.addWidget(f)

        # Main Button Toolbar
        bl = QHBoxLayout()
        self.load_b, self.play_b, self.play_sel_b = QPushButton("Load"), QPushButton("Play Full"), QPushButton(
            "Play Seq")
        self.stop_b, self.undo_b, self.clear_b = QPushButton("Stop"), QPushButton("Undo"), QPushButton("Clear All")
        self.save_b = QPushButton("Merge & Save")
        # self.save_b.setStyleSheet("background-color: #1B5E20; color: white; font-weight: bold;")
        for b in [self.load_b, self.play_b, self.play_sel_b, self.stop_b, self.undo_b, self.clear_b,
                  self.save_b]: bl.addWidget(b)
        layout.addLayout(bl)

        # Visualization Sliders
        zl = QHBoxLayout()
        self.zoom_s = QSlider(Qt.Orientation.Horizontal);
        self.zoom_s.setRange(10, 100);
        self.zoom_s.setValue(10)
        self.zoom_s.valueChanged.connect(self.update_zoom)
        self.contrast_s = QSlider(Qt.Orientation.Horizontal);
        self.contrast_s.setRange(1, 100);
        self.contrast_s.setValue(10)
        self.contrast_s.valueChanged.connect(self.update_contrast)
        zl.addWidget(QLabel("Zoom:"));
        zl.addWidget(self.zoom_s)
        zl.addWidget(QLabel("Contrast:"));
        zl.addWidget(self.contrast_s)
        layout.addLayout(zl)

        self.scroll = QScrollArea();
        self.wv = WaveformWidget();
        self.scroll.setWidget(self.wv);
        self.scroll.setWidgetResizable(True);
        layout.addWidget(self.scroll)
        self.progress_bar = QProgressBar();
        self.progress_bar.setVisible(False);
        layout.addWidget(self.progress_bar)

        # Signals
        self.load_b.clicked.connect(self.load_audio);
        self.play_b.clicked.connect(self.play_full)
        self.play_sel_b.clicked.connect(self.start_sequential_preview);
        self.stop_b.clicked.connect(self.stop_audio)
        self.undo_b.clicked.connect(self.undo_last);
        self.clear_b.clicked.connect(self.clear_all_regions)
        self.save_b.clicked.connect(self.save_optimized)

    def update_contrast(self):
        self.wv.contrast = self.contrast_s.value() / 10.0
        self.wv.generate_spectrogram()
        self.wv.update()

    def update_zoom(self):
        self.wv.zoom_factor = self.zoom_s.value() / 10.0;
        self.wv.update()

    def load_audio(self, path=None):
        if not path:
            from PyQt6.QtCore import QStandardPaths
            home_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.HomeLocation)
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Open Audio File",
                home_dir,
                "Audio Files (*.wav *.mp3 *.flac *.aac *.m4a *.ogg *.wma);;All Files (*)"
            )
        if not path: return
        self.input_file = path
        try:
            probe = ffmpeg.probe(path);
            audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
            sr, dur = int(audio["sample_rate"]), float(audio["duration"])
            out, _ = (ffmpeg.input(path).output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=sr).run(
                capture_stdout=True, quiet=True))
            self.wv.set_audio(np.frombuffer(out, dtype=np.int16).astype(np.float32), sr, dur)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def play_full(self):
        self.stop_audio();
        self.playback_process = subprocess.Popen(
            ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", self.input_file])
        self.playback_start = np.datetime64('now');
        self.cursor_timer.start(50)

    def start_sequential_preview(self):
        if not self.wv.regions: return
        self.stop_audio();
        self.sorted_regs = sorted(self.wv.regions, key=lambda x: x[0]);
        self.prev_idx = 0;
        self.play_next_region()

    def play_next_region(self):
        if self.prev_idx < len(self.sorted_regs):
            s, e = self.sorted_regs[self.prev_idx]
            self.playback_process = subprocess.Popen(
                ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", "-ss", str(s), "-t", str(e - s),
                 self.input_file])
            self.playback_start = np.datetime64('now') - np.timedelta64(int(s * 1000), 'ms');
            self.cursor_timer.start(50)
            self.preview_timer.singleShot(int((e - s) * 1000), self.check_preview)

    def check_preview(self):
        if self.playback_process: self.prev_idx += 1; self.play_next_region()

    def update_cursor(self):
        if self.playback_process and self.playback_process.poll() is None:
            elapsed = (np.datetime64('now') - self.playback_start) / np.timedelta64(1, 's');
            self.wv.playback_position = elapsed;
            self.wv.update()
        else:
            self.stop_audio()

    def stop_audio(self):
        if self.playback_process: self.playback_process.terminate(); self.playback_process = None
        self.cursor_timer.stop();
        self.preview_timer.stop();
        self.wv.playback_position = -1.0;
        self.wv.update()

    def undo_last(self):
        if self.wv.regions: self.wv.regions.pop(); self.wv.update()

    def clear_all_regions(self):
        if self.wv.regions:
            if QMessageBox.question(self, "Confirm", "Clear all selected regions?") == QMessageBox.StandardButton.Yes:
                self.wv.regions.clear()
                self.wv.update()

    def save_optimized(self):
        if not self.wv.regions: return
        path, _ = QFileDialog.getSaveFileName(self, "Save", "", "WAV (*.wav)")
        if not path: return
        self.progress_bar.setVisible(True);
        self.progress_bar.setRange(0, 0)
        cfg = {'fade': self.fade_s.value(), 'padding': self.pad_s.value(), 'curve': self.curve_c.currentText(),
               'normalize': self.norm_check.isChecked()}
        self.worker = FFmpegWorker(self.input_file, self.wv.regions, path, cfg);
        self.thread = QThread()
        self.worker.moveToThread(self.thread);
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_success);
        self.worker.error.connect(self.on_error)
        self.thread.start()

    def on_success(self, path):
        self.progress_bar.setVisible(False);
        self.thread.quit()
        QMessageBox.information(self, "Success", f"Optimized file re-loaded: {path}")
        self.load_audio(path)

    def on_error(self, err):
        self.progress_bar.setVisible(False);
        self.thread.quit()
        QMessageBox.critical(self, "Error", err)


if __name__ == "__main__":
    app = QApplication(sys.argv);
    w = AudioTrimmerApp();
    w.show();
    sys.exit(app.exec())