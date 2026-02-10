import sys
import os
import subprocess
import ffmpeg
import numpy as np
import signal
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QAbstractItemView, QLabel, QProgressBar,
    QFrame, QFormLayout, QDoubleSpinBox, QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPainter, QColor


class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.samples = None
        self.sample_rate = None
        self.duration = 0.0
        self.playback_position = -1.0
        self.setMinimumHeight(140)

    def set_audio(self, samples, sample_rate, duration):
        self.samples = samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.update()

    def paintEvent(self, event):
        if self.samples is None or len(self.samples) == 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        center_y = height // 2
        painter.fillRect(event.rect(), Qt.GlobalColor.white)

        num_points = min(len(self.samples), width * 2)
        step = max(1, len(self.samples) // num_points)
        indices = np.arange(0, len(self.samples), step)
        values = self.samples[indices]
        times = indices / self.sample_rate
        x_vals = (times / self.duration * width).astype(int)

        max_val = np.max(np.abs(values)) + 1e-6
        y_scale = (height // 3) / max_val
        y_vals = center_y - (values * y_scale).astype(int)

        painter.setPen(Qt.GlobalColor.blue)
        for i in range(len(x_vals) - 1):
            painter.drawLine(x_vals[i], y_vals[i], x_vals[i + 1], y_vals[i + 1])

        if self.playback_position >= 0:
            playback_x = int((self.playback_position / self.duration) * width)
            painter.setPen(QColor(255, 0, 0, 200))
            painter.drawLine(playback_x, 0, playback_x, height)

    def set_playback_position(self, position):
        self.playback_position = position
        self.update()


class MergeThread(QThread):
    merge_finished = pyqtSignal(str)
    merge_error = pyqtSignal(str)

    def __init__(self, file_paths, output_path, crossfade_duration=0.05, silence_gap=0.0, normalize=True, curve='tri'):
        super().__init__()
        self.file_paths = file_paths
        self.output_path = output_path
        self.crossfade_duration = crossfade_duration
        self.silence_gap = silence_gap
        self.normalize = normalize
        self.curve = curve

    def run(self):
        try:
            # Format Unification
            probe = ffmpeg.probe(self.file_paths[0])
            audio_stream = next((s for s in probe["streams"] if s["codec_type"] == "audio"), None)
            target_sr = int(audio_stream["sample_rate"])
            target_ch = int(audio_stream.get("channels", 1))

            processed_inputs = []
            for file_path in self.file_paths:
                inp = ffmpeg.input(file_path)
                inp = ffmpeg.filter(inp, 'aresample', target_sr)
                inp = ffmpeg.filter(inp, 'aformat', channel_layouts='stereo' if target_ch >= 2 else 'mono')

                # Volume Normalization
                if self.normalize:
                    inp = ffmpeg.filter(inp, 'loudnorm', I=-16, TP=-1.5, LRA=11)
                processed_inputs.append(inp)

            if self.silence_gap > 0:
                final_chain = []
                for i, stream in enumerate(processed_inputs):
                    final_chain.append(stream)
                    if i < len(processed_inputs) - 1:
                        silence = ffmpeg.input(f'anullsrc=r={target_sr}:cl={"stereo" if target_ch >= 2 else "mono"}',
                                               f='lavfi', t=self.silence_gap)
                        final_chain.append(silence)
                concatenated = ffmpeg.concat(*final_chain, v=0, a=1)
            elif self.crossfade_duration > 0 and len(processed_inputs) > 1:
                # Use high-quality acrossfade with user-selected curve
                concatenated = processed_inputs[0]
                for i in range(1, len(processed_inputs)):
                    concatenated = ffmpeg.filter([concatenated, processed_inputs[i]], 'acrossfade',
                                                 d=self.crossfade_duration, c1=self.curve, c2=self.curve)
            else:
                concatenated = ffmpeg.concat(*processed_inputs, v=0, a=1)

            output = ffmpeg.output(concatenated, self.output_path, acodec='pcm_s16le', ar=target_sr, ac=target_ch)
            ffmpeg.run(output, overwrite_output=True, quiet=True)
            self.merge_finished.emit(self.output_path)
        except Exception as e:
            self.merge_error.emit(str(e))


class AudioMergerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro Audio Merger & Vocoder Optimizer")
        self.setGeometry(100, 100, 1000, 850)
        self.file_paths = []
        self.is_playing_merged = False
        self.is_paused_merged = False
        self.playback_position = 0.0
        self.playback_process = None
        self.selected_playback_process = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Settings Panel
        settings_frame = QFrame()
        settings_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        settings_layout = QHBoxLayout(settings_frame)

        self.crossfade_spin = QDoubleSpinBox()
        self.crossfade_spin.setRange(0.0, 1.0);
        self.crossfade_spin.setValue(0.05);
        self.crossfade_spin.setSuffix(" s")

        # Curve Selection Dropdown
        self.curve_combo = QComboBox()
        # Mapping common names to FFmpeg curve shorthand
        self.curves = {
            "Triangle (Default)": "tri",
            "Linear": "lin",
            "Constant Power": "iqsin",
            "Exponential": "exp",
            "Logarithmic": "log",
            "Parabolic": "par"
        }
        self.curve_combo.addItems(list(self.curves.keys()))

        self.silence_spin = QDoubleSpinBox()
        self.silence_spin.setRange(0.0, 2.0);
        self.silence_spin.setValue(0.0);
        self.silence_spin.setSuffix(" s")
        self.normalize_check = QCheckBox("Normalize");
        self.normalize_check.setChecked(True)

        settings_layout.addWidget(QLabel("Crossfade:"))
        settings_layout.addWidget(self.crossfade_spin)
        settings_layout.addWidget(QLabel("Curve:"))
        settings_layout.addWidget(self.curve_combo)
        settings_layout.addWidget(QLabel("Silence Gap:"))
        settings_layout.addWidget(self.silence_spin)
        settings_layout.addWidget(self.normalize_check)
        main_layout.addWidget(settings_frame)

        # File List Controls
        list_controls = QHBoxLayout()
        self.add_btn = QPushButton("Add Audio")
        self.remove_btn = QPushButton("Remove Selected")
        self.play_selected_btn = QPushButton("Play Selected")
        self.up_btn = QPushButton("Move Up")
        self.down_btn = QPushButton("Move Down")

        for btn in [self.add_btn, self.remove_btn, self.play_selected_btn, self.up_btn, self.down_btn]:
            list_controls.addWidget(btn)
        main_layout.addLayout(list_controls)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        main_layout.addWidget(self.file_list)

        # Merged Audio Controls
        self.waveform_widget = WaveformWidget()
        main_layout.addWidget(self.waveform_widget)

        merge_controls = QHBoxLayout()
        self.merge_btn = QPushButton("Merge All Files")
        # self.merge_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; height: 40px;")

        self.play_merged_btn = QPushButton("Play Merged")
        self.pause_resume_btn = QPushButton("Pause")
        self.stop_merged_btn = QPushButton("Stop")
        self.pause_resume_btn.setEnabled(False)

        merge_controls.addWidget(self.merge_btn)
        merge_controls.addWidget(self.play_merged_btn)
        merge_controls.addWidget(self.pause_resume_btn)
        merge_controls.addWidget(self.stop_merged_btn)
        main_layout.addLayout(merge_controls)

        self.progress_bar = QProgressBar();
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Signal Connections
        self.add_btn.clicked.connect(self.add_files)
        self.remove_btn.clicked.connect(self.remove_selected)
        self.play_selected_btn.clicked.connect(self.toggle_selected_play)
        self.up_btn.clicked.connect(self.move_up)
        self.down_btn.clicked.connect(self.move_down)
        self.merge_btn.clicked.connect(self.start_merge)
        self.play_merged_btn.clicked.connect(self.play_merged)
        self.pause_resume_btn.clicked.connect(self.toggle_pause_resume)
        self.stop_merged_btn.clicked.connect(self.stop_merged)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Audio", "", "Audio (*.mp3 *.wav *.flac *.m4a)")
        for f in files:
            item = QListWidgetItem(Path(f).name)
            item.setData(Qt.ItemDataRole.UserRole, f)
            self.file_list.addItem(item)
            self.file_paths.append(f)

    def remove_selected(self):
        row = self.file_list.currentRow()
        if row >= 0:
            self.file_list.takeItem(row)
            del self.file_paths[row]

    def move_up(self):
        row = self.file_list.currentRow()
        if row > 0:
            item = self.file_list.takeItem(row)
            self.file_list.insertItem(row - 1, item)
            self.file_list.setCurrentRow(row - 1)
            self.file_paths[row], self.file_paths[row - 1] = self.file_paths[row - 1], self.file_paths[row]

    def move_down(self):
        row = self.file_list.currentRow()
        if 0 <= row < self.file_list.count() - 1:
            item = self.file_list.takeItem(row)
            self.file_list.insertItem(row + 1, item)
            self.file_list.setCurrentRow(row + 1)
            self.file_paths[row], self.file_paths[row + 1] = self.file_paths[row + 1], self.file_paths[row]

    def toggle_selected_play(self):
        if self.selected_playback_process:
            self.selected_playback_process.terminate()
            self.selected_playback_process = None
            self.play_selected_btn.setText("Play Selected")
        else:
            row = self.file_list.currentRow()
            if row >= 0:
                path = self.file_paths[row]
                self.selected_playback_process = subprocess.Popen(
                    ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", path])
                self.play_selected_btn.setText("Stop Selected")

    def start_merge(self):
        if not self.file_paths: return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Merged", "", "WAV (*.wav)")
        if not out_path: return

        self.progress_bar.setVisible(True);
        self.progress_bar.setRange(0, 0)

        # Get the FFmpeg curve code from selection
        selected_text = self.curve_combo.currentText()
        curve_code = self.curves[selected_text]

        self.thread = MergeThread(
            self.file_paths,
            out_path,
            self.crossfade_spin.value(),
            self.silence_spin.value(),
            self.normalize_check.isChecked(),
            curve=curve_code
        )
        self.thread.merge_finished.connect(self.on_merge_success)
        self.thread.merge_error.connect(self.on_merge_error)
        self.thread.start()

    def on_merge_success(self, path):
        self.progress_bar.setVisible(False)
        self.merged_file_path = path
        self.load_merged_waveform(path)
        QMessageBox.information(self, "Success", f"Merge completed using {self.curve_combo.currentText()} curve.")

    def on_merge_error(self, err):
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", f"Merge failed: {err}")

    def load_merged_waveform(self, path):
        probe = ffmpeg.probe(path)
        audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        sr, dur = int(audio['sample_rate']), float(audio['duration'])
        out, _ = (
            ffmpeg.input(path).output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=sr).run(capture_stdout=True,
                                                                                                quiet=True))
        samples = np.frombuffer(out, dtype=np.int16).astype(np.float32)
        self.waveform_widget.set_audio(samples, sr, dur)

    def play_merged(self):
        if hasattr(self, 'merged_file_path'):
            self.stop_merged()
            self.playback_process = subprocess.Popen(
                ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", self.merged_file_path])
            self.is_playing_merged = True
            self.is_paused_merged = False
            self.pause_resume_btn.setEnabled(True)
            self.pause_resume_btn.setText("Pause")

            self.playback_timer = QTimer()
            self.playback_timer.timeout.connect(self.update_playback)
            self.playback_timer.start(100)

    def toggle_pause_resume(self):
        if not self.playback_process: return

        if not self.is_paused_merged:
            if sys.platform != "win32":
                os.kill(self.playback_process.pid, signal.SIGSTOP)
            else:
                self.playback_process.terminate()

            self.is_paused_merged = True
            self.pause_resume_btn.setText("Resume")
            self.playback_timer.stop()
        else:
            if sys.platform != "win32":
                os.kill(self.playback_process.pid, signal.SIGCONT)
            else:
                self.playback_process = subprocess.Popen([
                    "ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet",
                    "-ss", str(self.playback_position), self.merged_file_path
                ])

            self.is_paused_merged = False
            self.pause_resume_btn.setText("Pause")
            self.playback_timer.start(100)

    def stop_merged(self):
        if self.playback_process:
            self.playback_process.terminate()
            self.playback_process = None
        self.is_playing_merged = False
        self.is_paused_merged = False
        self.playback_position = 0.0
        self.waveform_widget.set_playback_position(0.0)
        self.pause_resume_btn.setText("Pause")
        self.pause_resume_btn.setEnabled(False)
        if hasattr(self, 'playback_timer'):
            self.playback_timer.stop()

    def update_playback(self):
        if self.is_playing_merged and not self.is_paused_merged:
            self.playback_position += 0.1
            self.waveform_widget.set_playback_position(self.playback_position)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioMergerApp()
    window.show()
    sys.exit(app.exec())