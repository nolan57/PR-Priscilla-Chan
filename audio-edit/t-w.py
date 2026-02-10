#!/usr/bin/env python3
"""
Real-time Audio Visualizer using PyQt6, pyqtgraph, and FFmpeg
Displays Waveform, PSD, and Spectrogram simultaneously
"""

import sys
import warnings

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=".*sipPyTypeDict.*"
)
import numpy as np
import subprocess
import threading
import queue
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QFrame,
)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
import pyqtgraph as pg

# === Configuration ===
DEVICE_NAME = (
    "0"  # Use BlackHole 64ch (device 0) - Change to "1" for Microsoft Teams Audio
)
SAMPLE_RATE = 48000  # Match the detected sample rate from BlackHole
CHUNK_SIZE = 512
CHANNELS = 2
GAIN = 5.0  # Increased gain to make waveform more visible
PLOT_LENGTH = 1024
FFT_WINDOW = 2048

# FFmpeg configuration
FFMPEG_PATH = "ffmpeg"
BUFFER_SIZE = 8192

# Audio buffer and thread communication
audio_queue = queue.Queue(maxsize=100)
audio_buffer = np.zeros(PLOT_LENGTH * CHANNELS, dtype=np.float32)


class FFmpegAudioThread(QThread):
    """Thread to capture audio using FFmpeg"""

    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        try:
            # FFmpeg command to capture audio from device with proper channel handling
            cmd = [
                FFMPEG_PATH,
                "-f",
                "avfoundation",
                "-i",
                f":{DEVICE_NAME}",  # :N means audio device N
                "-af",
                "pan=stereo|c0=c0|c1=c1",  # Explicitly map first two channels to stereo
                "-f",
                "f32le",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                str(CHANNELS),
                "pipe:1",
            ]

            print(f"🎙️  Starting FFmpeg with command: {' '.join(cmd)}")

            # On Linux, you might need to use pulse or alsa instead:
            # ['-f', 'pulse', '-i', DEVICE_NAME] or ['-f', 'alsa', '-i', DEVICE_NAME]

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=BUFFER_SIZE
            )

            # Start a thread to read stderr for debugging
            def read_stderr():
                for line in iter(process.stderr.readline, b""):
                    print(f"🔧 FFmpeg stderr: {line.decode('utf-8').strip()}")

            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            while self.running:
                # Read audio data
                raw_data = process.stdout.read(
                    CHUNK_SIZE * 4 * CHANNELS
                )  # 4 bytes per float32
                if not raw_data:
                    break

                # Convert to numpy array
                audio_data = np.frombuffer(raw_data, dtype=np.float32)

                # Put in queue (non-blocking)
                try:
                    audio_queue.put_nowait(audio_data)
                except queue.Full:
                    # Drop oldest data if queue is full
                    try:
                        audio_queue.get_nowait()
                        audio_queue.put_nowait(audio_data)
                    except:
                        pass

            process.terminate()
            process.wait()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self.running = False


class AudioVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Audio Visualizer - {DEVICE_NAME}")
        self.setGeometry(100, 100, 1200, 900)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # === Row 1: Waveform ===
        waveform_frame = self.create_frame("Waveform (Left & Right Channels)")
        layout.addWidget(waveform_frame)

        waveform_layout = waveform_frame.layout()

        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setBackground("k")
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.3)
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.setLabel("bottom", "Sample")
        self.waveform_plot.setYRange(-2.0, 2.0)
        self.waveform_plot.setXRange(0, PLOT_LENGTH)

        # Waveform curves
        self.waveform_left = self.waveform_plot.plot(
            pen=pg.mkPen(color="g", width=1), name="Left"
        )
        self.waveform_right = self.waveform_plot.plot(
            pen=pg.mkPen(color="c", width=1), name="Right"
        )

        waveform_layout.addWidget(self.waveform_plot)

        # === Row 2: PSD (Power Spectral Density) ===
        psd_frame = self.create_frame("Power Spectral Density")
        layout.addWidget(psd_frame)

        psd_layout = psd_frame.layout()

        self.psd_plot = pg.PlotWidget()
        self.psd_plot.setBackground("k")
        self.psd_plot.showGrid(x=True, y=True, alpha=0.3)
        self.psd_plot.setLabel("left", "Power/Frequency")
        self.psd_plot.setLabel("bottom", "Frequency (Hz)")
        self.psd_plot.setLogMode(x=True, y=False)
        self.psd_plot.setXRange(np.log10(20), np.log10(12000))
        # Auto-scale PSD Y-axis to fit the data range
        self.psd_plot.enableAutoRange(axis=pg.ViewBox.YAxis)

        # PSD curve
        self.psd_curve = self.psd_plot.plot(pen=pg.mkPen(color="y", width=1.5))

        psd_layout.addWidget(self.psd_plot)

        # === Row 3: Spectrogram ===
        spec_frame = self.create_frame("Spectrogram")
        layout.addWidget(spec_frame)

        spec_layout = spec_frame.layout()

        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setBackground("k")
        self.spec_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spec_plot.setLabel("left", "Frequency (Hz) - Low ↓ High ↑")
        self.spec_plot.setLabel("bottom", "Time (s)")
        self.spec_plot.setYRange(0, 20000)

        # Spectrogram image item
        self.spec_img = pg.ImageItem()
        self.spec_plot.addItem(self.spec_img)

        # Color map for spectrogram
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        color = np.array(
            [
                [0, 0, 0, 255],
                [0, 0, 255, 255],
                [0, 255, 255, 255],
                [255, 255, 0, 255],
                [255, 0, 0, 255],
            ],
            dtype=np.ubyte,
        )
        cmap = pg.ColorMap(pos, color)
        self.spec_img.setLookupTable(cmap.getLookupTable())

        spec_layout.addWidget(self.spec_plot)

        # RMS Label
        self.rms_label = QLabel("RMS L: 0.000 (0.0 dBFS)  RMS R: 0.000 (0.0 dBFS)")
        self.rms_label.setFont(QFont("Menlo", 10))  # Use Menlo instead of Monospace
        self.rms_label.setStyleSheet(
            "color: white; background-color: black; padding: 5px;"
        )
        layout.addWidget(self.rms_label)

        # Spectrogram history
        self.spec_history = []
        self.spec_time_bins = 150
        self.spec_freq_bins = 128
        self.spec_window_size = 1024

        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)  # 50ms = 20 FPS

        # Start audio capture thread
        self.audio_thread = FFmpegAudioThread()
        self.audio_thread.error_occurred.connect(self.handle_audio_error)
        self.audio_thread.start()

    def handle_audio_error(self, error_msg):
        print(f"❌ Audio thread error: {error_msg}")

    def create_frame(self, title):
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        frame.setStyleSheet(
            "QFrame { background-color: #1a1a1a; border: 1px solid #444; }"
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        frame.setLayout(layout)

        title_label = QLabel(title)
        title_label.setFont(
            QFont("Menlo", 10)
        )  # Use Menlo instead of Monospace on macOS
        title_label.setStyleSheet("color: white; padding: 2px;")
        layout.addWidget(title_label)

        return frame

    def update_plots(self):
        global audio_buffer, audio_queue

        # Debug: Print audio buffer status
        if hasattr(self, "debug_counter"):
            self.debug_counter += 1
        else:
            self.debug_counter = 0
            print(f"🔍 Debug: Starting audio visualization")
            print(
                f"🔍 Buffer size: {len(audio_buffer)}, Queue size: {audio_queue.qsize()}"
            )

        # Get new audio data from queue
        data_updated = False
        while not audio_queue.empty():
            try:
                new_data = audio_queue.get_nowait()
                if len(new_data) > 0:
                    new_len = len(new_data)
                    audio_buffer = np.roll(audio_buffer, -new_len)
                    audio_buffer[-new_len:] = new_data
                    data_updated = True
                    if self.debug_counter % 100 == 0:  # Print every 100 updates
                        print(
                            f"📊 Data received: {new_len} samples, buffer mean: {np.mean(np.abs(audio_buffer[-1000:])):.6f}"
                        )
            except queue.Empty:
                break
            except Exception as e:
                print(f"❌ Queue error: {e}")
                break

        # If no real audio data, generate stronger test signal for demonstration
        if not data_updated:
            # Generate a louder test tone continuously
            phase = (self.debug_counter * 0.1) % (2 * np.pi)
            t = np.arange(512) / SAMPLE_RATE
            test_signal = 0.5 * np.sin(
                2 * np.pi * 440 * t + phase
            )  # Increased amplitude to 0.5
            test_stereo = np.repeat(test_signal, 2)  # Make stereo
            audio_buffer = np.roll(audio_buffer, -len(test_stereo))
            audio_buffer[-len(test_stereo) :] = test_stereo
            data_updated = True
            if self.debug_counter % 50 == 0:
                print(
                    f"🎵 Generated test signal (440 Hz, amp=0.5) - No real audio input detected"
                )

        # Extract channel data
        if CHANNELS == 2:
            full_left = audio_buffer[0::2][-PLOT_LENGTH:]
            full_right = audio_buffer[1::2][-PLOT_LENGTH:]
            if len(full_left) < PLOT_LENGTH:
                pad = PLOT_LENGTH - len(full_left)
                full_left = np.pad(full_left, (pad, 0), constant_values=0)
                full_right = np.pad(full_right, (pad, 0), constant_values=0)
        else:
            mono = audio_buffer[-PLOT_LENGTH:]
            if len(mono) < PLOT_LENGTH:
                mono = np.pad(mono, (PLOT_LENGTH - len(mono), 0), constant_values=0)
            full_left = full_right = mono

        # Calculate PSD
        fft_input = audio_buffer[-FFT_WINDOW:]
        if len(fft_input) < FFT_WINDOW:
            fft_input = np.pad(fft_input, (FFT_WINDOW - len(fft_input), 0))

        if CHANNELS == 2:
            fft_signal = fft_input[0::2][: FFT_WINDOW // 2]
        else:
            fft_signal = fft_input[: FFT_WINDOW // 2]

        fft_signal = fft_signal * np.hanning(len(fft_signal))
        fft_result = np.fft.rfft(fft_signal)
        psd = np.abs(fft_result) ** 2 / len(fft_result)

        freqs = np.fft.rfftfreq(FFT_WINDOW // 2, 1.0 / SAMPLE_RATE)

        # Debug: Check actual audio values
        if self.debug_counter % 100 == 0:
            buffer_max = np.max(np.abs(audio_buffer))
            left_max = np.max(np.abs(full_left))
            right_max = np.max(np.abs(full_right))
            print(
                f"🔊 Buffer stats - Max: {buffer_max:.6f}, Left: {left_max:.6f}, Right: {right_max:.6f}"
            )
            print(
                f"📊 Buffer shape: {audio_buffer.shape}, Non-zero samples: {np.count_nonzero(audio_buffer)}"
            )

            # Debug PSD calculation
            psd_max = np.max(psd) if len(psd) > 0 else 0
            psd_mean = np.mean(psd) if len(psd) > 0 else 0
            print(
                f"📈 PSD stats - Max: {psd_max:.2e}, Mean: {psd_mean:.2e}, Freq points: {len(freqs)})"
            )

        # Debug: Check if we have actual audio data
        if self.debug_counter % 50 == 0:  # Print every 50 updates
            left_max = np.max(np.abs(full_left))
            right_max = np.max(np.abs(full_right))
            if left_max < 0.001 and right_max < 0.001:
                print(
                    f"⚠️  Warning: Low audio levels - Left: {left_max:.6f}, Right: {right_max:.6f}"
                )
                print(f"⚠️  Check if audio device '{DEVICE_NAME}' is receiving input")
            else:
                print(f"✅ Audio levels - Left: {left_max:.4f}, Right: {right_max:.4f}")

        # Update Waveform
        x = np.arange(len(full_left))
        self.waveform_left.setData(x, full_left * GAIN)
        self.waveform_right.setData(x, full_right * GAIN)

        # Calculate PSD
        fft_input = audio_buffer[-FFT_WINDOW:]
        if len(fft_input) < FFT_WINDOW:
            fft_input = np.pad(fft_input, (FFT_WINDOW - len(fft_input), 0))

        if CHANNELS == 2:
            fft_signal = fft_input[0::2][: FFT_WINDOW // 2]
        else:
            fft_signal = fft_input[: FFT_WINDOW // 2]

        fft_signal = fft_signal * np.hanning(len(fft_signal))
        fft_result = np.fft.rfft(fft_signal)
        psd = np.abs(fft_result) ** 2 / len(fft_result)

        freqs = np.fft.rfftfreq(FFT_WINDOW // 2, 1.0 / SAMPLE_RATE)

        # Update PSD plot
        if len(psd) == len(freqs):
            # Only plot frequencies > 20 Hz
            mask = freqs > 20
            if np.any(mask):  # Check if we have valid data to plot
                psd_data = psd[mask]
                freq_data = freqs[mask]
                # Filter out extreme outliers for better visualization
                psd_filtered = np.clip(
                    psd_data, np.percentile(psd_data, 1), np.percentile(psd_data, 99)
                )
                self.psd_curve.setData(freq_data, psd_filtered)
                # Force auto-range update
                self.psd_plot.enableAutoRange()
            else:
                # Clear the plot if no valid data
                self.psd_curve.setData([], [])

        # Calculate and display RMS
        rms_l = np.sqrt(np.mean(full_left**2))
        rms_r = np.sqrt(np.mean(full_right**2))
        db_l = 20 * np.log10(rms_l + 1e-6)
        db_r = 20 * np.log10(rms_r + 1e-6)
        self.rms_label.setText(
            f"RMS L: {rms_l:.3f} ({db_l:+.1f} dBFS)  "
            f"RMS R: {rms_r:.3f} ({db_r:+.1f} dBFS)"
        )

        # Update Spectrogram
        if CHANNELS == 2:
            spec_signal = audio_buffer[0::2][-self.spec_window_size :]
        else:
            spec_signal = audio_buffer[-self.spec_window_size :]

        if len(spec_signal) >= self.spec_window_size:
            try:
                # Apply window
                windowed = spec_signal * np.hanning(self.spec_window_size)

                # FFT
                fft_result = np.fft.rfft(windowed)
                magnitude_db = 20 * np.log10(np.abs(fft_result) + 1e-10)

                # Limit to 0-20kHz
                max_freq_idx = int(20000 * len(magnitude_db) / (SAMPLE_RATE / 2))
                magnitude_db = magnitude_db[:max_freq_idx]

                # Reshape to target bins
                if len(magnitude_db) > self.spec_freq_bins:
                    step = len(magnitude_db) // self.spec_freq_bins
                    magnitude_db = magnitude_db[::step][: self.spec_freq_bins]
                elif len(magnitude_db) < self.spec_freq_bins:
                    magnitude_db = np.pad(
                        magnitude_db, (0, self.spec_freq_bins - len(magnitude_db))
                    )

                # Add to history for right-to-left scrolling (newest at the end)
                self.spec_history.append(magnitude_db)
                if len(self.spec_history) > self.spec_time_bins:
                    self.spec_history.pop(0)

                # Update spectrogram image
                if len(self.spec_history) > 1:
                    # Create spectrogram data with time as horizontal axis (no transpose)
                    spec_data = np.array(self.spec_history)

                    # Flip vertically so low frequencies are at bottom (conventional orientation)
                    spec_data = np.flipud(spec_data)

                    # Normalize for display with improved contrast
                    vmin = np.percentile(
                        spec_data, 10
                    )  # Lower percentile for better visibility
                    vmax = np.percentile(spec_data, 95)

                    self.spec_img.setImage(
                        spec_data, autoLevels=False, levels=(vmin, vmax)
                    )

                    # Update extent - time flows left to right, frequencies bottom to top
                    max_time = (
                        len(self.spec_history) * self.spec_window_size / SAMPLE_RATE
                    )
                    self.spec_img.setRect(0, 0, max_time, 20000)

            except Exception as e:
                print(f"Spectrogram error: {e}")

    def closeEvent(self, event):
        # Cleanup
        if hasattr(self, "audio_thread"):
            self.audio_thread.stop()
            self.audio_thread.wait(2000)  # Wait up to 2 seconds
        print("✅ Audio stream closed")
        event.accept()


def list_audio_devices():
    """List available audio devices using FFmpeg"""
    try:
        print("🔍 Detecting audio devices...")
        # Try avfoundation (macOS)
        cmd = [FFMPEG_PATH, "-f", "avfoundation", "-list_devices", "true", "-i", ""]
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        _, stderr = process.communicate(timeout=5)

        devices_found = []
        for line in stderr.decode("utf-8").split("\n"):
            if "] [" in line and "]" in line.split("] [", 1)[1]:
                print(f"  {line.strip()}")
                # Extract device index
                after_first = line.split("] [", 1)[1]
                device_index = after_first.split("]")[0]
                device_name = (
                    after_first.split("] ", 1)[1] if "] " in after_first else "Unknown"
                )
                devices_found.append(device_index)
                print(f"    Device {device_index}: {device_name}")

        if devices_found:
            print(f"\n📋 Available devices: {devices_found}")
            if DEVICE_NAME not in devices_found:
                print(f"⚠️  Warning: Configured device '{DEVICE_NAME}' not found!")
                print(f"💡 Available options: {', '.join(devices_found)}")
                print(
                    f"🔧 Suggestion: Change DEVICE_NAME to '0' or '1' in the configuration"
                )
            return devices_found
        else:
            print("⚠️  No audio devices found")
            return []

    except subprocess.TimeoutExpired:
        print("⚠️  Device detection timed out")
        return []
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return []


def main():
    # Check FFmpeg availability
    try:
        subprocess.run([FFMPEG_PATH, "-version"], check=True, capture_output=True)
        print(f"✅ FFmpeg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            f"❌ FFmpeg not found. Please install FFmpeg and make sure '{FFMPEG_PATH}' is in PATH"
        )
        return

    # List audio devices
    devices = list_audio_devices()

    # Set pyqtgraph options
    pg.setConfigOptions(antialias=True, useOpenGL=True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark theme
    app.setStyleSheet("""
        QMainWindow {
            background-color: #000000;
        }
        QWidget {
            background-color: #000000;
        }
    """)

    window = AudioVisualizer()
    window.show()

    print(f"\n🎧 Audio Visualizer Started")
    print(
        f"📊 Listening to: {DEVICE_NAME} | Sample Rate: {SAMPLE_RATE} Hz | Channels: {CHANNELS}"
    )
    print(f"🔄 Update rate: 20 FPS (50ms intervals)")
    print(f"🚪 Close the window to exit")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
