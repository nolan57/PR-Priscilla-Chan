import sys
import os
import tempfile
import signal
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QStatusBar,
    QProgressBar,
    QSpacerItem,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal as Signal, QThread, QObject, QTimer, QRectF
from PyQt6.QtGui import QPainter, QColor, QCursor

import ffmpeg
import threading
import subprocess


class WaveformWidget(QWidget):
    region_selected = Signal(float, float)
    region_modified = Signal(int, float, float)  # region_index, start_time, end_time

    def __init__(self, parent=None):
        super().__init__(parent)
        self.samples = None
        self.sample_rate = None
        self.duration = 0.0
        self.regions = []
        self.playback_position = (
            -1.0
        )  # -1 means no playback, 0 to duration is current position
        self.drag_start_x = None
        self.drag_start_time = None
        self.current_drag_end_x = None
        self.current_drag_end_time = None
        self.setMinimumHeight(140)

        # Region interaction states
        self.selected_region_index = -1  # -1 means no region selected
        self.hovered_region_index = -1
        self.hovered_handle = None  # None, 'left', 'right', 'center'
        self.dragging_region = False
        self.resizing_region = False
        self.drag_offset_x = 0  # Offset from region edge when dragging
        self.handle_width = 8  # Width of resize handles in pixels
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # Accept keyboard focus

    def get_region_at_pos(self, x):
        """Get the region index at the given x position"""
        width = self.width()
        time = (x / width) * self.duration

        for i, (start_t, end_t) in enumerate(self.regions):
            if start_t <= time <= end_t:
                return i
        return -1

    def get_handle_at_pos(self, x, region_index):
        """Determine which handle (left, right, center) is at the given x position"""
        if region_index < 0 or region_index >= len(self.regions):
            return None

        width = self.width()
        start_t, end_t = self.regions[region_index]
        start_x = (start_t / self.duration) * width
        end_x = (end_t / self.duration) * width

        # Check left handle
        if abs(x - start_x) <= self.handle_width:
            return "left"
        # Check right handle
        if abs(x - end_x) <= self.handle_width:
            return "right"
        # Check center (dragging)
        if start_x <= x <= end_x:
            return "center"
        return None

    def update_mouse_cursor(self):
        """Update mouse cursor based on hover state"""
        if self.hovered_handle == "left" or self.hovered_handle == "right":
            self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
        elif self.hovered_handle == "center":
            self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def set_audio(self, samples, sample_rate, duration):
        self.samples = samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.regions.clear()
        self.selected_region_index = -1
        self.hovered_region_index = -1
        self.update()

    def paintEvent(self, event):
        if self.samples is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = self.width()
        height = self.height()
        center_y = height // 2

        painter.fillRect(event.rect(), Qt.GlobalColor.white)

        # Draw waveform
        # Downsample for performance
        num_points = min(len(self.samples), width * 2)
        step = max(1, len(self.samples) // num_points)
        indices = np.arange(0, len(self.samples), step)
        values = self.samples[indices]
        times = indices / self.sample_rate

        x_vals = (times / self.duration * width).astype(int)
        y_scale = height // 3 / (np.max(np.abs(values)) + 1e-6)
        y_vals = center_y - (values * y_scale).astype(int)

        painter.setPen(Qt.GlobalColor.blue)
        for i in range(len(x_vals) - 1):
            painter.drawLine(x_vals[i], y_vals[i], x_vals[i + 1], y_vals[i + 1])

        # Draw confirmed regions (permanent selections)
        for i, (start_t, end_t) in enumerate(self.regions):
            x1 = int(start_t / self.duration * width)
            x2 = int(end_t / self.duration * width)

            # Different appearance for selected vs unselected regions
            if i == self.selected_region_index:
                painter.setPen(QColor(0, 100, 255, 200))  # Blue for selected
                painter.setBrush(QColor(0, 100, 255, 100))
            else:
                painter.setPen(QColor(255, 0, 0, 150))  # Red for unselected
                painter.setBrush(QColor(255, 0, 0, 80))

            painter.drawRect(x1, 0, x2 - x1, height)

            # Draw resize handles for selected region
            if i == self.selected_region_index:
                # Left handle
                painter.setPen(QColor(0, 50, 200))
                painter.setBrush(QColor(0, 50, 200, 150))
                painter.drawRect(
                    x1 - self.handle_width // 2, height // 2 - 10, self.handle_width, 20
                )
                # Right handle
                painter.drawRect(
                    x2 - self.handle_width // 2, height // 2 - 10, self.handle_width, 20
                )

            # Add time labels for confirmed regions
            duration_label = f"{end_t - start_t:.2f}s"
            if i == self.selected_region_index:
                painter.setPen(QColor(0, 50, 200))
            else:
                painter.setPen(QColor(180, 0, 0))
            painter.drawText(x1 + 5, 20, duration_label)

        # Draw current drag selection (real-time preview)
        if self.drag_start_x is not None and self.current_drag_end_x is not None:
            painter.setPen(QColor(0, 150, 0, 200))  # Green for active selection
            painter.setBrush(QColor(0, 255, 0, 60))
            x1 = int(min(self.drag_start_x, self.current_drag_end_x))
            x2 = int(max(self.drag_start_x, self.current_drag_end_x))
            painter.drawRect(x1, 0, x2 - x1, height)
            # Add time label for current selection
            current_duration = abs(self.current_drag_end_time - self.drag_start_time)
            if current_duration > 0.01:
                duration_label = f"{current_duration:.2f}s"
                painter.setPen(QColor(0, 100, 0))
                painter.drawText(x1 + 5, 40, duration_label)
                painter.setPen(QColor(0, 150, 0, 200))  # Reset pen color

        # Draw playback position line
        if self.playback_position >= 0 and self.duration > 0:
            playback_x = int((self.playback_position / self.duration) * width)
            if 0 <= playback_x <= width:
                # Draw a red vertical line for playback position
                painter.setPen(QColor(255, 0, 0, 200))  # Red with some transparency
                painter.drawLine(playback_x, 0, playback_x, height)

                # Add time label above the line
                time_str = f"{self.playback_position:.2f}s"
                painter.setPen(QColor(200, 0, 0))
                painter.drawText(playback_x - 20, 15, time_str)

        # Time labels
        painter.setPen(Qt.GlobalColor.black)
        for t_sec in np.linspace(0, self.duration, min(10, int(self.duration) + 1)):
            x = int(t_sec / self.duration * width)
            painter.drawLine(x, center_y - 5, x, center_y + 5)
            painter.drawText(x - 15, height - 5, f"{t_sec:.1f}s")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()  # Ensure widget has focus for keyboard events
            x = event.position().x()
            region_index = self.get_region_at_pos(x)
            handle = self.get_handle_at_pos(x, region_index)

            if region_index >= 0 and handle:
                # Interaction with existing region
                self.selected_region_index = region_index
                self.drag_start_x = x
                self.drag_start_time = (x / self.width()) * self.duration

                if handle == "center":
                    # Start dragging region
                    self.dragging_region = True
                    self.resizing_region = False
                    start_t, end_t = self.regions[region_index]
                    self.drag_offset_x = x - ((start_t / self.duration) * self.width())
                elif handle in ["left", "right"]:
                    # Start resizing region
                    self.dragging_region = False
                    self.resizing_region = True
                    self.hovered_handle = handle
                else:
                    self.dragging_region = False
                    self.resizing_region = False

                self.update()
            else:
                # Start new selection (deselect current)
                self.selected_region_index = -1
                self.drag_start_x = x
                self.drag_start_time = (
                    self.drag_start_x / self.width()
                ) * self.duration
                self.current_drag_end_x = self.drag_start_x
                self.current_drag_end_time = self.drag_start_time
                self.dragging_region = False
                self.resizing_region = False
                self.update()
                # Emit signal to notify parent about drag start
                self.region_selected.emit(
                    -1, -1
                )  # Special signal indicating drag started

    def mouseMoveEvent(self, event):
        x = event.position().x()

        # Update hover state
        if not self.dragging_region and not self.resizing_region:
            region_index = self.get_region_at_pos(x)
            handle = self.get_handle_at_pos(x, region_index)

            if (
                region_index != self.hovered_region_index
                or handle != self.hovered_handle
            ):
                self.hovered_region_index = region_index
                self.hovered_handle = handle
                self.update_mouse_cursor()
                self.update()

        # Handle dragging/resizing
        if self.dragging_region and self.selected_region_index >= 0:
            # Move region
            new_x = x - self.drag_offset_x
            new_time = (new_x / self.width()) * self.duration

            start_t, end_t = self.regions[self.selected_region_index]
            duration = end_t - start_t

            # Keep region within bounds
            new_start = max(0, min(new_time, self.duration - duration))
            new_end = new_start + duration

            self.regions[self.selected_region_index] = (new_start, new_end)
            self.region_modified.emit(self.selected_region_index, new_start, new_end)
            self.update()

        elif self.resizing_region and self.selected_region_index >= 0:
            # Resize region
            new_time = (x / self.width()) * self.duration
            start_t, end_t = self.regions[self.selected_region_index]

            if self.hovered_handle == "left":
                new_start = min(new_time, end_t - 0.01)  # Ensure minimum size
                self.regions[self.selected_region_index] = (max(0, new_start), end_t)
                self.region_modified.emit(
                    self.selected_region_index, max(0, new_start), end_t
                )
            elif self.hovered_handle == "right":
                new_end = max(new_time, start_t + 0.01)  # Ensure minimum size
                self.regions[self.selected_region_index] = (
                    start_t,
                    min(new_end, self.duration),
                )
                self.region_modified.emit(
                    self.selected_region_index, start_t, min(new_end, self.duration)
                )

            self.update()

        elif (
            self.drag_start_x is not None
            and not self.dragging_region
            and not self.resizing_region
        ):
            # New selection drag
            self.current_drag_end_x = x
            self.current_drag_end_time = (x / self.width()) * self.duration
            self.update()  # Real-time visual feedback during drag

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.dragging_region or self.resizing_region:
                # Finish region modification
                self.dragging_region = False
                self.resizing_region = False
                self.hovered_handle = None
                self.update_mouse_cursor()
            elif self.drag_start_x is not None:
                # Finish new selection
                end_x = event.position().x()
                end_time = (end_x / self.width()) * self.duration
                start_t = min(self.drag_start_time, end_time)
                end_t = max(self.drag_start_time, end_time)
                region_added = False
                if end_t - start_t > 0.01:
                    self.regions.append((start_t, end_t))
                    self.selected_region_index = len(self.regions) - 1
                    self.region_selected.emit(start_t, end_t)
                    region_added = True
                # Clear drag state
                self.drag_start_x = None
                self.drag_start_time = None
                self.current_drag_end_x = None
                self.current_drag_end_time = None
                self.update()
                # Emit signal to notify parent about drag end
                if region_added:
                    self.region_selected.emit(
                        -3, -3
                    )  # Special signal indicating region added
                else:
                    self.region_selected.emit(
                        -2, -2
                    )  # Special signal indicating drag ended without selection

    def keyPressEvent(self, event):
        """Handle keyboard events"""
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            deleted_region = self.delete_selected_region()
            if deleted_region:
                start_t, end_t = deleted_region
                duration = end_t - start_t
                self.region_selected.emit(-4, -4)  # Signal region deletion
        else:
            super().keyPressEvent(event)

    def set_playback_position(self, position):
        """Set the current playback position in seconds"""
        self.playback_position = position
        self.update()

    def clear_regions(self):
        """Clear all permanent selections"""
        self.regions.clear()
        self.selected_region_index = -1
        self.update()

    def delete_selected_region(self):
        """Delete the currently selected region"""
        if self.selected_region_index >= 0 and self.selected_region_index < len(
            self.regions
        ):
            deleted_region = self.regions.pop(self.selected_region_index)
            self.selected_region_index = -1  # Deselect after deletion
            self.update()
            return deleted_region
        return None

    def cancel_current_drag(self):
        """Cancel current drag operation without adding selection"""
        # Store previous state for debugging
        had_drag = self.drag_start_x is not None

        # Clear all drag state variables
        self.drag_start_x = None
        self.drag_start_time = None
        self.current_drag_end_x = None
        self.current_drag_end_time = None

        # Force repaint to remove visual feedback
        self.update()

        # Return whether there was an active drag to cancel
        return had_drag


class FFmpegWorker(QObject):
    finished = Signal(str)  # output path
    error = Signal(str)

    def __init__(
        self, input_path, regions, output_path, muting_method="harmonic_residual"
    ):
        super().__init__()
        self.input_path = input_path
        self.regions = regions  # list of (start, end) in seconds
        self.output_path = output_path
        self.muting_method = muting_method  # New parameter
        self._is_running = False

    def create_muted_segment(
        self, input_file, start, end, total_duration,
        sample_rate=44100, channels=1
    ):
        """
        Improved Version: Create Optimized Silent Segments

        Parameters:
            muting_method: Silence processing method
                - 'original': Original fade in/out silence (default)
                - 'harmonic_residual': Harmonic residual method (recommended for singing) ⭐⭐
                - 'adaptive_ducking': Adaptive compression method (general recommendation) ⭐
                - 'noise_replacement': Environmental noise replacement
                - 'spectral_subtraction': Spectral subtraction method
                - 'pink_noise_blend': Pink noise blending
        """
        duration = end - start

        # Original method (backward compatible)
        if self.muting_method == 'original':
            fade = min(0.004, duration / 2)
            seg = ffmpeg.input(input_file, ss=start, t=duration)

            if fade >= 0.002:
                seg = ffmpeg.filter(seg, "afade", t="out", st=0, d=fade)
                mid_start = fade
                mid_end = duration - fade
                if mid_end > mid_start:
                    seg = ffmpeg.filter(
                        seg, "volume",
                        enable=f"between(t,{mid_start},{mid_end})",
                        volume=0
                    )
                seg = ffmpeg.filter(seg, "afade", t="in", st=duration - fade, d=fade)
            else:
                seg = ffmpeg.filter(seg, "volume", volume=0)

            return seg

        # ===== Method 1: Harmonic Residual (Best for Singing) =====
        elif self.muting_method == 'harmonic_residual':
            seg = ffmpeg.input(input_file, ss=start, t=duration)

            # Preserve high-frequency harmonics (>3kHz)
            seg_high = ffmpeg.filter(seg, 'highpass', f='3000', poles=2)

            # Preserve full spectrum at very low volume
            seg_full = ffmpeg.input(input_file, ss=start, t=duration)
            seg_full = ffmpeg.filter(seg_full, 'volume', volume='0.01')

            # Mix: 30% high frequency + 100% full spectrum at very low volume
            mixed = ffmpeg.filter(
                [seg_high, seg_full],
                'amix',
                inputs=2,
                weights='0.3 1'
            )

            # Overall volume control
            mixed = ffmpeg.filter(mixed, 'volume', volume='0.08')

            # Smooth transition
            fade = min(0.025, duration / 4)
            mixed = ffmpeg.filter(mixed, 'afade', t='in', st=0, d=fade)
            mixed = ffmpeg.filter(mixed, 'afade', t='out', st=duration-fade, d=fade)

            return mixed

        # ===== Method 2: Adaptive Compression (General Recommendation) =====
        elif self.muting_method == 'adaptive_ducking':
            seg = ffmpeg.input(input_file, ss=start, t=duration)

            # Dynamic compression to very low levels
            seg = ffmpeg.filter(
                seg,
                'compand',
                attacks='0.001',
                decays='0.1',
                points='-80/-80|-60/-40|-40/-40|-20/-40|0/-40',
                volume='0'
            )

            # Additional volume reduction
            seg = ffmpeg.filter(seg, 'volume', volume='0.03')

            # Smooth transition
            fade = min(0.02, duration / 4)
            seg = ffmpeg.filter(seg, 'afade', t='in', st=0, d=fade)
            seg = ffmpeg.filter(seg, 'afade', t='out', st=duration-fade, d=fade)

            return seg

        # ===== Method 3: Environmental Noise Replacement =====
        elif self.muting_method == 'noise_replacement':
            # Extract noise samples from audio beginning (assuming first 0.5 seconds is background noise)
            try:
                noise_sample = ffmpeg.input(input_file, ss=0, t=0.5)

                # Loop noise to match duration
                noise_looped = ffmpeg.filter(
                    noise_sample,
                    'aloop',
                    loop=-1,
                    size=int(sample_rate * duration)
                )

                # Trim to exact length
                noise_segment = ffmpeg.filter(noise_looped, 'atrim', duration=duration)

                # Fade in/out
                fade = min(0.02, duration / 4)
                noise_segment = ffmpeg.filter(noise_segment, 'afade', t='in', st=0, d=fade)
                noise_segment = ffmpeg.filter(
                    noise_segment,
                    'afade',
                    t='out',
                    st=duration-fade,
                    d=fade
                )

                return noise_segment
            except:
                # If noise extraction fails, fall back to adaptive compression
                self.muting_method = 'adaptive_ducking'
                return self.create_muted_segment(
                    input_file, start, end, total_duration,
                    sample_rate, channels
                )

        # ===== Method 4: Spectral Subtraction =====
        elif self.muting_method == 'spectral_subtraction':
            seg = ffmpeg.input(input_file, ss=start, t=duration)

            # Multi-band attenuation
            seg = ffmpeg.filter(seg, 'equalizer', f='100', width_type='h', width='50', g='-25')
            seg = ffmpeg.filter(seg, 'equalizer', f='1000', width_type='h', width='500', g='-30')
            seg = ffmpeg.filter(seg, 'equalizer', f='5000', width_type='h', width='2000', g='-25')

            # Reduce overall volume
            seg = ffmpeg.filter(seg, 'volume', volume='0.05')

            # Smooth transition
            fade = min(0.015, duration / 4)
            seg = ffmpeg.filter(seg, 'afade', t='in', st=0, d=fade)
            seg = ffmpeg.filter(seg, 'afade', t='out', st=duration-fade, d=fade)

            return seg

        # ===== Method 5: Pink Noise Blend =====
        elif self.muting_method == 'pink_noise_blend':
            seg = ffmpeg.input(input_file, ss=start, t=duration)

            # Reduce original audio to extremely low volume
            seg_quiet = ffmpeg.filter(seg, 'volume', volume='0.02')

            # Generate pink noise (simulated with white noise)
            try:
                noise = ffmpeg.input(
                    f'anoisesrc=duration={duration}:color=pink:sample_rate={sample_rate}',
                    f='lavfi'
                )
                noise = ffmpeg.filter(noise, 'volume', volume='0.05')

                # Mix
                mixed = ffmpeg.filter([seg_quiet, noise], 'amix', inputs=2, weights='1 0.5')

                # Smooth transition
                fade = min(0.02, duration / 4)
                mixed = ffmpeg.filter(mixed, 'afade', t='in', st=0, d=fade)
                mixed = ffmpeg.filter(mixed, 'afade', t='out', st=duration-fade, d=fade)

                return mixed
            except:
                # If noise generation fails, fall back to harmonic residual method
                self.muting_method = 'harmonic_residual'
                return self.create_muted_segment(
                    input_file, start, end, total_duration,
                    sample_rate, channels
                )

        else:
            # Unknown method, use original method
            self.muting_method = 'original'
            return self.create_muted_segment(
                input_file, start, end, total_duration,
                sample_rate, channels
            )

    def run(self):
        self._is_running = True
        try:
            if not self.regions:
                # No regions: just copy
                stream = ffmpeg.input(self.input_path)
                stream = ffmpeg.output(stream, self.output_path, acodec="pcm_s16le")
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
            else:
                # Get audio info for proper output settings
                probe = ffmpeg.probe(self.input_path)
                audio_stream = next(
                    (s for s in probe["streams"] if s["codec_type"] == "audio"), None
                )
                if not audio_stream:
                    raise ValueError("No audio stream found")

                total_dur = float(audio_stream["duration"])
                sr = int(audio_stream["sample_rate"])
                ch = int(audio_stream.get("channels", 1))

                # Sort regions for proper processing
                regions_sorted = sorted(self.regions, key=lambda x: x[0])

                segments = []
                current = 0.0

                # Process timeline: keep segments + muted segments
                for start, end in regions_sorted:
                    # Keep segment before region (unchanged)
                    if start > current:
                        keep = ffmpeg.input(
                            self.input_path, ss=current, t=start - current
                        )
                        segments.append(keep)

                    # Create muted segment with micro fades
                    muted = self.create_muted_segment(
                        self.input_path, start, end, total_dur, sr, ch
                    )
                    segments.append(muted)

                    current = end

                # Final keep segment after last region
                if current < total_dur:
                    keep = ffmpeg.input(
                        self.input_path, ss=current, t=total_dur - current
                    )
                    segments.append(keep)

                # Concatenate all segments
                if len(segments) == 1:
                    out_stream = segments[0]
                else:
                    out_stream = ffmpeg.concat(*segments, v=0, a=1)

                # Output with proper audio parameters
                out = ffmpeg.output(
                    out_stream, self.output_path, acodec="pcm_s16le", ar=sr, ac=ch
                )
                ffmpeg.run(out, overwrite_output=True, quiet=True)

            if self._is_running:  # Check if still running before emitting
                self.finished.emit(self.output_path)
        except Exception as e:
            if self._is_running:  # Check if still running before emitting
                self.error.emit(str(e))
        finally:
            self._is_running = False


class AudioMuterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Muter (FFmpeg Backend)")
        self.resize(1000, 600)
        self.input_file = None
        self.audio_duration = 0.0
        self.region_history = []  # Stack to track region addition order
        self.playback_process = None
        self.is_playing = False
        self.is_paused = False
        self.playback_start_time = 0.0
        self.pause_time = 0.0
        self.total_paused_time = 0.0
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)

        # Audio metadata
        self.audio_info = {
            "filename": "",
            "format": "",
            "duration": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "bitrate": 0,
            "codec": "",
            "file_size": 0,
        }

        self.init_ui()

    def update_button_states(self):
        """Update button enabled states based on current state"""
        has_audio = self.input_file is not None
        has_regions = False
        has_active_drag = False
        can_undo_selection = False

        # Check if waveform_view exists and is properly initialized
        if hasattr(self, "waveform_view") and self.waveform_view is not None:
            has_regions = len(self.waveform_view.regions) > 0
            has_active_drag = self.waveform_view.drag_start_x is not None
            can_undo_selection = has_regions or has_active_drag
        else:
            has_regions = False
            has_active_drag = False
            can_undo_selection = False

        # Update play button text and state
        if self.is_playing:
            if self.is_paused:
                self.play_btn.setText("Resume")
                self.play_btn.setEnabled(True)  # Enable resume functionality
                self.pause_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
            else:
                self.play_btn.setText("Playing...")
                self.play_btn.setEnabled(False)
                self.pause_btn.setEnabled(True)
                self.stop_btn.setEnabled(True)
        else:
            if hasattr(self, "waveform_view") and self.waveform_view is not None:
                has_regions = len(self.waveform_view.regions) > 0
                if has_regions:
                    self.play_btn.setText("Play (Mute Regions)")
                else:
                    self.play_btn.setText("Play Original")
            else:
                self.play_btn.setText("Play Original")
            self.play_btn.setEnabled(has_audio)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

        self.clear_selections_btn.setEnabled(has_regions)
        # Enable cancel button when there are regions or active drag
        self.cancel_drag_btn.setEnabled(can_undo_selection)
        self.save_btn.setEnabled(has_audio)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 3, 10, 3)  # Minimize top and bottom margins
        layout.setSpacing(0)  # Remove spacing between widgets

        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Audio")
        self.play_btn = QPushButton("Play (Mute Regions)")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.clear_selections_btn = QPushButton("Clear All Regions")
        self.cancel_drag_btn = QPushButton("Cancel Drag/Undo Last")
        self.save_btn = QPushButton("Apply Silence & Save")

        self.load_btn.clicked.connect(self.load_audio)
        self.play_btn.clicked.connect(self.play_audio)
        self.pause_btn.clicked.connect(self.pause_audio)
        self.stop_btn.clicked.connect(self.stop_audio)
        self.clear_selections_btn.clicked.connect(self.clear_all_selections)
        self.cancel_drag_btn.clicked.connect(self.cancel_current_selection)
        self.save_btn.clicked.connect(self.apply_silence)

        # Button states will be updated after waveform_view is created

        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.play_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.clear_selections_btn)
        btn_layout.addWidget(self.cancel_drag_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Time display section
        time_layout = QHBoxLayout()
        time_layout.setContentsMargins(0, 0, 0, 0)  # Remove all vertical margins
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setMinimumWidth(100)
        time_layout.addWidget(self.time_label)
        time_layout.addStretch()
        layout.addLayout(time_layout)

        # Remove vertical spacer completely
        # vertical_spacer = QSpacerItem(20, 2, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        # layout.addItem(vertical_spacer)

        self.waveform_view = WaveformWidget()
        self.waveform_view.region_selected.connect(self.on_region_selected)
        self.waveform_view.region_modified.connect(self.on_region_modified)
        layout.addWidget(self.waveform_view)
        layout.setStretchFactor(self.waveform_view, 1)  # Allow waveform to expand

        # Update button states after waveform_view is created
        # Add muting method selection
        self.muting_method = 'harmonic_residual'  # Default to harmonic residual method
        
        # Add method selection dropdown to button layout
        method_layout = QHBoxLayout()
        method_label = QLabel("Muting Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Harmonic Residual (Recommended-Singing) ⭐⭐",
            "Adaptive Compression (General) ⭐",
            "Environmental Noise Replacement",
            "Spectral Subtraction",
            "Pink Noise Blend",
            "Original Method (Complete Silence)"
        ])
        self.method_combo.setCurrentIndex(0)  # Default to harmonic residual
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)

        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()

        # Add method selection layout to main layout (after time display)
        layout.insertLayout(2, method_layout)  # Insert at appropriate position

        # Add method description label
        self.method_description = QLabel()
        self.method_description.setWordWrap(True)
        self.method_description.setStyleSheet("color: #555; font-size: 11px; padding: 5px;")
        self.update_method_description()
        layout.insertWidget(3, self.method_description)
        
        self.update_button_states()

    def on_method_changed(self, index):
        """Handle muting method change"""
        method_map = {
            0: 'harmonic_residual',
            1: 'adaptive_ducking',
            2: 'noise_replacement',
            3: 'spectral_subtraction',
            4: 'pink_noise_blend',
            5: 'original'
        }
        self.muting_method = method_map.get(index, 'harmonic_residual')
        self.update_method_description()

        # Update status bar
        method_names = {
            'harmonic_residual': 'Harmonic Residual Method',
            'adaptive_ducking': 'Adaptive Compression Method',
            'noise_replacement': 'Environmental Noise Replacement Method',
            'spectral_subtraction': 'Spectral Subtraction Method',
            'pink_noise_blend': 'Pink Noise Blend Method',
            'original': 'Original Silence Method'
        }
        self.statusBar().showMessage(
            f"Switched to: {method_names.get(self.muting_method, 'Unknown Method')}"
        )

    def update_method_description(self):
        """Update method description"""
        descriptions = {
            'harmonic_residual':
                "💡 Preserve high-frequency harmonics and harmonic structure, most suitable for singing training. "
                "Vocoder can learn timbre 'DNA' and vocal characteristics.",

            'adaptive_ducking':
                "💡 Preserve audio dynamic contour, adaptively compress volume to background level. "
                "Suitable for various audio types, most versatile choice.",

            'noise_replacement':
                "💡 Replace selections with background noise from recording environment. "
                "Helps vocoder learn recording environment characteristics, suitable for studio recordings.",

            'spectral_subtraction':
                "💡 Preserve spectral distribution but greatly reduce energy. "
                "Maintain 'sound is there' feeling, suitable for scenarios needing spectral reference.",

            'pink_noise_blend':
                "💡 Mix original sound with pink noise. "
                "Provide natural spectral reference, suitable for high-quality recordings.",

            'original':
                "💡 Traditional fade in/out complete silence method. "
                "Suitable for general purposes, but will lose timbre and environmental information."
        }

        desc = descriptions.get(self.muting_method, "")
        self.method_description.setText(desc)

    def update_status_bar_info(self):
        """Update the enhanced status bar with current audio information"""
        # Check if status bar labels have been created
        if not hasattr(self, "file_info_label"):
            return

        if not self.input_file:
            self.file_info_label.setText("No file loaded")
            self.audio_info_label.setText("")
            self.region_info_label.setText("")
            return

        # File information
        filename = Path(self.input_file).name
        self.file_info_label.setText(f"📁 {filename}")

        # Audio information
        duration_str = (
            f"{int(self.audio_duration // 60):02d}:{int(self.audio_duration % 60):02d}"
        )
        if self.audio_info["sample_rate"] > 0:
            sr_khz = self.audio_info["sample_rate"] / 1000
            audio_text = f"🎵 {duration_str} | {sr_khz:.1f}kHz"
            if self.audio_info["channels"] > 0:
                audio_text += f" | {self.audio_info['channels']}ch"
            if self.audio_info["bitrate"] > 0:
                bitrate_kb = self.audio_info["bitrate"] / 1000
                audio_text += f" | {bitrate_kb:.0f}kbps"
            if self.audio_info["codec"]:
                audio_text += f" | {self.audio_info['codec'].upper()}"
            self.audio_info_label.setText(audio_text)
        else:
            self.audio_info_label.setText("")

        # Region information
        if hasattr(self, "waveform_view") and self.waveform_view:
            num_regions = len(self.waveform_view.regions)
            if num_regions > 0:
                total_muted = sum(
                    end - start for start, end in self.waveform_view.regions
                )
                muted_percent = (total_muted / self.audio_duration) * 100
                self.region_info_label.setText(
                    f"🔇 {num_regions} regions | {total_muted:.1f}s ({muted_percent:.1f}%)"
                )
            else:
                self.region_info_label.setText("🔊 No muted regions")
        else:
            self.region_info_label.setText("")

    def format_file_size(self, size_bytes):
        """Format file size in human readable format"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def load_audio(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a)"
        )
        if not filepath:
            return

        self.input_file = filepath
        try:
            # Probe file info
            probe = ffmpeg.probe(filepath)
            audio_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "audio"), None
            )
            if not audio_stream:
                raise ValueError("No audio stream found")

            duration = float(audio_stream["duration"])
            sample_rate = int(audio_stream["sample_rate"])
            self.audio_duration = duration

            # Store detailed audio information
            from pathlib import Path
            import os

            self.audio_info = {
                "filename": Path(filepath).name,
                "format": Path(filepath).suffix.upper().lstrip("."),
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": int(audio_stream.get("channels", 0)),
                "bitrate": int(probe["format"].get("bit_rate", 0)),
                "codec": audio_stream.get("codec_name", ""),
                "file_size": os.path.getsize(filepath),
            }

            # Extract raw PCM for waveform (mono, 16-bit)
            out, _ = (
                ffmpeg.input(filepath)
                .output(
                    "pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate
                )
                .run(capture_stdout=True, quiet=True)
            )
            samples = np.frombuffer(out, dtype=np.int16).astype(np.float32)

            self.waveform_view.set_audio(samples, sample_rate, duration)
            self.update_time_display(0.0)

            # Update enhanced status bar
            self.update_status_bar_info()

            # Main status message
            self.statusBar().showMessage(
                f"Loaded: {Path(filepath).name} | Duration: {duration:.2f}s | Click and drag to select regions | Click regions to select/move | Use Delete key to remove"
            )
            self.update_button_states()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio:\n{str(e)}")

    def update_time_display(self, current_time):
        """Update time display and waveform position"""
        # Update waveform playback position
        if hasattr(self, "waveform_view") and self.waveform_view:
            self.waveform_view.set_playback_position(current_time)

        # Format time as MM:SS / MM:SS
        current_str = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
        total_str = (
            f"{int(self.audio_duration // 60):02d}:{int(self.audio_duration % 60):02d}"
        )
        self.time_label.setText(f"{current_str} / {total_str}")

    def update_progress(self):
        """Update progress during playback"""
        if self.is_playing and not self.is_paused:
            import time

            current_time = (
                time.time() - self.playback_start_time - self.total_paused_time
            )
            current_time = min(
                current_time, self.audio_duration
            )  # Don't exceed duration
            self.update_time_display(current_time)

            if current_time >= self.audio_duration:
                # Playback finished
                self.stop_audio()
                self.statusBar().showMessage("Playback completed")
            else:
                # Continue updating
                QTimer.singleShot(100, self.update_progress)

    def play_audio(self):
        if not self.input_file:
            QMessageBox.warning(self, "Warning", "No audio loaded.")
            return

        if self.is_paused:
            # Resume playback
            self.resume_audio()
            return

        # Stop any existing playback
        self.stop_audio()

        # Create a temporary file with silence applied to selected regions
        if self.waveform_view.regions:
            self.statusBar().showMessage("Preparing playback with mute regions...")
            try:
                self.play_with_mute_regions()
            except Exception as e:
                # Clean up any temporary files that might have been created
                if hasattr(self, "temp_playback_file") and self.temp_playback_file:
                    try:
                        import os

                        os.unlink(self.temp_playback_file)
                        self.temp_playback_file = None
                    except:
                        pass  # Ignore cleanup errors

                # Show error message to user
                import traceback

                QMessageBox.warning(
                    self,
                    "Playback Error",
                    f"Could not create muted audio:\n{str(e)}\n\nCheck console for detailed traceback.",
                )
                # Print traceback for debugging
                traceback.print_exc()
        else:
            self.statusBar().showMessage("Playing original audio...")
            self.play_original_audio()

        self.update_button_states()

    def create_muted_segment(
        self, input_file, start, end, total_duration,
        sample_rate=44100, channels=1
    ):
        """
        Improved Version: Create Optimized Silent Segments

        Parameters:
            muting_method: Silence processing method
                - 'original': Original fade in/out silence (default)
                - 'harmonic_residual': Harmonic residual method (recommended for singing) ⭐⭐
                - 'adaptive_ducking': Adaptive compression method (general recommendation) ⭐
                - 'noise_replacement': Environmental noise replacement
                - 'spectral_subtraction': Spectral subtraction method
                - 'pink_noise_blend': Pink noise blending
        """
        duration = end - start

        # Original method (backward compatible)
        if self.muting_method == 'original':
            fade = min(0.004, duration / 2)
            seg = ffmpeg.input(input_file, ss=start, t=duration)

            if fade >= 0.002:
                seg = ffmpeg.filter(seg, "afade", t="out", st=0, d=fade)
                mid_start = fade
                mid_end = duration - fade
                if mid_end > mid_start:
                    seg = ffmpeg.filter(
                        seg, "volume",
                        enable=f"between(t,{mid_start},{mid_end})",
                        volume=0
                    )
                seg = ffmpeg.filter(seg, "afade", t="in", st=duration - fade, d=fade)
            else:
                seg = ffmpeg.filter(seg, "volume", volume=0)

            return seg

        # ===== Method 1: Harmonic Residual (Best for Singing) =====
        elif self.muting_method == 'harmonic_residual':
            seg = ffmpeg.input(input_file, ss=start, t=duration)

            # Preserve high-frequency harmonics (>3kHz)
            seg_high = ffmpeg.filter(seg, 'highpass', f='3000', poles=2)

            # Preserve full spectrum at very low volume
            seg_full = ffmpeg.input(input_file, ss=start, t=duration)
            seg_full = ffmpeg.filter(seg_full, 'volume', volume='0.01')

            # Mix: 30% high frequency + 100% full spectrum at very low volume
            mixed = ffmpeg.filter(
                [seg_high, seg_full],
                'amix',
                inputs=2,
                weights='0.3 1'
            )

            # Overall volume control
            mixed = ffmpeg.filter(mixed, 'volume', volume='0.08')

            # Smooth transition
            fade = min(0.025, duration / 4)
            mixed = ffmpeg.filter(mixed, 'afade', t='in', st=0, d=fade)
            mixed = ffmpeg.filter(mixed, 'afade', t='out', st=duration-fade, d=fade)

            return mixed

        # ===== Method 2: Adaptive Compression (General Recommendation) =====
        elif self.muting_method == 'adaptive_ducking':
            seg = ffmpeg.input(input_file, ss=start, t=duration)

            # Dynamic compression to very low levels
            seg = ffmpeg.filter(
                seg,
                'compand',
                attacks='0.001',
                decays='0.1',
                points='-80/-80|-60/-40|-40/-40|-20/-40|0/-40',
                volume='0'
            )

            # Additional volume reduction
            seg = ffmpeg.filter(seg, 'volume', volume='0.03')

            # Smooth transition
            fade = min(0.02, duration / 4)
            seg = ffmpeg.filter(seg, 'afade', t='in', st=0, d=fade)
            seg = ffmpeg.filter(seg, 'afade', t='out', st=duration-fade, d=fade)

            return seg

        # ===== Method 3: Environmental Noise Replacement =====
        elif self.muting_method == 'noise_replacement':
            # Extract noise samples from audio beginning (assuming first 0.5 seconds is background noise)
            try:
                noise_sample = ffmpeg.input(input_file, ss=0, t=0.5)

                # Loop noise to match duration
                noise_looped = ffmpeg.filter(
                    noise_sample,
                    'aloop',
                    loop=-1,
                    size=int(sample_rate * duration)
                )

                # Trim to exact length
                noise_segment = ffmpeg.filter(noise_looped, 'atrim', duration=duration)

                # Fade in/out
                fade = min(0.02, duration / 4)
                noise_segment = ffmpeg.filter(noise_segment, 'afade', t='in', st=0, d=fade)
                noise_segment = ffmpeg.filter(
                    noise_segment,
                    'afade',
                    t='out',
                    st=duration-fade,
                    d=fade
                )

                return noise_segment
            except:
                # If noise extraction fails, fall back to adaptive compression
                self.muting_method = 'adaptive_ducking'
                return self.create_muted_segment(
                    input_file, start, end, total_duration,
                    sample_rate, channels
                )

        # ===== Method 4: Spectral Subtraction =====
        elif self.muting_method == 'spectral_subtraction':
            seg = ffmpeg.input(input_file, ss=start, t=duration)

            # Multi-band attenuation
            seg = ffmpeg.filter(seg, 'equalizer', f='100', width_type='h', width='50', g='-25')
            seg = ffmpeg.filter(seg, 'equalizer', f='1000', width_type='h', width='500', g='-30')
            seg = ffmpeg.filter(seg, 'equalizer', f='5000', width_type='h', width='2000', g='-25')

            # Reduce overall volume
            seg = ffmpeg.filter(seg, 'volume', volume='0.05')

            # Smooth transition
            fade = min(0.015, duration / 4)
            seg = ffmpeg.filter(seg, 'afade', t='in', st=0, d=fade)
            seg = ffmpeg.filter(seg, 'afade', t='out', st=duration-fade, d=fade)

            return seg

        # ===== Method 5: Pink Noise Blend =====
        elif self.muting_method == 'pink_noise_blend':
            seg = ffmpeg.input(input_file, ss=start, t=duration)

            # Reduce original audio to extremely low volume
            seg_quiet = ffmpeg.filter(seg, 'volume', volume='0.02')

            # Generate pink noise (simulated with white noise)
            try:
                noise = ffmpeg.input(
                    f'anoisesrc=duration={duration}:color=pink:sample_rate={sample_rate}',
                    f='lavfi'
                )
                noise = ffmpeg.filter(noise, 'volume', volume='0.05')

                # Mix
                mixed = ffmpeg.filter([seg_quiet, noise], 'amix', inputs=2, weights='1 0.5')

                # Smooth transition
                fade = min(0.02, duration / 4)
                mixed = ffmpeg.filter(mixed, 'afade', t='in', st=0, d=fade)
                mixed = ffmpeg.filter(mixed, 'afade', t='out', st=duration-fade, d=fade)

                return mixed
            except:
                # If noise generation fails, fall back to harmonic residual method
                self.muting_method = 'harmonic_residual'
                return self.create_muted_segment(
                    input_file, start, end, total_duration,
                    sample_rate, channels
                )

        else:
            # Unknown method, use original method
            self.muting_method = 'original'
            return self.create_muted_segment(
                input_file, start, end, total_duration,
                sample_rate, channels
            )

    def play_with_mute_regions(self):
        """Play audio with selected regions muted (with smooth fades) - keep timeline, just mute regions"""
        import tempfile
        import time
        import os

        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)
        self.temp_playback_file = temp_path

        try:
            regions = sorted(self.waveform_view.regions)
            if not regions:
                self.play_original_audio()
                return

            # Validate input file exists and is readable
            if not os.path.exists(self.input_file):
                raise FileNotFoundError(f"Input file not found: {self.input_file}")
            
            if not os.access(self.input_file, os.R_OK):
                raise PermissionError(f"Cannot read input file: {self.input_file}")

            total_dur = self.audio_duration
            # Get audio properties from stored info
            sr = self.audio_info.get("sample_rate", 44100)
            ch = self.audio_info.get("channels", 1)
            
            # Validate audio properties
            if sr <= 0 or ch <= 0:
                raise ValueError(f"Invalid audio properties: sample_rate={sr}, channels={ch}")

            segments = []
            current = 0.0

            for start, end in regions:
                # Keep segment before region (unchanged)
                if start > current:
                    keep = ffmpeg.input(self.input_file, ss=current, t=start - current)
                    segments.append(keep)

                # Process muted segment with micro fades
                muted = self.create_muted_segment(
                    self.input_file, start, end, total_dur, sr, ch
                )
                segments.append(muted)

                current = end

            # Final keep segment after last region
            if current < total_dur:
                keep = ffmpeg.input(self.input_file, ss=current, t=total_dur - current)
                segments.append(keep)

            # Concatenate all segments
            if len(segments) == 1:
                out_stream = segments[0]
            else:
                out_stream = ffmpeg.concat(*segments, v=0, a=1)

            # Output to temporary file with original audio parameters
            out = ffmpeg.output(out_stream, temp_path, acodec="pcm_s16le", ar=sr, ac=ch)
            
            # Run FFmpeg with better error handling
            try:
                ffmpeg.run(out, overwrite_output=True, quiet=False)
            except ffmpeg.Error as e:
                # Log the actual error for debugging
                error_msg = f"FFmpeg Error:\nSTDOUT: {e.stdout.decode() if e.stdout else 'None'}\nSTDERR: {e.stderr.decode() if e.stderr else 'None'}"
                print(error_msg)  # Print to console for debugging
                raise RuntimeError(f"Failed to process audio: {error_msg}") from e

            # Play the muted file
            self.playback_process = subprocess.Popen(
                ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", temp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Initialize progress tracking
            self.is_playing = True
            self.is_paused = False
            self.playback_start_time = time.time()
            self.pause_time = 0.0
            self.total_paused_time = 0.0

            # Update status message
            total_muted_time = sum(end - start for start, end in regions)
            self.statusBar().showMessage(
                f"Playing with {len(regions)} muted region(s) ({total_muted_time:.2f}s muted)"
            )

            # Start progress tracking
            QTimer.singleShot(100, self.update_progress)
            QTimer.singleShot(100, self.check_playback_status)

        except Exception as e:
            # Clean up
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
            self.temp_playback_file = None
            
            # Provide user-friendly error messages
            error_msg = str(e)
            if "FFmpeg" in error_msg or "ffmpeg" in error_msg.lower():
                user_msg = (
                    "Audio processing failed. This could be due to:\n\n"
                    "• FFmpeg not installed or misconfigured\n"
                    "• Unsupported audio file format\n"
                    "• Corrupted audio file\n"
                    "• Insufficient system permissions\n\n"
                    f"Technical details: {error_msg}"
                )
            elif isinstance(e, FileNotFoundError):
                user_msg = f"Audio file not found: {self.input_file}"
            elif isinstance(e, PermissionError):
                user_msg = f"Permission denied accessing: {self.input_file}"
            else:
                user_msg = f"An unexpected error occurred: {error_msg}"
            
            QMessageBox.critical(self, "Playback Error", user_msg)
            self.statusBar().showMessage("Playback failed")
            
            # Offer fallback option
            reply = QMessageBox.question(
                self,
                "Fallback Option",
                "Would you like to play the original audio file without muting?\n\n"
                "This will bypass FFmpeg processing and use your system's default player.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.play_original_audio()
            return

    def play_original_audio(self):
        """Play original audio without modifications"""
        import time

        try:
            self.playback_process = subprocess.Popen(
                [
                    "ffplay",
                    "-autoexit",
                    "-nodisp",
                    "-loglevel",
                    "quiet",
                    self.input_file,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Initialize progress tracking
            self.is_playing = True
            self.is_paused = False
            self.playback_start_time = time.time()
            self.pause_time = 0.0
            self.total_paused_time = 0.0

            # Start progress tracking
            QTimer.singleShot(100, self.update_progress)
            QTimer.singleShot(100, self.check_playback_status)

        except FileNotFoundError:
            # FFplay not found, fall back to system player
            try:
                if sys.platform == "win32":
                    os.startfile(self.input_file)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", self.input_file])
                else:
                    subprocess.Popen(["xdg-open", self.input_file])
                QMessageBox.information(
                    self,
                    "Playback",
                    "Using system player (ffplay not found). Install FFmpeg for better playback.",
                )
            except Exception as e:
                QMessageBox.warning(self, "Play Error", f"Could not play audio:\n{e}")
        except Exception as e:
            QMessageBox.warning(self, "Play Error", f"Unexpected error:\n{e}")

    def pause_audio(self):
        """Pause currently playing audio"""
        if self.playback_process and self.is_playing and not self.is_paused:
            try:
                # Try SIGTSTP first (Unix-like systems)
                if hasattr(signal, "SIGTSTP"):
                    self.playback_process.send_signal(signal.SIGTSTP)
                else:
                    # On Windows, try to pause by sending SIGINT (may not work perfectly)
                    self.playback_process.send_signal(signal.SIGINT)

                import time

                self.pause_time = time.time()
                self.is_paused = True
                self.statusBar().showMessage("Audio paused")
                self.update_button_states()
            except Exception as e:
                QMessageBox.warning(self, "Pause Error", f"Could not pause audio:\n{e}")

    def resume_audio(self):
        """Resume paused audio playback"""
        if self.playback_process and self.is_playing and self.is_paused:
            try:
                # Try SIGCONT first (Unix-like systems)
                if hasattr(signal, "SIGCONT"):
                    import time

                    self.total_paused_time += time.time() - self.pause_time
                    self.playback_process.send_signal(signal.SIGCONT)
                    self.is_paused = False
                    self.statusBar().showMessage("Resuming audio...")
                    self.update_button_states()
                    QTimer.singleShot(100, self.update_progress)
                    QTimer.singleShot(100, self.check_playback_status)
                else:
                    # On Windows, restart playback since resume may not work
                    self.statusBar().showMessage("Restarting playback (Windows)...")
                    self.stop_audio()
                    self.play_audio()
            except Exception as e:
                QMessageBox.warning(
                    self, "Resume Error", f"Could not resume audio:\n{e}"
                )

    def stop_audio(self):
        """Stop audio playback"""
        if self.playback_process:
            try:
                self.playback_process.terminate()
                self.playback_process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.playback_process.kill()
                self.playback_process.wait()
            except:
                pass
            finally:
                self.playback_process = None

        # Clean up temporary playback file if it exists
        if hasattr(self, "temp_playback_file"):
            try:
                import os

                os.unlink(self.temp_playback_file)
                self.temp_playback_file = None
            except:
                pass

        self.is_playing = False
        self.is_paused = False
        self.playback_start_time = 0.0
        self.pause_time = 0.0
        self.total_paused_time = 0.0
        if hasattr(self, "waveform_view") and self.waveform_view:
            self.waveform_view.set_playback_position(-1.0)  # Hide playback line
        self.update_time_display(0.0)
        self.statusBar().showMessage("Playback stopped")
        self.update_button_states()

    def check_playback_status(self):
        """Check if playback has finished"""
        if self.playback_process:
            return_code = self.playback_process.poll()
            if return_code is not None:
                # Playback has finished
                self.is_playing = False
                self.is_paused = False
                self.playback_process = None
                self.playback_start_time = 0.0
                self.pause_time = 0.0
                self.total_paused_time = 0.0
                if hasattr(self, "waveform_view") and self.waveform_view:
                    self.waveform_view.set_playback_position(
                        self.audio_duration
                    )  # Show at end
                self.update_time_display(self.audio_duration)  # Show full duration
                self.statusBar().showMessage("Playback completed")
                self.update_button_states()
            elif self.is_playing and not self.is_paused:
                # Still playing, check again in 100ms
                QTimer.singleShot(100, self.check_playback_status)

    def clear_all_selections(self):
        """Clear all permanent selections"""
        reply = QMessageBox.question(
            self,
            "Clear All Selections",
            "Are you sure you want to clear all selected regions?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.waveform_view.clear_regions()
            self.region_history.clear()
            self.statusBar().showMessage("All selections cleared")
            self.update_status_bar_info()  # Update region info
            self.update_button_states()

    def cancel_current_selection(self):
        """Cancel current drag selection in progress OR undo regions in reverse order"""
        # First try to cancel active drag
        had_active_drag = self.waveform_view.cancel_current_drag()
        if had_active_drag:
            self.statusBar().showMessage("Active drag operation cancelled")
        else:
            # If no active drag, undo regions from history (LIFO order)
            if self.region_history and self.waveform_view.regions:
                # Find and remove the last region from both lists
                last_region = self.region_history.pop()

                # Remove from waveform regions (find by value since order might differ)
                if last_region in self.waveform_view.regions:
                    self.waveform_view.regions.remove(last_region)
                    duration = last_region[1] - last_region[0]
                    remaining_count = len(self.waveform_view.regions)

                    if remaining_count > 0:
                        self.statusBar().showMessage(
                            f"Selection undone: {last_region[0]:.2f}s to {last_region[1]:.2f}s (duration: {duration:.2f}s) - {remaining_count} regions remaining"
                        )
                    else:
                        self.statusBar().showMessage(
                            f"Last selection undone: {last_region[0]:.2f}s to {last_region[1]:.2f}s (duration: {duration:.2f}s) - No regions remaining"
                        )
                else:
                    # Fallback: remove last region from waveform view
                    removed_region = self.waveform_view.regions.pop()
                    duration = removed_region[1] - removed_region[0]
                    remaining_count = len(self.waveform_view.regions)
                    self.statusBar().showMessage(
                        f"Selection undone: {removed_region[0]:.2f}s to {removed_region[1]:.2f}s (duration: {duration:.2f}s) - {remaining_count} regions remaining"
                    )

        self.update_status_bar_info()  # Update region info
        self.update_button_states()

    def apply_silence(self):
        if not self.input_file:
            QMessageBox.warning(self, "Warning", "No audio loaded.")
            return
        if not self.waveform_view.regions:
            reply = QMessageBox.question(
                self,
                "No Regions",
                "No silence regions selected. Save original?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processed Audio",
            "",
            "WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*)",
        )
        if not save_path:
            return

        # Ensure extension
        if not any(
            save_path.endswith(ext) for ext in [".wav", ".mp3", ".flac", ".ogg"]
        ):
            save_path += ".wav"

        self.statusBar().showMessage("Processing with FFmpeg...")
        self.save_btn.setEnabled(False)

        self.worker = FFmpegWorker(
            self.input_file, self.waveform_view.regions, save_path, muting_method=self.muting_method
        )
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.worker.finished.connect(self.on_success)
        self.worker.error.connect(self.on_error)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def cleanup_worker(self):
        """Clean up worker and thread resources"""
        if hasattr(self, "worker") and self.worker:
            try:
                self.worker.deleteLater()
                self.worker = None
            except:
                pass
        if (
            hasattr(self, "thread")
            and self.thread
            and hasattr(self.thread, "isRunning")
            and callable(self.thread.isRunning)
        ):
            try:
                if self.thread.isRunning():
                    self.thread.quit()
                    self.thread.wait(1000)
                self.thread.deleteLater()
                self.thread = None
            except:
                pass
        # Re-enable UI
        self.update_button_states()

    def on_success(self, output_path):
        self.statusBar().showMessage(f"Saved: {output_path}")
        QMessageBox.information(
            self,
            "Success",
            f"Audio saved successfully to:\n{output_path}\n\nLoading modified file...",
        )
        self.cleanup_worker()

        # Automatically load the modified audio file
        self.input_file = output_path

        try:
            # Probe file info
            probe = ffmpeg.probe(output_path)
            audio_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "audio"), None
            )
            if not audio_stream:
                raise ValueError("No audio stream found")

            duration = float(audio_stream["duration"])
            sample_rate = int(audio_stream["sample_rate"])
            self.audio_duration = duration

            # Extract raw PCM for waveform (mono, 16-bit)
            out, _ = (
                ffmpeg.input(output_path)
                .output(
                    "pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate
                )
                .run(capture_stdout=True, quiet=True)
            )
            samples = np.frombuffer(out, dtype=np.int16).astype(np.float32)

            # Clear all selections before loading new waveform
            self.waveform_view.clear_regions()
            self.region_history.clear()

            # Load the new waveform
            self.waveform_view.set_audio(samples, sample_rate, duration)
            self.update_time_display(0.0)

            # Update audio info for the modified file
            self.audio_info["filename"] = Path(output_path).name
            self.audio_info["duration"] = duration
            self.audio_info["file_size"] = os.path.getsize(output_path)

            # Update status bar with new file info
            self.update_status_bar_info()

            self.statusBar().showMessage(
                f"Loaded modified file: {Path(output_path).name} | Duration: {duration:.2f}s"
            )
            self.update_button_states()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Modified File",
                f"Failed to load modified audio:\n{str(e)}",
            )

    def on_error(self, error_msg):
        self.statusBar().showMessage("Processing failed")
        QMessageBox.critical(
            self,
            "FFmpeg Error",
            f"Failed to process audio:\n{error_msg}\n\nPlease check:\n1. FFmpeg is properly installed\n2. Input file is not corrupted\n3. Output path is writable",
        )
        self.cleanup_worker()

    def closeEvent(self, event):
        """Handle application closing"""
        # Stop any ongoing playback
        self.stop_audio()

        # Cancel any ongoing processing
        if hasattr(self, "worker") and self.worker:
            self.worker._is_running = False

        # Clean up worker and thread safely
        self.cleanup_worker()

        # Additional cleanup for any remaining threads
        if hasattr(self, "thread") and self.thread:
            try:
                # Check if thread is still valid before accessing it
                if self.thread and hasattr(self.thread, "isRunning"):
                    if self.thread.isRunning():
                        self.thread.quit()
                        self.thread.wait(2000)  # Wait up to 2 seconds
                        if self.thread.isRunning():
                            self.thread.terminate()
                            self.thread.wait(1000)
            except RuntimeError:
                # Thread already deleted, ignore the error
                pass
            except Exception:
                # Handle any other exceptions during cleanup
                pass
            finally:
                # Ensure thread reference is cleared
                self.thread = None

        event.accept()

    def on_region_modified(self, region_index, start_time, end_time):
        """Handle region modification events from waveform widget"""
        duration = end_time - start_time
        self.statusBar().showMessage(
            f"Region {region_index + 1} modified: {start_time:.2f}s to {end_time:.2f}s (duration: {duration:.2f}s)"
        )
        self.update_status_bar_info()  # Update region info
        self.update_button_states()

    def on_region_selected(self, start_time, end_time):
        """Handle region selection events from waveform widget"""
        # Only show status messages if status bar exists
        # Special case: drag started (-1, -1)
        if start_time == -1 and end_time == -1:
            self.statusBar().showMessage("Dragging... Release to select region")
        # Special case: drag ended without selection (-2, -2)
        elif start_time == -2 and end_time == -2:
            self.statusBar().showMessage("Drag completed without selection")
        # Special case: region added (-3, -3)
        elif start_time == -3 and end_time == -3:
            # Add the newly added region to history
            if self.waveform_view.regions:
                new_region = self.waveform_view.regions[-1]
                self.region_history.append(new_region)
                duration = new_region[1] - new_region[0]
                remaining_count = len(self.waveform_view.regions)
                self.statusBar().showMessage(
                    f"Region added: {new_region[0]:.2f}s to {new_region[1]:.2f}s (duration: {duration:.2f}s) - {remaining_count} regions selected"
                )
        # Special case: region deleted (-4, -4)
        elif start_time == -4 and end_time == -4:
            remaining_count = len(self.waveform_view.regions)
            if remaining_count > 0:
                self.statusBar().showMessage(
                    f"Region deleted - {remaining_count} regions remaining"
                )
            else:
                self.statusBar().showMessage("Region deleted - No regions remaining")
        # Normal region selection
        elif start_time >= 0 and end_time >= 0:
            duration = end_time - start_time
            self.statusBar().showMessage(
                f"Region selected: {start_time:.2f}s to {end_time:.2f}s (duration: {duration:.2f}s)"
            )

        # Update status bar info and button states after any selection event
        self.update_status_bar_info()
        self.update_button_states()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioMuterApp()
    window.show()
    sys.exit(app.exec())
