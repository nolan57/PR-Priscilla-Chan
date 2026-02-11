"""
PyAnnote Singer Separator - 基于目标说话人提取的歌手分离应用

功能：从预分离的干声音频中提取特定目标歌手的声音
技术：PyAnnote.Audio TargetSpeakerExtraction + PyQt6 GUI
"""

import sys
import os
import warnings
import tempfile
import subprocess
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple, List
import torch
import torchaudio

# --- 设备检测 ---
def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"

DEVICE = get_best_device()
print(f"Using device: {DEVICE}")

# --- 离线模式配置 ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 抑制警告
warnings.filterwarnings("ignore", message=".*Unknown device for graph fuser.*")
warnings.filterwarnings("ignore", message=".*mps.*fallback.*")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QGroupBox, QSlider, QProgressBar, QScrollArea,
    QStatusBar, QLineEdit, QComboBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect, QObject, QSize
from PyQt6.QtGui import QPainter, QColor, QImage, QCursor, QLinearGradient, QPen, QBrush

# 音频参数
SAMPLE_RATE = 16000
HOP_LENGTH = 512
N_FFT = 1024


class PyAnnoteSeparator:
    """
    PyAnnote.Audio 目标说话人分离核心类
    支持参考音频嵌入提取和混合音频分离
    """
    
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.embedding_model = None
        self.separation_model = None
        self.target_embedding = None
        self.is_initialized = False
        
    def load_models(self, cache_dir: Optional[str] = None) -> bool:
        """
        加载 PyAnnote 模型
        
        Args:
            cache_dir: 模型缓存目录
            
        Returns:
            bool: 是否成功加载
        """
        try:
            from pyannote.audio import Model
            from pyannote.audio.pipelines import SpeakerDiarization
            
            # 设置缓存目录
            if cache_dir:
                os.environ["PYANNOTE_CACHE"] = cache_dir
            
            # 加载声纹嵌入模型 (ECAPA-TDNN)
            print("Loading speaker embedding model...")
            self.embedding_model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=False,
                cache_dir=cache_dir
            )
            self.embedding_model.to(self.device)
            
            # 加载分离模型
            print("Loading separation model...")
            self.separation_model = Model.from_pretrained(
                "pyannote/segmentation-3.0",
                use_auth_token=False,
                cache_dir=cache_dir
            )
            self.separation_model.to(self.device)
            
            self.is_initialized = True
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        从参考音频提取声纹嵌入
        
        Args:
            audio_path: 参考音频路径
            
        Returns:
            声纹嵌入向量 (256维)
        """
        if not self.is_initialized:
            return None
            
        try:
            # 加载音频
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # 重采样到 16kHz
            if sample_rate != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 提取嵌入
            waveform = waveform.to(self.device)
            with torch.no_grad():
                embedding = self.embedding_model(waveform)
            
            # 归一化
            embedding = embedding.squeeze().cpu().numpy()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            self.target_embedding = embedding
            return embedding
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def separate_target(
        self, 
        audio_path: str, 
        threshold: float = 0.5,
        progress_callback=None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        从混合音频中分离目标说话人
        
        Args:
            audio_path: 混合音频路径
            threshold: 分离阈值 (0-1)
            progress_callback: 进度回调函数
            
        Returns:
            (分离后的音频, 相似度热力图)
        """
        if not self.is_initialized or self.target_embedding is None:
            return None, None
        
        try:
            # 加载混合音频
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if progress_callback:
                progress_callback("Resampling audio...")
            
            # 重采样
            if sample_rate != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
                waveform = resampler(waveform)
            
            # 单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            waveform = waveform.to(self.device)
            
            if progress_callback:
                progress_callback("Computing spectrogram...")
            
            # 计算 STFT
            stft = torch.stft(
                waveform.squeeze(),
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                return_complex=True
            )
            
            # 提取帧级嵌入
            if progress_callback:
                progress_callback("Extracting frame embeddings...")
            
            num_frames = stft.shape[-1]
            frame_embeddings = []
            
            # 分块处理避免内存溢出
            chunk_size = 100
            for i in range(0, num_frames, chunk_size):
                end_idx = min(i + chunk_size, num_frames)
                chunk_stft = stft[:, :, i:end_idx]
                
                # 转回时域提取嵌入
                chunk_waveform = torch.istft(
                    chunk_stft,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    length=(end_idx - i) * HOP_LENGTH
                )
                
                with torch.no_grad():
                    chunk_emb = self.embedding_model(chunk_waveform.unsqueeze(0))
                    frame_embeddings.append(chunk_emb.squeeze().cpu().numpy())
                
                if progress_callback and i % (chunk_size * 5) == 0:
                    progress = int((i / num_frames) * 50)
                    progress_callback(f"Processing frames... {progress}%")
            
            # 计算相似度
            if progress_callback:
                progress_callback("Computing similarities...")
            
            frame_embeddings = np.array(frame_embeddings)
            similarities = np.dot(frame_embeddings, self.target_embedding)
            similarities = np.clip(similarities, -1, 1)
            
            # 插值到完整帧数
            similarities_full = np.interp(
                np.linspace(0, len(similarities), num_frames),
                np.arange(len(similarities)),
                similarities
            )
            
            # 生成软 mask
            if progress_callback:
                progress_callback("Generating separation mask...")
            
            # 使用 sigmoid 平滑过渡
            mask_scores = 1 / (1 + np.exp(-10 * (similarities_full - threshold)))
            
            # 应用 mask 到频谱
            mask = torch.tensor(mask_scores, device=self.device).unsqueeze(0).unsqueeze(0)
            mask = mask.expand_as(stft)
            
            masked_stft = stft * mask
            
            # 重建波形
            if progress_callback:
                progress_callback("Reconstructing waveform...")
            
            separated_waveform = torch.istft(
                masked_stft,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                length=waveform.shape[-1]
            )
            
            # 转回 numpy
            separated_audio = separated_waveform.cpu().numpy()
            
            # 创建热力图数据 (时间 x 频率 x 相似度)
            heatmap = np.tile(similarities_full, (stft.shape[0], 1)).T
            
            return separated_audio, heatmap
            
        except Exception as e:
            print(f"Error in separation: {e}")
            import traceback
            traceback.print_exc()
            return None, None


class SeparationWorker(QThread):
    """分离处理后台线程"""
    finished = pyqtSignal(object, object)  # 分离结果, 热力图
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, separator: PyAnnoteSeparator, audio_path: str, threshold: float):
        super().__init__()
        self.separator = separator
        self.audio_path = audio_path
        self.threshold = threshold
        self._is_cancelled = False
    
    def cancel(self):
        self._is_cancelled = True
    
    def run(self):
        try:
            def progress_cb(msg):
                if not self._is_cancelled:
                    self.progress.emit(msg)
            
            separated, heatmap = self.separator.separate_target(
                self.audio_path,
                self.threshold,
                progress_cb
            )
            
            if not self._is_cancelled:
                if separated is not None:
                    self.finished.emit(separated, heatmap)
                else:
                    self.error.emit("Separation failed. Check model and audio.")
        except Exception as e:
            if not self._is_cancelled:
                self.error.emit(str(e))


class EmbeddingWorker(QThread):
    """参考音频嵌入提取后台线程"""
    finished = pyqtSignal(bool, str, object)
    progress = pyqtSignal(str)
    
    def __init__(self, separator: PyAnnoteSeparator, audio_path: str):
        super().__init__()
        self.separator = separator
        self.audio_path = audio_path
    
    def run(self):
        try:
            self.progress.emit("Extracting reference embedding...")
            embedding = self.separator.extract_embedding(self.audio_path)
            
            if embedding is not None:
                self.finished.emit(True, "Reference loaded successfully!", embedding)
            else:
                self.finished.emit(False, "Failed to extract embedding.", None)
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}", None)


class ModelLoadWorker(QThread):
    """模型加载后台线程"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)
    
    def __init__(self, separator: PyAnnoteSeparator, cache_dir: str):
        super().__init__()
        self.separator = separator
        self.cache_dir = cache_dir
    
    def run(self):
        try:
            self.progress.emit("Loading PyAnnote models...")
            success = self.separator.load_models(self.cache_dir)
            
            if success:
                self.finished.emit(True, "Models loaded successfully!")
            else:
                self.finished.emit(False, "Failed to load models. Check cache directory.")
        except Exception as e:
            self.finished.emit(False, f"Error loading models: {str(e)}")


class WaveformWidget(QWidget):
    """
    增强版波形显示组件
    支持：波形显示、频谱图、相似度热力图叠加
    """
    region_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.samples = None
        self.duration = 0.0
        self.regions = []  # 保留区域列表
        self.heatmap = None  # 相似度热力图
        self.spectrogram = None
        
        self.playback_position = -1.0
        self.zoom_factor = 1.0
        self.contrast = 1.0
        
        # 交互状态
        self.drag_start_pos = None
        self.drag_current_x = None
        self.interaction_mode = None
        self.active_idx = -1
        self.resize_edge = None
        self.edge_threshold = 10
        
        self.setMouseTracking(True)
        self.setMinimumHeight(400)
        
    def set_audio(self, samples: np.ndarray, duration: float):
        """设置音频数据"""
        self.samples = samples
        self.duration = duration
        self.regions.clear()
        self.generate_spectrogram()
        self.update()
    
    def set_heatmap(self, heatmap: np.ndarray):
        """设置相似度热力图"""
        self.heatmap = heatmap
        self.update()
    
    def generate_spectrogram(self):
        """生成频谱图"""
        if self.samples is None:
            return
        
        # 简化的频谱图生成
        n_fft, hop_length = 1024, 512
        data = self.samples[:2000000] if len(self.samples) > 2000000 else self.samples
        
        # 计算 STFT
        spec = []
        for i in range(0, len(data) - n_fft, hop_length):
            frame = data[i:i + n_fft] * np.hanning(n_fft)
            fft = np.abs(np.fft.rfft(frame))
            spec.append(fft)
        
        spec = np.array(spec).T
        spec = np.log1p(spec * self.contrast)
        
        if spec.size == 0:
            return
        
        # 归一化
        norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        
        # 创建彩色图像
        img_data = np.zeros((norm.shape[0], norm.shape[1], 4), dtype=np.uint8)
        img_data[..., 0] = (norm * 255).astype(np.uint8)  # R
        img_data[..., 1] = (np.sin(norm * np.pi) * 255).astype(np.uint8)  # G
        img_data[..., 2] = ((1 - norm) * 150).astype(np.uint8)  # B
        img_data[..., 3] = 255  # Alpha
        
        self.spectrogram = QImage(
            np.ascontiguousarray(img_data).tobytes(),
            norm.shape[1],
            norm.shape[0],
            QImage.Format.Format_RGBA8888
        )
    
    def get_view_width(self):
        return int(self.width() * self.zoom_factor)
    
    def t_to_x(self, t: float) -> int:
        if self.duration > 0:
            return int((t / self.duration) * self.get_view_width())
        return 0
    
    def x_to_t(self, x: int) -> float:
        if self.get_view_width() > 0:
            return (x / self.get_view_width()) * self.duration
        return 0.0
    
    def _detect_hovered_edge(self, x: int):
        """检测鼠标是否悬停在区域边缘"""
        for i, (s, e) in enumerate(self.regions):
            xs, xe = self.t_to_x(s), self.t_to_x(e)
            if abs(x - xs) <= self.edge_threshold:
                return 'left', i
            if abs(x - xe) <= self.edge_threshold:
                return 'right', i
        return None, -1
    
    def mousePressEvent(self, event):
        if self.samples is None:
            return
        
        x = int(event.position().x())
        self.drag_start_pos = x
        
        # 检测边缘拖拽
        edge, idx = self._detect_hovered_edge(x)
        if edge:
            self.interaction_mode = 'resize'
            self.active_idx = idx
            self.resize_edge = edge
            self.drag_start_region = list(self.regions[idx])
            self.drag_start_time = self.x_to_t(x)
            return
        
        # 检测区域移动
        for i, (s, e) in enumerate(self.regions):
            xs, xe = self.t_to_x(s), self.t_to_x(e)
            if xs < x < xe:
                self.interaction_mode = 'move'
                self.active_idx = i
                self.drag_start_region = list(self.regions[i])
                self.update()
                return
        
        # 新建区域
        self.interaction_mode = 'draw'
        self.active_idx = -1
        self.drag_start_time = self.x_to_t(x)
        self.drag_current_x = x
        self.update()
    
    def mouseMoveEvent(self, event):
        if self.samples is None:
            return
        
        x = int(event.position().x())
        
        if self.drag_start_pos is None:
            # 悬停状态检测
            edge, _ = self._detect_hovered_edge(x)
            if edge:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif any(self.t_to_x(s) < x < self.t_to_x(e) for s, e in self.regions):
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
            return
        
        # 拖拽处理
        if self.interaction_mode == 'draw':
            self.drag_current_x = x
        elif self.interaction_mode == 'move':
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            dt = self.x_to_t(x) - self.x_to_t(self.drag_start_pos)
            s, e = self.drag_start_region
            new_s = max(0.0, s + dt)
            new_e = min(self.duration, e + dt)
            if new_e - new_s >= 0.05:
                self.regions[self.active_idx] = (new_s, new_e)
            self.region_changed.emit()
        elif self.interaction_mode == 'resize':
            s, e = self.drag_start_region
            if self.resize_edge == 'left':
                new_s = max(0.0, min(self.x_to_t(x), e - 0.05))
                self.regions[self.active_idx] = (new_s, e)
            elif self.resize_edge == 'right':
                new_e = min(self.duration, max(self.x_to_t(x), s + 0.05))
                self.regions[self.active_idx] = (s, new_e)
            self.region_changed.emit()
        
        self.update()
    
    def mouseReleaseEvent(self, event):
        if self.interaction_mode == 'draw' and self.drag_start_pos is not None:
            s, e = sorted([self.drag_start_time, self.x_to_t(int(event.position().x()))])
            if e - s > 0.05:
                self.regions.append((s, e))
                self.region_changed.emit()
        
        self.drag_start_pos = None
        self.interaction_mode = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()
    
    def paintEvent(self, event):
        if self.samples is None:
            return
        
        painter = QPainter(self)
        vw = self.get_view_width()
        h = self.height()
        mid_h = h // 2
        
        # 黑色背景
        painter.fillRect(event.rect(), Qt.GlobalColor.black)
        
        # 绘制频谱图 (上半部分)
        if self.spectrogram:
            # 如果有热力图，叠加显示
            if self.heatmap is not None:
                self._draw_heatmap_overlay(painter, vw, mid_h)
            else:
                painter.drawImage(QRect(0, 0, vw, mid_h), self.spectrogram)
        
        # 绘制波形 (下半部分)
        painter.setPen(QColor(0, 255, 255))
        step = max(1, len(self.samples) // (vw * 2))
        vals = self.samples[::step]
        y_scale = (mid_h // 1.5) / (np.max(np.abs(vals)) + 1e-6)
        
        for i in range(len(vals) - 1):
            x1 = int((i * step / len(self.samples)) * vw)
            x2 = int(((i + 1) * step / len(self.samples)) * vw)
            y1 = int(h * 0.75 - vals[i] * y_scale)
            y2 = int(h * 0.75 - vals[i + 1] * y_scale)
            painter.drawLine(x1, y1, x2, y2)
        
        # 绘制保留区域
        for i, (s, e) in enumerate(self.regions):
            xs, xe = self.t_to_x(s), self.t_to_x(e)
            
            # 区域填充
            painter.setPen(QColor(46, 204, 113, 220))  # 绿色边框
            painter.setBrush(QColor(46, 204, 113, 80))  # 半透明绿色
            painter.drawRect(xs, 0, xe - xs, h)
            
            # 边缘手柄
            handle_color = QColor(255, 255, 0, 200) if i == self.active_idx else QColor(255, 200, 0, 180)
            painter.setPen(handle_color)
            painter.setBrush(handle_color)
            painter.drawRect(int(xs - 3), 0, 6, h)
            painter.drawRect(int(xe - 3), 0, 6, h)
        
        # 绘制拖拽中的新区域
        if self.drag_start_pos and self.drag_current_x and self.interaction_mode == 'draw':
            xs, xe = sorted([int(self.drag_start_pos), int(self.drag_current_x)])
            painter.setPen(QColor(255, 165, 0, 220))  # 橙色
            painter.setBrush(QColor(255, 165, 0, 80))
            painter.drawRect(xs, 0, xe - xs, h)
        
        # 播放位置指示线
        if self.playback_position >= 0:
            px = self.t_to_x(self.playback_position)
            painter.setPen(QColor(255, 0, 0))
            painter.drawLine(px, 0, px, h)
        
        painter.end()
    
    def _draw_heatmap_overlay(self, painter: QPainter, width: int, height: int):
        """绘制相似度热力图叠加"""
        if self.heatmap is None:
            return
        
        # 创建热力图图像
        heatmap_data = self.heatmap[:height, :width] if self.heatmap.shape[0] > height else self.heatmap
        
        # 归一化到 0-1
        h_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-6)
        
        # 创建彩色热力图 (绿->黄->红)
        img_data = np.zeros((heatmap_data.shape[0], heatmap_data.shape[1], 4), dtype=np.uint8)
        
        # 红色表示低相似度，绿色表示高相似度
        img_data[..., 0] = ((1 - h_norm) * 255).astype(np.uint8)  # R: 低相似=高红
        img_data[..., 1] = (h_norm * 255).astype(np.uint8)  # G: 高相似=高绿
        img_data[..., 2] = 50  # B: 固定蓝色
        img_data[..., 3] = 180  # Alpha: 半透明
        
        heatmap_img = QImage(
            np.ascontiguousarray(img_data).tobytes(),
            heatmap_data.shape[1],
            heatmap_data.shape[0],
            QImage.Format.Format_RGBA8888
        )
        
        painter.drawImage(QRect(0, 0, width, height), heatmap_img)


class PyAnnoteSingerSeparatorApp(QMainWindow):
    """PyAnnote 歌手分离应用主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyAnnote Singer Separator")
        self.resize(1400, 1000)
        
        # 状态变量
        self.input_file: Optional[str] = None
        self.reference_file: Optional[str] = None
        self.waveform: Optional[np.ndarray] = None
        self.separated_waveform: Optional[np.ndarray] = None
        self.heatmap: Optional[np.ndarray] = None
        self.sr = SAMPLE_RATE
        
        # 模型和线程
        self.separator = PyAnnoteSeparator(DEVICE)
        self.model_worker: Optional[ModelLoadWorker] = None
        self.embedding_worker: Optional[EmbeddingWorker] = None
        self.separation_worker: Optional[SeparationWorker] = None
        self.playback_process = None
        
        # 定时器
        self.cursor_timer = QTimer()
        self.cursor_timer.timeout.connect(self.update_cursor)
        
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        
        # === 模型配置区域 ===
        model_group = QGroupBox("Model Configuration")
        model_layout = QHBoxLayout()
        
        self.model_path_input = QLineEdit()
        self.model_path_input.setText(os.path.join(os.getcwd(), "models", "pyannote"))
        self.model_path_input.setPlaceholderText("Path to PyAnnote models...")
        
        self.btn_browse_model = QPushButton("Browse")
        self.btn_browse_model.clicked.connect(self.browse_model_dir)
        
        self.btn_load_model = QPushButton("Load Models")
        self.btn_load_model.clicked.connect(self.load_models)
        
        model_layout.addWidget(self.model_path_input, 3)
        model_layout.addWidget(self.btn_browse_model, 1)
        model_layout.addWidget(self.btn_load_model, 1)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # === 参考音频区域 ===
        ref_group = QGroupBox("Reference Audio (Target Singer)")
        ref_layout = QHBoxLayout()
        
        self.ref_path_input = QLineEdit()
        self.ref_path_input.setPlaceholderText("Select reference audio of target singer...")
        
        self.btn_browse_ref = QPushButton("Browse")
        self.btn_browse_ref.clicked.connect(self.browse_reference)
        
        self.btn_load_ref = QPushButton("Load Reference")
        self.btn_load_ref.clicked.connect(self.load_reference)
        self.btn_load_ref.setEnabled(False)
        
        ref_layout.addWidget(self.ref_path_input, 4)
        ref_layout.addWidget(self.btn_browse_ref, 1)
        ref_layout.addWidget(self.btn_load_ref, 1)
        ref_group.setLayout(ref_layout)
        layout.addWidget(ref_group)
        
        # === 目标音频区域 ===
        target_group = QGroupBox("Target Audio (Mixed Vocals)")
        target_layout = QHBoxLayout()
        
        self.target_path_input = QLineEdit()
        self.target_path_input.setPlaceholderText("Select mixed vocal audio...")
        
        self.btn_browse_target = QPushButton("Browse")
        self.btn_browse_target.clicked.connect(self.browse_target)
        
        self.btn_load_target = QPushButton("Load Target")
        self.btn_load_target.clicked.connect(self.load_target)
        
        target_layout.addWidget(self.target_path_input, 4)
        target_layout.addWidget(self.btn_browse_target, 1)
        target_layout.addWidget(self.btn_load_target, 1)
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)
        
        # === 分离控制区域 ===
        control_group = QGroupBox("Separation Controls")
        control_layout = QHBoxLayout()
        
        # 分离按钮
        self.btn_separate = QPushButton("🎵 Separate Target Singer")
        self.btn_separate.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self.btn_separate.clicked.connect(self.start_separation)
        self.btn_separate.setEnabled(False)
        
        # 阈值滑块
        control_layout.addWidget(QLabel("Threshold:"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        control_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel("0.50")
        control_layout.addWidget(self.threshold_label)
        
        # 相似度模式
        control_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Soft Mask", "Hard Mask", "Binary"])
        control_layout.addWidget(self.mode_combo)
        
        control_layout.addWidget(self.btn_separate)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # === 可视化区域 ===
        viz_group = QGroupBox("Visualization (Green = Target, Red = Others)")
        viz_layout = QVBoxLayout()
        
        # 缩放和对比度控制
        viz_controls = QHBoxLayout()
        viz_controls.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 100)
        self.zoom_slider.setValue(10)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        viz_controls.addWidget(self.zoom_slider)
        
        viz_controls.addWidget(QLabel("Contrast:"))
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(1, 100)
        self.contrast_slider.setValue(10)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        viz_controls.addWidget(self.contrast_slider)
        
        viz_controls.addWidget(QLabel("Heatmap Opacity:"))
        self.heatmap_opacity = QSlider(Qt.Orientation.Horizontal)
        self.heatmap_opacity.setRange(0, 100)
        self.heatmap_opacity.setValue(70)
        viz_controls.addWidget(self.heatmap_opacity)
        
        viz_layout.addLayout(viz_controls)
        
        # 波形显示
        self.scroll = QScrollArea()
        self.wv = WaveformWidget(self)
        self.wv.region_changed.connect(self.update_region_list)
        self.scroll.setWidget(self.wv)
        self.scroll.setWidgetResizable(True)
        viz_layout.addWidget(self.scroll)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group, stretch=1)
        
        # === 播放控制 ===
        play_group = QGroupBox("Playback")
        play_layout = QHBoxLayout()
        
        self.btn_play_original = QPushButton("▶ Play Original")
        self.btn_play_original.clicked.connect(self.play_original)
        
        self.btn_play_separated = QPushButton("▶ Play Separated")
        self.btn_play_separated.clicked.connect(self.play_separated)
        self.btn_play_separated.setEnabled(False)
        
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.clicked.connect(self.stop_audio)
        
        play_layout.addWidget(self.btn_play_original)
        play_layout.addWidget(self.btn_play_separated)
        play_layout.addWidget(self.btn_stop)
        play_group.setLayout(play_layout)
        layout.addWidget(play_group)
        
        # === 区域管理 ===
        region_group = QGroupBox("Selected Regions (Force Keep)")
        region_layout = QVBoxLayout()
        
        self.region_list = QListWidget()
        self.region_list.setMaximumHeight(120)
        self.region_list.itemClicked.connect(self.preview_region)
        
        region_buttons = QHBoxLayout()
        self.btn_remove_region = QPushButton("Remove Selected")
        self.btn_remove_region.clicked.connect(self.remove_selected_region)
        
        self.btn_clear_regions = QPushButton("Clear All")
        self.btn_clear_regions.clicked.connect(self.clear_regions)
        
        self.btn_save = QPushButton("💾 Save Separated Audio")
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover { background-color: #229954; }
        """)
        self.btn_save.clicked.connect(self.save_audio)
        self.btn_save.setEnabled(False)
        
        region_buttons.addWidget(self.btn_remove_region)
        region_buttons.addWidget(self.btn_clear_regions)
        region_buttons.addStretch()
        region_buttons.addWidget(self.btn_save)
        
        region_layout.addWidget(self.region_list)
        region_layout.addLayout(region_buttons)
        region_group.setLayout(region_layout)
        layout.addWidget(region_group)
        
        # === 状态栏 ===
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.progress_label = QLabel("Ready - Load models to start")
        self.status.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status.addPermanentWidget(self.progress_bar)
    
    # === 事件处理 ===
    def on_threshold_changed(self):
        value = self.threshold_slider.value() / 100
        self.threshold_label.setText(f"{value:.2f}")
    
    def update_zoom(self):
        self.wv.zoom_factor = self.zoom_slider.value() / 10.0
        self.wv.update()
    
    def update_contrast(self):
        self.wv.contrast = self.contrast_slider.value() / 10.0
        self.wv.generate_spectrogram()
        self.wv.update()
    
    # === 模型管理 ===
    def browse_model_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if path:
            self.model_path_input.setText(path)
    
    def load_models(self):
        cache_dir = self.model_path_input.text()
        self.btn_load_model.setEnabled(False)
        self.progress_label.setText("Loading models...")
        
        self.model_worker = ModelLoadWorker(self.separator, cache_dir)
        self.model_worker.progress.connect(self.progress_label.setText)
        self.model_worker.finished.connect(self.on_model_loaded)
        self.model_worker.start()
    
    def on_model_loaded(self, success: bool, msg: str):
        self.btn_load_model.setEnabled(True)
        self.progress_label.setText(msg)
        
        if success:
            self.btn_load_ref.setEnabled(True)
            QMessageBox.information(self, "Success", "Models loaded successfully!")
        else:
            QMessageBox.critical(self, "Error", msg)
    
    # === 参考音频 ===
    def browse_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", "",
            "Audio (*.wav *.mp3 *.flac *.m4a)"
        )
        if path:
            self.ref_path_input.setText(path)
    
    def load_reference(self):
        path = self.ref_path_input.text()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Warning", "Please select a valid reference file.")
            return
        
        self.reference_file = path
        self.btn_load_ref.setEnabled(False)
        self.progress_label.setText("Extracting reference embedding...")
        
        self.embedding_worker = EmbeddingWorker(self.separator, path)
        self.embedding_worker.progress.connect(self.progress_label.setText)
        self.embedding_worker.finished.connect(self.on_reference_loaded)
        self.embedding_worker.start()
    
    def on_reference_loaded(self, success: bool, msg: str, embedding):
        self.btn_load_ref.setEnabled(True)
        self.progress_label.setText(msg)
        
        if success:
            QMessageBox.information(self, "Success", f"Reference loaded! Embedding shape: {embedding.shape}")
        else:
            QMessageBox.critical(self, "Error", msg)
    
    # === 目标音频 ===
    def browse_target(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Target Audio", "",
            "Audio (*.wav *.mp3 *.flac *.m4a)"
        )
        if path:
            self.target_path_input.setText(path)
    
    def load_target(self):
        path = self.target_path_input.text()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Warning", "Please select a valid audio file.")
            return
        
        self.input_file = path
        
        # 加载音频
        try:
            self.progress_label.setText("Loading audio...")
            
            # 使用 ffmpeg 转换
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            
            subprocess.run([
                'ffmpeg', '-i', path, '-ar', str(SAMPLE_RATE),
                '-ac', '1', '-y', tmp_path
            ], check=True, capture_output=True)
            
            self.waveform, _ = sf.read(tmp_path)
            os.unlink(tmp_path)
            
            duration = len(self.waveform) / SAMPLE_RATE
            self.wv.set_audio(self.waveform.astype(np.float32), duration)
            
            self.progress_label.setText(f"Audio loaded: {duration:.2f}s")
            
            # 检查是否可以分离
            if self.separator.target_embedding is not None:
                self.btn_separate.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio: {e}")
    
    # === 分离处理 ===
    def start_separation(self):
        if not self.input_file:
            QMessageBox.warning(self, "Warning", "Please load target audio first.")
            return
        
        if self.separator.target_embedding is None:
            QMessageBox.warning(self, "Warning", "Please load reference audio first.")
            return
        
        self.btn_separate.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 无限进度
        
        threshold = self.threshold_slider.value() / 100
        
        self.separation_worker = SeparationWorker(
            self.separator, self.input_file, threshold
        )
        self.separation_worker.progress.connect(self.progress_label.setText)
        self.separation_worker.finished.connect(self.on_separation_finished)
        self.separation_worker.error.connect(self.on_separation_error)
        self.separation_worker.start()
    
    def on_separation_finished(self, separated_audio: np.ndarray, heatmap: np.ndarray):
        self.separated_waveform = separated_audio
        self.heatmap = heatmap
        
        self.btn_separate.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Separation complete!")
        
        # 更新可视化
        self.wv.set_heatmap(heatmap)
        self.wv.update()
        
        # 启用播放和保存
        self.btn_play_separated.setEnabled(True)
        self.btn_save.setEnabled(True)
        
        QMessageBox.information(
            self, "Success",
            f"Separation complete!\n"
            f"Output shape: {separated_audio.shape}"
        )
    
    def on_separation_error(self, error_msg: str):
        self.btn_separate.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Separation failed")
        QMessageBox.critical(self, "Error", error_msg)
    
    # === 区域管理 ===
    def update_region_list(self):
        self.region_list.clear()
        for i, (s, e) in enumerate(self.wv.regions):
            item = QListWidgetItem(f"Region {i+1}: {s:.2f}s - {e:.2f}s ({e-s:.2f}s)")
            item.setData(Qt.ItemDataRole.UserRole, (s, e))
            self.region_list.addItem(item)
    
    def remove_selected_region(self):
        row = self.region_list.currentRow()
        if row >= 0:
            del self.wv.regions[row]
            self.wv.update()
            self.update_region_list()
    
    def clear_regions(self):
        self.wv.regions.clear()
        self.wv.update()
        self.update_region_list()
    
    def preview_region(self, item):
        self.stop_audio()
        s, e = item.data(Qt.ItemDataRole.UserRole)
        start_sample = int(s * SAMPLE_RATE)
        end_sample = int(e * SAMPLE_RATE)
        
        if self.waveform is not None:
            sd.play(self.waveform[start_sample:end_sample], SAMPLE_RATE)
    
    # === 播放控制 ===
    def play_original(self):
        if self.input_file:
            self.stop_audio()
            self.playback_process = subprocess.Popen([
                "ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet",
                self.input_file
            ])
            self.playback_start = np.datetime64('now')
            self.cursor_timer.start(50)
    
    def play_separated(self):
        if self.separated_waveform is not None:
            self.stop_audio()
            sd.play(self.separated_waveform, SAMPLE_RATE)
    
    def stop_audio(self):
        if self.playback_process:
            self.playback_process.terminate()
            self.playback_process = None
        sd.stop()
        self.cursor_timer.stop()
        self.wv.playback_position = -1.0
        self.wv.update()
    
    def update_cursor(self):
        if self.playback_process and self.playback_process.poll() is None:
            elapsed = (np.datetime64('now') - self.playback_start) / np.timedelta64(1, 's')
            self.wv.playback_position = elapsed
            self.wv.update()
        else:
            self.stop_audio()
    
    # === 保存 ===
    def save_audio(self):
        if self.separated_waveform is None:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Separated Audio", "",
            "WAV (*.wav);;FLAC (*.flac)"
        )
        if not path:
            return
        
        try:
            sf.write(path, self.separated_waveform, SAMPLE_RATE)
            self.progress_label.setText(f"Saved to {path}")
            QMessageBox.information(self, "Success", f"Audio saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置应用样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #2c3e50;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #34495e;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            padding: 6px 12px;
            border-radius: 3px;
            background-color: #3498db;
            color: white;
            border: none;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        QPushButton:disabled {
            background-color: #7f8c8d;
        }
        QLineEdit {
            padding: 5px;
            border: 1px solid #bdc3c7;
            border-radius: 3px;
        }
        QListWidget {
            border: 1px solid #bdc3c7;
            border-radius: 3px;
        }
    """)
    
    window = PyAnnoteSingerSeparatorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
