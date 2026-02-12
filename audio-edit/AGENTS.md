# AGENTS.md

This file provides guidance to agentic coding agents working with this audio editing application repository.

## Repository Overview

This is a Python-based audio editing toolkit with PyQt6 GUI applications for various audio processing tasks:
- **audio_merger.py** - Merge multiple audio files with waveform visualization
- **audio_muter.py** - Remove background noise from audio files with region selection
- **audio_trimmer.py** - Trim and edit audio segments
- **singer_cleaner.py** - Voice activity detection and speaker separation
- **t-w.py** - Real-time audio visualization with FFmpeg

## Build/Run/Test Commands

### Running Applications
Each Python file is a standalone PyQt6 application:

```bash
# Run individual applications
python audio_merger.py
python audio_muter.py  
python audio_trimmer.py
python singer_cleaner.py
python t-w.py
```

### Dependencies Installation
```bash
# Core dependencies
pip install PyQt6 pyqtgraph numpy ffmpeg-python soundfile sounddevice

# ML/AI dependencies (for singer_cleaner.py)
pip install torch torchaudio silero-vad speechbrain huggingface_hub onnxruntime

# Optional: Install ruff for linting
pip install ruff
```

### Linting Commands
```bash
# Run ruff linter on all files
ruff check .

# Run ruff with auto-fix
ruff check --fix .

# Run ruff on specific file
ruff check audio_merger.py
```

### Testing Commands
```bash
# Run basic integration test
python test_integration.py
```

Manual testing is primarily done by running individual applications.

## Code Style Guidelines

### Import Organization
1. **Standard library imports first** (sys, os, tempfile, signal, warnings, subprocess)
2. **Third-party imports next** (numpy, PyQt6 modules, ffmpeg, soundfile, torch, etc.)
3. **Local imports last** (none in this repository)

Import style:
```python
# Standard library
import sys
import os
import tempfile
import warnings

# Third-party
import numpy as np
import ffmpeg
import soundfile as sf
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QPushButton, QLabel, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPainter, QColor

```

### Naming Conventions
- **Classes**: PascalCase (`AudioMergerApp`, `WaveformWidget`)
- **Methods/Functions**: snake_case (`set_audio_data`, `process_file()`)
- **Variables**: snake_case (`sample_rate`, `audio_buffer`, `playback_position`)
- **Constants**: UPPER_SNAKE_CASE (`SAMPLE_RATE`, `CHUNK_SIZE`, `MIN_HEIGHT`)
- **Private members**: Use leading underscore (`_process_data`, `_update_ui`)

### PyQt6 Patterns & Error Handling
- Use `pyqtSignal` for inter-thread communication
- Always call `super().__init__(parent)` in widget `__init__`
- Use QThread for background processing, emit signals for UI updates
- Override PyQt6 methods (`paintEvent`, `mousePressEvent`, etc.)
- Use specific exception handling, provide QMessageBox feedback
- Catch `Exception` at top level, more specific in deeper code

```python
try:
    result = process_audio(file_path)
    return result
except FileNotFoundError:
    QMessageBox.warning(self, "Error", "File not found")
    return None
except Exception as e:
    QMessageBox.critical(self, "Error", f"Processing failed: {str(e)}")
    return None
```

### Audio Processing Guidelines
- **Sample rates**: Common rates are 44100Hz and 48000Hz
- **Data types**: Use numpy arrays, typically float32 for audio processing
- **FFmpeg integration**: Use python-ffmpeg for audio processing
- **Memory management**: Process audio files in chunks for large files

### File Structure Patterns
Each application follows this structure:
1. Imports (organized as described above)
2. Global constants/configuration
3. Custom widget classes (WaveformWidget, etc.)
4. Worker thread classes (for background processing)
5. Main application class
6. `if __name__ == "__main__"` block

### Threading Patterns
- Use QThread for background audio processing
- Emit signals to communicate results back to main thread
- Always include error handling in worker threads
- Use QTimer for periodic UI updates

### GUI Development Guidelines
- Use QVBoxLayout/HHBoxLayout for responsive layouts
- Set minimum sizes for widgets (`setMinimumHeight(140)`)
- Use QProgressDialog for long-running operations
- Provide status updates during processing
- Handle window closing gracefully (cleanup resources)

### File Paths and Resources
- Use `pathlib.Path` for path manipulation
- Handle cross-platform path differences
- Use QFileDialog for file operations
- Check file existence before processing

### Memory and Performance
- Process audio files in chunks for large files
- Use numpy arrays efficiently
- Release audio resources when done
- Monitor memory usage with ML models (singer_cleaner.py)

### Code Comments and Documentation
- Use docstrings for classes and complex methods
- Inline comments for complex audio processing logic
- TODO comments for future improvements
- Include platform-specific notes in comments when relevant

## Special Considerations

### Offline Mode (singer_cleaner.py)
The singer_cleaner.py includes mandatory offline compatibility patches at the very top to force local-only operation. These patches cannot be modified or moved.

### Platform Compatibility
- macOS: AVFoundation for audio capture
- Linux: PulseAudio or ALSA backends  
- Windows: DirectShow backend
- FFmpeg must be installed and in PATH for audio processing

### ML Model Usage
When working with singer_cleaner.py:
- Models are loaded from `./models/` directory
- All network access is disabled by design
- Use the provided huggingface_hub patches
- Be patient with model loading times