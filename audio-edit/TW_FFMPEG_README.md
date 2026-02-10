# Audio Visualizer - FFmpeg Version

This is a real-time audio visualization tool that displays waveform, power spectral density (PSD), and spectrogram using PyQt6 and FFmpeg.

## Features

- Real-time waveform display (left/right channels)
- Power Spectral Density (PSD) analysis
- Spectrogram visualization
- RMS level monitoring
- Cross-platform audio capture via FFmpeg

## Requirements

### Core Dependencies
```bash
pip install PyQt6 pyqtgraph numpy
```

### FFmpeg Installation

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [FFmpeg official website](https://ffmpeg.org/download.html)

## Usage

### Basic Usage
```bash
python t-w.py
```

### Device Configuration
Modify the `DEVICE_NAME` variable in the script:
```python
DEVICE_NAME = "BlackHole 64ch"  # Your audio device name
```

To list available devices, run:
```bash
python t-w.py
```
The program will automatically detect and list available audio devices.

## Platform-Specific Notes

### macOS
Uses AVFoundation backend:
```bash
ffmpeg -f avfoundation -list_devices true -i ""
```

### Linux
Uses PulseAudio or ALSA backend:
```bash
# PulseAudio
ffmpeg -f pulse -sources

# ALSA  
ffmpeg -f alsa -devices
```

### Windows
Uses DirectShow backend:
```bash
ffmpeg -list_devices true -f dshow -i dummy
```

## Configuration Options

Adjust these parameters in the script:

```python
SAMPLE_RATE = 48000      # Audio sample rate
CHUNK_SIZE = 512         # Audio chunk size
CHANNELS = 2             # Number of channels (1=mono, 2=stereo)
GAIN = 1.0               # Display gain
PLOT_LENGTH = 1024       # Waveform display length
FFT_WINDOW = 2048        # FFT window size
BUFFER_SIZE = 8192       # FFmpeg buffer size
```

## Troubleshooting

### FFmpeg Not Found
```
❌ FFmpeg not found. Please install FFmpeg and make sure 'ffmpeg' is in PATH.
```
Solution: Install FFmpeg and ensure it's in your system PATH.

### Device Not Found
```
⚠️ Could not detect audio devices automatically.
```
Solution: 
1. Check device name spelling
2. Verify device is available in system audio settings
3. Try alternative device names

### No Audio Data
- Check if the audio device is active and receiving input
- Verify sample rate compatibility
- Ensure sufficient permissions for audio capture

## Technical Details

### Architecture
- **Main Thread**: GUI rendering and plotting
- **Audio Thread**: FFmpeg subprocess capturing audio data
- **Queue**: Thread-safe communication between audio and GUI threads

### Data Flow
1. FFmpeg captures audio from device → raw float32 PCM
2. Audio thread processes data → puts in queue
3. Main thread retrieves data → updates plots
4. Visualization updates at 20 FPS (50ms timer)

### Supported Formats
- Input: Device audio capture
- Output: Real-time visualization
- Internal: 32-bit float PCM

## Customization

### Adding New Visualizations
Extend the `AudioVisualizer` class by adding new plot widgets in `__init__`.

### Changing Color Schemes
Modify the color definitions in the spectrogram colormap section.

### Adjusting Performance
- Increase `CHUNK_SIZE` for better performance (less frequent updates)
- Decrease `BUFFER_SIZE` for lower latency
- Adjust `timer.start()` interval for different update rates

## License
MIT License - See source code for details.