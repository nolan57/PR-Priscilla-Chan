# Ultimate Vocal Remover GUI - Project Documentation

## Project Overview

Ultimate Vocal Remover (UVR) is a powerful audio source separation tool that uses state-of-the-art deep learning models to separate vocals and instruments from audio files. The project implements three different architectures for audio separation: VR (Vocal Remover), MDX-Net, and Demucs, allowing users to choose the most appropriate method for their specific use case.

### Key Features:
- Multiple processing methods (VR, MDX-Net, Demucs)
- Ensemble mode for combining multiple models
- Command-line interface for easy automation
- Support for various output formats (WAV, MP3, FLAC)
- GPU acceleration for faster processing
- Batch processing capability
- Denoising and de-reverb options
- Sample creation for quick testing

### Technologies Used:
- Python 3.8+
- PyTorch for deep learning models
- ONNX Runtime for model inference
- SoundFile and Librosa for audio processing
- Various audio processing libraries (pydub, pyrubberband, etc.)

## Project Structure

```
ultimatevocalremovergui/
├── UVR.py              # Main entry point (CLI)
├── cli.py              # Command-line interface implementation
├── separate.py         # Audio separation implementation
├── modeldata.py        # Model data handling
├── __version__.py      # Version information
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
├── USER_GUIDE.md      # User manual
├── REBUILD_PROCESS.md # Rebuild instructions
├── data.pkl           # Saved configuration data
├── core/              # Core functionality modules
│   ├── __init__.py    # Module initialization
│   ├── models.py      # Model management
│   ├── ensemble.py    # Ensemble processing
│   ├── utils.py       # Utility functions
│   └── config.py      # Configuration management
├── demucs/            # Demucs implementation
├── gui_data/          # GUI-related files (minimal)
├── lib_v5/            # Core libraries for audio processing
│   ├── __init__.py
│   ├── mdxnet.py      # MDX-Net implementation
│   ├── modules.py     # Audio processing modules
│   ├── pyrb.py        # Rubber band implementation
│   ├── results.py     # Result handling
│   ├── spec_utils.py  # Spectral utilities
│   ├── tfc_tdf_v3.py  # TFC-TDF network implementation
│   └── vr_network/    # VR network implementation
├── models/            # Model directories
│   ├── VR_Models/     # VR architecture models
│   ├── MDX_Net_Models/ # MDX-Net models
│   └── Demucs_Models/  # Demucs models
├── output/            # Default output directory
└── config/            # Configuration files
```

## Building and Running

### Prerequisites:
1. Python 3.8 or higher installed on your system
2. A virtual environment (recommended)
3. Git for cloning the repository
4. FFmpeg for processing non-WAV audio files
5. Rubber Band library for time-stretch and pitch-shift options

### Installation Steps:
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Anjok07/ultimatevocalremovergui.git
   cd ultimatevocalremovergui
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   # On Windows: venv\Scripts\activate
   # On macOS/Linux: source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch with CUDA support (optional but recommended):**
   ```bash
   # For CUDA 11.7
   pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/cu117
   ```

5. **Download models:**
   - VR models should be placed in `models/VR_Models/`
   - MDX models should be placed in `models/MDX_Net_Models/`
   - Demucs models should be placed in `models/Demucs_Models/v3_v4_repo/`

### Running the Application:
```bash
# Basic usage
python UVR.py [options] input [input ...]

# Example: Separate vocals from instrumental using VR method
python UVR.py -m vr -t UVR-Model-5 -o output audio_file.wav

# Example: Separate all stems using Demucs
python UVR.py -m demucs -t htdemucs -o output --demucs-stems all audio_file.wav
```

## Core Components

### 1. Main Entry Point (UVR.py)
The main entry point initializes the application and runs in CLI mode by default. It creates a CLI instance and executes it.

### 2. Command-Line Interface (cli.py)
Implements the command-line interface with extensive options for:
- Input/output configuration
- Processing method selection (VR, MDX, Demucs, Ensemble)
- Method-specific parameters
- Output format and quality settings
- Device selection (CPU/GPU)

### 3. Audio Separation Engine (separate.py)
Contains the core separation algorithms for all three methods:
- `SeperateVR`: VR architecture implementation
- `SeperateMDX`: MDX-Net implementation
- `SeperateMDXC`: MDX-C implementation
- `SeperateDemucs`: Demucs implementation
- `SeperateAttributes`: Base class with shared functionality

### 4. Model Management (core/models.py)
Handles model loading, validation, and metadata management:
- Model path resolution
- Hash calculation for model identification
- Parameter loading from JSON files
- Model-specific configuration

### 5. Configuration Management (core/config.py)
Manages application configuration with:
- Default configuration values
- User configuration persistence
- Validation and error handling
- Process-specific configurations

### 6. Core Libraries (lib_v5/)
Contains the core audio processing implementations:
- `tfc_tdf_v3.py`: TFC-TDF network for MDX-Net
- `vr_network/`: VR network implementation
- `mdxnet.py`: MDX-Net processing
- `spec_utils.py`: Spectral processing utilities

## Processing Methods

### 1. VR (Vocal Remover)
Original Vocal Remover architecture based on deep learning models:
- Optimized for vocal/instrumental separation
- Configurable aggression settings
- Post-processing options
- Test-Time Augmentation (TTA)

### 2. MDX-Net
Advanced model based on TFC-TDF networks:
- Often provides better separation quality
- Particularly effective for complex mixes
- Configurable segment sizes and overlaps
- Compensation for volume differences

### 3. Demucs
State-of-the-art source separation model:
- Can separate into multiple stems (vocals, drums, bass, other)
- Advanced neural network architecture
- Flexible configuration options
- Ensemble capabilities

### 4. Ensemble Mode
Combines multiple models to improve separation quality:
- Averages outputs from different models
- Supports various combination algorithms (mean, median, sum)
- Can use models from different architectures

## Development Conventions

### Code Style:
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Include docstrings for public functions and classes
- Maintain consistent formatting throughout the codebase

### Testing:
- The application includes extensive command-line options for testing
- Sample creation feature allows quick testing of models
- Configuration validation helps prevent runtime errors

### Error Handling:
- Comprehensive error handling throughout the codebase
- Logging for debugging and monitoring
- Graceful degradation when resources are unavailable

## Usage Examples

### Basic Separation:
```bash
# VR method with default settings
python UVR.py -m vr -t UVR-Model-5 -o output input.wav

# MDX-Net with custom segment size
python UVR.py -m mdx -t UVR-MDX-NET-Vocal_1 -o output --segment-size 256 input.wav

# Demucs separating all stems
python UVR.py -m demucs -t htdemucs -o output --demucs-stems all input.wav
```

### Advanced Options:
```bash
# Use GPU with custom overlap
python UVR.py -m mdx -t UVR-MDX-NET-Inst_HQ_1 -o output --overlap-mdx 0.5 --gpu input.wav

# Ensemble multiple models
python UVR.py -m ensemble -o output --ensemble-models UVR-Model-5 UVR-Model-4 input.wav

# Create MP3 output with high bitrate
python UVR.py -m vr -t UVR-Model-5 -o output --export-format mp3 --mp3-bitrate 320 input.wav
```

## Model Management

### Model Directories:
- `models/VR_Models/`: VR architecture models (.pth files)
- `models/MDX_Net_Models/`: MDX-Net models (.onnx files)
- `models/Demucs_Models/v3_v4_repo/`: Demucs models (.ckpt/.yaml files)

### Model Metadata:
- Model parameters are stored in JSON files in respective model_data directories
- Each model has associated metadata including primary/secondary stems
- Model hashes are used for validation and identification

## Troubleshooting

### Common Issues:
1. **CUDA Out of Memory**: Reduce batch size, segment size, or use CPU
2. **Model Not Found**: Verify model placement in correct directory
3. **FFmpeg Missing**: Install FFmpeg for non-WAV file support
4. **Permission Errors**: Ensure write permissions to output directory

### Performance Tips:
- Use GPU when available for faster processing
- Adjust segment/chunk sizes based on available memory
- Try different models to find the best for your audio type
- Use ensemble mode for improved quality (at cost of processing time)