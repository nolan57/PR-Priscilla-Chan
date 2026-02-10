# Ultimate Vocal Remover: Complete User Guide

## 1. Introduction

Ultimate Vocal Remover (UVR) is a powerful audio source separation tool that uses deep learning models to separate vocals and instruments from audio files. It offers multiple processing methods and models to achieve high-quality separations for various use cases.

### Key Features:
- Multiple processing methods (VR, MDX-Net, Demucs)
- Ensemble mode for combining multiple models
- Command-line interface for easy automation
- Support for various output formats (WAV, MP3, FLAC)
- GPU acceleration for faster processing
- Batch processing capability
- Denoising options
- Sample creation for quick testing

### System Requirements:
- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended for faster processing)
- 8GB+ RAM
- 5GB+ free disk space for models and outputs

## 2. Installation

### Prerequisites:
1. Python 3.8 or higher installed on your system
2. A virtual environment (recommended)
3. Git for cloning the repository

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

4. **Download models:**
   - VR models should be placed in `models/VR_Models/`
   - MDX models should be placed in `models/MDX_Net_Models/`
   - Demucs models should be placed in `models/Demucs_Models/v3_v4_repo/`

## 3. Directory Structure

```
ultimatevocalremovergui/
├── UVR.py              # Main entry point (CLI)
├── cli.py              # Command-line interface implementation
├── separate.py         # Audio separation implementation
├── core/               # Core functionality modules
│   ├── __init__.py     # Module initialization
│   ├── models.py       # Model management
│   ├── ensemble.py      # Ensemble processing
│   ├── utils.py        # Utility functions
│   └── config.py       # Configuration management
├── demucs/             # Demucs implementation
├── gui_data/           # GUI-related files (minimal)
├── lib_v5/             # Core libraries for audio processing
├── models/             # Model directories
│   ├── VR_Models/      # VR architecture models
│   ├── MDX_Net_Models/ # MDX-Net models
│   └── Demucs_Models/   # Demucs models
└── output/             # Default output directory
```

## 4. Basic Usage

### Command-Line Interface:
The primary way to use UVR is through the command-line interface:

```bash
python3 UVR.py [options] input [input ...]
```

### Common Parameters:
- `-m, --method`: Processing method (`vr`, `mdx`, `demucs`, `ensemble`)
- `-t, --model`: Model name
- `-o, --output`: Output directory
- `-d, --device`: Device to use (`cpu`, `cuda`)
- `--export-format`: Output format (`wav`, `mp3`, `flac`)
- `--mp3-bitrate`: MP3 bitrate (`128`, `192`, `256`, `320`)
- `--normalize`: Normalize output
- `--primary-only`: Save only primary stem
- `--secondary-only`: Save only secondary stem

## 5. Processing Methods

### 5.1 VR (Vocal Remover)
VR is the original Vocal Remover architecture, ideal for general vocal/instrumental separation.

**Example Usage:**
```bash
# Basic VR separation
python3 UVR.py -m vr -t UVR-DeNoise -o output input.wav

# With aggression setting
python3 UVR.py -m vr -t UVR-Model-5 -o output --aggression 10 input.wav

# With post-processing
python3 UVR.py -m vr -t UVR-Model-5 -o output --post-process input.wav
```

**VR-Specific Parameters:**
- `--aggression`: Aggression setting (0-100)
- `--window-size`: Window size (320, 512, 1024)
- `--batch-size`: Batch size
- `--crop-size`: Crop size
- `--tta`: Enable Test-Time Augmentation
- `--post-process`: Enable post-processing
- `--high-end-process`: Enable high-end processing

### 5.2 MDX-Net
MDX-Net is a more advanced model that often provides better separation quality, especially for complex mixes.

**Example Usage:**
```bash
# Basic MDX separation
python3 UVR.py -m mdx -t UVR-MDX-NET-Vocal_1 -o output input.wav

# With custom segment size
python3 UVR.py -m mdx -t UVR-MDX-NET-Vocal_1 -o output --segment-size 256 input.wav

# With compensation
python3 UVR.py -m mdx -t UVR-MDX-NET-Vocal_1 -o output --compensate 1.035 input.wav
```

**MDX-Specific Parameters:**
- `--margin`: Margin (44100, 22050, 11025)
- `--compensate`: Volume compensation
- `--chunks`: Chunks
- `--segment-size`: Segment size
- `--overlap-mdx`: Overlap for MDX processing

### 5.3 Demucs
Demucs is a state-of-the-art source separation model that can separate audio into multiple stems (vocals, drums, bass, other).

**Example Usage:**
```bash
# Basic Demucs separation
python3 UVR.py -m demucs -t htdemucs -o output input.wav

# Separate all stems
python3 UVR.py -m demucs -t htdemucs -o output --demucs-stems all input.wav

# With custom shifts
python3 UVR.py -m demucs -t htdemucs -o output --shifts 3 input.wav
```

**Demucs-Specific Parameters:**
- `--segment`: Segment duration
- `--overlap`: Overlap
- `--shifts`: Shifts
- `--demucs-stems`: Stems to separate (`vocals`, `drums`, `bass`, `other`, `all`)

### 5.4 Ensemble Mode
Ensemble mode combines multiple models to achieve better separation results by averaging their outputs.

**Example Usage:**
```bash
# Ensemble with multiple VR models
python3 UVR.py -m ensemble -t UVR-Model-5 -o output --ensemble-models UVR-Model-5 UVR-Model-4 input.wav

# Ensemble with custom algorithm
python3 UVR.py -m ensemble -t UVR-Model-5 -o output --ensemble-models UVR-Model-5 UVR-Model-4 --ensemble-algorithm mean input.wav
```

**Ensemble-Specific Parameters:**
- `--ensemble-models`: List of models to ensemble
- `--ensemble-algorithm`: Ensemble algorithm (`mean`, `median`, `sum`)

## 6. Model Management

### Model Types:
- **VR Models**: End with `.pth` (e.g., `UVR-Model-5.pth`)
- **MDX Models**: End with `.onnx` or `.ckpt` (e.g., `UVR-MDX-NET-Vocal_1.onnx`)
- **Demucs Models**: YAML configuration files (e.g., `htdemucs.yaml`)

### Model Directories:
- `models/VR_Models/`: For VR architecture models
- `models/MDX_Net_Models/`: For MDX-Net models
- `models/Demucs_Models/v3_v4_repo/`: For Demucs v3 and v4 models

### Adding Custom Models:
1. Place model files in the appropriate directory
2. For VR models, ensure model data is available in `models/VR_Models/model_data/`
3. For MDX models, ensure model data is available in `models/MDX_Net_Models/model_data/`

## 7. Configuration Options

### Command-Line Configuration:
Most configuration options can be set via command-line parameters. For example:

```bash
# Set output format to MP3
python3 UVR.py -m vr -t UVR-Model-5 -o output --export-format mp3 input.wav

# Set MP3 bitrate
python3 UVR.py -m vr -t UVR-Model-5 -o output --export-format mp3 --mp3-bitrate 320 input.wav

# Use CPU instead of GPU
python3 UVR.py -m vr -t UVR-Model-5 -o output --cpu input.wav
```

### Advanced Configuration:
For more advanced configuration, you can modify the default configuration in `core/config.py`. This allows you to set default values for all parameters.

## 8. Output Formats

### Supported Formats:
- **WAV**: Lossless audio format (highest quality)
- **MP3**: Lossy audio format (smaller file size)
- **FLAC**: Lossless audio format (compressed)

### Example Usage:
```bash
# Save as WAV
python3 UVR.py -m vr -t UVR-Model-5 -o output --export-format wav input.wav

# Save as MP3 with 320kbps
python3 UVR.py -m vr -t UVR-Model-5 -o output --export-format mp3 --mp3-bitrate 320 input.wav

# Save as FLAC
python3 UVR.py -m vr -t UVR-Model-5 -o output --export-format flac input.wav
```

## 9. Advanced Features

### Batch Processing:
Process multiple files at once by specifying multiple input files or a directory:

```bash
# Process multiple files
python3 UVR.py -m vr -t UVR-Model-5 -o output file1.wav file2.wav file3.wav

# Process all files in a directory
python3 UVR.py -m vr -t UVR-Model-5 -o output /path/to/audio/files/
```

### Sample Creation:
Create a short sample of the output for quick testing:

```bash
# Create a 10-second sample
python3 UVR.py -m vr -t UVR-Model-5 -o output --sample 10 input.wav
```

### GPU Usage:
Force GPU usage for faster processing:

```bash
# Force GPU usage
python3 UVR.py -m vr -t UVR-Model-5 -o output --gpu input.wav
```

### Denoising:
Apply denoising to the output:

```bash
# Enable denoising
python3 UVR.py -m vr -t UVR-Model-5 -o output --denoise input.wav
```

## 10. Troubleshooting

### Common Errors:
1. **Model Not Found**: Ensure the model is in the correct directory and the model name is correct.
2. **CUDA Out of Memory**: Reduce batch size, segment size, or use CPU.
3. **Invalid Input File**: Ensure the input file is a supported audio format.
4. **Permission Denied**: Ensure you have write permissions to the output directory.

### Solutions:
- **Model Issues**: Check model paths and ensure models are correctly placed.
- **Memory Issues**: Reduce processing parameters or use a smaller model.
- **Input Issues**: Convert audio files to WAV format before processing.
- **Permission Issues**: Run the command with appropriate permissions or change the output directory.

## 11. Conclusion

Ultimate Vocal Remover is a powerful tool for audio source separation, offering multiple processing methods and models to achieve high-quality results. With its command-line interface, it's easy to integrate into workflows and automate processing tasks.

### Key Takeaways:
- Choose the right processing method based on your needs:
  - **VR**: General purpose separation
  - **MDX**: Better for complex mixes
  - **Demucs**: Best for multi-stem separation
  - **Ensemble**: Combine multiple models for optimal results
- Experiment with different models to find the best one for your specific audio
- Adjust parameters based on your hardware capabilities
- Use batch processing for efficient workflow

### Resources:
- [GitHub Repository](https://github.com/Anjok07/ultimatevocalremovergui)
- [Model Repository](https://github.com/TRvlvr/model_repo)
- [Documentation](https://github.com/Anjok07/ultimatevocalremovergui/wiki)

For more detailed information and troubleshooting, please refer to the official documentation and GitHub repository.