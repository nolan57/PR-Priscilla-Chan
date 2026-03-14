# AGENTS.md

This file provides guidance for AI coding agents when working with code in this repository.

## Repository Overview

This is a collection of AI/ML projects focused on **singing voice synthesis**, specifically the DiffSinger ecosystem. The main workflow involves preparing datasets, training acoustic and variance models, and synthesizing singing voices. The repository contains multiple interconnected projects that form a complete pipeline from raw audio to trained models.

## Key Projects Architecture

### Core Training System: DiffSinger
- **Location**: `DiffSinger/`
- **Purpose**: Main singing voice synthesis system using shallow diffusion mechanisms
- **Language**: Python with PyTorch
- **Key Directories**:
  - `scripts/` - Entry points for all operations (binarize, train, infer, export)
  - `training/` - Training tasks for acoustic and variance models
  - `preprocessing/` - Binarizers that convert raw data to training format
  - `inference/` - Inference logic for generating audio
  - `modules/` - Neural network architectures
  - `configs/` - Configuration templates and examples
  - `basics/` - Base classes for datasets, tasks, and modules

### Vocoder Training: SingingVocoders
- **Location**: `SingingVocoders/`
- **Purpose**: Neural vocoder training for singing voice synthesis
- **Supported Models**: NSF-HiFiGAN, UnivNet, DDSP-GAN, LVC-DDSP-GAN, DDSP-UnivNet
- **Key Files**:
  - `train.py` - Main training entry point
  - `process.py` - Audio preprocessing
  - `export_ckpt.py` - Checkpoint export utility
  - `configs/` - Model-specific configuration files

### Audio Processing Pipeline: pipeline
- **Location**: `pipeline/`
- **Purpose**: Batch processing utilities for dataset preparation
- **Key Scripts**:
  - `separate_vocals.py` / `separate_vocals_all.py` - Batch vocal separation using UVR
  - `build_npzs.py` / `build_npzs_batch.py` - Build NPZ files for vocoder training
  - `extract_perfect_f0.py` / `extract_perfect_f0_batch.py` - F0 extraction
  - `align_lyrics.py` - Lyric alignment
  - `dsf_to_wav_converter.py` - Convert DSF to WAV format

### Audio Editing Tools: audio-edit
- **Location**: `audio-edit/`
- **Purpose**: PyQt6 GUI applications for audio processing
- **Applications**:
  - **audio_merger.py** - Merge multiple audio files with waveform visualization
  - **audio_muter.py** - Remove background noise from audio files
  - **audio_trimmer.py** - Trim and edit audio segments
  - **singer_cleaner.py** - Voice activity detection and speaker separation
  - **pro_singer_separator.py** - Professional singer voice isolation (Demucs + SpeechBrain)
  - **pyannote_singer_separator.py** - Target speaker extraction using PyAnnote.Audio
  - **harmony_remover.py** - Harmony/vocal background removal
  - **t-w.py** - Real-time audio visualization with FFmpeg

### Alignment Tools
- **Montreal-Forced-Aligner** (`Montreal-Forced-Aligner/`): Command-line utility for forced alignment using Kaldi
- **crepe** (`crepe/`): Pitch estimation library

### Supporting Components
- **HKCantonese_models/**: Pre-trained acoustic models and lexicons for Hong Kong Cantonese
- **ultimatevocalremovergui/**: Vocal removal system using MDX-Net and Demucs (used by pipeline scripts)
- **whisper-small/**: Whisper ASR model for transcription

## Common Development Commands

### DiffSinger (Main Project)

#### Setup
```bash
cd DiffSinger
pip install -r requirements.txt  # Main dependencies (requires PyTorch 2.0+ installed separately)
pip install -r requirements-onnx.txt  # For ONNX export (requires PyTorch 1.13 specifically)
```

#### Preprocessing (Binarization)
```bash
python scripts/binarize.py --config <config.yaml>
```
- Converts raw datasets to binary format for training
- Set `binarization_args.num_workers` in config for multiprocessing acceleration

#### Training
```bash
python scripts/train.py --config <config.yaml> --exp_name <experiment_name> --reset
```
- Checkpoints saved to `checkpoints/<experiment_name>/`
- Remove `--reset` to resume from latest checkpoint
- For multi-GPU training with TensorBoard: `tensorboard --logdir checkpoints/ --reload_multifile=true`

#### Inference
```bash
# Variance model
python scripts/infer.py variance <song.ds> --exp <experiment_name>

# Acoustic model
python scripts/infer.py acoustic <song.ds> --exp <experiment_name>
```
- Requires `.ds` files (JSON format with phoneme sequences, durations, and scores)

#### Export to ONNX (Deployment)
```bash
# Create separate environment with PyTorch 1.13
python scripts/export.py variance --exp <experiment_name>
python scripts/export.py acoustic --exp <experiment_name>
python scripts/export.py nsf-hifigan --config <config.yaml> --ckpt <checkpoint_path>
```

#### Other Utilities
```bash
python scripts/drop_spk.py  # Remove speaker embeddings from checkpoints
python scripts/vocode.py    # Run vocoder on mel-spectrograms
```

### SingingVocoders (Vocoder Training)

#### Setup
```bash
cd SingingVocoders
# Recommended: Create conda environment from vocoder.yaml
conda env create -f vocoder.yaml -n vocoder
conda activate vocoder
```

#### Preprocessing
```bash
python process.py --config configs/<config_file>.yaml --num_cpu <num_cores> --strx 1
```

#### Training
```bash
# Basic training
python train.py --config configs/<config_file>.yaml --exp_name <experiment_name>

# With custom work directory
python train.py --config configs/base_hifi_gpu.yaml --exp_name my_vocoder_exp --work_dir ./my_experiments

# RTX 5090 optimized (32GB VRAM)
python train.py --config configs/base_hifi_rtx5090_optimized.yaml --exp_name rtx5090_exp
```

#### Export Checkpoint
```bash
python export_ckpt.py --ckpt_path <path_to_ckpt> --save_path <output_path>
```

#### Monitoring
```bash
tensorboard --logdir experiments/<exp_name>/lightning_logs/
```

### Audio Processing Pipeline

#### Vocal Separation (UVR Integration)
```bash
cd pipeline

# Single folder
python separate_vocals.py --input <input_dir> --output <output_dir> --model <model_name>

# Batch processing (all subfolders)
python separate_vocals_all.py --input <input_dir> --output <output_dir> --model <model_name> --workers 2

# Dual output (vocals + instrumentals)
python separate_vocals_all_dual.py --input <input_dir> --output <output_dir> --model <model_name>
```

#### DSF to WAV Conversion
```bash
python dsf_to_wav_converter.py -i <input_dir> -o <output_dir> --sample-rate 44100
```

#### F0 Extraction
```bash
python extract_perfect_f0.py --input <input_file> --output <output_file>
python extract_perfect_f0_batch.py --input <input_dir> --output <output_dir>
```

#### Build NPZs for Vocoder Training
```bash
python build_npzs.py --config <config.yaml>
python build_npzs_batch.py --config <config.yaml>
```

### audio-edit Applications

#### Running Applications
```bash
cd audio-edit

# Individual applications
python audio_merger.py      # Audio merger
python audio_muter.py       # Noise removal
python audio_trimmer.py     # Audio trimming
python singer_cleaner.py    # Voice activity detection & speaker separation
python pro_singer_separator.py  # Professional singer isolation (Demucs + SpeechBrain)
python pyannote_singer_separator.py  # Target speaker extraction
python harmony_remover.py   # Harmony removal
python t-w.py               # Real-time waveform visualization
```

#### Dependencies Installation
```bash
# Core dependencies
pip install PyQt6 pyqtgraph numpy ffmpeg-python soundfile sounddevice

# For singer_cleaner.py
pip install torch torchaudio silero-vad speechbrain huggingface_hub onnxruntime

# For pro_singer_separator.py
pip install -r requirements_pro_separator.txt

# For pyannote_singer_separator.py
pip install -r requirements_pyannote.txt
```

### Montreal Forced Aligner

```bash
# Installation
conda install -c conda-forge montreal-forced-aligner

# Basic alignment
mfa align <corpus_dir> <dictionary> <acoustic_model> <output_dir>

# Check version
mfa version
```

## Configuration System (DiffSinger)

DiffSinger uses cascading YAML configurations:
- **Base**: `configs/base.yaml` - Common parameters
- **Acoustic**: `configs/acoustic.yaml` - For acoustic model training
- **Variance**: `configs/variance.yaml` - For variance model training

### Key Configuration Patterns

1. **Inherit and override** using `base_config`:
```yaml
base_config:
  - configs/base.yaml
```

2. **Always customize** these fields:
   - `dictionaries` - Language-to-dictionary mappings
   - `datasets` - Raw dataset paths, speakers, languages
   - `binary_data_dir` - Output directory for binarized data
   - `vocoder_ckpt` - Path to vocoder checkpoint

3. **Model types** are determined by `task_cls`:
   - `training.acoustic_task.AcousticTask` - Acoustic models
   - `training.variance_task.VarianceTask` - Variance models

4. **Multiprocessing**: Set `binarization_args.num_workers` and `ds_workers` based on CPU cores

5. **Check schemas**: See `docs/ConfigurationSchemas.md` for all parameters

## SingingVocoders Configuration

### Configuration Selection Guide
- **RTX 5090 (32GB VRAM)**: Use `configs/base_hifi_rtx5090_optimized.yaml`
- **High-end GPU**: Use `configs/base_hifi_gpu.yaml`
- **CPU Training**: Use `configs/base_hifi_cpu.yaml`

### Key Parameters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `batch_size` | Samples per batch | 32 for RTX 5090 |
| `crop_mel_frames` | Time frames per sample | Affects memory |
| `upsample_initial_channel` | Model channel count | Affects model size |
| `pl_trainer_precision` | Training precision | `16-mixed` (fast) or `32-true` (stable) |
| `ds_workers` | Data loading workers | Based on CPU cores |
| `optimizer_cls` | Optimizer type | Muon for faster training |

### RTX 5090 Performance Reference
- Per-step training time: ~0.3-0.5 seconds
- Memory usage: ~25-28GB
- GPU utilization: ~85-95%
- Full training time: 24-48 hours

## Dataset Structure

### Raw Dataset Format
```
dataset_name/
  raw/
    wavs/
      recording1.wav
      recording2.wav
    transcriptions.csv
```

### Transcriptions CSV Columns
- **Acoustic models**: `name`, `ph_seq`, `ph_dur`
- **Variance models** (additional): `ph_num`, `note_seq`, `note_dur`
- `ph_seq`: Phoneme sequence (space-separated)
- `ph_dur`: Phoneme durations in seconds (space-separated)

### Multi-Language Support
- Use ISO 639 language codes (`zh`, `en`, `ja`, `yue`)
- Language-specific phonemes prefixed with language: `zh/a`, `en/eh`
- Global phonemes (`SP`, `AP`) have no prefix
- Define `extra_phonemes` and `merged_phoneme_groups` in config

## Architecture Insights

### DiffSinger Two-Model System
1. **Variance Model** (optional): Predicts phoneme durations and pitch curves from high-level music info (MIDI, notes)
2. **Acoustic Model** (required): Generates mel-spectrograms from low-level features (phoneme sequence, durations, F0)
3. **Vocoder**: Converts mel-spectrograms to waveforms (NSF-HiFiGAN, UnivNet, or DDSP-based)

### Training Pipeline
Raw Data → (MFA/LyricFA) → Aligned Labels → (Binarization) → Training → (Export) → ONNX Models

### Diffusion Types
- **DDPM** (Denoising Diffusion Probabilistic Models)
- **Reflow** (Rectified Flow) - Set `diffusion_type: reflow` in config
- Sampling acceleration: DDIM, PNDM, DPM-Solver++, UniPC

### Shallow Diffusion
- Enabled by default: `use_shallow_diffusion: true`
- Uses auxiliary decoder to reduce diffusion steps
- Parameters: `T_start`, `K_step` control the diffusion depth

### Singer Separator Architecture
```
Audio Input → Demucs Separation → Clean Vocals
                ↓
Reference Clips → Speaker Embeddings → Target Profile
                ↓
Clean Vocals → Adaptive Frame Analysis → Speaker Matching
                ↓
Matching Scores → Morphological Mask → Isolated Vocal
```

## Important Technical Details

### Audio Parameters (Do NOT Change)
- `audio_sample_rate: 44100`
- `hop_size: 512`
- `fft_size: 2048`
- `audio_num_mel_bins: 128` (for acoustic models)

### Model Persistence
- Checkpoints include full model state and optimizer state
- Saved at `checkpoints/<exp_name>/model_ckpt_steps_<N>.ckpt`
- Final config saved in experiment directory (detached from inheritance chain)

### Phoneme Naming Rules
- No `/`, `-`, `+` in names (reserved characters)
- Avoid `@`, `#`, `&`, `|`, `<`, `>` (potential future use)
- ASCII preferred for compatibility

### Offline Mode (Important for audio-edit apps)
Several applications require offline operation:
- `singer_cleaner.py`, `pro_singer_separator.py`, `pyannote_singer_separator.py` have mandatory offline patches
- Set environment variables: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
- Models must be pre-downloaded to `./models/` directory

### Deployment
- Production: Use OpenUTAU or DiffScope (under development)
- Export models to ONNX format for deployment
- Include co-author line in commits: `Co-Authored-By: <agent> <agent@email>`

## Testing and Validation

### Single Test Commands
```bash
# DiffSinger (validation through training)
python scripts/train.py --config <config.yaml> --exp_name <test> --reset --fast_dev_run

# Montreal Forced Aligner
tox -e py38  # Run all tests
python -m pytest tests/test_specific.py -v  # Run single test

# crepe
python -m pytest tests/test_sweep.py::test_sweep -v  # Single test function

# audio-edit (manual testing)
python audio_merger.py  # Run application
```

### Lint/Format Commands
```bash
# audio-edit
ruff check .  # Run ruff linter
ruff check --fix .  # Auto-fix issues
ruff check audio_merger.py  # Single file

# Montreal Forced Aligner
tox -e check-formatting  # Check black formatting
tox -e format  # Apply black formatting
tox -e lint  # Run flake8
```

### Validation Methods
- **DiffSinger**: Training validation steps, TensorBoard monitoring, inference on sample DS files
- **SingingVocoders**: Validation during training with `val_check_interval`
- **MFA**: Unit tests with tox, coverage reporting
- **Manual checks**: `validate_lengths.py`, `validate_labels.py`, `check_tg.py`

## Code Style Guidelines

### Python Import Organization
1. **Standard library imports first** (os, sys, pathlib, warnings)
2. **Third-party imports next** (numpy, torch, PyQt6, yaml)
3. **Local imports last** (project-specific modules)

```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
import torch
import torch.nn as nn
import yaml
from PyQt6.QtWidgets import QApplication, QMainWindow

# Local imports
from utils.config_utils import read_full_config
```

### Naming Conventions
- **Classes**: PascalCase (`TFC`, `DenseTFC`, `AudioMergerApp`)
- **Methods/Functions**: snake_case (`forward`, `process_audio`, `set_audio_data`)
- **Variables**: snake_case (`audio_buffer`, `sample_rate`, `config_path`)
- **Constants**: UPPER_SNAKE_CASE (`SAMPLE_RATE`, `CHUNK_SIZE`)
- **Private members**: Leading underscore (`_process_data`, `_update_ui`)

### PyTorch/Deep Learning Patterns
- Extend `nn.Module` for neural network components
- Use `super().__init__()` in all module constructors
- Implement `forward()` method for all neural modules
- Use `nn.ModuleList` for dynamic layer lists
- Extend `pl.LightningModule` for training tasks

### Error Handling
- Use specific exception types when possible
- Return error information as tuples: `(False, error_message)`
- Provide user-friendly error messages in GUI applications
- Use try-except blocks for file operations and model loading

### Configuration Management
- YAML-based configuration with inheritance
- Use `base_config` for cascading configurations
- Snake_case for config keys: `audio_sample_rate`, `data_input_path`
- Validate configurations before use

### Audio Processing Guidelines
- **Sample rates**: Common rates are 44100Hz and 48000Hz
- **Data types**: Use numpy arrays, typically float32
- **Memory management**: Process large files in chunks
- **Format support**: Support .wav and .flac when possible

### Code Formatting Standards
- **Python**: Black with line-length 99 (MFA standard), flake8 for linting
- **YAML**: 2-space indentation for configurations
- **Comments**: Use docstrings for classes and complex methods
- **Type hints**: Use when beneficial, especially in function signatures

### GUI Development Guidelines (PyQt6)
- Use `pyqtSignal` for inter-thread communication
- Use QThread for background processing
- Emit signals for UI updates from worker threads
- Use QTimer for periodic UI updates
- Handle window closing gracefully
- Provide status updates during long operations

### File Organization Patterns
- **Configs**: `configs/` directory with YAML files
- **Models**: `models/[model_name]/` with implementation
- **Modules**: `modules/[module_type]/` for reusable components
- **Scripts**: `scripts/` for CLI entry points
- **Utils**: `utils/` directory for helper functions

## Troubleshooting Common Issues

### PyTorch Version Conflicts
- Training: PyTorch 2.0+ (recommended)
- ONNX Export: PyTorch 1.13 (required) - Create separate conda environment

### MFA Format Issues
- MFA requires 16kHz 16bit PCM WAVs - Use `reformat_wavs.py`
- Try different `--beam` values (default 100) if alignment fails

### Binarization Errors
- Ensure all phonemes in labels exist in dictionary
- Check `validate_labels.py` output for coverage issues
- Verify dataset paths in config are correct

### Multi-GPU Training
- Set `pl_trainer_devices: auto` or specific GPU IDs
- Use `pl_trainer_strategy.name: ddp` for distributed training
- Launch TensorBoard with `--reload_multifile=true`

### SingingVocoders Memory Issues
- Reduce `batch_size`
- Decrease `crop_mel_frames`
- Reduce `upsample_initial_channel`
- Use mixed precision training (`pl_trainer_precision: 16-mixed`)

### Singer Separator Issues
- **Slow Processing**: Reduce audio length or disable Demucs
- **Poor Separation**: Add more diverse reference segments
- **Memory Issues**: Use smaller audio files or basic separation mode
- **Demucs Not Available**: Install `pip install demucs` or `pip install julius`

## Key Reference Documents

### DiffSinger
- **Getting Started**: `DiffSinger/docs/GettingStarted.md`
- **Best Practices**: `DiffSinger/docs/BestPractices.md`
- **Configuration Reference**: `DiffSinger/docs/ConfigurationSchemas.md`

### SingingVocoders
- **Quick Start**: `SingingVocoders/TRAINING_QUICK_START.md`
- **Fine-tuning Guide**: `SingingVocoders/FINETUNE_PARAMETER_GUIDE.md`
- **RTX 5090 Optimization**: `SingingVocoders/RTX5090_OPTIMIZATION_GUIDE.md`
- **Config Comparison**: `SingingVocoders/CONFIG_COMPARISON.md`

### audio-edit
- **Pro Singer Separator**: `audio-edit/PRO_SINGER_SEPARATOR_README.md`
- **T-W FFmpeg**: `audio-edit/TW_FFMPEG_README.md`
- **Vocoder Analysis**: `audio-edit/vocoder_audio_processing_analysis.md`

### Pipeline
- **README**: `pipeline/README.md`
- **UVR Guide**: `pipeline/uvr_diffsinger_guide.md`

## License
All main projects use Apache 2.0 License. Always obtain permission before training models on someone's voice.