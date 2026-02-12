# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

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

### Dataset Creation: MakeDiffSinger
- **Location**: `MakeDiffSinger/`
- **Purpose**: Pipelines for building DiffSinger datasets from raw recordings
- **Key Subdirectories**:
  - `acoustic-forced-alignment/` - Create datasets from scratch using MFA (Montreal Forced Aligner)
  - `variance-temp-solution/` - Extend acoustic datasets to variance datasets

### Dataset Processing: dataset-tools
- **Location**: `dataset-tools/`
- **Purpose**: GUI tools for audio processing and labeling
- **Language**: C++ with Qt 6.8+, CMake build system
- **Applications**:
  - **MinLabel**: Label `.lab` files with word transcriptions
  - **SlurCutter**: Edit MIDI sequences in `.ds` files
  - **AudioSlicer**: Segment audio into short clips

### Alignment Tools
- **Montreal-Forced-Aligner** (`Montreal-Forced-Aligner/`): Command-line utility for forced alignment using Kaldi
- **LyricFA** (`LyricFA/`): Automatic lyric forced alignment using ASR

### Supporting Components
- **HKCantonese_models/**: Pre-trained acoustic models for Hong Kong Cantonese
- **ultimatevocalremovergui/**: Vocal removal GUI using MDX-Net and Demucs
- **pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/**: Neural vocoder for waveform reconstruction

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
- See `samples/` for example DS files

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

### Dataset Creation (MakeDiffSinger)

#### Forced Alignment Pipeline
```bash
cd MakeDiffSinger/acoustic-forced-alignment

# 1. Validate segment lengths
python validate_lengths.py --dir <path/to/segments>

# 2. Validate labels against dictionary
python validate_labels.py --dir <path/to/segments> --dictionary <path/to/dict.txt>

# 3. Reformat WAVs for MFA (16kHz 16bit PCM)
python reformat_wavs.py --src <path/to/segments> --dst <tmp/dir>

# 4. Run Montreal Forced Aligner
mfa align <segments/> <dictionary.txt> <model.zip> <textgrids/> --beam 100 --clean --overwrite

# 5. Check TextGrid generation
python check_tg.py --wavs <path/to/segments> --tg <path/to/textgrids>

# 6. Enhance TextGrids (detect AP/SP)
python enhance_tg.py --wavs <segments/> --dictionary <dict.txt> --src <raw_tg/> --dst <final_tg/>

# 7. Build final dataset
python build_dataset.py --wavs <segments/> --tg <final_tg/> --dataset <output/dataset/>

# 8. Select validation set
python select_test_set.py <config.yaml>
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

### LyricFA (Automatic Lyric Alignment)

```bash
cd LyricFA
pip install -r requirements.txt

# Run ASR to get lab results
python fun_asr.py --language zh/en --wav_folder <wav_folder> --lab_folder <lab_folder>

# Match lyrics and generate JSON for MinLabel
python match_lyric.py --lyric_folder <lyric> --lab_folder <lab> --json_folder <json> --language zh/en
```

### dataset-tools (C++ GUI Applications)

#### Build from Source
```bash
cd dataset-tools

# Setup ONNX Runtime
cd src/libs
cmake -Dep=dml -P ../../scripts/setup-onnxruntime.cmake  # Windows
cmake -Dep=cpu -P ../../scripts/setup-onnxruntime.cmake  # Unix

# Setup vcpkg (Windows)
cd ../../
set QT_DIR=<qt_install_dir>
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
bootstrap-vcpkg.bat
vcpkg install --x-manifest-root=../scripts/vcpkg-manifest --x-install-root=./installed --triplet=x64-windows

# Setup vcpkg (Unix)
export QT_DIR=<qt_install_dir>
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install --x-manifest-root=../scripts/vcpkg-manifest --x-install-root=./installed --triplet=<x64-osx|arm64-osx|x64-linux|arm64-linux>

# Build
cmake -B build -G Ninja \
    -DCMAKE_INSTALL_PREFIX=<install_dir> \
    -DCMAKE_PREFIX_PATH=<qt_dir> \
    -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --target all
cmake --build build --target install
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

5. **Check schemas**: See `docs/ConfigurationSchemas.md` for all parameters and their customizability levels

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

### Two-Model System
1. **Variance Model** (optional): Predicts phoneme durations and pitch curves from high-level music info (MIDI, notes)
2. **Acoustic Model** (required): Generates mel-spectrograms from low-level features (phoneme sequence, durations, F0)
3. **Vocoder**: Converts mel-spectrograms to waveforms (NSF-HiFiGAN or PC-DDSP)

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

### Deployment
- Production: Use OpenUTAU or DiffScope (under development)
- Export models to ONNX format for deployment
- Include co-author line in commits: `Co-Authored-By: Warp <agent@warp.dev>`

## Testing and Validation

### Single Test Commands
```bash
# DiffSinger (validation through training)
python scripts/train.py --config <config.yaml> --exp_name <test> --reset --fast_dev_run

# Montreal Forced Aligner
tox -e py38  # Run all tests
python -m pytest tests/test_specific.py -v  # Run single test
coverage run -m pytest tests/test_specific.py -v  # With coverage

# crepe
python -m pytest tests/test_sweep.py::test_sweep -v  # Single test function
python -m pytest tests/test_sweep.py -v  # All tests in file

# audio-edit (manual testing)
python test_integration.py  # Basic integration test
```

### Lint/Format Commands
```bash
# Montreal Forced Aligner
tox -e check-formatting  # Check black formatting
tox -e format  # Apply black formatting
tox -e lint  # Run flake8

# audio-edit
ruff check .  # Run ruff linter
ruff check --fix .  # Auto-fix issues
ruff check audio_merger.py  # Single file

# DiffSinger (no formal linting, follow style guidelines)
```

### Validation Methods
- **DiffSinger**: Training validation steps, TensorBoard monitoring, inference on sample DS files
- **MFA**: Unit tests with tox, coverage reporting
- **crepe**: pytest with sweep test patterns
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

```python
class TFC(nn.Module):
    def __init__(self, c, l, k, norm):
        super(TFC, self).__init__()
        self.H = nn.ModuleList()
        for i in range(l):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=k, stride=1, padding=k // 2),
                    norm(c),
                    nn.ReLU(),
                )
            )
    
    def forward(self, x):
        for h in self.H:
            x = h(x)
        return x
```

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

### File Organization Patterns
- **Configs**: `configs/` directory with YAML files
- **Models**: `models/[model_name]/` with implementation
- **Modules**: `modules/[module_type]/` for reusable components
- **Tests**: `tests/` directory with test files
- **Utils**: `utils/` directory for helper functions

### GUI Development Guidelines
- Use QThread for background processing in PyQt6
- Emit signals for inter-thread communication
- Use QTimer for periodic UI updates
- Handle window closing gracefully
- Provide status updates during long operations

### Documentation Standards
- Use docstrings for all classes and public methods
- Include parameter types and return values
- Add inline comments for complex algorithms
- Document configuration parameters in YAML files

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

### Testing Issues
- **MFA**: Use `tox -e py38` for full test suite, `coverage run -m pytest` for single tests
- **crepe**: Ensure test audio files exist in `tests/` directory
- **DiffSinger**: Use `--fast_dev_run` flag for quick validation during development

## Key Reference Documents

When working in DiffSinger:
- **Getting Started**: `DiffSinger/docs/GettingStarted.md`
- **Best Practices**: `DiffSinger/docs/BestPractices.md`
- **Configuration Reference**: `DiffSinger/docs/ConfigurationSchemas.md`

When building datasets:
- **Forced Alignment**: `MakeDiffSinger/acoustic_forced_alignment/README.md`
- **Variance Extension**: `MakeDiffSinger/variance-temp-solution/README.md`

## License
All main projects use Apache 2.0 License. Always obtain permission before training models on someone's voice.
