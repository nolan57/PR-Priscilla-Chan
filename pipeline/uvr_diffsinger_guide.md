# Optimizing Ultimate Vocal Remover for DiffSinger Preprocessing

## Table of Contents
1. [Introduction](#introduction)
2. [DiffSinger Requirements](#diffsinger-requirements)
3. [UVR Model Types Explained](#uvr-model-types-explained)
4. [Optimal Settings for DiffSinger](#optimal-settings-for-diffsinger)
5. [Workflow Recommendations](#workflow-recommendations)
6. [Quality Control](#quality-control)
7. [Troubleshooting](#troubleshooting)

## Introduction

Ultimate Vocal Remover (UVR) is a powerful audio source separation tool that can be optimized to extract high-quality vocals for DiffSinger training. This guide explains how to achieve the best results for DiffSinger preprocessing workflows.

## DiffSinger Requirements

DiffSinger expects audio data with the following characteristics:
- **Sample Rate**: 44.1 kHz
- **Format**: WAV (recommended), though FLAC is also supported
- **Bit Depth**: Preferably 32-bit float for maximum quality
- **Mono Audio**: Though stereo is accepted and converted internally
- **Clean Audio**: Minimal noise and artifacts for better feature extraction

For harmonic-noise separation (hnsep), DiffSinger can use either:
- `world`: Uses PyWorld for harmonic/percussive separation
- `vr`: Uses neural network-based vocal remover models (based on UVR architecture)

## UVR Model Types Explained

### 1. VR Architecture (VR)
- Based on deep learning with cascaded networks
- Excellent for general vocal/instrumental separation
- Good for harmonic-noise separation
- Models with names like "UVR-DeNoise", "UVR-DeReverb" belong to this category

### 2. MDX-Net (MDX)
- Based on DenseNet architecture
- Excellent for preserving musical details
- Models like "UVR-MDX-NET-Inst_HQ_4" provide high-quality separation
- Good for both vocals and instruments

### 3. Demucs
- Facebook's Demucs models
- Effective for multi-instrument separation
- Good for extracting specific stems like bass, drums, etc.

## Optimal Settings for DiffSinger

### For High-Quality Vocal Extraction:

```
Process Method: MDX-Net
Model: UVR-MDX-NET-Inst_HQ_4 (or similar high-quality model)
Aggression Setting: 5 (moderate, preserves more detail)
Window Size: 512 (good balance)
Segment Size: 256-512 (adjust based on GPU memory)
Overlap: 0.25 (lower for better quality)
Enable Denoise: Yes
Primary Stem Only: Yes (for vocals)
Save Format: WAV, 32-bit float
```

### For Instrumental Extraction:

```
Process Method: MDX-Net
Model: UVR-MDX-NET-Inst_HQ_4
Aggression Setting: 5
Window Size: 512
Segment Size: 256-512
Overlap: 0.25
Enable Denoise: Yes
Secondary Stem Only: Yes (for instrumental)
Save Format: WAV, 32-bit float
```

### For Noise Reduction:

```
Process Method: VR Architecture
Model: UVR-DeNoise-Lite (or similar)
Aggression Setting: 3 (conservative)
Window Size: 512
Crop Size: 256
Enable TTA: No (increases processing time with marginal gains)
Save Format: WAV, 32-bit float
```

## Workflow Recommendations

### Basic Workflow
1. **Validate Input**: Ensure all audio files are at 44.1kHz or resample to this rate
2. **Vocal Extraction**: Use MDX-Net with high-quality model
3. **Noise Reduction**: Apply denoising model to clean vocals
4. **Format Conversion**: Save as 32-bit float WAV files

### Advanced Workflow
1. **Initial Separation**: Use MDX-Net for initial vocal/instrumental split
2. **Secondary Processing**: Apply VR model with denoising to vocal track
3. **Ensemble Processing**: Combine results from multiple models
4. **Quality Enhancement**: Apply post-processing for artifact reduction

### Ensemble Mode for Maximum Quality
- Combine multiple models to improve separation quality
- Use "Max Spec/Min Spec" for vocal extraction
- Models to combine: UVR-MDX-NET-Inst_HQ_4 + UVR-DeNoise-Lite

## Quality Control

### Checking Results
- Listen to separated vocals for artifacts or unwanted noise
- Compare original and separated audio levels
- Verify frequency spectrum preservation (especially 100Hz-8kHz range)
- Check for temporal alignment issues

### Common Issues and Fixes
- **Over-suppression**: Reduce aggression setting
- **Artifacts**: Increase segment size or enable denoising
- **Phase issues**: Try different phase options if available
- **Clipping**: Ensure normalization is applied appropriately

## Troubleshooting

### Processing Issues
- **Out of memory**: Reduce segment size or use CPU processing
- **Slow processing**: Use larger chunks or enable GPU acceleration
- **Model not found**: Verify model files are in correct directories

### Quality Issues
- **Vocals contain instrumental artifacts**: Increase aggression or use different model
- **Vocals sound muffled**: Check for over-filtering, try different model
- **Missing high frequencies**: Use models known for preserving HF content

### Integration with DiffSinger
- Ensure separated vocals are at 44.1kHz sample rate
- Convert stereo to mono if needed
- Save in formats compatible with DiffSinger preprocessing
- Validate that pitch extraction works properly on separated audio

## Conclusion

By following these guidelines, you can optimize Ultimate Vocal Remover for producing high-quality separated vocals suitable for DiffSinger training. The key is balancing separation quality with artifact introduction, ensuring the resulting audio maintains the natural characteristics needed for accurate feature extraction in DiffSinger's preprocessing pipeline.

# Script Usage Guide

This section provides detailed instructions on how to use the three automation scripts included in this project.

## 1. optimize_uvr_for_diffsinger.py

This script provides optimized settings for UVR specifically tailored for DiffSinger preprocessing.

### Purpose:
- Provides UVR settings optimized for DiffSinger
- Recommends models and parameters best suited for DiffSinger
- Offers advanced preprocessing workflows

### How to Use:
```bash
python optimize_uvr_for_diffsinger.py --input_dir input_directory --output_dir output_directory [--workflow basic|advanced]
```

### Example:
```bash
# Basic workflow
python optimize_uvr_for_diffsinger.py --input_dir ./audio_input --output_dir ./separated_vocals

# Advanced workflow
python optimize_uvr_for_diffsinger.py --input_dir ./audio_input --output_dir ./separated_vocals --workflow advanced
```

### Features:
- Automatically validates input audio for DiffSinger compliance (44.1kHz sample rate)
- Provides model recommendations (e.g., UVR-MDX-NET-Inst_HQ_4)
- Shows optimized parameters (aggression setting 5, window size 512, etc.)

## 2. automatic_uvr_optimizer.py

This script automatically configures UVR with optimal settings for DiffSinger preprocessing.

### Purpose:
- Automatically creates optimized UVR configurations
- Validates that audio meets DiffSinger requirements
- Generates configuration files for later use

### How to Use:
```bash
python automatic_uvr_optimizer.py --input_dir input_directory --output_dir output_directory [--config_path config_path] [--validate_only]
```

### Example:
```bash
# Create optimized config and validate audio
python automatic_uvr_optimizer.py --input_dir ./audio_input --output_dir ./output

# Validation only mode
python automatic_uvr_optimizer.py --input_dir ./audio_input --output_dir ./output --validate_only
```

### Features:
- Creates JSON configuration files with optimal settings
- Validates audio compliance with DiffSinger requirements
- Provides configurations for different scenarios (high-quality vocals, noise reduction, instrumental extraction)

## 3. run_uvr_for_diffsinger.py

This is a complete end-to-end processing script that handles all steps from input validation to output validation.

### Purpose:
- Complete end-to-end processing pipeline
- Includes preprocessing, UVR separation, and output validation
- Audio processing specifically prepared for DiffSinger

### How to Use:
```bash
python run_uvr_for_diffsinger.py --input_dir input_directory --output_dir output_directory [--model_name model_name] [--vocals_only] [--use_gpu] [--skip_preprocessing]
```

### Example:
```bash
# Standard processing
python run_uvr_for_diffsinger.py --input_dir ./audio_input --output_dir ./diffsinger_ready

# Vocals only with GPU acceleration
python run_uvr_for_diffsinger.py --input_dir ./audio_input --output_dir ./diffsinger_ready --vocals_only --use_gpu

# Skip preprocessing (if audio already meets requirements)
python run_uvr_for_diffsinger.py --input_dir ./audio_input --output_dir ./diffsinger_ready --skip_preprocessing
```

### Features:
- Step 1: Validates input directory for audio files
- Step 2: Preprocesses audio (resamples to 44.1kHz, converts to mono)
- Step 3: Runs UVR separation (with optimized settings)
- Step 4: Validates output for DiffSinger compliance
- Step 5: Cleans up temporary files

## Recommended Workflow

For DiffSinger preprocessing, the recommended workflow is:

1. **Initial Check**: Use automatic_uvr_optimizer.py to validate your audio files
   ```bash
   python automatic_uvr_optimizer.py --input_dir ./your_audio --output_dir ./temp --validate_only
   ```

2. **Run Complete Processing**: Use run_uvr_for_diffsinger.py for end-to-end processing
   ```bash
   python run_uvr_for_diffsinger.py --input_dir ./your_audio --output_dir ./diffsinger_ready --vocals_only --use_gpu
   ```

3. **For More Control**: Use optimize_uvr_for_diffsinger.py to get detailed parameter suggestions
   ```bash
   python optimize_uvr_for_diffsinger.py --input_dir ./your_audio --output_dir ./output --workflow advanced
   ```

## Notes

- Ensure required dependencies are installed (librosa, soundfile, numpy, etc.)
- Processing may take time for large audio files
- If using GPU, ensure you have CUDA-compatible PyTorch installed
- Output audio files will be 44.1kHz sample rate, mono WAV format, fully compliant with DiffSinger requirements

These three scripts work together to provide a complete solution for processing audio into high-quality, separated vocals suitable for DiffSinger training.
