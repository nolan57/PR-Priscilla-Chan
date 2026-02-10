# PCS Directory - AI Audio Processing Collection

## Overview

This directory contains a collection of AI/ML projects focused on audio processing, singing voice synthesis, and speech technology. The main focus is on the DiffSinger ecosystem and related tools for creating and processing singing voice synthesis datasets.

## Project Components

### 1. DiffSinger
A singing voice synthesis system based on shallow diffusion mechanisms. This is the main project in the collection, offering:
- High-quality singing voice synthesis at 44.1kHz
- Variance models for controlling pitch, energy, breathiness, etc.
- Production-ready architecture with improved acoustic models
- Support for both acoustic and variance model training

**Key Features:**
- Cleaner code structure than the original implementation
- Better sound quality with 44.1kHz sampling rate
- Higher fidelity with improved models
- More controllability through variance parameters

### 2. dataset-tools
Comprehensive tools for DiffSinger dataset processing, including:
- **MinLabel**: Labeling tool for *.lab files with word transcriptions
- **SlurCutter**: MIDI sequence editor for *.ds files in variance model training
- **AudioSlicer**: Audio segmentation tool for cutting recordings into short segments
- ASR (Automatic Speech Recognition) integration for LyricFA

### 3. MakeDiffSinger
Pipelines and tools for building custom DiffSinger datasets:
- acoustic-forced-alignment: Dataset creation from scratch using MFA
- variance-temp-solution: Extension of acoustic datasets to variance datasets
- Standard dataset structure guidance

### 4. Montreal-Forced-Aligner (MFA)
A command-line utility for forced alignment of speech datasets using Kaldi:
- Used for aligning audio with transcriptions
- Essential for creating training datasets
- Cross-platform support (Windows, macOS, Linux)

### 5. LyricFA
A tool for automatic lyric forced alignment:
- Uses ASR to obtain syllables
- Matches text from lyrics
- Generates JSON for Minlabel preloading
- Supports both Chinese and English

### 6. HKCantonese_models
Pre-trained acoustic models for Hong Kong Cantonese:
- Multiple versions trained on Common Voice datasets
- Different phone set options (v1, v2, v3)
- Includes lexicons and dictionaries
- Tutorial scripts for Kaldi and MFA implementations

### 7. ultimatevocalremovergui
GUI application for vocal removal using state-of-the-art source separation:
- Uses MDX-Net and Demucs models
- Supports both CPU and GPU processing
- Cross-platform with optimized builds for Windows, macOS, and Linux
- Used for preprocessing audio data

### 8. pc_nsf_hifigan_44.1k_hop512_128bin_2025.02
A neural vocoder model (likely PC-NSF-HiFiGAN) for waveform reconstruction:
- 44.1kHz sampling rate
- Hop size of 512
- 128 bin configuration
- Used in the DiffSinger pipeline for audio generation

## Architecture & Workflow

The typical workflow involves:
1. **Data Preparation**: Using AudioSlicer and other tools to prepare audio segments
2. **Alignment**: Using MFA or LyricFA for forced alignment of audio and text
3. **Labeling**: Using MinLabel for transcription labeling
4. **Model Training**: Training acoustic and variance models with DiffSinger
5. **Synthesis**: Generating singing voice using trained models
6. **Post-processing**: Using tools like Ultimate Vocal Remover for audio enhancement

## Technical Stack

- **Python**: Primary language for most tools and training
- **PyTorch**: Deep learning framework for DiffSinger models
- **C++/Qt**: For GUI applications like dataset-tools
- **Kaldi**: Backend for Montreal Forced Aligner
- **ONNX Runtime**: For model inference in dataset-tools
- **FFmpeg**: Audio processing utilities

## Building and Running

### For DiffSinger:
- Follow the installation guide in the DiffSinger README
- Requires Python with PyTorch
- Various configuration schemas for different training scenarios

### For dataset-tools:
- Requires Qt 6.8+, C++17 compiler, CMake 3.17+, Python 3.8+
- Build using CMake with provided instructions
- Platform-specific setup for Windows/macOS/Linux

### For Montreal-Forced-Aligner:
- Available via conda: `conda install -c conda-forge montreal-forced-aligner`
- Or build from source with Python 3.11 and Kaldi dependencies

## Development Conventions

- Most projects follow Apache 2.0 license
- Clean code structure with modular components
- Configuration-driven workflows
- Emphasis on reproducible research and production deployment

## Use Cases

This collection is ideal for:
- Singing voice synthesis research
- Creating custom singing voice models
- Audio dataset preparation and processing
- Speech technology development
- Cantonese and other language-specific voice synthesis

## Modification Goals and Principles

### Goal
All modifications are aimed at achieving the goal of "reproducing any song with the specific singer's timbre, vocal techniques, and stylistic interpretation."

### Principles
All modifications adhere to the following principles:
- **Minimizing changes**: Making the smallest possible modifications to achieve the desired outcome
- **Preserving existing functionality and data**: Maintaining all current capabilities and data integrity as much as possible
- **Conducting comprehensive analysis**: Thoroughly analyzing the call chain and data input/output before making changes
- **Thorough understanding**: Ensuring complete comprehension of existing documentation and codebase before implementing modifications