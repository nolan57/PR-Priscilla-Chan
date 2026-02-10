# SingingVocoders Project Overview

## Project Description

SingingVocoders is a collection of neural vocoders specifically designed for singing voice synthesis tasks. The project provides implementations of various state-of-the-art vocoder architectures including NSF-HiFiGAN, DDSP, UnivNet, and HiFiVAE models. It's built on PyTorch and uses PyTorch Lightning for training orchestration.

## Key Features

- Multiple neural vocoder architectures (NSF-HiFiGAN, DDSP, UnivNet, HiFiVAE)
- Preprocessing pipeline for audio data
- Training scripts with configurable parameters
- Checkpoint export functionality for use in other projects
- Online data augmentation capabilities
- Support for fine-tuning pre-trained models

## Architecture

The project is organized into several key directories:

- `configs/` - YAML configuration files for different models and training scenarios
- `models/` - Model definitions and architectures
- `modules/` - Reusable components like discriminators, loss functions, etc.
- `preprocess/` - Data preprocessing utilities
- `training/` - Training task definitions
- `utils/` - Utility functions for configuration, audio processing, etc.

## Building and Running

### Prerequisites
- Python 3.x
- PyTorch
- PyTorch Lightning
- Torchaudio
- Additional dependencies specified in requirements (if any)

### Preprocessing
```sh
python process.py --config (your config path) --num_cpu (Number of cpu threads used in preprocessing) --strx (1 for a forced absolute path 0 for a relative path)
```

### Training
```sh
python train.py --config (your config path) --exp_name (your ckpt name) --work_dir (working directory, optional)
```

### Export Checkpoint
```sh
python export_ckpt.py --ckpt_path (your ckpt path) --save_path (output ckpt path) --work_dir (working directory, optional)
```

## Configuration Files

The project uses a hierarchical configuration system:

- `base.yaml` - Base configuration with common parameters
- `base_hifi.yaml` - Extends base with HiFi-specific parameters
- `ft_hifigan.yaml` - Fine-tuning specific configurations
- Model-specific configs like `ddsp_univnet.yaml`, `univnet.yaml`, etc.

## Models Supported

- NSF-HiFiGAN (Neural Source Filter HiFi-GAN)
- DDSP (Differentiable Digital Signal Processing) models
- UnivNet
- HiFiVAE (HiFi Variational Autoencoder)
- Mixed architectures like DDSP-UnivNet

## Fine-tuning

The project supports fine-tuning pre-trained models with specific configurations. For fine-tuning NSF-HiFiGAN, download pre-trained weights from releases and adjust the `finetune_ckpt_path` in the configuration file.

## Integration

Exported checkpoints can be used in various singing voice synthesis projects like:
- DDSP-SVC
- Diffusion-SVC
- so-vits-svc
- DiffSinger (openvpi)
- OpenUtau (via ONNX conversion)

## Development Conventions

- Configuration-driven approach for model and training parameters
- PyTorch Lightning for training orchestration
- Modular design allowing easy extension of new architectures
- Comprehensive preprocessing pipeline for audio data

## Important Notes

- The project recommends using 44.1kHz sample rate audio for fine-tuning
- Online data augmentation is available but may affect sound quality
- Configuration inheritance follows the pattern: base.yaml → base_hifi.yaml → ft_hifigan.yaml
- The project uses a crop_mel_frames parameter that affects GPU memory usage