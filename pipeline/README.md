# DSF to WAV Converter

This script recursively converts all `.dsf` files in a specified directory to `.wav` files, preserving the original directory structure in the output. This is useful for preparing audio files for DiffSinger which requires standard audio formats like WAV.

## Purpose

DiffSinger requires audio files in standard formats like WAV for training and synthesis. DSF (DSD Stream File) is a high-resolution audio format that needs to be converted to a format that DiffSinger can process. This script automates the conversion process while maintaining the directory structure.

## Prerequisites

Before using this script, you need to install the required dependencies:

```bash
# Install required Python packages
pip install pydub

# Install ffmpeg (required for audio conversion)
# On macOS:
brew install ffmpeg

# On Ubuntu/Debian:
sudo apt update
sudo apt install ffmpeg

# On Windows (using Chocolatey):
choco install ffmpeg
```

## Usage

Basic usage:
```bash
python dsf_to_wav_converter.py -i /path/to/input/dir -o /path/to/output/dir
```

With custom sample rate:
```bash
python dsf_to_wav_converter.py --input /path/to/input --output /path/to/output --sample-rate 44100
```

With custom bit depth:
```bash
python dsf_to_wav_converter.py -i /path/to/input -o /path/to/output --bit-depth s24
```

## Options

- `-i, --input`: Input directory containing `.dsf` files (required)
- `-o, --output`: Output directory where `.wav` files will be saved (required)
- `--sample-rate`: Sample rate for output WAV files (default: 44100)
- `--bit-depth`: Bit depth for output WAV files (default: s16, options: s16, s24, s32, flt, dbl)

## Notes

- The script preserves the original directory structure in the output
- Output files are saved with the same relative paths as the input files
- The script uses ffmpeg for the actual conversion process
- By default, the script sets the sample rate to 44.1kHz which is commonly used in audio processing