#!/usr/bin/env python3
"""
DSF to WAV Converter Script

This script recursively converts all .dsf files in a specified directory to .wav files,
preserving the original directory structure in the output. This is useful for preparing
audio files for DiffSinger which requires standard audio formats like WAV.

Requirements:
- python-dsd (install with: pip install dsd)
- pydub (install with: pip install pydub)
- ffmpeg (for audio conversion)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def convert_dsf_to_wav(dsf_path, wav_path):
    """
    Convert a DSF file to WAV using ffmpeg.
    
    Args:
        dsf_path (str): Path to the input DSF file
        wav_path (str): Path to the output WAV file
    """
    try:
        # Use ffmpeg to convert DSF to WAV
        cmd = [
            'ffmpeg',
            '-i', dsf_path,
            '-ar', '44100',  # Set sample rate to 44.1kHz (required by DiffSinger)
            '-ac', '1',      # Set to mono (required by DiffSinger and MFA)
            '-sample_fmt', 's16',  # Set sample format to 16-bit
            wav_path,
            '-y'  # Overwrite output files without asking
        ]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"Successfully converted: {dsf_path} -> {wav_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {dsf_path}: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert all .dsf files in a directory to .wav files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i /path/to/input/dir -o /path/to/output/dir
  %(prog)s --input /path/to/input --output /path/to/output --sample-rate 44100
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Input directory containing .dsf files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output directory where .wav files will be saved'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=44100,
        help='Sample rate for output WAV files (default: 44100)'
    )
    
    parser.add_argument(
        '--bit-depth',
        type=str,
        default='s16',
        choices=['s16', 's24', 's32', 'flt', 'dbl'],
        help='Bit depth for output WAV files (default: s16)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .dsf files recursively
    dsf_files = list(input_dir.rglob('*.dsf'))
    
    if not dsf_files:
        print(f"No .dsf files found in '{input_dir}'")
        return
    
    print(f"Found {len(dsf_files)} .dsf files to convert...")
    
    success_count = 0
    failure_count = 0
    
    for dsf_path in dsf_files:
        # Calculate relative path from input directory
        relative_path = dsf_path.relative_to(input_dir)
        
        # Create corresponding path in output directory
        wav_path = output_dir / relative_path.with_suffix('.wav')
        
        # Create parent directories if they don't exist
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert the file
        if convert_dsf_to_wav(str(dsf_path), str(wav_path)):
            success_count += 1
        else:
            failure_count += 1
    
    print(f"\nConversion complete!")
    print(f"Successful conversions: {success_count}")
    print(f"Failed conversions: {failure_count}")


if __name__ == "__main__":
    main()