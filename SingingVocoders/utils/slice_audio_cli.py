#!/usr/bin/env python3
"""
Script to recursively process all subdirectories in an input directory using dataset-tools-cli slice-audio command.
"""

import os
import subprocess
import argparse
from pathlib import Path


def process_directories(dataset_tools_cli_path, input_dir, output_base_dir, **kwargs):
    """
    Recursively process all subdirectories in a directory using dataset-tools-cli slice-audio.
    For each directory containing WAV files, process all WAV files in that directory,
    maintaining the directory structure in the output.
    
    Args:
        dataset_tools_cli_path (str): Path to the dataset-tools-cli executable
        input_dir (str): Input directory containing subdirectories to process
        output_base_dir (str): Base output directory for processed files
        **kwargs: Additional options for slice-audio command
    """
    input_path = Path(input_dir)
    output_path = Path(output_base_dir)
    
    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all directories that contain WAV files
    for current_dir in [input_path] + [x for x in input_path.rglob("*") if x.is_dir()]:
        # Find only the WAV files in this specific directory (not subdirectories)
        wav_files = [f for f in current_dir.iterdir() if f.suffix.lower() == '.wav']
        
        if wav_files:
            print(f"Processing directory: {current_dir} ({len(wav_files)} WAV files)")
            
            # Calculate relative path from input directory to preserve folder structure
            if current_dir == input_path:
                relative_path = Path(".")
            else:
                relative_path = current_dir.relative_to(input_path)
            output_subdir = output_path / relative_path
            
            # Create the corresponding subdirectory in output
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Process each WAV file in this directory
            for wav_file in wav_files:
                print(f"  Processing file: {wav_file.name}")
                
                # Build the command for this specific file
                cmd = [dataset_tools_cli_path, "slice-audio", str(wav_file)]
                
                # Add optional parameters
                if kwargs.get('threshold'):
                    cmd.extend(["-t", str(kwargs['threshold'])])
                if kwargs.get('min_length'):
                    cmd.extend(["-l", str(kwargs['min_length'])])
                if kwargs.get('min_interval'):
                    cmd.extend(["-i", str(kwargs['min_interval'])])
                if kwargs.get('hop_size'):
                    cmd.extend(["-s", str(kwargs['hop_size'])])
                if kwargs.get('max_sil_kept'):
                    cmd.extend(["-m", str(kwargs['max_sil_kept'])])
                if kwargs.get('output_format'):
                    cmd.extend(["-f", str(kwargs['output_format'])])
                if kwargs.get('digits'):
                    cmd.extend(["-d", str(kwargs['digits'])])
                
                # Set output directory to maintain the same subdirectory structure
                cmd.extend(["-o", str(output_subdir)])
                
                try:
                    # Execute the command
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    print(f"  Successfully processed: {wav_file.name}")
                    if result.stdout:
                        print(f"  Output: {result.stdout.strip()}")
                except subprocess.CalledProcessError as e:
                    print(f"  Error processing {wav_file.name}: {e}")
                    if e.stderr:
                        print(f"  Error output: {e.stderr}")
                except FileNotFoundError:
                    print("  Error: dataset-tools-cli command not found. Please check that the provided executable path is correct and the file is executable.")
                    return


def main():
    parser = argparse.ArgumentParser(
        description="Recursively process all subdirectories in an input directory using dataset-tools-cli slice-audio"
    )
    parser.add_argument("dataset_tools_cli_path", help="Path to the dataset-tools-cli executable")
    parser.add_argument("input_dir", help="Input directory containing subdirectories to process")
    parser.add_argument("output_dir", help="Base output directory for processed files")
    
    # Slice-audio specific options
    parser.add_argument("--threshold", type=int, default=-40, 
                       help="Threshold (dB) for silence detection (default: -40)")
    parser.add_argument("--min-length", dest="min_length", type=int, default=5000,
                       help="Minimum length (ms) for each audio (default: 5000)")
    parser.add_argument("--min-interval", dest="min_interval", type=int, default=300,
                       help="Minimum interval between slices (default: 300)")
    parser.add_argument("--hop-size", dest="hop_size", type=int, default=10,
                       help="Hop size (ms) for frame processing (default: 10)")
    parser.add_argument("--max-sil-kept", dest="max_sil_kept", type=int, default=500,
                       help="Maximum silence (ms) to keep (default: 500)")
    parser.add_argument("--output-format", dest="output_format", type=int, default=0,
                       help="Output format (0: 16-bit, 1: 24-bit, 2: 32-bit, 3: float32) (default: 0)")
    parser.add_argument("--digits", type=int, default=3,
                       help="Minimum digits for output file names (default: 3)")

    args = parser.parse_args()
    
    # Convert argparse namespace to dictionary
    options = {
        'threshold': args.threshold,
        'min_length': args.min_length,
        'min_interval': args.min_interval,
        'hop_size': args.hop_size,
        'max_sil_kept': args.max_sil_kept,
        'output_format': args.output_format,
        'digits': args.digits
    }
    
    process_directories(args.dataset_tools_cli_path, args.input_dir, args.output_dir, **options)


if __name__ == "__main__":
    main()