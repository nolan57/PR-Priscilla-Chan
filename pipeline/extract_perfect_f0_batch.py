#!/usr/bin/env python3
"""
Batch F0 extractor for AI-separated vocal tracks.
Recursively processes all WAV files in a directory while maintaining directory structure.
"""

import os
import argparse
from pathlib import Path

# Import the main function from extract_perfect_f0
from extract_perfect_f0 import main

def process_directory(input_dir, output_dir, save_vad):
    """
    Recursively process all WAV files in input_dir and save outputs to output_dir
    while maintaining the directory structure.
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir) if output_dir else input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    total_files = 0
    processed_files = 0
    
    # Recursively walk through the input directory
    for root, _, files in os.walk(input_dir):
        # Filter for WAV files
        wav_files = [f for f in files if f.lower().endswith('.wav')]
        total_files += len(wav_files)
        
        for wav_file in wav_files:
            # Full path to the WAV file
            wav_path = os.path.join(root, wav_file)
            
            # Compute relative path from input_dir
            rel_path = os.path.relpath(wav_path, input_dir)
            
            # Extract directory component from relative path
            rel_dir = os.path.dirname(rel_path)
            
            # Create corresponding output directory
            output_subdir = os.path.join(output_dir, rel_dir)
            os.makedirs(output_subdir, exist_ok=True)
            
            print(f"Processing: {rel_path}")
            
            try:
                # Call the main function from extract_perfect_f0
                main(wav_path, output_dir=output_subdir, save_vad=save_vad)
                processed_files += 1
            except Exception as e:
                print(f"Error processing {rel_path}: {e}")
    
    print(f"\n✅ Batch processing completed!")
    print(f"Total files found: {total_files}")
    print(f"Successfully processed: {processed_files}")
    
    return processed_files

def main_batch():
    """
    Main function for batch processing.
    """
    parser = argparse.ArgumentParser(description="Batch F0 extractor for AI-separated vocals.")
    parser.add_argument("input_dir", type=str, help="Directory containing WAV files to process")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as input_dir)")
    parser.add_argument("--save_vad", action="store_true", help="Also save VAD mask (.vad.npy)")
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return 1
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a directory.")
        return 1
    
    # Process the directory
    process_directory(args.input_dir, args.output_dir, args.save_vad)
    
    return 0

if __name__ == "__main__":
    main_batch()
