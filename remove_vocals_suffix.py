#!/usr/bin/env python3
"""
Script to recursively remove '_(Vocals)' suffix from all files in the vr_wavs directory.
"""

import os
import re


def remove_vocals_suffix(directory):
    """
    Recursively removes '_(Vocals)' suffix from all files in the given directory.
    
    Args:
        directory (str): The directory to process recursively
    """
    # Counter for renamed files
    renamed_count = 0
    
    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check if the filename ends with '_(Vocals)' followed by an extension
            if re.search(r'_\(Vocals\)(\.[^.]+)$', filename):
                # Construct the old and new file paths
                old_path = os.path.join(root, filename)
                
                # Create the new filename by removing '_(Vocals)'
                new_filename = re.sub(r'_\(Vocals\)(\.[^.]+)$', r'\1', filename)
                new_path = os.path.join(root, new_filename)
                
                # Rename the file
                os.rename(old_path, new_path)
                print(f'Renamed: {filename} -> {new_filename}')
                
                # Increment counter
                renamed_count += 1
    
    print(f'\nTotal files renamed: {renamed_count}')


if __name__ == '__main__':
    # Define the directory to process
    vr_wavs_dir = 'Montreal-Forced-Aligner/datasets/vr_wavs'
    
    # Check if directory exists
    if not os.path.exists(vr_wavs_dir):
        print(f'Directory does not exist: {vr_wavs_dir}')
    else:
        print(f'Starting to process directory: {vr_wavs_dir}')
        remove_vocals_suffix(vr_wavs_dir)
        print('Processing complete!')