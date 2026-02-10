#!/usr/bin/env python3
"""
Script to recursively process files in a directory and move them to subfolders
based on the prefix before the first or second dash in the filename.
"""

import os
import shutil
import argparse
from pathlib import Path


def extract_prefix(filename):
    """
    Extract the prefix from a filename based on dash separators.
    
    Args:
        filename (str): The filename to process
        
    Returns:
        str: The extracted prefix
    """
    name = Path(filename).stem
    parts = name.split('-')
    
    if len(parts) >= 2:
        # Return part before the second dash (first two parts joined)
        return '-'.join(parts[:2])
    elif len(parts) >= 1:
        # Return part before the first dash (first part only)
        return parts[0]
    else:
        # No dashes found, return the whole name
        return name


def find_target_folder(destination_dir, folder_name):
    """
    Recursively search for a subfolder with the given name in the destination directory.
    
    Args:
        destination_dir (str): The directory to search in
        folder_name (str): The name of the folder to find
        
    Returns:
        Path: The path to the found folder, or None if not found
    """
    destination_path = Path(destination_dir)
    
    # First, check if the folder exists directly in the destination
    direct_path = destination_path / folder_name
    if direct_path.is_dir():
        return direct_path
    
    # If not found directly, recursively search in subdirectories
    for root, dirs, files in os.walk(destination_path):
        if folder_name in dirs:
            return Path(root) / folder_name
    
    return None


def process_files(source_dir, destination_dir):
    """
    Process all files in the source directory and move them to appropriate subfolders.

    Args:
        source_dir (str): The directory containing files to process
        destination_dir (str): The directory where files should be moved
    """
    source_path = Path(source_dir)
    destination_path = Path(destination_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    if not destination_path.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {destination_dir}")

    if not source_path.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    if not destination_path.is_dir():
        raise NotADirectoryError(f"Destination path is not a directory: {destination_dir}")

    # Walk through all files in the source directory recursively
    for root, dirs, files in os.walk(source_path):
        for file in files:
            source_file_path = Path(root) / file

            # Skip hidden files or non-files
            if source_file_path.name.startswith('.') or not source_file_path.is_file():
                continue

            try:
                # Extract the prefix from the filename
                prefix = extract_prefix(file)

                if not prefix.strip():  # Skip if prefix is empty or only whitespace
                    print(f"Warning: Empty prefix extracted from file '{file}', skipping...")
                    continue

                # Find the target folder in the destination directory
                target_folder = find_target_folder(destination_dir, prefix)

                if target_folder is None:
                    print(f"Target folder '{prefix}' not found in destination directory '{destination_dir}'. "
                          f"Creating new folder.")

                    # Create the target folder in the destination directory
                    target_folder = Path(destination_dir) / prefix
                    target_folder.mkdir(parents=True, exist_ok=True)

                # Check if file already exists in target location
                destination_file_path = target_folder / file
                if destination_file_path.exists():
                    print(f"Warning: File '{destination_file_path}' already exists. Skipping...")
                    continue

                # Move the file to the target folder
                shutil.move(str(source_file_path), str(destination_file_path))

                print(f"Moved '{source_file_path}' to '{destination_file_path}'")

            except PermissionError:
                print(f"Permission denied when trying to move '{source_file_path}'")
            except OSError as e:
                print(f"OS error occurred while moving '{source_file_path}': {e}")
            except Exception as e:
                print(f"Unexpected error occurred while moving '{source_file_path}': {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively process files and move them to subfolders based on filename prefixes."
    )
    parser.add_argument(
        "source_dir",
        help="Source directory containing files to process"
    )
    parser.add_argument(
        "destination_dir",
        help="Destination directory where files should be moved"
    )
    
    args = parser.parse_args()
    
    try:
        process_files(args.source_dir, args.destination_dir)
        print("File processing completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()