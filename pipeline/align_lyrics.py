#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lyrics Alignment for Cantonese Songs using MFA Python API
This script is designed to be compatible with the original align_lyrics.py but uses MFA Python API instead of command line
"""

import os
import sys
import logging
import argparse
import tempfile
import shutil
import time
import glob
import re
import random
import uuid
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Set environment variables before importing MFA modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Note: We no longer set a global MFA_ROOT_DIR here to avoid conflicts when processing multiple files
# Each process will set its own unique MFA_ROOT_DIR in process_single_file function

# Check if MFA Python API is available
HAS_MFA_API = False
try:
    import montreal_forced_aligner as mfa
    HAS_MFA_API = True
except ImportError:
    pass

# Check and fix numpy/numba compatibility issue
def check_and_fix_numpy_version():
    """
    Check if numpy version is compatible with numba and fix if needed
    """
    try:
        import numpy as np
        import numba
        # Get numpy version
        numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
        # Numba requires numpy <= 2.3
        if numpy_version > (2, 3):
            print(f"Warning: NumPy version {np.__version__} is incompatible with Numba. Installing compatible version...")
            # Try to downgrade numpy to a compatible version
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "numpy<=2.3"])
            print("Compatible NumPy version installed. Restarting...")
            # Restart the script to load the correct version
            os.execv(sys.executable, ['python'] + sys.argv)
    except ImportError:
        print("Could not import numpy or numba to check versions")
    except Exception as e:
        print(f"Error trying to fix numpy/numba compatibility: {e}")

# Run compatibility check early
check_and_fix_numpy_version()

# Set up logging
logger = logging.getLogger(__name__)

def preprocess_chinese_text(input_file, output_file):
    """
    Convert Chinese text to a format suitable for MFA processing
    
    Args:
        input_file (str): Input text file path
        output_file (str): Output text file path
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                    
                # Remove punctuation marks, keep Chinese characters and spaces
                cleaned_line = re.sub(r'[^\u4e00-\u9fff\s]', '', line)
                cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
                
                # Separate each character with spaces, which is a common way for MFA to process Chinese
                chars = list(cleaned_line.replace(' ', ''))
                spaced_text = ' '.join(chars)
                
                outfile.write(spaced_text + '\n')
    except FileNotFoundError as e:
        print(f"File not found: {input_file}")
        raise e
    except IOError as e:
        print(f"IO error occurred when processing file: {input_file}")
        raise e
    except Exception as e:
        print(f"Unexpected error in preprocess_chinese_text: {str(e)}")
        raise e

def preprocess_lyrics():
    """
    Process lyrics files:
    1. Read all WAV files from dataset/wavs recursively
    2. For each WAV file, find the corresponding lyrics file in dataset/lyrics with same directory structure
    3. Process the lyrics and save to corpus/ directory with same directory structure and .txt extension
    """
    try:
        # Get project root directory - same as original script
        project_root = Path(__file__).resolve().parent.parent
        mfa_root = os.path.join(project_root, "Montreal-Forced-Aligner")
        datasets_root = os.path.join(mfa_root, "datasets")
        corpus_dir = os.path.join(datasets_root, "corpus")
        wavs_dir = os.path.join(datasets_root, "vr_wavs")  # Changed to vr_wavs to match the original processing path
        lyrics_dir = os.path.join(datasets_root, "lyrics")

        os.makedirs(datasets_root, exist_ok=True)
        os.makedirs(wavs_dir, exist_ok=True)
        os.makedirs(lyrics_dir, exist_ok=True)
        os.makedirs(corpus_dir, exist_ok=True)
        
        processed_files = []
        # Get all WAV files recursively
        wav_files = glob.glob(os.path.join(wavs_dir, "**", "*.wav"), recursive=True)
        
        if not wav_files:
            print("No WAV files found in dataset/wavs directory")
            return []
        
        print(f"Found {len(wav_files)} WAV files")
        
        # Process each WAV file
        for wav_file in wav_files:
            try:
                # Get relative path from wavs_dir to maintain directory structure
                rel_path = os.path.relpath(wav_file, wavs_dir)
                # Get base name with directory structure but without extension
                basename_without_ext = os.path.splitext(rel_path)[0]
                # Get just the filename without directory for display
                display_name = os.path.basename(wav_file)
                
                # Find corresponding lyrics file with same directory structure
                lyrics_file = os.path.join(lyrics_dir, basename_without_ext + ".txt")
                
                if not os.path.exists(lyrics_file):
                    print(f"Warning: No lyrics file found for {rel_path}")
                    continue
                
                # Define output file path in corpus directory with same directory structure
                output_file = os.path.join(corpus_dir, basename_without_ext + ".txt")
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Process the lyrics
                print(f"Processing: {rel_path}")
                preprocess_chinese_text(lyrics_file, output_file)
                print(f"Saved processed lyrics to: {output_file}")
                processed_files.append((wav_file, output_file))
            except Exception as e:
                print(f"Error processing file {wav_file}: {str(e)}")
                continue
        
        return processed_files
    except Exception as e:
        print(f"Error in preprocess_lyrics: {str(e)}")
        return []
    

class ProgressCallback:
    """
    Progress callback for MFA alignment
    """
    def __init__(self, file_name):
        self.file_name = file_name
        self.last_progress = 0
        self.start_time = time.time()
    
    def update(self, progress):
        """
        Update progress
        
        Args:
            progress (float): Progress percentage (0-1)
        """
        current_progress = int(progress * 100)
        if current_progress > self.last_progress:
            self.last_progress = current_progress
            elapsed = time.time() - self.start_time
            logger.info(f"[{self.file_name}] Progress: {current_progress}% ({elapsed:.1f}s)")


class BatchProgressTracker:
    """
    Track progress of batch alignment
    """
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.succeeded = 0
        self.failed = 0
        self.start_time = time.time()
    
    def update(self, success):
        """
        Update progress
        
        Args:
            success (bool): Whether the alignment succeeded
        """
        self.completed += 1
        if success:
            self.succeeded += 1
        else:
            self.failed += 1
        
        elapsed = time.time() - self.start_time
        logger.info(f"Batch progress: {self.completed}/{self.total} ({elapsed:.1f}s)")


def run_mfa_in_separate_process(args):
    """
    Function to run MFA in a separate process to avoid conflicts
    """
    wav_file, txt_file, output_dir = args
    
    try:
        # Import necessary modules inside the function for multiprocessing
        import os
        import shutil
        from pathlib import Path
        import tempfile
        import uuid
        from montreal_forced_aligner.alignment import PretrainedAligner
        
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Use the same directory structure as in preprocess_lyrics function
        wavs_dir = os.path.join(project_root, "Montreal-Forced-Aligner", "datasets", "vr_wavs")
        
        # Get relative path from wavs_dir to maintain directory structure
        # This is crucial for maintaining the same directory structure as corpus
        rel_path = os.path.relpath(wav_file, wavs_dir)
        
        file_name = os.path.basename(wav_file)
        print(f"Processing {rel_path} with MFA Python API in subprocess...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get project root and model paths - using EXACT same paths as original script
        dict_path = os.path.join(project_root, "HKCantonese_models", "cv19_val_lexicon_v3.txt")
        
        # EXACT same acoustic model path as original script
        acoustic_model_path = os.path.join(project_root, "HKCantonese_models","pretrained_models", "acoustic_model_cv19_v3.zip")
        
        # Check if dictionary exists
        if not os.path.exists(dict_path):
            return False, file_name, f"Cantonese dictionary not found at {dict_path}"
        
        # Create unique temporary directory for this specific file to avoid MFA conflicts
        unique_id = str(uuid.uuid4())[:8]  # Short UUID for directory name
        temp_base_dir = os.path.join(tempfile.gettempdir(), "mfa_processing")
        os.makedirs(temp_base_dir, exist_ok=True)
        temp_dir = os.path.join(temp_base_dir, f"mfa_{unique_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create dedicated audio subdirectory for this process
        temp_audio_dir = os.path.join(temp_dir, "audio")
        os.makedirs(temp_audio_dir, exist_ok=True)
        
        # Create necessary directories in temporary audio folder to match original structure
        temp_wav_dir = os.path.join(temp_audio_dir, os.path.dirname(rel_path))
        os.makedirs(temp_wav_dir, exist_ok=True)
        
        # Create copy with same directory structure
        temp_wav = os.path.join(temp_audio_dir, rel_path)
        temp_txt = os.path.join(temp_audio_dir, os.path.splitext(rel_path)[0] + ".txt")
        
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(temp_wav), exist_ok=True)
            os.makedirs(os.path.dirname(temp_txt), exist_ok=True)
            
            shutil.copy2(wav_file, temp_wav)
            shutil.copy2(txt_file, temp_txt)
        except Exception as e:
            return False, file_name, f"Failed to copy files: {str(e)}"
        
        # Configure MFA settings exactly as in original script
        config = {
            "beam": 1000,  # Same as original script
            "retry_beam": 4000,  # Same as original script
            "clean": True,
            "num_jobs": 1
        }
        
        try:
            # Use MFA Python API for alignment
            print(f"[{rel_path}] Starting alignment with beam={config['beam']}, retry_beam={config['retry_beam']}")
            
            # Create aligner with unique temporary directory
            aligner = PretrainedAligner(
                corpus_directory=temp_audio_dir,
                dictionary_path=dict_path,
                acoustic_model_path=acoustic_model_path,
                beam=config['beam'],
                retry_beam=config['retry_beam'],
                output_directory=Path(temp_dir)  # Use unique temp directory as output directory to isolate database files
            )

            # Run alignment
            aligner.align()

            # Export TextGrids to the specified output directory
            print(f"[{rel_path}] Exporting TextGrid files...")
            # First export to temp directory, then move to final output directory to preserve directory structure
            temp_output_dir = Path(temp_dir) / "output_textgrids"
            temp_output_dir.mkdir(exist_ok=True)
            aligner.export_files(output_directory=temp_output_dir)

            # Move TextGrid files from temp to final output directory while preserving directory structure
            for temp_textgrid in temp_output_dir.rglob("*.TextGrid"):
                # Calculate relative path from temp output directory
                rel_textgrid_path = temp_textgrid.relative_to(temp_output_dir)
                # Construct final destination path
                final_textgrid_path = Path(output_dir) / rel_textgrid_path
                # Create parent directories if needed
                final_textgrid_path.parent.mkdir(parents=True, exist_ok=True)
                # Move the file
                shutil.move(str(temp_textgrid), str(final_textgrid_path))

            # Locate the generated TextGrid with same directory structure
            textgrid_rel_path = os.path.splitext(rel_path)[0] + ".TextGrid"
            output_textgrid = os.path.join(output_dir, textgrid_rel_path)

            # Explicitly cleanup the aligner to release database connections
            aligner.cleanup()
            del aligner

            if not os.path.exists(output_textgrid):
                raise FileNotFoundError(f"TextGrid not found at expected path: {output_textgrid}")
            print(f"[{rel_path}] Alignment completed successfully")
            print(f"TextGrid saved with directory structure preserved: {output_textgrid}")
            return True, file_name, f"TextGrid created: {output_textgrid}"

        except Exception as e:
            print(f"[{rel_path}] Alignment failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, file_name, f"MFA alignment failed: {str(e)}"
            
    except Exception as e:
        if 'rel_path' in locals():
            path_info = rel_path
        else:
            path_info = wav_file
        print(f"Unexpected error processing {path_info}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, os.path.basename(wav_file), f"Unexpected error: {str(e)}"
    finally:
        # Clean up temporary files
        try:
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Failed to clean up temporary directory: {str(e)}")


def process_single_file(wav_file, txt_file, output_dir):
    """
    Process a single audio file with MFA Python API - compatible with original script
    
    Args:
        wav_file (str): Path to WAV file
        txt_file (str): Path to TXT file
        output_dir (str): Output directory for TextGrid
        
    Returns:
        tuple: (success, file_name, message)
    """
    # Directly call the subprocess function since multiprocessing.Process doesn't return values easily
    return run_mfa_in_separate_process((wav_file, txt_file, output_dir))

def batch_align(output_dir, workers=1):
    """
    Main alignment function with batch processing using MFA Python API
    
    Args:
        output_dir (str): Output directory for temporary files
        workers (int): Number of parallel workers
    
    Returns:
        bool: True if any alignment succeeded
    """
    try:
        print(f"Starting lyrics alignment process using MFA Python API...")
        print(f"Output directory: {output_dir}")
        print(f"Processing mode: Sequential (recommended for MFA API)")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Preprocess lyrics
        processed_files = preprocess_lyrics()
        
        if not processed_files:
            print("No files to process")
            return False
        
        print(f"Found {len(processed_files)} files to align")
        
        # Initialize progress tracker
        progress_tracker = BatchProgressTracker(len(processed_files))
        
        # Process files sequentially (safer with MFA API)
        success_count = 0
        print("Starting sequential processing...")
        for wav_file, txt_file in tqdm(processed_files, desc="Lyrics Alignment"):
            try:
                ok, name, msg = process_single_file(wav_file, txt_file, output_dir)
                if ok:
                    success_count += 1
                    print(f"✅ {name}: {msg}")
                else:
                    print(f"❌ {name}: {msg}")
                progress_tracker.update(ok)
            except KeyboardInterrupt:
                print("Processing interrupted by user")
                return success_count > 0
            except Exception as e:
                print(f"Error processing file {os.path.basename(wav_file)}: {str(e)}")
                progress_tracker.update(False)
        
        # Print summary
        total_time = time.time() - progress_tracker.start_time
        print(f"Batch processing complete in {total_time:.2f} seconds")
        print(f"Summary: Total={len(processed_files)}, Succeeded={success_count}, Failed={len(processed_files) - success_count}")
        
        return success_count > 0
    except Exception as e:
        print(f"Unexpected error in batch_align: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function with command line arguments - compatible with original script interface
    """
    project_root = Path(__file__).resolve().parent.parent
    mfa_root = os.path.join(project_root, "Montreal-Forced-Aligner")
    datasets_root = os.path.join(mfa_root, "datasets")
    output_dir = os.path.join(datasets_root, "textgrids")
    try:
        parser = argparse.ArgumentParser(description="Lyrics Alignment for Cantonese Songs using MFA Python API")
        
        # Use same default output path as original script would use
        parser.add_argument("--output", default=output_dir,
                           help="Output directory for TextGrid files")
        
        # Keep worker parameter for compatibility but ensure safe limits for MFA
        parser.add_argument("--workers", type=int, default=1,
                           help="Number of parallel workers (1 is recommended for MFA stability)")
        
        # Debug mode option
        parser.add_argument("--debug", action="store_true",
                           help="Enable debug output")
        
        args = parser.parse_args()
        
        # Log MFA API availability
        print(f"MFA Python API available: {HAS_MFA_API}")
        if not HAS_MFA_API:
            print("ERROR: MFA Python API not available. Please install Montreal Forced Aligner with Python API support.")
            print("You can install it with: pip install montreal-forced-aligner")
            return 1
        
        print(f"Using Montreal Forced Aligner Python API")
        print(f"Output directory: {args.output}")
        print(f"Processing mode: Sequential (optimal for MFA API)")
        
        # Run batch alignment
        start_time = time.time()
        success = batch_align(
            output_dir=args.output,
            workers=1  # Force sequential for MFA stability
        )
        total_time = time.time() - start_time
        
        # Print final summary - similar style to original
        print(f"\n=== Alignment Process Summary ===")
        print(f"Total processing time: {total_time:.2f} seconds")
        
        if success:
            print(f"✅ All alignment tasks completed successfully!")
            print(f"TextGrid files saved in: {args.output}")
            return 0
        else:
            print(f"❌ Some alignment tasks failed. Please check the output above.")
            return 1
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 130  # SIGINT
    except Exception as e:
        print(f"\nCritical error in main function: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())