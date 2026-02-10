#!/usr/bin/env python3
"""
Recursively count audio files and calculate their total duration in a specified folder.
Supports various audio formats and provides detailed statistics.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported audio file extensions
AUDIO_EXTENSIONS = {
    '.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma', 
    '.aiff', '.aif', '.au', '.snd', '.opus', '.webm', '.amr'
}

def get_audio_duration(file_path: Path) -> float:
    """
    Get duration of an audio file using ffprobe.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Duration in seconds, or 0 if unable to determine
    """
    try:
        import subprocess
        import json
        
        # Use ffprobe to get audio duration
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(file_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            
            # Try to get duration from format section first
            if 'format' in data and 'duration' in data['format']:
                return float(data['format']['duration'])
            
            # If not available, try to get from streams
            if 'streams' in data and len(data['streams']) > 0:
                for stream in data['streams']:
                    if stream.get('codec_type') == 'audio' and 'duration' in stream:
                        return float(stream['duration'])
        
        logger.warning(f"Could not determine duration for {file_path}")
        return 0
        
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout while processing {file_path}")
        return 0
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return 0

def scan_audio_files(root_path: Path, recursive: bool = True) -> List[Tuple[Path, float]]:
    """
    Recursively scan for audio files and get their durations.
    
    Args:
        root_path: Root directory to scan
        recursive: Whether to scan subdirectories recursively
        
    Returns:
        List of tuples containing (file_path, duration_in_seconds)
    """
    audio_files = []
    
    try:
        if recursive:
            # Walk through all subdirectories
            for root, dirs, files in os.walk(root_path):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                        duration = get_audio_duration(file_path)
                        audio_files.append((file_path, duration))
        else:
            # Only scan the root directory
            for item in root_path.iterdir():
                if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS:
                    duration = get_audio_duration(item)
                    audio_files.append((item, duration))
                    
    except PermissionError:
        logger.error(f"Permission denied accessing {root_path}")
    except Exception as e:
        logger.error(f"Error scanning directory {root_path}: {e}")
    
    return audio_files

def move_files(file_list: List[Tuple[Path, float]], source_folder: Path, target_folder: str, description: str) -> Tuple[int, List[Tuple[Path, str]]]:
    """
    Move files to target folder while preserving directory structure.
    
    Args:
        file_list: List of (file_path, duration) tuples to move
        source_folder: Source folder path
        target_folder: Target folder path
        description: Description for logging
        
    Returns:
        Tuple of (moved_count, failed_moves_list)
    """
    move_folder = Path(target_folder)
    try:
        # Create target folder if it doesn't exist
        move_folder.mkdir(parents=True, exist_ok=True)
        
        moved_count = 0
        failed_moves = []
        
        print(f"\nMoving {description.lower()} to: {move_folder}")
        print("-" * 50)
        
        for file_path, duration in file_list:
            try:
                # Preserve relative directory structure
                relative_path = file_path.relative_to(source_folder)
                target_path = move_folder / relative_path
                
                # Create subdirectories if needed
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Move the file
                file_path.rename(target_path)
                moved_count += 1
                print(f"Moved: {relative_path} -> {target_path.name}")
                
            except Exception as e:
                failed_moves.append((file_path, str(e)))
                logger.error(f"Failed to move {file_path}: {e}")
        
        return moved_count, failed_moves
        
    except Exception as e:
        logger.error(f"Failed to create or access move folder {move_folder}: {e}")
        print(f"\nError: Could not access target folder {move_folder}")
        return 0, [(Path(), str(e))]

def log_move_results(moved_count: int, failed_moves: List[Tuple[Path, str]], folder_path: str):
    """
    Log the results of file moving operations.
    """
    print(f"\nMove operation completed!")
    print(f"Successfully moved: {moved_count} files")
    
    if failed_moves:
        print(f"Failed to move: {len(failed_moves)} files")
        for file_path, error in failed_moves:
            if file_path.exists():
                print(f"  {file_path.name}: {error}")
    
    logger.info(f"Moved {moved_count} files to {folder_path}")

def handle_empty_result(folder_path: str, file_type: str, threshold: float):
    """
    Handle case when no files match the criteria.
    """
    if folder_path:
        try:
            empty_file = Path(folder_path) / f"no_{file_type}_files.txt"
            empty_file.parent.mkdir(parents=True, exist_ok=True)
            with open(empty_file, 'w', encoding='utf-8') as f:
                f.write(f"No audio files found {file_type}er than {threshold} seconds\n")
            print(f"\nEmpty result indicator created: {empty_file}")
        except Exception as e:
            logger.error(f"Failed to create empty result file: {e}")

def format_duration(seconds: float) -> str:
    """
    Convert seconds to human-readable format (HH:MM:SS).
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def get_file_stats(audio_files: List[Tuple[Path, float]], min_duration: float = None, max_duration: float = None) -> Dict:
    """
    Calculate statistics from audio files list.
    
    Args:
        audio_files: List of (file_path, duration) tuples
        min_duration: Minimum duration threshold for long files
        
    Returns:
        Dictionary containing statistics
    """
    if not audio_files:
        return {
            'total_files': 0,
            'total_duration': 0,
            'average_duration': 0,
            'longest_file': None,
            'shortest_file': None,
            'extension_counts': {}
        }
    
    durations = [duration for _, duration in audio_files]
    file_paths = [str(path) for path, _ in audio_files]
    
    # Extension counts
    extension_counts = {}
    for path, _ in audio_files:
        ext = path.suffix.lower()
        extension_counts[ext] = extension_counts.get(ext, 0) + 1
    
    # Find files longer than minimum duration
    long_files = []
    if min_duration is not None:
        long_files = [(path, duration) for path, duration in audio_files if duration > min_duration]
    
    # Find files shorter than maximum duration
    short_files = []
    if max_duration is not None:
        short_files = [(path, duration) for path, duration in audio_files if duration < max_duration]
    
    stats = {
        'total_files': len(audio_files),
        'total_duration': sum(durations),
        'average_duration': sum(durations) / len(durations),
        'longest_file': max(zip(file_paths, durations), key=lambda x: x[1]),
        'shortest_file': min(zip(file_paths, durations), key=lambda x: x[1]),
        'extension_counts': extension_counts,
        'long_files': long_files,
        'short_files': short_files,
        'min_duration': min_duration,
        'max_duration': max_duration
    }
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Count audio files and calculate total duration recursively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python count_audio_files.py /path/to/audio/folder
  python count_audio_files.py /path/to/audio/folder --non-recursive
  python count_audio_files.py /path/to/audio/folder --min-duration 60 --long-folder /path/to/long_files
  python count_audio_files.py /path/to/audio/folder --max-duration 10 --short-folder /path/to/short_files
  python count_audio_files.py /path/to/audio/folder --min-duration 60 --max-duration 10 --long-folder /path/to/long_files --short-folder /path/to/short_files
  python count_audio_files.py /path/to/audio/folder --verbose
        """
    )
    
    parser.add_argument(
        'folder_path',
        help='Path to the folder containing audio files'
    )
    
    parser.add_argument(
        '--non-recursive', '-n',
        action='store_true',
        help='Only scan the specified folder, not subdirectories'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information for each file'
    )
    
    parser.add_argument(
        '--extensions', '-e',
        nargs='+',
        help=f'Specific audio extensions to count (default: all common formats)'
    )
    
    parser.add_argument(
        '--min-duration', '-d',
        type=float,
        help='Minimum duration in seconds. Files longer than this will be moved to long-folder.'
    )
    
    parser.add_argument(
        '--max-duration', '-x',
        type=float,
        help='Maximum duration in seconds. Files shorter than this will be moved to short-folder.'
    )
    
    parser.add_argument(
        '--long-folder', '-l',
        help='Target folder path for moving files longer than min-duration'
    )
    
    parser.add_argument(
        '--short-folder', '-s',
        help='Target folder path for moving files shorter than max-duration'
    )
    
    args = parser.parse_args()
    
    # Validate folder path
    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        logger.error(f"Folder does not exist: {folder_path}")
        return 1
    
    if not folder_path.is_dir():
        logger.error(f"Path is not a directory: {folder_path}")
        return 1
    
    # Update supported extensions if specified
    if args.extensions:
        global AUDIO_EXTENSIONS
        AUDIO_EXTENSIONS = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                          for ext in args.extensions}
        logger.info(f"Using custom extensions: {AUDIO_EXTENSIONS}")
    
    logger.info(f"Scanning folder: {folder_path}")
    logger.info(f"Recursive mode: {'Yes' if not args.non_recursive else 'No'}")
    
    # Scan for audio files
    audio_files = scan_audio_files(folder_path, not args.non_recursive)
    
    if not audio_files:
        logger.info("No audio files found.")
        return 0
    
    # Calculate statistics
    stats = get_file_stats(audio_files, args.min_duration)
    
    # Display results
    print("\n" + "="*60)
    print("AUDIO FILES STATISTICS")
    print("="*60)
    print(f"Folder scanned: {folder_path}")
    print(f"Scan mode: {'Recursive' if not args.non_recursive else 'Non-recursive'}")
    print(f"Total audio files found: {stats['total_files']}")
    print(f"Total duration: {format_duration(stats['total_duration'])} ({stats['total_duration']:.2f} seconds)")
    print(f"Average duration: {format_duration(stats['average_duration'])} ({stats['average_duration']:.2f} seconds)")
    
    if stats['longest_file']:
        print(f"Longest file: {os.path.basename(stats['longest_file'][0])} ({format_duration(stats['longest_file'][1])})")
        print(f"Shortest file: {os.path.basename(stats['shortest_file'][0])} ({format_duration(stats['shortest_file'][1])})")
    
    # Extension breakdown
    print(f"\nFile format breakdown:")
    for ext, count in sorted(stats['extension_counts'].items()):
        percentage = (count / stats['total_files']) * 100
        print(f"  {ext}: {count} files ({percentage:.1f}%)")
    
    # Handle long files if minimum duration is specified
    if args.min_duration is not None:
        print(f"\nFiles longer than {args.min_duration} seconds ({format_duration(args.min_duration)}):")
        print(f"Found {len(stats['long_files'])} files exceeding the threshold")
        
        if stats['long_files']:
            # Sort by duration descending
            sorted_long_files = sorted(stats['long_files'], key=lambda x: x[1], reverse=True)
            
            # Display long files
            for file_path, duration in sorted_long_files:
                relative_path = file_path.relative_to(folder_path)
                print(f"  {relative_path} | {format_duration(duration)} | {duration:.2f}s")
            
            # Move long files to target folder
            if args.long_folder:
                moved_count, failed_moves = move_files(sorted_long_files, folder_path, args.long_folder, "Long files")
                log_move_results(moved_count, failed_moves, args.long_folder)
            else:
                print("\nNote: Use --long-folder option to actually move long files")
        else:
            print("  No files exceed the specified duration threshold.")
            handle_empty_result(args.long_folder, "long", args.min_duration)
    
    # Handle short files if maximum duration is specified
    if args.max_duration is not None:
        print(f"\nFiles shorter than {args.max_duration} seconds ({format_duration(args.max_duration)}):")
        print(f"Found {len(stats['short_files'])} files below the threshold")
        
        if stats['short_files']:
            # Sort by duration ascending
            sorted_short_files = sorted(stats['short_files'], key=lambda x: x[1])
            
            # Display short files
            for file_path, duration in sorted_short_files:
                relative_path = file_path.relative_to(folder_path)
                print(f"  {relative_path} | {format_duration(duration)} | {duration:.2f}s")
            
            # Move short files to target folder
            if args.short_folder:
                moved_count, failed_moves = move_files(sorted_short_files, folder_path, args.short_folder, "Short files")
                log_move_results(moved_count, failed_moves, args.short_folder)
            else:
                print("\nNote: Use --short-folder option to actually move short files")
        else:
            print("  No files are shorter than the specified duration threshold.")
            handle_empty_result(args.short_folder, "short", args.max_duration)

    # Verbose output (showing all files with movement indicators)
    if args.verbose and audio_files:
        print(f"\nDetailed file list (all files):")
        print("-" * 80)
        for file_path, duration in sorted(audio_files, key=lambda x: x[1], reverse=True):
            relative_path = file_path.relative_to(folder_path)
            long_marker = "LONG" if args.min_duration and duration > args.min_duration else "    "
            short_marker = "SHORT" if args.max_duration and duration < args.max_duration else "     "
            long_status = "[TO MOVE-LONG]" if args.min_duration and duration > args.min_duration and args.long_folder else "             "
            short_status = "[TO MOVE-SHORT]" if args.max_duration and duration < args.max_duration and args.short_folder else "              "
            print(f"{long_marker} {short_marker} {long_status} {short_status} {relative_path} | {format_duration(duration)} | {duration:.2f}s")
    
    print("="*60)
    
    return 0

if __name__ == "__main__":
    exit(main())