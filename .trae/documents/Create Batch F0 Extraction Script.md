# Batch F0 Extraction Script

## Overview
Create a new script `extract_perfect_f0_batch.py` that recursively processes all WAV files in a specified directory while maintaining directory structure consistency in the output.

## Implementation Steps

### 1. Create New Script File
- Create `extract_perfect_f0_batch.py` in the same directory as `extract_perfect_f0.py`

### 2. Import Dependencies
- Import necessary modules: `os`, `argparse`, `Path`, etc.
- Import `main` function from `extract_perfect_f0` for processing individual files

### 3. Command-Line Argument Parsing
- Add positional argument `input_dir` for the directory to process
- Keep existing `--output_dir` and `--save_vad` options
- Validate that `input_dir` exists and is a directory

### 4. Recursive Processing Logic
- Use `os.walk` to traverse `input_dir` recursively
- For each WAV file found:
  - Compute relative path from `input_dir`
  - Extract directory component from relative path
  - Create corresponding subdirectory in `output_dir`
  - Call `main` function with appropriate parameters

### 5. Progress Reporting
- Add print statements to indicate processing status
- Show total files processed at completion

## Key Features
- Maintains directory structure in output
- Processes all WAV files recursively
- Reuses existing single-file processing logic
- Provides clear progress feedback

## Usage Example
```bash
python extract_perfect_f0_batch.py /path/to/input/dir --output_dir /path/to/output/dir --save_vad
```