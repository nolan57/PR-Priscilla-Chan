#!/usr/bin/env python3
"""
Convert validated dictionary files in HKCantonese_models to MFA-compatible format (improved version)
"""
import os
import re
import argparse

# from scripts.tools.download_models import project_root


def convert_dictionary(input_file, output_file, separator='\t'):
    """
    Convert dictionary files with probability values to MFA-compatible or DiffSinger-compatible format
    
    Args:
        input_file (str): Input dictionary file path
        output_file (str): Output dictionary file path
        separator (str): Separator to use (' ' for MFA, '\\t' for DiffSinger)
    
    Raises:
        FileNotFoundError: If input file does not exist
        PermissionError: If no permission to read/write files
        IOError: If any I/O error occurs during file processing
    """
    # Use try-except block to ensure proper resource release
    try:
        # Validate input file existence
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' does not exist")
        
        # Validate output directory is writable
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                raise IOError(f"Failed to create output directory '{output_dir}': {e}")
        
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            # Write basic silence and unknown word markers
            outfile.write(f"sil{separator}sil\n")
            outfile.write(f"<unk>{separator}spn\n")
            outfile.write(f"<oov>{separator}spn\n")
            outfile.write(f"<cutoff>{separator}spn\n")
            
            line_count = 0
            valid_entries = 0
            
            for line in infile:
                line_count += 1
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    # Split line content
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    
                    word = parts[0]
                    # For Cantonese dictionaries, we need to retain all phoneme information
                    # Format: word prob1 prob2 prob3 prob4 phone1 phone2 ...
                    if len(parts) > 5:  # Ensure there are enough parts
                        # Phonemes start from the 5th position (index 4)
                        phones = parts[5:]
                        # Filter out empty strings
                        phones = [p for p in phones if p]
                        if phones:  # Only write when there are phonemes
                            phone_str = ' '.join(phones)
                            outfile.write(f"{word}{separator}{phone_str}\n")
                            valid_entries += 1
                    else:
                        # If the format is simpler (just word and phonemes), handle it
                        phonemes = parts[1:]  # Everything after the first part is considered phonemes
                        if phonemes:
                            phone_str = ' '.join(phonemes)
                            outfile.write(f"{word}{separator}{phone_str}\n")
                            valid_entries += 1
                except Exception as e:
                    # Log error line but continue processing other lines
                    print(f"Warning: Error processing line {line_count}: {e}")
                    print(f"  Line content: {line}")
                    continue
        
        print(f"Successfully converted {valid_entries} entries from '{input_file}' to '{output_file}'")
        print(f"Used separator: '{repr(separator)}' ({'space' if separator == ' ' else 'tab'})")
        
    except Exception as e:
        # Ensure all exceptions are caught and handled
        print(f"Error in convert_dictionary: {e}")
        raise

def extract_words_from_corpus(corpus_dir, output_file):
    """
    Extract all words from the corpus and save to file
    
    Args:
        corpus_dir (str): Corpus directory path
        output_file (str): Output file path
    
    Raises:
        FileNotFoundError: If corpus directory does not exist
        PermissionError: If no permission to read/write files
        IOError: If any I/O error occurs during file processing
    """
    import os
    import glob
    
    words = set()
    
    try:
        # Validate corpus directory exists
        if not os.path.exists(corpus_dir):
            raise FileNotFoundError(f"Corpus directory '{corpus_dir}' does not exist")
        
        if not os.path.isdir(corpus_dir):
            raise NotADirectoryError(f"'{corpus_dir}' is not a valid directory")
        
        # Validate output directory is writable
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                raise IOError(f"Failed to create output directory '{output_dir}': {e}")
        
        # Find all text files
        txt_files = glob.glob(os.path.join(corpus_dir, "*.txt"))
        
        if not txt_files:
            print(f"Warning: No .txt files found in corpus directory '{corpus_dir}'")
            return
        
        processed_files = 0
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Remove punctuation marks and split words
                    # For Chinese, we split by characters
                    for char in content:
                        if char.strip() and not char.isspace() and char not in '，。？！；：""''（）【】《》':
                            words.add(char)
                processed_files += 1
            except Exception as e:
                # Log error file but continue processing other files
                print(f"Warning: Error processing file '{txt_file}': {e}")
                continue
        
        # Write word list
        with open(output_file, 'w', encoding='utf-8') as f:
            for word in sorted(words):
                f.write(word + '\n')
        
        print(f"Successfully extracted {len(words)} unique characters from {processed_files} files")
        print(f"Results saved to '{output_file}'")
        
    except Exception as e:
        # Ensure all exceptions are caught and handled
        print(f"Error in extract_words_from_corpus: {e}")
        raise

def main():
    """
    Main function to run the dictionary conversion
    """
    parser = argparse.ArgumentParser(description="Convert dictionary to MFA or DiffSinger compatible format")
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help="Input dictionary file path")
    parser.add_argument('--output', '-o', type=str, required=True, 
                        help="Output dictionary file path")
    parser.add_argument('--format', '-f', type=str, choices=['mfa', 'diffsinger'], 
                        default='diffsinger', 
                        help="Output format: 'mfa' for space-separated, 'diffsinger' for tab-separated (default: diffsinger)")
    
    args = parser.parse_args()
    
    try:
        input_file = args.input
        output_file = args.output
        separator = ' ' if args.format == 'mfa' else '\t'

        print(f"Starting dictionary conversion...")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print(f"Output format: {args.format} (separator: {repr(separator)})")
        
        # Convert the dictionary file
        convert_dictionary(input_file, output_file, separator)
        
        print(f"\n✅ Dictionary conversion completed successfully!")
        print(f"Output file saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please check the file paths and try again.")
        exit(1)
    except PermissionError as e:
        print(f"\n❌ Error: {e}")
        print("Please check your file permissions and try again.")
        exit(1)
    except IOError as e:
        print(f"\n❌ Error: {e}")
        print("An I/O error occurred during processing.")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the error message above and try again.")
        exit(1)
    finally:
        # Ensure all resources have been released (with context manager handles file resources)
        print("\nAll resources have been properly released.")

if __name__ == "__main__":
    import sys
    main()