# build_vocab.py
import os
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime

def collect_all_phonemes(npz_dir):
    """Traverse all npz files and collect all occurring phonemes"""
    all_phonemes = Counter()
    npz_paths = Path(npz_dir).glob("**/*.npz")

    print("🔍 Scanning for phonemes in all .npz files...")
    for npz_path in npz_paths:
        try:
            print(f"🔄 Processing: {npz_path}")
            data = np.load(npz_path, allow_pickle=True)
            ph_seq = data['ph_seq'].tolist()  # Ensure it's a Python list
            all_phonemes.update(ph_seq)
        except Exception as e:
            print(f"⚠️ Failed to read: {npz_path} - {e}")
    
    print(f"✅ Found {len(all_phonemes)} different phonemes in total")
    return all_phonemes

def build_vocab(ph_counter, save_path=None):
    """Build vocabulary, numbering from the most common phonemes"""
    # Reserve special tokens
    special_tokens = ['PAD', 'EOS']  # You can add more as needed, such as 'UNK'
    vocab = {token: i for i, token in enumerate(special_tokens)}
    
    # Sort by frequency, then assign IDs
    sorted_phonemes = sorted(ph_counter.items(), key=lambda x: -x[1])
    for i, (ph, count) in enumerate(sorted_phonemes):
        vocab[ph] = len(vocab)  # Start numbering from len(special_tokens)

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            for ph, idx in sorted(vocab.items(), key=lambda x: x[1]):
                f.write(f"{ph}\t{idx}\n")
        print(f"💾 Phoneme table saved to: {save_path}")
    
    return vocab

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    diffsinger_dir = project_root / "DiffSinger"
    npz_directory = diffsinger_dir / "data/npzs"  # Fixed: Use diffsinger_dir instead of project_root
    output_vocab_file = diffsinger_dir / "data/singer_vocab.txt"  # Output vocabulary file

    # Check if file already exists, if so, append timestamp
    if output_vocab_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_vocab_file = diffsinger_dir / f"data/singer_vocab_{timestamp}.txt"
        print(f"⚠️  singer_vocab.txt already exists, creating new file: {output_vocab_file.name}")

    ph_counter = collect_all_phonemes(npz_directory)
    vocab = build_vocab(ph_counter, save_path=output_vocab_file)

    # Print the 20 most common phonemes
    print("\n📊 Top 20 most common phonemes:")
    for ph, count in ph_counter.most_common(20):
        print(f"  '{ph}': {count}")