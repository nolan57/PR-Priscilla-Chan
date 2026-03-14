# PCS Audio Processing Automation Scheme

## Overview

This document outlines an automated audio processing pipeline for training singing voice synthesis models using the PCS codebase. The scheme leverages AI assistants (via MCP) to orchestrate existing scripts and tools.

## Implemented Components

### 1. Audio Edit CLI (`audio-edit-cli/`)

Command-line tools for audio cleaning and processing.

```
audio-edit-cli/
├── __main__.py           # Module entry point
├── cli.py               # Main CLI entry
├── requirements.txt     # Dependencies
└── processors/
    ├── muter.py        # Silence region processing (multiple methods)
    ├── trimmer.py      # Audio trimming
    ├── harmony.py      # Harmony removal
    └── merge.py        # Audio merging
```

**Usage:**

```bash
# Silence processing (multiple methods available)
python -m audio_edit_cli.muter -i input.wav -o output.wav -r "1.5,3.0;5.0,6.0" --method harmonic_residual

# Audio trimming
python -m audio_edit_cli.trimmer -i input.wav -o output.wav -s 0.5 -e 10.0 --fade-in 0.01 --normalize

# Harmony removal
python -m audio_edit_cli.harmony -i input.wav -o output.wav -r "0.5,1.5" --threshold 0.5

# Merge audio
python -m audio_edit_cli.merge -i "file1.wav,file2.wav" -o merged.wav --crossfade 0.5
```

---

### 2. Workflow Pipeline (`workflow/`)

Batch processing workflows with state tracking and breakpoint resume.

```
workflow/
├── __init__.py
├── __main__.py
├── cli.py              # CLI entry point
├── state.py            # SQLite state tracking
├── vocoder_workflow.py    # Vocoder training data workflow
└── diffsinger_workflow.py # DiffSinger training data workflow
```

**Usage:**

```bash
# Vocoder training data workflow
python -m workflow vocoder -i ./raw_songs -o ./data --skip-existing

# DiffSinger training data workflow
python -m workflow diffsinger -i ./raw_songs -o ./data --skip-existing

# Check workflow status
python -m workflow status --db ./data/workflow_state.db

# Reset workflow
python -m workflow reset --workflow vocoder_20260301 --db ./data/workflow_state.db
```

**Features:**

- **Breakpoint Resume**: SQLite tracks processing state, skips completed files
- **Parallel Processing**: Configurable workers with `--workers` flag
- **Error Handling**: Failed files logged, continues with others
- **Manifest Generation**: Outputs `manifest.json` on completion

---

### 3. MCP Server (`mcp_server/`)

Model Context Protocol server exposing tools to AI assistants.

```
mcp_server/
├── __init__.py
├── __main__.py
├── server.py            # Main MCP server
├── requirements.txt
├── MCP_CONFIG.md       # Configuration guide
└── types/
    ├── __init__.py
    └── models.py       # Pydantic type definitions
```

**Configuration:**

```json
{
  "mcpServers": {
    "pcs-audio": {
      "command": "uv",
      "args": ["--directory", "/Users/lpcw/Documents/PCS", "run", "mcp_server"],
      "env": {
        "PYTHONPATH": "/Users/lpcw/Documents/PCS"
      }
    }
  }
}
```

---

## Project Architecture

```
PCS/
├── pipeline/                    # Preprocessing pipeline
│   ├── separate_vocals.py       # Vocal separation (UVR5)
│   ├── extract_perfect_f0.py    # High-precision F0 extraction
│   └── build_npzs.py           # DiffSinger training data builder
│
├── SingingVocoders/             # Vocoder training
│   ├── process.py              # Preprocessing (wav→npz: mel+f0+audio)
│   ├── train.py                 # Training entry point
│   └── configs/                # Configuration files
│
├── DiffSinger/                 # Acoustic + Variance model training
│   ├── scripts/binarize.py     # Binarization
│   ├── scripts/train.py        # Training entry
│   └── configs/                # Configuration files
│
├── audio-edit/                  # GUI audio cleaning tools
│   ├── audio_muter.py          # Silence region annotation
│   ├── audio_trimmer.py        # Audio trimming
│   ├── audio_merger.py         # Audio merging
│   ├── harmony_remover.py      # Harmony removal
│   └── pro_singer_separator_*.py  # Professional separator
│
├── audio-edit-cli/              # CLI tools (NEW)
├── workflow/                   # Batch workflows (NEW)
├── mcp_server/                  # MCP server (NEW)
│
└── ultimatevocalremovergui/    # UVR5 core
```

---

## Two Training Data Generation Flows

### Flow A: Vocoder Training (SingingVocoders)

```
Raw Audio → Separation → Cleaning → Preprocess → Training
              ↓                         ↓
           Dry Vocal WAV          npz (mel+f0+audio)
```

| Step          | Script/Tool                 | Input           | Output                |
| ------------- | --------------------------- | --------------- | --------------------- |
| 1. Separation | `separate_vocals`           | Mixed audio.wav | Dry vocal.wav         |
| 2. Cleaning   | `mute_audio` / `trim_audio` | Dry vocal.wav   | Cleaned dry vocal.wav |
| 3. Preprocess | `process.py --config`       | Dry vocal.wav   | .npz (mel/f0/audio)   |
| 4. Training   | `train.py --config`         | .npz            | Vocoder model         |

**npz contents (SingingVocoders):**

- `audio`: [T] waveform
- `mel`: [T, 128] mel spectrogram
- `f0`: [T] fundamental frequency
- `uv`: [T] voiced/unvoiced flag

---

### Flow B: Acoustic + Variance Model Training (DiffSinger)

```
Raw Audio → Separation → Cleaning → F0 Extract → MFA → NPZ Build → Train
              ↓            ↓          ↓           ↓         ↓
           Dry.wav    Cleaned   .f0.npy    TextGrid   .npz   Acoustic
```

| Step          | Script/Tool                | Input                | Output                |
| ------------- | -------------------------- | -------------------- | --------------------- |
| 1. Separation | `separate_vocals`          | Mixed audio          | Dry vocal.wav         |
| 2. Cleaning   | `mute_audio` etc.          | Dry vocal.wav        | Cleaned dry vocal.wav |
| 3. F0 Extract | `extract_f0`               | Dry vocal.wav        | .f0.npy               |
| 4. Alignment  | MFA                        | Dry vocal.wav + text | .TextGrid             |
| 5. Build NPZ  | `build_npzs.py`            | Dry+TextGrid+F0      | .npz                  |
| 6. Training   | `binarize.py` + `train.py` | .npz                 | Acoustic model        |

**npz contents (DiffSinger):**

- `f0`: [T] fundamental frequency
- `mel`: [128, T] mel spectrogram
- `ph_seq`: [N] phoneme sequence
- `ph_dur`: [N] phoneme duration (frames)
- `spk_id`: speaker ID

---

## MCP Tools

### Audio Processing Tools

| Tool              | Function                            | Parameters                                                        |
| ----------------- | ----------------------------------- | ----------------------------------------------------------------- |
| `separate_vocals` | Vocal separation using UVR5         | input_dir, output_dir, model, sample_rate                         |
| `mute_audio`      | Mute specified regions              | input_path, output_path, regions, method                          |
| `trim_audio`      | Trim audio to range                 | input_path, output_path, start, end, fade_in, fade_out, normalize |
| `remove_harmony`  | Remove harmony via spectral masking | input_path, output_path, ref_region, threshold, sensitivity       |
| `merge_audio`     | Merge multiple audio files          | inputs, output_path, crossfade, normalize                         |

### Pipeline Tools

| Tool                   | Function                                      | Parameters                                                 |
| ---------------------- | --------------------------------------------- | ---------------------------------------------------------- |
| `extract_f0`           | High-precision F0 (CREPE+Harvest+Parselmouth) | wav_path, output_dir, save_vad                             |
| `build_diffsinger_npz` | DiffSinger training data                      | wav_dir, textgrid_dir, f0_dir, output_dir, use_external_f0 |

### Workflow Tools (Skills)

| Tool                      | Function                  | Parameters                                                                           |
| ------------------------- | ------------------------- | ------------------------------------------------------------------------------------ |
| `run_vocoder_workflow`    | Full vocoder data prep    | input_dir, output_dir, model, config, skip_cleaning, skip_existing, workers, reset   |
| `run_diffsinger_workflow` | Full DiffSinger data prep | input_dir, output_dir, model, skip_cleaning, skip_mfa, skip_existing, workers, reset |
| `get_workflow_status`     | Query processing status   | db_path                                                                              |

---

## Output Directory Structure

```
data/
├── workflow_state.db   # SQLite state database
├── manifest.json      # Processing manifest
├── separated/        # Separated vocals
├── cleaned/          # Cleaned audio
├── f0/              # F0 files (DiffSinger)
├── aligned/          # TextGrid alignment
└── npz/             # Training data
```

---

## AI Usage Example

> "Use run_diffsinger_workflow to process `./data/raw` directory, skip already processed files, and generate manifest.jsonl"

The AI will automatically:

1. Scan `./data/raw` for audio files
2. Orchestrate: `separate_vocals` → `mute_audio` → `extract_f0` → `build_npz`
3. Log each stage to SQLite
4. Skip completed files (breakpoint resume)
5. Generate manifest on completion

---

## Key Technical Points

- **Breakpoint Resume**: SQLite records processing stage for each file; failed jobs can be retried
- **Type Safety**: Pydantic ensures AI accurately understands function signatures
- **Environment Isolation**: Run MCP server via `uv run mcp_server.server`
- **State Query**: AI can directly query progress via SQL

---

## Dependencies

### audio-edit-cli

```
numpy
scipy
soundfile
ffmpeg-python
```

### mcp_server

```
mcp
pydantic
```

---

## Reference

- See `/Users/lpcw/Documents/opencode/docs/autoaduioprocessing.md` for MCP architecture details
- See `/Users/lpcw/Documents/PCS/AGENTS.md` for code style guidelines
