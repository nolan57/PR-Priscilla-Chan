# AGENTS.md

This file contains guidelines and commands for agentic coding agents working in the SingingVocoders repository.

## Build/Lint/Test Commands

### Core Commands
- **Preprocessing**: `python process.py --config configs/[config_file].yaml --num_cpu [num_cores] --strx [1|0]`
- **Training**: `python train.py --config configs/[config_file].yaml --exp_name [experiment_name] --work_dir [optional_dir]`
- **Export Checkpoint**: `python export_ckpt.py --ckpt_path [path_to_ckpt] --save_path [output_path] --work_dir [optional_dir]`

### Testing
- **Run single test**: No formal test framework detected. Use Lightning's test functionality through training scripts
- **Validation**: Use built-in validation during training with `val_check_interval` config parameter
- **Model evaluation**: Use `test_dataloader()`, `test_step()` methods in training tasks

### Dependencies
- No formal package management files (requirements.txt, setup.py, pyproject.toml) found
- Uses PyTorch, Lightning, Click, YAML, Torchaudio
- Dependencies must be installed manually based on import statements

## Code Style Guidelines

### Import Style
- Standard library imports first, then third-party, then local imports
- Use absolute imports for local modules: `from utils.config_utils import read_full_config`
- Group related imports together
- Use `import pathlib` instead of `from pathlib import Path` when possible

### Naming Conventions
- **Variables**: snake_case (`config_path`, `work_dir`)
- **Functions**: snake_case (`read_full_config`, `get_latest_checkpoint_path`)
- **Classes**: PascalCase (`UnivNet`, `GLU`, `Upspamler`)
- **Constants**: UPPER_SNAKE_CASE (`LRELU_SLOPE`)
- **Files**: snake_case (`config_utils.py`, `training_utils.py`)

### Type Hints
- Use type hints for function parameters and return values
- Import typing from standard library: `from typing import Dict, Tuple, Union`
- Use `pathlib.Path` for file paths
- Example: `def preprocess(config, input_path, output_path, num_cpu, st_path):`

### Error Handling
- Use try-except blocks with specific exception types
- Return error information as tuples: `(False, error_message)`
- Log errors appropriately for debugging
- Use assertions for validation of critical conditions

### Configuration Management
- YAML-based configuration system with inheritance
- Base configs in `configs/base.yaml`, extended by model-specific configs
- Use `read_full_config()` for hierarchical config loading
- Config keys use snake_case: `data_input_path`, `audio_sample_rate`

### PyTorch/Lightning Patterns
- Extend `pl.LightningModule` for training tasks
- Use `@click.command()` decorators for CLI interfaces
- Implement `training_step()`, `validation_step()`, `test_step()` methods
- Use Lightning callbacks for model checkpointing and progress tracking
- Follow the established pattern: config → task → trainer → fit

### File Organization
- **Configs**: `configs/` directory with YAML files
- **Models**: `models/[model_name]/` with `__init__.py` and implementation
- **Modules**: `modules/[module_type]/` for reusable components
- **Training**: `training/` directory for task definitions
- **Utils**: `utils/` directory for helper functions

### Code Comments and Documentation
- Class docstrings should briefly describe purpose: `"""Parallel WaveGAN Generator module."""`
- Use inline comments for complex operations
- Document configuration parameters in YAML files
- Click commands include help text: `@click.option('--config', required=True, metavar='FILE', help='Path to the configuration file')`

### Logging
- Use Python's logging module with consistent format
- Lightning's `rank_zero_info()` for distributed training messages
- Progress bars with tqdm for long-running operations
- TensorBoard logging for training metrics

### GPU/CPU Handling
- Use Lightning's accelerator and device configurations
- Set multiprocessing strategy: `torch.multiprocessing.set_sharing_strategy('file_system')`
- Handle NCCL settings with environment variables
- Use CPU/GPU-agnostic PyTorch operations

### Data Processing
- Support both .wav and .flac audio formats
- Use ProcessPoolExecutor for parallel preprocessing
- Save processed data as .npz files with audio, mel, f0, uv, pe fields
- Implement proper data validation and error reporting