#!/usr/bin/env python3
"""
UVR Model Config Loader - Load and resolve model aliases to actual paths.

Usage:
    from uvr_model_config import get_model_path, get_model_info
    
    # Get full model path
    model_path = get_model_path("MDX23C-8KFFT")
    
    # Get model info dict
    info = get_model_info("MDX23C-8KFFT")
    print(info['type'], info['path'])
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

PCS_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_PATH = PCS_ROOT / "pipeline" / "uvr_models_config.yaml"

_model_config = None


def _load_config():
    """Load model configuration from YAML file."""
    global _model_config
    
    if _model_config is not None:
        return _model_config
    
    if not CONFIG_PATH.exists():
        _model_config = {"models": {}, "default_model": None}
        return _model_config
    
    try:
        import yaml
        with open(CONFIG_PATH, 'r') as f:
            _model_config = yaml.safe_load(f) or {"models": {}}
    except ImportError:
        # Fallback: simple YAML parser for basic configs
        _model_config = {"models": {}, "default_model": None}
        with open(CONFIG_PATH, 'r') as f:
            current_model = None
            for line in f:
                line = line.rstrip()
                if not line or line.strip().startswith('#'):
                    continue
                if line.startswith('models:'):
                    continue
                if line.startswith('default_model:'):
                    _model_config['default_model'] = line.split(':', 1)[1].strip().strip('"\'')
                    continue
                if line.startswith('  ') and ':' in line:
                    key = line.strip().split(':')[0].strip()
                    val = line.split(':', 1)[1].strip().strip('"\'')
                    if key.endswith(':') and not val:
                        # New model section
                        current_model = key[:-1]
                        _model_config['models'][current_model] = {}
                    elif current_model and key in ['path', 'model_dir', 'type', 'description']:
                        _model_config['models'][current_model][key] = val
    except Exception as e:
        print(f"Warning: Failed to load model config: {e}")
        _model_config = {"models": {}, "default_model": None}
    
    return _model_config


def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """
    Get model information by alias name.
    
    Args:
        model_name: Model alias (e.g., "MDX23C-8KFFT")
        
    Returns:
        Dict with model info (path, model_dir, type, description) or None if not found
    """
    config = _load_config()
    return config.get('models', {}).get(model_name)


def get_model_path(model_name: str) -> Optional[str]:
    """
    Get full model file path by alias name.
    
    Args:
        model_name: Model alias (e.g., "MDX23C-8KFFT")
        
    Returns:
        Full path to model file, or None if not found
    """
    config = _load_config()
    model_info = config.get('models', {}).get(model_name)
    
    if not model_info:
        return None
    
    model_base = config.get('model_base_dir', 'ultimatevocalremovergui/models')
    model_dir = model_info.get('model_dir', '')
    model_path = model_info.get('path', '')
    
    # Construct full path
    if model_dir:
        full_path = PCS_ROOT / model_base / model_dir / model_path
    else:
        full_path = PCS_ROOT / model_base / model_path
    
    return str(full_path)


def resolve_model_name(model_name: str) -> str:
    """
    Resolve model alias to actual model file name.
    
    If the model name is an alias in the config, returns the actual file name.
    Otherwise, returns the original name (assumed to be already a file name or path).
    
    Args:
        model_name: Model alias or file name
        
    Returns:
        Resolved model file name or original if not an alias
    """
    config = _load_config()
    model_info = config.get('models', {}).get(model_name)
    
    if model_info:
        return model_info.get('path', model_name)
    
    # Not an alias, return as-is
    return model_name


def get_default_model() -> str:
    """Get the default model alias."""
    config = _load_config()
    return config.get('default_model', 'MDX23C-8KFFT')


def list_models(model_type: Optional[str] = None) -> list:
    """
    List all available model aliases.
    
    Args:
        model_type: Optional filter by type ('mdx', 'vr', 'demucs')
        
    Returns:
        List of model alias names
    """
    config = _load_config()
    models = config.get('models', {})
    
    if model_type:
        return [name for name, info in models.items() if info.get('type') == model_type]
    
    return list(models.keys())


def is_model_alias(model_name: str) -> bool:
    """Check if the given name is a model alias in the config."""
    config = _load_config()
    return model_name in config.get('models', {})
