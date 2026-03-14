#!/usr/bin/env python3
"""
UVR Model Config - Model alias resolution for vocal separation.

Maps user-friendly model names to actual model file names.

Usage:
    from uvr_model_config import resolve_model_name, get_model_list
    
    # Resolve alias to actual model file name
    model_file = resolve_model_name("MDX23C-8KFFT")
    # Returns: "MDX23C-8KFFT-InstVoc_HQ.ckpt"
    
    # List available models
    models = get_model_list()
"""

from pathlib import Path
from typing import Optional, Dict, List

PCS_ROOT = Path(__file__).parent.parent.resolve()

# Model alias mapping
# Key: user-friendly alias
# Value: actual model file name (relative to models/<type>_Models/)
MODEL_ALIASES: Dict[str, str] = {
    # MDX-Net Models
    "MDX23C": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
    "MDX23C-8KFFT": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
    "MDX23C-HQ": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
    
    "MDX23": "model_bs_ro4_ep_317_sdr_12.9755.ckpt",
    "MDX23-BS": "model_bs_ro4_ep_317_sdr_12.9755.ckpt",
    
    # VR Models
    "UVR5-DeEcho": "UVR5-DeEcho-Normal.pth",
    "UVR5-DeEcho-Normal": "UVR5-DeEcho-Normal.pth",
    "UVR5-DeEcho-DeReverb": "UVR5-DeEcho-DeReverb.pth",
    
    # Demucs Models
    "Demucs-v3": "htdemucs.yaml",
    "Demucs-v3-Hybrid": "htdemucs.yaml",
    "Demucs-v4": "htdemucs.yaml",
}

# Default model directory for each model type
MODEL_DIRS = {
    "mdx": "ultimatevocalremovergui/models/MDX_Net_Models",
    "vr": "ultimatevocalremovergui/models/VR_Models",
    "demucs": "ultimatevocalremovergui/models/Demucs_Models",
}

# Default model to use if none specified
DEFAULT_MODEL = "MDX23C-8KFFT"


def resolve_model_name(model_name: str) -> str:
    """
    Resolve model alias to actual model file name.
    
    If the model name is an alias, returns the actual file name.
    Otherwise, returns the original name (assumed to be a file name or path).
    
    Args:
        model_name: Model alias (e.g., "MDX23C-8KFFT") or file name
        
    Returns:
        Resolved model file name
    """
    return MODEL_ALIASES.get(model_name, model_name)


def get_model_type(model_name: str) -> str:
    """
    Guess model type based on file extension or alias.
    
    Args:
        model_name: Model alias or file name
        
    Returns:
        Model type: 'mdx', 'vr', or 'demucs'
    """
    resolved = resolve_model_name(model_name)
    lower = resolved.lower()
    
    if lower.endswith('.ckpt') or lower.endswith('.onnx'):
        return 'mdx'
    elif lower.endswith('.pth') or lower.endswith('.th'):
        return 'vr'
    elif lower.endswith('.yaml') or lower.endswith('.th'):
        return 'demucs'
    
    # Guess by alias name
    if 'mdx' in lower:
        return 'mdx'
    elif 'demucs' in lower:
        return 'demucs'
    elif 'uvr' in lower or 'deecho' in lower:
        return 'vr'
    
    return 'mdx'  # Default to MDX


def get_model_dir(model_name: str) -> str:
    """
    Get model directory path based on model type.
    
    Args:
        model_name: Model alias or file name
        
    Returns:
        Directory path relative to PCS_ROOT
    """
    model_type = get_model_type(model_name)
    return MODEL_DIRS.get(model_type, MODEL_DIRS['mdx'])


def get_full_model_path(model_name: str) -> Path:
    """
    Get full model file path.
    
    Args:
        model_name: Model alias or file name
        
    Returns:
        Full path to model file
    """
    resolved = resolve_model_name(model_name)
    model_dir = get_model_dir(model_name)
    return PCS_ROOT / model_dir / resolved


def model_exists(model_name: str) -> bool:
    """
    Check if model file exists.
    
    Args:
        model_name: Model alias or file name
        
    Returns:
        True if model file exists
    """
    return get_full_model_path(model_name).exists()


def get_model_list() -> List[str]:
    """
    Get list of available model aliases.
    
    Returns:
        List of model alias names
    """
    return list(MODEL_ALIASES.keys())


def get_default_model() -> str:
    """Get the default model alias."""
    return DEFAULT_MODEL


def is_model_alias(model_name: str) -> bool:
    """Check if the given name is a model alias."""
    return model_name in MODEL_ALIASES
