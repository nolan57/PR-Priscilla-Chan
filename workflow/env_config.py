#!/usr/bin/env python3
"""
Environment Config Loader - Load and manage conda environment configurations.
"""

import os
from pathlib import Path
from typing import Optional
import subprocess


PCS_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_CONFIG = PCS_ROOT / "workflow" / "env_config.toml"


class EnvConfig:
    """Conda environment configuration manager."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_CONFIG
        self._config = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from TOML file."""
        if not self.config_path.exists():
            return

        try:
            current_section = None
            with open(self.config_path) as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue
                    # Section header [section] or [section.subsection]
                    if line.startswith("[") and line.endswith("]"):
                        current_section = line[1:-1]
                        continue
                    # Key = value pair
                    if "=" in line and current_section:
                        key, val = line.split("=", 1)
                        key = key.strip().strip('"')
                        # Remove inline comments (# ...)
                        val = val.split("#")[0].strip().strip('"')
                        # Store as section.key = value
                        full_key = f"{current_section}.{key}"
                        # Navigate/create nested dict
                        parts = full_key.split(".")
                        d = self._config
                        for p in parts[:-1]:
                            if p not in d:
                                d[p] = {}
                            d = d[p]
                        d[parts[-1]] = val
        except Exception as e:
            print(f"Warning: Failed to load config: {e}")

    def get_env(self, component: str) -> str:
        """Get conda environment for a component."""
        return self._config.get("environments", {}).get(component, "base")

    def get_workflow_env(self, workflow: str, step: str) -> str:
        """Get conda environment for a workflow step."""
        key = f"workflows.{workflow}.{step}"
        for env_key, env_val in self._config.get("environments", {}).items():
            if key.endswith(f".{env_key}"):
                return env_val
        return self._config.get(key) or "base"


def run_in_conda_env(
    env_name: str,
    cmd: list,
    cwd: Optional[str] = None,
    timeout: int = 3600,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run command in specified conda environment.

    Args:
        env_name: Conda environment name
        cmd: Command list to execute
        cwd: Working directory
        timeout: Timeout in seconds
        capture_output: Capture stdout/stderr

    Returns:
        subprocess.CompletedProcess result
    """
    full_cmd = ["conda", "run", "-n", env_name, *cmd]

    return subprocess.run(
        full_cmd,
        cwd=cwd,
        timeout=timeout,
        capture_output=capture_output,
        text=True,
    )


def check_env(env_name: str) -> bool:
    """Check if conda environment exists."""
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    return env_name in result.stdout


_default_config = None


def get_default_config() -> EnvConfig:
    """Get default environment configuration."""
    global _default_config
    if _default_config is None:
        _default_config = EnvConfig()
    return _default_config
