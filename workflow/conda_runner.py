#!/usr/bin/env python3
"""
Conda Environment Runner - Execute commands in specific conda environments.

Usage:
    from conda_runner import run_in_env

    # Run python script in specific env
    run_in_env("vocoder", ["python", "script.py"], ...)

    # Run module in specific env
    run_in_env("sola", ["-m", "audio_edit_cli.muter"], ...)
"""

import subprocess
import sys
from typing import Optional


def run_in_env(
    env_name: str,
    cmd: list,
    *args,
    cwd: Optional[str] = None,
    timeout: int = 3600,
    capture_output: bool = True,
    **kwargs,
) -> subprocess.CompletedProcess:
    """
    Run command in specified conda environment.

    Args:
        env_name: Conda environment name
        cmd: Command list to execute
        cwd: Working directory
        timeout: Timeout in seconds
        capture_output: Capture stdout/stderr
        **kwargs: Additional subprocess.run arguments

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
        **kwargs,
    )


def run_python_in_env(
    env_name: str,
    script_or_module: str,
    *script_args,
    cwd: Optional[str] = None,
    timeout: int = 3600,
) -> tuple[int, str, str]:
    """
    Run python script or module in specific conda environment.

    Args:
        env_name: Conda environment name
        script_or_module: Python script path or module (e.g., 'audio_edit_cli.muter')
        script_args: Arguments to pass to the script/module
        cwd: Working directory
        timeout: Timeout in seconds

    Returns:
        (returncode, stdout, stderr)
    """
    cmd = ["python"]

    if script_or_module.startswith("-m "):
        cmd.extend(script_or_module.split())
    else:
        cmd.append(script_or_module)

    cmd.extend(script_args)

    result = run_in_env(env_name, cmd, cwd=cwd, timeout=timeout)

    return result.returncode, result.stdout, result.stderr


def is_env_available(env_name: str) -> bool:
    """Check if conda environment exists."""
    result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True)
    return env_name in result.stdout


def get_env_python(env_name: str) -> Optional[str]:
    """Get python executable path for specific conda environment."""
    result = subprocess.run(
        ["conda", "run", "-n", env_name, "which", "python"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run commands in conda environments")
    parser.add_argument("-n", "--env", required=True, help="Conda environment name")
    parser.add_argument("command", nargs="+", help="Command to run")
    parser.add_argument("--cwd", help="Working directory")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds")

    args = parser.parse_args()

    result = run_in_env(args.env, args.command, cwd=args.cwd, timeout=args.timeout)

    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    sys.exit(result.returncode)
