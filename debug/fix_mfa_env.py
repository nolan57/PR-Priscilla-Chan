#!/usr/bin/env python3
"""
Script to fix MFA environment dependencies, specifically for NumPy/Numba compatibility
"""
import subprocess
import sys
import os
import platform


def fix_mfa_dependencies():
    """
    Fix the MFA environment dependencies to resolve NumPy/Numba compatibility issue
    """
    print("Fixing MFA environment dependencies...")
    print(f"Python executable: {sys.executable}")
    
    # Check current numpy version
    try:
        import numpy as np
        print(f"Current NumPy version: {np.__version__}")
        
        # Parse version
        numpy_version = tuple(map(int, np.__version__.split('.')[:2]))
        
        # Check if numpy version is greater than 2.3
        if numpy_version > (2, 3):
            print(f"NumPy version {np.__version__} is incompatible with Numba. Downgrading to 1.26.4...")
            
            # Attempt to downgrade numpy
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "numpy==1.26.4"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Failed to downgrade numpy: {result.stderr}")
                # Try alternative numpy version that's compatible
                print("Trying alternative compatible version...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "numpy<=2.3"
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Failed to install compatible numpy: {result.stderr}")
                    return False
            
            print("Successfully installed compatible NumPy version")
            return True
        else:
            print("NumPy version is already compatible")
            return True
            
    except ImportError:
        print("NumPy is not installed, installing a compatible version...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "numpy<=2.3"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Failed to install numpy: {result.stderr}")
            return False
        
        print("Successfully installed NumPy")
        return True
    except Exception as e:
        print(f"Error checking NumPy version: {e}")
        return False


def install_additional_packages():
    """
    Install additional packages that may be required for MFA to work properly
    """
    print("Installing additional packages that may be required...")
    
    packages = [
        "standard-aifc",
        "standard-sunau",
        "librosa<0.10.0"  # Older version that's more stable
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Warning: Failed to install {package}: {result.stderr}")
        else:
            print(f"Successfully installed {package}")


def main():
    """
    Main function to fix MFA environment
    """
    print("MFA Environment Fixer")
    print("=====================")
    
    # Fix dependencies
    if not fix_mfa_dependencies():
        print("Failed to fix MFA dependencies")
        return 1
    
    # Install additional packages
    install_additional_packages()
    
    print("\nEnvironment fixing completed!")
    print("You should now be able to run the align_lyrics.py script without NumPy/Numba compatibility errors.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())