#!/usr/bin/env python3
"""
Setup script for distributional reinforcement learning project.

This script helps set up the project environment and verify installation.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install project dependencies."""
    print("Installing dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("✗ requirements.txt not found")
        return False
    
    # Install dependencies
    return run_command("pip install -r requirements.txt", "Installing dependencies")


def verify_installation():
    """Verify that key packages are installed correctly."""
    print("Verifying installation...")
    
    packages_to_check = [
        ("torch", "PyTorch"),
        ("gymnasium", "Gymnasium"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("yaml", "PyYAML"),
        ("tqdm", "TQDM")
    ]
    
    all_good = True
    for package, name in packages_to_check:
        try:
            __import__(package)
            print(f"✓ {name} is installed")
        except ImportError:
            print(f"✗ {name} is not installed")
            all_good = False
    
    return all_good


def run_tests():
    """Run the test suite."""
    print("Running tests...")
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("✗ pytest is not installed. Install with: pip install pytest")
        return False
    
    # Run tests
    return run_command("python -m pytest tests/ -v", "Running test suite")


def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    
    directories = ["logs", "checkpoints", "notebooks"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up Distributional Reinforcement Learning Project")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Dependency installation failed. You may need to install them manually.")
        print("   Run: pip install -r requirements.txt")
    
    # Verify installation
    if not verify_installation():
        print("⚠️  Some packages are missing. Please install them manually.")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("⚠️  Tests failed. Please check the installation.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run a quick demo: python cli.py demo")
    print("2. Train an agent: python cli.py train --env MiniGrid --episodes 100")
    print("3. Explore the notebook: jupyter notebook notebooks/distributional_rl_demo.ipynb")
    print("4. Read the README.md for more information")


if __name__ == "__main__":
    main()
