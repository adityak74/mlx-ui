#!/usr/bin/env python3
"""
Quick Start Script for MLX Fine-tuning UI
This script helps you get started with the application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print("✅ Python version:", sys.version.split()[0])
    return True

def check_platform():
    """Check if platform is supported"""
    import platform
    system = platform.system()
    if system != "Darwin":  # macOS
        print("❌ MLX is only supported on macOS")
        return False
    print("✅ Platform:", system)
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        
        # Install additional ML dependencies
        print("\n📚 Installing additional ML dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "transformers"], 
                      check=True, capture_output=True)
        print("✅ ML dependencies installed successfully")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def install_mlx():
    """Install MLX framework"""
    print("\n🚀 Installing MLX...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "mlx"], 
                      check=True, capture_output=True)
        print("✅ MLX installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing MLX: {e}")
        return False

def install_mlx_lm():
    """Install mlx-lm in editable mode"""
    print("\n🔧 Installing mlx-lm...")
    try:
        mlx_lm_path = Path("mlx-lm-main")
        if not mlx_lm_path.exists():
            print("❌ mlx-lm-main directory not found")
            return False
        
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(mlx_lm_path)], 
                      check=True, capture_output=True)
        print("✅ mlx-lm installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing mlx-lm: {e}")
        return False

def test_installation():
    """Test if everything is working"""
    print("\n🧪 Testing installation...")
    try:
        import mlx
        print(f"✅ MLX imported successfully (version: {mlx.__version__})")
        
        import mlx_lm
        print("✅ mlx-lm imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def start_application():
    """Start the Streamlit application"""
    print("\n🎉 Starting MLX Fine-tuning UI...")
    print("The application will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    """Main function"""
    print("🚀 MLX Fine-tuning UI - Quick Start")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_platform():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please check the error above.")
        sys.exit(1)
    
    if not install_mlx():
        print("\n❌ Failed to install MLX. Please check the error above.")
        sys.exit(1)
    
    if not install_mlx_lm():
        print("\n❌ Failed to install mlx-lm. Please check the error above.")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation test failed. Please check the error above.")
        sys.exit(1)
    
    print("\n🎉 All dependencies installed successfully!")
    
    # Ask user if they want to start the application
    response = input("\nWould you like to start the application now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        start_application()
    else:
        print("\nTo start the application later, run:")
        print("  streamlit run app.py")
        print("  or")
        print("  make run")

if __name__ == "__main__":
    main()
