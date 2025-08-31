#!/usr/bin/env python3
"""
Run all tests to verify that the MLX fine-tuning functionality is working correctly.
"""

import subprocess
import sys
from pathlib import Path

def run_test(name, command):
    """Run a test and report results"""
    print(f"\n{'='*60}")
    print(f"🧪 Running {name}...")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print(f"✅ {name} PASSED")
            return True
        else:
            print(f"❌ {name} FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ {name} ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running MLX Fine-tuning Tests")
    
    python_exe = "/Users/aditya-mini/Projects/mlx-ui/.venv/bin/python"
    
    tests = [
        ("Minimal Training Test", [python_exe, "test_minimal_training.py"]),
        ("Pytest Training Test", [python_exe, "-m", "pytest", "tests/test_training.py::test_run_fine_tuning_local", "-v"]),
    ]
    
    results = []
    for name, command in tests:
        success = run_test(name, command)
        results.append((name, success))
    
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:.<40} {status}")
        if not success:
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! Fine-tuning is working correctly.")
        print("You can now use the Streamlit UI with confidence.")
        return 0
    else:
        print("💥 SOME TESTS FAILED! Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
