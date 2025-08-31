"""
Pytest configuration file for MLX Fine-tuning UI tests.
"""

import pytest
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add mlx-lm-main to path for imports
sys.path.append(str(project_root / "mlx-lm-main"))

@pytest.fixture(scope="session")
def project_root_path():
    """Provide the project root path."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def example_dataset_path(project_root_path):
    """Provide the example dataset path."""
    return project_root_path / "example_dataset"

@pytest.fixture
def temp_training_dir():
    """Create a temporary directory for training outputs."""
    temp_dir = tempfile.mkdtemp(prefix="test_training_")
    yield Path(temp_dir)
    # Cleanup after test
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)

@pytest.fixture
def cleanup_training_dirs():
    """Clean up any training directories created during tests."""
    yield
    # Clean up training directories after tests
    import glob
    project_root = Path(__file__).parent.parent
    for pattern in ["training_*", "test_training_*"]:
        for training_dir in glob.glob(str(project_root / pattern)):
            if Path(training_dir).exists():
                shutil.rmtree(training_dir)

@pytest.fixture(scope="session", autouse=True)
def suppress_streamlit_warnings():
    """Suppress Streamlit warnings during tests."""
    import warnings
    warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
    warnings.filterwarnings("ignore", message=".*Streamlit app.*")
    
    # Set environment variable to suppress streamlit warnings
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

@pytest.fixture
def minimal_config():
    """Provide a minimal configuration for testing."""
    return {
        "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "fine_tune_type": "lora",
        "num_layers": 2,
        "max_seq_length": 256,
        "batch_size": 1,
        "iters": 3,  # Very small for testing
        "learning_rate": 1e-5,
        "optimizer": "adam",
        "val_batches": 1,
        "steps_per_report": 1,
        "steps_per_eval": 2,
        "save_every": 3,
        "grad_checkpoint": False,
        "mask_prompt": False,
        "seed": 42,
        "test_batches": 1,
        "resume_adapter_file": None,
        "lr_schedule": None,
        "wandb": None,
        "report_to": "",
        "project_name": None,
        "optimizer_config": {"adam": {}, "adamw": {}, "muon": {}, "sgd": {}, "adafactor": {}},
        "lora_parameters": {"rank": 2, "scale": 1.0, "dropout": 0.0}
    }

class MockFile:
    """Mock file object that behaves like uploaded files in Streamlit."""
    
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Test file not found: {file_path}")
    
    def getvalue(self):
        return self.file_path.read_bytes()

@pytest.fixture
def mock_dataset_config(example_dataset_path):
    """Provide mock dataset configuration using example dataset."""
    return {
        "type": "local",
        "train_file": MockFile(example_dataset_path / "train.jsonl"),
        "valid_file": MockFile(example_dataset_path / "valid.jsonl"),
        "test_file": MockFile(example_dataset_path / "test.jsonl")
    }

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (fast, no model downloads)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (may take a long time)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring model download"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests that use actual models as slow
        if "test_run_fine_tuning" in item.name:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.requires_model)
        
        # Mark integration tests
        if "integration" in item.name or "test_training" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
