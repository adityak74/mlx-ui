# Testing Documentation

This document describes the comprehensive testing setup for the MLX Fine-tuning UI project.

## Test Structure

```
tests/
├── conftest.py           # Pytest configuration and fixtures
├── test_training.py      # Main training functionality tests
└── __pycache__/         # Pytest cache

# Test support files
├── pytest.ini           # Pytest configuration
├── run_tests.py         # Custom test runner script
└── test_minimal_training.py  # Direct MLX training test
```

## Available Test Commands

### Make Commands

| Command | Description | Speed | Coverage |
|---------|-------------|-------|----------|
| `make test` | Test MLX installation | Fast | N/A |
| `make test-unit` | Run unit tests only | Fast | No |
| `make test-integration` | Run integration tests (downloads models) | Slow | No |
| `make test-coverage` | Run all tests with coverage report | Medium | Yes |
| `make test-all` | Run comprehensive test suite | Slow | Yes |
| `make test-quick` | Run custom test runner | Medium | No |

### Direct Commands

```bash
# Run specific test types
.venv/bin/python -m pytest tests/ -m "unit"
.venv/bin/python -m pytest tests/ -m "integration"
.venv/bin/python -m pytest tests/ -m "slow"
.venv/bin/python -m pytest tests/ -m "requires_model"

# Run with coverage
.venv/bin/python -m pytest tests/ --cov=app --cov-report=html

# Run custom test runner
.venv/bin/python run_tests.py

# Run minimal training test directly
.venv/bin/python test_minimal_training.py
```

## Test Types and Markers

### Test Markers
- `@pytest.mark.unit`: Fast unit tests that don't download models
- `@pytest.mark.integration`: Integration tests that test complete workflows
- `@pytest.mark.slow`: Tests that may take several minutes
- `@pytest.mark.requires_model`: Tests that download ML models
- `@pytest.mark.skip()`: Skipped tests (e.g., requires internet)

### Test Categories

#### Unit Tests (Fast)
- Configuration validation
- File I/O operations
- Utility functions
- Mock data processing

#### Integration Tests (Slow)
- Complete fine-tuning workflow
- Model loading and processing
- Dataset loading and validation
- Adapter file generation

## Test Configuration

### Pytest Configuration (pytest.ini)
- Test discovery patterns
- Output formatting
- Coverage settings
- Marker definitions

### Conftest.py Features
- Automatic path setup for imports
- Streamlit warning suppression
- Reusable fixtures for configs and datasets
- Automatic cleanup of test artifacts

## Fixtures Available

### From conftest.py
- `minimal_config`: Basic training configuration
- `mock_dataset_config`: Mock dataset using example data
- `temp_training_dir`: Temporary directory for test outputs
- `cleanup_training_dirs`: Automatic cleanup after tests
- `project_root_path`: Project root directory path
- `example_dataset_path`: Path to example dataset

### Example Usage
```python
def test_training(minimal_config, mock_dataset_config, cleanup_training_dirs):
    result = run_fine_tuning(minimal_config, mock_dataset_config)
    assert result is not None
```

## Coverage Reporting

Coverage reports are generated in multiple formats:

### Terminal Output
Shows line-by-line coverage percentages during test runs.

### HTML Report
Detailed interactive coverage report available at `htmlcov/index.html`:
- Line-by-line coverage visualization
- Missing line highlighting
- Function and branch coverage
- Searchable and filterable results

### Coverage Configuration
- Source code coverage tracking
- Exclusion of test files and dependencies
- Missing line reporting
- HTML report generation

## Running Tests in Development

### Quick Development Cycle
```bash
# Fast feedback loop
make test-unit

# Full validation
make test-coverage

# Before commits
make test-all
```

### Debugging Tests
```bash
# Run with verbose output and no capture
.venv/bin/python -m pytest tests/ -v -s --tb=long

# Run specific test
.venv/bin/python -m pytest tests/test_training.py::test_save_config_to_file -v -s

# Run with pdb on failures
.venv/bin/python -m pytest tests/ --pdb
```

## Test Data

### Example Dataset
The tests use the example dataset in `example_dataset/`:
- `train.jsonl`: 20 training examples
- `valid.jsonl`: 10 validation examples  
- `test.jsonl`: 10 test examples

### Mock Objects
- `MockFile`: Simulates Streamlit uploaded files
- Temporary directories for training outputs
- Configurable model parameters for testing

## Continuous Integration

The test setup is designed to support CI/CD:
- Fast unit tests for quick feedback
- Slow integration tests for thorough validation
- Coverage reporting for quality metrics
- Proper cleanup to prevent resource leaks

## Best Practices

### Writing Tests
1. Use appropriate markers (`@pytest.mark.unit`, etc.)
2. Use fixtures for common test data
3. Clean up resources in test teardown
4. Test both success and failure cases
5. Use descriptive test names and docstrings

### Running Tests
1. Run unit tests during development
2. Run integration tests before commits
3. Check coverage reports regularly
4. Use appropriate test markers to control execution

### Maintaining Tests
1. Keep tests independent and isolated
2. Update fixtures when adding new features
3. Add new test markers as needed
4. Regular cleanup of unused test code
