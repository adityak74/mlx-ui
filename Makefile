# Makefile for MLX Fine-tuning UI project

.PHONY: help install run clean test test-unit test-integration test-coverage test-all dev-setup format lint setup start stop status

# Python executable (use virtual env if available)
PYTHON := $(shell if [ -f ".venv/bin/python" ]; then echo ".venv/bin/python"; else echo "python"; fi)

# Colors for better output
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
RED = \033[0;31m
NC = \033[0m # No Color

# Default target
help:
	@echo "$(BLUE)MLX Fine-tuning UI - Available Commands$(NC)"
	@echo "================================================"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@echo "  setup      - Complete setup (install + test)"
	@echo "  install    - Install all dependencies"
	@echo "  test       - Test MLX installation"
	@echo ""
	@echo "$(GREEN)Testing Commands:$(NC)"
	@echo "  test-unit  - Run unit tests only"
	@echo "  test-integration - Run integration tests (slow)"
	@echo "  test-training - Run actual fine-tuning tests (very slow)"
	@echo "  test-coverage - Run tests with coverage report"
	@echo "  test-all   - Run all tests with coverage"
	@echo ""
	@echo "$(GREEN)Run Commands:$(NC)"
	@echo "  start      - Start the application"
	@echo "  run        - Alias for start"
	@echo "  stop       - Stop running Streamlit processes"
	@echo "  status     - Check if app is running"
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@echo "  dev-setup  - Setup development environment"
	@echo "  format     - Format code with black"
	@echo "  lint       - Lint code with flake8"
	@echo ""
	@echo "$(GREEN)Utility Commands:$(NC)"
	@echo "  clean      - Clean up temporary files"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC) make setup && make start"

# Complete setup (install + test)
setup: install test
	@echo "$(GREEN)✅ Setup complete! You can now run 'make start'$(NC)"

# Install dependencies
install:
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(BLUE)🚀 Installing MLX...$(NC)"
	pip install mlx
	@echo "$(BLUE)🔧 Installing mlx-lm in editable mode...$(NC)"
	pip install -e mlx-lm-main/
	@echo "$(BLUE)📚 Installing additional ML dependencies...$(NC)"
	pip install datasets transformers
	@echo "$(GREEN)✅ Installation complete!$(NC)"

# Test MLX installation
test:
	@echo "$(BLUE)🧪 Testing MLX installation...$(NC)"
	@$(PYTHON) -c "import mlx; print('✅ MLX imported successfully')" || (echo "$(RED)❌ MLX import failed$(NC)" && exit 1)
	@$(PYTHON) -c "import mlx_lm; print('✅ mlx-lm imported successfully')" || (echo "$(RED)❌ mlx-lm import failed$(NC)" && exit 1)
	@echo "$(GREEN)✅ MLX installation test passed!$(NC)"

# Run unit tests only (fast)
test-unit:
	@echo "$(BLUE)🧪 Running unit tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v -m "unit" --tb=short
	@echo "$(GREEN)✅ Unit tests completed!$(NC)"

# Run integration tests (slow, downloads models)
test-integration:
	@echo "$(BLUE)🧪 Running integration tests (this may take a while)...$(NC)"
	@echo "$(YELLOW)⏳ These tests download models and run actual training$(NC)"
	$(PYTHON) -m pytest tests/ -v -m "integration or slow or requires_model" --tb=short
	@echo "$(GREEN)✅ Integration tests completed!$(NC)"

# Run all tests with coverage report
test-coverage:
	@echo "$(BLUE)📊 Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=app --cov-report=html --cov-report=term --tb=short
	@echo "$(GREEN)✅ Tests with coverage completed!$(NC)"
	@echo "$(BLUE)📄 HTML coverage report: htmlcov/index.html$(NC)"

# Run all tests (unit + integration) with coverage
test-all: test test-coverage test-integration
	@echo "$(GREEN)🎉 All tests completed successfully!$(NC)"

# Run quick tests for development
test-quick:
	@echo "$(BLUE)⚡ Running quick tests...$(NC)"
	$(PYTHON) run_tests.py
	@echo "$(GREEN)✅ Quick tests completed!$(NC)"

# Run actual fine-tuning tests (downloads models, runs training)
test-training:
	@echo "$(BLUE)🤖 Running fine-tuning tests (this will download models and run training)...$(NC)"
	@echo "$(YELLOW)⏳ This may take several minutes...$(NC)"
	$(PYTHON) -m pytest tests/ -v -m "integration and requires_model" --tb=short -s
	@echo "$(GREEN)✅ Fine-tuning tests completed!$(NC)"

# Start the application
start:
	@echo "$(BLUE)🚀 Starting MLX Fine-tuning UI...$(NC)"
	@echo "$(YELLOW)📱 The app will open in your browser at http://localhost:8501$(NC)"
	@echo "$(YELLOW)⏹️  Press Ctrl+C to stop the application$(NC)"
	@echo ""
	streamlit run app.py

# Alias for start
run: start

# Stop running Streamlit processes
stop:
	@echo "$(BLUE)🛑 Stopping Streamlit processes...$(NC)"
	@pkill -f "streamlit run app.py" 2>/dev/null || echo "$(YELLOW)No Streamlit processes found$(NC)"
	@echo "$(GREEN)✅ Stopped successfully$(NC)"

# Check if app is running
status:
	@echo "$(BLUE)📊 Checking application status...$(NC)"
	@if pgrep -f "streamlit run app.py" > /dev/null; then \
		echo "$(GREEN)✅ MLX Fine-tuning UI is running$(NC)"; \
		echo "$(BLUE)🌐 Access at: http://localhost:8501$(NC)"; \
	else \
		echo "$(YELLOW)⏸️  MLX Fine-tuning UI is not running$(NC)"; \
		echo "$(BLUE)💡 Run 'make start' to launch the app$(NC)"; \
	fi

# Clean up temporary files
clean:
	@echo "$(BLUE)🧹 Cleaning up temporary files...$(NC)"
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -delete 2>/dev/null || true
	@find . -type f -name "*.log" -delete 2>/dev/null || true
	@find . -type f -name "*.tmp" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup complete!$(NC)"

# Development setup
dev-setup: install
	@echo "$(BLUE)🔧 Setting up development environment...$(NC)"
	pip install black flake8 pytest pytest-cov coverage
	@echo "$(GREEN)✅ Development setup complete!$(NC)"

# Format code
format:
	@echo "$(BLUE)🎨 Formatting code with black...$(NC)"
	@black app.py || echo "$(RED)❌ Black not installed. Run 'make dev-setup' first$(NC)"
	@echo "$(GREEN)✅ Code formatting complete!$(NC)"

# Lint code
lint:
	@echo "$(BLUE)🔍 Linting code with flake8...$(NC)"
	@flake8 app.py || echo "$(RED)❌ Flake8 not installed. Run 'make dev-setup' first$(NC)"
	@echo "$(GREEN)✅ Linting complete!$(NC)"

# Quick commands for common tasks
quick-install: install
	@echo "$(GREEN)✅ Quick install complete!$(NC)"

quick-start: test start
	@echo "$(GREEN)✅ Quick start complete!$(NC)"

# Show system info
info:
	@echo "$(BLUE)💻 System Information$(NC)"
	@echo "Python: $(shell python --version)"
	@echo "Platform: $(shell uname -s)"
	@echo "Architecture: $(shell uname -m)"
	@echo "Working Directory: $(shell pwd)"

# Show available models (placeholder)
models:
	@echo "$(BLUE)🤖 Available Models$(NC)"
	@echo "Popular models you can fine-tune:"
	@echo "  • mistralai/Mistral-7B-v0.1"
	@echo "  • meta-llama/Llama-2-7b"
	@echo "  • google/gemma-7b"
	@echo "  • microsoft/phi-2"
	@echo "  • And many more at huggingface.co"

# Show example usage
examples:
	@echo "$(BLUE)📚 Example Usage$(NC)"
	@echo "1. Setup: make setup"
	@echo "2. Start: make start"
	@echo "3. Stop: make stop"
	@echo "4. Check status: make status"
	@echo "5. Clean up: make clean"
