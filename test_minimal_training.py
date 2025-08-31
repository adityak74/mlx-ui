#!/usr/bin/env python3
"""
Minimal test script to verify that training works correctly.
This bypasses the Streamlit UI and tests the core training functionality directly.
"""

import sys
import os
from pathlib import Path
import types
import tempfile
import shutil

# Add mlx-lm-main to path for imports
sys.path.append(str(Path(__file__).parent / "mlx-lm-main"))

def test_minimal_training():
    """Test training using direct mlx-lm calls, bypassing our UI wrapper"""
    
    # Use the direct training approach from mlx-lm
    from mlx_lm.lora import train_model
    from mlx_lm.utils import load
    from mlx_lm.tuner.datasets import load_dataset
    
    # Create minimal config with all required attributes
    config = {
        "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "data": "example_dataset",  # Use the example dataset directly
        "fine_tune_type": "lora",
        "num_layers": 2,
        "max_seq_length": 256,
        "batch_size": 1,
        "iters": 3,
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
        "train": True,
        "test": True,
        "resume_adapter_file": None,
        "lr_schedule": None,
        "report_to": "",
        "wandb": None,
        "project_name": None,
        "optimizer_config": {"adam": {}, "adamw": {}, "muon": {}, "sgd": {}, "adafactor": {}},
        "lora_parameters": {"rank": 2, "scale": 1.0, "dropout": 0.0}
    }
    
    # Convert config to namespace
    args = types.SimpleNamespace(**config)
    
    # Create temporary adapter directory
    with tempfile.TemporaryDirectory() as temp_dir:
        adapter_path = Path(temp_dir) / "adapters"
        adapter_path.mkdir()
        args.adapter_path = str(adapter_path)
        
        print(f"Loading model: {args.model}")
        model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
        print("✅ Model loaded successfully")
        
        print("Loading dataset...")
        train_set, valid_set, test_set = load_dataset(args, tokenizer)
        print(f"✅ Dataset loaded: train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)}")
        
        print("Starting training...")
        # Create a proper callback class instead of a simple lambda
        class SimpleCallback:
            def on_train_loss_report(self, info):
                print(f"Iteration {info['iteration']}: train loss = {info['train_loss']:.6f}")
            
            def on_val_loss_report(self, info):
                print(f"Iteration {info['iteration']}: val loss = {info['val_loss']:.6f}")
            
            def on_train_end(self, info):
                print("Training completed!")
        
        training_callback = SimpleCallback()
        
        try:
            # Call train_model directly with our callback
            train_model(args, model, train_set, valid_set, training_callback)
            print("✅ Training completed successfully!")
            
            # Check if adapter files were created
            adapter_files = list(adapter_path.rglob("*.safetensors")) + \
                           list(adapter_path.rglob("*.bin")) + \
                           list(adapter_path.rglob("*.json"))
            
            if adapter_files:
                print(f"✅ Adapter files created: {[f.name for f in adapter_files]}")
                return True
            else:
                print("❌ No adapter files found after training")
                return False
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("🧪 Testing minimal training functionality...")
    success = test_minimal_training()
    if success:
        print("🎉 All tests passed! Training is working correctly.")
        sys.exit(0)
    else:
        print("💥 Training test failed.")
        sys.exit(1)
