import streamlit as st
import os
import sys
import yaml
import json
import time
import queue
import uuid
from pathlib import Path
import tempfile
import shutil
from typing import Optional, Dict, Any

# Add mlx-lm-main to path for imports
sys.path.append(str(Path(__file__).parent / "mlx-lm-main"))

st.set_page_config(
    page_title="MLX Fine-tuning UI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .config-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def check_mlx_installation():
    """Check if MLX and mlx-lm are properly installed"""
    try:
        import mlx
        import mlx_lm
        return True, "MLX and mlx-lm are installed"
    except ImportError as e:
        return False, f"Import error: {str(e)}"

def create_fine_tuning_config():
    """Create fine-tuning configuration form"""
    st.header("🔧 Fine-tuning Configuration")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Settings")
            model_path = st.text_input(
                "Model Path/Repo",
                placeholder="Path to local model or HuggingFace repo (e.g., mistralai/Mistral-7B-v0.1)",
                help="Local path to model directory or HuggingFace repository name"
            )
            
            fine_tune_type = st.selectbox(
                "Fine-tuning Type",
                ["lora", "dora", "full"],
                help="LoRA: Low-rank adaptation, DoRA: Dropout LoRA, Full: Full fine-tuning"
            )
            
            num_layers = st.number_input(
                "Number of Layers to Fine-tune",
                min_value=-1,
                max_value=100,
                value=16,
                help="Number of layers to fine-tune. Use -1 for all layers"
            )
            
            max_seq_length = st.number_input(
                "Maximum Sequence Length",
                min_value=512,
                max_value=8192,
                value=2048,
                step=512,
                help="Maximum sequence length for training"
            )
        
        with col2:
            st.subheader("Training Settings")
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=32,
                value=4,
                help="Training batch size"
            )
            
            iters = st.number_input(
                "Training Iterations",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="Total number of training iterations"
            )
            
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-7,
                max_value=1e-2,
                value=1e-5,
                format="%.2e",
                help="Training learning rate"
            )
            
            optimizer = st.selectbox(
                "Optimizer",
                ["adam", "adamw", "muon", "sgd", "adafactor"],
                help="Optimizer for training"
            )
    
    # LoRA specific parameters
    if fine_tune_type in ["lora", "dora"]:
        st.subheader("LoRA Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lora_rank = st.number_input(
                "LoRA Rank",
                min_value=1,
                max_value=64,
                value=8,
                help="Rank of LoRA matrices"
            )
        
        with col2:
            lora_scale = st.number_input(
                "LoRA Scale",
                min_value=0.1,
                max_value=100.0,
                value=20.0,
                step=0.1,
                help="Scaling factor for LoRA weights"
            )
        
        with col3:
            lora_dropout = st.number_input(
                "LoRA Dropout",
                min_value=0.0,
                max_value=0.5,
                value=0.0,
                step=0.1,
                help="Dropout rate for LoRA layers"
            )
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            val_batches = st.number_input(
                "Validation Batches",
                min_value=-1,
                max_value=100,
                value=25,
                help="Number of validation batches (-1 for entire validation set)"
            )
            
            steps_per_report = st.number_input(
                "Steps per Report",
                min_value=1,
                max_value=100,
                value=10,
                help="Report training loss every N steps"
            )
            
            steps_per_eval = st.number_input(
                "Steps per Evaluation",
                min_value=50,
                max_value=500,
                value=200,
                help="Run validation every N steps"
            )
        
        with col2:
            save_every = st.number_input(
                "Save Every N Steps",
                min_value=50,
                max_value=500,
                value=100,
                help="Save model every N training steps"
            )
            
            grad_checkpoint = st.checkbox(
                "Gradient Checkpointing",
                help="Use gradient checkpointing to reduce memory usage"
            )
            
            mask_prompt = st.checkbox(
                "Mask Prompt in Loss",
                help="Mask the prompt tokens when computing training loss"
            )
    
    return {
        "model": model_path,
        "fine_tune_type": fine_tune_type,
        "num_layers": num_layers,
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "iters": iters,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "val_batches": val_batches,
        "steps_per_report": steps_per_report,
        "steps_per_eval": steps_per_eval,
        "save_every": save_every,
        "grad_checkpoint": grad_checkpoint,
        "mask_prompt": mask_prompt,
        "seed": 42,  # Add seed for reproducibility
        "test_batches": 500,  # Add test_batches
        "resume_adapter_file": None,  # Add resume option
        "lr_schedule": None,  # Add lr_schedule option
        "wandb": None,  # Add wandb option
    "report_to": "",  # Add report_to option
        "project_name": None,  # Add project_name option
        "optimizer_config": {  # Add optimizer_config that train_model expects
            "adam": {},
            "adamw": {},
            "muon": {},
            "sgd": {},
            "adafactor": {},
        },
        "lora_parameters": {
            "rank": lora_rank if fine_tune_type in ["lora", "dora"] else 8,
            "scale": lora_scale if fine_tune_type in ["lora", "dora"] else 20.0,
            "dropout": lora_dropout if fine_tune_type in ["lora", "dora"] else 0.0
        } if fine_tune_type in ["lora", "dora"] else None
    }

def create_dataset_config():
    """Create dataset configuration form"""
    st.header("📊 Dataset Configuration")
    
    dataset_type = st.radio(
        "Dataset Source",
        ["Local JSONL Files", "HuggingFace Dataset"],
        help="Choose between local files or HuggingFace dataset"
    )
    
    if dataset_type == "Local JSONL Files":
        st.subheader("Local Dataset Files")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_file = st.file_uploader(
                "Training Data (train.jsonl)",
                type=["jsonl"],
                help="Upload training data in JSONL format"
            )
        
        with col2:
            valid_file = st.file_uploader(
                "Validation Data (valid.jsonl)",
                type=["jsonl"],
                help="Upload validation data in JSONL format"
            )
        
        with col3:
            test_file = st.file_uploader(
                "Test Data (test.jsonl)",
                type=["jsonl"],
                help="Upload test data in JSONL format (optional)"
            )
        
        # Show dataset preview
        if train_file:
            st.subheader("Dataset Preview")
            try:
                import pandas as pd
                df = pd.read_json(train_file, lines=True)
                st.dataframe(df.head(), use_container_width=True)
                st.info(f"Training dataset: {len(df)} examples")
            except Exception as e:
                st.error(f"Error reading dataset: {str(e)}")
        
        return {
            "type": "local",
            "train_file": train_file,
            "valid_file": valid_file,
            "test_file": test_file
        }
    
    else:
        st.subheader("HuggingFace Dataset")
        hf_dataset = st.text_input(
            "Dataset Name",
            placeholder="e.g., mlx-community/wikisql",
            help="HuggingFace dataset identifier"
        )
        
        st.info("💡 **Tip**: HuggingFace datasets can sometimes have compatibility issues. For best results, try using local JSONL files first.")
        st.info("📚 **Recommended**: Start with the example dataset provided in the `example_dataset/` folder")
        
        return {
            "type": "huggingface",
            "dataset_name": hf_dataset
        }

def save_config_to_file(config: Dict[str, Any], file_path: str):
    """Save configuration to YAML file"""
    try:
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        st.error(f"Error saving config: {str(e)}")
        return False

def test_dataset_loading(config: Dict[str, Any], dataset_config: Dict[str, Any]):
    """Test if the dataset can be loaded properly"""
    st.header("🧪 Testing Dataset Loading")
    
    try:
        # Create a temporary directory for testing
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Handle dataset files
            if dataset_config["type"] == "local":
                data_dir = temp_path / "data"
                data_dir.mkdir(exist_ok=True)
                
                # Save uploaded files
                if dataset_config["train_file"]:
                    with open(data_dir / "train.jsonl", "wb") as f:
                        f.write(dataset_config["train_file"].getvalue())
                
                if dataset_config["valid_file"]:
                    with open(data_dir / "valid.jsonl", "wb") as f:
                        f.write(dataset_config["valid_file"].getvalue())
                
                if dataset_config["test_file"]:
                    with open(data_dir / "test.jsonl", "wb") as f:
                        f.write(dataset_config["test_file"].getvalue())
                
                config["data"] = str(data_dir)
            else:
                config["data"] = dataset_config["dataset_name"]
            
            # Update config for testing
            config["train"] = True
            config["test"] = True
            
            st.info("Testing dataset loading...")
            
            # Import and test
            from mlx_lm.tuner.datasets import load_dataset
            from mlx_lm.utils import load
            import types
            
            # Convert config to namespace
            args = types.SimpleNamespace(**config)
            
            # Load the model and tokenizer
            st.info("Loading model and tokenizer...")
            model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
            st.success("✅ Model and tokenizer loaded successfully")
            
            # Load the dataset
            st.info("Loading dataset...")
            train_set, valid_set, test_set = load_dataset(args, tokenizer)
            st.success(f"✅ Dataset loaded successfully!")
            st.info(f"Training samples: {len(train_set)}")
            st.info(f"Validation samples: {len(valid_set)}")
            st.info(f"Test samples: {len(test_set)}")
            
            # Test a sample
            if len(train_set) > 0:
                st.info("Testing first training sample...")
                sample = train_set[0]
                st.success(f"✅ Sample processed: {type(sample).__name__}")
                
                # Show sample info
                if hasattr(sample, 'process'):
                    processed = sample.process(sample[0])
                    st.info(f"Processed sample: {len(processed[0])} tokens")
            
            st.success("🎉 Dataset test completed successfully! You can now start fine-tuning.")
            
    except Exception as e:
        st.error(f"❌ Dataset test failed: {str(e)}")
        import traceback
        st.text_area("Error Details:", traceback.format_exc(), height=200)


def run_fine_tuning(config: Dict[str, Any], dataset_config: Dict[str, Any]):
    """Run the fine-tuning process"""
    st.header("🚀 Starting Fine-tuning")
    
    # Create a permanent directory for training (not temporary)
    import uuid
    training_id = str(uuid.uuid4())[:8]
    training_dir = Path(f"training_{training_id}")
    training_dir.mkdir(exist_ok=True)
    
    try:
        # Save configuration
        config_path = training_dir / "config.yaml"
        if not save_config_to_file(config, str(config_path)):
            return False
        
        # Handle dataset files
        if dataset_config["type"] == "local":
            data_dir = training_dir / "data"
            data_dir.mkdir(exist_ok=True)
            
            # Save uploaded files with correct names
            if dataset_config["train_file"]:
                with open(data_dir / "train.jsonl", "wb") as f:
                    f.write(dataset_config["train_file"].getvalue())
                st.success(f"✅ Training data saved: {len(dataset_config['train_file'].getvalue())} bytes")
            
            if dataset_config["valid_file"]:
                with open(data_dir / "valid.jsonl", "wb") as f:
                    f.write(dataset_config["valid_file"].getvalue())
                st.success(f"✅ Validation data saved: {len(dataset_config['valid_file'].getvalue())} bytes")
            
            if dataset_config["test_file"]:
                with open(data_dir / "test.jsonl", "wb") as f:
                    f.write(dataset_config["test_file"].getvalue())
                st.success(f"✅ Test data saved: {len(dataset_config['test_file'].getvalue())} bytes")
            
            config["data"] = str(data_dir)
            
            # Verify dataset files exist and check format
            st.info("Verifying dataset files...")
            if (data_dir / "train.jsonl").exists():
                st.success("✅ train.jsonl exists")
                
                # Check dataset format
                try:
                    with open(data_dir / "train.jsonl", 'r') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            sample_data = json.loads(first_line)
                            if "text" in sample_data:
                                st.success("✅ Dataset format: text-based (correct)")
                            elif "prompt" in sample_data and "completion" in sample_data:
                                st.success("✅ Dataset format: prompt-completion (correct)")
                            elif "messages" in sample_data:
                                st.success("✅ Dataset format: chat-based (correct)")
                            else:
                                st.error("❌ Dataset format not supported. Expected: text, prompt+completion, or messages")
                                st.info("Supported formats:")
                                st.info("• {\"text\": \"your text here\"}")
                                st.info("• {\"prompt\": \"question\", \"completion\": \"answer\"}")
                                st.info("• {\"messages\": [{\"role\": \"user\", \"content\": \"...\"}]}")
                                return False
                except Exception as e:
                    st.error(f"❌ Error reading dataset format: {str(e)}")
                    return False
            else:
                st.error("❌ train.jsonl not found")
                return False
                
            if (data_dir / "valid.jsonl").exists():
                st.success("✅ valid.jsonl exists")
            else:
                st.error("❌ valid.jsonl not found")
                return False
                
            if (data_dir / "test.jsonl").exists():
                st.success("✅ test.jsonl exists")
            else:
                st.warning("⚠️ test.jsonl not found (optional)")
        else:
            config["data"] = dataset_config["dataset_name"]
            st.info(f"Using HuggingFace dataset: {dataset_config['dataset_name']}")
        
        # Update config for training
        config["train"] = True
        config["test"] = True
        config["adapter_path"] = str(training_dir / "adapters")
        
        # Create adapters directory
        Path(config["adapter_path"]).mkdir(parents=True, exist_ok=True)
        
        # Display final configuration
        st.subheader("Final Configuration")
        st.json(config)
        
        # Run training
        try:
            st.info("Starting fine-tuning process...")
            st.info(f"Training directory: {training_dir}")
            
            # Import and run training
            from mlx_lm.lora import run
            import types
            
            # Convert config to namespace
            args = types.SimpleNamespace(**config)
            
            # Run training in a separate thread to avoid blocking UI
            import threading
            import queue
            
            # Create a queue for progress updates
            progress_queue = queue.Queue()
            
            def training_thread():
                try:
                    # Capture stdout to get training progress
                    import io
                    import sys
                    from contextlib import redirect_stdout
                    
                    # Redirect stdout to capture training output
                    output = io.StringIO()
                    with redirect_stdout(output):
                        print(f"Starting training with args: {vars(args)}")
                        print(f"Training directory: {training_dir}")
                        print(f"Data path: {args.data}")
                        print(f"Adapter path: {args.adapter_path}")
                        
                        # Import required modules
                        print("Importing required modules...")
                        from mlx_lm.tuner.datasets import load_dataset
                        from mlx_lm.utils import load
                        from mlx_lm.tuner.trainer import TrainingArgs
                        from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
                        from mlx_lm.tuner.callbacks import get_reporting_callbacks
                        import mlx.optimizers as optim
                        print("Modules imported successfully")
                        
                        # Load the model and tokenizer
                        print("Loading model and tokenizer...")
                        model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
                        print("Model and tokenizer loaded successfully")
                        print(f"Model type: {type(model)}")
                        print(f"Model layers: {len(model.layers) if hasattr(model, 'layers') else 'No layers attribute'}")
                        
                        # Load the dataset
                        print("Loading dataset...")
                        train_set, valid_set, test_set = load_dataset(args, tokenizer)
                        print(f"Dataset loaded: train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)}")
                        
                        # Verify dataset objects
                        print(f"Train set type: {type(train_set)}")
                        print(f"Valid set type: {type(valid_set)}")
                        if len(train_set) > 0:
                            print(f"First training sample: {train_set[0]}")
                        
                        # Use the same approach as the working command line version
                        print("Setting up training using train_model function...")
                        from mlx_lm.lora import train_model
                        
                        # Call train_model which handles all the setup automatically
                        print("Starting actual training...")
                        print(f"Training for {args.iters} iterations with batch size {args.batch_size}")

                        # Create a proper callback class instead of using get_reporting_callbacks
                        class TrainingCallback:
                            def on_train_loss_report(self, info):
                                print(f"Iteration {info['iteration']}: train loss = {info['train_loss']:.6f}")
                            
                            def on_val_loss_report(self, info):
                                print(f"Iteration {info['iteration']}: val loss = {info['val_loss']:.6f}")
                            
                            def on_train_end(self, info):
                                print("Training completed!")
                        
                        training_callback = TrainingCallback()

                        # train_model handles all the setup: LoRA layers, optimizer, training args, etc.
                        train_model(args, model, train_set, valid_set, training_callback)
                        
                        print("Training completed successfully!")
                    
                    # Send completion signal
                    progress_queue.put(("complete", output.getvalue()))
                    
                except Exception as e:
                    import traceback
                    error_msg = f"Training error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                    print(f"ERROR in training thread: {error_msg}")  # Also print to console
                    progress_queue.put(("error", error_msg))
            
            # Start training thread
            thread = threading.Thread(target=training_thread)
            thread.daemon = True
            thread.start()
            
            # Monitor progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Wait for training to complete with timeout
            import time
            start_time = time.time()
            timeout_seconds = 300  # 5 minutes timeout
            
            while thread.is_alive():
                try:
                    # Check for progress updates
                    try:
                        msg_type, msg_data = progress_queue.get_nowait()
                        if msg_type == "complete":
                            st.success("Fine-tuning completed successfully!")
                            st.text_area("Training Output:", msg_data, height=200)
                            break
                        elif msg_type == "error":
                            st.error(f"Training error: {msg_data}")
                            return False
                    except queue.Empty:
                        pass
                    
                    # Check timeout
                    elapsed_time = time.time() - start_time
                    if elapsed_time > timeout_seconds:
                        st.error(f"Training timed out after {timeout_seconds} seconds")
                        st.info("This might indicate an issue with the model or dataset")
                        return False
                    
                    # Update progress
                    time.sleep(1)
                    
                    # Show that training is in progress with elapsed time
                    status_text.text(f"Training in progress... Elapsed time: {elapsed_time:.1f}s")
                    
                except Exception as e:
                    st.error(f"Error monitoring training: {str(e)}")
                    return False
            
            # Wait for thread to finish
            thread.join(timeout=5)
            
            # Check results
            if Path(config["adapter_path"]).exists():
                # Look for actual model files
                model_files = list(Path(config["adapter_path"]).rglob("*.safetensors")) + \
                             list(Path(config["adapter_path"]).rglob("*.bin")) + \
                             list(Path(config["adapter_path"]).rglob("*.json"))
                
                if model_files:
                    st.subheader("🎉 Training Results")
                    
                    # List the files that were created
                    st.write("Files created:")
                    for file_path in model_files:
                        st.write(f"- {file_path.name}")
                    
                    # Create zip file
                    import zipfile
                    zip_path = training_dir / "fine_tuned_model.zip"
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for file_path in model_files:
                            zipf.write(file_path, file_path.relative_to(training_dir))
                    
                    # Download button
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            label="Download Fine-tuned Model",
                            data=f.read(),
                            file_name="fine_tuned_model.zip",
                            mime="application/zip"
                        )
                    
                    st.info(f"Training files are saved in: {training_dir}")
                    st.info("You can also manually copy the files from this directory.")
                    return True
                else:
                    st.warning("⚠️ Training completed but no model files were created!")
                    st.error("This usually means the training failed or the dataset is incompatible.")
                    st.info("Common issues:")
                    st.info("• Dataset format is not supported")
                    st.info("• Model failed to load")
                    st.info("• Training completed too quickly (check dataset size)")
                    st.info(f"Check the training directory: {training_dir}")
                    
                    # Show what's in the adapters directory
                    st.write("Contents of adapters directory:")
                    for item in Path(config["adapter_path"]).iterdir():
                        st.write(f"- {item.name} ({'dir' if item.is_dir() else 'file'})")
                    
                    return False
            else:
                st.warning("No adapter files found. Training may have failed.")
                st.info(f"Check the training directory: {training_dir}")
                return False
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            st.info(f"Check the training directory: {training_dir}")
            return False
            
    except Exception as e:
        st.error(f"Error setting up training: {str(e)}")
        return False

def create_inference_interface():
    """Create interface for running inference with fine-tuned models"""
    st.header("🤖 Model Inference")
    
    model_source = st.radio(
        "Model Source",
        ["Fine-tuned Model", "HuggingFace Model", "Local Model Path"],
        help="Choose the source of your model"
    )
    
    if model_source == "Fine-tuned Model":
        st.subheader("Load Fine-tuned Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            base_model = st.text_input(
                "Base Model Path/Repo",
                placeholder="Path to base model or HuggingFace repo",
                help="Path to the base model that was fine-tuned"
            )
        
        with col2:
            adapter_path = st.text_input(
                "Adapter Path",
                placeholder="Path to fine-tuned adapters",
                help="Path to the fine-tuned adapter weights"
            )
        
        if st.button("Load Fine-tuned Model"):
            if base_model and adapter_path:
                try:
                    # Load model with adapters
                    from mlx_lm.utils import load
                    from mlx_lm.tuner.utils import load_adapters
                    
                    model, tokenizer = load(base_model)
                    load_adapters(model, adapter_path)
                    
                    st.success("Fine-tuned model loaded successfully!")
                    return model, tokenizer
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    return None, None
    
    elif model_source == "HuggingFace Model":
        st.subheader("Load HuggingFace Model")
        hf_model = st.text_input(
            "Model Name",
            placeholder="e.g., mistralai/Mistral-7B-v0.1",
            help="HuggingFace model identifier"
        )
        
        if st.button("Load Model"):
            if hf_model:
                try:
                    from mlx_lm.utils import load
                    model, tokenizer = load(hf_model)
                    st.success("Model loaded successfully!")
                    return model, tokenizer
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    return None, None
    
    else:  # Local Model Path
        st.subheader("Load Local Model")
        local_path = st.text_input(
            "Local Model Path",
            placeholder="/path/to/local/model",
            help="Path to local model directory"
        )
        
        if st.button("Load Model"):
            if local_path:
                try:
                    from mlx_lm.utils import load
                    model, tokenizer = load(local_path)
                    st.success("Model loaded successfully!")
                    return model, tokenizer
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    return None, None
    
    return None, None

def run_inference(model, tokenizer):
    """Run inference with loaded model"""
    if model is None or tokenizer is None:
        st.warning("Please load a model first")
        return
    
    st.subheader("Generate Text")
    
    # Input options
    input_type = st.radio(
        "Input Type",
        ["Single Prompt", "Chat Mode", "Batch Generation"]
    )
    
    if input_type == "Single Prompt":
        prompt = st.text_area(
            "Enter your prompt:",
            height=100,
            placeholder="Enter your text prompt here..."
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=2048,
                value=100,
                help="Maximum number of tokens to generate"
            )
        
        with col2:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="Sampling temperature (higher = more random)"
            )
        
        with col3:
            top_p = st.slider(
                "Top-p",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.1,
                help="Nucleus sampling parameter"
            )
        
        if st.button("Generate") and prompt:
            try:
                with st.spinner("Generating..."):
                    # Generate response
                    from mlx_lm.generate import generate
                    
                    response = generate(
                        model,
                        tokenizer,
                        prompt,
                        max_tokens=max_tokens,
                        temp=temperature,
                        top_p=top_p
                    )
                    
                    st.subheader("Generated Response:")
                    st.write(response)
                    
            except Exception as e:
                st.error(f"Generation error: {str(e)}")
    
    elif input_type == "Chat Mode":
        st.info("Chat mode coming soon!")
    
    else:  # Batch Generation
        st.info("Batch generation coming soon!")

def main():
    st.markdown('<h1 class="main-header">MLX Fine-tuning UI</h1>', unsafe_allow_html=True)
    
    # Check MLX installation
    mlx_ok, mlx_status = check_mlx_installation()
    if mlx_ok:
        st.success(mlx_status)
    else:
        st.error(mlx_status)
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔧 Fine-tuning", "🤖 Inference", "📊 Training Status"]
    )
    
    if page == "🏠 Home":
        st.header("Welcome to MLX Fine-tuning UI! 🚀")
        
        st.markdown("""
        This application makes it easy to fine-tune large language models using Apple's MLX framework.
        
        ### Features:
        - **Easy Fine-tuning**: Configure and run LoRA, DoRA, or full fine-tuning
        - **Dataset Support**: Use local JSONL files or HuggingFace datasets
        - **Model Inference**: Run inference with fine-tuned models
        - **User-friendly Interface**: Simple forms and real-time progress tracking
        
        ### Quick Start:
        1. Go to the **Fine-tuning** page to configure your training
        2. Upload your dataset or specify a HuggingFace dataset
        3. Start training and monitor progress
        4. Use the **Inference** page to test your fine-tuned model
        
        ### Supported Models:
        - Mistral, Llama, Gemma, and many more
        - Local models and HuggingFace repositories
        - LoRA, DoRA, and full fine-tuning approaches
        """)
        
        # System info
        st.subheader("System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Python Version", sys.version.split()[0])
            st.metric("MLX Available", "✅" if mlx_ok else "❌")
        
        with col2:
            import platform
            st.metric("Platform", platform.system())
            st.metric("Architecture", platform.machine())
    
    elif page == "🔧 Fine-tuning":
        st.header("Fine-tune Your Model")
        
        # Configuration tabs
        tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📊 Dataset", "🚀 Start Training"])
        
        with tab1:
            config = create_fine_tuning_config()
        
        with tab2:
            dataset_config = create_dataset_config()
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🧪 Test Dataset", type="secondary"):
                    if config["model"] and dataset_config:
                        test_dataset_loading(config, dataset_config)
                    else:
                        st.error("Please complete all required fields in the Configuration and Dataset tabs.")
            
            with col2:
                if st.button("🚀 Start Fine-tuning", type="primary"):
                    if config["model"] and dataset_config:
                        run_fine_tuning(config, dataset_config)
                    else:
                        st.error("Please complete all required fields in the Configuration and Dataset tabs.")
    
    elif page == "🤖 Inference":
        st.header("Run Model Inference")
        
        model, tokenizer = create_inference_interface()
        run_inference(model, tokenizer)
    
    elif page == "📊 Training Status":
        st.header("Training Status")
        st.info("Training status monitoring coming soon!")

if __name__ == "__main__":
    main()
