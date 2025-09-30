# 🐱🥚-type1: GFlowNet-Fine-Tuned LLM for StrategyQA

An implementation of GFlowNet for optimizing reasoning chain generation on StrategyQA dataset.

## Overview

This project implements GFlowNet (Generative Flow Networks) to optimize large language model reasoning chains on the StrategyQA dataset. It supports both mock and real training modes with configurable parameters.

## Features
- **Better Reasoning**: Elevated reasoning chain generation by fine-tuning LLM with GFlowNet on StrategyQA dataset
- **Efficient Design**: Optimized for minimal resource usage
- **Dual Mode Support**: Mock mode for testing, real mode for training
- **Configurable**: JSON-based configuration system

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Sullivan07043/PersonalProject_GFlowNet4LLMs.git
cd PersonalProject_GFlowNet4LLMs

# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Test (Mock Mode)

```bash
# Run a quick test with mock components
python main.py --config config_mock.json --mode test
```

### 3. Real Training

```bash
# Run training with real components
python main.py --config config_real.json --mode experiment --experiment_name "real_experiment"
```

## Configuration

### Mock Configuration (`config_mock.json`)
- **Purpose**: Ultra-fast testing and development
- **Model**: Mock components (no real LLM)
- **Resources**: Minimal CPU usage
- **Time**: ~30 seconds for full test

### Real Configuration (`config_real.json`)
- **Purpose**: Real training with actual LLM
- **Model**: microsoft/DialoGPT-medium (~1GB)
- **Resources**: 16GB RAM, 20GB storage
- **Time**: 30-45 minutes for training

## Usage Examples

### Testing
```bash
# Quick mock test
python main.py --mode test --config config_mock.json

# Real component test
python main.py --mode test --config config_real.json --max_train_samples 50
```

### Training
```bash
# Mock training (for testing)
python main.py --mode train --config config_mock.json --num_epochs 2

# Real training
python main.py --mode train --config config_real.json --num_epochs 5

# Full training (requires more resources)
python main.py --mode train --model_name "microsoft/DialoGPT-medium" --num_epochs 20
```

### Evaluation
```bash
# Evaluate with mock components
python main.py --mode eval --config config_mock.json --max_eval_samples 20

# Evaluate with real components
python main.py --mode eval --config config_real.json --max_eval_samples 100
```

### Complete Experiment
```bash
# Run full experiment pipeline
python main.py --mode experiment --config config_real.json --experiment_name "my_experiment"
```

### Inference
```bash
# Interactive inference mode
python inference_example.py --interactive

# Single question inference
python inference_example.py --question "Did the Hopi Indians use a symbol that was similar to the swastika?"

# Batch inference from file
python inference_example.py --questions_file example_questions.txt --output results.json

# Run example questions
python inference_example.py
```

## Project Structure

```
PersonalProject/
├── main.py              # Main entry point
├── inference_example.py # Inference example script
├── config_mock.json     # Mock configuration
├── config_real.json     # Real configuration
├── requirements.txt     # Dependencies
├── README.md           # This file
├── data/               # Dataset handling
│   └── dataset.py      # StrategyQA dataset loader
├── llm/                # LLM generation
│   ├── generate.py     # LLM generator
│   └── promt.py        # Prompt templates
├── gfn/                # GFlowNet components
│   ├── model.py        # Policy network
│   ├── sampler.py      # Trajectory sampler
│   ├── losses.py       # Loss functions
│   └── train.py        # Training loop
├── evaluation/         # Evaluation system
│   ├── eval.py         # Evaluator
│   └── metrics.py      # Metrics calculation
├── results/            # Experiment results
└── checkpoints/        # Model checkpoints
```

## System Requirements

### Minimum Requirements (Mock Mode)
- **RAM**: 4GB
- **Storage**: 5GB
- **Python**: 3.8+
- **OS**: Linux, macOS, Windows

### Full Training Requirements (Real Mode)
- **RAM**: 16GB+
- **Storage**: 20GB+
- **GPU**: A100 GPU with 70GB+ VRAM
- **Python**: 3.10+

## Dependencies

### Core Dependencies
- `torch>=2.0.0` - PyTorch framework
- `transformers>=4.40.0` - HuggingFace transformers
- `datasets>=2.20.0` - Dataset handling
- `accelerate>=0.28.0` - Training acceleration
- `tqdm` - Progress bars
- `numpy` - Numerical computing
- `pandas` - Data manipulation

### Optional Dependencies
- `scikit-learn` - Machine learning utilities
- `matplotlib` - Plotting
- `tensorboard` - Training visualization
- `bitsandbytes` - Model quantization

## Performance Benchmarks

| Mode | Time | Memory | Storage | Model Size |
|------|------|--------|---------|------------|
| Mock | 30s | 1GB | 2GB | 0MB |
| Real | 30-45min | 16GB | 20GB | 1GB |
| Full | 1-2h | 16GB | 32GB | 1-5GB |

## Command Line Options

### Basic Options
- `--mode`: Run mode (test, train, eval, experiment)
- `--config`: Configuration file path
- `--experiment_name`: Name for experiment
- `--use_mock`: Use mock components

### Data Options
- `--max_train_samples`: Maximum training samples
- `--max_eval_samples`: Maximum evaluation samples
- `--batch_size`: Batch size
- `--train_split`: Training data split
- `--eval_split`: Evaluation data split

### Training Options
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--model_name`: LLM model name
- `--device`: Device (auto, cpu, cuda)

### GFlowNet Options
- `--hidden_dim`: Hidden dimension
- `--num_layers`: Number of layers
- `--max_steps`: Maximum reasoning steps
- `--temperature`: Sampling temperature
- `--num_candidates`: Number of action candidates

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size and model size
   python main.py --batch_size 2 --model_name "microsoft/DialoGPT-small"
   ```

2. **Slow Training**
   ```bash
   # Use mock mode for testing
   python main.py --use_mock --mode test
   ```

3. **Model Loading Issues**
   ```bash
   # Check internet connection and model name
   python main.py --model_name "microsoft/DialoGPT-medium"
   ```

4. **CUDA Issues**
   ```bash
   # Force CPU usage
   python main.py --device cpu
   ```

### Performance Tips

1. **Use Mock Mode**: For development and testing
2. **Reduce Parameters**: Lower `hidden_dim`, `num_layers`, `max_steps`
3. **Smaller Batches**: Reduce `batch_size` for memory constraints
4. **Fewer Samples**: Limit `max_train_samples` and `max_eval_samples`
5. **GPU Acceleration**: Use `--device cuda` when available

## Results and Outputs

### Generated Files
- `results/experiment_report_*.md` - Experiment reports
- `results/training_history_*.json` - Training metrics
- `results/evaluation_results_*.json` - Evaluation results
- `checkpoints/checkpoint_*.json` - Model checkpoints
- `gfn_project.log` - Training logs

### Example Output
```
Epoch 0: Loss=0.5234, Reward=0.6789, Acc=0.7500
Epoch 1: Loss=0.4567, Reward=0.7123, Acc=0.8000
Epoch 2: Loss=0.3890, Reward=0.7456, Acc=0.8500
```

## Using Trained Models

### Inference Example
The `inference_example.py` script provides a complete example of how to use trained GFlowNet models for reasoning:

```python
from inference_example import GFlowNetInference

# Initialize inference system
inference = GFlowNetInference("config_real.json")

# Generate reasoning for a question
result = inference.generate_reasoning(
    "Did the Hopi Indians use a symbol that was similar to the swastika?"
)

print(f"Question: {result['question']}")
print(f"Answer: {result['final_answer']}")
print(f"Reasoning steps: {result['num_steps']}")
```

### Inference Modes

1. **Interactive Mode**: Real-time question answering
2. **Single Question**: Process one question with detailed output
3. **Batch Processing**: Process multiple questions from a file
4. **Example Questions**: Run built-in example questions

### Output Format
```json
{
  "question": "Did the Hopi Indians use a symbol that was similar to the swastika?",
  "reasoning_steps": [
    "Step 1: Understanding the question",
    "Step 2: Researching Hopi culture",
    "Step 3: Analyzing symbol similarities"
  ],
  "final_answer": "Yes",
  "num_steps": 3,
  "trajectory_data": [...]
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [shz127@gmail.com].
