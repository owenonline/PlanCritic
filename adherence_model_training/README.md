# Adherence Model Training

This directory contains tools for training reward models that evaluate whether AI-generated plans adhere to natural language constraints in PDDL (Planning Domain Definition Language) environments.

## Overview

The reward model training pipeline consists of several stages:
1. **Data Generation**: Create training data with plans and adherence feedback
2. **Situation Information**: Generate natural language descriptions of problem environments
3. **Data Embedding**: Convert text data to numerical embeddings
4. **Model Training**: Train LSTM-based reward models on embedded data

## Prerequisites

Install required dependencies:
```bash
pip install -r requirements.txt
```

Set your OpenAI API key (required for embedding and text generation):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Set up Weights & Biases for experiment tracking:
```bash
wandb login
```

## File Descriptions

- **`create_training_data.py`**: Generates synthetic training data with plans and constraint adherence labels
- **`create_situation_info.py`**: Creates natural language descriptions of PDDL problem instances
- **`embed_data_api.py`**: Converts text data to numerical embeddings using OpenAI's embedding API
- **`train_reward_model.py`**: Trains reward model using distributed training (CUDA-based)
- **`train_reward_model_dynamic.py`**: Adaptive training script that works on CUDA/MPS/CPU
- **`requirements.txt`**: Python dependencies
- **`reward_model_embedded_data/`**: Directory for storing embedded training data

## Usage Workflow

### Step 1: Create Situation Information
Generate natural language descriptions for your domain's problem instances:

```bash
python create_situation_info.py --domain <domain_name>
```

This script:
- Reads PDDL domain and problem files from `/workspace/domains/<domain_name>/feedback/`
- Uses GPT-4 to generate situation descriptions
- Saves descriptions as `situation_info.txt` in each problem directory

### Step 2: Generate Training Data
Create synthetic training data with plan-constraint pairs:

```bash
python create_training_data.py --domain <domain_name> --problem <problem_name>
```

This script:
- Generates random PDDL constraints for the given problem
- Tests constraint solvability using the OPTIC planner
- Creates both positive (constraint-satisfying) and negative (constraint-violating) examples
- Converts constraints to natural language using GPT-4
- Saves training data as `v5data.json` files

**Key Features:**
- Creates balanced datasets with positive/negative examples
- Ensures constraints are meaningful (not vacuously true)
- Generates semantically similar but incorrect constraints via mutation
- Validates plan adherence using formal verification

### Step 3: Embed Training Data
Convert text data to numerical embeddings:

```bash
python embed_data_api.py --domain <domain_name>
```

This script:
- Loads training data from all problem instances in the domain
- Uses OpenAI's `text-embedding-3-small` model to embed:
  - Planning objectives (natural language constraints)
  - Plan steps (individual actions)
- Combines objective and plan embeddings
- Saves padded sequences as `reward_model_embedded_data/<domain_name>.npy`

### Step 4: Train the Reward Model
Train the LSTM-based reward model:

**Option A: Dynamic Training (Recommended)**
```bash
python train_reward_model_dynamic.py --domain <domain_name>
```

**Option B: CUDA-specific Training**
```bash
python train_reward_model.py --domain <domain_name>
```

The dynamic version automatically selects the best available device (CUDA > MPS > CPU).

**Model Architecture:**
- LSTM network for processing sequential plan data
- Input: Concatenated embeddings of constraints and plan steps
- Output: Scalar adherence score (0-1)
- Features: Dropout regularization, Xavier initialization, sigmoid activation

## Data Format

### Training Data Structure (`v5data.json`)
```json
[
  {
    "plan": [
      {"action": "move robot1 room1 room2", "time_step": 0, "duration": 1.0},
      ...
    ],
    "feedback": [
      {
        "feedback": "Ensure robot1 visits room3 before room2",
        "constraint": "(sometime-before (at robot1 room2) (at robot1 room3))",
        "obeyed": 1
      },
      ...
    ]
  },
  ...
]
```

### Embedded Data Format
- Numpy array shape: `(num_examples, max_sequence_length, embedding_dim)`
- Each sequence combines plan step embeddings with repeated constraint embeddings
- Zero-padded to handle variable-length plans

## Configuration

### Key Parameters
- **MAX_LEN**: Maximum sequence length for padding (default: 2048)
- **Embedding Model**: `text-embedding-3-small` for cost-effectiveness
- **Base Model**: `intfloat/e5-base-v2` for sentence embeddings (in training scripts)
- **Target Data Size**: 500 examples per problem instance

### Directory Structure Expected
```
/workspace/domains/<domain_name>/
├── domain.pddl
└── feedback/
    ├── <problem1>/
    │   ├── <problem1>.pddl
    │   ├── situation_info.txt
    │   └── v5data.json
    └── <problem2>/
        └── ...
```

## Monitoring and Logging

Training progress is logged to Weights & Biases with:
- Loss curves
- Validation metrics
- Model checkpoints
- Hardware utilization

## Troubleshooting

### Common Issues
1. **OPTIC Planner Not Found**: Ensure `/workspace/binaries/optic-cplex` exists
2. **Validation Binary Missing**: Ensure `/workspace/binaries/Validate` exists
3. **Memory Issues**: Reduce batch size or sequence length for large domains
4. **API Rate Limits**: The embedding script includes retry logic and threading

### Performance Tips
- Use the dynamic training script for better hardware compatibility
- For large domains, consider splitting data generation across multiple runs
- Monitor GPU memory usage during embedding generation

## Output

The trained model will be saved as PyTorch checkpoints in the `reward_model_lstm` directory, ready for use in plan evaluation and constraint adherence scoring.
