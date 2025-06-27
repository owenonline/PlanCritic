# Adherence Model Training

This directory contains tools for generating the additional files required to run the PlanCritic experiment for a new domain. This primarily involves training the LSTM model which will be used to predict if a plan adheres to the user's feedback during the progression of the genetic algorithm.

## Prerequisites

Set up Weights & Biases for experiment tracking:
```bash
wandb login
```

## Usage Workflow

### Step 1: Create Situation Information
Generate natural language descriptions for your domain's problem instances:

```bash
python3 create_situation_info.py --domain <domain_name>
```

This script:
- Reads PDDL domain and problem files from `/workspace/domains/<domain_name>/feedback/`
- Uses GPT-4 to generate situation descriptions
- Saves descriptions as `situation_info.txt` in each problem directory

### Step 2: Generate Training Data
Create synthetic training data with plan-constraint pairs:

```bash
python3 create_training_data.py --domain <domain_name> --problem <problem_name>
```

This script:
- Generates random PDDL constraints for the given problem
- Tests constraint solvability using the OPTIC planner
- Creates both positive (constraint-satisfying) and negative (constraint-violating) examples
- Converts constraints to natural language using GPT-4
- Saves training data as `v5data.json` files

### Step 3: Embed Training Data
Convert text data to numerical embeddings:

```bash
python3 embed_data_api.py --domain <domain_name>
```

This script:
- Loads training data from all problem instances in the domain
- Uses OpenAI's `text-embedding-3-small` model to embed:
  - Planning objectives (natural language constraints)
  - Plan steps (individual actions)
- Combines objective and plan embeddings
- Saves padded sequences as `reward_model_embedded_data/<domain_name>.npy`
- **This step may run out of memory if your docker container has less than 16GB of RAM.**

### Step 4: Train the Reward Model
Train the LSTM-based reward model:

```bash
python3 train_reward_model_dynamic.py --domain <domain_name>
```

**Model Architecture:**
- LSTM network for processing sequential plan data
- Input: Concatenated embeddings of constraints and plan steps
- Output: Scalar adherence score (0-1)
- Features: Dropout regularization, Xavier initialization, sigmoid activation
- Saves the model to the `/workspace/domains/<domain_name>/models/best_lstm_model.pth` file.
- **This step may run out of memory if your docker container has less than 16GB of RAM.**