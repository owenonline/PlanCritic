import os
import torch
import torch.nn as nn
from datasets import Dataset
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import wandb

DOMAIN_PROBLEMS = {
    # 'crew-planning-temporal-satisficing': ['instance-17',
    #                                                     'instance-9',
    #                                                     'instance-18',
    #                                                     'instance-12',
    #                                                     'instance-13',
    #                                                     'instance-1'],
    #                     'parking-temporal-satisficing': ['instance-13',
    #                                                     'instance-10',
    #                                                     'instance-9',
    #                                                     'instance-12',
    #                                                     'instance-14',
    #                                                     'instance-4',
    #                                                     'instance-8',
    #                                                     'instance-3'],
                        "restore_waterway_no_fuel": ['instance-1',
                                                    'instance-2',
                                                    'instance-3',
                                                    'instance-4',]}

MAX_LEN = 2048
OUTPUT_DIR = "reward_model_lstm"

class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = self.fc1(hn[-1])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def create_dataset(domain_problems, train_set=True):
    dict_to_populate = {
        "planning_objectives": [],
        "plan_steps": [],
        "adherence": []
    }

    for domain in domain_problems:
        for problem in domain_problems[domain]:
            problem_directory = os.path.join("temporal", domain, "feedback", problem)

            for data_file in ["v3data.json", "v4data.json", "v5data.json"]:
                data_path = os.path.join(problem_directory, data_file)
            
                if not os.path.exists(data_path):
                    continue
                else:
                    with open(data_path, "r") as f:
                        data = json.load(f)
                    
                    for feedback_instance in data:
                        actions = feedback_instance["plan"]
                        steps = [action['action'] for action in actions]

                        for feedback in feedback_instance["feedback"]:
                            dict_to_populate["planning_objectives"].append(feedback['feedback'])
                            dict_to_populate["plan_steps"].append(steps)

                            if "obeyed" in feedback:
                                adherence = feedback['obeyed']
                            elif "satisfied" in feedback:
                                adherence = feedback['satisfied']

                            dict_to_populate["adherence"].append(adherence)

    return dict_to_populate

def encode_plans(rank, world_size, aggregated_dataset, return_list):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.manual_seed(0)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # Split the dataset
    num_plans = len(aggregated_dataset["plan_steps"])
    chunk_size = (num_plans + world_size - 1) // world_size
    start = rank * chunk_size
    end = min(start + chunk_size, num_plans)
    local_plans = aggregated_dataset["plan_steps"][start:end]
    local_objectives = aggregated_dataset["planning_objectives"][start:end]

    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
    model = AutoModel.from_pretrained('intfloat/e5-base-v2').to(rank)
    # tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    # model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large").to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    with torch.no_grad():
        # handle the local objectives
        tokenized_local_objectives = tokenizer(local_objectives, max_length=512, padding=True, truncation=True, return_tensors='pt').to(rank)
        encoded_local_objectives = ddp_model.module(**tokenized_local_objectives)
        encoded_local_objectives = average_pool(encoded_local_objectives.last_hidden_state, tokenized_local_objectives['attention_mask']).cpu().detach().numpy()
        # tokenized_local_objectives = tokenizer(local_objectives, padding=True, truncation=True, return_tensors='pt').to(rank)
        # encoded_local_objectives = ddp_model.module(**tokenized_local_objectives, output_hidden_states=True, return_dict=True).pooler_output
        # encoded_local_objectives = encoded_local_objectives.cpu().detach().numpy()
        

        # handle the local plans
        encoded_local_plans = []
        for steps in local_plans:
            tokenized_steps = tokenizer(steps, max_length=512, padding=True, truncation=True, return_tensors='pt').to(rank)
            encoded_steps = ddp_model.module(**tokenized_steps)
            encoded_local_plans.append(average_pool(encoded_steps.last_hidden_state, tokenized_steps['attention_mask']).cpu().detach().numpy())
            # tokenized_steps = tokenizer(steps, padding=True, truncation=True, return_tensors='pt').to(rank)
            # encoded_steps = ddp_model.module(**tokenized_steps, output_hidden_states=True, return_dict=True).pooler_output
            # encoded_local_plans.append(encoded_steps.cpu().detach().numpy())

    # Gather results from all processes
    encoded_plans = [None for _ in range(world_size)]
    encoded_objectives = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(encoded_plans, encoded_local_plans)
    torch.distributed.all_gather_object(encoded_objectives, encoded_local_objectives)

    # Flatten the list of lists and store in the return list
    if rank == 0:
        flattened_encoded_plans = [item for sublist in encoded_plans for item in sublist]
        flattened_encoded_objectives = [item for sublist in encoded_objectives for item in sublist]

        combined_data = []
        for objective, plan in zip(flattened_encoded_objectives, flattened_encoded_plans):
            combined = np.repeat(objective.reshape(1, -1), plan.shape[0], axis=0)
            combined = np.concatenate((plan, combined), axis=1)
            combined_data.append(combined)

        return_list.extend(combined_data)

    torch.distributed.destroy_process_group()

def pad_sequences(combined_data, max_len):
    padded_data = []
    for plan in combined_data:
        if len(plan) < max_len:
            padding = np.zeros((max_len - len(plan), plan.shape[1]))
            padded_plan = np.vstack((plan, padding))
        else:
            padded_plan = plan
        padded_data.append(padded_plan)
    return np.array(padded_data)

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())

    wandb.login(key="57e77180c09aaee0b682991aef27ddc5ea72d9cb")
    os.environ['WANDB_LOG_MODEL'] = "True"

    # Initialize WandB
    wandb.init(project="huggingface", entity="owenonline")

    aggregated_dataset = create_dataset(DOMAIN_PROBLEMS)
    print(len(aggregated_dataset["planning_objectives"]), len(aggregated_dataset["plan_steps"]), len(aggregated_dataset["adherence"]))

    combined_data = np.load("/home/owen/workareas/online-preference-learning/reward_model_embedded_data/openai.npy")

    # Prepare data for training
    X = torch.tensor(combined_data, dtype=torch.float32)
    y = torch.tensor(aggregated_dataset['adherence'], dtype=torch.float32).view(-1, 1)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False)

    # Initialize the model, loss function, and optimizer
    input_dim = combined_data.shape[2]
    print("Input dimension:", input_dim)
    hidden_dim = 512
    output_dim = 1
    num_layers = 2
    model = LSTMNetwork(input_dim, hidden_dim, output_dim, num_layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Reduced learning rate

    # Watch the model with WandB
    wandb.watch(model, log="all")

    best_val_loss = float('inf')
    best_accuracy = 0

    for i in range(20):
        # Training loop with gradient clipping
        epochs = 5
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(train_loader)
            wandb.log({"Epoch": epoch+1, "Training Loss": avg_epoch_loss})
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")

        # Evaluation
        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        true_predictions = 0
        false_predictions = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                # Calculate accuracy
                predictions = (outputs > 0.5).float()
                correct_predictions += (predictions == batch_y).sum().item()
                total_predictions += batch_y.size(0)
                
                # Calculate true/false predictions
                true_predictions += predictions.sum().item()
                false_predictions += (predictions == 0).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        true_false_ratio = true_predictions / (false_predictions + 1e-6)  # Add a small value to avoid division by zero

        if accuracy > best_accuracy and avg_val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_lstm_model.pth"))
            wandb.save(os.path.join(OUTPUT_DIR, "best_lstm_model.pth"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            
        wandb.log({"Validation Loss": avg_val_loss, "Validation Accuracy": accuracy, "True to False Ratio": true_false_ratio})
        print(f"Validation Loss: {val_loss/len(val_loader)}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Ratio of Predicted Trues to Predicted Falses: {true_false_ratio:.2f}")
        print(f"true_predictions: {true_predictions}, false_predictions: {false_predictions}")

    print("Best validation loss:", best_val_loss)
    print("Best validation accuracy:", best_accuracy)

    # Save the model
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "lstm_model.pth"))

    # Log the model to W&B
    wandb.save(os.path.join(OUTPUT_DIR, "lstm_model.pth"))

    print("Training complete and model saved.")

if __name__ == "__main__":
    main()