from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from openai import OpenAI
import os
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
class FitnessEvaluator:
    def __init__(self):
        self.fitness_model = LSTMNetwork(3072, 512, 1, 2) # 1 for old model, 2 for new model
        # self.fitness_model = nn.DataParallel(self.fitness_model)
        self.fitness_model.to(device)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'best_lstm_model.pth')
        self.fitness_model.load_state_dict(torch.load(model_path, map_location=device))

    def __call__(self, plans, objectives):
        """
        Evaluate the plan and return the fitness score
        """
        
        def embed_plan(plan, index):
            return np.array([response.embedding for response in client.embeddings.create(input=plan, model="text-embedding-3-small").data]), index
        
        plans = [[action['action'] for action in plan] for plan in plans]
        
        embedded_objectives = [np.array(response.embedding) for response in client.embeddings.create(input=objectives, model="text-embedding-3-small").data]
        with ThreadPoolExecutor(max_workers=30) as executor:
            embedded_plans_futures = []
            for idx, plan in enumerate(plans):
                embedded_plans_futures.append(executor.submit(embed_plan, plan, idx))

            embedded_plans = [future.result() for future in as_completed(embedded_plans_futures)]
            embedded_plans = sorted(embedded_plans, key=lambda x: x[1])
            embedded_plans = [plan[0] for plan in embedded_plans]

        combined_list = []
        for plan in embedded_plans:
            for objective in embedded_objectives:
                combined = np.repeat(objective.reshape(1, -1), plan.shape[0], axis=0)
                combined = np.concatenate((plan, combined), axis=1)
                combined_list.append(combined)

        combined_data = [torch.tensor(thing, dtype=torch.float32).to(device) for thing in combined_list]

        with torch.no_grad():
            self.fitness_model.eval()
            fitness_scores = np.array([self.fitness_model(data).cpu().detach().numpy().flatten() for data in combined_data])

        fitness_scores = fitness_scores.reshape(len(plans), len(objectives))

        # Calculate the proportion of objectives fulfilled for each plan
        plan_fitness_scores = (fitness_scores > 0.5).mean(axis=1)
        raw_fitness_scores = fitness_scores.mean(axis=1)

        return plan_fitness_scores, raw_fitness_scores