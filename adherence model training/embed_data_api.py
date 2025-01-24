import os
import torch
import torch.nn as nn
from datasets import Dataset
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import os

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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

aggregated_dataset = create_dataset(DOMAIN_PROBLEMS)

embedded_objectives = []
for obj_idx in trange(0, len(aggregated_dataset['planning_objectives']), 1000):
    tmp_objectives = [np.array(response.embedding) for response in client.embeddings.create(input=aggregated_dataset['planning_objectives'][obj_idx:obj_idx+1000], model="text-embedding-3-small").data]
    embedded_objectives += tmp_objectives

def embed_plan(plan, index):
    return np.array([response.embedding for response in client.embeddings.create(input=plan, model="text-embedding-3-small").data]), index

with ThreadPoolExecutor(max_workers=30) as executor:
    embedded_plans_futures = []
    for idx, plan in enumerate(aggregated_dataset['plan_steps']):
        embedded_plans_futures.append(executor.submit(embed_plan, plan, idx))

    embedded_plans = [future.result() for future in as_completed(embedded_plans_futures)]
    embedded_plans = sorted(embedded_plans, key=lambda x: x[1])
    embedded_plans = [plan[0] for plan in embedded_plans]

combined_data = []
for objective, plan in zip(embedded_objectives, embedded_plans):
    combined = np.repeat(objective.reshape(1, -1), plan.shape[0], axis=0)
    combined = np.concatenate((plan, combined), axis=1)
    combined_data.append(combined)

max_steps = max([plan.shape[0] for plan in combined_data])
padded_data = pad_sequences(combined_data, max_steps)

os.makedirs("reward_model_embedded_data", exist_ok=True)
with open("reward_model_embedded_data/openai.npy", "wb") as f:
    np.save(f, padded_data)