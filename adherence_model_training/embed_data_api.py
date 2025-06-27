import os
import numpy as np
import json
from tqdm import trange, tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import argparse
import os
from collections import defaultdict

# PREFIX = "/Users/owenburns/workareas/Carnegie Mellon PlanCritic/PlanCritic/"

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=str, default=None)
parser.add_argument("--prefix", type=str, default="/workspace/")
args = parser.parse_args()
domain = args.domain

def create_domain_problems(domain: str):
    problem_directory = f"{args.prefix}domains/{domain}/feedback"
    domain_problems = defaultdict(list)
    for problem in os.listdir(problem_directory):
        domain_problems[domain].append(problem)
    return domain_problems

DOMAIN_PROBLEMS = create_domain_problems(domain)
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
            problem_directory = os.path.join(f"{args.prefix}domains", domain, "feedback", problem)

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

def pad_sequences(plans):
    """
    Takes a list of 2‑D numpy arrays (steps × embed_dim) and returns a single
    3‑D tensor (num_plans × max_steps × embed_dim) with zero‑padding.

    • Pre‑allocates the full tensor once, so only one copy is ever in RAM.
    • Uses float32 to cut the footprint in half without touching how
      the embeddings were *generated* earlier in the script.
    """
    if not plans:
        return np.empty((0, 0, 0), dtype=np.float32)

    num_plans  = len(plans)
    max_steps  = max(p.shape[0] for p in plans)
    embed_dim  = plans[0].shape[1]

    padded = np.zeros((num_plans, max_steps, embed_dim), dtype=np.float32)

    for i, plan in enumerate(plans):
        steps = plan.shape[0]
        padded[i, :steps, :] = plan.astype(np.float32, copy=False)

    return padded

aggregated_dataset = create_dataset(DOMAIN_PROBLEMS)

embedded_objectives = []
for obj_idx in trange(0, len(aggregated_dataset['planning_objectives']), 1000):
    tmp_objectives = [np.array(response.embedding) for response in client.embeddings.create(input=aggregated_dataset['planning_objectives'][obj_idx:obj_idx+1000], model="text-embedding-3-small").data]
    embedded_objectives += tmp_objectives

print("Objective embedding complete")

def embed_plan(plan, index):
    return np.array([response.embedding for response in client.embeddings.create(input=plan, model="text-embedding-3-small").data]), index

with ThreadPoolExecutor(max_workers=30) as executor:
    embedded_plans_futures = []
    for idx, plan in enumerate(aggregated_dataset['plan_steps']):
        embedded_plans_futures.append(executor.submit(embed_plan, plan, idx))

    embedded_plans = [future.result() for future in tqdm(as_completed(embedded_plans_futures), total=len(aggregated_dataset['plan_steps']))]
    embedded_plans = sorted(embedded_plans, key=lambda x: x[1])
    embedded_plans = [plan[0] for plan in embedded_plans]

print("Plan embedding complete")

combined_data = []
for objective, plan in zip(embedded_objectives, embedded_plans):
    combined = np.repeat(objective.reshape(1, -1), plan.shape[0], axis=0)
    combined = np.concatenate((plan, combined), axis=1)
    combined_data.append(combined)

print("Concatenated data")

max_steps = max([plan.shape[0] for plan in combined_data])
print(max_steps)
padded_data = pad_sequences(combined_data)

print("Padded data")

np.savez(f"{args.prefix}adherence_model_training/reward_model_embedded_data/{domain}.npz", X=padded_data, y=np.asarray(aggregated_dataset['adherence']))
# with open(f"{PREFIX}adherence_model_training/reward_model_embedded_data/{domain}.npy", "wb") as f:
#     np.save(f, padded_data)

print("Saved data")