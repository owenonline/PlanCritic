"""
This script goes through each of the problem instances and creates the situation info for each one.
"""

import os
from openai import OpenAI, RateLimitError
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import backoff
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=str, default=None)
parser.add_argument("--prefix", type=str, default="/workspace/")
args = parser.parse_args()
domain = args.domain

assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable"

API_KEY = os.environ["OPENAI_API_KEY"]

@backoff.on_exception(backoff.expo,
                      RateLimitError)
def make_situation_info(domain_text, problem_text, instance_directory):
    messages = [
        {"role": "system", "content": "You are an assistant helping a user describe their pddl environment. Write a paragraph of information describing the goal of the problem in natural language and explaining the dynamics of the environment. However, include the problem specific names of objects in the problem.pddl."},
        {"role": "user", "content": f"domain pddl:\n{domain_text}\n\nproblem pddl:\n{problem_text}"}
    ]

    if "situation_info.txt" in os.listdir(instance_directory):
        return

    with OpenAI(api_key=API_KEY) as client:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7
        )
        situation_info = completion.choices[0].message.content

    with open(os.path.join(instance_directory, "situation_info.txt"), "w") as situation_info_file:
        situation_info_file.write(situation_info)

with ThreadPoolExecutor(max_workers=20) as executor:

    futures = []

    domain_path = f"{args.prefix}domains/{domain}/domain.pddl"
    problem_directory = f"{args.prefix}domains/{domain}/feedback"

    with open(domain_path, "r") as domain_file:
        domain_text = domain_file.read()

    for problem in os.listdir(problem_directory):

        instance_directory = os.path.join(problem_directory, problem)
        problem_path = os.path.join(problem_directory, problem, f"{problem}.pddl")

        for problem_file in os.listdir(instance_directory):
            if problem_file.endswith(".pddl"):
                problem_path = os.path.join(instance_directory, problem_file)

            with open(problem_path, "r") as problem_file:
                problem_text = problem_file.read()

            future = executor.submit(make_situation_info, domain_text, problem_text, instance_directory)
            futures.append(future)

    for future in tqdm(as_completed(futures)):
        future.result()