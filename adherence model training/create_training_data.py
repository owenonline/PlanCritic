import os
import re
import resource
import tempfile
import time
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel
from parsers import OpticParser
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm, trange
import random
from itertools import permutations, product, zip_longest, combinations
import chromadb
from chromadb.utils import embedding_functions
import json
import argparse
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
import json_repair
import instructor
import signal

parser = argparse.ArgumentParser()
parser.add_argument("--domain", type=str, default=None)
parser.add_argument("--problem", type=str, default=None)

args = parser.parse_args()
domain = args.domain
problem = args.problem

domain_problems = {'crew-planning-temporal-satisficing': ['instance-17',
                                                    'instance-9',
                                                    'instance-18',
                                                    'instance-12',
                                                    'instance-13',
                                                    'instance-1'],
                    'parking-temporal-satisficing': ['instance-13',
                                              'instance-10',
                                              'instance-9',
                                              'instance-12',
                                              'instance-14',
                                              'instance-4',
                                              'instance-8',
                                              'instance-3'],
                    'match-cellar-temporal-satisficing': ['instance-15',
                                                   'instance-17',
                                                   'instance-19',
                                                   'instance-4',
                                                   'instance-3',
                                                   'instance-16'],
                    "restore_waterway_no_fuel": ['instance-1',
                                                    'instance-2',
                                                    'instance-3',
                                                    'instance-4',]}

EXAMPLES = [
    {
        "situation_info": "The goal of the problem in this PDDL environment is to move all crates to their respective depots efficiently. Specifically, crates `crate0`, `crate1`, `crate2`, and `crate3` need to be moved to `depot0`, while crates `crate4`, `crate5`, `crate6`, and `crate7` need to be moved to `depot1`. This involves using hoists `hoist0` and `hoist1` to lift and drop the crates at the designated storeareas within the depots. The environment consists of various storeareas (`depot0-1-1`, `depot0-1-2`, `depot0-2-1`, `depot0-2-2`, `depot1-1-1`, `depot1-1-2`, `depot1-2-1`, `depot1-2-2`) and a transitarea (`loadarea`). The hoists can move between connected storeareas, and from storeareas to the transitarea and back. The goal is to achieve this movement while minimizing total time, ensuring that the hoists are used efficiently to transport the crates to their final positions in the depots.",
        "constraints":[
            "sometime-before (in crate1 depot0) (in crate0 depot0)",
            "at-end (not (lifting hoist0 crate1))",
            "at-end (not (lifting hoist1 crate4))"
        ],
        "feedback": [
            "Ensure crate0 is placed in depot0 before crate1.",
            "Ensure crate1 is not being lifted by hoist0 at the end.",
            "Ensure crate4 is not being lifted by hoist1 at the end."
        ]
        # "Ensure crate0 is placed in depot0 before crate1, and that crates crate1 and crate4 are not being lifted by hoists at the end."
    },
    {
        "situation_info": "The goal of the CrewPlanning problem revolves around ensuring that a team of crew members, specifically c1 and c2, follow a structured daily routine over a span of four days while completing various assigned tasks. The main objectives include ensuring that both crew members complete their sleep cycles (done_sleep) each day, finish medical conferences (mcs_finished) with medical states mcs1 and mcs2, change the spaceship filter (spaceshipFilter) daily, and complete a series of payload activities (pa1_1 to pa3_15). The day starts with post_sleep activities, followed by meals, exercises, and specific tasks such as changing filters or conducting medical conferences. The tasks are interdependent and need to be completed in a specific order while ensuring the constraints on the crew members' availability and the equipment usage. The environment operates on a temporal scale where each day is modeled to last 1440 minutes, and actions have defined durations that must fit within this timeframe. The ultimate goal is to efficiently manage the crew's time to meet all specified objectives within the given period, minimizing the total time spent on all activities.",
        "constraints": [
            "(sometime-before (done_exercise c1 d1) (done_meal c1 d1))",
            "(sometime-before (done_exercise c2 d1) (done_meal c2 d1))"
        ],
        "feedback": [
            "Ensure that c1 eats their meal before exercising on day d1.",
            "Ensure that c2 eats their meal before exercising on day d1."
        ]
        # "Ensure that c1 and c2 eat their meals before exercising on day d1."
    },
    {
        "situation_info": "The goal of the problem is to clear the waterway of debris and salvage a shipwreck, ensuring safe passage from the initial waypoint (wpt_ini) to the endpoint (wpt_end). Specifically, this involves removing all normal_debris and underwater_debris blocking the path from wpt_a_2 and wpt_b_2 to wpt_end, and towing the shipwreck (shp_0) to the ship dock (shp_dck_0). The environment includes various assets such as debris_asset (deb_ast_0), scout_asset (sct_ast_0), and ship_salvage_asset (shp_sal_ast_0). Debris assets can remove visible debris, scout assets are used to detect and make underwater debris visible, and ship salvage assets can tow and dock shipwrecks. The environment also includes locations like debris stations (deb_stn_0), refuel stations (rfl_stn_0), and ship docks (shp_dck_0). The dynamics of the environment require managing fuel levels, ensuring assets are not damaged, and coordinating the traversal, debris removal, scouting, and ship salvaging actions while adhering to the constraints of the assets and the waterway authorities (aut_a, aut_b) that control the restriction status of certain locations. The ultimate objective is to ensure all specified waypoints are unblocked and the shipwreck is docked, with all assets returning to the initial waypoint.",
        "constraints": [
            "(sometime (at u_deb_ini_b_0 wpt_ini))",
            "(always (at u_deb_b_0_b_1 wpt_b_0))",
            "(always (at u_deb_b_1_b_2 wpt_b_1))",
            "(always (at u_deb_b_2_end wpt_b_2))"
        ],
        "feedback": [
            "Ensure that underwater debris is at wpt_ini at some point.",
            "Ensure that underwater debris is always at wpt_b_0.",
            "Ensure that underwater debris is always at wpt_b_1.",
            "Ensure that underwater debris is always at wpt_b_2."
        ]
        # "Ensure that underwater debris is never removed from wpt_ini, wpt_b_0, wpt_b_1, and wpt_b_2."
    },
    # {
    #     "situation_info": "The goal of the problem is to clear the waterway of debris and salvage a shipwreck, ensuring safe passage from the initial waypoint (wpt_ini) to the endpoint (wpt_end). Specifically, this involves removing all normal_debris and underwater_debris blocking the path from wpt_a_2 and wpt_b_2 to wpt_end, and towing the shipwreck (shp_0) to the ship dock (shp_dck_0). The environment includes various assets such as debris_asset (deb_ast_0), scout_asset (sct_ast_0), and ship_salvage_asset (shp_sal_ast_0). Debris assets can remove visible debris, scout assets are used to detect and make underwater debris visible, and ship salvage assets can tow and dock shipwrecks. The environment also includes locations like debris stations (deb_stn_0), refuel stations (rfl_stn_0), and ship docks (shp_dck_0). The dynamics of the environment require managing fuel levels, ensuring assets are not damaged, and coordinating the traversal, debris removal, scouting, and ship salvaging actions while adhering to the constraints of the assets and the waterway authorities (aut_a, aut_b) that control the restriction status of certain locations. The ultimate objective is to ensure all specified waypoints are unblocked and the shipwreck is docked, with all assets returning to the initial waypoint.",
    #     "constraints": [
    #         "(always (at u_deb_ini_b_0 wpt_ini))",
    #         "(always (at u_deb_b_0_b_1 wpt_b_0))",
    #         "(always (at u_deb_b_1_b_2 wpt_b_1))",
    #         "(always (at u_deb_b_2_end wpt_b_2))"
    #     ],
    #     "feedback": "Ensure that underwater debris is never removed from wpt_ini, wpt_b_0, wpt_b_1, and wpt_b_2."
    # }
]

def test_constraint_solvability(domain_path: str, problem_path: str, constraints: list[str], timeout: Optional[str]="10s") -> bool:
    with open(problem_path, "r") as f:
        problem_text = f.read()

    if len(constraints) > 1:
        constraints_block = "\n\t".join(constraints)
        constraints_tag = f"(:constraints (and\n\t{constraints_block}\n))"
    elif len(constraints) == 1:
        constraints_tag = f"(:constraints {constraints[0]})"
    else:
        constraints_tag = ""

    new_problem = problem_text.replace("(:goal", f"{constraints_tag}\n(:goal")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete_on_close=False) as tf:
        tf.write(new_problem)
        candidate_problem_path = tf.name
        tf.close()

        command = ["timeout", timeout, "./optic-cplex", "-N", domain_path, candidate_problem_path]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, 
                universal_newlines=True, text=True, encoding="utf-8")
        
        stdout, stderr = process.communicate()
        parser = OpticParser()
        plan = parser.injest_stdout(stdout)

        if plan is None:
            return False, None
        else:
            return True, plan['actions']
        
def timeout_handler(signum, frame):
    raise Exception
        
def test_plan_adherence(domain_path: str, problem_path: str, constraint: str, actions: list[dict]) -> bool:
    with open(problem_path, "r") as f:
        problem_text = f.read()

    constraints_tag = f"(:constraints {constraint})"

    new_problem = problem_text.replace("(:goal", f"{constraints_tag}\n(:goal")

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # Set the alarm for 10 seconds

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete_on_close=False) as temp_problem_file:
            temp_problem_file.write(new_problem)
            candidate_problem_path = temp_problem_file.name
            temp_problem_file.close()

            candidate_plan_pddl_steps = [f"{action['time_step']}: {action['action']} [{action['duration']}]" for action in actions]
            candidate_plan_block = "\n".join(candidate_plan_pddl_steps)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete_on_close=False) as temp_plan_file:
                temp_plan_file.write(candidate_plan_block)
                candidate_plan_path = temp_plan_file.name
                temp_plan_file.close()

                command = ["./Validate", "-t", "0.001", "-v", domain_path, candidate_problem_path, candidate_plan_path]
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, 
                        universal_newlines=True, text=True, encoding="utf-8")
                
                stdout, _ = process.communicate(timeout=5)

                signal.alarm(0)
                
                if "Successful plans:" in stdout:
                    return 1
                else:#if "Failed plans:" in stdout:
                    return 0
    except Exception as e:
        print("timed out...")
        return 0

def create_nl_feedback(constraints: list[str], situation_info: str) -> str:
    instructor_model = instructor.from_openai(OpenAI(api_key=os.environ['OPENAI_API_KEY']))

    class Feedback(BaseModel):
        feedback_list: list[str]

    user_messages = [#
        {
            "role": "user",
            "content": f"Situation info: {example['situation_info']}\nConstraints: {example['constraints']}"
        }
        for example in EXAMPLES
    ]
    ai_messages = [
        {
            "role": "assistant",
            "content": f"{example['feedback']}"
        }
        for example in EXAMPLES
    ]
    messages = [
        {
            "role": "system",
            "content": "You are an assistant helping a user describe their PDDL environment. Given a summary of the environment dynamics and a set of constraints, state what each of the constraints will do to the plan as if you are commanding someone to make that change. Be sure that everything in your statement directly corresponds to the constraints, and don't omit anything. If there are object names, use those names in your response. A predicate of the form (sometime-before (p1) (p2)) means that p1 happens AFTER p2. In other words, it means \"sometime before p1 happens, p2 happens\". A predicate of the form (always-within number (condition) (predicate)) means that after condition is true, no more than number time steps pass before predicate becomes true. A predicate of the form (hold-during number1 number2 (predicate)) means that predicate is true between time steps number1 and number2. A predicate of the form (hold-after number (predicate)) means that predicate is true after time step number."
        }
    ] + [message for pair in zip_longest(user_messages, ai_messages) for message in pair] + [
        {
            "role": "user",
            "content": f"Situation info: {situation_info}\nConstraints: {constraints}"
        }
    ]

    response_list = instructor_model.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0,
                        response_model=Feedback,
                        messages=messages,
                    ).feedback_list

    return response_list

# for domain in domain_problems:
#     domain_path = os.path.join("temporal", domain, "domain.pddl")

#     for problem in domain_problems[domain]:
domain_path = os.path.join("temporal", domain, "domain.pddl")
problem_directory = os.path.join("temporal", domain, "feedback", problem)
problem_path = os.path.join(problem_directory, f"{problem}.pddl")
situation_info_path = os.path.join(problem_directory, "situation_info.txt")

with open(situation_info_path, "r") as f:
    situation_info = f.read()

if os.path.exists(os.path.join(problem_directory, "v5data.json")):
    with open(os.path.join(problem_directory, "v5data.json"), "r") as f:
        complete_feedback_instances = json.load(f)
    print(f"loaded {len(complete_feedback_instances)} feedback instances.")
else:
    complete_feedback_instances = []

# create the constraints
class pddltype:
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
        self.objects = []

    def add_object(self, instance):
        self.objects.append(instance)
        if not self.parent is None:
            self.parent.add_object(instance)

    def __repr__(self):
        return str(self.objects)

with open(domain_path, "r") as f:
    domain_text = f.read()

with open(problem_path, "r") as f:
    problem_text = f.read()

raw_predicates = domain_text.split("(:predicates")[1].split("(:")[0].split("\n")
predicates = {}

for raw_predicate in raw_predicates:
    raw_predicate = raw_predicate.strip()

    if ";" in raw_predicate:
        raw_predicate = raw_predicate.split(";")[0]

    # extract name
    pred_name_match = re.search(r'\(([^ ]*)', raw_predicate)
    if pred_name_match:
        pred_name = pred_name_match.group(1)
        pred_name = pred_name.replace(")", "")
    else:
        continue
    raw_predicate = raw_predicate.replace(f"({pred_name} ", "").strip()

    if raw_predicate[-1] == ")":
        raw_predicate = raw_predicate[:-1]

    # extract arguments
    arguments = []
    while len(raw_predicate) > 0:
        arg_match = re.search(r'\?([^ ]*) - ([^\?]*)', raw_predicate)
        if arg_match:
            arg_name = arg_match.group(1)
            arg_type = arg_match.group(2).strip()
            arguments.append(arg_type)
            raw_predicate = raw_predicate.replace(f"?{arg_name} - {arg_type}", "")
        else:
            break

    predicates[pred_name] = arguments

objects = {"object": pddltype(None, "object")}
raw_types = domain_text.split("(:types")[1].split("(:")[0].split("\n")

for raw_type_line in raw_types:
    raw_type_line = raw_type_line.strip()

    if ";" in raw_type_line:
        raw_type_line = raw_type_line.split(";")[0]

    raw_type_line = raw_type_line.replace(")", "").replace("(", "")

    if " - " in raw_type_line:
        type_names = raw_type_line.split(" - ")[0].strip().split()
        type_parent = raw_type_line.split(" - ")[1].strip()
        
        if type_parent in objects:
            type_parent_object = objects[type_parent]
        else:
            type_parent_object = pddltype(None, type_parent)
            objects[type_parent] = type_parent_object

        for new_type in type_names:
            objects[new_type] = pddltype(type_parent_object, new_type)
    else:
        type_names = raw_type_line.split()
        for new_type in type_names:
            objects[new_type] = pddltype(None, new_type)

raw_objects = problem_text.split("(:objects")[1].split("(:")[0].split("\n")
buffer = []

for raw_object_line in raw_objects:
    raw_object_line = raw_object_line.strip()

    if ";" in raw_object_line:
        raw_object_line = raw_object_line.split(";")[0]

    raw_object_line = raw_object_line.replace(")", "").replace("(", "")

    if " - " in raw_object_line:
        object_names = raw_object_line.split(" - ")[0].split()
        object_type = raw_object_line.split(" - ")[1]

        instance_names = object_names + buffer
        for instance_name in instance_names:
            objects[object_type].add_object(instance_name)
        buffer = []
    else:
        object_names = raw_object_line.split()
        if len(object_names) > 0:
            buffer.extend(object_names)

predicate_instances = []
for pred_name, arguments in predicates.items():
    objects_list = []
    for argument in arguments:
        if "(either" in argument:
            arg_options = argument.split("(either ")[1].split(")")[0].split()
            combined_objects = objects[arg_options[0]].objects + objects[arg_options[1]].objects
            objects_list.append(combined_objects)
        else:
            objects_list.append(objects[argument].objects)

    argument_combinations = list(product(*objects_list))

    for combination in argument_combinations:
        if len(list(set(combination))) == len(combination):
            predicate_instances.append(f"({pred_name} {' '.join(combination)})")

raw_init_predicates = problem_text.split("(:init")[1].split("(:")[0].split("\n")
init_predicates_cleaned = []
for raw_init_pred in raw_init_predicates:
    raw_init_pred = raw_init_pred.strip()

    if ";" in raw_init_pred:
        raw_init_pred = raw_init_pred.split(";")[0]

    if raw_init_pred != "" and raw_init_pred[0] != ")":
        init_predicates_cleaned.append(raw_init_pred)

action_durations = re.findall(r":duration \(= \?duration ([^)]*)\)", domain_text)
try:
    action_duration_nums = [float(duration) for duration in action_durations if duration != "?duration"]
except:
    action_duration_nums = list(range(10))

pred_combinations = list(combinations(predicate_instances, 2))
pred_permutations = list(permutations(predicate_instances, 2))

constraints = {
    "always": [f"(always {pred})" for pred in predicate_instances if pred in init_predicates_cleaned],
    "always-not": [f"(always (not {pred}))" for pred in predicate_instances if not pred in init_predicates_cleaned],
    "always-or": [f"(always (or {pred1} {pred2}))" for pred1, pred2 in pred_combinations if pred1 in init_predicates_cleaned or pred2 in init_predicates_cleaned],
    "always-and": [f"(always (and {pred1} {pred2}))" for pred1, pred2 in pred_combinations if pred1 in init_predicates_cleaned and pred2 in init_predicates_cleaned],
    "sometime": [f"(sometime {pred})" for pred in predicate_instances],
    "sometime-or": [f"(sometime (or {pred1} {pred2}))" for pred1, pred2 in pred_combinations],
    "at-most-once": [f"(at-most-once {pred})" for pred in predicate_instances],
    "at-most-once-or": [f"(at-most-once (or {pred1} {pred2}))" for pred1, pred2 in pred_combinations],
    "sometime-before": [f"(sometime-before {pred1} {pred2})" for pred1, pred2 in pred_permutations],
    "sometime-after": [f"(sometime-after {pred1} {pred2})" for pred1, pred2 in pred_permutations],
    "at-end": [f"(at end {pred})" for pred in predicate_instances],
    "at-end-not": [f"(at end (not {pred}))" for pred in predicate_instances],
    "at-end-or": [f"(at end (or {pred1} {pred2}))" for pred1, pred2 in pred_combinations],
    "always-within": [f"(always-within {random.choice(action_duration_nums)} {pred})" for pred in predicate_instances],
    "hold-during": [f"(hold-during {random.choice(action_duration_nums)} {random.choice(action_duration_nums)} {pred})" for pred in predicate_instances],
    "hold-after": [f"(hold-after {random.choice(action_duration_nums)} {pred})" for pred in predicate_instances],
}

def get_random_constraint():
    constraint_type = random.choice(list(constraints.keys()))
    return random.choice(constraints[constraint_type])

def mutate_constraint(llm_constraint: str) -> str:

    # choose the kind of mutation to perform
    mutation_type = random.choice(['negate_predicate', 'change_modifier', 'change_arg'])

    try:

        # extract the modifier, and change it if the mutation type is change_modifier
        modifier = llm_constraint.split("(")[1]

        if mutation_type == 'change_modifier':
            valid_modifier_regex = [r"always ", r"sometime ", r"within \d+ ", r"at-most-once ", r"sometime-after ", r"sometime-before ", r"always-within \d+ ", r"hold-during \d+ \d+ ", r"hold-after \d+ ", r"at end "]
            valid_modifier = random.choice(valid_modifier_regex)
            valid_modifier = valid_modifier.replace("\\d+", str(random.choice(action_duration_nums)))
            llm_constraint = llm_constraint.replace(modifier, valid_modifier)

        def mutate_pred_content(pred_content, negate=False):

            # check if the pred is starting out negative, so we know how to change it if we're negating it
            negative = False
            if pred_content.startswith("not "):
                negative = True
                pred_content = pred_content.split("(")[1].split(")")[0]

            # if performing the change_arg mutation, flip a coin to decide whether to change each argument
            if mutation_type == 'change_arg':

                # separate the predicate name and its current arguments
                predicate = pred_content.split()[0]
                args = pred_content.split()[1:]

                # construct a list of available values for each argument of the predicate
                predargs = predicates[predicate]
                objects_list = []
                for argument in predargs:
                    if "(either" in argument:
                        arg_options = argument.split("(either ")[1].split(")")[0].split()
                        combined_objects = objects[arg_options[0]].objects + objects[arg_options[1]].objects
                        objects_list.append(combined_objects)
                    else:
                        objects_list.append(objects[argument].objects)
                
                new_args = []
                for arg, ol in zip(args, objects_list):
                    if mutation_type == 'change_arg' and random.random() < 0.5:
                        repaired_arg = random.choice(ol)
                        new_args.append(repaired_arg)
                    else:
                        new_args.append(arg)

                pred_content = f"{predicate} {' '.join(new_args)}"

            if negate:
                negative = not negative

            if negative:
                pred_content = f"not ({pred_content})"

            return pred_content

        arg1 = llm_constraint.split("(")[2].split()[0]
        if arg1 in ["and", "or"]:

            # extract both predicates for individual processing

            preds = []
            buffer = ""
            paren_depth = 0
            for char in llm_constraint.split(arg1)[1].strip():
                if char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth -= 1

                if paren_depth == 0 and char == ")":
                    preds.append(buffer + ")")
                    buffer = ""
                elif paren_depth == 0:
                    continue
                else:
                    buffer += char

            # if the mutation type is negate_predicate, randomly negate each predicate
            if mutation_type == 'negate_predicate':
                negate = [random.random() < 0.5 for _ in range(len(preds))]
            else:
                negate = [False for _ in range(len(preds))]

            mutated_preds = [
                f"({mutate_pred_content(pred.strip()[1:-1], negate=should_negate)})"
                for pred, should_negate in zip(preds, negate)
            ]
            for pred, mutated_pred in zip(preds, mutated_preds):
                llm_constraint = llm_constraint.replace(pred, mutated_pred)
        else:
            # extract the one predicate and process it

            pred = "(" + '('.join(llm_constraint.split("(")[2:])
            paren_depth = 0
            buffer = ""
            for char in pred:
                if char == "(":
                    paren_depth += 1
                elif char == ")":
                    paren_depth -= 1

                if paren_depth == 0:
                    buffer += char
                    break
                else:
                    buffer += char

            mutated_pred = mutate_pred_content(buffer[1:-1])
            llm_constraint = llm_constraint.replace(buffer[1:-1], mutated_pred)

        return llm_constraint
    except Exception as e:
        print(e)
        return None

success, base_plan_steps = test_constraint_solvability(domain_path, problem_path, [], timeout="30s")

if not success:
    print(f"[{domain} - {problem} - 0.00%] Base plan is unsolvable. Exiting.")
    exit()

desired_len = 500

while len(complete_feedback_instances) < desired_len:
    new_instance = {}

    # find a random set of ground truth constraints to create the feedback from
    print(f"[{domain} - {problem} - {(len(complete_feedback_instances)/desired_len)*100:.2f}%] Finding solvable constraint ground truth...")
    while True:
        num_constraints = random.randint(2, 5)
        ground_truth_constraints = [get_random_constraint() for _ in range(num_constraints)]

        solvable, current_plan_steps = test_constraint_solvability(domain_path, problem_path, ground_truth_constraints, timeout="10s")

        if solvable:
            # determine how many of the selected constraints are also true in the base plan
            try:
                num_satisfied_verified = sum([test_plan_adherence(domain_path, problem_path, ground_truth_constraint, base_plan_steps) for ground_truth_constraint in ground_truth_constraints])
            except:
                continue
            if num_satisfied_verified < len(ground_truth_constraints): # as I came to find out, this is one of the most important lines in the whole program since so many of the constraints are effectively meaningless because they're vacuously true whether specified or not.
                print(f"[{domain} - {problem} - {(len(complete_feedback_instances)/desired_len)*100:.2f}%] Found solvable and interesting ground truth constraints.")
                
                # convert the set of "true" constraints for the plan into natural language feedback
                gt_feedback = create_nl_feedback(ground_truth_constraints, situation_info)
                new_instance['feedback'] = [
                    {
                        "feedback": feedback,
                        "constraint": constraint,
                        "obeyed": 1
                    }
                    for feedback, constraint in zip(gt_feedback, ground_truth_constraints)
                ]
                
                # create an equal number of constraints that are false with respect to the plan but within a small edit distance of
                # one of the true constraints. This teaches the model to identify the small semantic differences that can make a constraint
                # untrue.
                false_constraints = []
                for true_constraint in tqdm(ground_truth_constraints):

                    while True:

                        false_constraint = mutate_constraint(true_constraint)
                        
                        if false_constraint is None:
                            continue
                            
                        if test_plan_adherence(domain_path, problem_path, false_constraint, current_plan_steps) == 0:
                            false_constraints.append(false_constraint)
                            break

                false_feedback = create_nl_feedback(false_constraints, situation_info)
                
                new_instance['feedback'] += [
                    {
                        "feedback": feedback,
                        "constraint": constraint,
                        "obeyed": 0
                    }
                    for feedback, constraint in zip(false_feedback, false_constraints)
                ]

                new_instance['plan'] = current_plan_steps
                break

    complete_feedback_instances.append(new_instance)

    # saving progress
    with open(os.path.join(problem_directory, "v5data.json"), "w") as f:
        json.dump(complete_feedback_instances, f, indent=4)

        
        

