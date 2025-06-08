from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import get_close_matches
from itertools import combinations, permutations, product
import logging
import random
import subprocess
import tempfile
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from ..parsers.optic_parser import OpticParser
from .evaluation import FitnessEvaluator
import re
import os

API_KEY = os.environ['OPENAI_API_KEY']

class OptimizationResult:
    def __init__(self, success: bool, plan: Optional[list[dict]]=None, failure_reason: Optional[str]=None):
        self.success = success
        self.plan = plan
        self.failure_reason = failure_reason

    success: bool
    plan: Optional[list[dict]]
    failure_reason: Optional[str]

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

class GeneticOptimizer:
    def __init__(self, problem_path, domain_path, max_iter: Optional[int]=3):
        with open(problem_path, "r") as f:
            self.problem_text = f.read()

        self.goal = self.problem_text.split("(:goal")[1].split("(:")[0].strip()[:-1]

        self.problem_path = problem_path
        self.domain_path = domain_path
        self.max_iter = max_iter

        self.fitness_evaluator = FitnessEvaluator()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.optic_path = os.path.join(current_dir, 'optic-cplex')

        # process the problem and domain text to be ready to sample predicates
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

        self.predicates = predicates
        self.objects = objects
        self.init_predicates_cleaned = init_predicates_cleaned
        self.action_duration_nums = action_duration_nums
        self.all_constraints = constraints

    @property
    def predicate_templates(self):
        return [
            f"({pred_name} {' '.join(['?'+arg for arg in arguments])})"
            for pred_name, arguments in self.predicates.items()
        ]

    def _plan(self, constraints: list[str], timeout: Optional[str]="10s", **kwargs) -> tuple[bool, list | None]:
        if len(constraints) > 1:
            constraints_block = "\n\t".join(constraints)
            constraints_tag = f"(:constraints (and\n\t{constraints_block}\n))"
        elif len(constraints) == 1:
            constraints_tag = f"(:constraints {constraints[0]})"
        else:
            constraints_tag = ""

        new_problem = self.problem_text.replace("(:goal", f"{constraints_tag}\n(:goal")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as tf:
            tf.write(new_problem)
            candidate_problem_path = tf.name
            tf.close()

            
            command = ["timeout", timeout, self.optic_path, "-N", self.domain_path, candidate_problem_path]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, 
                    universal_newlines=True, text=True, encoding="utf-8")
            
            stdout, stderr = process.communicate()
            parser = OpticParser()
            plan = parser.injest_stdout(stdout)

            os.remove(candidate_problem_path)

            if plan is None:
                return False, None
            else:
                return True, plan['actions']
            
    def repair_constraint(self, llm_constraint: str, mutate=False, mutation_type=None) -> str:
        """
        Repair a constraint that potentially invalid
        """

        original_constraint = llm_constraint

        try:
            if not llm_constraint.startswith("("):
                llm_constraint = "(" + llm_constraint
            if not llm_constraint.endswith(")"):
                llm_constraint = llm_constraint + ")"

            modifier = llm_constraint.split("(")[1]

            valid_modifier_regex = [r"always ", r"sometime ", r"within \d+ ", r"at-most-once ", r"sometime-after ", r"sometime-before ", r"always-within \d+ ", r"hold-during \d+ \d+ ", r"hold-after \d+ ", r"at end "]
            if not mutate:
                if not any([re.match(valid_modifier, modifier) for valid_modifier in valid_modifier_regex]):
                    valid_modifier = get_close_matches(modifier, valid_modifier_regex, n=1, cutoff=0)[0]
                    valid_modifier = valid_modifier.replace("\\d+", str(random.choice(self.action_duration_nums)))
                    llm_constraint = llm_constraint.replace(modifier, valid_modifier)
            else:
                if mutation_type == 'change_modifier':
                    valid_modifier = random.choice(valid_modifier_regex)
                    valid_modifier = valid_modifier.replace("\\d+", str(random.choice(self.action_duration_nums)))
                    llm_constraint = llm_constraint.replace(modifier, valid_modifier)

            def repair_pred_content(pred_content, negate=False):

                negative = False
                if pred_content.startswith("not "):
                    negative = True
                    pred_content = pred_content.split("(")[1].split(")")[0]

                predicate = pred_content.split()[0]
                args = pred_content.split()[1:]

                if not predicate in self.predicates:
                    repaired_predicate = get_close_matches(predicate, self.predicates.keys(), n=1, cutoff=0)[0]
                    pred_content = pred_content.replace(predicate, repaired_predicate)
                    predargs = self.predicates[repaired_predicate]
                else:
                    predargs = self.predicates[predicate]

                objects_list = []
                for argument in predargs:
                    if "(either" in argument:
                        arg_options = argument.split("(either ")[1].split(")")[0].split()
                        combined_objects = self.objects[arg_options[0]].objects + self.objects[arg_options[1]].objects
                        objects_list.append(combined_objects)
                    else:
                        objects_list.append(self.objects[argument].objects)

                new_args = []
                for arg, ol in zip(args, objects_list):
                    if not mutate:
                        if arg not in ol:
                            repaired_arg = get_close_matches(arg, ol, n=1, cutoff=0)[0]
                            new_args.append(repaired_arg)
                        else:
                            new_args.append(arg)
                    else: # for mutation, randomly replace an argument
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

                if mutate and mutation_type == 'negate_predicate':
                    negate = [random.random() < 0.5 for _ in range(len(preds))]
                else:
                    negate = [False for _ in range(len(preds))]

                repaired_preds = [
                    f"({repair_pred_content(pred.strip()[1:-1], negate=should_negate)})"
                    for pred, should_negate in zip(preds, negate)
                ]
                for pred, repaired_pred in zip(preds, repaired_preds):
                    llm_constraint = llm_constraint.replace(pred, repaired_pred)
            else:
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

                repaired_pred = repair_pred_content(buffer[1:-1])
                llm_constraint = llm_constraint.replace(buffer[1:-1], repaired_pred)

            return llm_constraint
        except Exception as e:
            logging.exception(f"Error repairing constraint: {e} {llm_constraint} (original: {original_constraint})")
            return None
    
    def mutate(self, individual: list[str]) -> list[str]:
        """
        Mutate an individual by either adding, removing, modifying, or duplicating and modifying a constraint
        """

        mutation_rand = random.random()

        arg_mutation_type = None
        
        if mutation_rand < 0.1 and len(individual) > 1:
            individual.remove(random.choice(individual))
            arg_mutation_type = 'remove_constraint'
        elif mutation_rand < 0.2:
            constraint_type = random.choice(list(self.all_constraints.keys()))
            new_constraint = random.choice(self.all_constraints[constraint_type])
            individual.append(new_constraint)
            arg_mutation_type = 'add_constraint'
        elif mutation_rand < 0.5:
            to_duplicate = random.choice(individual)
            mutation_type = random.choice(['negate_predicate', 'change_modifier', 'change_arg'])
            mutated_duplicate = self.repair_constraint(to_duplicate, mutate=True, mutation_type=mutation_type)
            individual.append(mutated_duplicate)
            arg_mutation_type = f'duplicate_constraint_and_modify[{mutation_type}]'
        else:
            to_mutate = random.randint(0, len(individual) - 1)
            mutation_type = random.choice(['negate_predicate', 'change_modifier', 'change_arg'])    
            individual[to_mutate] = self.repair_constraint(individual[to_mutate], mutate=True, mutation_type=mutation_type)
            arg_mutation_type = f'modify_constraint[{mutation_type}]'

        return individual, arg_mutation_type
    
    def crossover(self, parent1: list[str], parent2: list[str]) -> list[str]:
        """
        Perform crossover between two parents to produce a child
        """

        if min(len(parent1), len(parent2)) - 1 <= 1:
            return parent1 + parent2
        else:
            crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            return child
            
    def optimize_constraints(self, nl_constraints: list[str], ga_candidates: list, **kwargs):
        """
        Optimize the constraints using the genetic optimizer
        """

        num_generations = self.max_iter
        population = ga_candidates
        population_size = 20

        print(f"NL Constraints: {nl_constraints}")

        for generation in range(num_generations):
            logging.info(f"\tstarted generation {generation}")

            # Score the individuals in the generation
            plans = [plan for _, plan, _, _ in population]
            scores, raw_scores = self.fitness_evaluator(plans, nl_constraints) # raw score is logits. If number of adherences assessed is tied, use this as a proxy for model confidence for the tiebreaker
            scored_population = [(constraints, plan, satisfied, arg_mutation_type, raw_score) for (constraints, plan, _, arg_mutation_type), satisfied, raw_score in zip(population, scores, raw_scores)]

            # Check the best score
            scored_population.sort(key=lambda x: (x[2], x[4]), reverse=True)
            logging.info(f"\tbest score: {scored_population[0][2]}")
            if scored_population[0][2] == 1.0: # stop early if all constraints are satisfied
                logging.info("\tAll constraints satisfied, ending optimization...")
                break

            # Select the top half of the population to be the parents
            next_generation = [individual[:-1] for individual in scored_population[:population_size//2]]
            parents = [individual[0] for individual in next_generation]

            # Create children to bring the population back to its original size
            children = []
            while len(children) < population_size//2:
                temporary_children = []
                for i in range(population_size):
                    parent1, parent2 = random.sample(parents, 2)
                    child = self.crossover(parent1, parent2)
                    if random.random() < 0.5:
                        child, arg_mutation_type = self.mutate(child)
                    else:
                        arg_mutation_type = "no_mutation"
                    temporary_children.append((child, arg_mutation_type))

                with ThreadPoolExecutor(max_workers=20) as executor:
                    futures = {executor.submit(self._plan, child[0]): child for child in temporary_children}

                    for future in as_completed(futures):
                        success, plan = future.result()
                        child, arg_mutation_type = futures[future]
                        if success:
                            children.append((child, plan, None, arg_mutation_type))

                # success, plan = self._plan(child)
                # if success:
                #     children.append((child, plan, None))

            population = next_generation + children

        best_constraints, best_plan, best_score, arg_mutation_type, best_raw_score = scored_population[0]
        return best_constraints, best_plan, best_score, best_raw_score, arg_mutation_type