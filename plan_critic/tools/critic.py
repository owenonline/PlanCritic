from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import zip_longest
import json
import operator
import random
import re
import signal
import subprocess
import tempfile
import time
import os
from typing import Annotated, List, Optional, Sequence, TypedDict
from pydantic import BaseModel
from tqdm import trange
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from collections import defaultdict
import json_repair
import threading
import logging
import uuid
import couchdb
from openai import OpenAI
import instructor
from .constraint_optimization import GeneticOptimizer, OptimizationResult
from .map_handling import MapHandler
import pickle

# FEEDBACK_PROCESS_EXAMPLES = [
#     {
#         "new_goal": ["Make sure the scout asset only visits the endpoint once", "Make sure the scout asset and salvage asset cross paths at most once"],
#         "old_goals": [
#             "Limit the scout asset (`sct_ast_0`) to visiting the endpoint (`wpt_end`) at most one time throughout the entire plan.",
#             "Ensure that the ship salvage asset (`shp_sal_ast_0`) is only on top of the scout asset (`sct_ast_0`) at most once throughout the entire plan."
#         ],
#         "new_goal_list": [
#             "Ensure that either the debris asset (`deb_ast_0`) is only on underwater debris `u_deb_b_0_end` at most once, or the ship salvage asset (`shp_sal_ast_0`) is only on top of the scout asset (`sct_ast_0`) at most once throughout the entire plan."
#         ]
#     },
#     {
#         "new_goal": ["We need to clear the route from debris station 0 to the endpoint within 5 hours"],
#         "new_goal_list": [
#             "Ensure that after time step 5, the route between `deb_stn_0` and `wpt_end` is always unblocked."
#         ]
#     },
#     {
#         "new_goal": ["Don't remove any underwater debris"],
#         "new_goal_list": [
#             "Ensure that the underwater debris u_deb_ini_b_0 remains at wpt_ini at all times.",
#             "Ensure that the underwater debris u_deb_b_0_end remains at wpt_b_0 at all times.",
#         ]
#     },
#     {
#         "new_goal": ["Ensure that we never visit waypoint a"],
#         "new_goal_list": [
#             "Ensure that the scout asset `sct_ast_0` never visits `wpt_a_0`.",
#             "Ensure that the debris asset `deb_ast_0` never visits `wpt_a_0`.",
#             "Ensure that the ship salvage asset `shp_sal_ast_0` never visits `wpt_a_0`."
#         ]
#     }
# ]

# CONSTRAINT_TRANSLATION_EXAMPLES = [
#     {
#         "nl": "At the end of the plan, ensure that either there is a connected route from `wpt_a_0` to `shp_dck_0`, or the route from `wpt_b_0` to `wpt_ini` is not blocked.",
#         "pred": "(at end (or (is_location_connected wpt_a_0 shp_dck_0) (is_location_not_blocked wpt_b_0 wpt_ini)))",
#     },
#     {
#         "nl": "Ensure that either the scout asset `sct_ast_0` never visits `wpt_end` or location `deb_stn_0` is always blocked for traversal to `wpt_end`, but not both at the same time.",
#         "pred": "(at-most-once (or (at sct_ast_0 wpt_end) (is_location_not_blocked deb_stn_0 wpt_end)))",
#     },
#     {
#         "nl": "Ensure that, at all times, there is a clear path between `wpt_ini` and `wpt_a_0`, and also between `deb_stn_0` and `wpt_ini`. Keep these routes unblocked throughout the entire process of restoring the waterway.",
#         "pred": "(always (and (is_location_connected wpt_ini wpt_a_0) (is_location_connected deb_stn_0 wpt_ini)))"
#     },
#     {
#         "nl": "Ensure that, two time steps into the plan, the scout asset `sct_ast_0` is at location `n_deb_ini_a_0`.",
#         "pred": "(hold-after 2 (on sct_ast_0 n_deb_ini_a_0))"
#     },
#     {
#         "nl": "At the end of the plan, ensure that the underwater debris `u_deb_ini_b_0` is at location `wpt_b_0`.",
#         "pred": "(at end (at u_deb_ini_b_0 wpt_b_0))"
#     }
# ]

# ACTION_EXPLANATIONS = {
#     "move_debris_asset": "(?loc1 - location ?loc2 - location ?ast - debris_asset) -> moves debris_asset from loc1 to loc2",
#     "remove_normal_debris_total": "(?loc1 - location ?loc2 - location ?deb - normal_debris ?ast - debris_asset) -> removes the debris normal_debris covering loc1 and loc2 using debris_asset",
#     "remove_normal_debris_partial": "(?loc1 - location ?loc2 - location ?deb - normal_debris ?ast - debris_asset) -> removes the debris normal_debris covering loc1 and loc2 partially using debris_asset",
#     "remove_underwater_debris_total": "(?loc1 - location ?loc2 - location ?deb - underwater_debris ?ast - debris_asset) -> removes the debris underwater_debris covering loc1 and loc2 using debris_asset",
#     "remove_underwater_debris_partial": "(?loc1 - location ?loc2 - location ?deb - underwater_debris ?ast - debris_asset) -> removes the debris underwater_debris covering loc1 and loc2 partially using debris_asset",
#     "unload_debris_debris_station": "(?loc - location ?ast - debris_asset) -> debris_asset unloads all of the debris it is carrying at debris station loc",
#     "move_scout_asset": "(?loc1 - location ?loc2 - location ?ast - scout_asset) -> moves scout_asset from loc1 to loc2",
#     "scout_location": "(?loc - location ?ast - scout_asset) -> scout_asset scouts location loc, making any underwater debris at loc visible to the other assets",
#     "move_ship_salvage_asset": "(?loc1 - location ?loc2 - location ?ast - ship_salvage_asset) -> moves ship_salvage_asset from loc1 to loc2",
#     "ship_salvage_asset_salvage_ship": "(?loc - location ?shp - ship ?ast - ship_salvage_asset) -> salvage asset ship_salvage_asset salvages ship shp located at loc",
#     "ship_salvage_asset_dock_ship": "(?loc - location ?shp - ship ?ast - ship_salvage_asset) -> ship_salvage_asset docks ship shp (which it had previously salvaged) at a dock specified by loc",
#     "authority_make_location_unrestricted": "(?loc - location ?aut - authority) -> authority aut makes location loc unrestricted, allowing assets to travel to it."
# }

# PROBLEM_ARCHETYPES = {
#     "All underwater debris is removed": ["(at end (is_location_not_blocked wpt_ini wpt_b_0))", "(at end (is_location_not_blocked wpt_b_0 wpt_end))"],
#     "Waypoint b is made unrestricted": ["(at end (is_location_unrestricted wpt_b_0))"],
#     "Scout asset reaches end point before debris asset moves": ["(sometime-before (at deb_ast_0 wpt_a_0) (at sct_ast_0 wpt_end))", "(sometime-before (at deb_ast_0 wpt_b_0) (at sct_ast_0 wpt_end))", "(sometime-before (at deb_ast_0 shp_dck_0) (at sct_ast_0 wpt_end))", "(sometime-before (at deb_ast_0 deb_stn_0) (at sct_ast_0 wpt_end))"],
#     "No assets visit waypoint a": ["(always (not (at deb_ast_0 wpt_a_0)))", "(always (not (at sct_ast_0 wpt_a_0)))", "(always (not (at shp_sal_ast_0 wpt_a_0)))"],
#     "Step 6 happens before step 5": ["(sometime-before (not (at shp_sal_ast_0 wpt_ini)) (is_underwater_debris_visible wpt_end))", "(sometime-before (not (at deb_ast_0 wpt_ini)) (is_underwater_debris_visible wpt_end))"],
#     "All of the underwater debris is removed and none of the normal debris is removed": ["(at end (is_location_not_blocked wpt_ini wpt_b_0))", "(at end (is_location_not_blocked wpt_b_0 wpt_end))", "(at end (at n_deb_ini_a_0 wpt_a_0))", "(at end (at n_deb_a_0_end wpt_a_0))"],
#     "Debris asset ends at waypoint b": ["(at end (at deb_ast_0 wpt_b_0))"],
#     "All assets are at the ship dock at the end of the plan": ["(at end (at deb_ast_0 shp_dck_0))", "(at end (at shp_sal_ast_0 shp_dck_0))", "(at end (at sct_ast_0 shp_dck_0))"],
#     "Scout asset reaches shipwreck before debris asset reaches shipwreck": ["(sometime-before (at deb_ast_0 wpt_end) (at sct_ast_0 wpt_end))"],
#     "Scout asset reaches shipwreck before debris asset reaches shipwreck and no underwater debris is removed": ["(sometime-before (at deb_ast_0 wpt_end) (at sct_ast_0 wpt_end))", "(at end (at u_deb_ini_b_0 wpt_b_0))", "(at end (at u_deb_b_0_end wpt_b_0))"]
# }

POP_SIZE = 20

def timeout_handler(signum, frame):
    raise Exception

class FeedbackPlannerState(TypedDict):
    chat_messages: Annotated[Sequence[BaseMessage], operator.add]
    optimization_outcome: OptimizationResult
    nl_feedback: Sequence[str]
    optimizer: GeneticOptimizer
    identifier: tuple[str, str]
    session_id: str
    use_ga: bool

class FeedbackPlanner:

    def __init__(self, domain, problem, problem_archetypes, couchdb_database, constraint_model="gpt-4o") -> None:

        # set up the Validate path
        self.validate_path = os.path.join("workspace", "plan_critic", "tools", "Validate")
        
        # set up the message store
        nosql_server = couchdb.Server(url=os.environ["COUCHDB_URL"])
        if not couchdb_database in nosql_server:
            self.database = nosql_server.create(couchdb_database)
        else:
            self.database = nosql_server[couchdb_database]

        # set up the utilities we'll be using
        self.chat_model = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o", temperature=0)
        self.constraint_model = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model=constraint_model, temperature=0)
        self.constraint_model_id = constraint_model

        # store the class-level values we'll be using
        self.domain_path = f"/workspace/domains/{domain}/domain.pddl"
        self.problem_path = f"/workspace/domains/{domain}/feedback/instance-{problem}/instance-{problem}.pddl"
        self.problem_archetypes = problem_archetypes

        # retreive and store the domain specific context
        with open(f"/workspace/domains/{domain}/domain_context.json", "r") as f:
            domain_context = json.load(f)

            self.feedback_process_examples = domain_context['feedback_process_examples']
            self.constraint_translation_examples = domain_context['constraint_translation_examples']
            self.action_explanations = domain_context['action_explanations']

        # set up the genetic optimizer
        self.genetic_optimizer = GeneticOptimizer(self.problem_path, self.domain_path)

        # create the initial plan, to be used as the base plan for each replanning attempt
        success, plan = self.genetic_optimizer._plan([])
        if not success:
            raise Exception("This should never happen")
        self.base_plan, _, _ = self.format_plan(plan)

    def format_plan(self, plan):
        """
        Given a raw PDDL plan, create the summaries and visual queues that allow it to be nicely displayed in the frontend
        """

        if plan is None:

            formatted_plan = {
                "duration": 0,
                "steps": []
            }

        else:

            duration = float(plan[-1]['time_step']) + float(plan[-1]['duration'])

            # aggregate actions by time step
            steps = defaultdict(list)
            for step in plan:
                time_step = step.pop('time_step')
                steps[time_step].append(step)

            # summarize each of the time steps
            batch_messages = []
            tsa = []
            for time_step, actions in steps.items():
                messages = [
                    HumanMessage(content="At time step {step}, the following actions will be executed: {actions}\nSummarize it in 20 words or less and convert pddl object names into natural language if possible without losing information (e.g. wpt_a_0 to waypoint a0 is ok, shp_dck_0 to ship dock 0 is ok). Do not mention the time step. For example, say 'Salvage asset moves to waypoint a' instead of 'at time step, moves salvage asset to waypoint'. The action explanations are here: {action_explanations}".format(step=time_step, actions=", ".join([action['action'] for action in actions]), action_explanations=self.action_explanations))
                ]
                batch_messages.append(messages)
                tsa.append((time_step, actions))

            batch_responses = self.chat_model.batch(batch_messages)
            mapper = MapHandler()
            # format the plan
            summarized_steps = {}
            for (time_step, actions), agent_message in zip(tsa, batch_responses):
                step_summary = agent_message.content
                step_summary = step_summary.replace('"', "")
                if "\n" in step_summary:
                    step_summary = step_summary.split("\n")[-1]

                summarized_steps[time_step] = {"summary": step_summary, "details": mapper([action['action'] for action in actions])}

            formatted_plan = {
                "duration": duration,
                "steps": [
                    {
                        "time_step": time_step,
                        "summary": step_summary['summary'],
                        "details": step_summary['details']
                    }
                    for time_step, step_summary in summarized_steps.items()
                ]
            }

        step_number = 0
        step_string = ""
        raw_step_string = ""
        for step in formatted_plan['steps']:
            step_number += 1
            step_string += f"Step {step_number}: {step['summary']}\n"
            raw_step_string += f"{step['time_step']}: {step['details']}\n"

        return formatted_plan, step_string, raw_step_string

    def test_plan_adherence(self, domain_path: str, problem_path: str, plan_archetype: str, actions: list[dict], timeout: Optional[str]="10s") -> bool:
        """
        Given a plan and a plan archetype, verify that the plan is a valid realization of the plan archetype
        """

        if actions is None:
            # Planning failed, so the constraints generated are definitely not valid
            return False

        with open(problem_path, "r") as f:
            problem_text = f.read()

        required_constraints = self.problem_archetypes[plan_archetype]

        for constraint in required_constraints:

            constraints_tag = f"(:constraints {constraint})"

            new_problem = problem_text.replace("(:goal", f"{constraints_tag}\n(:goal")

            # try:
                # signal.signal(signal.SIGALRM, timeout_handler)
                # signal.alarm(10)  # Set the alarm for 10 seconds

            with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as temp_problem_file:
                temp_problem_file.write(new_problem)
                candidate_problem_path = temp_problem_file.name
                temp_problem_file.close()

                candidate_plan_pddl_steps = [f"{action['time_step']}: {action['action']} [{action['duration']}]" for action in actions]
                candidate_plan_block = "\n".join(candidate_plan_pddl_steps)

                with tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False) as temp_plan_file:
                    temp_plan_file.write(candidate_plan_block)
                    candidate_plan_path = temp_plan_file.name
                    temp_plan_file.close()

                    command = ["timeout", timeout, self.validate_path, "-t", "0.001", "-v", domain_path, candidate_problem_path, candidate_plan_path] # timeout is linux, gtimeout is mac
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, 
                            universal_newlines=True, text=True, encoding="utf-8")
                    
                    stdout, _ = process.communicate()

                    os.remove(candidate_problem_path)
                    os.remove(candidate_plan_path)

                    # signal.alarm(0)
                    
                    # if any of the required constraints are violated, return False since the plan does not adhere to the archetype specifications
                    if not "Successful plans:" in stdout:
                        return False
                        
            # except Exception as e:
            #     print(f"Validation failed: {e}")
            #     return 0
            
        return True

    def run_plan(self, message) -> None:
        """
        Given some user feedback, generate a new plan revision
        """

        # unpack the message
        preferences = message['preferences']
        problem_archetype = message['problem_archetype']

        # initialize the testing results
        testing_results = {
            "after_llm" : {
                "plan": None,
                "constraints": None,
                "optic_success": False,
                "lstm_approved": False,
            },
            "after_ga" : {
                "plan": None,
                "constraints": None,
                "optic_success": False,
                "lstm_approved": False,
            },
            "constraint_llm": self.constraint_model_id,
            "preferences": preferences,
            "problem_archetype": problem_archetype,
        }

        # create the mid-level goals

        ## get the steps referenced in the preferences
        all_prefs = "\n".join(preferences)
        included_steps = re.findall(r"step (\d+)", all_prefs.lower())
        if len(included_steps) > 0:
            included_steps = [f"Step {int(step)}. {self.base_plan['steps'][int(step)-1]['summary']}" for step in included_steps]
            included_steps = "\n".join(included_steps)
        else:
            included_steps = ""

        ## assemble the few-shot prompt for the agent
        user_example_messages = [
            {
                "role": "user",
                "content": f"Here are my high-level goals: {example['new_goal']}"
            }
            for example in self.feedback_process_examples
        ]
        assistant_example_messages = [
            {
                "role": "assistant",
                "content": f"{example['new_goal_list']}"
            }
            for example in self.feedback_process_examples
        ]
        messages = [
            {
                "role": "system",
                "content": "You are helping a user taking a list of high-level goals and create a list of mid-level (BUT STILL NATURAL LANGUAGE) goals to reflect the high level goals in a symbolically grounded manner. Be sure to mention all objects to which the feedback applies by name. Return all of your responses in JSON with no additional commentary. The user should be able to directly serialize your response."
            }
        ] + [message for pair in zip_longest(user_example_messages, assistant_example_messages) for message in pair] + [
            {
                "role": "system",
                "content": f"To help you out with this next one, you have been provided with the list of available predicates in the domain, the currently instantiated objects, and the plan steps the user is referring to. Also when a waterway is referred to as restricted, this ONLY EVER MEANS restricted by an authority; it is NOT related to the location of any ships or assets.\nPredicates: {self.genetic_optimizer.predicate_templates}\nObjects: {self.genetic_optimizer.objects}"
            },
            {
                "role": "user",
                "content": f"Here are my high-level goals: {preferences}\n{included_steps}"
            }
        ]
        class MidlevelResponse(BaseModel):
            midlevel_goals: List[str]

        ## invoke the agent to generate the mid-level goals, and load the resulting JSON
        midlevel_model = instructor.from_openai(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
        goal_list = midlevel_model.chat.completions.create(
            model="gpt-4o-mini",
            response_model=MidlevelResponse,
            messages=messages,
        ).midlevel_goals

        # guess initial constraints

        ## assemble the few-shot prompt for the agent
        user_messages = [
            HumanMessage(content=f"{example['nl']}")
            for example in self.constraint_translation_examples
        ]
        assistant_messages = [
            AIMessage(content=f"{example['pred']}")
            for example in self.constraint_translation_examples
        ]
        messages = [ # Modified to indicate that roundabout results are sometimes required
            SystemMessage(content="You are helping a user understand how to translate natural language feedback into PDDL constraints that can be used to generate a plan. Your response should be a single PDDL constraint. You may have to use a roundabout solution to achieve your goal, as predicates can only say something is true. For example, if you are trying to ensure that u_deb_ini_b_0 isn't removed, your constraint might be 'at end (at u_deb_ini_b_0 wpt_b_0)'; you're saying that the debris DOES remain at its original location rather than that it DOESN'T get removed."),
        ] + [message for pair in zip_longest(user_messages, assistant_messages) for message in pair]
        batch_messages = [
            messages + [
                SystemMessage(content=f"To help you out with this next one, you have been provided with the list of available predicates in the domain and the currently instantiated objects. You also get the list of valid conditionals. Each constraint must have a conditional\nPredicates: {self.genetic_optimizer.predicate_templates}\nObjects: {self.genetic_optimizer.objects}\nConditionals: ['always', 'sometime', 'within <number>', 'at-most-once', 'sometime-after', 'sometime-before', 'always-within', 'hold during <number> <number>', 'hold-after <number>', 'at end']"),
                HumanMessage(content=goal)
            ]
            for goal in goal_list
        ]

        ## generate at least one syntactically correct constraint set
        def process_message(message):
            message_content = message.content
            if "\n" in message_content:
                message_content = message_content.split("\n")[0]
            return message_content.strip()

        valid_constraints = []
        while len(valid_constraints) == 0:
            try:
                agent_messages = self.constraint_model.batch(batch_messages)
                message_contents = [process_message(message) for message in agent_messages]
                valid_constraints = [self.genetic_optimizer.repair_constraint(content) for content in message_contents]
                valid_constraints = [constraint for constraint in valid_constraints if constraint is not None]
            except Exception as e:
                logging.warning(f"Constraint generation failed: {e}")
                pass

        logging.info(f"Generated {len(valid_constraints)} SYNTACTICALLY valid constraints: {json.dumps(valid_constraints, indent=4)}")

        ### Test LLM generated plan for plannability AND alignment to the goals

        success, plan = self.genetic_optimizer._plan(valid_constraints)

        if success:
            plan_adherence, _ = self.genetic_optimizer.fitness_evaluator([plan], goal_list)
            adherence = plan_adherence[0] == 1
        else:
            adherence = False

        testing_results["after_llm"]["plan"] = plan
        testing_results["after_llm"]["constraints"] = valid_constraints
        testing_results["after_llm"]["optic_success"] = bool(success)
        testing_results["after_llm"]["lstm_approved"] = bool(adherence)

        ground_truth_correct_after_llm = self.test_plan_adherence(self.genetic_optimizer.domain_path, self.genetic_optimizer.problem_path, problem_archetype, plan)
        testing_results["after_llm"]["ground_truth_verification"] = bool(ground_truth_correct_after_llm)

        # run genetic algorithm

        ## generate the first round of individuals via mutation for the genetic algorithm
        if success:
            candidate_individuals = [(valid_constraints, plan, None, "no_mutation")]
        else:
            candidate_individuals = []

        with tqdm(total=POP_SIZE, desc="Generating candidates") as pbar:
            while len(candidate_individuals) < POP_SIZE:
                temp_individuals = []
                for i in range(50):
                    new_individual = valid_constraints.copy()
                    for _ in range(random.randint(1, 5)):
                        new_individual, arg_mutation_type = self.genetic_optimizer.mutate(new_individual)

                    # remove any potential None constraints that could arise from an issue with the mutation
                    new_individual = [constraint for constraint in new_individual if constraint is not None]

                    temp_individuals.append((new_individual, arg_mutation_type))

                with ThreadPoolExecutor(max_workers=POP_SIZE) as executor:
                    futures = {executor.submit(self.genetic_optimizer._plan, individual[0]): individual for individual in temp_individuals}

                    for future in as_completed(futures):
                        success, plan = future.result()
                        individual, arg_mutation_type = futures[future]
                        if success:
                            candidate_individuals.append((individual, plan, None, arg_mutation_type))
                            pbar.update(1)

        logging.info(f"Generated {len(candidate_individuals)} candidate individuals")

        ## run the genetic algorithm
        logging.info(f"Starting genetic algorithm")
        best_constraints, best_plan, best_score, best_raw_score, arg_mutation_type = self.genetic_optimizer.optimize_constraints(goal_list, candidate_individuals)
        logging.info("Genetic algorithm complete")

        # analyze planning outcome
        ground_truth_correct_after_ga = self.test_plan_adherence(self.genetic_optimizer.domain_path, self.genetic_optimizer.problem_path, problem_archetype, best_plan)
        testing_results["after_ga"]["ground_truth_verification"] = bool(ground_truth_correct_after_ga)

        ### Log the success of planning after the GA
        testing_results["after_ga"]["plan"] = best_plan
        testing_results["after_ga"]["constraints"] = best_constraints
        testing_results["after_ga"]["optic_success"] = bool(best_plan is not None)
        testing_results["after_ga"]["lstm_approved"] = bool(best_score == 1)
        testing_results["after_ga"]["mutation_type"] = arg_mutation_type
        testing_results["after_ga"]["logit_score"] = str(best_raw_score)

        # publish the testing results
        self.database.save(testing_results)

        logging.info("Plan revision complete")