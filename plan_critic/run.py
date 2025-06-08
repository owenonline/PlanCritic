import argparse
import logging
import time
from typing import Optional
import pickle
import os
import yaml
import json
from tqdm import tqdm
import base64
from openai import OpenAI
from pydantic import BaseModel
from .tools.critic import FeedbackPlanner

# DOMAIN_PATH = "/Users/owenburns/workareas/Carnegie Mellon PlanCritic/plan-critic/test_domain.pddl"
# PROBLEM_PATH = "/Users/owenburns/workareas/Carnegie Mellon PlanCritic/plan-critic/test_instance.pddl"
# PROBLEM_ARCHETYPES = [
#     "All underwater debris is removed",
#     "Waypoint b is made unrestricted",
#     "No assets visit waypoint a",
#     "Step 6 happens before step 5",
#     "All of the underwater debris is removed and none of the normal debris is removed",
#     "Debris asset ends at waypoint b",
#     "All assets are at the ship dock at the end of the plan",
#     "Scout asset reaches shipwreck before debris asset reaches shipwreck",
#     "Scout asset reaches shipwreck before debris asset reaches shipwreck and no underwater debris is removed",
#     "Scout asset reaches end point before debris asset moves",
# ]
EXPERIMENT_CONFIG = json.load(open("/workspace/experiment_config.json"))

if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO,
						format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

	logging.getLogger("http.client").disabled = True
	logging.getLogger("urllib3").disabled = True
	logging.getLogger("requests").disabled = True
	logging.getLogger("httpx").disabled = True
	logging.getLogger("httpcore.http11").disabled = True
	logging.getLogger("httpcore.connection").disabled = True
	logging.getLogger("openai._base_client").disabled = True

	if "rephrased_goals.pkl" in os.listdir():
		with open("rephrased_goals.pkl", "rb") as f:
			rephrased_goals = pickle.load(f)
	else:
		# rephrase the goals
		class Goal(BaseModel):
			phrasing: list[str]

		# load the UI image
		with open("cai_ui.png", "rb") as image_file:
			encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
		img_type="image/png"

		client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

		rephrased_goals = {}

		for goal in EXPERIMENT_CONFIG["problem_archetypes"].keys():
			response = client.beta.chat.completions.parse(
				model="gpt-4o",
				temperature=1,
				messages=[
					{
						"role": "user",
						"content": [
							{"type": "text", "text": f"""Imagine you are a human testing out the following user interface for a descision support system designed to assist with planning disaster relief operations. The test scenario is a shipwreck that needs to be recovered, and can be reached by traversing one of two waterways. One waterway has debris above the water that can be readily removed, and the other has debris below the water that must be scouted before it can be removed. You will be an objective that represents an additional goal that a user might ask for to modify the existing plan for ship recovery. Your job is to provide 30 possible rephrasings of that goal that represent the breadth of possible ways (e.g., some more conversational and some more authoritative, with a variety of language) that end users may state that desire while interacting with the system. These statements will then be used with the system to assess its performance. Return your response as a JSON list.

			Your objective: \"{goal}\""""},
							{
								"type": "image_url",
								"image_url": {"url": f"data:{img_type};base64,{encoded_image}"},
							}
						]
					}
				],
				response_format=Goal
			)

			parsed_response: Goal = response.choices[0].message.parsed
			print(f"Got {len(parsed_response.phrasing)} rephrasings for goal: {goal}")
			rephrased_goals[goal] = parsed_response.phrasing

		with open("rephrased_goals.pkl", "wb") as f:
			pickle.dump(rephrased_goals, f)
	
	planner = FeedbackPlanner(
		domain=EXPERIMENT_CONFIG["domain"],
		problem=EXPERIMENT_CONFIG["problem"],
		problem_archetypes=EXPERIMENT_CONFIG["problem_archetypes"],
		couchdb_database=EXPERIMENT_CONFIG["couchdb_database"]
	)

	archetypes = list(rephrased_goals.keys())

	for current_archetype in archetypes:

		current_rephrasing_list = rephrased_goals[current_archetype]

		logging.info(f"Processing {len(current_rephrasing_list)} rephrasings for archetype {current_archetype}")
		for current_rephrasing in tqdm(current_rephrasing_list):

			message = {
				"preferences": [current_rephrasing], # the natural language rephrasing of the goal, simulating how a user might describe their goal to the system
				"problem_archetype": current_archetype, # the goal that the user is trying to achieve. This is crucial for measuring performance on the correct goal.
			}
			planner.run_plan(message)

	logging.info("All rephrasings have been processed.")
