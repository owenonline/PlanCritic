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
import uuid

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

	parser = argparse.ArgumentParser()
	parser.add_argument("--domain", type=str, default=None, required=True)
	parser.add_argument("--prefix", type=str, default="/workspace/")
	parser.add_argument("--rephrasing_path", type=str, default=None)
	parser.add_argument("--save_rephrasings", type=bool, default=True)
	args = parser.parse_args()
	PREFIX = args.prefix
	DOMAIN = args.domain

	EXPERIMENT_CONFIG = json.load(open(f"{PREFIX}domains/{DOMAIN}/experiment_config.json"))

	if args.rephrasing_path is not None:
		with open(args.rephrasing_path, "rb") as f:
			rephrased_goals = pickle.load(f)
	else:
		# rephrase the goals
		class Goal(BaseModel):
			phrasing: list[str]

		# load the UI image
		with open(f"{PREFIX}cai_ui.png", "rb") as image_file:
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

		if args.save_rephrasings:
			with open(f"{PREFIX}rephrased_goals.pkl", "wb") as f:
				pickle.dump(rephrased_goals, f)
				print(f"Saved rephrased goals to {PREFIX}rephrased_goals.pkl")
	
	planner = FeedbackPlanner(
		domain=DOMAIN,
		problem=EXPERIMENT_CONFIG["problem"],
		problem_archetypes=EXPERIMENT_CONFIG["problem_archetypes"],
		couchdb_database=EXPERIMENT_CONFIG["couchdb_database"],
		prefix=PREFIX
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
