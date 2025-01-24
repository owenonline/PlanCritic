import argparse
import logging
import time
from typing import Optional
import pickle
import os
import yaml
from tqdm import tqdm
from .tools.critic import FeedbackPlanner

DOMAIN_PATH = "/Users/owenburns/workareas/Carnegie Mellon PlanCritic/plan-critic/test_domain.pddl"
PROBLEM_PATH = "/Users/owenburns/workareas/Carnegie Mellon PlanCritic/plan-critic/test_instance.pddl"

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

	component = FeedbackPlanner("PlanCritic", condition=3)
	component.start()

	if "rephrased_goals.pkl" in os.listdir():
		with open("rephrased_goals.pkl", "rb") as f:
			rephrased_goals = pickle.load(f)
	else:
		raise Exception("No rephrased goals found")

	initialization_response = component.initialize({
		"domain": DOMAIN_PATH,
		"problem": PROBLEM_PATH
	})

	# rephrased_goals.pop("All underwater debris is removed")
	# rephrased_goals.pop("Waypoint b is made unrestricted")
	# rephrased_goals.pop("Scout asset reaches end point before debris asset moves")
	# rephrased_goals.pop("No assets visit waypoint a")
	# rephrased_goals.pop("Step 6 happens before step 5")
	# rephrased_goals['All of the underwater debris is removed and none of the normal debris is removed'] = rephrased_goals['All of the underwater debris is removed and none of the normal debris is removed'][7+14:]

	rephrased_goals = {
		"Scout asset reaches end point before debris asset moves": rephrased_goals["Scout asset reaches end point before debris asset moves"],
	}

	archetypes = list(rephrased_goals.keys())

	assert len(initialization_response["history"]) == 1, "Plan critic failed to initialize"

	for current_archetype in archetypes:

		current_rephrasing_list = rephrased_goals[current_archetype]

		logging.info(f"Processing {len(current_rephrasing_list)} rephrasings for archetype {current_archetype}")
		for current_rephrasing in tqdm(current_rephrasing_list):

			message = {
				"preferences": [current_rephrasing],
				"problem_archetype": current_archetype,
				"domain": DOMAIN_PATH,
				"problem": PROBLEM_PATH,
			}
			planning_response = component.run_plan(message)

	logging.info("All rephrasings have been processed.")
