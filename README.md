
A LLM+GA approach to adaptying symbolic plans to user feedback while maintaining a guarantee of feasibility.

### Installation

Local installation of the repository can be performed with the following steps::

1.  Clone the repository
    ```
    git clone git@github.com:owenonline/PlanCritic.git
    cd PlanCritic
    ```

2.  Build the docker image
    ```
    docker compose up --build -d
    ```

3.  Install the dependencies
    ```
    docker exec -it <container_id> /bin/bash
    pip install -e .
    pip install -r adherence_model_training/requirements.txt
    ```

## Usage

This system can work with arbitrary PDDL3 domains, so long as they are structured in the following way:
```
/workspace/domains/<domain_name>/
├── domain.pddl
├── domain_context.json
├── experiment_config.json
└── feedback/
    ├── <problem1>/
    │   ├── <problem1>.pddl
    └── <problem2>/
        └── ...
```

If you are adding a new domain, you will need to follow the instructions in `adherence model training/README.md` to generate the additional files needed to run the experiment. If you are only using existing domains, you may skip this step. 

**You can validate that this is the case by checking if your directory structure is as follows:**
```
/workspace/domains/<domain_name>/
├── domain.pddl
├── domain_context.json
├── experiment_config.json
├── model/
│   └── best_lstm_model.pth
└── feedback/
    ├── <problem1>/
    │   ├── <problem1>.pddl
    │   ├── situation_info.txt
    │   └── v5data.json
    └── <problem2>/
        └── ...
```

To configure the experiment, you will need to create a `domain_context.json` file for your domain. This file should contain the following information:
- feedback_process_examples: A list of examples of how the user might describe their goal to the system.
- constraint_translation_examples: A list of examples of how the user might describe their constraint to the system.
- action_explanations: A dictionary of action names to their natural language explanations.

You will also need to create a `experiment_config.json` file for your domain. This file should contain the following information:
- problem: The number of the problem instance you want to use. (e.g. 1)
- problem_archetypes: A dictionary of problem archetypes to the set of predicates that are required to be true for you to consider the plan to have satisfied the archetype (a validator will be used to strictly check if the generated plan satisfies all of the specified predicates). Consider this to be the set of user intentions you want to test.
- couchdb_database: The name of the couchdb database you want to use to store the experiment results. **This will append new records to the database if it already exists.**

> Note about problem archetypes: These should each involve a different objective. During execution, the system will create rephrasings of each archetype to simulate how different users might describe the same goal.

To run the experiment, you will need to run the following command:

```
python3 -m plan_critic.run --domain <domain_name>
```

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


## Contact

Owen Burns - burnso@acm.org
