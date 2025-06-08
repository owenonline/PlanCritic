# Plan Critic

A system for executing a plan-feedback-replan loop on a user's command, sending a natural language plan summary to the user interface, receiving their feedback, and using a genetic algorithm to evolve the plan to match that feedback.

### Installation

Local installation of the repository can be performed with the following steps::

1.  Clone the repository
    ```
    git clone git@github.com:owenonline/PlanCritic.git
    ```

2.  cd to this folder

    ```
    cd PlanCritic
    ```

3.  Build the docker image
    ```
    docker compose up --build -d
    ```

## Usage

Once installed via `pip`, the test data must be generated. Follow the instructions in `adherence model training/README.md` to generate the test data. When that is finished, run the following command to execute the experiment:

```
docker exec <container_id> python -m plan_critic.run
```

## Configuring the experiment

To add new domains, create a new folder in the `domains` folder and add the `domain.pddl` file to it. Follow the structure of the existing domain. The files you will need to add for a new domain are:

- `domain.pddl`
- `domain_context.json`
- `feedback/instance-*/instance-*.pddl`

**Additionally, you will need to follow the instructions in `adherence model training/README.md` to train an instance of the critic model for that domain.**

To use your new domain in an experiment, you will need to update the `experiment_config.json` file with the name of your domain and the number of the instance you want to use. Then add the problem archetypes you want to test. Each archetype should be a natural language objective that is not incompatible with the base objective of the problem instance. The value of that archetype in the dictionary should be the set of predicates that are required to be true for you to consider the plan to have satisfied the archetype. These are the constraints that will be used to determine if the critic model was correct.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


## Contact

Owen Burns - oburns@andrew.cmu.edu

Project Link: [https://gitlab.com/cmu_aart/onr_cai/environments/marine_cadastre](https://gitlab.com/cmu_aart/onr_cai/environments/marine_cadastre)