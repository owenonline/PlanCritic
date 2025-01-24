# Plan Critic

A component designed to modify plans based on user feedback.

## About

This repository contains a component designed to execute a plan-feedback-replan loop on a user's command, sending a natural language plan summary to the user interface, receiving their feedback, and using a genetic algorithm to evolve the plan to match that feedback.

## Getting Started

The client code in this repository assumes that the pycourier package is installed, as well as the requirements for the Paho MQTT client.  See [https://gitlab.com/cmu_aart/onr_cai/messaging_backbone](https://gitlab.com/cmu_aart/onr_cai/messaging_backbone) for details on installing.

### Installation

Local installation of the repository can be performed with the following steps::

1.  Clone the repository
    ```
    git clone https://gitlab.com/cmu_aart/onr_cai/plan-critic.git
    ```

2.  cd to this folder

    ```
    cd plan-critic
    ```

3.  Install locally via pip
    ```
    pip install -e .
    ```

## Usage

Once installed via `pip`, an environment can be started with the following command

```
python -m plan_critic.run [--config <config.yaml>]
```

The command line arguments are defined as follows:

- `config.yaml` (required):  Defines the mode in which the planner runs. Run in mode 1 for round trip translation, 2 for plan critic with genetic algorithm, and 3 when running the repharsing experiment.

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


## Contact

Owen Burns - oburns@andrew.cmu.edu

Project Link: [https://gitlab.com/cmu_aart/onr_cai/environments/marine_cadastre](https://gitlab.com/cmu_aart/onr_cai/environments/marine_cadastre)