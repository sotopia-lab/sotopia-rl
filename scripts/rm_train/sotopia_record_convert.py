import json
from collections import defaultdict
from typing import Dict, List, Tuple

from src.prompting.sotopia_utils import Agent, Environment

ENVIRONMENT_PROFILES = "../../data/profiles/environmentprofiles_v1.jsonl"
AGENT_PROFILES = "../../data/profiles/agentprofiles_v1.jsonl"
RELATIONSHIP_PROFILES = "../../data/profiles/relationshipprofiles_v1.jsonl"


def get_sotopia_profiles(env_file: str=ENVIRONMENT_PROFILES, agent_file: str=AGENT_PROFILES, relationship_file: str=RELATIONSHIP_PROFILES) -> Tuple[List[Tuple[str, str]], Dict[str, Environment], Dict[str, Agent], Dict[str, Dict[str, List[str]]]]:
    with open(env_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]

    code_names_count: Dict[str, int] = defaultdict(int)
    environments = []
    environment_dict = {}
    for profile in sorted(data, key=lambda x: x['codename']):
        env_obj = Environment(profile)
        if profile['codename'] in code_names_count:
            environments.append((
                "{}_{:05d}".format(profile['codename'],
                                    code_names_count[profile['codename']]
                                    ),
                env_obj._id
                ))
        else:
            environments.append((profile['codename'], env_obj._id))
        environment_dict[env_obj._id] = env_obj
        code_names_count[profile['codename']] += 1

    with open(agent_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]

    agent_dict = {}
    for profile in data:
        agent_obj = Agent(profile)
        agent_dict[agent_obj._id] = agent_obj

    with open(relationship_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]

    relationship_dict: Dict[str, Dict[str, List[str]]] = defaultdict(lambda : defaultdict(list))
    for profile in data:
        relationship_dict[profile['relationship']][profile['agent1_id']].append(profile['agent2_id'])
        relationship_dict[profile['relationship']][profile['agent2_id']].append(profile['agent1_id'])

    return environments, environment_dict, agent_dict, relationship_dict


if __name__ == "__main__":

    environments, environment_dict, agent_dict, relationship_dict = get_sotopia_profiles()
    with open(
        "../../data/sft_gpt4_selfplay.json", "r"
    ) as f:
        dataset = json.load(f)

    systems = []
    prompts = []
    history_pairs = []

    for data in dataset:
        agent_name = data["agent"]
        goal_score = data["goal_score"]
        scenario = data["scenario"]
        goal = data["goal"]
        is_first_speaker = data["is_first_speaker"]
        history = []

        for speaker, utter in data["attributed_utterances"].items():
            # Append all utterances to history
            history.append(f"{speaker} {utter[0]}")

            if agent_name in speaker:
                # Store the current history excluding the agent's current utterance
                if is_first_speaker:
                    systems.append(
                        f"The scenario is {scenario}. The goal of {agent_name} is {goal}."
                    )
                    prompts.append(
                        f"{speaker} {utter[0]}\nHow much does this utterance contribute to the goal of {agent_name}?"
                    )
                    # Create a copy of the current history excluding the last utterance for pairing
                    history_pairs.append(history[:-1])

                if not is_first_speaker:
                    # Store the current history excluding the agent's previous utterance
                    systems.append(
                        f"The scenario is {scenario}. The goal of {agent_name} is {goal}."
                    )
                    prompts.append(
                        f"{speaker} {utter[0]}\nHow much does this utterance contribute to the goal of {agent_name}?"
                    )
                    # Create a copy of the current history excluding the last two utterances for pairing
                    history_pairs.append(history[:-1] + [""])

    formulated_dataset = []
    for prompt, system, history in zip(
        prompts, systems, history_pairs
    ):
        data = {
            "instruction": prompt,
            "input": "",
            "output": "",
            "system": system,
            "history": [
                (history[i], history[i + 1])
                for i in range(0, len(history) - 1, 2)
            ],

        }
        formulated_dataset.append(data)

    with open("./data/sotopia_pi_gpt4_selfplay.json", "w") as writer:
        json.dump(formulated_dataset, writer, indent=4)
