import json
import jsonlines


def calc_reward(utter_attrib: float, goal_score: float) -> float:
    if utter_attrib == -1:
        reward = -1.0
    else:
        reward = utter_attrib / 3 * goal_score
    return reward


if __name__ == "__main__":

    with jsonlines.open(
        "./data/sotopia_pi_openai_log_attribution.jsonl", "r"
    ) as reader:
        dataset = list(reader)

    rewards = []
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
                    reward = calc_reward(utter[1], goal_score)
                    rewards.append(reward)

                if not is_first_speaker:
                    # Store the current history excluding the agent's previous utterance
                    systems.append(
                        f"The scenario is {scenario}. The goal of {agent_name} is {goal}."
                    )
                    prompts.append(
                        f"{speaker} {utter[0]}\nHow much does this utterance contribute to the goal of {agent_name}?"
                    )
                    # Create a copy of the current history excluding the last two utterances for pairing
                    history_pairs.append(history[1:-1])
                    reward = calc_reward(utter[1], goal_score)
                    rewards.append(reward)

    formulated_dataset = []
    for prompt, reward, system, history in zip(
        prompts, rewards, systems, history_pairs
    ):
        data = {
            "instruction": prompt,
            "input": "",
            "output": "",
            "value": reward,
            "system": system,
            "history": [
                (history[i], history[i + 1])
                for i in range(0, len(history) - 1, 2)
            ],
        }
        formulated_dataset.append(data)

    with open("./data/sotopia_pi_utterance_reward.json", "w") as writer:
        json.dump(formulated_dataset, writer, indent=4)
