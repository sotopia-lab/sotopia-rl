import json

import jsonlines


def calc_reward(utter_attrib: str | float, goal_score: float) -> float:
    global fail_count, total_count
    if type(utter_attrib) == int and utter_attrib == -1:
    if type(utter_attrib) is float and utter_attrib == -1:
        return 0
    if utter_attrib == "YES":
        return goal_score
    if utter_attrib == "NO":
        return 0
    raise ValueError(f"Invalid utterance attribute: {utter_attrib}")


def add_discounted_reward(temp_rewards: list[float]) -> list[float]:
    if temp_rewards[-1] != 0:
        gamma = 0.9
        for prev_idx in range(len(temp_rewards) - 2, -1, -1):
            temp_rewards[prev_idx] = (
                0.5 * temp_rewards[prev_idx] + 0.5 * temp_rewards[-1] * gamma
            )
            gamma *= 0.9
    return temp_rewards


if __name__ == "__main__":
    with jsonlines.open(
        "./data/sotopia_pi_openai_log_key_utterance.jsonl", "r"
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

        temp_rewards = []
        for speaker, utter in data["key_utterance_judgement"].items():
            # Append all utterances to history
            history.append(f"{speaker} {utter[0]}")

            if agent_name in speaker:
                # Store the current history excluding the agent's current utterance
                if is_first_speaker:
                    systems.append(
                        f"The scenario is {scenario}. The goal of {agent_name} is {goal}."
                    )
                    prompts.append(
                        f"{speaker} {utter[0]}\nDo you think it is a key utterance contributing to the success or failure of {agent_name}?"
                    )
                    # Create a copy of the current history excluding the last utterance for pairing
                    history_pairs.append(history[:-1])
                    reward = calc_reward(utter[1], goal_score)
                    temp_rewards.append(reward)
                    temp_rewards = add_discounted_reward(temp_rewards)

                if not is_first_speaker:
                    # Store the current history excluding the agent's previous utterance
                    systems.append(
                        f"The scenario is {scenario}. The goal of {agent_name} is {goal}."
                    )
                    prompts.append(
                        f"{speaker} {utter[0]}\nDo you think it is a key utterance contributing to the success or failure of {agent_name}?"
                    )
                    # Create a copy of the current history excluding the last two utterances for pairing
                    history_pairs.append(history[:-1] + [""])
                    reward = calc_reward(utter[1], goal_score)
                    temp_rewards.append(reward)
                    temp_rewards = add_discounted_reward(temp_rewards)

        # normalize temp_rewards
        if len(temp_rewards) > 0 and max(temp_rewards) > 0:
            temp_rewards = [
                reward / max(temp_rewards) for reward in temp_rewards
            ]
        rewards.extend(temp_rewards)

    formulated_dataset = []
    for prompt, reward, system, history in zip(
        prompts, rewards, systems, history_pairs
    ):
        data = {
            "instruction": prompt,
            "input": "",
            "output": "",
            "value": reward if reward > 0 else 0.1,
            "system": system,
            "history": [
                (history[i], history[i + 1])
                for i in range(0, len(history) - 1, 2)
            ],
        }
        formulated_dataset.append(data)

    # print(f"Fail count: {fail_count}/{total_count}")
    with open(
        "./data/sotopia_pi_key_utterance_discounted_reward.json", "w"
    ) as writer:
        json.dump(formulated_dataset, writer, indent=4)
