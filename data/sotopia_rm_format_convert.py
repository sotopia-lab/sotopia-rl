import json
from pprint import pprint

import jsonlines


def calc_reward(utter, goal_score):
    utter_attrib = utter[1]
    if utter_attrib == -1:
        reward = -1
    else:
        reward = utter_attrib / 3 * goal_score
    return reward


if __name__ == "__main__":

    with jsonlines.open(
        "sotopia_pi_openai_log_attribution.jsonl", "r"
    ) as reader:
        dataset = list(reader)

    rewards = []
    historys = []
    systems = []
    prompts = []
    for data in dataset:
        agent_name = data["agent"]
        goal_score = data["goal_score"]
        scenario = data["scenario"]
        goal = data["goal"]
        history = []

        for speaker, utter in data["attributed_utterances"].items():
            if agent_name in speaker:
                systems.append(
                    f"The scenario is {scenario}. The goal of {agent_name} is {goal}."
                )
                prompts.append(
                    f'How much does "{speaker} {utter[0]}" contribute to the goal of {agent_name}?'
                )
                history += [f"{speaker} {utter[0]}"]
                historys.append(history)
                reward = calc_reward(utter, goal_score)
                rewards.append(reward)

    formulated_dataset = []
    for prompt, reward, system, history in zip(
        prompts, rewards, systems, historys
    ):
        data = {
            "instruction": prompt,
            "input": "",
            "output": "",
            "value": reward,
            "system": system,
            "history": history,
        }
        formulated_dataset.append(data)

    with open("sotopia_pi_utterance_reward.json", "w") as writer:
        json.dump(formulated_dataset, writer, indent=4)
