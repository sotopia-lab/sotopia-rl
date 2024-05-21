import json
from pprint import pprint

import jsonlines

with jsonlines.open("sotopia_pi_openai_log_attribution.jsonl", "r") as reader:
    dataset = list(reader)

rewards = []
prompts = []
for data in dataset:
    agent_name = data["agent"]
    goal_score = data["goal_score"]
    goal = data["goal"]
    dialogues = []
    dialogue = ""

    goal_description = f"The goal of the agent is {goal}."
    for speaker, utter in data["attributed_utterances"].items():
        if agent_name in speaker:
            dialogue += f"{speaker}: {utter[0]}\n"
            utter_attrib = utter[1]
            if utter_attrib == -1:
                reward = -1
            else:
                reward = utter_attrib / 3 * goal_score
            prompt = f"{dialogue}\n{goal_description}\nFocus on the last utterance. What is the reward for the last utterance for achieving the goal? {reward}\n"
            prompts.append(prompt)
            rewards.append(reward)

formulated_dataset = []
for prompt, reward in zip(prompts, rewards):
    data = {"instruction": prompt, "input": "", "output": "", "value": reward}
    formulated_dataset.append(data)

with open("sotopia_pi_utterance_reward.json", "w") as writer:
    json.dump(formulated_dataset, writer, indent=4)
