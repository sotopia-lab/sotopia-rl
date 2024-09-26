import json
from copy import deepcopy

if __name__ == "__main__":

    with open(
        "./data/gpt35_gpt4_prompt_response_pairs.json", "r"
    ) as f:
        dataset = json.load(f)

    rewards = []
    systems = []
    prompts = []
    history_pairs = []
    model_names = []
    pair_ids = []

    for i in range(len(dataset)):
        data = dataset[i]
        agent_name = data["end_agent_name"]
        scenario = data["scenario"]
        goal = data["end_agent_goal"]
        raw_history = data["history"]
        history = deepcopy(raw_history)
        message = data["message"]
        history.append([message, ""])
        # raw_history = data["history"]
        # history = []
        # for pair in raw_history:
        #     history.append(pair[0])
        #     history.append(pair[1])
        # history.append(data["message"])
        # history = history[1:]

        for model_name in ["gpt-3.5-turbo", "gpt-4o"]:
            model_response = data[model_name]
            if not model_response.startswith("[action] "):
                formatted_response = f"Utterance {len(raw_history)} by {agent_name} said: {model_response}"
            else:
                formatted_response = f"Utterance {len(raw_history)} by {agent_name} {model_response[9:]}"
            systems.append(
                f"The scenario is {scenario}. The goal of {agent_name} is {goal}."
            )
            prompts.append(
                f"{formatted_response}\nDo you think it is a key utterance contributing to the success or failure of {agent_name}?"
            )
            history_pairs.append(history)
            model_names.append(model_name)
            pair_ids.append(i)
            rewards.append(0.0)

    formulated_dataset = []
    for prompt, reward, system, history, model_name, pair_id in zip(
        prompts, rewards, systems, history_pairs, model_names, pair_ids
    ):
        data = {
            "instruction": prompt,
            "input": "",
            "output": "",
            "value": reward,
            "system": system,
            "history": history,
            # [
            #     (history[i], history[i + 1])
            #     for i in range(0, len(history) - 1, 2)
            # ],
            "model_name": model_name,
            "pair_id": pair_id,
        }
        formulated_dataset.append(data)

    with open("./data/sotopia_pi_preference_data.json", "w") as writer:
        json.dump(formulated_dataset, writer, indent=4)
