import json
import os
from collections import OrderedDict
from pprint import pprint

import jsonlines
from openai import OpenAI
from tqdm import tqdm

# Set environment variables for OpenAI API
# with open("openai_api.key", "r") as f:
#    os.environ["OPENAI_API_KEY"] = f.readline().strip()

client = OpenAI()


def openai_call(prompt):
    model_name = "gpt-3.5-turbo"
    """Make a call to OpenAI API with a specific prompt."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    # not robust
    reply = response.choices[0].message.content
    if "gpt-4" in model_name:
        # extract { ... } from the response
        reply = reply[reply.find("{") : reply.rfind("}") + 1]
    return reply


def parse_conversation(episode):
    """Extract and parse conversation and goals from the episode."""
    conversation = episode["social_interactions"].split("\n\n")
    goals = episode["social_goals"]
    agent1, agent2 = list(goals.keys())
    parsed_conversation = []
    for i, utterance in enumerate(conversation):
        if utterance.startswith(agent1):
            speaker = agent1
        elif utterance.startswith(agent2):
            speaker = agent2
        else:
            continue  # Skip any unparsable utterances
        parsed_conversation.append((speaker, utterance))
    return parsed_conversation, goals


PRELOGUE_INSTRUCTIONS = """
For this task, you will receive the dialogue history between two conversational agents, the social goal of one of the agents, and the final goal achieving score recieved by this agent. Your objective is to find the critical utterances (might be one utterance or more than one utterance) that directly decide the success or failure of the agent's goal.

The critical utterance should be labeled as 'YES' and other to be labeled as 'NO'. Whether one utterance is critical or not depends on the response of that utterance. If the response from the other agent directly indicates the success of the agent's goal or failure of the agent's goal, then that utterance is critical. If the response from the other agent does not directly indicate the success or failure of the agent's goal, then that utterance is not critical.

For the goal achieving score, if it is <5, the agent fails, so you need to think which utterance is the most important one that leads to the failure of the goal and assign the critical utterance that leads to the failure to be "YES". If it is >=5, the agent succeeds, so you need to think which utterances is the most important one that leads to the success and assign that utterance to be "3".
"""


def get_epilogue_instructions(agent):
    return f"""
Please provide YES or NO for each of the utterances made by {agent}. If you believe an utterance directly leads to the success or failure of the agent's goal, assign it as 'YES'. Otherwise, assign it as 'NO'.

Please format your response as JSON with the following structure:
{{
    "Utterance 1 by {agent}": "YES",
    "Utterance 2 by {agent}": "NO",
    ...
}}
The utterance numbers should correspond to their order in the conversation. Each score should reflect how much the utterance contributed to achieving the agent's goals.
"""


def generate_single_attribution_prompt(conversation, goal, score, agent):
    """Generate a single prompt for GPT based on the entire conversation, agent's goals, and final goal achieving score."""
    prompt = f"{PRELOGUE_INSTRUCTIONS}\n\n"
    prompt += f"Agent Goal: {goal}\n\n"
    prompt += f"Final goal achieving score: {score}\n\n"
    prompt += "Conversation:\n"
    key_utterance_dict = OrderedDict()
    for i, (speaker, utterance) in enumerate(conversation):
        prompt += f"Utterance {i//2} by {speaker}: {utterance}\n"
        key_utterance_dict[f"Utterance {i//2} by {speaker}"] = [
            utterance,
            -1,
        ]
    prompt += "\n" + get_epilogue_instructions(agent)
    return prompt, key_utterance_dict


def assign_attributions_for_conversation(prompt):
    """Assign attributions to the entire conversation based on a GPT response."""
    response = openai_call(prompt)
    return json.loads(response)


if __name__ == "__main__":
    with jsonlines.open(
        "../data/example_episodes_with_scores.jsonl", "r"
    ) as reader:
        data = list(reader)

    print(len(data))
    results = []
    for episode in tqdm(data[1:]):
        conversation, goals = parse_conversation(episode)
        agents = list(goals.keys())
        for agent in agents:
            prompt, key_prompt_dict = generate_single_attribution_prompt(
                conversation, goals[agent], episode["scores"][agent], agent
            )
            key_utterance_judgements = assign_attributions_for_conversation(
                prompt
            )
            for key in key_prompt_dict:
                if agent in key and key in key_utterance_judgements:
                    key_prompt_dict[key][1] = key_utterance_judgements[key]
            results.append(
                {
                    "episode_id": episode["episode_id"],
                    "scenario": episode["scenario"],
                    "agent": agent,
                    "goal": goals[agent],
                    "key_utterance_judgement": key_prompt_dict,
                    "is_first_speaker": agent == agents[0],
                    "goal_score": episode["scores"][agent],
                }
            )

            with jsonlines.open(
                "../data/openai_log_key_utterance.jsonl", "w"
            ) as writer:
                writer.write_all(results)
