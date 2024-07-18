import json
import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import jsonlines
from openai import OpenAI
from tqdm import tqdm

from ..utils.openai import openai_call
from ..utils.preprocess import parse_conversation

client = OpenAI()

PRELOGUE_INSTRUCTIONS = """
For this task, you will receive the dialogue history between two conversational agents, the social goal of one of the agents, and the final goal achieving score recieved by this agent. Your objective is to find the critical utterances (might be one utterance or more than one utterance) that contribute significantly to the success or failure of the agent's goal.

The critical utterance should be labeled as 'YES' and other to be labeled as 'NO'. Whether one utterance is critical or not depends on the response of that utterance. If the response from the other agent directly indicates the success of the agent's goal or failure of the agent's goal, then that utterance is critical. If the response from the other agent does not directly indicate the success or failure of the agent's goal, then that utterance is not critical.

For the goal achieving score, if it is <5, the agent fails, so you need to think which utterance is the most important one that leads to the failure of the goal and assign the critical utterance that leads to the failure to be "YES". If it is >=5, the agent succeeds, so you need to think which utterances is the most important one that leads to the success and assign that utterance to be "NO".
"""


def get_epilogue_instructions(agent: str) -> str:
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


def generate_single_key_utterance_prompt(
    conversation: List[Tuple[str, str]],
    goal: Dict[str, Any],
    score: float,
    agent: str,
) -> Tuple[str, Dict[str, List[Any]]]:
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


def assign_key_utterances_for_conversation(
    prompt: str,
    llm_name: str = "gpt-3.5-turbo",
) -> Dict[str, int] | Any:
    """Assign key_utterances to the entire conversation based on a GPT response."""
    response = openai_call(prompt, llm_name)
    if response is None:
        return None
    else:
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            formatted_response = extract_json(response)
            if formatted_response is None:
                print(
                    "Failed to extract JSON string from response; returning empty dictionary"
                )
                print(response)
                return {}
            try:
                result = json.loads(formatted_response)
            except json.JSONDecodeError:
                print(
                    "Failed to load formatted JSON string; returning empty dictionary"
                )
                print(formatted_response)
                return {}
        return result


def extract_json(text: str) -> str | None:
    # Use regex to find the JSON string within the text
    match = re.search(r"\{\n.*?\n\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        return json_str
    else:
        return None


def generate_key_utterance_recognition(
    data_dir: str, llm_name: str, input_file: str, output_file: str
) -> None:
    with jsonlines.open(os.path.join(data_dir, input_file), "r") as reader:
        data = list(reader)

    print(len(data))
    results = []
    for episode in tqdm(data):
        conversation, goals = parse_conversation(episode)
        agents = list(goals.keys())
        for agent in agents:
            prompt, key_prompt_dict = generate_single_key_utterance_prompt(
                conversation, goals[agent], episode["scores"][agent], agent
            )
            key_utterance_judgements = assign_key_utterances_for_conversation(
                prompt, llm_name
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
                os.path.join(data_dir, output_file), "w"
            ) as writer:
                writer.write_all(results)
