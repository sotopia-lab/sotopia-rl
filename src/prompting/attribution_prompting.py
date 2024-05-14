import json
import os
import re
from collections import OrderedDict
from pprint import pprint
from typing import Any, Dict, List, Tuple

import jsonlines
from openai import OpenAI
from tqdm import tqdm

# from ..utils.openai import openai_call
# from ..utils.preprocess import parse_conversation
from openai import OpenAI


def openai_call(prompt: str, model: str = "gpt-3.5-turbo") -> str | None:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content

def parse_conversation(
    episode: Dict[str, Any]
) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """Extract and parse conversation and goals from the episode."""
    conversation = episode["social_interactions"].split("\n\n")
    goals = episode["social_goals"]
    agent1, agent2 = list(goals.keys())
    parsed_conversation = []
    for utterance in conversation:
        if utterance.startswith(agent1):
            speaker = agent1
        elif utterance.startswith(agent2):
            speaker = agent2
        else:
            continue  # Skip any unparsable utterances
        parsed_conversation.append(
            (speaker, utterance[len(speaker) + 1 :].strip())
        )  # Strip the speaker from the utterance
    return parsed_conversation, goals

client = OpenAI()


PRELOGUE_INSTRUCTIONS = """
For this task, you will receive the dialogue history between two conversational agents, the social goal of one of the agents, and the final goal achieving score recieved by this agent. Your objective is to assess how much each of the agent's utterance (marked by the agent's name and the utterance number) contributed to the final goal achieving score. You also need to consider the response of the other agent in the conversation to evaluate the impact of the utterance. 

For the goal achieving score, if it is <5, the agent fails, so you need to think which utterance is the most important one that leads to the failure of the goal and assign the critical utterance that leads to the failure to be 3. If it is >=5, the agent succeeds, so you need to think which utterance is the most important one that leads to the success of the goal and assign the critical utterance that leads to the success to be 3. 

Following the same logic, if you believe an utterance had no impact on the final goal achieving score, please provide a score of 0. If you believe an utterance had a significant impact on the final goal achieving score, please provide a score of 3. If you believe an utterance had a moderate impact on the final goal achieving score, please provide a score of 1 or 2. As a special case, if you believe an utterance is redundant and unnecessary, please provide a score of -1.
"""


def get_epilogue_instructions(agent: str, goal: str, score: int) -> str:
    return f"""
Please format your response as JSON with the following structure:
{{
    "Utterance 1 by {agent}": 0,
    "Utterance 2 by {agent}": 2,
    ...
}}
The utterance numbers should correspond to their order in the conversation. Each score should reflect how much the utterance contributed to achieving the agent's goals.

For the goal achieving score, if it is <5, the agent fails, so you need to think which utterance is the most important one that leads to the failure of the goal and assign the critical utterance that leads to the failure to be 3. If it is >=5, the agent succeeds, so you need to think which utterance is the most important one that leads to the success of the goal and assign the critical utterance that leads to the success to be 3. 

Following the same logic, if you believe an utterance had no impact on the final goal achieving score, please provide a score of 0. If you believe an utterance had a significant impact on the final goal achieving score, please provide a score of 3. If you believe an utterance had a moderate impact on the final goal achieving score, please provide a score of 1 or 2. As a special case, if you believe an utterance is redundant and unnecessary, please provide a score of -1.
"""


def generate_single_attribution_prompt(
    conversation: List[Tuple[str, str]],
    goal: str,
    score: float,
    agent: str,
) -> Tuple[str, Dict[str, List[Any]]]:
    """Generate a single prompt for GPT based on the entire conversation, agent's goals, and final goal achieving score."""
    prompt = f"{PRELOGUE_INSTRUCTIONS}\n\n"
    prompt += f"Conversation between two agents:\n\n"
    prompt += f"Agent for Evaluation: {agent}\n\n"
    prompt += f"Agent Goal: {goal}\n\n"
    prompt += f"Final goal achieving score: {score}\n\n"
    prompt += "Conversation:\n"
    key_utterance_dict: Dict[str, List[Any]] = OrderedDict()
    for i, (speaker, utterance) in enumerate(conversation):
        prompt += f"Utterance {i//2} by {speaker}: {utterance}\n"
        key_utterance_dict[f"Utterance {i//2} by {speaker}"] = [
            utterance,
            0,
        ]
    prompt += "\n" + get_epilogue_instructions(agent, goal, score)
    return prompt, key_utterance_dict


def assign_attributions_for_conversation(prompt: str, llm_name: str = "gpt-3.5-turbo") -> Dict[str, int] | Any:
    """Assign attributions to the entire conversation based on a GPT response."""
    response = openai_call(prompt, llm_name)
    if response is None:
        print("Failed to get response from OpenAI; returning empty dictionary")
        return {}
    else:
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            print("Failed to decode JSON response; trying to extract JSON string from response")
            try:
                result = json.loads(extract_json(response))
            except json.JSONDecodeError:
                print("Failed to decode JSON string from response after extracting it; returning empty dictionary")
                return {}
        return result
        

def extract_json(text):
    # Use regex to find the JSON string within the text
    match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json_str
    else:
        return None

def generate_reward_attribution(data_dir: str, llm_name: str="gpt-3.5-turbo") -> None:
    with jsonlines.open(
        os.path.join(data_dir, "example_episodes_with_scores.jsonl"), "r"
    ) as reader:
        data = list(reader)

    print(len(data))
    results = []
    for episode in tqdm(data):
        conversation, goals = parse_conversation(episode)
        agents = list(goals.keys())
        for agent in agents:
            prompt, key_prompt_dict = generate_single_attribution_prompt(
                conversation, goals[agent], episode["scores"][agent], agent
            )
            attribution_scores = assign_attributions_for_conversation(prompt, llm_name=llm_name)
            for key in key_prompt_dict:
                if agent in key and key in attribution_scores:
                    key_prompt_dict[key][1] = attribution_scores[key]
            results.append(
                {
                    "episode_id": episode["episode_id"],
                    "scenario": episode["scenario"],
                    "agent": agent,
                    "goal": goals[agent],
                    "attributed_utterances": key_prompt_dict,
                    "is_first_speaker": agent == agents[0],
                    "goal_score": episode["scores"][agent],
                }
            )

            with jsonlines.open(
                os.path.join(data_dir, "openai_log_attribution.jsonl"), "w"
            ) as writer:
                writer.write_all(results)
