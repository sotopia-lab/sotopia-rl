import json
import re
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm

from sotopia_rl.prompter.direct_attribution_instructions import (
    ATTRIBUTION_INSTRUCTIONS_DICT,
)


def openai_call(prompt: str, model: str = "gpt-3.5-turbo") -> str | None:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )
    return response.choices[0].message.content

def extract_json(text: str) -> str | None:
    # Use regex to find the JSON string within the text
    match = re.search(r"\{\n.*?\n\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        return json_str
    else:
        return None

def get_attribution_formatting_instructions(agent: str) -> str:
    return f"""
Please format your response as JSON with the following structure:
{{
    "Utterance 1 by {agent}": 0,
    "Utterance 2 by {agent}": 2,
    ...
}}
The utterance numbers should correspond to their order in the conversation. Each score should reflect how much the utterance contributed to achieving the agent's goals.
"""

def get_single_attribution_prompt(
    conversation: List[Tuple[str, str]],
    goal: str,
    score: float,
    agent: str,
    attribution_instruction: str
) -> Tuple[str, Dict[str, List[Any]]]:
    """Generate a single prompt for GPT based on the entire conversation, agent's goals, and final goal achieving score."""
    prompt = f"{attribution_instruction}\n\n"
    prompt += "Conversation between two agents:\n\n"
    prompt += f"Agent for Evaluation: {agent}\n\n"
    prompt += f"Agent Goal: {goal}\n\n"
    prompt += f"Final goal achieving score: {score}\n\n"
    prompt += "Conversation:\n"
    for i, (speaker, utterance) in enumerate(conversation):
        prompt += f"Utterance {i//2} by {speaker}: {utterance}\n"
    prompt += "\n" + get_attribution_formatting_instructions(agent)
    return prompt

def assign_attributions_for_conversation(
    prompt: str, llm_name: str = "gpt-3.5-turbo"
) -> Dict[str, int] | Any:
    """Assign attributions to the entire conversation based on a GPT response."""
    response = openai_call(prompt, llm_name)
    if response is None:
        print("Failed to get response from OpenAI; returning empty dictionary")
        return {}
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
            result = json.loads(formatted_response)
        return result

def calc_reward(utter_attrib: float, attribution_instruction_name: str, goal_score: float) -> float:
    denominator = {"default": 3, "5-scale": 5}[attribution_instruction_name]
    if utter_attrib == -1:
        reward = -1.0
    else:
        reward = utter_attrib / denominator * goal_score
    return reward

def calc_attributed_reward(attributed_data: List[Dict[str, float | int]], attribution_instruction_name: str, goal_score: float | int) -> List[Dict[str, Any]]:
    utterance_reward_map = {}
    for k, v in tqdm(attributed_data.items()):
        utterance_reward_map[k] = calc_reward(v, attribution_instruction_name, goal_score)
    return utterance_reward_map

# unified function
def get_attribution_single_conv(conversation, agent, goals, episode, llm_name, attribution_instruction_name):
    attribution_instruction = ATTRIBUTION_INSTRUCTIONS_DICT[attribution_instruction_name]
    prompt = get_single_attribution_prompt(
        conversation, goals[agent], episode["scores"][agent], agent, attribution_instruction=attribution_instruction
    )
    attribution_scores = assign_attributions_for_conversation(
        prompt, llm_name=llm_name
    )
    attribution_rewards = calc_attributed_reward(attribution_scores, attribution_instruction_name, episode["scores"][agent])
    return attribution_rewards
