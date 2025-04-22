import json
import re
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from sotopia_rl.prompter.generic_templates import (
    SCALE_GUIDELINE_DICT, DIMENSION_DESCRIPTION_DICT, DIRECT_ATTRIBUTION_TEMPLATE, 
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
    "Utterance 0 by {agent}": 0,
    "Utterance 1 by {agent}": 2,
    ...
}}
The utterance numbers should correspond to their order in the conversation. Each score should reflect how much the utterance contributed to achieving the agent's goals. Please annotate every utterance made by an agent in the conversation, denoted "Utterance X by agent_name". For example, "Utterance 6 by Finnegan O'Malley". Please give a score even if the utterance is the end of the conversation.
"""

def get_single_attribution_prompt(
    conversation: List[Tuple[str, str]],
    agent: str,
    agent_goal: str,
    agent_background: str,
    dimension: str,
    scale: str,
) -> Tuple[str, Dict[str, List[Any]]]:
    scoring_guidelines = SCALE_GUIDELINE_DICT[scale]
    dimension_description = DIMENSION_DESCRIPTION_DICT[dimension]
    conversation_prompt = "Conversation:\n"
    for i, (speaker, utterance) in enumerate(conversation):
        conversation_prompt += f"Utterance {i//2} by {speaker}: {utterance}\n"
    formatting_instruction = get_attribution_formatting_instructions(agent)
    prompt = DIRECT_ATTRIBUTION_TEMPLATE.format(
        scoring_guidelines=scoring_guidelines,
        agent=agent,
        goal=agent_goal,
        agent_background=agent_background,
        conversation=conversation_prompt,
        dimension=dimension,
        dimension_description=dimension_description,
        formatting_instructions=formatting_instruction,
    )
    return prompt

def assign_attributions_for_conversation(
    prompt: str, conversation: list, agent: str, llm_name: str = "gpt-3.5-turbo"
) -> Dict[str, int] | Any:
    for i in range(5):
        uttr_count = 0
        for j, (speaker, _) in enumerate(conversation):
            if speaker == agent:
                uttr_count += 1
        response = openai_call(prompt + f"\nYou are supposed to be returning {uttr_count} attributions.", llm_name)
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
            try:
                for key in result:
                    result[key] = int(result[key])
            except ValueError:
                print("Failed to convert all values to integers; retrying")
                continue
        
        if uttr_count != len(result) and i < 4:
            print("Response length does not match the number of agent utterances; retrying")
        elif uttr_count == len(result):
            break
        else:
            print("Response length does not match the number of agent utterances after 5 attempts; returning original dictionary")
    return result

def calc_reward(utter_attrib: float, scale: str, dim_score: float) -> float:
    denominator = {"default": 3, "5_scale": 5, "10_scale": 10}[scale]
    if utter_attrib == -1:
        reward = -1.0
    else:
        reward = utter_attrib / denominator * dim_score
    return reward

def calc_attributed_reward(attributed_data: List[Dict[str, float | int]], scale: str, dim_score: float | int) -> List[Dict[str, Any]]:
    utterance_reward_map = {}
    for k, v in attributed_data.items():
        utterance_reward_map[k] = {"reward": calc_reward(v, scale, dim_score), "attribution": v}
    return utterance_reward_map


def fill_in_attribution_scores(
    conversation: List[Tuple[str, str]],
    raw_attribution_scores: Dict[str, Any],
    agent: str,
) -> Dict[str, Any]:
    attribution_dict = {}
    for i, (speaker, utterance) in enumerate(conversation):
        if speaker != agent:
            continue
        key = f"Utterance {i//2} by {speaker}"
        attribution_dict[key] = raw_attribution_scores.get(key, 0)
    return attribution_dict

# unified function
def get_attribution_single_conv(conversation, agent, goals, episode, rewards, llm_name, attribution_instruction_name):
    scale, dimension = attribution_instruction_name.split("-")
    assert scale in SCALE_GUIDELINE_DICT, f"Scale {scale} not in scale dict"
    assert dimension in DIMENSION_DESCRIPTION_DICT, f"Dimension {dimension} not in dimension dict"
    agent_goal = goals[agent]
    agent_background = episode["agents_background"][agent]
    prompt = get_single_attribution_prompt(
        conversation, agent, agent_goal, agent_background, dimension, scale
    )
    attribution_scores = assign_attributions_for_conversation(
        prompt, conversation, agent, llm_name=llm_name
    )
    attribution_rewards = calc_attributed_reward(attribution_scores, scale, rewards[agent][dimension])
    for key in attribution_rewards:
        attribution_rewards[key]["dimension"] = dimension
        attribution_rewards[key]["scale"] = scale
        attribution_rewards[key]["dim_score"] = rewards[agent][dimension]
    return attribution_rewards
