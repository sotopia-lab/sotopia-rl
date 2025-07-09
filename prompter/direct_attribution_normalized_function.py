import json
import re
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from sotopia_rl.prompter.one_pass_instructions import ATTRIBUTION_INSTRUCTIONS_DICT


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
    goal: str,
    agent: str,
    attribution_instruction: str
) -> Tuple[str, Dict[str, List[Any]]]:
    """Generate a single prompt for GPT based on the entire conversation, agent's goals, and final goal achieving score."""
    prompt = f"{attribution_instruction}\n\n"
    prompt += f"Chosen agent for Evaluation: {agent}\n\n"
    prompt += f"Agent's Goal: {goal}\n\n"
    # prompt += f"Final goal achieving score: {score}\n\n"
    prompt += "Conversation:\n"
    for i, (speaker, utterance) in enumerate(conversation):
        prompt += f"Utterance {i//2} by {speaker}: {utterance}\n"
    prompt += "\n" + get_attribution_formatting_instructions(agent)
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

        if uttr_count != len(result) and i < 4:
            print("Response length does not match the number of agent utterances; retrying")
        elif uttr_count == len(result):
            break
        else:
            print("Response length does not match the number of agent utterances after 5 attempts; returning original dictionary")
    return result

def calc_reward(utter_attrib: float, attribution_instruction_name: str, goal_score: float, total_attributions: float) -> float:
    if total_attributions == 0:
        return 0.0
    reward = utter_attrib / total_attributions * goal_score
    return reward

def calc_attributed_reward(attributed_data: List[Dict[str, float | int]], attribution_instruction_name: str, goal_score: float | int) -> List[Dict[str, Any]]:
    utterance_reward_map = {}
    total_attributions = 0
    for k, v in attributed_data.items():
        total_attributions += v
    for k, v in attributed_data.items():
        utterance_reward_map[k] = {"reward": calc_reward(v, attribution_instruction_name, goal_score, total_attributions),
                                    "attribution": v}
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
def get_attribution_single_conv(conversation, agent, goals, episode, llm_name, attribution_instruction_name):
    attribution_instruction = ATTRIBUTION_INSTRUCTIONS_DICT[attribution_instruction_name]
    prompt = get_single_attribution_prompt(
        conversation, goals[agent], agent, attribution_instruction=attribution_instruction
    )
    attribution_scores = assign_attributions_for_conversation(
        prompt, conversation, agent, llm_name=llm_name
    )
    # attribution_scores = fill_in_attribution_scores(conversation, raw_attribution_scores, agent)
    attribution_rewards = calc_attributed_reward(attribution_scores, attribution_instruction_name, episode["scores"][agent])
    return attribution_rewards
