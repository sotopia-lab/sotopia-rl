import re
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from sotopia_rl.prompter.one_pass_instructions import ATTRIBUTION_INSTRUCTIONS_DICT

REGEX = "^Utterance (?:[0-9]|[1-9][0-9]) by {agent}$"

def check_regex_formatting(target: str, agent: str, regex: str = REGEX) -> bool:
    return bool(re.match(regex.format(agent=agent), target))

def extract_turn_number(text: str) -> int:
    match = re.search(r"Utterance ([0-9]+) by", text)
    if match:
        return int(match.group(1))
    else:
        return -1

def openai_call(prompt: str, model: str = "gpt-3.5-turbo") -> str | None:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def get_attribution_formatting_instructions(agent: str) -> str:
    return f"""
Your format should strictly follow the regex pattern below:
{REGEX.format(agent=agent)}
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
    prompt += f"Chosen agent for Evaluation: {agent}\n\n"
    prompt += f"Agent's Goal: {goal}\n\n"
    prompt += "Conversation:\n"
    for i, (speaker, utterance) in enumerate(conversation):
        prompt += f"Utterance {i//2} by {speaker}: {utterance}\n"
    prompt += "\n" + get_attribution_formatting_instructions(agent)
    return prompt

def assign_attributions_for_conversation(
    prompt: str, conversation: list, agent: str, llm_name: str = "gpt-3.5-turbo"
) -> Dict[str, int] | Any:
    for i in range(5):
        uttr_attr_dict = {}
        uttr_count = 0
        for j, (speaker, _) in enumerate(conversation):
            if speaker == agent:
                uttr_count += 1
            uttr_attr_dict[f"Utterance {j//2} by {speaker}"] = 0
        response = openai_call(prompt, llm_name).strip()

        if response is None:
            print("Failed to get response from OpenAI; returning empty dictionary")
            return {}
        else:
            try:
                result = check_regex_formatting(response, agent)
                assert -1 < extract_turn_number(response) < uttr_count
                assert response in uttr_attr_dict
            except Exception:
                if i < 4:
                    print("Response does not match the regex expression; retrying")
                else:
                    print("Response length does not match the number of agent utterances after 5 attempts; returning original dictionary")
    uttr_attr_dict[response] = 1
    return uttr_attr_dict

def calc_reward(utter_attrib: float, goal_score: float) -> float:
    return utter_attrib * goal_score

def calc_attributed_reward(attributed_data: List[Dict[str, float | int]], attribution_instruction_name: str, goal_score: float | int) -> List[Dict[str, Any]]:
    utterance_reward_map = {}
    for k, v in attributed_data.items():
        utterance_reward_map[k] = {"reward": calc_reward(v, goal_score), "attribution": v}
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
        conversation, goals[agent], episode["scores"][agent], agent, attribution_instruction=attribution_instruction
    )
    attribution_scores = assign_attributions_for_conversation(
        prompt, conversation, agent, llm_name=llm_name
    )
    # attribution_scores = fill_in_attribution_scores(conversation, raw_attribution_scores, agent)
    attribution_rewards = calc_attributed_reward(attribution_scores, attribution_instruction_name, episode["scores"][agent])
    return attribution_rewards
