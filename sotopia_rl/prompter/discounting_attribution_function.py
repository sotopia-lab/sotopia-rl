from typing import Any, Dict, List, Tuple, Optional, Type, Union, TypeVar
from pydantic import BaseModel, Field
from openai import OpenAI
import json

def assign_attributions_for_conversation(
    agent: str,
    conversation: List[Tuple[str, str]],
    discounting_factor: float,
) -> Dict[str, int] | Any:
    count_utterances = 0
    for i, (speaker, _) in enumerate(conversation):
        if speaker == agent:
            count_utterances += 1
    
    attribution_dict = {}
    for i, (speaker, _) in enumerate(conversation):
        if speaker == agent:
            attribution_dict[f"Utterance {i//2} by {speaker}"] = discounting_factor ** (count_utterances - 1 - i//2)
    return attribution_dict

def calc_attributed_reward(attributed_data: List[Dict[str, float | int]], attribution_instruction_name: str, goal_score: float | int) -> List[Dict[str, Any]]:
    utterance_reward_map = {}
    for k, v in attributed_data.items():
        utterance_reward_map[k] = v * goal_score
    return utterance_reward_map

# unified function
def get_attribution_single_conv(conversation, agent, goals, episode, llm_name, attribution_instruction_name):
    attribution_scores = assign_attributions_for_conversation(
        agent, conversation, discounting_factor=0.9
    )
    attribution_rewards = calc_attributed_reward(attribution_scores, attribution_instruction_name, episode["scores"][agent])
    return attribution_rewards