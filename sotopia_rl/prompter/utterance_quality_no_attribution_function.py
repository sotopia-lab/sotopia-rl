import json
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from openai import OpenAI
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

DEFAULT_PROMPT = """
Two agents are in a conversation. For now, you are the judge of the utterance of one of the agents. You are given the utterance or action of that agent at a certain point and the conversation before it. Your task is to judge how much would the utterance contribute to the final goal in a scale of 0 to 10. 0 means the utterance is not contributing at all, and 10 means the utterance is fully contributing to the final goal.

### Your Agent's Name:
{agent}
### Your Agent's Goal:
{goal}
### Conversation History:
{conversation}
### Your Agent's Utterance:
{utterance}
"""

class UtteranceScore(BaseModel):
    score: int = Field(ge=0, le=10)
    reasoning: str

def openai_call(prompt: str, model: str = "gpt-3.5-turbo") -> str | None:
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )
    return response.choices[0].message.content

def openai_call_with_response_model(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    response_model: Optional[Type[T]] = None
) -> Union[T, str, None]:
    client = OpenAI()
    prompt = prompt + "\n\n" + "### Your response should follow this json schema: \n" + str(response_model.model_json_schema())
    content = None
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = json.loads(response.choices[0].message.content)

            if response_model:
                # Assuming the content is already a dict; if it's a JSON string, you might need to load it first.
                return response_model.model_validate(content)
        except Exception:
            if not i == 2:
                print("Error in openai_call_with_response_model, trying again")
            else:
                print("Error in openai_call_with_response_model, tried 3 times and failed")

    return content

def assign_attributions_for_conversation(
    prompt_format: str,
    agent: str,
    goal: str,
    final_goal_score: int,
    conversation: List[Tuple[str, str]],
    llm_name: str = "gpt-3.5-turbo"
) -> Dict[str, int] | Any:
    prev_score = 0
    attribution_dict = {}
    for i, (speaker, utterance) in enumerate(conversation):
        if speaker == agent and i + 1 < len(conversation):
            prompt = prompt_format.format(
                agent=agent,
                goal=goal,
                score=final_goal_score,
                conversation="\n".join([f"{s}: {u}" for s, u in conversation[:i]]),
                utterance=utterance
            )
            response = openai_call_with_response_model(prompt, llm_name, UtteranceScore)
            score = response.score if response else prev_score
            attribution_dict[f"Utterance {i//2} by {speaker}"] = score
            prev_score = score
    return attribution_dict

def calc_attributed_reward(attributed_data: List[Dict[str, float | int]], attribution_instruction_name: str, goal_score: float | int) -> List[Dict[str, Any]]:
    utterance_reward_map = {}
    for k, v in attributed_data.items():
        utterance_reward_map[k] = v
    return utterance_reward_map

# unified function
def get_attribution_single_conv(conversation, agent, goals, episode, llm_name, attribution_instruction_name):
    prompt_format = DEFAULT_PROMPT
    attribution_scores = assign_attributions_for_conversation(
        prompt_format, agent, goals[agent], episode["scores"][agent], conversation, llm_name=llm_name
    )
    attribution_rewards = calc_attributed_reward(attribution_scores, attribution_instruction_name, episode["scores"][agent])
    return attribution_rewards
