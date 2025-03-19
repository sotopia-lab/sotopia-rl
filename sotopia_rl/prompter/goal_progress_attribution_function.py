from typing import Any, Dict, List, Tuple, Optional, Type, Union, TypeVar
from pydantic import BaseModel, Field
from openai import OpenAI
import json
T = TypeVar("T", bound=BaseModel)

DEFAULT_PROMPT = """
You are given a conversation history between two agents. For now, you are the judge of the utterance of one of the agent, and your task is to judge how much of your agent's goal is achieved at certain point of the conversation. You will also be provided with the agent's final goal achieving score, which would help you in making the decision better. Note, the goal achieving score is between 0 and 10, where 0 means the goal is not achieved at all, and 10 means that the goal is fully achieved. 

### Your Agent's Name:
{agent}
### Your Agent's Goal:
{goal}
### Final Goal Achieving Score out of 10:
{score}
### Conversation:
{conversation}
"""

class GoalAchievingScore(BaseModel):
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
    curr_conv = []
    prev_score = 0
    attribution_dict = {}
    for i, (speaker, utterance) in enumerate(conversation):
        if speaker == agent and len(curr_conv) > 0:
            prompt = prompt_format.format(
                agent=agent,
                goal=goal,
                score=final_goal_score,
                conversation="\n".join(curr_conv)
            )
            response = openai_call_with_response_model(prompt, llm_name, GoalAchievingScore)
            score = response.score if response else prev_score
            attribution_dict[f"Utterance {i//2} by {speaker}"] = score - prev_score
            prev_score = score
        
        curr_conv.append(f"Utterance {i//2} by {speaker}: {utterance}")
    return attribution_dict

def calc_attributed_reward(attributed_data: List[Dict[str, float | int]], attribution_instruction_name: str, goal_score: float | int) -> List[Dict[str, Any]]:
    utterance_reward_map = {}
    for k, v in attributed_data.items():
        utterance_reward_map[k] = v / 10 * goal_score
    return utterance_reward_map

# unified function
def get_attribution_single_conv(conversation, agent, goals, episode, llm_name, attribution_instruction_name):
    prompt_format = DEFAULT_PROMPT
    attribution_scores = assign_attributions_for_conversation(
        prompt_format, agent, goals[agent], episode["scores"][agent], conversation, llm_name=llm_name
    )
    attribution_rewards = calc_attributed_reward(attribution_scores, attribution_instruction_name, episode["scores"][agent])
    return attribution_rewards