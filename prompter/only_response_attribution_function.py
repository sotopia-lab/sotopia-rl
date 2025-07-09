import json
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from openai import OpenAI
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

# DEFAULT_PROMPT = """
# Two agents are in a conversation. For now, you are the judge of the utterance of one of the agents. You are given the utterance or action of that agent and the immediate response it recieves. Your task is to judge how successful the utterance is and give a score between 0 and 10. 0 means the utterance is not successful at all, and 10 means the utterance is fully successful.

# You will also be provided with the agent's final goal achieving score, which would help you in making the decision better. Note, the goal achieving score is between 0 and 10, where 0 means the goal is not achieved at all, and 10 means that the goal is fully achieved.

# ### Your Agent's Name:
# {agent}
# ### Your Agent's Goal:
# {goal}
# ### Final Goal Achieving Score out of 10:
# {score}
# ### Your Agent's Utterance:
# {utterance}
# ### Immediate Response:
# {response}
# """

ONLY_RESPONSE_DIRECT_INSTRUCTIONS = """
## Reward Attribution Instructions for LLMs

Two agents are in a conversation. For now, you are the judge of the utterance of one of the agents.

1. Input Context:
   - You will receive the utterance or action of an agent in one turn of the conversation.
   - You will recieve the immediate response it receives after the utterance/action.
   - You will also be provided with the social goal of the agent and its final goal achievement score.

2. Objective:
   - Assign an importance value to the utterance based on its contribution to the final goal achievement score, judging from how postive or negative the response is. Note, you should only consider the immediate response it recieves, not the quality of the utterance itself. The agent's utterance is provided only for context.

3. Scoring Based on Outcome:
   - Failure (Final Score < 5):
     - Identify the utterance that most critically led to the failure.
     - Assign that key utterance an importance of 3.
   - Success (Final Score ≥ 5):
     - Identify the utterance that most critically led to the success.
     - Assign that key utterance an importance of 3.

4. Additional Reward Guidelines:
   - If an utterance has no impact on the final goal achievement, assign it an importance of 0.
   - If an utterance has a moderate impact on the final goal achievement, assign it an importance of 1 or 2 (depending on the degree of impact).
   - If an utterance has a significant impact on the final goal achievement (aside from the key critical utterance already identified), assign it an importance of 3.

   Note:
   - Please only assign a score between 0 and 3.

### Your Agent's Name:
{agent}
### Your Agent's Goal:
{goal}
### Final Goal Achieving Score out of 10:
{score}
### Conversation History:
{conversation}
### Your Agent's Utterance:
{utterance}
### Immediate Response recieved:
{response}
"""

DEFAULT_PROMPT = """
## Reward Attribution Instructions for LLMs

Two agents are in a conversation. For now, you are the judge of the utterance of one of the agents.

1. Input Context:
   - You will receive the utterance or action of an agent in one turn of the conversation and the conversation before it.
   - You will recieve the immediate response it receives after the utterance/action.
   - You will also be provided with the social goal of the agent.

2. Objective:
   - Assign an importance value to the utterance based on its contribution to the final goal achievement, judging from how postive or negative the response is. Note, you should only consider the immediate response it recieves, not the quality of the utterance itself. The agent's utterance and the conversation history are provided only for context.

3. Additional Reward Guidelines:
   - If an utterance has no impact on the final goal achievement, assign it an importance of 0.
   - If an utterance has a moderate impact on the final goal achievement, assign it an importance of 1 or 2 (depending on the degree of impact).
   - If an utterance has a significant impact on the final goal achievement (aside from the key critical utterance already identified), assign it an importance of 3.

   Note:
   - Please only assign a score between 0 and 3.

### Your Agent's Name:
{agent}
### Your Agent's Goal:
{goal}
### Conversation History:
{conversation}
### Your Agent's Utterance:
{utterance}
### Immediate Response recieved:
{response}
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
    prev_score = 0
    attribution_dict = {}
    for i, (speaker, utterance) in enumerate(conversation):
        if speaker == agent:
            prompt = prompt_format.format(
                agent=agent,
                goal=goal,
                score=final_goal_score,
                conversation="\n".join([f"{s}: {u}" for s, u in conversation[:i]]),
                utterance=utterance,
                response=conversation[i + 1][-1] if i + 1 < len(conversation) else ""
            )
            response = openai_call_with_response_model(prompt, llm_name, GoalAchievingScore)
            score = response.score if response else prev_score
            attribution_dict[f"Utterance {i//2} by {speaker}"] = score
            prev_score = score
    return attribution_dict

def calc_attributed_reward(attributed_data: List[Dict[str, float | int]], attribution_instruction_name: str, goal_score: float | int) -> List[Dict[str, Any]]:
    utterance_reward_map = {}
    for k, v in attributed_data.items():
        utterance_reward_map[k] = {"reward": v / 3 * goal_score, "attribution": v}
    return utterance_reward_map

# unified function
def get_attribution_single_conv(conversation, agent, goals, episode, llm_name, attribution_instruction_name):
    prompt_format = DEFAULT_PROMPT
    attribution_scores = assign_attributions_for_conversation(
        prompt_format, agent, goals[agent], episode["scores"][agent], conversation, llm_name=llm_name
    )
    attribution_rewards = calc_attributed_reward(attribution_scores, attribution_instruction_name, episode["scores"][agent])
    return attribution_rewards
