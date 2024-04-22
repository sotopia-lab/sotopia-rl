import json
import os
from openai import OpenAI
from collections import OrderedDict
from pprint import pprint
from tqdm import tqdm

# Set environment variables for OpenAI API
with open("openai_api.key", 'r') as f:
    os.environ["OPENAI_API_KEY"] = f.readline().strip()

client = OpenAI()

def openai_call(prompt):
    """Make a call to OpenAI API with a specific prompt."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content

def parse_conversation(episode):
    """Extract and parse conversation and goals from the episode."""
    conversation = episode['social_interactions'].split("\n\n")
    goals = episode['social_goals']
    agent1, agent2 = list(goals.keys())
    parsed_conversation = []
    for i, utterance in enumerate(conversation):
        if utterance.startswith(agent1):
            speaker = agent1
        elif utterance.startswith(agent2):
            speaker = agent2
        else:
            continue  # Skip any unparsable utterances
        parsed_conversation.append((speaker, utterance))
    return parsed_conversation, goals

PRELOGUE_INSTRUCTIONS = """
For this task, you will receive the dialogue history between two conversational agents, the social goal of one of the agents, and the final reward score recieved by this agent. Your objective is to assess how much each of the agent's utterance (marked by the agent's name and the utterance number) contributed to the final reward score.
"""

def get_epilogue_instructions(agent):
    return f"""
Please provide a score between 0 and 10 for each of the utterances made by {agent}. If you believe an utterance had no impact on the final reward score, please provide a score of 0. If you believe an utterance had a significant impact on the final reward score, please provide a score of 10. If you believe an utterance had a moderate impact on the final reward score, please provide a score of 5. You can provide any score between 0 and 10 based on your judgment.

Please format your response as JSON with the following structure:
{{
    "Utterance 1 by {agent}": 5,
    "Utterance 2 by {agent}": 7,
    ... 
}}
The utterance numbers should correspond to their order in the conversation. Each score should reflect how much the utterance contributed to achieving the agent's goals.
"""

def generate_single_reward_prompt(conversation, goal, score, agent):
    """Generate a single prompt for GPT based on the entire conversation, agent's goals, and final reward."""
    prompt = f"{PRELOGUE_INSTRUCTIONS}\n\n"
    prompt += f"Agent Goal: {goal}\n\n"
    prompt += f"Final Reward Received: {score}\n\n" 
    prompt += "Conversation:\n"
    key_utterance_dict = OrderedDict()
    for i, (speaker, utterance) in enumerate(conversation):
        prompt += f"Utterance {i//2 + 1} by {speaker}: {utterance}\n"
        key_utterance_dict[f"Utterance {i//2 + 1} by {speaker}"] = [utterance, -1]
    prompt +="\n" + get_epilogue_instructions(agent)
    return prompt, key_utterance_dict

def assign_rewards_for_conversation(prompt):
    """Assign rewards to the entire conversation based on a GPT response."""
    response = openai_call(prompt)
    return response

# Main code execution with modified approach
with open("example_episodes_with_scores.jsonl", 'r') as f:
    data = [json.loads(line, object_hook=OrderedDict) for line in f]

with open("openai_log_reward_attribution.jsonl", 'w') as f:
    f.write("")

print(len(data))
results = []
for episode in tqdm(data[1:]):
    conversation, goals = parse_conversation(episode)
    agents = list(goals.keys())
    for agent in agents:
        prompt, key_prompt_dict  = generate_single_reward_prompt(conversation, goals[agent], episode['scores'][agent], agent)
        import pdb; pdb.set_trace()
        reward_scores = json.loads(assign_rewards_for_conversation(prompt))
        for key in key_prompt_dict:
            if agent in key and key in reward_scores:
                key_prompt_dict[key][1] = reward_scores[key]
        results.append({"episode_id": episode["episode_id"], "scenario": episode["scenario"], "agent": agent, "goal": goals[agent], "rewarded_utterances": key_prompt_dict, "is_first_speaker": agent == agents[0]})
        
        with open("openai_log_reward_attribution.jsonl", 'a') as f:
            f.write(json.dumps(results[-1]) + "\n")