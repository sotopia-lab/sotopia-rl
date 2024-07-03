import os
import json, jsonlines
from collections import defaultdict
from functools import cache
from typing import Literal, Tuple, List, Dict
from copy import copy

from tqdm import tqdm

from src.prompting.sotopia_utils import Environment, Agent, get_context_prompt, dialogue_history_prompt
from src.prompting.sotopia_generate import generate_action

ENVIRONMENT_PROFILES = "../../data/profiles/environmentprofiles_v1.jsonl"
AGENT_PROFILES = "../../data/profiles/agentprofiles_v1.jsonl"
RELATIONSHIP_PROFILES = "../../data/profiles/relationshipprofiles_v1.jsonl"

Action = Literal['none', 'action', 'non-verbal communication', 'speak', 'leave']
ACTION_TYPES: list[Action] = ['none', 'action', 'non-verbal communication', 'speak', 'leave']
TEMPERATURE = 0.7

@cache
def get_sotopia_profiles(env_file: str=ENVIRONMENT_PROFILES, agent_file: str=AGENT_PROFILES, relationship_file: str=RELATIONSHIP_PROFILES) -> Tuple[List[Tuple[str, str]], Dict[str, Environment], Dict[str, Agent], Dict[str, Dict[str, List[str]]]]:
    with open(env_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    code_names_count: Dict[str, int] = defaultdict(int)
    environments = []
    environment_dict = {}
    for profile in sorted(data, key=lambda x: x['codename']):
        env_obj = Environment(profile)
        if profile['codename'] in code_names_count:
            environments.append((
                "{}_{:05d}".format(profile['codename'], 
                                    code_names_count[profile['codename']]
                                    ), 
                env_obj._id
                ))
        else:
            environments.append((profile['codename'], env_obj._id))
        environment_dict[env_obj._id] = env_obj
        code_names_count[profile['codename']] += 1
    
    with open(agent_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    agent_dict = {}
    for profile in data:
        agent_obj = Agent(profile)
        agent_dict[agent_obj._id] = agent_obj
        
    with open(relationship_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    relationship_dict: Dict[str, Dict[str, List[str]]] = defaultdict(lambda : defaultdict(list))
    for profile in data:
        relationship_dict[profile['relationship']][profile['agent1_id']].append(profile['agent2_id'])
        relationship_dict[profile['relationship']][profile['agent2_id']].append(profile['agent1_id'])
    
    return environments, environment_dict, agent_dict, relationship_dict

def run_chat(
    message: str,
    history: List[List[str]],
    bot_agent:Agent,
    user_agent:Agent,
    environment:Environment,
    model_selection:str,
) -> Tuple[str, str]:
    context = get_context_prompt(bot_agent, user_agent, environment)
    dialogue_history, next_turn_idx = dialogue_history_prompt(message, history, user_agent, bot_agent)
    prompt_history = f"{context}{dialogue_history}"
    prompt, agent_action = generate_action(model_selection, prompt_history, next_turn_idx, ACTION_TYPES, bot_agent.name, TEMPERATURE)
    return prompt, agent_action.to_natural_language()

def load_sotopia_pi_data(
    data_path:str,
    environment_dict: Dict[str, Environment],
    agent_dict: Dict[str, Agent]
) -> Tuple[List[Environment], List[Agent], List[Agent], List[str]]:
    with jsonlines.open(data_path, "r") as f:
        dataset = [line for line in f]
    
    envs = []
    start_agents = []
    end_agents = []
    social_interactions = []
    for data in tqdm(dataset):
        if not 'gpt' in data['experiment_model_name_pairs'][1] or not 'gpt' in data['experiment_model_name_pairs'][2]:
            continue
        envs.append(environment_dict[data['environment_id']])
        start_agents.append(agent_dict[data['agent_ids'][0]])
        end_agents.append(agent_dict[data['agent_ids'][1]])
        social_interactions.append(data['social_interactions'])
    return envs, start_agents, end_agents, social_interactions

def generate_prompt_response_pairs(output_dir: str, model_selections: List[str], envs: List[Environment], 
                                   start_agents: List[Agent], end_agents: List[Agent], 
                                   social_interactions: List[str], num_episodes: int=2) -> None:
    if not os.path.exists(output_dir):
        with open(output_dir, "w") as f:
            f.write("[]")
    
    with open(output_dir, "r") as f:
        result_pairs = json.load(f)
    
    all_ids = set()
    for result in result_pairs:
        all_ids.add(f"{result['environment_id']}_{result['start_agent_id']}_{result['end_agent_id']}")
    
    count = 0
    for env, start_agent, end_agent, social_interaction in tqdm(zip(envs, start_agents, end_agents, social_interactions), total=len(envs)):
        if f"{env._id}_{start_agent._id}_{end_agent._id}" in all_ids:
            count += 1
            continue
        
        full_history = social_interaction.split("\n\n")
        curr_history = []
        for i in range(0, len(full_history), 2):
            if i > 0:
                curr_history.append([f"Utterance {i//2 - 1} by " + full_history[i-2], 
                                     f"Utterance {i//2 - 1} by " + full_history[i-1]])
            message = f"Utterance {i//2} by " + full_history[i]
            try:
                prompt, response0 = run_chat(message, curr_history, end_agent, start_agent, env, model_selections[0])
                prompt, response1 = run_chat(message, curr_history, start_agent, end_agent, env, model_selections[1])
            except:
                continue
                
            result_pairs.append({"prompt": prompt, 
                                 "message": message, 
                                 "history": copy(curr_history), 
                                 model_selections[0]: response0, 
                                 model_selections[1]: response1, 
                                 "environment_id": env._id, 
                                 "start_agent_id": start_agent._id, 
                                 "end_agent_id": end_agent._id, 
                                 "scenario": env.scenario, 
                                 "start_agent_name": start_agent.name, 
                                 "end_agent_name": end_agent.name, 
                                 "start_agent_goal": env.agent_goals[0], 
                                 "end_agent_goal": env.agent_goals[1]})
        
        count += 1
        all_ids.add(f"{env._id}_{start_agent._id}_{end_agent._id}")
        
        with open(output_dir, "w") as f:
            f.write(json.dumps(result_pairs))
        if count >= num_episodes:
            break

if __name__ == "__main__":
    environments, environment_dict, agent_dict, relationship_dict = get_sotopia_profiles()
    envs, start_agents, end_agents, social_interactions = load_sotopia_pi_data(
        "../../data/sotopia_pi_episodes.jsonl",
        environment_dict,
        agent_dict
    )
    print("Loaded data with {} episodes".format(len(envs)))
    generate_prompt_response_pairs("../../data/gpt35_gpt4_prompt_response_pairs.json", ["gpt-3.5-turbo", "gpt-4o"], envs, start_agents, end_agents, social_interactions, 1000000)