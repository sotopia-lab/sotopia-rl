from sotopia_generate import generate_action
import json, jsonlines
from utils import Environment, Agent, get_context_prompt, dialogue_history_prompt
from collections import defaultdict
from functools import cache
from typing import Dict

ENVIRONMENT_PROFILES = "../../data/environment_profiles.jsonl"
AGENT_PROFILES = "../../data/agent_profiles.jsonl"
RELATIONSHIP_PROFILES = "../../data/relationship_profiles.jsonl"

@cache
def get_sotopia_profiles(env_file=ENVIRONMENT_PROFILES, agent_file=AGENT_PROFILES, relationship_file=RELATIONSHIP_PROFILES):
    with open(env_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    code_names_count = defaultdict(int)
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
    
    relationship_dict = defaultdict(lambda : defaultdict(list))
    for profile in data:
        relationship_dict[profile['relationship']][profile['agent1_id']].append(profile['agent2_id'])
        relationship_dict[profile['relationship']][profile['agent2_id']].append(profile['agent1_id'])
    
    return environments, environment_dict, agent_dict, relationship_dict



def run_chat(
    message,
    history,
    model_selection:str
):
    context = get_context_prompt(bot_agent, user_agent, environment)
    dialogue_history, next_turn_idx = dialogue_history_prompt(message, history, user_agent, bot_agent)
    prompt_history = f"{context}{dialogue_history}"
    agent_action = generate_action(model_selection, prompt_history, next_turn_idx, ACTION_TYPES, bot_agent.name, TEMPERATURE)
    return agent_action.to_natural_language()


def load_sotopia_pi_data(
    data_path:str,
    environment_dict,
    agent_dict,
):
    with jsonlines.open(data_path, "r") as f:
        dataset = [line for line in f]
    
    envs = []
    start_agents = []
    end_agents = []
    for data in dataset:
        envs.append(environment_dict[data['environment_id']])
        start_agents.append(agent_dict[data['agent_id'][0]])
        end_agents.append(agent_dict[data['agent_id'][1]])
    return envs, start_agents, end_agents


if __name__ == "__main__":
    environments, environment_dict, agent_dict, relationship_dict = get_sotopia_profiles()
    envs, start_agents, end_agetns = load_sotopia_pi_data(
        "../../data/sotopia_pi_episodes.jsonl",
        environment_dict,
        agent_dict
    )
    import pdb; pdb.set_trace()
    for d in envs:
        message = d["message"]
        history = d["history"]
        model_selection = d["model_selection"]
        print(run_chat(message, history, model_selection))