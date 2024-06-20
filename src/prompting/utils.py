from typing import List, Tuple
import ast
import re

class Agent:
    def __init__(self, agent_profile):
        self._id = agent_profile["agent_id"]
        
        self.agent_profile = agent_profile
        self.agent_id = agent_profile["agent_id"]
        self.name = self.get_name(agent_profile)
        self.background = self.get_background(agent_profile)
        self.secret = agent_profile["secret"]
        self.personality = agent_profile["personality_and_values"]
        self.goal = ""
        
    def get_name(self, agent_profile):
        return agent_profile["first_name"] + " " + agent_profile["last_name"]
    
    def get_background(self, agent_profile):
        name = self.name
        return f"{name} is a {agent_profile['age']}-year-old {agent_profile['gender'].lower()} {agent_profile['occupation']}. {agent_profile['public_info']}"
    
class Environment:
    
    def __init__(self, env_profile):
        self._id = env_profile["env_id"]
        
        self.environment_profile = env_profile
        self.codename = env_profile["codename"]
        self.scenario = env_profile["scenario"]
        self.agent_goals = env_profile["agent_goals"]
        self.relationship = env_profile["relationship"]
        
        
def get_context_prompt(machine_agent, human_agent, environment):
    return f"Here is the context of this interaction:\n Scenario: {environment.scenario}\nParticipants: {human_agent.name} and {machine_agent.name}\n{human_agent.name}'s background: {human_agent.background} Personality and values description: {human_agent.personality} \n{machine_agent.name}'s background: {machine_agent.background} Personality and values description: {machine_agent.personality} {machine_agent.name}'s secrets: {machine_agent.secret}\n{human_agent.name}'s goal: Unknown\n{machine_agent.name}'s goal: {environment.agent_goals[1]}\nConversation Starts:"
        
def dialogue_history_prompt(message, history, user_agent, bot_agent):
    dialogue_history = ""
    for idx, turn in enumerate(history):
        user_message, bot_message = turn
        # TODOTODO (haofeiyu): we first assume that human talks first
        user_turn_idx = idx * 2
        bot_turn_idx = idx * 2 + 1
        dialogue_history = f"""{dialogue_history}\n\nTurn #{user_turn_idx} {user_message}"\n\nTurn #{bot_turn_idx} {bot_message}"""
    curr_turn_idx = len(history) * 2
    dialogue_history = f"""{dialogue_history}\n\nTurn #{curr_turn_idx} {message}\n"""
    return dialogue_history, curr_turn_idx + 1

def format_docstring(docstring: str) -> str:
    """Format a docstring for use in a prompt template."""
    return re.sub("\n +", "\n", docstring).strip()