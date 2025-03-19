ARCHIVE_PRELOGUE_INSTRUCTIONS = """
For this task, you will receive the dialogue history between two conversational agents, the social goal of one of the agents, and the final goal achieving score recieved by this agent. Your objective is to assess how much each of the agent's utterance (marked by the agent's name and the utterance number) contributed to the final goal achieving score. You also need to consider the response of the other agent in the conversation to evaluate the impact of the utterance.

For the goal achieving score, if it is <5, the agent fails, so you need to think which utterance is the most important one that leads to the failure of the goal and assign the critical utterance that leads to the failure to be 3. If it is >=5, the agent succeeds, so you need to think which utterance is the most important one that leads to the success of the goal and assign the critical utterance that leads to the success to be 3.

Following the same logic, if you believe an utterance had no impact on the final goal achieving score, please provide a score of 0. If you believe an utterance had a significant impact on the final goal achieving score, please provide a score of 3. If you believe an utterance had a moderate impact on the final goal achieving score, please provide a score of 1 or 2. As a special case, if you believe an utterance is redundant and unnecessary, please provide a score of -1.
"""


DEFAULT_DIRECT_INSTRUCTIONS = """ 
Reward Attribution Instructions for LLMs

1. Input Context:
   - You will receive the dialogue history between two conversational agents.
   - You will also be provided with the social goal of one of the agents and the final goal achievement score.

2. Objective:
   - Assign a reward value to each utterance (identified by the agent's name and utterance number) based on its contribution to the final goal achievement score. Note, you should only consider how critical an utterance is to the final goal achievement, not the quality of the utterance itself.
   - Consider both the individual utterance and the responses from the other agent, as both affect the final outcome.

3. Scoring Based on Outcome:
   - Failure (Final Score < 5):
     - Identify the utterance that most critically led to the failure.
     - Assign that key utterance a reward of 3.
   - Success (Final Score ≥ 5):
     - Identify the utterance that most critically led to the success.
     - Assign that key utterance a reward of 3.

4. Additional Reward Guidelines:
   - If an utterance has no impact on the final goal achievement, assign it a reward of 0.
   - If an utterance has a moderate impact on the final goal achievement, assign it a reward of 1 or 2 (depending on the degree of impact).
   - If an utterance has a significant impact on the final goal achievement (aside from the key critical utterance already identified), assign it a reward of 3.
   - If an utterance is redundant or unnecessary, assign it a reward of -1.

   Note:
   - Please provide a score for each utterance in the conversation.
   - Please only assign a score between -1 and 3.
"""

DIRECT_5_SCALE_INSTRUCTIONS = """
Reward Attribution Instructions for LLMs

1. Input Context:
   - You will receive the dialogue history between two conversational agents.
   - You will also be provided with the social goal of one of the agents and the final goal achievement score.

2. Objective:
   - Assign a reward value to each utterance (identified by the agent's name and utterance number) based on its contribution to the final goal achievement score.Note, you should only consider how critical an utterance is to the final goal achievement, not the quality of the utterance itself.
   - Consider both the individual utterance and the responses from the other agent, as both affect the final outcome.

3. Scoring Based on Outcome:
   - Failure (Final Score < 5):
     - Identify the utterance that most critically led to the failure.
     - Assign that key utterance a reward of 4 to 5, depending on how critical it is.
   - Success (Final Score ≥ 5):
     - Identify the utterance that most critically led to the success.
     - Assign that key utterance a reward of 4 to 5, depending on how critical it is.

4. Additional Reward Guidelines:
   - If an utterance has no impact on the final goal achievement, assign it a reward of 0.
   - If an utterance has a moderate impact on the final goal achievement, assign it a reward of 1 to 3 (depending on the degree of impact).
   - If an utterance has a significant impact on the final goal achievement (aside from the key critical utterance already identified), assign it a reward of 5.
   - If an utterance is redundant or unnecessary, assign it a reward of -1.
"""

ATTRIBUTION_INSTRUCTIONS_DICT = {
   "default": DEFAULT_DIRECT_INSTRUCTIONS,
   "5-scale": DIRECT_5_SCALE_INSTRUCTIONS,
}