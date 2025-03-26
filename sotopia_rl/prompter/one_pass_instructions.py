DEFAULT_DIRECT_INSTRUCTIONS = """
Reward Attribution Instructions for LLMs

1. Input Context:
   - You will receive the dialogue history between two conversational agents.
   - You will also be provided with the social goal of one of the agents.

2. Objective:
   - Assign am importance value to each utterance (identified by the agent's name and utterance number) based on its contribution to the final goal achievement. Note, you should only consider how critical an utterance is to the final goal achievement, not the quality of the utterance itself.
   - Consider both the individual utterance and the responses from the other agent, as both affect the final outcome.

3. Additional Reward Guidelines:
   - If an utterance has no impact on the final goal achievement, assign it an importance of 0.
   - If an utterance has a moderate impact on the final goal achievement, assign it an importance of 1 or 2 (depending on the degree of impact).
   - If an utterance has a significant impact on the final goal achievement (aside from the key critical utterance already identified), assign it an importance of 3.

   Note:
   - Please provide a score for each utterance of the chosen agent in the conversation. Do not provide scores for the other agent's utterances.
   - Please only assign a score between 0 and 3.
"""

DIRECT_10_SCALE_INSTRUCTIONS = """
Reward Attribution Instructions for LLMs

1. Input Context:
   - You will receive the dialogue history between two conversational agents.
   - You will also be provided with the social goal of one of the agents.

2. Objective:
   - Assign am importance value to each utterance (identified by the agent's name and utterance number) based on its contribution to the final goal achievement. Note, you should only consider how critical an utterance is to the final goal achievement, not the quality of the utterance itself.
   - Consider both the individual utterance and the responses from the other agent, as both affect the final outcome.

3. Additional Reward Guidelines:
   - If an utterance has no impact on the final goal achievement, assign it an importance of 0.
   - If an utterance has a minor impact on the final goal achievement, assign it an importance of 1 to 5 (depending on the degree of impact).
   - If an utterance has a major impact on the final goal achievement, assign it an importance of 6 to 10 (depending on the degree of impact).
   - If an utterance has a significant impact on the final goal achievement (aside from the key critical utterances already identified), assign it an importance of 10.

   Note:
   - Please provide a score for each utterance of the chosen agent in the conversation. Do not provide scores for the other agent's utterances.
   - Please only assign a score between 0 and 10.
"""

KEY_UTTERANCE_INSTRUCTIONS = """
Reward Attribution Instructions for LLMs

1. Input Context:
   - You will receive the dialogue history between two conversational agents.
   - You will also be provided with the social goal of one of the agents.

2. Objective:
   - Identify the most critical utterance that has the highest impact on the final goal achievement, whether it is bad or good impact. Note, you should only consider how critical an utterance is to the final goal achievement, not the quality of the utterance itself.
   - Consider both the individual utterance and the responses from the other agent, as both affect the final outcome.

3. Additional Guidelines:
   - The conversation history will be given in a unique key of ""Utterance {utterance number} by {agent name}"" for each utterance. Please only return the key of the most critical utterance.

   Note:
   - You will also be given a formatting instruction for the output. Please follow the instruction to ensure the evaluation process runs smoothly.
"""

ATTRIBUTION_INSTRUCTIONS_DICT = {
   "default": DEFAULT_DIRECT_INSTRUCTIONS,
   "10-scale": DIRECT_10_SCALE_INSTRUCTIONS,
   "key_utterance": KEY_UTTERANCE_INSTRUCTIONS,
}