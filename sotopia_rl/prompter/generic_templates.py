THREE_SCALE_SCORING_GUIDELINES = """
   - If an utterance has no impact on the final achievement on the dimension, assign it an importance of 0.
   - If an utterance has a minor impact on the final achievement on the dimension, assign it an importance of 1 to 2 (depending on the degree of impact).
   - If an utterance has a major impact on the final achievement on the dimension, assign it an importance of 2 to 3 (depending on the degree of impact).
   - If an utterance has a significant impact on the final achievement on the dimension, assign it an importance of 3.

   Note:
   - Please provide a score for each utterance of the chosen agent in the conversation. Do not provide scores for the other agent's utterances.
   - Please only assign a score between 0 and 3.
"""

FIVE_SCALE_SCORING_GUIDELINES = """
   - If an utterance has no impact on the final achievement on the dimension, assign it an importance of 0.
   - If an utterance has a minor impact on the final achievement on the dimension, assign it an importance of 1 to 3 (depending on the degree of impact).
   - If an utterance has a major impact on the final achievement on the dimension, assign it an importance of 3 to 5 (depending on the degree of impact).
   - If an utterance has a significant impact on the final achievement on the dimension, assign it an importance of 5.

   Note:
   - Please provide a score for each utterance of the chosen agent in the conversation. Do not provide scores for the other agent's utterances.
   - Please only assign a score between 0 and 5.
"""

TEN_SCALE_SCORING_GUIDELINES = """
3. Specific Scoring Guidelines:
   - If an utterance has no impact on the final achievement on the dimension, assign it an importance of 0.
   - If an utterance has a minor impact on the final achievement on the dimension, assign it an importance of 1 to 5 (depending on the degree of impact).
   - If an utterance has a major impact on the final achievement on the dimension, assign it an importance of 6 to 10 (depending on the degree of impact).
   - If an utterance has a significant impact on the final achievement on the dimension, assign it an importance of 10.

   Note:
   - Please provide a score for each utterance of the chosen agent in the conversation. Do not provide scores for the other agent's utterances.
   - Please only assign a score between 0 and 10.
"""

BELIEVABILITY_DESCRIPTION = """
Believability refers to the extent to which the agents in the conversation are perceived as realistic and relatable. This includes evaluating if the agent interacts with others in a natural and realistic manner and analyzing whether the actions of the agent align with their character traits. A higher score indicates that the utterance contributes significantly to the believability of the agent's character and the overall conversation.
"""

RELATIONSHIP_DESCRIPTION = """
Relationship refers to the analysis of the pre- and post-interaction relationships between agents. This includes evaluating whether the interactions enhance or harm social ties or status. A higher score indicates that the interaction significantly improves the relationship, while a lower score indicates harm to the relationship or social status.
"""

KNOWLEDGE_DESCRIPTION = """
Knowledge refers to the assessment of information gained through the interaction. This includes evaluating whether the information is new, important, and relevant. A higher score indicates that the interaction contributes significantly to the acquisition of valuable knowledge.
"""

SECRET_DESCRIPTION = """
Secret refers to the analysis of what secret or intention the participant wants to keep and whether it is successfully kept. A higher (less negative) score indicates that the secret or intention is well-protected, while a lower (more negative) score indicates a failure to keep the secret or intention.
"""

SOCIAL_RULES_DESCRIPTION = """
Social Rules refers to the evaluation of whether the agent violates any moral rules or laws during the interaction. A higher (less negative) score indicates adherence to social norms and rules, while a lower (more negative) score indicates violations of these norms or laws.
"""

FINANCIAL_AND_MATERIAL_BENEFITS_DESCRIPTION = """
Financial and Material Benefits refers to the evaluation of whether the interactions yield financial or material gain or loss. A higher score indicates significant financial or material gain, while a lower score indicates a loss.
"""

GOAL_BAREBONE_DESCRIPTION = """
Goal refers to the reiteration of the agent's social goals and the analysis of their achievement. A higher score indicates significant progress or achievement of the stated goals, while a lower score indicates minimal or no progress.
"""

GOAL_DESCRIPTION = """
Goal refers to the reiteration of the agent's social goals and the analysis of their achievement. A higher score indicates significant progress or achievement of the stated goals, while a lower score indicates minimal or no progress.

DOMAIN SPECIFIC SCORING GUIDELINES:
! Note: The following scoring guidelines are specific to the domain of goal and should be used in conjunction with the general scoring scale. The domain specific rules ultimately override the general scoring scale. Here are the specific rules:
   - The highest score should be assigned to the utterance that is most relevant to the goal. In general, avoid assigning the highest score to more than one utterance unless they are equally critical.
   - A lower score should be assigned to the utterances that are not relevant to the goal or do not contribute to its achievement. The lowest score should be assigned to the utterances that do not make any progress towards the goal juding by the goal description and the conversation history.
   - A lower score should be assigned to the utterances that are not effective in achieving the goal, juding by the response of the other agent. Effective utterances are those that lead to a positive response from the other agent, while ineffective utterances are those that lead to a negative or neutral response.
   - Note that you should only consider the contribution to the goal achievement. For each utterance, assess whether the goal is achieved. If a goal is already achieved, the utterance should not be assigned a score higher than 1.
"""

CONVERSATION_BEHAVIOR_DESCRIPTION = """   
Conversation Behavior refers to the evaluation of the agent's conversation behavior, including the avoidance of repetitive utterances, proper ending of the conversation, and overall performance as a social agent. A higher score indicates effective conversation behavior, while a lower score indicates poor performance in these areas.

DOMAIN SPECIFIC SCORING GUIDELINES:
! Note: The following scoring guidelines are specific to the domain of conversation behavior and should be used in conjunction with the general scoring scale. The domain specific rules ultimately override the general scoring scale. Here are the specific rules:
   - If an utterance is repetitive of the previous utterance, assign it an importance of 0. Note that repetitive utterances are not necessarily identical, but they should be similar in meaning or content.
   - If an utterance is the last utterance in the conversation and it is a proper ending of the conversation, assign it an importance of the highest score. If not a proper ending, assign it an importance of 0.
   - If an utterance is a proper ending of the conversation but not the last utterance, assign it an importance of 0.
   - Otherwise, assign an importance of 1.
   - Do not assign a score higher than 1 for any utterance in this domain except for the last utterance in the conversation.
"""

SCALE_GUIDELINE_DICT = {
      "default": THREE_SCALE_SCORING_GUIDELINES,
      "5_scale": FIVE_SCALE_SCORING_GUIDELINES,
      "10_scale": TEN_SCALE_SCORING_GUIDELINES,
   }

DIMENSION_DESCRIPTION_DICT = {
    "believability": BELIEVABILITY_DESCRIPTION,
    "relationship": RELATIONSHIP_DESCRIPTION,
    "knowledge": KNOWLEDGE_DESCRIPTION,
    "secret": SECRET_DESCRIPTION,
    "social_rules": SOCIAL_RULES_DESCRIPTION,
    "financial_and_material_benefits": FINANCIAL_AND_MATERIAL_BENEFITS_DESCRIPTION,
    "goal": GOAL_DESCRIPTION,
    "conversation_behavior": CONVERSATION_BEHAVIOR_DESCRIPTION,
    "goal_barebone": GOAL_BAREBONE_DESCRIPTION,
}

DIRECT_ATTRIBUTION_TEMPLATE = """
Reward Attribution Instructions for LLMs

Your task is to evaluate the importance of each utterance in a conversation between two agents on a certain dimension of evaluation. You will be provided with the dialogue history, the social goal of one of the agents, and the certain dimension to be evaluated. For example, the dimension can be common social values such as adherence to social rules, relationship maintainance or improvement, or secret keeping. The dimension can also be objectives such as goal achieving, financial and material gains, or discovery of new knowledge. Moreover, the dimension can also be about performance of a language model as an social agent, such as the agent's believability as a human, avoidance of repetitions, and properly ending the conversation. However, you will be only provided with one dimension to be evaluated and you should only focus on that dimension.

1. Input Context:
   - You will receive the dialogue history between two conversational agents each with their own social goal.
   - You will be provided with the social goal of one of the agents.
   - You will be provided with the dimension to be evaluated and the discription of the dimension.

2. Objective:
   - Assign am importance value to each utterance (identified by the agent's name and utterance number) based on its contribution to the achievement on the provided dimension. Note, you should only consider how critical an utterance is to the achievment of the dimension, not the quality of the utterance itself.
   - Consider both the individual utterance and the responses from the other agent, as both affect the final outcome.

3. General Scoring Scale:
{scoring_guidelines}

4. Chosen Agent for Evaluation:
{agent}

5. Agent's Goal:
{goal}

6. Agent's Background:
{agent_background}

7. Conversation History:
{conversation}

8. Dimension to be Evaluated:
{dimension}

9. Dimension Description:
{dimension_description}

10. Formatting Instructions:
{formatting_instructions}
"""