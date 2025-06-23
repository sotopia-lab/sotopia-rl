import json
from collections import defaultdict
from statistics import mean as average

TAG = "grpo_rm_goal_0503_w_relationship_knowledge_0507_5_10_step_2200_vs_sft_0510_epoch500_step_200-0512"
MODEL_NAME = "together_ai/deepseek-ai/DeepSeek-V3"
FILE_NAME = f".cache/{TAG}_eval_{MODEL_NAME.replace('/', '_')}_results.json"

with open(FILE_NAME, "r") as f:
    results = json.load(f)
    print(len(results))

model_goal_score_dict = defaultdict(list)
model_overall_score_dict = defaultdict(list)
judge_model = None
for result in results.values():
    if result is None:
        continue
    judge_model = result["models"][0]
    agent1_dict = defaultdict(dict)
    agent2_dict = defaultdict(dict)
    for rating in result["rating"]:
        agent_name = rating[0]
        dim_name, score = rating[1][0][0], rating[1][0][1]
        if agent_name == 'agent_1':
            agent1_dict[dim_name] = score
        elif agent_name == 'agent_2':
            agent2_dict[dim_name] = score
        else:
            raise ValueError(f"Unknown agent name: {agent_name}")
    print("Agent 1 Ratings:")
    print(agent1_dict)
    print("Agent 2 Ratings:")
    print(agent2_dict)
    model_1 = result["models"][1]
    model_2 = result["models"][2]
    overall_score_1 = average(agent1_dict.values())
    overall_score_2 = average(agent2_dict.values())
    
    if agent1_dict and agent2_dict:
        model_goal_score_dict[model_1].append(agent1_dict["goal"])
        model_goal_score_dict[model_2].append(agent2_dict["goal"])
        model_overall_score_dict[model_1].append(overall_score_1)
        model_overall_score_dict[model_2].append(overall_score_2)

print(f"Judge Model: {judge_model}")
for model_name, scores in model_goal_score_dict.items():
    print(f"Model: {model_name}")
    print(f"Average Goal: {sum(scores)/len(scores)}")
for model_name, scores in model_overall_score_dict.items():
    print(f"Model: {model_name}")
    print(f"Average Overall: {sum(scores)/len(scores)}")