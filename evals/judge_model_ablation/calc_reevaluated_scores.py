import json
from collections import defaultdict

FILE_NAME = "/root/sotopia-rl/Untitled/.cache/rm_reward_direct_default_no_goal_gpt-4o_without_goal_leak_rej_sampling_num10_vs_sft_qwen25_7b_sft_round_1_bc_data_top_2_0326_v0_eval_claude_claude-3-7-sonnet-20250219_results.json"

with open(FILE_NAME, "r") as f:
    results = json.load(f)
    print(len(results))

model_goal_score_dict = defaultdict(list)
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
    if agent1_dict and agent2_dict:
        model_goal_score_dict[model_1].append(agent1_dict["goal"])
        model_goal_score_dict[model_2].append(agent2_dict["goal"])

print(f"Judge Model: {judge_model}")
for model_name, scores in model_goal_score_dict.items():
    print(f"Model: {model_name}")
    print(f"Average Score: {sum(scores)/len(scores)}")