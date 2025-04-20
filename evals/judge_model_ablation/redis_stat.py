from sotopia.database.logs import EpisodeLog
from collections import defaultdict

def analyze_episodes_with_positions(tag):
    # Find episodes with the specified tag
    episodes = EpisodeLog.find(EpisodeLog.tag == tag).all()
    print(f"Total episodes found: {len(episodes)}")
    
    # Track rewards by model name
    model_rewards_list = defaultdict(lambda: defaultdict(list))
    model_rewards = defaultdict(lambda: defaultdict(float))
    model_counts = defaultdict(int)
    
    # Track position counts (how many times a model appears as agent1 vs agent2)
    position_counts = defaultdict(lambda: {'agent1': 0, 'agent2': 0})
    
    # Track rewards by position
    position_rewards = defaultdict(lambda: {
        'agent1': defaultdict(float),
        'agent2': defaultdict(float)
    })
    
    # Process each episode
    for episode in episodes:
        try:
            # Skip if no models or rewards
            if not hasattr(episode, 'models') or len(episode.models) < 3:
                continue
            if not hasattr(episode, 'rewards') or len(episode.rewards) < 2:
                continue
            
            # Get model names
            model1_name = episode.models[1]  # agent1's model
            model2_name = episode.models[2]  # agent2's model
            
            # Get rewards (handle both list and direct formats)
            try:
                reward1 = episode.rewards[0][-1]
                reward2 = episode.rewards[1][-1]
            except (IndexError, TypeError):
                continue
            
            # Skip if rewards are not dictionaries
            if not isinstance(reward1, dict) or not isinstance(reward2, dict):
                continue
            
            # Add rewards to model accumulators
            for key, value in reward1.items():
                model_rewards[model1_name][key] += value
                position_rewards[model1_name]['agent1'][key] += value
                model_rewards_list[model1_name][key].append(value)
                
            for key, value in reward2.items():
                model_rewards[model2_name][key] += value
                position_rewards[model2_name]['agent2'][key] += value
                model_rewards_list[model2_name][key].append(value)
            
            # Count model appearances
            model_counts[model1_name] += 1
            model_counts[model2_name] += 1
            
            # Count position appearances
            position_counts[model1_name]['agent1'] += 1
            position_counts[model2_name]['agent2'] += 1
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Calculate overall averages
    print("\n===== OVERALL MODEL PERFORMANCE =====")
    for model, rewards in model_rewards.items():
        print(f"\nModel: {model} (appeared in {model_counts[model]} episodes)")
        print(f"  As agent1: {position_counts[model]['agent1']} times")
        print(f"  As agent2: {position_counts[model]['agent2']} times")
        
        for key, value in rewards.items():
            avg = value / model_counts[model]
            print(f"  {key}: {avg:.4f}")
            # Update the dict with average value
            model_rewards[model][key] = avg
    
    model_pairs = set()
    for model1, rewards1 in model_rewards_list.items():
        for model2, rewards2 in model_rewards_list.items():
            if model1 == model2:
                continue
            if (model1, model2) in model_pairs or (model2, model1) in model_pairs:
                continue
            model_pairs.add((model1, model2))
            # calculate stat sig between model1 and model2 for 'goal' reward
            # use scipy.stats.ttest_ind to calculate the t-statistic and p-value
            # for the two independent samples
            import scipy.stats as stats
            # calculate the t-statistic and p-value
            t_stat, p_val = stats.ttest_ind(rewards1['goal'], rewards2['goal'])
            print(f"Model1: {model1} vs Model2: {model2} t-statistic: {t_stat} p-value: {p_val}")
            
    # Calculate position-specific averages
    print("\n===== PERFORMANCE BY POSITION =====")
    for model in position_rewards:
        print(f"\nModel: {model}")
        
        # Agent1 position
        if position_counts[model]['agent1'] > 0:
            print(f"  As agent1 ({position_counts[model]['agent1']} episodes):")
            for key, value in position_rewards[model]['agent1'].items():
                avg = value / position_counts[model]['agent1']
                print(f"    {key}: {avg:.4f}")
                position_rewards[model]['agent1'][key] = avg
        else:
            print(f"  Never appeared as agent1")
                
        # Agent2 position
        if position_counts[model]['agent2'] > 0:
            print(f"  As agent2 ({position_counts[model]['agent2']} episodes):")
            for key, value in position_rewards[model]['agent2'].items():
                avg = value / position_counts[model]['agent2']
                print(f"    {key}: {avg:.4f}")
                position_rewards[model]['agent2'][key] = avg
        else:
            print(f"  Never appeared as agent2")
    
    # Count model pairs
    print("\n===== MODEL PAIRINGS =====")
    model_pairs = defaultdict(int)
    for episode in episodes:
        try:
            if not hasattr(episode, 'models') or len(episode.models) < 3:
                continue
                
            model1 = episode.models[1]
            model2 = episode.models[2]
            pair_key = f"{model1} vs {model2}"
            model_pairs[pair_key] += 1
        except Exception:
            continue
    
    for pair, count in model_pairs.items():
        print(f"{pair}: {count} episodes")
    
    return {
        'model_rewards': dict(model_rewards),
        'position_counts': dict(position_counts),
        'position_rewards': dict(position_rewards)
    }

# Run the analysis
# results = analyze_episodes_with_positions("sotopia-sft-goal-progress-ppo-3-19-v4")
# results = analyze_episodes_with_positions("baseline-baseline-3-19-v4")
results = analyze_episodes_with_positions("rm_reward_direct_default_no_goal_gpt-4o_without_goal_leak_rej_sampling_num10_vs_sft_qwen25_7b_sft_round_1_bc_data_top_2_0326_v0")

# baseline-baseline-3-19-v5 7b vs 7b
# baseline-sotopia-sft-3-19-v9 first stable version
# baseline-sotopia-sft-3-19-v10 1.5b vs sft
# baseline-gpt-3-21-v1 7b vs gpt
# sotopia-sft-gpt-3-21-v1 sft 4500 vs gpt
# sotopia-sft-gpt-3-21-v2 sft 3000 vs gpt
# sotopia-sft-gpt-3-21-v3 sft 2000 vs gpt
# sotopia-sft-gpt-3-21-v4 sft 1000 vs gpt
# sotopia-sft-gpt-3-21-v5 sft 500 vs gpt
# sotopia-sft-sotopia-sft-3-21-v1 sft 1000 vs sft 1000

# sotopia-sft-sotopia-sft-3-24-v1 sft 1000 vs sft 1000
# baseline-sotopia-sft-3-25-v2 baseline vs sft 1000
# qwen-sft-qwen-sft-3-26-v2 sft 1500 vs sft 1500