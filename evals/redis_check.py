from sotopia.database.logs import EpisodeLog
from sotopia.database.persistent_profile import EnvironmentProfile
import rich
from collections import Counter

# tag = "grpo_rm_reward_goal_w_conversation_behavior_4_23_step_1400_vs_sft_qwen25_7b_sft_round_1_bc_data_top_2_step_1500-0422"
tag = "sft_round_1_bc_data_top_2_with_aligned_format_instruction_prompt_weight_decay_0_0509_step_700_vs_sft_round_1_bc_data_top_2_with_aligned_format_instruction_prompt_weight_decay_0_0509_step_700-0510_v1"
all_episodes = EpisodeLog.find(EpisodeLog.tag == tag).all()
filtered_episodes = []
for episode in all_episodes:
    if episode.models[0] != "gpt-4o":
        continue
    filtered_episodes.append(episode)
print(f"Filtered episodes: {len(filtered_episodes)}")
all_episodes = filtered_episodes
print(Counter(episode.models[0] for episode in all_episodes))
print(Counter(episode.models[1] for episode in all_episodes))
print(Counter(episode.models[2] for episode in all_episodes))
print(f"Total episodes found: {len(all_episodes)}")

convo_lens = []
for episode in all_episodes:
    if not hasattr(episode, "messages"):
        continue
    convo_lens.append(len(episode.messages))
print(f"Distribution of conversation lengths: {Counter(convo_lens)}")


first_episode = all_episodes[1]
rich.print(first_episode.environment)
rich.print(first_episode.models)
rich.print(first_episode.render_for_humans())


breakpoint()