## all the same
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name all_the_same \
--attribution_instruction_name default \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_all_the_same.jsonl

python post_process_annotation.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--input_file sotopia_pi_bc_episodes_annotated_all_the_same.jsonl \
--reward_output_file sotopia_pi_bc_episodes_reward_all_the_same.json 

## key utterance
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name key_utterance \
--attribution_instruction_name key_utterance \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_key_utterance_no_goal_gpt-4o.jsonl \
--max_concurrency 64

python post_process_annotation.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--input_file sotopia_pi_bc_episodes_annotated_key_utterance_no_goal_gpt-4o.jsonl \
--reward_output_file sotopia_pi_bc_episodes_reward_key_utterance_no_goal_gpt-4o.json 

# utterance quality
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name utterance_quality \
--attribution_instruction_name default \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_utterance_quality_no_goal_gpt-4o.jsonl \
--max_concurrency 64

python post_process_annotation.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--input_file sotopia_pi_bc_episodes_annotated_utterance_quality_no_goal_gpt-4o.jsonl \
--reward_output_file sotopia_pi_bc_episodes_reward_utterance_quality_no_goal_gpt-4o.json 

## direct default 
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name direct \
--attribution_instruction_name default \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_direct_default_no_goal_gpt-4o.jsonl \
--max_concurrency 20

python post_process_annotation.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--input_file sotopia_pi_bc_episodes_annotated_direct_default_no_goal_gpt-4o.jsonl \
--reward_output_file sotopia_pi_bc_episodes_reward_direct_default_no_goal_gpt-4o.json 

## direct normalized 
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name direct_normalized \
--attribution_instruction_name default \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_direct_default_normalized_no_goal_gpt-4o.jsonl \
--max_concurrency 20

python post_process_annotation.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--input_file sotopia_pi_bc_episodes_annotated_direct_default_normalized_no_goal_gpt-4o.jsonl \
--reward_output_file sotopia_pi_bc_episodes_reward_direct_default_normalized_no_goal_gpt-4o.json 

## direct 10-scale 
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name direct \
--attribution_instruction_name 10-scale \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_direct_10-scale_no_goal_gpt-4o.jsonl \
--max_concurrency 20

python post_process_annotation.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--input_file sotopia_pi_bc_episodes_annotated_direct_10-scale_no_goal_gpt-4o.jsonl \
--reward_output_file sotopia_pi_bc_episodes_reward_direct_10-scale_no_goal_gpt-4o.json 
