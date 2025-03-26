## direct default 
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name direct \
--attribution_instruction_name default \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_direct_default_no_goal_gpt-4o.jsonl \
--max_concurrency 50

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
--output_file sotopia_pi_bc_episodes_annotated_direct_normalized_no_goal_gpt-4o.jsonl \
--max_concurrency 50

python post_process_annotation.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--input_file sotopia_pi_bc_episodes_annotated_direct_normalized_no_goal_gpt-4o.jsonl \
--reward_output_file sotopia_pi_bc_episodes_reward_direct_normalized_no_goal_gpt-4o.json 

# only response
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name only_response \
--attribution_instruction_name default \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_only_response_no_goal_gpt-4o.jsonl \
--max_concurrency 20

python post_process_annotation.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--input_file sotopia_pi_bc_episodes_annotated_only_response_no_goal_gpt-4o.jsonl \
--reward_output_file sotopia_pi_bc_episodes_reward_only_response_no_goal_gpt-4o.json

# utterance quality
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name utterance_quality \
--attribution_instruction_name default \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_utterance_quality_no_goal_gpt-4o.jsonl \
--max_concurrency 20

python post_process_annotation.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--input_file sotopia_pi_bc_episodes_annotated_utterance_quality_no_goal_gpt-4o.jsonl \
--reward_output_file sotopia_pi_bc_episodes_reward_utterance_quality_no_goal_gpt-4o.json 

# utterance quality normalized
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name utterance_quality_normalized \
--attribution_instruction_name default \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_utterance_quality_normalized_no_goal_gpt-4o.jsonl \
--max_concurrency 20

python post_process_annotation.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--input_file sotopia_pi_bc_episodes_annotated_utterance_quality_normalized_no_goal_gpt-4o.jsonl \
--reward_output_file sotopia_pi_bc_episodes_reward_utterance_quality_normalized_no_goal_gpt-4o.json 