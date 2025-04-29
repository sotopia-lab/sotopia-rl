# direct attribution gpt-4o on goal
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name direct_generic \
--attribution_instruction_name default-goal \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_direct_default-goal_gpt-4o.jsonl \
--max_concurrency 32

# other dimensions:  believability, relationship, knowledge, secret, social_rules, financial_and_material_benefits

# direct attribution gpt-4o on believability
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name direct_generic \
--attribution_instruction_name default-believability \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_direct_default-believability_gpt-4o.jsonl \
--max_concurrency 32

# direct attribution gpt-4o on relationship
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name direct_generic \
--attribution_instruction_name default-relationship \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_direct_default-relationship_gpt-4o.jsonl \
--max_concurrency 32

# direct attribution gpt-4o on knowledge
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name direct_generic \
--attribution_instruction_name default-knowledge \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_direct_default-knowledge_gpt-4o.jsonl \
--max_concurrency 32

# direct attribution gpt-4o on secret
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name direct_generic \
--attribution_instruction_name default-secret \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_direct_default-secret_gpt-4o.jsonl \
--max_concurrency 32

# direct attribution gpt-4o on social_rules
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name direct_generic \
--attribution_instruction_name default-social_rules \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_direct_default-social_rules_gpt-4o.jsonl \
--max_concurrency 32

# direct attribution gpt-4o on financial_and_material_benefits
python sample_episodes_and_annotate.py \
--data_dir /Users/zhengyangqi/repos/sotopia-rl/data \
--llm_name gpt-4o \
--attribution_method_name direct_generic \
--attribution_instruction_name default-financial_and_material_benefits \
--input_file sotopia_pi_bc_episodes.jsonl \
--output_file sotopia_pi_bc_episodes_annotated_direct_default-financial_and_material_benefits_gpt-4o.jsonl \
--max_concurrency 32