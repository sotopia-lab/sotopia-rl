CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model /workspace/gemma-2-2b-it --chat-template /workspace/sotopia-rl/evals/gemma-2-2b-it.jinja 



poetry run python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHN7A1ZX5KSMT2YN9RXC4", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="gpt-3.5-turbo"' \
'--gin.AGENT2_MODEL="gpt-3.5-turbo"' \
'--gin.BATCH_SIZE=1' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="gpt-3.5-turbo_vs_gpt-3.5-turbo-1003"'

poetry run python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHN7A1ZX5KSMT2YN9RXC4", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="gpt-4o-mini"' \
'--gin.AGENT2_MODEL="gpt-3.5-turbo"' \
'--gin.BATCH_SIZE=1' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="gpt-4o-mini_vs_gpt-3.5-turbo-1003"'

poetry run python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHN7A1ZX5KSMT2YN9RXC4", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="gpt-4o"' \
'--gin.AGENT2_MODEL="gpt-3.5-turbo"' \
'--gin.BATCH_SIZE=1' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="gpt-4o_vs_gpt-3.5-turbo-1003"'

poetry run python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHN7A1ZX5KSMT2YN9RXC4", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="custom/workspace/gemma-2-2b-it@http://localhost:8000/v1"' \
'--gin.AGENT2_MODEL="gpt-3.5-turbo"' \
'--gin.BATCH_SIZE=1' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="gemma-2-2b-it_vs_gpt-3.5-turbo-1003"'



CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model /workspace/gemma-2-2b-it --port 8000 --chat-template /workspace/sotopia-rl/evals/gemma-2-2b-it.jinja  --served-model-name gemma-2-2b --enable-lora --lora-modules sotopia_gemma-2-2b-sft=/workspace/sotopia-rl/saves/gemma-2-2b/lora/sft/

poetry run python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHN7A1ZX5KSMT2YN9RXC4", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="custom/sotopia_gemma-2-2b-sft@http://localhost:8000/v1"' \
'--gin.AGENT2_MODEL="gpt-3.5-turbo"' \
'--gin.BATCH_SIZE=1' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="sotopia_gemma-2-2b-sft_vs_gpt-3.5-turbo-1003"'



CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model /workspace/gemma-2-2b-it --port 8000 --chat-template /workspace/sotopia-rl/evals/gemma-2-2b-it.jinja --served-model-name gemma-2-2b --enable-lora --disable-sliding-window --max-model-len 4096 --lora-modules sotopia_gemma-2-2b-sft-ppo-rm-baseline=/workspace/sotopia-rl/saves/gemma-2-2b/lora/ppo_rm_baseline/checkpoint-250

poetry run python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHN7A1ZX5KSMT2YN9RXC4", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="custom/sotopia_gemma-2-2b-sft-ppo-rm-baseline@http://localhost:8000/v1"' \
'--gin.AGENT2_MODEL="gpt-3.5-turbo"' \
'--gin.BATCH_SIZE=1' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="sotopia_gemma-2-2b-sft-ppo-rm-baseline_vs_gpt-3.5-turbo-1003"'



CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model /workspace/gemma-2-2b-it --port 8000 --chat-template /workspace/sotopia-rl/evals/gemma-2-2b-it.jinja --served-model-name gemma-2-2b --enable-lora --disable-sliding-window --max-model-len 4096 --lora-modules sotopia_gemma-2-2b-sft-ppo-rm-key=/workspace/sotopia-rl/saves/gemma-2-2b/lora/ppo_rm_key/checkpoint-250

poetry run python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHN7A1ZX5KSMT2YN9RXC4", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="custom/sotopia_gemma-2-2b-sft-ppo-rm-key@http://localhost:8007/v1"' \
'--gin.AGENT2_MODEL="gpt-3.5-turbo"' \
'--gin.BATCH_SIZE=1' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="sotopia_gemma-2-2b-sft-rm-key_vs_gpt-3.5-turbo-1003"'



CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model /workspace/gemma-2-2b-it --port 8000 --chat-template /workspace/sotopia-rl/evals/gemma-2-2b-it.jinja --served-model-name gemma-2-2b --enable-lora --disable-sliding-window --max-model-len 4096 --lora-modules sotopia_gemma-2-2b-sft-ppo-rm-direct=/workspace/sotopia-rl/saves/gemma-2-2b/lora/ppo_rm_direct/checkpoint-250

poetry run python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHN7A1ZX5KSMT2YN9RXC4", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="custom/sotopia_gemma-2-2b-sft-ppo-rm-direct@http://localhost:8008/v1"' \
'--gin.AGENT2_MODEL="gpt-3.5-turbo"' \
'--gin.BATCH_SIZE=1' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="sotopia_gemma-2-2b-sft-ppo-rm-direct_vs_gpt-3.5-turbo-1003"'

