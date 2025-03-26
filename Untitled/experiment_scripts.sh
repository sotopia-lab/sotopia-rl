## Baseline vs Sotopia SFT
# baseline
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-7B-Instruct \
--port 8005 \
--chat-template /root/sotopia-rl/Untitled/qwen2.5-7b.jinja \
--served-model-name qwen-2.5-instruct

# sotopia-sft
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-7B-Instruct \
--port 8006 \
--chat-template /root/sotopia-rl/Untitled/qwen2.5-7b.jinja \
--enable-lora \
--lora-modules sotopia-sft=/root/sotopia-rl/Untitled/.cache/final_sft/checkpoint-1000 \
--served-model-name xx

python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="custom/qwen-2.5-instruct@http://localhost:8005/v1"' \
'--gin.AGENT2_MODEL="custom/sotopia-sft@http://localhost:8006/v1"' \
'--gin.BATCH_SIZE=20' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="baseline-sotopia-sft-3-25-v2"'

python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="custom/sotopia-sft@http://localhost:8006/v1"' \
'--gin.AGENT2_MODEL="custom/qwen-2.5-instruct@http://localhost:8005/v1"' \
'--gin.BATCH_SIZE=20' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="baseline-sotopia-sft-3-25-v2"'

## Sotopia SFT vs Sotopia SFT
# sotopia-sft
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-7B-Instruct \
--port 8005 \
--chat-template /root/sotopia-rl/Untitled/qwen2.5-7b.jinja \
--enable-lora \
--lora-modules sotopia-sft-1=/root/sotopia-rl/Untitled/.cache/final_sft/checkpoint-1000/ \
--served-model-name xx

# sotopia-sft
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-7B-Instruct \
--port 8006 \
--chat-template /root/sotopia-rl/Untitled/qwen2.5-7b.jinja \
--enable-lora \
--lora-modules sotopia-sft-2=/root/sotopia-rl/Untitled/.cache/final_sft/checkpoint-1000/ \
--served-model-name xx

python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="custom/sotopia-sft-1@http://localhost:8005/v1"' \
'--gin.AGENT2_MODEL="custom/sotopia-sft-2@http://localhost:8006/v1"' \
'--gin.BATCH_SIZE=20' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="sotopia-sft-sotopia-sft-3-21-v1"'

python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="custom/sotopia-sft-2@http://localhost:8006/v1"' \
'--gin.AGENT2_MODEL="custom/sotopia-sft-1@http://localhost:8005/v1"' \
'--gin.BATCH_SIZE=20' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="sotopia-sft-sotopia-sft-3-21-v1"'

## Baseline vs Baseline
# baseline 1
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-7B-Instruct \
--port 8005 \
--chat-template /root/sotopia-rl/Untitled/qwen2.5-7b.jinja \
--served-model-name qwen-2.5-instruct-1 &

# baseline 2
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-7B-Instruct \
--port 8006 \
--chat-template /root/sotopia-rl/Untitled/qwen2.5-7b.jinja \
--served-model-name qwen-2.5-instruct-2 &

# with sotopia env, launch the evaluation
python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="custom/qwen-2.5-instruct-1@http://localhost:8005/v1"' \
'--gin.AGENT2_MODEL="custom/qwen-2.5-instruct-2@http://localhost:8006/v1"' \
'--gin.BATCH_SIZE=10' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="baseline-baseline-3-19-v5"'

python examples/experiment_eval.py \
--gin_file sotopia_conf/generation_utils_conf/generate.gin \
--gin_file sotopia_conf/server_conf/server.gin \
--gin_file sotopia_conf/run_async_server_in_batch.gin \
'--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
'--gin.AGENT1_MODEL="custom/qwen-2.5-instruct-2@http://localhost:8006/v1"' \
'--gin.AGENT2_MODEL="custom/qwen-2.5-instruct-1@http://localhost:8005/v1"' \
'--gin.BATCH_SIZE=10' \
'--gin.PUSH_TO_DB=True' \
'--gin.TAG="baseline-baseline-3-19-v5"'
