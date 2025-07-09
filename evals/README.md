# 📊 Model Evaluation

This section describes how to deploy and evaluate trained models (e.g., Behavior Cloning vs. GRPO) using vLLM, Django, and the sotopia evaluation framework.

### Sotopia Evaluation Framework

We use `sotopia==0.1.0rc5` for evaluation. First, create the environment and install the correct version:

```bash
conda create -n sotopia python=3.10
conda activate sotopia
pip install sotopia==0.1.0rc5
```

### Environment Setup

Make sure to set the required environment variables **in all terminal windows** (one for each model server).

```bash
conda activate sotopia-rl

# Set paths
export REPO_FOLDER_NAME="<your_sotopia-rl_repo_path>"
export MODEL_PATH="<your_base_model_path>"
export CHAT_TEMPLATE="${REPO_FOLDER_NAME}/evals/qwen2.5-7b.jinja"

# Set GPUs and ports
export SFT_GPU=0
export GRPO_GPU=1
export SFT_PORT=7010
export GRPO_PORT=7020

# Model folders and checkpoints
export SFT_MODEL_FOLDER_NAME="<your_sft_model_folder_name>"
export GRPO_MODEL_FOLDER_NAME="<your_grpo_model_folder_name>"
export SFT_MODEL_CKPT_STEP=<your_best_sft_checkpoint>
export GRPO_MODEL_CKPT_STEP=<your_best_grpo_checkpoint>

# Full checkpoint paths
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/${SFT_MODEL_FOLDER_NAME}/checkpoint-${SFT_MODEL_CKPT_STEP}/"
export GRPO_MODEL_PATH="${REPO_FOLDER_NAME}/${GRPO_MODEL_FOLDER_NAME}/checkpoint-${GRPO_MODEL_CKPT_STEP}/"

# Names for served models
export SFT_MODEL_NAME="${SFT_MODEL_FOLDER_NAME}-gpu${SFT_GPU}"
export GRPO_MODEL_NAME="${GRPO_MODEL_FOLDER_NAME}-gpu${GRPO_GPU}"

# Final evaluation tags
export ENV_MODEL="gpt-4o"
export TAG="${GRPO_MODEL_FOLDER_NAME}_step_${GRPO_MODEL_CKPT_STEP}_vs_${SFT_MODEL_FOLDER_NAME}_step_${SFT_MODEL_CKPT_STEP}"

# Endpoint URLs
export MODEL_A="custom/${GRPO_MODEL_NAME}@http://localhost:${GRPO_PORT}/v1"
export MODEL_B="custom/${SFT_MODEL_NAME}@http://localhost:${SFT_PORT}/v1"
```

### Launch Model Servers (LoRA-enabled)

**Terminal 1: Serve SFT Model**

```bash
CUDA_VISIBLE_DEVICES=$SFT_GPU python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port "$SFT_PORT" \
    --max-lora-rank 64 \
    --chat-template $CHAT_TEMPLATE \
    --served-model-name qwen25-7b-instruct \
    --enable-lora \
    --lora-modules "$SFT_MODEL_NAME=$SFT_MODEL_PATH"
```

**Terminal 2: Serve GRPO Model**

```bash
CUDA_VISIBLE_DEVICES=$GRPO_GPU python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port "$GRPO_PORT" \
    --max-lora-rank 64 \
    --chat-template $CHAT_TEMPLATE \
    --served-model-name qwen25-7b-instruct \
    --enable-lora \
    --lora-modules "$GRPO_MODEL_NAME=$GRPO_MODEL_PATH"
```

### Run Evaluation with Sotopia

##### Terminal 3: Run Evaluation

```bash
git clone https://github.com/sotopia-lab/sotopia.git
cd sotopia
conda activate sotopia
git checkout tags/v0.1.0-rc.5
cd sotopia
```

Ensure **all environment variables** listed above are exported before running the evaluation.

```bash
python examples/experiment_eval.py \
  --gin_file sotopia_conf/generation_utils_conf/generate.gin \
  --gin_file sotopia_conf/server_conf/server.gin \
  --gin_file sotopia_conf/run_async_server_in_batch.gin \
  --gin.BATCH_SIZE=20 \
  --gin.PUSH_TO_DB=True \
  '--gin.ENV_IDS=[your_env_ids]' \
  "--gin.ENV_MODEL='${ENV_MODEL}'" \
  "--gin.AGENT1_MODEL='${MODEL_A}'" \
  "--gin.AGENT2_MODEL='${MODEL_B}'" \
  "--gin.TAG='${TAG}'"

# Reverse agents
python examples/experiment_eval.py \
  --gin_file sotopia_conf/generation_utils_conf/generate.gin \
  --gin_file sotopia_conf/server_conf/server.gin \
  --gin_file sotopia_conf/run_async_server_in_batch.gin \
  --gin.BATCH_SIZE=20 \
  --gin.PUSH_TO_DB=True \
  '--gin.ENV_IDS=[your_env_ids]' \
  "--gin.ENV_MODEL='${ENV_MODEL}'" \
  "--gin.AGENT1_MODEL='${MODEL_B}'" \
  "--gin.AGENT2_MODEL='${MODEL_A}'" \
  "--gin.TAG='${TAG}'"
```

For ENV_IDS in ` sotopia-hard`  and ` sotopia-all` , please see [this file](https://github.com/sotopia-lab/sotopia-rl/tree/main/data/env_ids.txt).
