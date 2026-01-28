#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh

# conda activate base
# ./evals/serve_dpo_checkpoints.sh 1000 &
# sleep 100
# conda activate sotopia-rl
# ./evals/run_dpo_eval.sh 1000

# conda activate base
# ./evals/serve_dpo_checkpoints.sh 2000 &
# sleep 100
# conda activate sotopia-rl
# ./evals/run_dpo_eval.sh 2000

conda activate base
./evals/serve_dpo_checkpoints.sh 3000 &
sleep 100
conda activate sotopia-rl
./evals/run_dpo_eval.sh 3000

# conda activate base
# ./evals/serve_dpo_checkpoints.sh 4000 &
# sleep 100
# conda activate sotopia-rl
# ./evals/run_dpo_eval.sh 4000

conda activate base
./evals/serve_dpo_checkpoints.sh 5000 &
sleep 100
conda activate sotopia-rl
./evals/run_dpo_eval.sh 5000

# conda activate base
# ./evals/serve_dpo_checkpoints.sh 6000 &
# sleep 100
# conda activate sotopia-rl
# ./evals/run_dpo_eval.sh 6000

# conda activate base
# ./evals/serve_dpo_checkpoints.sh 7000 &
# sleep 100
# conda activate sotopia-rl
# ./evals/run_dpo_eval.sh 7000
