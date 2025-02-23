# sotopia-rl

## Install

Create a conda environment with python 3.10
```
conda create -n sotopia-rl python=3.10
conda activate sotopia-rl
```

Then install poetry and use it to install the dependencies. Currently the package is under development so it's recommended to use the `--no-root` flag to avoid installing the package itself.
```
pip install poetry
poetry install --no-root
```

## Generating LLM Annotations

To generate LLM annotations, you need to download the original sotopia-pi episodes file from the [huggingface repository](https://huggingface.co/datasets/cmu-lti/sotopia-pi/tree/main) and place it in the `data` folder. Then run the following command:
```
cd scripts/annotate
python process_sotopia_pi.py --data_dir /workspace/sotopia-rl/data --input_file sotopia_pi_episodes.jsonl --output_file sotopia_pi_bc_episodes.jsonl
```
This will generate a new file `sotopia_pi_bc_episodes.jsonl` in the `data` folder.

Then run the following command to generate the LLM annotations:
```
python sample_episodes_and_annotate.py --data_dir /workspace/sotopia-rl/data --llm_name gpt-4o --input_file sotopia_pi_bc_episodes.jsonl --output_file sotopia_pi_bc_episodes_annotated.jsonl
```
The annotations will need to be furuther processed into the format required by the training script. This can be done by running the following command:
```
cd ../data_process
python process_annotation_direct_attribution.py --data_dir /workspace/sotopia-rl/data --input_file sotopia_pi_bc_episodes_annotated.jsonl --reward_output_file sotopia_pi_bc_episodes_reward.json --ppo_output_file sotopia_pi_bc_episodes_ppo.json
```