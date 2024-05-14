import sys
sys.path.append('../../')
import argparse

from src.utils.preprocess import add_score
from src.human_annotate.episode_sampling import sample_episodes
from src.prompting.attribution_prompting import generate_reward_attribution
from src.human_annotate.form_creation import create_forms

def main(data_dir: str, llm_name: str) -> None:
    add_score(data_dir, "sotopia_pi_episodes.jsonl", "sotopia_pi_episodes_with_scores.jsonl")
    generate_reward_attribution(data_dir, llm_name=llm_name, input_file="sotopia_pi_episodes_with_scores.jsonl", output_file="sotopia_pi_openai_log_attribution.jsonl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing data files')
    parser.add_argument('--llm_name', type=str, required=True, help='Name of the language model')
    
    args = parser.parse_args()
    
    main(args.data_dir, args.llm_name)