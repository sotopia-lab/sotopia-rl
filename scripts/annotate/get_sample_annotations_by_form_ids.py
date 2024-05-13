import sys
sys.path.append('../../')
import argparse

from src.utils.preprocess import add_score
from src.human_annotate.form_response_retrieval import get_episodes_from_form_ids, retrieve_responses
from src.prompting.attribution_prompting import generate_reward_attribution

def main(data_dir: str, llm_name: str, gcp_key: str) -> None:
    # get_episodes_from_form_ids(data_dir, gcp_key)
    # generate_reward_attribution(data_dir, llm_name)
    retrieve_responses(data_dir, gcp_key)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing data files')
    parser.add_argument('--llm_name', type=str, required=True, help='Name of the language model')
    parser.add_argument('--gcp_key', type=str, required=True, help='Path to GCP credentials')
    
    args = parser.parse_args()
    
    main(args.data_dir, args.llm_name, args.gcp_key)