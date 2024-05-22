import sys

sys.path.append("../../")
import argparse

from src.human_annotate.form_response_retrieval import (
    get_episodes_from_form_ids,
    retrieve_responses,
)
from src.prompting.key_utterance_prompting import (
    generate_key_utterance_recognition,
)
from src.utils.preprocess import add_score


def main(data_dir: str, llm_name: str) -> None:
    generate_key_utterance_recognition(data_dir, llm_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing data files",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        required=True,
        help="Name of the language model",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=False,
        default="example_episodes_with_scores.jsonl",
        help="Input file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        default="openai_log_attribution.jsonl",
        help="Output file",
    )

    args = parser.parse_args()
    print(args.data_dir, args.llm_name)

    main(args.data_dir, args.llm_name)
