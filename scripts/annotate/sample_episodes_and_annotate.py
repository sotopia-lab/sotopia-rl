import sys

sys.path.append("../../")
import argparse

from sotopia_rl.prompter.attribution_prompting import (
    parallel_generate_reward_attribution,
)
from sotopia_rl.utils.preprocess import add_score


def main(data_dir: str,
        llm_name: str,
        input_file: str,
        output_file: str,
        attribution_method_name: str,
        attribution_instruction_name: str,
        max_concurrency: int = 1
    ) -> None:

    add_score(
        data_dir,
        input_file,
        "sotopia_pi_episodes_with_scores.jsonl",
    )
    parallel_generate_reward_attribution(
        data_dir,
        llm_name=llm_name,
        input_file="sotopia_pi_episodes_with_scores.jsonl",
        output_file=output_file,
        attribution_method_name=attribution_method_name,
        attribution_instruction_name=attribution_instruction_name,
        max_concurrency=max_concurrency
    )


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
        "--attribution_method_name",
        type=str,
        required=False,
        help="Type of attribution method",
    )
    parser.add_argument(
        "--attribution_instruction_name",
        type=str,
        required=False,
        help="Type of attribution instruction",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input file containing episodes",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file containing episodes with reward attribution",
    )
    parser.add_argument(
        "--max_concurrency",
        type=int,
        required=False,
        default=1,
        help="Maximum number of concurrent episodes",
    )


    args = parser.parse_args()

    main(args.data_dir,
        args.llm_name,
        args.input_file,
        args.output_file,
        args.attribution_method_name,
        args.attribution_instruction_name,
        args.max_concurrency
        )
