import sys

sys.path.append("../../")
import argparse
from direct_attribution_instructions import ATTRIBUTION_INSTRUCTIONS_DICT
from sotopia_rl.prompter.attribution_prompting import generate_reward_attribution
from sotopia_rl.utils.preprocess import add_score

def main(data_dir: str, llm_name: str, input_file: str, output_file: str, attribution_instruction_name: str = "default") -> None:
    add_score(
        data_dir,
        input_file,
        "sotopia_pi_episodes_with_scores.jsonl",
    )
    generate_reward_attribution(
        data_dir,
        llm_name=llm_name,
        input_file="sotopia_pi_episodes_with_scores.jsonl",
        output_file=output_file,
        attribution_instruction=ATTRIBUTION_INSTRUCTIONS_DICT[attribution_instruction_name],
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
    

    args = parser.parse_args()

    main(args.data_dir, args.llm_name, args.input_file, args.output_file, args.attribution_instruction_name if args.attribution_instruction_name else None)
