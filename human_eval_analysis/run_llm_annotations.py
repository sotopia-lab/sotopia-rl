"""
Run LLM attribution annotations on all human_eval episodes for all 7 dimensions.

This script:
1. Adds scores to the episodes JSONL (required preprocessing step)
2. Runs parallel_generate_reward_attribution for each dimension
3. Outputs one JSONL per dimension

Usage:
    cd human_eval_analysis
    OPENAI_API_KEY=sk-... python run_llm_annotations.py [--dimensions goal,believability,...] [--llm gpt-4o] [--concurrency 16]
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

ALL_DIMENSIONS = [
    "goal",
    "believability",
    "relationship",
    "knowledge",
    "secret",
    "social_rules",
    "financial_and_material_benefits",
]

INPUT_FILE = "human_eval_episodes.jsonl"
SCORES_FILE = "human_eval_episodes_with_scores.jsonl"


def setup_imports():
    """Workaround: stub sotopia_rl package to avoid importing training code."""
    env_file = REPO_ROOT / "human_eval_interface" / ".env"
    if env_file.exists():
        for line in env_file.read_text().strip().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

    sys.modules["sotopia_rl"] = types.ModuleType("sotopia_rl")
    sys.modules["sotopia_rl"].__path__ = [str(REPO_ROOT / "sotopia_rl")]

    mod = types.ModuleType("sotopia_rl.prompter")
    mod.__path__ = [str(REPO_ROOT / "prompter")]
    sys.modules["sotopia_rl.prompter"] = mod

    umod = types.ModuleType("sotopia_rl.utils")
    umod.__path__ = [str(REPO_ROOT / "sotopia_rl" / "utils")]
    sys.modules["sotopia_rl.utils"] = umod


def main():
    parser = argparse.ArgumentParser(description="Run LLM annotations on human_eval episodes")
    parser.add_argument("--dimensions", type=str, default=",".join(ALL_DIMENSIONS),
                        help="Comma-separated list of dimensions to annotate")
    parser.add_argument("--llm", type=str, default="gpt-4o", help="LLM model name")
    parser.add_argument("--concurrency", type=int, default=16, help="Max concurrent API calls")
    parser.add_argument("--scale", type=str, default="default", help="Attribution scale (default=3-point)")
    parser.add_argument("--run", type=int, default=1, help="Run number (for multi-run aggregation)")
    args = parser.parse_args()

    dimensions = [d.strip() for d in args.dimensions.split(",")]
    print(f"Dimensions: {dimensions}")
    print(f"LLM: {args.llm}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Scale: {args.scale}")
    print(f"Data dir: {DATA_DIR}")
    print()

    setup_imports()

    from sotopia_rl.utils.preprocess import add_score
    from sotopia_rl.prompter.attribution_prompting import parallel_generate_reward_attribution

    input_path = DATA_DIR / INPUT_FILE
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run extract_episodes_from_redis.py first.")
        sys.exit(1)

    print("Step 1: Adding scores to episodes...")
    add_score(str(DATA_DIR), INPUT_FILE, SCORES_FILE)
    print(f"  Wrote {SCORES_FILE}")
    print()

    for dim in dimensions:
        output_file = f"human_eval_annotated_{args.scale}-{dim}_{args.llm}_run{args.run}.jsonl"
        output_path = DATA_DIR / output_file

        print(f"Step 2: Annotating dimension '{dim}'...")
        print(f"  Output: {output_file}")

        instruction_name = f"{args.scale}-{dim}"

        parallel_generate_reward_attribution(
            data_dir=str(DATA_DIR),
            llm_name=args.llm,
            input_file=SCORES_FILE,
            output_file=output_file,
            attribution_method_name="direct_generic",
            attribution_instruction_name=instruction_name,
            max_concurrency=args.concurrency,
        )
        print(f"  Done: {dim}")
        print()

    print("All dimensions annotated.")


if __name__ == "__main__":
    main()
