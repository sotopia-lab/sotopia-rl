import click
import json
import os
import glob
from typing import Any, Dict, List
from tqdm import tqdm
from db_free_reverse_engineering import run_reverse_by_pk_agent

@click.command()
@click.option("--data_dir", type=str, required=True, help="Directory containing data files.")
@click.option("--utterances_output_subdir", type=str, required=True, help="Directory to save the utterances.")
@click.option("--episodes_file", type=str, required=True, help="Path to the raw JSON file.")
@click.option("--sft_output_file", type=str, required=False, help="Path to the processed JSON file.")
def main(data_dir: str, utterances_output_subdir: str, episodes_file: str, sft_output_file: str) -> None:
    episode_path = os.path.join(data_dir, episodes_file)
    if not os.path.exists(episode_path):
        raise Exception(f"Episodes file not found: {episode_path}")
    
    with open(episode_path, 'r') as f:
        data: List[Dict[str, Any]] = [json.loads(d) for d in f.readlines()]
    
    cache_dir = os.path.join(data_dir, utterances_output_subdir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        for d in tqdm(data):
            run_reverse_by_pk_agent(d['episode_id'], True, cache_dir, episode_path)
            run_reverse_by_pk_agent(d['episode_id'], False, cache_dir, episode_path)
    
    utterances = []
    for record in glob.glob(f"{cache_dir}/*.json"):
        with open(record, 'r') as f:
            uttr = json.load(f)
            utterances.append(uttr)
    
    sft_utterances = []
    for uttr in utterances:
        sft_utterances.append({
            "input": uttr['prompt'] + " Your available action types are\nspeak none action leave non-verbal communication.\nNote: You can \"leave\" this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.\n\nPlease only generate a JSON string including the action type and the argument.\nYour action should follow the given format:\nThe output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}\nthe object {\"foo\": [\"bar\", \"baz\"]} is a well-formatted instance of the schema. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{\"properties\": {\"action_type\": {\"description\": \"whether to speak at this turn or choose to not do anything\", \"enum\": [\"none\", \"speak\", \"non-verbal communication\", \"action\", \"leave\"], \"title\": \"Action Type\", \"type\": \"string\"}, \"argument\": {\"description\": \"the utterance if choose to speak, the expression or gesture if choose non-verbal communication, or the physical action if choose action\", \"title\": \"Argument\", \"type\": \"string\"}}, \"required\": [\"action_type\", \"argument\"]}\n```",
            "output": uttr['result'],
        })
    
    print(f"Total utterances: {len(sft_utterances)}")
    with open(os.path.join(data_dir, sft_output_file), 'w') as f:
        json.dump(sft_utterances, f, indent=4)

if __name__ == "__main__":
    main()
