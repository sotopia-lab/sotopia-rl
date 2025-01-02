import click
import json
import jsonlines
from rich import print
import os

@click.command()
@click.option("--data_dir", type=str, required=True, help="Directory containing data files.")
@click.option("--input_file", type=str, required=True, help="Path to the raw JSON file.")
@click.option("--output_file", type=str, required=True, help="Path to the processed JSON file.")
def main(data_dir: str, input_file: str, output_file: str) -> None:
    """
    Process the JSON file containing Sotopia episodes.
    """
    with open(os.path.join(data_dir, input_file), "r") as f:
        episodes = [json.loads(line) for line in f]
    print("[bold green]Successfully loaded episodes:[/bold green]")
    
    behavior_cloning_episodes = []
    for episode in episodes:
        if episode["experiment_model_name_pairs"][0] == "gpt-4" and episode["experiment_model_name_pairs"][1] == "gpt-4":
            behavior_cloning_episodes.append(episode)
    print(f"[bold green]Successfully filtered {len(behavior_cloning_episodes)} episodes [/bold green]")
    with open(os.path.join(data_dir, output_file), "w") as f:
        jsonlines.Writer(f).write_all(behavior_cloning_episodes)

if __name__ == "__main__":
    main()
