import json

from sotopia.database.logs import BaseEpisodeLog


class FakeEpisodeLog(BaseEpisodeLog):
    pk: str


def jsonl_to_episodes(
    jsonl_file_path: str,
) -> list[FakeEpisodeLog]:
    """Load episodes from a jsonl file.

    Args:
        jsonl_file_path (str): The file path.

    Returns:
        list[FakeEpisodeLog]: List of episodes that fakes an EpisodeLog object.
    """
    episodes = []
    with open(jsonl_file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            episode = FakeEpisodeLog(
                pk=data["episode_id"],
                environment=data["environment_id"],
                agents=data["agent_ids"],
                tag=data["experiment_tag"],
                models=data["experiment_model_name_pairs"],
                messages=data["raw_messages"],
                reasoning=data["reasoning"],
                rewards=data["raw_rewards"],
                rewards_prompt=data["raw_rewards_prompt"],
            )
            episodes.append(episode)
    return episodes
