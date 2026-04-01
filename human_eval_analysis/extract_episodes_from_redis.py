"""
Extract human_eval episodes from Redis and convert to the JSONL format
expected by the annotation pipeline (sample_episodes_and_annotate.py).

Required JSONL fields:
  - episode_id
  - scenario
  - social_interactions  (agent utterances joined by \n\n)
  - social_goals         (OrderedDict: agent_name -> goal)
  - agents_background    (OrderedDict: agent_name -> background text)
  - rewards              (list of [overall_score, {goal: X, believability: Y, ...}])
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

import redis

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_PATH = ANALYSIS_DIR / "episode_models.json"
OUTPUT_FILE = DATA_DIR / "human_eval_episodes.jsonl"


def get_all_episode_ids() -> set[str]:
    """Collect episode IDs from human_eval markdown files."""
    ids = set()
    human_eval_dir = REPO_ROOT / "human_eval"
    for split_dir in human_eval_dir.iterdir():
        if not split_dir.is_dir():
            continue
        for md_file in split_dir.glob("*.md"):
            ids.add(md_file.stem)
    return ids


def build_agent_background(profile: dict) -> str:
    """Build a background text string from an agent profile."""
    name = f"{profile['first_name']} {profile['last_name']}"
    age = profile.get("age", "")
    gender = profile.get("gender", "").lower()
    occupation = profile.get("occupation", "")
    pronouns = profile.get("gender_pronoun", "")
    public_info = profile.get("public_info", "")
    personality = profile.get("personality_and_values", "")
    secret = profile.get("secret", "")

    parts = [
        f"{name} is a {age}-year-old {gender} {occupation.lower()}.",
    ]
    if pronouns:
        parts.append(f"{pronouns} pronouns.")
    if public_info:
        parts.append(public_info)
    if personality:
        parts.append(f"Personality and values description: {personality}")
    if secret:
        parts.append(f"{name}'s secrets: {secret}")

    return " ".join(parts)


def extract_utterances(messages: list) -> list[tuple[str, str]]:
    """Extract (speaker, utterance_text) pairs from raw messages.

    Each message group has [env->a1, env->a2, a1->env, a2->env].
    The agent->env messages contain "said: ..." or "left the conversation" etc.
    """
    utterances = []
    for turn in messages:
        for msg in turn:
            sender, receiver, content = msg[0], msg[1], msg[2]
            if receiver == "Environment" and sender != "Environment":
                content = content.strip()
                if content == "did nothing":
                    continue
                if content.startswith("said: "):
                    text = content[6:].strip().strip('"').strip('\u201c').strip('\u201d')
                    utterances.append((sender, f'{sender}: "{text}"'))
                elif "left the conversation" in content:
                    utterances.append((sender, f"{sender}: left the conversation"))
    return utterances


def main():
    redis_url = os.environ.get("REDIS_OM_URL", "")
    if not redis_url:
        print("ERROR: REDIS_OM_URL environment variable not set.")
        sys.exit(1)

    r = redis.from_url(redis_url, decode_responses=True)
    ep_prefix = ":sotopia.database.logs.EpisodeLog:"
    env_prefix = ":sotopia.database.persistent_profile.EnvironmentProfile:"
    agent_prefix = ":sotopia.database.persistent_profile.AgentProfile:"

    episode_ids = get_all_episode_ids()
    print(f"Found {len(episode_ids)} episode IDs from human_eval/")

    DATA_DIR.mkdir(exist_ok=True)

    results = []
    errors = []

    for eid in sorted(episode_ids):
        ep_data = r.json().get(f"{ep_prefix}{eid}")
        if not ep_data:
            errors.append(f"Episode {eid}: not found in Redis")
            continue

        env_id = ep_data["environment"]
        agent_ids = ep_data["agents"]
        messages = ep_data.get("messages", [])
        rewards_raw = ep_data.get("rewards", [])

        env_data = r.json().get(f"{env_prefix}{env_id}")
        if not env_data:
            errors.append(f"Episode {eid}: environment {env_id} not found")
            continue

        agent_profiles = []
        for aid in agent_ids:
            prof = r.json().get(f"{agent_prefix}{aid}")
            if not prof:
                errors.append(f"Episode {eid}: agent {aid} not found")
                break
            agent_profiles.append(prof)
        if len(agent_profiles) != 2:
            continue

        agent_names = [f"{p['first_name']} {p['last_name']}" for p in agent_profiles]

        utterances = extract_utterances(messages)
        social_interactions = "\n\n".join(utt_text for _, utt_text in utterances)

        social_goals = OrderedDict()
        for i, name in enumerate(agent_names):
            goal = env_data["agent_goals"][i] if i < len(env_data.get("agent_goals", [])) else ""
            social_goals[name] = goal

        agents_background = OrderedDict()
        for i, name in enumerate(agent_names):
            agents_background[name] = build_agent_background(agent_profiles[i])

        episode_record = {
            "episode_id": eid,
            "scenario": env_data.get("scenario", ""),
            "codename": env_data.get("codename", ""),
            "social_interactions": social_interactions,
            "social_goals": social_goals,
            "agents_background": agents_background,
            "rewards": rewards_raw,
            "tag": ep_data.get("tag", ""),
        }
        results.append(episode_record)

    with open(OUTPUT_FILE, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(results)} episodes to {OUTPUT_FILE}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
