"""
Fetch model information for annotated episodes from Redis.

Uses raw redis client to avoid sotopia Python version constraints.
Redis-om stores EpisodeLog as JSON at key:
    :sotopia.database.logs.EpisodeLog:<pk>

The `models` field is [judge_model, agent1_model, agent2_model] and is the
ground truth for which model played which agent position. The `tag` field
names the experiment and encodes step/checkpoint info but does NOT reflect
agent ordering.

We use `models` for position assignment and cross-reference the tag to
recover the step number for each base model name.

Outputs a JSON mapping: episode_id -> {agent_1_model, agent_2_model, tag, ...}
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import redis

ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = ANALYSIS_DIR / "episode_models.json"

ANNOTATOR_FOLDERS = {
    "an01_split_1": "split_1",
    "an01_split_2": "split_2",
    "an02_split_1": "split_1",
    "an02_split_2": "split_2",
    "an03_split_1": "split_1",
}


def get_all_episode_ids() -> set[str]:
    """Collect all unique episode IDs from annotation folders."""
    ids = set()
    annotations_dir = ANALYSIS_DIR / "annotations"
    for folder_name in ANNOTATOR_FOLDERS:
        folder = annotations_dir / folder_name
        if not folder.exists():
            continue
        for fp in folder.glob("*.json"):
            with open(fp) as f:
                ann = json.load(f)
            ids.add(ann["episode_id"])
    return ids


def clean_model_name(raw: str) -> str:
    """Extract the base model name from a raw model string.

    'custom/sft_0510_epoch_500-gpu2@http://localhost:7010/v1' -> 'sft_0510_epoch_500'
    """
    name = raw.split("@")[0]
    if "/" in name:
        name = name.split("/", 1)[1]
    name = name.rsplit("-gpu", 1)[0]
    return name


MODEL_ALIASES = {
    "sft_0510_epoch500": "sft_0510_epoch_500",
}

TAG_PATTERN = re.compile(
    r"^(?P<model1>.+?)_vs_(?P<model2>.+?)-(?P<date>\d{4})(?:_v\d+)?$"
)

STEP_PATTERN = re.compile(r"^(?P<base>.+?)_step_(?P<step>\d+)$")


def build_step_map(tag: str) -> dict[str, str]:
    """Parse the tag to build a mapping from base model name -> full name with step.

    E.g. tag 'pi_sft_step_1500_vs_sft_0510_epoch_500_step_200-0509_v0'
    returns {'pi_sft': 'pi_sft_step_1500', 'sft_0510_epoch_500': 'sft_0510_epoch_500_step_200'}
    """
    m = TAG_PATTERN.match(tag)
    if not m:
        return {}
    result = {}
    for model_str in [m.group("model1"), m.group("model2")]:
        normalized = model_str
        for old, new in MODEL_ALIASES.items():
            normalized = normalized.replace(old, new)
        sm = STEP_PATTERN.match(normalized)
        if sm:
            result[sm.group("base")] = normalized
    return result


def fetch_from_redis(episode_ids: set[str], redis_url: str) -> dict:
    """Fetch model info for each episode from Redis."""
    r = redis.from_url(redis_url, decode_responses=True)

    results = {}
    key_prefix = ":sotopia.database.logs.EpisodeLog:"

    for eid in sorted(episode_ids):
        key = f"{key_prefix}{eid}"
        try:
            data = r.json().get(key)
        except Exception:
            data = None

        if data is None:
            try:
                data = r.hgetall(key)
            except Exception:
                data = None

        if not data:
            print(f"  WARNING: episode {eid} not found in Redis")
            results[eid] = {"agent_1_model": "UNKNOWN", "agent_2_model": "UNKNOWN"}
            continue

        models = data.get("models", [])
        if isinstance(models, str):
            models = json.loads(models)

        tag = data.get("tag", "")
        if isinstance(tag, bytes):
            tag = tag.decode()

        step_map = build_step_map(tag)

        if len(models) >= 3:
            base1 = clean_model_name(models[1])
            base2 = clean_model_name(models[2])
            base1 = MODEL_ALIASES.get(base1, base1)
            base2 = MODEL_ALIASES.get(base2, base2)

            results[eid] = {
                "agent_1_model": step_map.get(base1, base1),
                "agent_2_model": step_map.get(base2, base2),
                "agent_1_model_raw": models[1],
                "agent_2_model_raw": models[2],
                "judge_model": models[0],
                "tag": tag,
            }
        else:
            print(f"  WARNING: episode {eid} has unexpected models field: {models}")
            results[eid] = {"agent_1_model": "UNKNOWN", "agent_2_model": "UNKNOWN"}

    return results


def main():
    redis_url = os.environ.get("REDIS_OM_URL", "")
    if not redis_url:
        print("ERROR: REDIS_OM_URL environment variable not set.")
        print("Usage: REDIS_OM_URL='redis://:password@host:port' python fetch_episode_models.py")
        sys.exit(1)

    episode_ids = get_all_episode_ids()
    print(f"Found {len(episode_ids)} unique episode IDs in annotations")

    print(f"Connecting to Redis: {redis_url.split('@')[-1] if '@' in redis_url else redis_url}")
    results = fetch_from_redis(episode_ids, redis_url)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"Saved model info to {OUTPUT_PATH}")

    found = sum(1 for v in results.values() if v["agent_1_model"] != "UNKNOWN")
    print(f"  {found}/{len(results)} episodes resolved")

    models_seen = set()
    for v in results.values():
        models_seen.add(v.get("agent_1_model", ""))
        models_seen.add(v.get("agent_2_model", ""))
    models_seen.discard("UNKNOWN")
    print(f"  Models found: {sorted(models_seen)}")


if __name__ == "__main__":
    main()
