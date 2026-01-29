#!/usr/bin/env python3
"""
Delete EpisodeLog entries from Redis by tag.

Usage:
    python evals/delete_redis_episodes.py "tag_name"
    python evals/delete_redis_episodes.py "tag1" "tag2" "tag3"
"""

import argparse

from sotopia.database.logs import EpisodeLog


def find_episodes_by_tag(tag: str) -> list:
    """Find all episodes with exact tag match."""
    return EpisodeLog.find(EpisodeLog.tag == tag).all()


def delete_episodes(episodes: list) -> int:
    """Delete episodes from Redis."""
    deleted_count = 0
    errors = 0
    
    for i, episode in enumerate(episodes):
        try:
            EpisodeLog.delete(episode.pk)
            deleted_count += 1
            if (i + 1) % 100 == 0:
                print(f"  Deleted {i + 1}/{len(episodes)}...")
        except Exception as e:
            errors += 1
            print(f"  Error deleting {episode.pk}: {e}")
    
    return deleted_count, errors


def main():
    parser = argparse.ArgumentParser(
        description="Delete EpisodeLog entries from Redis by tag"
    )
    parser.add_argument(
        "tags",
        nargs="+",
        help="Tag(s) to delete (exact match)"
    )
    
    args = parser.parse_args()
    
    total_deleted = 0
    total_errors = 0
    
    for tag in args.tags:
        print(f"Deleting tag: '{tag}'")
        episodes = find_episodes_by_tag(tag)
        print(f"  Found {len(episodes)} episodes")
        
        if episodes:
            deleted, errors = delete_episodes(episodes)
            total_deleted += deleted
            total_errors += errors
            print(f"  Deleted {deleted} episodes")
    
    print(f"\nTotal: {total_deleted} deleted, {total_errors} errors")


if __name__ == "__main__":
    main()
