import json
import os
from typing import Any, Dict, List

from tqdm import tqdm

from src.human_annotate.google_form_apis import (
    get_form,
    get_form_responses,
)

from ..utils.preprocess import extract_goal_scores

GoogleResource = Any


def retrieve_responses(data_dir: str, gcp_key: str) -> None:
    with open(
        os.path.join(data_dir, "openai_log_attribution.jsonl"), "r"
    ) as f:
        log = [json.loads(line) for line in f]

    with open(os.path.join(data_dir, "form_uris.jsonl"), "r") as f:
        form_uris = [json.loads(line) for line in f]

    log = add_responses_to_sheet(log, form_uris, gcp_key)

    with open(os.path.join(data_dir, "human_log_attribution.jsonl"), "w") as f:
        for item in log:
            f.write(json.dumps(item))
            f.write("\n")


def get_episodes_from_form_ids(data_dir: str, gcp_key: str) -> None:
    with open(os.path.join(data_dir, "sotopia_episodes_v1.jsonl"), "r") as f:
        episodes = [json.loads(line) for line in f]

    with open(os.path.join(data_dir, "form_ids.txt"), "r") as f:
        form_ids = f.readlines()

    form_ids = [form_id.strip() for form_id in form_ids]

    print("retrieving episodes from form ids")
    example_episodes = []
    visited = set()
    for form_id in tqdm(form_ids):
        form = get_form(form_id, gcp_key)
        episode_id = form["info"]["title"].split(" ")[-1]
        for episode in episodes:
            if (
                episode["episode_id"] == episode_id
                and episode_id not in visited
            ):
                visited.add(episode_id)
                example_episodes.append(episode)
                break

    with open(os.path.join(data_dir, "example_episodes.jsonl"), "w") as f:
        for episode in example_episodes:
            f.write(json.dumps(episode) + "\n")

    example_episodes_with_scores = extract_goal_scores(example_episodes)
    with open(
        os.path.join(data_dir, "example_episodes_with_scores.jsonl"), "w"
    ) as f:
        for episode in example_episodes_with_scores:
            f.write(json.dumps(episode) + "\n")


def add_responses_to_sheet(
    log: List[Dict[str, Any]], form_uris: List[Dict[str, str]], gcp_key: str
) -> List[Dict[str, Any]]:
    """Add responses to rewarded attribution log."""
    for i in range(len(log)):
        form_id = form_uris[i]["formId"]
        print(f"Log: {i}")
        print(f"Form ID: {form_id}")
        form_schema = get_form(form_id, gcp_key)
        responses = get_form_responses(form_id, gcp_key)
        print(f"Responses: {responses}")
        for key in log[i]["attributed_utterances"]:
            print(f"  Key: {key}")
            item, next_item = None, None
            for item_idx in range(len(form_schema["items"])):
                item = form_schema["items"][item_idx]
                if item["title"].split(":")[0] == key:
                    if item_idx + 1 < len(form_schema["items"]):
                        next_item = form_schema["items"][item_idx + 1]
                    break

            if item and "questionItem" in item:
                question_id = item["questionItem"]["question"]["questionId"]
                for response in responses:
                    response_answer = response["answers"][question_id]
                    if len(log[i]["attributed_utterances"][key]) == 2:
                        log[i]["attributed_utterances"][key].append({})
                    log[i]["attributed_utterances"][key][2].update(
                        {
                            response["lastSubmittedTime"]: int(
                                response_answer["textAnswers"]["answers"][0][
                                    "value"
                                ]
                            )
                        }
                    )
                if next_item and "questionItem" in next_item:
                    next_question_id = next_item["questionItem"]["question"][
                        "questionId"
                    ]
                    for response in responses:
                        next_response_answer = response["answers"][
                            next_question_id
                        ]
                        if len(log[i]["attributed_utterances"][key]) == 3:
                            log[i]["attributed_utterances"][key].append({})
                        log[i]["attributed_utterances"][key][3].update(
                            {
                                response["lastSubmittedTime"]: str(
                                    next_response_answer["textAnswers"][
                                        "answers"
                                    ][0]["value"]
                                )
                            }
                        )
                    print(log[i]["attributed_utterances"][key][3])
            else:
                print("  No question item found")
        print(f"Updated log")
    return log
