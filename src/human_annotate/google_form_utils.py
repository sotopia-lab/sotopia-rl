import json
from typing import Any, Dict, List

from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials

from ..utils.preprocess import extract_goal_scores
from google_form_api_wrapper import authenticate_google_services, get_form, get_form_responses

GoogleResource = Any

def add_responses_to_sheet(
    log: List[Dict[str, Any]], form_uris: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """Add responses to rewarded attribution log."""
    for i in range(len(log)):
        print(f"Log: {i}")
        form_id = form_uris[i]["formId"]
        form_schema = get_form(form_id)
        responses = get_form_responses(form_id)
        print(f"Form ID: {form_id}")
        print(f"Responses: {responses}")
        for key in log[i]["rewarded_utterances"]:
            print(f"  Key: {key}")
            item = next(
                (
                    item
                    for item in form_schema["items"]
                    if item["title"].split(":")[0] == key
                ),
                None,
            )
            if item and "questionItem" in item:
                question_id = item["questionItem"]["question"]["questionId"]
                for response in responses:
                    response_id = response["responseId"]
                    for _, response_item in response["answers"].items():
                        response_question_id = response_item["questionId"]
                        if response_question_id == question_id:
                            print(f"  Response ID: {response_id}")
                            if len(log[i]["rewarded_utterances"][key]) == 2:
                                log[i]["rewarded_utterances"][key].append({})
                            log[i]["rewarded_utterances"][key][2].update(
                                {
                                    response["lastSubmittedTime"]: int(
                                        response_item["textAnswers"][
                                            "answers"
                                        ][0]["value"]
                                    )
                                }
                            )
                            break
            else:
                print("  No question item found")
        print(f"Updated log")
    return log

def get_form_responses():
    with open("openai_log_reward_attribution.jsonl", "r") as f:
        log = [json.loads(line) for line in f]

    with open("form_uris.jsonl", "r") as f:
        form_uris = [json.loads(line) for line in f]

    log = add_responses_to_sheet(log, form_uris)

    with open("reward_attribution_log_combined.jsonl", "w") as f:
        for item in log:
            f.write(json.dumps(item))
            f.write("\n")

def get_episodes_from_form_ids():
    with open("../sotopia_episodes_v1.jsonl", "r") as f:
        episodes = [json.loads(line) for line in f]

    with open("../data/form_ids.txt", "r") as f:
        form_ids = f.readlines()

    form_ids = [form_id.strip() for form_id in form_ids]

    example_episodes = []
    visited = set()
    for form_id in tqdm(form_ids):
        form = get_form(form_id)
        episode_id = form["info"]["title"].split(" ")[-1]
        for episode in episodes:
            if episode["episode_id"] == episode_id and episode_id not in visited:
                visited.add(episode_id)
                example_episodes.append(episode)
                break

    with open("../data/example_episodes.jsonl", "w") as f:
        for episode in example_episodes:
            f.write(json.dumps(episode) + "\n")

    example_episodes_with_scores = extract_goal_scores(example_episodes)
    with open("../data/example_episodes_with_scores.jsonl", "w") as f:
        for episode in example_episodes_with_scores:
            f.write(json.dumps(episode) + "\n")

def format_goals(text: str) -> str:
    """Format the goals text by removing unwanted markers and tags."""
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("<extra_info>", "\n\n")
    text = text.replace("</extra_info>", "")
    text = text.replace("<strategy_hint>", "\n\n")
    text = text.replace("</strategy_hint>", "")
    return text

def creat_forms() -> None:
    with open("openai_log_reward_attribution.jsonl", "r") as f:
        all_data: List[Dict[str, Any]] = [json.loads(line) for line in f]

    form_uris: List[Dict[str, str]] = []

    for data in all_data:
        requests: List[Dict[str, Any]] = []
        service = authenticate_google_services()

        form_body: Dict[str, Any] = {
            "info": {
                "title": f"Utterance Reward Annotation {data['agent']} {data['episode_id']}",
            }
        }
        form = service.forms().create(body=form_body).execute()
        formId: str = form["formId"]
        responderUri: str = form["responderUri"]
        print("Form created with ID:", formId)
        print("Access your form at:", responderUri)
        form_uris.append({"formId": formId, "responderUri": responderUri})

        update_request: Dict[str, Any] = {
            "updateFormInfo": {
                "info": {
                    "title": f"Utterance Reward Attribution {data['agent']} {data['episode_id']}",
                    "description": (
                        "For this task, you will receive the dialogue history between two conversational agents, the social goal of one of the agents, and the final reward score received by this agent. Your objective is to assess how much each of the agent's utterance (marked by the agent's name and the utterance number) contributed to the final reward score. \n\nPlease provide a score between 0 and 3 for each of the utterances made by {data['agent']}. If you believe an utterance had no impact on the final reward score, please provide a score of 0. If you believe an utterance had a significant impact on the final reward score, please provide a score of 3. If you believe an utterance had a moderate impact on the final reward score, please provide a score of 5. You can provide any score between 0 and 10 based on your judgment."
                    ),
                },
                "updateMask": "*",
            }
        }
        requests.append(deepcopy(update_request))

        index: int = 1
        for utterance, details in data["rewarded_utterances"].items():
            quotes_text = re.findall(r'"([^"]*)"', details[0])
            action_text = re.findall(r"\[.*?\].*", details[0])
            full_text = ""
            if quotes_text:
                full_text = f"{utterance}: {quotes_text[0]} "
            elif action_text:
                full_text += f"{utterance}: {action_text[0]} "
            else:
                full_text = f"{utterance}: {details[0]} "

            if data["agent"] in details[0]:  # Only add if reward is not -1
                request: Dict[str, Any] = {
                    "createItem": {
                        "item": {
                            "title": full_text,
                            "questionItem": {
                                "question": {
                                    "scaleQuestion": {
                                        "low": 0,
                                        "high": 3,
                                        "lowLabel": "No Attribution",
                                        "highLabel": "Significant Attribution",
                                    },
                                    "required": True,
                                }
                            },
                        },
                        "location": {"index": index},
                    }
                }
            else:
                request = {
                    "createItem": {
                        "item": {"title": full_text, "textItem": {}},
                        "location": {"index": index},
                    }
                }
            requests.append(deepcopy(request))
            index += 1

        if requests:
            update_body: Dict[str, Any] = {"requests": requests}
            service.forms().batchUpdate(
                formId=formId, body=update_body
            ).execute()
            print("Questions added to the form.")

    with open("form_uris.jsonl", "w") as f:
        for form_uri in form_uris:
            f.write(json.dumps(form_uri) + "\n")
        print("Form URIs written to form_uris.jsonl")