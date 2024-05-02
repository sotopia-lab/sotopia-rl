import json
import re
from copy import deepcopy
from html import escape

from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials

# Prepare questions for batchUpdate
with open("openai_log_reward_attribution.jsonl", "r") as f:
    all_data = [json.loads(line) for line in f]

form_uris = []

for data in all_data:
    requests = []
    # Authentication and service creation
    scopes = [
        "https://www.googleapis.com/auth/forms",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        "credentials.json", scopes
    )
    service = build("forms", "v1", credentials=credentials)

    # Create a new form with just the title
    form_body = {
        "info": {
            "title": f"Utterance Reward Annotation {data['agent']} {data['episode_id']}",
        }
    }
    form = service.forms().create(body=form_body).execute()
    formId = form["formId"]
    responderUri = form["responderUri"]
    print("Form created with ID:", formId)
    print("Access your form at:", responderUri)
    form_uris.append({"formId": formId, "responderUri": responderUri})

    update_request = {
        "updateFormInfo": {
            "info": {
                "title": f"Utterance Reward Attribution {data['agent']} {data['episode_id']}",
                "description": f"""For this task, you will receive the dialogue history between two conversational agents, the social goal of one of the agents, and the final reward score recieved by this agent. Your objective is to assess how much each of the agent's utterance (marked by the agent's name and the utterance number) contributed to the final reward score. \n\nPlease provide a score between 0 and 3 for each of the utterances made by {data['agent']}. If you believe an utterance had no impact on the final reward score, please provide a score of 0. If you believe an utterance had a significant impact on the final reward score, please provide a score of 3. If you believe an utterance had a moderate impact on the final reward score, please provide a score of 5. You can provide any score between 0 and 10 based on your judgment.

                """,
            },
            "updateMask": "*",
        }
    }
    requests.append(deepcopy(update_request))

    def format_goals(text):
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.replace("<extra_info>", "\n\n")
        text = text.replace("</extra_info>", "")
        text = text.replace("<strategy_hint>", "\n\n")
        text = text.replace("</strategy_hint>", "")
        return text

    background_request = {
        "createItem": {
            "item": {
                "title": "Background Information",
                "description": (
                    "\n"
                    f"Episode ID:\n{data['episode_id']}\n\n"
                    f"Scenario:\n{data['scenario']}\n\n"
                    f"Agent:\n{data['agent']}\n\n"
                    f"Goal:\n{format_goals(data['goal'])}\n\n"
                    f"Final Reward Score:\n{data['score']}\n\n"
                    f"Is First Speaker:\n{data['is_first_speaker']}"
                ),
                "textItem": {},
            },
            "location": {"index": 0},
        }
    }
    requests.append(background_request)

    index = 1
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
            request = {
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

    # Use batchUpdate to add questions
    if requests:
        update_body = {"requests": requests}
        service.forms().batchUpdate(formId=formId, body=update_body).execute()
        print("Questions added to the form.")

with open("form_uris.jsonl", "w") as f:
    for form_uri in form_uris:
        f.write(json.dumps(form_uri) + "\n")
    print("Form URIs written to form_uris.jsonl")
