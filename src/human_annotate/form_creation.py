import os
import json
from typing import Any, Dict, List
from copy import deepcopy
import re

from src.human_annotate.google_form_apis import authenticate_google_services

GoogleResource = Any

def format_goals(text: str) -> str:
    """Format the goals text by removing unwanted markers and tags."""
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("<extra_info>", "\n\n")
    text = text.replace("</extra_info>", "")
    text = text.replace("<strategy_hint>", "\n\n")
    text = text.replace("</strategy_hint>", "")
    return text

def create_forms(data_dir, gcp_key) -> None:
    with open(os.path.join(data_dir, "openai_log_attribution.jsonl"), "r") as f:
        all_data: List[Dict[str, Any]] = [json.loads(line) for line in f]

    form_uris: List[Dict[str, str]] = []

    for data in all_data:
        requests: List[Dict[str, Any]] = []
        service = authenticate_google_services(gcp_key)

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
        
        # Form title and instruction
        update_request: Dict[str, Any] = {
            "updateFormInfo": {
                "info": {
                    "title": f"Utterance Reward Attribution {data['agent']} {data['episode_id']}",
                    "description": 
                        "For this task, you will receive the dialogue history between two conversational agents, the social goal of one of the agents, and the final reward score received by this agent. Your objective is to assess how much each of the agent's utterance (marked by the agent's name and the utterance number) contributed to the final reward score. \n\nPlease provide a score between 0 and 3 for each of the utterances made by {data['agent']}. If you believe an utterance had no impact on the final reward score, please provide a score of 0. If you believe an utterance had a significant impact on the final reward score, please provide a score of 3. If you believe an utterance had a moderate impact on the final reward score, please provide a score of 5. You can provide any score between 0 and 10 based on your judgment."
                },
                "updateMask": "*",
            }
        }
        requests.append(deepcopy(update_request))
        
        # Add background information
        background_request = {
            'createItem': {
                'item': {
                    'title': "Background Information",
                    'description': (
                        "\n"
                        f"Episode ID:\n{data['episode_id']}\n\n"
                        f"Scenario:\n{data['scenario']}\n\n"
                        f"Agent:\n{data['agent']}\n\n"
                        f"Goal:\n{format_goals(data['goal'])}\n\n"
                        f"Final Reward Score:\n{data['goal_score']}\n\n"
                        f"Is First Speaker:\n{data['is_first_speaker']}"
                    ),
                    'textItem': {
                    }
                },
                'location': {
                    'index': 0
                }
            }
        }
        requests.append(background_request)
        
        # Add question about the user's name
        request: Dict[str, Any] = {
            "createItem": {
                "item": {
                    "title": "Please enter your name",
                    "questionItem": {
                        "question": {
                            "textQuestion": {
                                "paragraph": False
                            },
                            "required": True,
                        }
                    },
                },
                "location": {"index": 1},
            }
        }
        requests.append(deepcopy(request))
        
        # Add questions for each utterance
        index: int = 2
        for utterance, details in data["attributed_utterances"].items():
            quotes_text = re.findall(r'"([^"]*)"', details[0])
            action_text = re.findall(r"\[.*?\].*", details[0])
            full_text = ""
            if quotes_text:
                full_text = f"{utterance}: {quotes_text[0]} "
            elif action_text:
                full_text += f"{utterance}: {action_text[0]} "
            else:
                full_text = f"{utterance}: {details[0]} "
            
            # Add question for each utterance
            if data["agent"] in utterance:
                request: Dict[str, Any] = {
                    "createItem": {
                        "item": {
                            "title": full_text,
                            "questionItem": {
                                "question": {
                                    "choiceQuestion": {
                                        "type": "RADIO",
                                        "options": [
                                            {"value": "-1"},
                                            {"value": "0"},
                                            {"value": "1"},
                                            {"value": "2"},
                                            {"value": "3"}
                                            ],
                                    },
                                    "required": True,
                                }
                            },
                        },
                        "location": {"index": index},
                    }
                }
                requests.append(deepcopy(request))
                index += 1
                
                # Add question for key utterance
                request: Dict[str, Any] = {
                    "createItem": {
                        "item": {
                            "title": "Was this utterance a key utterance?",
                            "questionItem": {
                                "question": {
                                    "choiceQuestion": {
                                        "type": "RADIO",
                                        "options": [
                                            {"value": "No"},
                                            {"value": "Yes"},
                                            ],
                                    },
                                    "required": True,
                                }
                            },
                        },
                        "location": {"index": index},
                    }
                }
                requests.append(deepcopy(request))
                index += 1
            else:
                request: Dict[str, Any] = {
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

    with open(os.path.join(data_dir, "form_uris.jsonl"), "w") as f:
        for form_uri in form_uris:
            f.write(json.dumps(form_uri) + "\n")
        print("Form URIs written to form_uris.jsonl")