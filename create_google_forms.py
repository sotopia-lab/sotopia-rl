from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from copy import deepcopy

# Authentication and service creation
scopes = ['https://www.googleapis.com/auth/forms', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scopes)
service = build('forms', 'v1', credentials=credentials)

# Create a new form with just the title
form_body = {
    "info": {
        "title": "Utterance Reward Annotation"
    }
}
form = service.forms().create(body=form_body).execute()
form_id = form['formId']
form_url = f"https://docs.google.com/forms/d/{form_id}/edit"
print("Form created with ID:", form_id)
print("Access your form at:", form_url)

# Prepare questions for batchUpdate
requests = []
data = {
    "rewarded_utterances": {
        "Utterance 1 by Benjamin Jackson": ["Some text here 1", 5],
        "Utterance 1 by Jason Qi": ["Some text here 2", -1],
        "Utterance 2 by Benjamin Jackson": ["Some text here 3", 5],
        "Utterance 2 by Jason Qi": ["Some text here 3", -1],
    }
}


index = 0
for utterance, details in data['rewarded_utterances'].items():
    full_text = utterance + ": " + details[0]
    if details[1] != -1:  # Only add if reward is not -1
        request = {
            'createItem': {
                'item': {
                    'title': full_text,
                    'questionItem': {
                        'question': {
                            'scaleQuestion': {
                                'low': 0,
                                'high': 3,
                                'lowLabel': "No Attribution",
                                'highLabel': "Significant Attribution",
                            },
                            'required': True
                        }
                    }
                },
                'location': {
                    'index': index
                }
            }
        }
    else:
        request = {
            'createItem': {
                'item': {
                    'title': full_text,
                    'textItem': {
                    }
                },
                'location': {
                    'index': index
                }
            }
        }
    requests.append(deepcopy(request))
    index += 1

# Use batchUpdate to add questions
if requests:
    update_body = {
        'requests': requests
    }
    service.forms().batchUpdate(formId=form_id, body=update_body).execute()
    print("Questions added to the form.")