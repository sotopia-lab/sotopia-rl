from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
import json

def authenticate_google_services():
    """Authenticate and return Google service client."""
    scopes = ['https://www.googleapis.com/auth/forms.responses.readonly', 'https://www.googleapis.com/auth/drive.readonly']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scopes)
    service = build('forms', 'v1', credentials=credentials)
    return service

def get_form(form_id):
    """Fetch and return a specified Google Form."""
    service = authenticate_google_services()
    # Fetch the form
    form = service.forms().get(formId=form_id).execute()
    return form

def get_form_responses(form_id):
    """Fetch and return all responses from a specified Google Form."""
    service = authenticate_google_services()
    # Fetch the form responses
    results = service.forms().responses().list(formId=form_id).execute()
    responses = results.get('responses', [])
    return responses

def print_responses(responses):
    """Print the responses nicely formatted."""
    print(json.dumps(responses, indent=4))

def add_responses_to_sheet(log, form_uris):
    """Add responses to rewared attribution log."""
    for i in range(len(log)):
        print(f"Log: {i}")
        form_id = form_uris[i]["formId"]
        form_schema = get_form(form_id)
        responses = get_form_responses(form_id)
        # print(f"Log: {log[i]}")
        print(f"Form ID: {form_id}")
        # print(f"Form schema: {form_schema}")
        # print(f"Responses: {responses}")
        for key in log[i]['rewarded_utterances']:
            print(f"  Key: {key}")
            for item in form_schema['items']:
                if item['title'].split(":")[0] == key:
                    break
            if 'questionItem' in item:
                question_id = item['questionItem']['question']['questionId']
            else:
                print("  No question item found")
                continue
            for response in responses:
                response_id = response['responseId']
                for _, response_item in response['answers'].items():
                    response_question_id = response_item['questionId']
                    if response_question_id == question_id:
                        print(f"  Response ID: {response_id}")
                        if len(log[i]['rewarded_utterances'][key]) == 2:
                            log[i]['rewarded_utterances'][key].append({})
                        log[i]['rewarded_utterances'][key][2].update({response_id: int(response_item['textAnswers']['answers'][0]['value'])})
                        break
        print(f"Updated log")
    return log

if __name__ == "__main__":
    # form_id = '1q_rV-4pvw78o_zAmsyjitMR7R7KO-EsJpRDBQFBVxY8'  # Replace 'your-form-id-here' with your actual Google Form ID
    # responses = get_form_responses(form_id)
    # print(get_form(form_id))
    # print_responses(responses)
    with open("openai_log_reward_attribution.jsonl", "r") as f:
        log = [json.loads(line) for line in f]
    
    with open("form_uris.jsonl", "r") as f:
        form_uris = [json.loads(line) for line in f]
    
    log = add_responses_to_sheet(log, form_uris)
    
    with open("reward_attribution_log_combined.jsonl", "w") as f:
        for item in log:
            f.write(json.dumps(item))
            f.write("\n")