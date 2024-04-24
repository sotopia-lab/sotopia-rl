from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
import json

def authenticate_google_services():
    """Authenticate and return Google service client."""
    scopes = ['https://www.googleapis.com/auth/forms.responses.readonly', 'https://www.googleapis.com/auth/drive.readonly']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scopes)
    service = build('forms', 'v1', credentials=credentials)
    return service

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

if __name__ == "__main__":
    form_id = '1q_rV-4pvw78o_zAmsyjitMR7R7KO-EsJpRDBQFBVxY8'  # Replace 'your-form-id-here' with your actual Google Form ID
    responses = get_form_responses(form_id)
    print_responses(responses)