from typing import Any, Dict, List

from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials

# Use a generic type for Google service resources
GoogleResource = Any


def authenticate_google_services(gcp_key: str) -> GoogleResource:
    """Authenticate and return Google service client."""
    scopes = [
        "https://www.googleapis.com/auth/forms",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        gcp_key, scopes
    )
    service = build("forms", "v1", credentials=credentials)
    return service


def get_form(form_id: str, gcp_key: str) -> Dict[str, Any] | Any:
    """Fetch and return a specified Google Form."""
    service = authenticate_google_services(gcp_key)
    form = service.forms().get(formId=form_id).execute()
    return form


def get_form_responses(
    form_id: str, gcp_key: str
) -> List[Dict[str, Any]] | Any:
    """Fetch and return all responses from a specified Google Form."""
    service = authenticate_google_services(gcp_key)
    results = service.forms().responses().list(formId=form_id).execute()
    responses = results.get("responses", [])
    return responses
