from typing import Any, List

class ServiceAccountCredentials:
    def __init__(self, json_keyfile_name: str, scopes: List[str]): ...
    def from_json_keyfile_name(
        cls, filename: str, scopes: List[str]
    ) -> "ServiceAccountCredentials": ...
