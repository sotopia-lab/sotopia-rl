# Contents of stubs/oauth2client/service_account/__init__.pyi

from typing import Any, Dict, List, Optional

class ServiceAccountCredentials:
    def __init__(
        self,
        service_account_email: str,
        signer: Any,
        scopes: List[str],
        user_agent: Optional[str] = None,
        token_uri: str = "https://oauth2.googleapis.com/token",
        revoke_uri: Optional[str] = None,
        **kwargs: Any
    ) -> None: ...
    @classmethod
    def from_json_keyfile_name(
        cls,
        filename: str,
        scopes: List[str],
        token_uri: Optional[str] = None,
        revoke_uri: Optional[str] = None,
    ) -> "ServiceAccountCredentials": ...
    @classmethod
    def from_json_keyfile_dict(
        cls,
        keyfile_dict: Dict[str, Any],
        scopes: List[str],
        token_uri: Optional[str] = None,
        revoke_uri: Optional[str] = None,
    ) -> "ServiceAccountCredentials": ...

# You might need to add more methods or classes depending on what your code uses.
