from pydantic import BaseModel
from typing import Optional

class NotionAuthCode(BaseModel):
    code: str

class NotionTokenResponse(BaseModel):
    access_token: str
    token_type: str
    bot_id: str
    workspace_name: str
    workspace_icon: Optional[str]
    workspace_id: str

class UserResponse(BaseModel):
    email: str
    name: Optional[str]
    access_token: str 