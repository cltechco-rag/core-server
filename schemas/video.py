from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class VideoBase(BaseModel):
    title: str
    file_path: str
    user_id: int

class VideoCreate(VideoBase):
    pass

class Video(VideoBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True 