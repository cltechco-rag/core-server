from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class TranscriptResponse(BaseModel):
    id: int
    video_id: int
    content: Optional[str]
    summary: Optional[str]
    language: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True  # SQLAlchemy 모델을 Pydantic 모델로 변환할 때 필요

class SummaryResponse(BaseModel):
    video_id: int
    summary: str

class ErrorResponse(BaseModel):
    error: str
    message: str 