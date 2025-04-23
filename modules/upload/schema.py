from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class VideoUploadResponse(BaseModel):
    message: str
    video_id: str
    video_path: str
    status: str
    created_at: datetime = datetime.now()


class VideoProcessingStatus(BaseModel):
    video_id: str
    status: str  # PENDING, PROCESSING, COMPLETED, FAILED
    progress: float  # 0 to 100
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
