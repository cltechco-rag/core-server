from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from core.database import get_db
from .service import VideoService
from schemas.video import Video, VideoCreate
from utils.auth import get_current_user
from models.user import User

router = APIRouter()

@router.post("/", response_model=Video)
def create_video(video: VideoCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    video.user_id = current_user.id
    return VideoService(db).create_video(video)

@router.get("/{video_id}", response_model=Video)
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = VideoService(db).get_video(video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

@router.get("/user/list", response_model=List[Video])
def get_user_videos(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return VideoService(db).get_videos_by_user(current_user.id) 