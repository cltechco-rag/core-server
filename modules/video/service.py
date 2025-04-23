from sqlalchemy.orm import Session
from models.video import Video
from schemas.video import VideoCreate
from typing import List

class VideoService:
    def __init__(self, db: Session):
        self.db = db

    def create_video(self, video: VideoCreate) -> Video:
        db_video = Video(**video.dict())
        self.db.add(db_video)
        self.db.commit()
        self.db.refresh(db_video)
        return db_video

    def get_video(self, video_id: int) -> Video:
        return self.db.query(Video).filter(Video.id == video_id).first()

    def get_videos_by_user(self, user_id: int) -> List[Video]:
        return self.db.query(Video).filter(Video.user_id == user_id).all() 