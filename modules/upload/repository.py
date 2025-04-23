from typing import Optional
from sqlalchemy.orm import Session
from models.video import Video
from models.transcript import Transcript
from .schema import VideoProcessingStatus
from .constants import ProcessingStatus


class UploadRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_video_metadata(
        self, title: str, file_path: str, user_id: int, status: str = None
    ) -> Video:
        video = Video(title=title, file_path=file_path, user_id=user_id)
        self.db.add(video)
        self.db.flush()
        self.db.refresh(video)
        return video

    def get_video_status(self, video_id: int) -> Optional[VideoProcessingStatus]:
        video = self.db.query(Video).filter(Video.id == video_id).first()
        if not video:
            return None

        # 트랜스크립트 여부로 처리 상태 판단
        has_transcript = (
            video.transcript is not None and video.transcript.content is not None
        )
        status = (
            ProcessingStatus.COMPLETED
            if has_transcript
            else ProcessingStatus.PROCESSING
        )

        return VideoProcessingStatus(
            video_id=str(video.id),
            status=status,
            progress=100 if status == ProcessingStatus.COMPLETED else 0,
            created_at=video.created_at,
            updated_at=video.updated_at,
        )

    def create_video(self, title: str, file_path: str) -> Video:
        video = Video(title=title, file_path=file_path)
        self.db.add(video)
        self.db.flush()
        self.db.refresh(video)
        return video

    def get_video_by_id(self, video_id: int) -> Video:
        return self.db.query(Video).filter(Video.id == video_id).first()

    def get_transcript_by_video_id(self, video_id: int) -> Optional[Transcript]:
        return self.db.query(Transcript).filter(Transcript.video_id == video_id).first()

    def update_video(self, video_id: int, **kwargs) -> Optional[Video]:
        video = self.get_video_by_id(video_id)
        if not video:
            return None

        for key, value in kwargs.items():
            if hasattr(video, key):
                setattr(video, key, value)

        self.db.flush()
        self.db.refresh(video)
        return video

    def update_video_transcript(self, video_id: int, transcript_content: str) -> Video:
        video = self.get_video_by_id(video_id)
        if not video:
            return None

        # 이미 트랜스크립트가 있으면 업데이트, 없으면 생성
        if video.transcript:
            video.transcript.content = transcript_content
        else:
            transcript = Transcript(video_id=video_id, content=transcript_content)
            self.db.add(transcript)

        self.db.flush()
        self.db.refresh(video)
        return video
