from sqlalchemy.orm import Session
from models.transcript import Transcript
from core.database import SessionLocal
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class OpenAIRepository:
    def __init__(self):
        self.db = SessionLocal()

    def get_transcript_by_video_id(self, video_id: int) -> Optional[Transcript]:
        """비디오 ID로 트랜스크립트를 조회합니다."""
        return self.db.query(Transcript).filter(Transcript.video_id == video_id).first()

    def get_all_transcripts(self) -> List[Transcript]:
        """모든 트랜스크립트를 조회합니다."""
        return self.db.query(Transcript).all()

    def update_transcript(self, video_id: int, summary: str) -> bool:
        """트랜스크립트에 요약 내용을 업데이트합니다."""
        try:
            transcript = self.get_transcript_by_video_id(video_id)
            if transcript:
                transcript.summary = summary
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(f"트랜스크립트 요약 업데이트 실패: {str(e)}", exc_info=True)
            return False 