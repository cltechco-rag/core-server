from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from core.database import Base

class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    language = Column(String(10), default="ko")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # 🔥 관계 설정
    video = relationship("Video", back_populates="transcript")

    def to_dict(self):
        return {
            "id": self.id,
            "video_id": self.video_id,
            "content": self.content,
            "summary": self.summary,
            "language": self.language,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
