from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from core.database import Base


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    transcription_status = Column(String(20), nullable=False, default="PENDING")
    rag_status = Column(String(20), nullable=False, default="NOT_INCLUDED")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        CheckConstraint(
            transcription_status.in_(["SUCCESS", "FAILED", "PENDING"]),
            name="check_transcription_status",
        ),
        CheckConstraint(
            rag_status.in_(["INCLUDED", "PENDING", "NOT_INCLUDED"]),
            name="check_rag_status",
        ),
    )

    user = relationship("User", back_populates="videos")
    transcript = relationship(
        "Transcript",
        back_populates="video",
        uselist=False,
        cascade="all, delete-orphan",
    )
    rag_documents = relationship(
        "RAGDocument",
        back_populates="video",
        cascade="all, delete-orphan",
    )

    def to_dict(self):
        result = {
            "id": self.id,
            "title": self.title,
            "file_path": self.file_path,
            "user_id": self.user_id,
            "transcription_status": self.transcription_status,
            "rag_status": self.rag_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.transcript:
            result["transcript"] = self.transcript.content
        return result
