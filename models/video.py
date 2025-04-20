from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from core.database import Base

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 트랜스크립트와의 관계 설정
    transcript = relationship("Transcript", back_populates="video", uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        result = {
            "id": self.id,
            "title": self.title,
            "file_path": self.file_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
        
        # 트랜스크립트가 있으면 포함
        if self.transcript:
            result["transcript"] = self.transcript.content
            
        return result 