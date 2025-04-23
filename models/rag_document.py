from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text, Index, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from core.database import Base


class RAGDocument(Base):
    __tablename__ = "rag_documents"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(
        Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    title = Column(String(255))
    content = Column(Text)
    tokenized_text = Column(Text)  # For BM25 search
    content_embedding = Column(Vector(3072))  # OpenAI text-embedding-3-large dimension
    title_embedding = Column(Vector(3072))  # OpenAI text-embedding-3-large dimension
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship to Video
    video = relationship("Video", back_populates="rag_documents")

    # Index for BM25 search
    __table_args__ = (
        # GIN index for full-text search on tokenized_text
        Index(
            "ix_documents_tokenized_text_gin",
            "tokenized_text",
            postgresql_using="gin",
            postgresql_ops={"tokenized_text": "gin_trgm_ops"},
        ),
    )

    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title}')>"
