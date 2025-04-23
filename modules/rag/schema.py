from typing import Optional
from pydantic import BaseModel
from datetime import datetime
from config.config import settings


class RAGPreprocessingRequest(BaseModel):
    video_id: int


class RAGPreprocessingResponse(BaseModel):
    message: str
    video_id: int
    status: str
    created_at: datetime = datetime.now()


class SearchRequest(BaseModel):
    query: str
    content_vector_weight: float = settings.DEFAULT_CONTENT_VECTOR_WEIGHT
    title_vector_weight: float = settings.DEFAULT_TITLE_VECTOR_WEIGHT
    bm25_weight: float = settings.DEFAULT_BM25_WEIGHT
    top_k: Optional[int] = 10
    rerank: Optional[bool] = False


class AnswerResponse(BaseModel):
    answer: str
    query_time: float
    total_documents: int
