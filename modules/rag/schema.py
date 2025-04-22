from typing import Optional
from pydantic import BaseModel
from datetime import datetime
from langchain.output_parsers import ResponseSchema


class RAGPreprocessingRequest(BaseModel):
    video_id: int


class RAGPreprocessingResponse(BaseModel):
    message: str
    video_id: int
    status: str
    created_at: datetime = datetime.now()


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    rerank: Optional[bool] = False


class AnswerResponse(BaseModel):
    answer: str
    query_time: float
    total_documents: int
