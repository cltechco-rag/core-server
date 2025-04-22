from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from .service import OpenAIService
from .schema import SummaryResponse, ErrorResponse, TranscriptResponse
from typing import List

router = APIRouter(prefix="/openai", tags=["openai"])

@router.get("/transcripts", response_model=List[TranscriptResponse])
async def get_all_transcripts():
    """모든 트랜스크립트를 조회합니다."""
    service = OpenAIService()
    try:
        transcripts = service.get_all_transcripts()
        return transcripts
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e),
                message="트랜스크립트 조회 중 오류가 발생했습니다."
            ).dict()
        )

@router.get("/transcripts/{video_id}", response_model=TranscriptResponse)
async def get_transcript(video_id: int):
    """비디오 ID로 트랜스크립트를 조회합니다."""
    service = OpenAIService()
    try:
        transcript = service.get_transcript(video_id)
        return transcript
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e),
                message="트랜스크립트 조회 중 오류가 발생했습니다."
            ).dict()
        )

@router.post("/summarize/{video_id}", response_model=SummaryResponse)
async def summarize_transcript(video_id: int):
    """비디오 ID로 트랜스크립트를 조회하여 요약합니다."""
    service = OpenAIService()
    try:
        summary = service.summarize_transcript(video_id)
        return SummaryResponse(video_id=video_id, summary=summary)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e),
                message="회의록 요약 중 오류가 발생했습니다."
            ).dict()
        ) 