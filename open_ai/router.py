from sqlalchemy.orm import Session
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from core.database import get_db
from .service import OpenAIService
from .schema import SummaryResponse, ErrorResponse, TranscriptResponse
from typing import List

router = APIRouter(prefix="/openai", tags=["openai"])


def get_openai_service(db: Session = Depends(get_db)) -> OpenAIService:
    """OpenAIService 의존성 주입을 위한 함수"""
    return OpenAIService(db)


@router.get("/transcripts", response_model=List[TranscriptResponse])
async def get_all_transcripts(
    service: OpenAIService = Depends(get_openai_service), db: Session = Depends(get_db)
):
    """모든 트랜스크립트를 조회합니다."""
    try:
        transcripts = service.get_all_transcripts()
        db.commit()
        return transcripts
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e), message="트랜스크립트 조회 중 오류가 발생했습니다."
            ).dict(),
        )


@router.get("/transcripts/{video_id}", response_model=TranscriptResponse)
async def get_transcript(
    video_id: int,
    service: OpenAIService = Depends(get_openai_service),
):
    """비디오 ID로 트랜스크립트를 조회합니다."""
    try:
        transcript = service.get_transcript(video_id)
        return transcript
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e), message="트랜스크립트 조회 중 오류가 발생했습니다."
            ).dict(),
        )


@router.post("/summarize/{video_id}", response_model=SummaryResponse)
async def summarize_transcript(
    video_id: int,
    db: Session = Depends(get_db),
    service: OpenAIService = Depends(get_openai_service),
):
    """비디오 ID로 트랜스크립트를 조회하여 요약합니다."""
    try:
        summary = service.summarize_transcript(video_id)
        db.commit()
        return SummaryResponse(video_id=video_id, summary=summary)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=str(e), message="회의록 요약 중 오류가 발생했습니다."
            ).dict(),
        )
