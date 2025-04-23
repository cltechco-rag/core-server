from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    status,
    Depends,
    BackgroundTasks,
)
from sqlalchemy.orm import Session
from .service import UploadService
from .schema import VideoUploadResponse
from .constants import ALLOWED_VIDEO_TYPES
from typing import List
from models.video import Video
from core.database import SessionLocal
import logging
from models.user import User
from core.database import get_db
from utils.auth import get_current_user

router = APIRouter(prefix="/upload", tags=["upload"])


# 의존성 주입을 위한 함수
def get_upload_service():
    return UploadService()


@router.post(
    "/video",
    response_model=VideoUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="영상 파일 업로드",
    description="""
    영상 파일을 업로드하고 처리를 위한 큐에 등록합니다.

    ## 기능
    * 영상 파일 업로드 및 검증
    * 파일 메타데이터 저장
    * 처리 작업 큐 등록
    * STT(음성-텍스트 변환) 처리

    ## 제한사항
    * 허용된 파일 형식: MP4, AVI, MOV, WMV
    * 최대 파일 크기: 2GB

    ## 응답
    * video_id: 업로드된 영상의 고유 식별자
    * status: 처리 상태 (PROCESSING)
    * estimated_time: 예상 처리 시간
    """,
)
async def upload_video(
    title: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    영상 파일을 업로드하고 STT 처리를 시작합니다.
    """
    service = UploadService(db)
    return await service.process_video_upload(
        file=file,
        title=title,
        background_tasks=background_tasks,
        user_id=current_user.id
    )


@router.post("/transcript", summary="비디오 트랜스크립트 업로드")
async def upload_transcript(
    background_tasks: BackgroundTasks,
    title: str = Form(),
    file: UploadFile = File(...),
    service: UploadService = Depends(get_upload_service),
):
    """
    비디오 트랜스크립트를 업로드합니다.

    - title: 비디오 제목
    - file: 트랜스크립트 파일 (txt, srt 등)
    """
    try:
        # 1. 파일 형식 검증
        if not file.content_type in ["text/plain", "application/x-subrip"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Allowed types are: txt, srt",
            )

        # 2. 서비스 계층에 처리 위임
        result = await service.process_transcript_upload(
            title=title, file=file, background_tasks=background_tasks
        )

        return {
            "message": "Transcript upload successful. Processing has been queued.",
            "video_id": result["video_id"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/videos", summary="모든 비디오 목록 조회")
def get_all_videos():
    """데이터베이스에 저장된 모든 비디오 목록을 반환합니다."""
    db = SessionLocal()
    try:
        videos = db.query(Video).all()
        return [video.to_dict() for video in videos]
    finally:
        db.close()


@router.get("/videos/{video_id}", summary="특정 비디오 조회")
def get_video_by_id(video_id: str):
    """ID로 특정 비디오의 정보와 변환된 텍스트를 조회합니다."""
    from core.database import SessionLocal
    from models.video import Video

    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == int(video_id)).first()
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video with id {video_id} not found",
            )
        return video.to_dict()
    finally:
        db.close()


@router.get("/videos/{video_id}/stt-status", summary="STT 처리 상태 확인")
def get_stt_status(video_id: str):
    """ID로 특정 비디오의 STT 처리 상태를 조회합니다."""
    from core.database import SessionLocal
    from models.video import Video

    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == int(video_id)).first()
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Video with id {video_id} not found",
            )

        # transcript 필드로 처리 상태 판단
        if video.transcript:
            status = "완료"
            transcript_preview = (
                video.transcript[:200] + "..."
                if len(video.transcript) > 200
                else video.transcript
            )
        else:
            status = "처리 중"
            transcript_preview = None

        return {
            "video_id": str(video.id),
            "title": video.title,
            "processing_status": status,
            "transcript_preview": transcript_preview,
            "created_at": video.created_at,
            "updated_at": video.updated_at,
        }
    finally:
        db.close()


@router.get("/videos/{video_id}/background-task", summary="백그라운드 작업 상태 확인")
def get_background_task_status(
    video_id: str, service: UploadService = Depends(get_upload_service)
):
    """비디오 처리를 위한 백그라운드 작업의 상태를 확인합니다."""
    try:
        result = service.get_background_task_status(int(video_id))
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/videos/{video_id}/cancel", summary="백그라운드 작업 취소")
def cancel_background_task(
    video_id: str, service: UploadService = Depends(get_upload_service)
):
    """비디오 처리를 위한 백그라운드 작업을 취소합니다."""
    try:
        result = service.cancel_background_task(int(video_id))
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
