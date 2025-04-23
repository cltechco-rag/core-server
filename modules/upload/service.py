from fastapi import UploadFile, BackgroundTasks
from config.config import settings
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .repository import UploadRepository
from .constants import ProcessingStatus
from sqlalchemy.orm import Session
from utils.stt_processor import STTProcessorParallel
from open_ai.service import OpenAIService
from models.video import Video
import os
import logging
import time
from typing import Dict, Any, Set
import aiofiles

logger = logging.getLogger(__name__)

# 백그라운드 작업 추적을 위한 전역 딕셔너리
background_tasks_status: Dict[int, Dict[str, Any]] = {}

# 취소 요청된 작업 ID 세트
cancelled_tasks: Set[int] = set()


class UploadService:
    def __init__(self, db: Session):
        self.repository = UploadRepository(db)
        self.openai_service = OpenAIService(db)
        self.db = db

    async def process_video_upload(
        self, file: UploadFile, title: str, background_tasks: BackgroundTasks, user_id: int
    ):
        # 파일 저장 경로 생성
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        # 파일 저장
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 메타데이터 저장
        video = self.repository.save_video_metadata(title=title, file_path=file_path, user_id=user_id)

        # 백그라운드 작업 상태 초기화
        video_id = video.id
        background_tasks_status[video_id] = {
            "status": "pending",
            "start_time": None,
            "progress": 0,
            "last_update": time.time(),
            "log_messages": ["작업이 대기열에 추가됨"],
        }

        # 백그라운드 작업으로 STT 처리 등록
        background_tasks.add_task(
            self._process_video_transcript, video_id=video_id, file_path=file_path
        )

        return {"video_id": str(video_id)}

    def clean_transcript(self, raw_transcript: str) -> str:
        # Initialize the LLM
        llm = AzureChatOpenAI(
            temperature=0,
            deployment_name=settings.AZURE_CHAT_DEPLOYMENT,
            openai_api_version=settings.AZURE_CHAT_API_VERSION,
            openai_api_key=settings.AZURE_CHAT_API_KEY,
            azure_endpoint=settings.AZURE_CHAT_ENDPOINT,
        )

        # Split transcript into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=0)
        chunks = splitter.split_text(raw_transcript)

        def make_prompt(text: str) -> list:
            return [
                HumanMessage(
                    content=f"""
                    You are a transcription editor. The following text is a transcript of a lecture.
                    Do not summarize or paraphrase. Only fix transcription errors, grammar, and punctuation.
                    Do not remove repetitions unless clearly wrong.
                    Only return the cleaned text, do not add any additional information.

                    Transcript:
                    \"\"\"{text}\"\"\"
                    """
                )
            ]

        # Process each chunk
        cleaned_chunks = []
        for idx, chunk in enumerate(chunks):
            response = llm(make_prompt(chunk))
            cleaned_chunks.append(response.content)
            print(f"Chunk {idx + 1}/{len(chunks)} processed.")

        return "\n\n".join(cleaned_chunks)


    async def process_transcript_upload(
        self, file: UploadFile, title: str, background_tasks: BackgroundTasks
    ):
        # 파일 저장 경로 생성
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        # 파일 저장
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 메타데이터 저장
        video = self.repository.save_video_metadata(title=title, file_path=file_path)
        background_tasks.add_task(
            self.clean_and_upload_transcript, video_id=video.id, raw_transcript=content
        )
        return {"video_id": str(video.id)}

    def clean_and_upload_transcript(self, video_id: int, raw_transcript: str):
        cleaned_transcript = self.clean_transcript((raw_transcript.decode("utf-8")))
        self.repository.update_video_transcript(video_id, cleaned_transcript)
        self.openai_service.summarize_transcript(video_id)

    def _process_video_transcript(self, video_id: int, file_path: str):
        """
        비디오 파일을 처리하여 텍스트로 변환하는 백그라운드 작업
        """
        try:
            # 작업 시작 상태 업데이트
            background_tasks_status[video_id]["status"] = "processing"
            background_tasks_status[video_id]["start_time"] = time.time()
            background_tasks_status[video_id]["log_messages"].append(
                f"[STT] 비디오 처리 시작 (ID: {video_id})"
            )

            logger.info(f"[STT] 비디오 처리 시작 (ID: {video_id}, 파일: {file_path})")

            # 취소 확인
            if video_id in cancelled_tasks:
                raise ValueError("작업이 사용자에 의해 취소되었습니다.")

            # 파일 크기 확인
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            log_msg = f"[STT] 파일 크기: {file_size_mb:.2f} MB"
            logger.info(log_msg)
            background_tasks_status[video_id]["log_messages"].append(log_msg)
            background_tasks_status[video_id]["progress"] = 10

            # 취소 확인
            if video_id in cancelled_tasks:
                raise ValueError("작업이 사용자에 의해 취소되었습니다.")

            # STT 처리
            log_msg = f"[STT] 오디오 추출 및 변환 처리 시작..."
            logger.info(log_msg)
            background_tasks_status[video_id]["log_messages"].append(log_msg)
            background_tasks_status[video_id]["progress"] = 20

            # STTProcessorParallel을 직접 사용하여 STT 처리 실행
            processor = STTProcessorParallel(model_name="medium", num_workers=4)

            # 임시 출력 디렉토리 설정
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)

            # STT 처리 실행
            transcription_result = processor.process_video_to_text(
                file_path, output_dir=temp_dir
            )

            # 취소 확인
            if video_id in cancelled_tasks:
                raise ValueError("작업이 사용자에 의해 취소되었습니다.")

            # 전체 텍스트 추출
            full_text = transcription_result["text"]
            log_msg = f"[STT] 텍스트 변환 완료: {len(full_text)} 문자"
            logger.info(log_msg)
            background_tasks_status[video_id]["log_messages"].append(log_msg)
            background_tasks_status[video_id]["progress"] = 80

            # 데이터베이스에 저장
            self.repository.update_video_transcript(video_id, full_text)

            log_msg = f"[STT] 비디오 처리 완료 (ID: {video_id})"
            logger.info(log_msg)
            background_tasks_status[video_id]["log_messages"].append(log_msg)
            background_tasks_status[video_id]["status"] = "completed"
            background_tasks_status[video_id]["progress"] = 100
            background_tasks_status[video_id]["last_update"] = time.time()

            # OpenAI 서비스를 사용하여 트랜스크립트 요약 실행
            try:
                self.openai_service.summarize_transcript(video_id)
                log_msg = f"[요약] 트랜스크립트 요약 완료 (ID: {video_id})"
                self.db.commit()
                logger.info(log_msg)
                background_tasks_status[video_id]["log_messages"].append(log_msg)
            except Exception as e:
                error_msg = f"[요약] 트랜스크립트 요약 실패 (ID: {video_id}): {str(e)}"
                logger.error(error_msg, exc_info=True)
                background_tasks_status[video_id]["log_messages"].append(error_msg)

            # 취소 목록에서 제거 (만약 있다면)
            if video_id in cancelled_tasks:
                cancelled_tasks.remove(video_id)

        except Exception as e:
            error_msg = f"[STT] 비디오 처리 실패 (ID: {video_id}): {str(e)}"
            logger.error(error_msg, exc_info=True)
            background_tasks_status[video_id]["status"] = "failed"
            background_tasks_status[video_id]["log_messages"].append(error_msg)
            background_tasks_status[video_id]["last_update"] = time.time()

            # 취소 목록에서 제거 (만약 있다면)
            if video_id in cancelled_tasks:
                cancelled_tasks.remove(video_id)

    def get_processing_status(self, video_id: int):
        status = self.repository.get_video_status(video_id)
        if not status:
            raise ValueError(f"Video with id {video_id} not found")
        return status

    def get_background_task_status(self, video_id: int) -> Dict[str, Any]:
        """백그라운드 작업 상태를 반환합니다."""
        if video_id not in background_tasks_status:
            return {
                "video_id": video_id,
                "status": "unknown",
                "message": "작업 상태를 찾을 수 없습니다.",
            }

        result = {"video_id": video_id, **background_tasks_status[video_id]}

        # 취소 요청 상태 추가
        result["cancel_requested"] = video_id in cancelled_tasks

        # 경과 시간 계산
        if result["start_time"]:
            result["elapsed_seconds"] = time.time() - result["start_time"]

        return result

    def cancel_background_task(self, video_id: int) -> Dict[str, Any]:
        """백그라운드 작업 취소를 요청합니다."""
        if video_id not in background_tasks_status:
            raise ValueError(f"Video with id {video_id} not found")

        status = background_tasks_status[video_id]["status"]
        if status == "completed" or status == "failed":
            return {
                "video_id": video_id,
                "success": False,
                "message": f"작업이 이미 {status} 상태입니다. 취소할 수 없습니다.",
            }

        # 취소 요청 등록
        cancelled_tasks.add(video_id)
        background_tasks_status[video_id]["log_messages"].append(
            "사용자에 의한 취소 요청됨"
        )

        return {
            "video_id": video_id,
            "success": True,
            "message": "작업 취소가 요청되었습니다. 진행 중인 단계가 완료된 후 취소됩니다.",
        }

    async def upload_video(self, file: UploadFile, title: str, user_id: int) -> Video:
        # 파일 저장 경로 생성
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)

        # 파일 저장
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # DB에 비디오 정보 저장
        db_video = Video(
            title=title,
            file_path=file_path,
            user_id=user_id
        )
        self.db.add(db_video)
        self.db.commit()
        self.db.refresh(db_video)

        return db_video
