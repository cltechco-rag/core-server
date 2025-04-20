from fastapi import UploadFile, BackgroundTasks
import uuid
from .repository import UploadRepository
from .constants import ProcessingStatus
from utils.stt_processor import process_video_to_text
import os
import logging
import time
import threading
from typing import Dict, Any, Set

logger = logging.getLogger(__name__)

# 백그라운드 작업 추적을 위한 전역 딕셔너리
background_tasks_status: Dict[int, Dict[str, Any]] = {}

# 취소 요청된 작업 ID 세트
cancelled_tasks: Set[int] = set()

class UploadService:
    def __init__(self):
        self.repository = UploadRepository()

    async def process_video_upload(
        self,
        file: UploadFile,
        title: str,
        background_tasks: BackgroundTasks
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
        video = self.repository.save_video_metadata(
            title=title,
            file_path=file_path
        )
        
        # 백그라운드 작업 상태 초기화
        video_id = video.id
        background_tasks_status[video_id] = {
            "status": "pending",
            "start_time": None,
            "progress": 0,
            "last_update": time.time(),
            "log_messages": ["작업이 대기열에 추가됨"]
        }
        
        # 백그라운드 작업으로 STT 처리 등록
        background_tasks.add_task(
            self._process_video_transcript,
            video_id=video_id,
            file_path=file_path
        )
        
        return {"video_id": str(video_id)}

    def _process_video_transcript(self, video_id: int, file_path: str):
        """
        비디오 파일을 처리하여 텍스트로 변환하는 백그라운드 작업
        """
        try:
            # 작업 시작 상태 업데이트
            background_tasks_status[video_id]["status"] = "processing"
            background_tasks_status[video_id]["start_time"] = time.time()
            background_tasks_status[video_id]["log_messages"].append(f"[STT] 비디오 처리 시작 (ID: {video_id})")
            
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
            
            # STT 처리 실행
            transcription_result = process_video_to_text(file_path)
            
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
                "message": "작업 상태를 찾을 수 없습니다."
            }
        
        result = {
            "video_id": video_id,
            **background_tasks_status[video_id]
        }
        
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
                "message": f"작업이 이미 {status} 상태입니다. 취소할 수 없습니다."
            }
            
        # 취소 요청 등록
        cancelled_tasks.add(video_id)
        background_tasks_status[video_id]["log_messages"].append("사용자에 의한 취소 요청됨")
        
        return {
            "video_id": video_id,
            "success": True,
            "message": "작업 취소가 요청되었습니다. 진행 중인 단계가 완료된 후 취소됩니다."
        } 