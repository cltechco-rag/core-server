import os
from openai import AzureOpenAI
import logging
from .repository import OpenAIRepository
from typing import List
from models.transcript import Transcript

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        self.repository = OpenAIRepository()
        
    def get_transcript(self, video_id: int) -> Transcript:
        """비디오 ID로 트랜스크립트를 조회합니다."""
        transcript = self.repository.get_transcript_by_video_id(video_id)
        if not transcript:
            raise ValueError(f"트랜스크립트를 찾을 수 없습니다. (video_id: {video_id})")
        return transcript

    def get_all_transcripts(self) -> List[Transcript]:
        """모든 트랜스크립트를 조회합니다."""
        return self.repository.get_all_transcripts()

    def summarize_text(self, content: str) -> str:
        """텍스트 내용을 요약합니다."""
        try:
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )

            response = client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[
                    {"role": "system", "content": "당신은 회의록 요약 어시스턴트입니다. 내용을 요약해주세요."},
                    {"role": "user", "content": content}
                ],
                temperature=0.2,
                max_tokens=1000
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"텍스트 요약 중 오류 발생: {str(e)}", exc_info=True)
            raise

    def summarize_transcript(self, video_id: int) -> str:
        """비디오 ID로 트랜스크립트를 조회하여 요약합니다."""
        # 트랜스크립트 조회
        transcript = self.repository.get_transcript_by_video_id(video_id)
        if not transcript or not transcript.content:
            raise ValueError("요약할 트랜스크립트가 없습니다.")

        # 요약 생성
        summary = self.summarize_text(transcript.content)
        
        # 요약 내용 저장
        self.repository.update_transcript(video_id, summary)
        
        return summary 