from sqlalchemy.orm import Session
from openai import AzureOpenAI
import logging
from .repository import OpenAIRepository
from typing import List
from models.transcript import Transcript
from config.config import settings

logger = logging.getLogger(__name__)


class OpenAIService:
    def __init__(self, db: Session):
        self.repository = OpenAIRepository(db)

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
                api_key=settings.AZURE_CHAT_API_KEY,
                api_version=settings.AZURE_CHAT_API_VERSION,
                azure_endpoint=settings.AZURE_CHAT_ENDPOINT,
            )

            system_prompt = """
                                당신은 전문적인 회의록 요약 어시스턴트입니다.
                                다음 지침에 따라 회의 내용을 요약해주세요:

                                1. 회의의 주요 주제와 핵심 결정사항을 먼저 제시
                                2. 중요한 논의 사항들을 시간 순서대로 정리
                                3. 주요 참석자들의 핵심 의견이나 제안 포함
                                4. 향후 조치사항이나 후속 단계가 있다면 별도로 정리
                                5. 회의에서 나온 중요한 수치나 데이터는 반드시 포함

                                요약은 다음 형식으로 구성해주세요:

                                [회의 핵심 요약]
                                - 주요 결정사항
                                - 핵심 논의 내용

                                [상세 내용]
                                1. 주요 논의사항
                                2. 참석자 의견
                                3. 후속 조치사항

                                * 전문 용어나 특정 맥락이 있는 내용은 가능한 유지해주세요.
                                * 명확한 문장으로 간단명료하게 작성해주세요.
                            """

            response = client.chat.completions.create(
                model=settings.AZURE_CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=0.3,  # 더 일관된 출력을 위해 temperature 조정
                max_tokens=1500,  # 더 긴 요약을 위해 토큰 수 증가
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
