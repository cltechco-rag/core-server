import whisper
import os
from pathlib import Path
import logging
import ssl

# SSL 인증서 검증 우회 설정
ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger(__name__)

class STTProcessor:
    def __init__(self, model_name="medium"):
        """
        Speech-to-Text 프로세서 초기화
        
        Args:
            model_name: Whisper 모델 크기 (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.model = None
        
    def load_model(self):
        """모델을 메모리에 로드합니다."""
        if self.model is None:
            logger.info(f"Whisper {self.model_name} 모델을 로드합니다...")
            self.model = whisper.load_model(self.model_name)
            logger.info("모델 로드 완료")
        return self.model
        
    def transcribe_audio(self, audio_path):
        """
        오디오 파일을 텍스트로 변환합니다.
        
        Args:
            audio_path: 오디오 파일 경로
            
        Returns:
            transcription: 변환된 텍스트 및 세그먼트 정보
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_path}")
            
        # 모델 로드
        model = self.load_model()
        
        # 트랜스크립션 수행
        logger.info(f"파일 변환 시작: {audio_path}")
        result = model.transcribe(audio_path, language="ko")
        logger.info("변환 완료")
        
        return result
        
    def extract_audio_from_video(self, video_path, output_path=None):
        """
        비디오에서 오디오를 추출합니다. (FFmpeg 필요)
        
        Args:
            video_path: 비디오 파일 경로
            output_path: 출력 오디오 파일 경로 (None이면 자동 생성)
            
        Returns:
            audio_path: 추출된 오디오 파일 경로
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
            
        if output_path is None:
            video_filename = Path(video_path).stem
            output_path = f"temp/{video_filename}.mp3"
            
        # temp 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # FFmpeg로 오디오 추출
        import subprocess
        command = f"ffmpeg -i \"{video_path}\" -q:a 0 -map a \"{output_path}\" -y"
        
        try:
            subprocess.run(command, shell=True, check=True, capture_output=True)
            logger.info(f"오디오 추출 완료: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"오디오 추출 실패: {e}")
            raise

def process_video_to_text(video_path):
    """
    비디오 파일을 텍스트로 변환하는 편의 함수
    
    Args:
        video_path: 비디오 파일 경로
        
    Returns:
        transcription: 변환된 텍스트
    """
    processor = STTProcessor(model_name="small")
    audio_path = processor.extract_audio_from_video(video_path)
    result = processor.transcribe_audio(audio_path)
    
    # 임시 오디오 파일 삭제
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    return result 