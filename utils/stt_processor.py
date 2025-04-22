import os
import time
import logging
import numpy as np
import torch
import whisper
import ffmpeg
import psutil
import librosa
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Union
import tqdm
import threading
import ssl
from pathlib import Path

# SSL 인증서 검증 우회 설정
ssl._create_default_https_context = ssl._create_unverified_context

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 스레드별 모델 저장용 전역 딕셔너리
_thread_local = threading.local()

class STTProcessorParallel:
    """
    침묵 감지 기반으로 세그먼트를 나누고 병렬 처리하는 STT 프로세서
    """
    
    def __init__(self, model_name: str = "medium", device: Optional[str] = None, num_workers: int = None, use_mixed_precision: bool = True):
        """
        STT 프로세서를 초기화합니다.
        
        Args:
            model_name: 사용할 Whisper 모델 이름 ("tiny", "base", "small", "medium", "large")
            device: 사용할 장치 (None, "cpu", "cuda", "mps" 등)
            num_workers: 병렬 처리에 사용할 작업자 수 (None이면 CPU 코어 수 자동 설정)
            use_mixed_precision: MPS에서 작동하지 않는 연산을 CPU로 실행할지 여부
        """
        self.model_name = model_name
        self.model = None
        self.use_mixed_precision = use_mixed_precision
        
        # 장치 설정 (GPU, MPS 또는 CPU)
        if device is None:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = "mps"
                logger.info("MPS(Metal Performance Shaders) 가속이 활성화되었습니다.")
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # 병렬 처리 작업자 수 설정
        if num_workers is None:
            # 기본값으로 CPU 코어 수의 3/4 설정 (최소 2개)
            self.num_workers = max(2, int(os.cpu_count() * 0.75))
        else:
            self.num_workers = num_workers
        
        # MPS 가속 정보 로깅
        if self.device == "mps" and self.use_mixed_precision:
            logger.info("혼합 정밀도 연산 모드가 활성화됨: 일부 연산은 CPU에서, 나머지는 MPS에서 실행됩니다.")
        
        logger.info(f"Whisper {model_name} 모델을 초기화합니다...")
        self.model = self._load_model_with_device()
        logger.info(f"모델 초기화 완료 (device: {self.device}, 작업자 수: {self.num_workers})")
    
    def _load_model_with_device(self):
        """
        장치에 따라 모델을 로드합니다. MPS 사용 시 혼합 정밀도 옵션을 고려합니다.
        """
        if self.device == "mps" and self.use_mixed_precision:
            # MPS를 사용하지만 일부 연산은 CPU에서 실행되도록 처리
            logger.info("혼합 정밀도 모드로 Whisper 모델 로드 중 (MPS+CPU)...")
            model = whisper.load_model(self.model_name, device="cpu")
            
            # 특정 sparse 텐서 연산을 제외하고 MPS로 이동
            # 원본 모델의 인코더와 디코더만 MPS로 이동
            try:
                model.encoder.to("mps")
                model.decoder.to("mps")
                logger.info("Whisper 모델의 인코더와 디코더를 MPS로 이동했습니다.")
            except Exception as e:
                logger.warning(f"MPS 이동 중 오류: {str(e)}. CPU 모드로 계속합니다.")
            
            return model
        else:
            # 다른 장치(CPU, CUDA)는 그대로 사용
            return whisper.load_model(self.model_name, device=self.device)
    
    def extract_audio_from_video(self, video_path: str, output_path: str, enhance_audio: bool = False) -> str:
        """
        비디오 파일에서 오디오를 추출합니다.
        
        Args:
            video_path: 비디오 파일 경로
            output_path: 오디오 출력 파일 경로
            enhance_audio: 오디오 향상 여부 (노이즈 감소, 볼륨 정규화)
            
        Returns:
            추출된 오디오 파일 경로
        """
        try:
            # 비디오 파일이 존재하는지 확인
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
            
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # 기존 ffmpeg 라이브러리 대신 subprocess로 직접 ffmpeg 명령 실행
            command = ["ffmpeg", "-i", video_path, "-vn"]  # 비디오 스트림 제거
            
            if enhance_audio:
                # 오디오 향상 필터 추가
                command.extend(["-af", "afftdn,dynaudnorm"])  # 노이즈 감소 및 볼륨 정규화
            
            # 16kHz, mono, PCM 형식으로 변환
            command.extend([
                "-acodec", "pcm_s16le",  # 16비트 PCM 오디오 형식
                "-ar", "16000",          # 16kHz 샘플링 레이트
                "-ac", "1",              # 모노 채널
                "-y",                    # 기존 파일 덮어쓰기
                output_path
            ])
            
            # 명령 실행 (더 자세한 오류 메시지 표시)
            logger.info(f"ffmpeg 명령 실행: {' '.join(command)}")
            import subprocess
            
            # 로그 파일에 ffmpeg 출력 저장 (디버깅용)
            log_dir = os.path.dirname(os.path.abspath(output_path))
            log_file = os.path.join(log_dir, "ffmpeg_log.txt")
            
            with open(log_file, "w") as log:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # 로그 저장
                log.write(f"STDOUT:\n{result.stdout}\n\n")
                log.write(f"STDERR:\n{result.stderr}\n\n")
                log.write(f"RETURN CODE: {result.returncode}\n")
                
            # 결과 확인
            if result.returncode != 0:
                raise Exception(f"ffmpeg 명령 실패 (코드: {result.returncode}): {result.stderr}")
            
            # 출력 파일 확인
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise Exception(f"오디오 파일이 생성되지 않았거나 크기가 0입니다: {output_path}")
            
            logger.info(f"오디오 추출 완료: {output_path} (크기: {os.path.getsize(output_path)/1024:.1f} KB)")
            return output_path
            
        except Exception as e:
            logger.error(f"오디오 추출 실패: {str(e)}")
            # 더 자세한 오류 정보 출력
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _detect_silence(self, audio: np.ndarray, sr: int = 16000, 
                        threshold: float = 0.02, min_silence_duration: float = 0.7, 
                        min_speech_duration: float = 1.0) -> List[float]:
        """
        오디오에서 침묵 구간을 감지합니다.
        
        Args:
            audio: 오디오 데이터 (numpy 배열)
            sr: 샘플링 레이트
            threshold: 침묵으로 간주할 진폭 임계값
            min_silence_duration: 최소 침묵 지속 시간(초)
            min_speech_duration: 최소 음성 지속 시간(초)
            
        Returns:
            침묵 구간의 끝 지점 시간(초) 목록
        """
        # 진폭 계산
        amplitude = np.abs(audio)
        
        # 임계값 이하는 침묵으로 간주
        is_silence = amplitude < threshold
        
        # 최소 침묵 지속 시간을 샘플 수로 변환
        min_silence_samples = int(min_silence_duration * sr)
        min_speech_samples = int(min_speech_duration * sr)
        
        # 침묵 구간 찾기
        silence_segments = []
        speech_segments = []
        
        in_silence = False
        silence_start = 0
        speech_start = 0
        
        for i, silent in enumerate(is_silence):
            if silent and not in_silence:
                # 침묵 시작
                in_silence = True
                silence_start = i
                
                # 이전 음성 구간 기록
                if i - speech_start >= min_speech_samples:
                    speech_segments.append((speech_start, i))
                
            elif not silent and in_silence:
                # 침묵 종료
                in_silence = False
                speech_start = i
                
                # 충분히 긴 침묵 구간만 기록
                if i - silence_start >= min_silence_samples:
                    silence_segments.append((silence_start, i))
        
        # 마지막 구간 처리
        if in_silence and len(audio) - silence_start >= min_silence_samples:
            silence_segments.append((silence_start, len(audio)))
        elif not in_silence and len(audio) - speech_start >= min_speech_samples:
            speech_segments.append((speech_start, len(audio)))
        
        # 침묵 구간의 끝 지점을 시간(초)으로 변환
        silence_boundaries = [segment[1] / sr for segment in silence_segments]
        
        logger.info(f"침묵 구간 감지: {len(silence_boundaries)}개의 구간 찾음")
        return silence_boundaries
    
    def _adaptive_chunk_audio(self, audio: np.ndarray, sr: int = 16000, 
                              min_chunk: float = 5.0, max_chunk: float = 30.0,
                              overlap_sec: float = 1.0) -> List[Tuple[np.ndarray, float]]:
        """
        침묵 구간을 기반으로 오디오를 적응적으로 분할합니다.
        
        Args:
            audio: 오디오 데이터
            sr: 샘플링 레이트
            min_chunk: 최소 청크 길이(초)
            max_chunk: 최대 청크 길이(초)
            overlap_sec: 세그먼트 간 중복 구간 길이(초)
            
        Returns:
            (오디오 청크, 시작 시간(초)) 튜플의 리스트
        """
        # 침묵 구간 감지
        silence_points = self._detect_silence(audio, sr)
        
        # 최소, 최대 청크 길이를 샘플 수로 변환
        min_chunk_samples = int(min_chunk * sr)
        max_chunk_samples = int(max_chunk * sr)
        overlap_samples = int(overlap_sec * sr)
        
        # 청크 나누기
        chunks = []
        start_sample = 0
        
        for silence_time in silence_points:
            silence_sample = int(silence_time * sr)
            chunk_length = silence_sample - start_sample
            
            # 청크가 너무 짧으면 건너뜀
            if chunk_length < min_chunk_samples:
                continue
                
            # 청크가 너무 길면 최대 길이로 제한하여 여러 청크로 나눔
            if chunk_length > max_chunk_samples:
                # 최대 길이로 여러 개의 청크로 나눔
                num_subchunks = (chunk_length + max_chunk_samples - 1) // max_chunk_samples
                for i in range(num_subchunks):
                    sub_start = start_sample + i * max_chunk_samples
                    # 첫 번째 청크가 아닌 경우 중복 구간 추가
                    if i > 0:
                        sub_start = max(0, sub_start - overlap_samples)
                    
                    sub_end = min(sub_start + max_chunk_samples, silence_sample)
                    sub_chunk = audio[sub_start:sub_end]
                    chunks.append((sub_chunk, sub_start / sr))
            else:
                # 적절한 길이의 청크
                chunk = audio[start_sample:silence_sample]
                chunks.append((chunk, start_sample / sr))
            
            # 다음 시작점 업데이트
            start_sample = silence_sample
        
        # 마지막 부분이 처리되지 않았으면 처리
        if start_sample < len(audio) - min_chunk_samples:
            remaining_length = len(audio) - start_sample
            
            if remaining_length > max_chunk_samples:
                # 최대 길이로 여러 개의 청크로 나눔
                num_subchunks = (remaining_length + max_chunk_samples - 1) // max_chunk_samples
                for i in range(num_subchunks):
                    sub_start = start_sample + i * max_chunk_samples
                    # 첫 번째 청크가 아닌 경우 중복 구간 추가
                    if i > 0:
                        sub_start = max(0, sub_start - overlap_samples)
                    
                    sub_end = min(sub_start + max_chunk_samples, len(audio))
                    sub_chunk = audio[sub_start:sub_end]
                    chunks.append((sub_chunk, sub_start / sr))
            else:
                # 마지막 청크 추가
                chunk = audio[start_sample:]
                chunks.append((chunk, start_sample / sr))
        
        logger.info(f"적응형 청크 분할: {len(chunks)}개의 청크 생성 (중복 구간: {overlap_sec}초)")
        return chunks

    def _get_thread_local_model(self):
        """
        현재 스레드에 대한 로컬 모델 인스턴스를 반환합니다.
        이 메소드는 CPU 전용 스레드와 MPS 지원 스레드 모두에서 작동합니다.
        """
        thread_id = threading.get_ident()
        
        # 현재 스레드에 대한 모델이 없으면 새로 생성
        if not hasattr(_thread_local, 'models'):
            _thread_local.models = {}
        
        if thread_id not in _thread_local.models:
            # 스레드별 모델 로드
            if self.device == "mps" and self.use_mixed_precision:
                # MPS 모델 - 인코더/디코더는 MPS로 이동
                model = whisper.load_model(self.model_name, device="cpu")
                try:
                    # 인코더와 디코더만 MPS로 이동
                    model.encoder.to("mps")
                    model.decoder.to("mps")
                    logger.info(f"스레드 {thread_id}에 하이브리드(MPS+CPU) 모델 로드됨")
                except Exception as e:
                    logger.warning(f"MPS로 모델 일부 이동 실패, CPU 모드로 계속합니다: {str(e)}")
                _thread_local.models[thread_id] = model
            else:
                # 병렬 모드에서는 항상 CPU 사용, 단일 모드에서는 self.device 사용
                device = "cpu" if self.num_workers > 1 else self.device
                _thread_local.models[thread_id] = whisper.load_model(self.model_name, device=device)
                logger.info(f"스레드 {thread_id}에 {device} 모델 로드됨")
            
        return _thread_local.models[thread_id]

    def _process_chunk(self, chunk_data: Tuple[np.ndarray, float], chunk_id: int) -> Dict:
        """
        단일 오디오 청크를 처리합니다 (병렬 처리용).
        
        Args:
            chunk_data: (오디오 청크, 시작 시간) 튜플
            chunk_id: 청크 ID
            
        Returns:
            처리된 세그먼트 정보
        """
        chunk, start_time_offset = chunk_data
        chunk_start = time.time()
        
        # 청크 길이를 초로 변환
        sr = 16000  # 샘플링 레이트
        chunk_duration = len(chunk) / sr
        end_time_offset = start_time_offset + chunk_duration
        
        try:
            # 각 스레드별로 별도의 모델 인스턴스를 사용하여 청크 처리
            with torch.no_grad():
                # 스레드별 모델 가져오기
                model = self._get_thread_local_model()
                
                # MPS 장치인 경우 메모리 관리를 위해 캐시 정리
                if self.device == "mps":
                    torch.mps.empty_cache()
                
                # MPS와 CPU 중 적절한 장치 선택
                if self.device == "mps" and self.use_mixed_precision:
                    # MPS에서는 특정 연산에서 지속적으로 오류가 발생하므로
                    # 중요한 부분만 MPS에서 실행하고, 실제 디코딩은 CPU에서 진행
                    try:
                        # 메인 모델 컴포넌트는 MPS 유지 (인코더/디코더)
                        # 실제 추론은 CPU로 임시 전환하여 수행
                        temp_device = "cpu"
                        logger.info(f"청크 {chunk_id}: 하이브리드 모드로 처리 (MPS+CPU)")
                        
                        # 기존 모델에서 CPU 버전 복제 (무거운 작업이므로 스레드당 한 번만 수행)
                        if not hasattr(model, '_cpu_clone'):
                            # CPU 모델 생성 (경량 복제)
                            model._cpu_clone = whisper.load_model(self.model_name, device="cpu")
                            logger.info(f"청크 {chunk_id}: CPU 백업 모델 생성 완료")
                        
                        # CPU 모델을 사용하여 추론
                        segment_result = model._cpu_clone.transcribe(
                            chunk, 
                            language="ko",
                            fp16=False
                        )
                    except Exception as e:
                        logger.warning(f"청크 {chunk_id}: 하이브리드 모드 처리 실패, 순수 CPU로 전환: {str(e)}")
                        segment_result = whisper.load_model(self.model_name, device="cpu").transcribe(
                            chunk, 
                            language="ko",
                            fp16=False
                        )
                else:
                    # 일반 모드에서 실행 (CPU 또는 CUDA)
                    segment_result = model.transcribe(
                        chunk, 
                        language="ko",
                        fp16=False if self.device == "cpu" else True
                    )
            
            chunk_end = time.time()
            
            # 결과가 없으면 빈 텍스트 설정
            if not segment_result or not isinstance(segment_result, dict) or "text" not in segment_result:
                segment_text = ""
                logger.warning(f"청크 {chunk_id}: 유효한 텍스트 결과가 없습니다.")
            else:
                segment_text = segment_result["text"].strip()
            
            # 세그먼트 결과 저장
            segment_key = f"segment_{start_time_offset:.2f}_{end_time_offset:.2f}"
            
            # 처리 정보 저장
            segment_info = {
                "id": chunk_id,
                "start": start_time_offset,
                "end": end_time_offset,
                "text": segment_text,
                "processing_time": chunk_end - chunk_start,
                "device_used": f"{self.device}+cpu" if self.device == "mps" and self.use_mixed_precision else self.device
            }
            
            return {
                "segment_key": segment_key,
                "segment_info": segment_info,
                "text": segment_text,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"청크 {chunk_id} 처리 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "segment_key": f"segment_{start_time_offset:.2f}_{end_time_offset:.2f}",
                "segment_info": {
                    "id": chunk_id,
                    "start": start_time_offset,
                    "end": end_time_offset,
                    "error": str(e)
                },
                "text": "",
                "error": str(e)
            }
    
    def transcribe_audio(self, audio_path: str, min_chunk: float = 5.0, max_chunk: float = 30.0, 
                         threshold: float = 0.02, min_silence: float = 0.7,
                         overlap_sec: float = 1.0) -> dict:
        """
        오디오 파일을 텍스트로 변환합니다 (병렬 처리).
        
        Args:
            audio_path: 오디오 파일 경로
            min_chunk: 최소 청크 길이(초)
            max_chunk: 최대 청크 길이(초)
            threshold: 침묵 감지 임계값
            min_silence: 최소 침묵 지속 시간(초)
            overlap_sec: 세그먼트 간 중복 구간 길이(초)
            
        Returns:
            텍스트 변환 결과와 메타데이터
        """
        start_time = time.time()
        
        try:
            # 모델 로드 확인
            if self.model is None:
                logger.info(f"Whisper {self.model_name} 모델을 초기화합니다...")
                self.model = whisper.load_model(self.model_name, device=self.device)
                logger.info(f"모델 초기화 완료 (device: {self.device})")
            
            # 오디오 파일 로드
            logger.info(f"오디오 파일 로드 중: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # 메모리 사용량 측정
            memory_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
            
            # 침묵 기반 적응형 분할
            chunks = self._adaptive_chunk_audio(audio, sr, min_chunk, max_chunk, overlap_sec)
            
            # 세그먼트별 결과 저장
            results = {
                "text": "", 
                "segments": [], 
                "segment_results": {},
                "parallel_processing": {
                    "num_workers": self.num_workers,
                    "num_chunks": len(chunks)
                }
            }
            
            logger.info(f"병렬 처리 시작: {len(chunks)}개 세그먼트, {self.num_workers}개 작업자")
            
            # 스레드 로컬 스토리지 초기화
            _thread_local.models = {}
            
            # 병렬 처리 실행
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # 작업 실행
                futures = [executor.submit(self._process_chunk, chunk, i) 
                          for i, chunk in enumerate(chunks)]
                
                # 진행 상황 표시
                completed = 0
                for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="세그먼트 처리"):
                    completed += 1
                    if completed % 5 == 0 or completed == len(futures):
                        logger.info(f"세그먼트 처리 중: {completed}/{len(futures)} 완료")
            
            # 결과 수집 (시간 순으로 정렬)
            segment_results = [future.result() for future in futures]
            segment_results.sort(key=lambda x: x["segment_info"]["start"])
            
            # 정렬된 결과로 텍스트 구성
            all_text = []
            for result in segment_results:
                segment_key = result["segment_key"]
                segment_info = result["segment_info"]
                
                if "error" not in segment_info or segment_info["error"] is None:
                    results["segments"].append(segment_info)
                    results["segment_results"][segment_key] = segment_info
                    all_text.append(result["text"])
                else:
                    results["segment_results"][segment_key] = {"error": segment_info["error"]}
            
            # 전체 텍스트 구성
            results["text"] = " ".join(all_text)
            
            # 처리 종료 시간 및 메모리 사용량 측정
            end_time = time.time()
            memory_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
            
            # 메타데이터 추가
            results["processing_time"] = end_time - start_time
            results["memory_usage_mb"] = memory_after - memory_before
            results["model"] = self.model_name
            results["silence_detection"] = {
                "threshold": threshold,
                "min_silence": min_silence,
                "min_chunk": min_chunk,
                "max_chunk": max_chunk,
                "overlap_sec": overlap_sec,
                "num_chunks": len(chunks)
            }
            
            logger.info(f"변환 완료: 처리 시간 {results['processing_time']:.2f}초, 텍스트 길이 {len(results['text'])}")
            
            # 스레드 로컬 스토리지 정리
            _thread_local.models = {}
            
            return results
            
        except Exception as e:
            logger.error(f"오디오 변환 실패: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            end_time = time.time()
            return {
                "error": str(e),
                "processing_time": end_time - start_time
            }
            
    def process_video_to_text(self, video_path: str, output_dir: str = None, 
                             min_chunk: float = 5.0, max_chunk: float = 30.0,
                             threshold: float = 0.02, min_silence: float = 0.7,
                             overlap_sec: float = 1.0) -> dict:
        """
        비디오 파일을 텍스트로 변환하는 전체 프로세스를 실행합니다.
        
        Args:
            video_path: 비디오 파일 경로
            output_dir: 결과 저장 디렉토리 (None이면 임시 디렉토리 사용)
            min_chunk: 최소 청크 길이(초)
            max_chunk: 최대 청크 길이(초)
            threshold: 침묵 감지 임계값
            min_silence: 최소 침묵 지속 시간(초)
            overlap_sec: 세그먼트 간 중복 구간 길이(초)
            
        Returns:
            텍스트 변환 결과와 메타데이터
        """
        start_time = time.time()
        
        try:
            # 출력 디렉토리 설정
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(video_path), "stt_output")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 오디오 추출 경로
            audio_path = os.path.join(output_dir, os.path.basename(video_path) + ".wav")
            
            # 오디오 추출
            self.extract_audio_from_video(video_path, audio_path, enhance_audio=True)
            
            # 오디오 변환
            result = self.transcribe_audio(
                audio_path, 
                min_chunk=min_chunk,
                max_chunk=max_chunk,
                threshold=threshold,
                min_silence=min_silence,
                overlap_sec=overlap_sec
            )
            
            # 텍스트 결과 저장
            text_path = os.path.join(output_dir, os.path.basename(video_path) + ".txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
            
            # 처리 시간 기록
            end_time = time.time()
            result["total_processing_time"] = end_time - start_time
            result["output_text_path"] = text_path
            
            # 임시 오디오 파일 삭제
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return result
            
        except Exception as e:
            logger.error(f"비디오 처리 실패: {str(e)}")
            end_time = time.time()
            return {
                "error": str(e),
                "total_processing_time": end_time - start_time
            }

# 모듈 직접 실행 테스트
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="병렬 처리 기반 STT 프로세서")
    parser.add_argument("--video", type=str, required=True, help="변환할 비디오 파일 경로")
    parser.add_argument("--output", type=str, default=None, help="결과 저장 디렉토리")
    parser.add_argument("--model", type=str, default="small", help="사용할 Whisper 모델 (tiny, base, small, medium, large)")
    parser.add_argument("--device", type=str, default=None, help="사용할 장치 (cpu, cuda, mps, None=자동)")
    parser.add_argument("--mixed-precision", action="store_true", help="MPS에서 일부 연산을 CPU로 실행 (MPS 사용 시)")
    parser.add_argument("--min-chunk", type=float, default=5.0, help="최소 청크 길이(초)")
    parser.add_argument("--max-chunk", type=float, default=30.0, help="최대 청크 길이(초)")
    parser.add_argument("--threshold", type=float, default=0.02, help="침묵 감지 임계값")
    parser.add_argument("--min-silence", type=float, default=0.7, help="최소 침묵 지속 시간(초)")
    parser.add_argument("--overlap-sec", type=float, default=1.0, help="세그먼트 간 중복 구간 길이(초)")
    parser.add_argument("--workers", type=int, default=None, help="병렬 처리에 사용할 작업자 수 (기본값: CPU 코어 수의 3/4)")
    
    args = parser.parse_args()
    
    processor = STTProcessorParallel(
        model_name=args.model, 
        device=args.device,
        num_workers=args.workers,
        use_mixed_precision=args.mixed_precision
    )
    
    result = processor.process_video_to_text(
        args.video, 
        args.output,
        min_chunk=args.min_chunk,
        max_chunk=args.max_chunk,
        threshold=args.threshold,
        min_silence=args.min_silence,
        overlap_sec=args.overlap_sec
    )
    
    print("\n============= 처리 결과 =============")
    print(f"모델: {args.model}")
    print(f"장치: {processor.device}")
    if processor.device == "mps":
        print(f"혼합 정밀도 모드: {'활성화' if processor.use_mixed_precision else '비활성화'}")
    print(f"병렬 작업자 수: {processor.num_workers}")
    print(f"처리 시간: {result.get('total_processing_time', 0):.2f}초")
    print(f"텍스트 길이: {len(result.get('text', ''))}")
    print(f"세그먼트 수: {len(result.get('segments', []))}")
    print(f"청크 수: {result.get('silence_detection', {}).get('num_chunks', 0)}")
    print(f"중복 구간: {args.overlap_sec}초")
    print("======================================")
