#!/usr/bin/env python3
import os
import time
import argparse
import json
import logging
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 병렬 처리 STT 모듈 임포트
try:
    from utils.stt_processor_parallel import STTProcessorParallel
except ImportError:
    # 상대 경로 시도
    from stt_processor_parallel import STTProcessorParallel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stt_parallel_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_parallel_stt_test(video_path, output_dir, model="medium", 
                          min_chunk=5.0, max_chunk=30.0, 
                          threshold=0.02, min_silence=0.7,
                          num_workers=None):
    """
    병렬 처리 STT 테스트를 실행합니다.
    
    Args:
        video_path: 비디오 파일 경로
        output_dir: 결과 저장 디렉토리
        model: 사용할 모델 크기
        min_chunk: 최소 청크 길이(초)
        max_chunk: 최대 청크 길이(초)
        threshold: 침묵 감지 임계값
        min_silence: 최소 침묵 지속 시간(초)
        num_workers: 병렬 처리에 사용할 작업자 수
    
    Returns:
        테스트 결과 딕셔너리
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # 테스트 시작
        logger.info(f"병렬 처리 STT 테스트 시작: {video_path}")
        logger.info(f"설정: 모델={model}, 최소 청크={min_chunk}초, 최대 청크={max_chunk}초, 임계값={threshold}, 작업자={num_workers}명")
        
        # STT 프로세서 초기화
        processor = STTProcessorParallel(model_name=model, num_workers=num_workers)
        
        # 비디오 처리
        start_time = time.time()
        result = processor.process_video_to_text(
            video_path,
            output_dir=str(output_dir),
            min_chunk=min_chunk,
            max_chunk=max_chunk,
            threshold=threshold,
            min_silence=min_silence
        )
        end_time = time.time()
        
        # 결과 처리
        processing_time = end_time - start_time
        
        # 텍스트 결과 저장
        with open(output_dir / "result.txt", "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        # 성능 결과 저장
        performance_result = {
            "video_path": video_path,
            "model": model,
            "min_chunk": min_chunk,
            "max_chunk": max_chunk,
            "threshold": threshold,
            "min_silence": min_silence,
            "num_workers": processor.num_workers,
            "processing_time": processing_time,
            "memory_usage_mb": result.get("memory_usage_mb", 0),
            "text_length": len(result.get("text", "")),
            "segment_count": len(result.get("segments", [])),
            "chunk_count": result.get("silence_detection", {}).get("num_chunks", 0)
        }
        
        with open(output_dir / "performance_results.json", "w") as f:
            json.dump(performance_result, f, indent=2)
        
        logger.info(f"테스트 완료: 처리 시간={processing_time:.2f}초, 세그먼트 수={performance_result['segment_count']}")
        
        # 세그먼트 시각화
        if len(result.get("segments", [])) > 0:
            create_segment_visualization(result["segments"], output_dir)
        
        return performance_result
        
    except Exception as e:
        logger.error(f"테스트 실패: {str(e)}")
        with open(output_dir / "error.txt", "w") as f:
            f.write(str(e))
        return {"error": str(e)}

def create_segment_visualization(segments, output_dir):
    """
    세그먼트 분포를 시각화합니다.
    
    Args:
        segments: 세그먼트 목록
        output_dir: 출력 디렉토리
    """
    # 세그먼트 길이 계산
    segment_lengths = [(s["end"] - s["start"]) for s in segments]
    segment_starts = [s["start"] for s in segments]
    segment_ends = [s["end"] for s in segments]
    processing_times = [s.get("processing_time", 0) for s in segments]
    
    # 세그먼트 길이 히스토그램
    plt.figure(figsize=(10, 6))
    plt.hist(segment_lengths, bins=20)
    plt.title('세그먼트 길이 분포')
    plt.xlabel('세그먼트 길이 (초)')
    plt.ylabel('빈도')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "segment_length_histogram.png")
    
    # 세그먼트 타임라인
    plt.figure(figsize=(12, 6))
    for i, seg in enumerate(segments):
        plt.plot([seg["start"], seg["end"]], [1, 1], 'b-', linewidth=2)
        plt.text(seg["start"], 1.05, f"{i+1}", fontsize=8, ha='center')
    
    plt.title('세그먼트 타임라인')
    plt.xlabel('시간 (초)')
    plt.yticks([])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "segment_timeline.png")
    
    # 세그먼트 처리 시간
    if any(t > 0 for t in processing_times):
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(processing_times)), processing_times)
        plt.title('세그먼트별 처리 시간')
        plt.xlabel('세그먼트 ID')
        plt.ylabel('처리 시간 (초)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "segment_processing_time.png")
    
    # 세그먼트 정보 데이터프레임 저장
    df = pd.DataFrame({
        "segment_id": [i+1 for i in range(len(segments))],
        "start_time": segment_starts,
        "end_time": segment_ends,
        "duration": segment_lengths,
        "text_length": [len(s.get("text", "")) for s in segments],
        "processing_time": processing_times
    })
    
    df.to_csv(output_dir / "segment_info.csv", index=False)
    
    # 평균 세그먼트 길이
    avg_length = sum(segment_lengths) / len(segment_lengths)
    with open(output_dir / "segment_stats.txt", "w") as f:
        f.write(f"세그먼트 수: {len(segments)}\n")
        f.write(f"평균 세그먼트 길이: {avg_length:.2f}초\n")
        f.write(f"최소 세그먼트 길이: {min(segment_lengths):.2f}초\n")
        f.write(f"최대 세그먼트 길이: {max(segment_lengths):.2f}초\n")
        if any(t > 0 for t in processing_times):
            f.write(f"평균 처리 시간: {sum(processing_times)/len(processing_times):.2f}초\n")
            f.write(f"최대 처리 시간: {max(processing_times):.2f}초\n")

def compare_processors(video_path, output_dir, model="medium", num_workers_list=None):
    """
    다양한 병렬 작업자 수와 기존 모델 간의 성능을 비교합니다.
    
    Args:
        video_path: 비디오 파일 경로
        output_dir: 결과 저장 디렉토리
        model: 사용할 모델 크기
        num_workers_list: 테스트할 작업자 수 목록
    
    Returns:
        비교 결과 딕셔너리
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if num_workers_list is None:
        # 기본 작업자 수 목록 (1, 2, 코어 수/2, 코어 수*3/4)
        cpu_count = os.cpu_count()
        num_workers_list = [1, 2, max(2, cpu_count // 2), max(2, int(cpu_count * 0.75))]
    
    try:
        # 결과 저장
        comparison_results = {
            "video_path": video_path,
            "model": model,
            "parallel": {},
            "silence_based": {}
        }
        
        # 1. 병렬 처리 테스트
        for num_workers in num_workers_list:
            test_name = f"parallel_{num_workers}"
            test_dir = output_dir / test_name
            
            logger.info(f"병렬 처리 테스트 (작업자 수: {num_workers}) 실행 중...")
            result = run_parallel_stt_test(
                video_path,
                test_dir,
                model=model,
                num_workers=num_workers
            )
            
            # 결과 저장
            comparison_results["parallel"][test_name] = {
                "workers": num_workers,
                "processing_time": result.get("processing_time", 0),
                "text_length": result.get("text_length", 0),
                "segment_count": result.get("segment_count", 0),
                "memory_usage_mb": result.get("memory_usage_mb", 0)
            }
        
        # 2. 기존 침묵 기반 테스트
        try:
            from utils.stt_processor_silence import STTProcessorSilence
            
            test_name = "silence_based"
            test_dir = output_dir / test_name
            
            logger.info("기존 침묵 기반 STT 테스트 실행 중...")
            
            # 기존 STT 프로세서로 테스트
            processor = STTProcessorSilence(model_name=model)
            
            # 비디오 처리
            start_time = time.time()
            result = processor.process_video_to_text(
                video_path,
                output_dir=str(test_dir)
            )
            end_time = time.time()
            
            processing_time = end_time - start_time
            text_length = len(result.get("text", ""))
            segment_count = len(result.get("segments", []))
            memory_usage = result.get("memory_usage_mb", 0)
            
            # 결과 저장
            comparison_results["silence_based"]["original"] = {
                "processing_time": processing_time,
                "text_length": text_length,
                "segment_count": segment_count,
                "memory_usage_mb": memory_usage
            }
            
        except ImportError:
            logger.warning("기존 STT 프로세서를 찾을 수 없어 비교 테스트를 건너뜁니다.")
        
        # 결과 저장
        with open(output_dir / "comparison_results.json", "w") as f:
            json.dump(comparison_results, f, indent=2)
        
        # 결과 시각화
        create_comparison_visualization(comparison_results, output_dir)
        
        return comparison_results
        
    except Exception as e:
        logger.error(f"비교 테스트 실패: {str(e)}")
        with open(output_dir / "error.txt", "w") as f:
            f.write(str(e))
        return {"error": str(e)}

def create_comparison_visualization(comparison_results, output_dir):
    """
    세그먼트 비교 결과를 시각화합니다.
    
    Args:
        comparison_results: 비교 결과 딕셔너리
        output_dir: 출력 디렉토리
    """
    # 1. 처리 시간 비교
    plt.figure(figsize=(12, 6))
    
    # 병렬 처리 결과
    parallel_names = []
    parallel_times = []
    
    for name, result in comparison_results["parallel"].items():
        parallel_names.append(f"병렬 ({result['workers']})")
        parallel_times.append(result["processing_time"])
    
    # 기존 침묵 기반 결과
    silence_names = []
    silence_times = []
    
    for name, result in comparison_results["silence_based"].items():
        silence_names.append("기존 STT")
        silence_times.append(result["processing_time"])
    
    # 바 차트
    x = list(range(len(parallel_names) + len(silence_names)))
    plt.bar(x[:len(parallel_names)], parallel_times, color='green', alpha=0.7, label='병렬 처리')
    plt.bar(x[len(parallel_names):], silence_times, color='blue', alpha=0.7, label='기존 STT')
    
    plt.title('처리 방식별 처리 시간 비교')
    plt.ylabel('처리 시간 (초)')
    plt.xticks(x, parallel_names + silence_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "processing_time_comparison.png")
    
    # 2. 효율성 비교 (자 / 초)
    plt.figure(figsize=(12, 6))
    
    # 병렬 처리 효율성
    parallel_efficiency = [result["text_length"] / result["processing_time"] 
                          if result["processing_time"] > 0 else 0 
                          for result in comparison_results["parallel"].values()]
    
    # 기존 침묵 기반 효율성
    silence_efficiency = [result["text_length"] / result["processing_time"] 
                         if result["processing_time"] > 0 else 0 
                         for result in comparison_results["silence_based"].values()]
    
    # 바 차트
    plt.bar(x[:len(parallel_names)], parallel_efficiency, color='green', alpha=0.7, label='병렬 처리')
    plt.bar(x[len(parallel_names):], silence_efficiency, color='blue', alpha=0.7, label='기존 STT')
    
    plt.title('처리 방식별 효율성 비교 (자 / 초)')
    plt.ylabel('효율성 (자 / 초)')
    plt.xticks(x, parallel_names + silence_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "efficiency_comparison.png")
    
    # 3. 메모리 사용량 비교
    if any("memory_usage_mb" in result for result in comparison_results["parallel"].values()):
        plt.figure(figsize=(12, 6))
        
        # 병렬 처리 메모리 사용량
        parallel_memory = [result.get("memory_usage_mb", 0) for result in comparison_results["parallel"].values()]
        
        # 기존 침묵 기반 메모리 사용량
        silence_memory = [result.get("memory_usage_mb", 0) for result in comparison_results["silence_based"].values()]
        
        # 바 차트
        plt.bar(x[:len(parallel_names)], parallel_memory, color='green', alpha=0.7, label='병렬 처리')
        plt.bar(x[len(parallel_names):], silence_memory, color='blue', alpha=0.7, label='기존 STT')
        
        plt.title('처리 방식별 메모리 사용량 비교 (MB)')
        plt.ylabel('메모리 사용량 (MB)')
        plt.xticks(x, parallel_names + silence_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "memory_usage_comparison.png")
    
    # 4. 요약 테이블 저장
    data = []
    
    # 병렬 처리 데이터
    for name, result in comparison_results["parallel"].items():
        efficiency = result["text_length"] / result["processing_time"] if result["processing_time"] > 0 else 0
        data.append({
            "방식": "병렬 처리",
            "설정": f"작업자 {result['workers']}명",
            "처리 시간(초)": result["processing_time"],
            "텍스트 길이(자)": result["text_length"],
            "세그먼트 수": result["segment_count"],
            "메모리 사용(MB)": result.get("memory_usage_mb", 0),
            "효율성(자/초)": efficiency
        })
    
    # 기존 침묵 기반 데이터
    for name, result in comparison_results["silence_based"].items():
        efficiency = result["text_length"] / result["processing_time"] if result["processing_time"] > 0 else 0
        data.append({
            "방식": "기존 STT",
            "설정": "단일 스레드",
            "처리 시간(초)": result["processing_time"],
            "텍스트 길이(자)": result["text_length"],
            "세그먼트 수": result["segment_count"],
            "메모리 사용(MB)": result.get("memory_usage_mb", 0),
            "효율성(자/초)": efficiency
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "comparison_summary.csv", index=False)
    
    # 효율성 기준 정렬
    df_sorted = df.sort_values(by='효율성(자/초)', ascending=False)
    
    # 최고 효율 설정 저장
    with open(output_dir / "best_settings.txt", "w") as f:
        f.write(f"최고 효율 설정: {df_sorted.iloc[0]['설정']}\n")
        f.write(f"방식: {df_sorted.iloc[0]['방식']}\n")
        f.write(f"효율성: {df_sorted.iloc[0]['효율성(자/초)']:.2f} 자/초\n")
        f.write(f"처리 시간: {df_sorted.iloc[0]['처리 시간(초)']:.2f}초\n")
        f.write(f"메모리 사용: {df_sorted.iloc[0]['메모리 사용(MB)']:.2f} MB\n")
        
        # 기존 대비 성능 개선
        if len(comparison_results["silence_based"]) > 0:
            baseline = next(iter(comparison_results["silence_based"].values()))
            baseline_time = baseline["processing_time"]
            best_time = df_sorted.iloc[0]['처리 시간(초)']
            improvement = (baseline_time - best_time) / baseline_time * 100
            f.write(f"기존 대비 속도 개선: {improvement:.2f}%\n")

def main():
    parser = argparse.ArgumentParser(description="병렬 처리 STT 테스트")
    parser.add_argument("--video", type=str, required=True, help="테스트할 비디오 파일 경로")
    parser.add_argument("--output", type=str, default="./test_results_parallel", help="결과 저장 디렉토리")
    parser.add_argument("--model", type=str, default="medium", help="사용할 모델 크기 (tiny, base, small, medium, large)")
    parser.add_argument("--test-type", type=str, choices=["single", "compare"], default="single", 
                       help="테스트 유형 - single: 단일 설정 테스트, compare: 다양한 작업자 수 비교")
    parser.add_argument("--workers", type=int, default=None, help="병렬 처리에 사용할 작업자 수")
    
    args = parser.parse_args()
    
    # 비디오 파일 확인
    if not os.path.exists(args.video):
        logger.error(f"비디오 파일을 찾을 수 없습니다: {args.video}")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    logger.info(f"테스트 시작: {args.video}")
    logger.info(f"테스트 유형: {args.test_type}")
    
    if args.test_type == "single":
        # 단일 설정 테스트
        run_parallel_stt_test(
            args.video,
            args.output,
            model=args.model,
            num_workers=args.workers
        )
    elif args.test_type == "compare":
        # 다양한 작업자 수 비교 테스트
        compare_processors(
            args.video,
            args.output,
            model=args.model
        )
    
    logger.info(f"테스트 완료. 결과는 {args.output} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 