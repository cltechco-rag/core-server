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
                          overlap_sec=1.0,
                          num_workers=None):
    """
    Run parallel STT test.
    
    Args:
        video_path: Video file path
        output_dir: Result directory
        model: Model size to use
        min_chunk: Minimum chunk length (seconds)
        max_chunk: Maximum chunk length (seconds)
        threshold: Silence detection threshold
        min_silence: Minimum silence duration (seconds)
        overlap_sec: Segment overlap duration (seconds)
        num_workers: Number of workers for parallel processing
    
    Returns:
        Test result dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Start test
        logger.info(f"Starting parallel STT test: {video_path}")
        logger.info(f"Settings: model={model}, min_chunk={min_chunk}s, max_chunk={max_chunk}s, threshold={threshold}, overlap={overlap_sec}s, workers={num_workers}")
        
        # Initialize STT processor
        processor = STTProcessorParallel(model_name=model, num_workers=num_workers)
        
        # Process video
        start_time = time.time()
        result = processor.process_video_to_text(
            video_path,
            output_dir=str(output_dir),
            min_chunk=min_chunk,
            max_chunk=max_chunk,
            threshold=threshold,
            min_silence=min_silence,
            overlap_sec=overlap_sec
        )
        end_time = time.time()
        
        # Process results
        processing_time = end_time - start_time
        
        # Save text result
        with open(output_dir / "result.txt", "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        # Save performance result
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
        
        logger.info(f"Test completed: processing_time={processing_time:.2f}s, segment_count={performance_result['segment_count']}")
        
        # Segment visualization
        if len(result.get("segments", [])) > 0:
            create_segment_visualization(result["segments"], output_dir)
        
        return performance_result
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        with open(output_dir / "error.txt", "w") as f:
            f.write(str(e))
        return {"error": str(e)}

def create_segment_visualization(segments, output_dir):
    """
    Create segment distribution visualizations.
    
    Args:
        segments: List of segments
        output_dir: Output directory
    """
    # Calculate segment lengths
    segment_lengths = [(s["end"] - s["start"]) for s in segments]
    segment_starts = [s["start"] for s in segments]
    segment_ends = [s["end"] for s in segments]
    processing_times = [s.get("processing_time", 0) for s in segments]
    
    # Segment length histogram
    plt.figure(figsize=(10, 6))
    plt.hist(segment_lengths, bins=20)
    plt.title('Segment Length Distribution')
    plt.xlabel('Segment Length (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "segment_length_histogram.png")
    
    # Segment timeline
    plt.figure(figsize=(12, 6))
    for i, seg in enumerate(segments):
        plt.plot([seg["start"], seg["end"]], [1, 1], 'b-', linewidth=2)
        plt.text(seg["start"], 1.05, f"{i+1}", fontsize=8, ha='center')
    
    plt.title('Segment Timeline')
    plt.xlabel('Time (seconds)')
    plt.yticks([])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "segment_timeline.png")
    
    # Segment processing time
    if any(t > 0 for t in processing_times):
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(processing_times)), processing_times)
        plt.title('Segment Processing Time')
        plt.xlabel('Segment ID')
        plt.ylabel('Processing Time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "segment_processing_time.png")
    
    # Save segment info dataframe
    df = pd.DataFrame({
        "segment_id": [i+1 for i in range(len(segments))],
        "start_time": segment_starts,
        "end_time": segment_ends,
        "duration": segment_lengths,
        "text_length": [len(s.get("text", "")) for s in segments],
        "processing_time": processing_times
    })
    
    df.to_csv(output_dir / "segment_info.csv", index=False)
    
    # Average segment length
    avg_length = sum(segment_lengths) / len(segment_lengths)
    with open(output_dir / "segment_stats.txt", "w") as f:
        f.write(f"Total segments: {len(segments)}\n")
        f.write(f"Average segment length: {avg_length:.2f}s\n")
        f.write(f"Minimum segment length: {min(segment_lengths):.2f}s\n")
        f.write(f"Maximum segment length: {max(segment_lengths):.2f}s\n")
        if any(t > 0 for t in processing_times):
            f.write(f"Average processing time: {sum(processing_times)/len(processing_times):.2f}s\n")
            f.write(f"Maximum processing time: {max(processing_times):.2f}s\n")

def compare_processors(video_path, output_dir, model="medium", num_workers_list=None, overlap_sec=1.0):
    """
    Compare performance between different worker counts and existing model.
    
    Args:
        video_path: Video file path
        output_dir: Result directory
        model: Model size to use
        num_workers_list: List of worker counts to test
        overlap_sec: Segment overlap duration (seconds)
    
    Returns:
        Comparison result dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if num_workers_list is None:
        # Default worker count list (1, 2, cpu_count/2, cpu_count*3/4)
        cpu_count = os.cpu_count()
        num_workers_list = [1, 2, max(2, cpu_count // 2), max(2, int(cpu_count * 0.75))]
    
    try:
        # Store results
        comparison_results = {
            "video_path": video_path,
            "model": model,
            "overlap_sec": overlap_sec,
            "parallel": {},
            "silence_based": {}
        }
        
        # 1. Parallel processing test
        for num_workers in num_workers_list:
            test_name = f"parallel_{num_workers}"
            test_dir = output_dir / test_name
            
            logger.info(f"Running parallel test (workers: {num_workers}, overlap: {overlap_sec}s)...")
            result = run_parallel_stt_test(
                video_path,
                test_dir,
                model=model,
                num_workers=num_workers,
                overlap_sec=overlap_sec
            )
            
            # Store results
            comparison_results["parallel"][test_name] = {
                "workers": num_workers,
                "processing_time": result.get("processing_time", 0),
                "text_length": result.get("text_length", 0),
                "segment_count": result.get("segment_count", 0),
                "memory_usage_mb": result.get("memory_usage_mb", 0)
            }
        
        # 2. Existing silence-based test
        try:
            from utils.stt_processor_silence import STTProcessorSilence
            
            test_name = "silence_based"
            test_dir = output_dir / test_name
            
            logger.info("Running existing silence-based STT test...")
            
            # Test with existing STT processor
            processor = STTProcessorSilence(model_name=model)
            
            # Process video
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
            
            # Store results
            comparison_results["silence_based"]["original"] = {
                "processing_time": processing_time,
                "text_length": text_length,
                "segment_count": segment_count,
                "memory_usage_mb": memory_usage
            }
            
        except ImportError:
            logger.warning("Could not find existing STT processor, skipping comparison test.")
        
        # Store results
        with open(output_dir / "comparison_results.json", "w") as f:
            json.dump(comparison_results, f, indent=2)
        
        # Visualize results
        create_comparison_visualization(comparison_results, output_dir)
        
        return comparison_results
        
    except Exception as e:
        logger.error(f"Comparison test failed: {str(e)}")
        with open(output_dir / "error.txt", "w") as f:
            f.write(str(e))
        return {"error": str(e)}

def create_comparison_visualization(comparison_results, output_dir):
    """
    Create segment comparison visualizations.
    
    Args:
        comparison_results: Comparison result dictionary
        output_dir: Output directory
    """
    # 1. Processing time comparison
    plt.figure(figsize=(12, 6))
    
    # Parallel processing results
    parallel_names = []
    parallel_times = []
    
    for name, result in comparison_results["parallel"].items():
        parallel_names.append(f"Parallel ({result['workers']})")
        parallel_times.append(result["processing_time"])
    
    # Existing silence-based results
    silence_names = []
    silence_times = []
    
    for name, result in comparison_results["silence_based"].items():
        silence_names.append("Original STT")
        silence_times.append(result["processing_time"])
    
    # Bar chart
    x = list(range(len(parallel_names) + len(silence_names)))
    plt.bar(x[:len(parallel_names)], parallel_times, color='green', alpha=0.7, label='Parallel')
    plt.bar(x[len(parallel_names):], silence_times, color='blue', alpha=0.7, label='Original')
    
    plt.title('Processing Time Comparison by Method')
    plt.ylabel('Processing Time (seconds)')
    plt.xticks(x, parallel_names + silence_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "processing_time_comparison.png")
    
    # 2. Efficiency comparison (chars/second)
    plt.figure(figsize=(12, 6))
    
    # Parallel processing efficiency
    parallel_efficiency = [result["text_length"] / result["processing_time"] 
                          if result["processing_time"] > 0 else 0 
                          for result in comparison_results["parallel"].values()]
    
    # Existing silence-based efficiency
    silence_efficiency = [result["text_length"] / result["processing_time"] 
                         if result["processing_time"] > 0 else 0 
                         for result in comparison_results["silence_based"].values()]
    
    # Bar chart
    plt.bar(x[:len(parallel_names)], parallel_efficiency, color='green', alpha=0.7, label='Parallel')
    plt.bar(x[len(parallel_names):], silence_efficiency, color='blue', alpha=0.7, label='Original')
    
    plt.title('Efficiency Comparison by Method (chars/second)')
    plt.ylabel('Efficiency (chars/second)')
    plt.xticks(x, parallel_names + silence_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "efficiency_comparison.png")
    
    # 3. Memory usage comparison
    if any("memory_usage_mb" in result for result in comparison_results["parallel"].values()):
        plt.figure(figsize=(12, 6))
        
        # Parallel processing memory usage
        parallel_memory = [result.get("memory_usage_mb", 0) for result in comparison_results["parallel"].values()]
        
        # Existing silence-based memory usage
        silence_memory = [result.get("memory_usage_mb", 0) for result in comparison_results["silence_based"].values()]
        
        # Bar chart
        plt.bar(x[:len(parallel_names)], parallel_memory, color='green', alpha=0.7, label='Parallel')
        plt.bar(x[len(parallel_names):], silence_memory, color='blue', alpha=0.7, label='Original')
        
        plt.title('Memory Usage Comparison by Method (MB)')
        plt.ylabel('Memory Usage (MB)')
        plt.xticks(x, parallel_names + silence_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "memory_usage_comparison.png")
    
    # 4. Save summary table
    data = []
    
    # Parallel processing data
    for name, result in comparison_results["parallel"].items():
        efficiency = result["text_length"] / result["processing_time"] if result["processing_time"] > 0 else 0
        data.append({
            "Method": "Parallel",
            "Setting": f"{result['workers']} workers",
            "Processing Time (s)": result["processing_time"],
            "Text Length (chars)": result["text_length"],
            "Segment Count": result["segment_count"],
            "Memory Usage (MB)": result.get("memory_usage_mb", 0),
            "Efficiency (chars/s)": efficiency
        })
    
    # Existing silence-based data
    for name, result in comparison_results["silence_based"].items():
        efficiency = result["text_length"] / result["processing_time"] if result["processing_time"] > 0 else 0
        data.append({
            "Method": "Original STT",
            "Setting": "Single thread",
            "Processing Time (s)": result["processing_time"],
            "Text Length (chars)": result["text_length"],
            "Segment Count": result["segment_count"],
            "Memory Usage (MB)": result.get("memory_usage_mb", 0),
            "Efficiency (chars/s)": efficiency
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "comparison_summary.csv", index=False)
    
    # Sort by efficiency
    df_sorted = df.sort_values(by='Efficiency (chars/s)', ascending=False)
    
    # Save best settings
    with open(output_dir / "best_settings.txt", "w") as f:
        f.write(f"Best efficiency setting: {df_sorted.iloc[0]['Setting']}\n")
        f.write(f"Method: {df_sorted.iloc[0]['Method']}\n")
        f.write(f"Efficiency: {df_sorted.iloc[0]['Efficiency (chars/s)']:.2f} chars/s\n")
        f.write(f"Processing time: {df_sorted.iloc[0]['Processing Time (s)']:.2f}s\n")
        f.write(f"Memory usage: {df_sorted.iloc[0]['Memory Usage (MB)']:.2f} MB\n")
        
        # Performance improvement compared to baseline
        if len(comparison_results["silence_based"]) > 0:
            baseline = next(iter(comparison_results["silence_based"].values()))
            baseline_time = baseline["processing_time"]
            best_time = df_sorted.iloc[0]['Processing Time (s)']
            improvement = (baseline_time - best_time) / baseline_time * 100
            f.write(f"Speed improvement over baseline: {improvement:.2f}%\n")

def main():
    parser = argparse.ArgumentParser(description="Parallel STT Testing")
    parser.add_argument("--video", type=str, required=True, help="Video file path to test")
    parser.add_argument("--output", type=str, default="./test_results_parallel", help="Output directory for results")
    parser.add_argument("--model", type=str, default="medium", help="Model size to use (tiny, base, small, medium, large)")
    parser.add_argument("--test-type", type=str, choices=["single", "compare"], default="single", 
                       help="Test type - single: single setting test, compare: compare various worker counts")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers for parallel processing")
    parser.add_argument("--overlap", type=float, default=1.0, help="Segment overlap duration (seconds)")
    
    args = parser.parse_args()
    
    # Check video file
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    logger.info(f"Starting test: {args.video}")
    logger.info(f"Test type: {args.test_type}")
    
    if args.test_type == "single":
        # Single setting test
        run_parallel_stt_test(
            args.video,
            args.output,
            model=args.model,
            num_workers=args.workers,
            overlap_sec=args.overlap
        )
    elif args.test_type == "compare":
        # Compare different worker counts
        compare_processors(
            args.video,
            args.output,
            model=args.model,
            overlap_sec=args.overlap
        )
    
    logger.info(f"Test completed. Results saved to {args.output} directory.")

if __name__ == "__main__":
    main() 