#!/bin/bash

# 병렬 처리 기반 STT 테스트 스크립트

# 테스트 파일 경로
VIDEO_DIR="./test_videos"
RESULT_DIR="./test_results_hybrid"

# Python 3.10 경로 확인
PYTHON="python3.10"
if ! command -v $PYTHON &> /dev/null; then
    echo "$PYTHON을 찾을 수 없습니다. Python 3.10이 설치되어 있는지 확인해주세요."
    exit 1
fi

# 테스트 비디오 파일이 없는 경우 안내
if [ ! -d "$VIDEO_DIR" ]; then
    echo "테스트 비디오 디렉토리($VIDEO_DIR)가 존재하지 않습니다."
    echo "다음 명령어로 디렉토리를 생성하고 테스트할 비디오 파일을 넣어주세요:"
    echo "  mkdir -p $VIDEO_DIR"
    exit 1
fi

# 테스트 비디오 파일 목록 확인
VIDEO_FILES=$(find "$VIDEO_DIR" -type f -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv")
if [ -z "$VIDEO_FILES" ]; then
    echo "테스트할 비디오 파일을 찾을 수 없습니다."
    echo "$VIDEO_DIR 디렉토리에 비디오 파일(.mp4, .avi, .mov, .mkv)을 넣어주세요."
    exit 1
fi

# 결과 디렉토리 생성
mkdir -p "$RESULT_DIR"
echo "테스트 결과는 $RESULT_DIR 디렉토리에 저장됩니다."

# 필요한 디렉토리 생성
mkdir -p uploads/audio uploads/cache uploads/models uploads/temp

# 필요한 패키지 확인
echo "필요한 패키지 확인 중..."
$PYTHON -c "
required_packages = ['torch', 'whisper', 'matplotlib', 'pandas', 'numpy', 'psutil', 'librosa', 'tqdm', 'concurrent.futures']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f'✅ {package} 패키지가 설치되어 있습니다.')
    except ImportError:
        missing_packages.append(package)
        print(f'❌ {package} 패키지가 없습니다.')

if missing_packages:
    print(f'\n다음 패키지를 설치해야 합니다: {missing_packages}')
    print('명령어: pip install ' + ' '.join(missing_packages))
    exit(1)
else:
    print('\n모든 필요한 패키지가 설치되어 있습니다.')
"

if [ $? -ne 0 ]; then
    echo "패키지 확인 중 오류가 발생했습니다. 필요한 패키지를 설치한 후 다시 시도해주세요."
    exit 1
fi

# CPU 코어 수 확인
CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "CPU 코어 수: $CPU_CORES"

# 테스트 유형 선택
echo "[ 병렬 처리 STT 테스트 ]"
echo "stt_processor_parallel.py를 사용하여 병렬 처리 기반의 STT 테스트를 수행합니다."
echo ""

# 테스트 선택 옵션
PS3="테스트 옵션을 선택하세요: "
options=("단일 비디오 테스트" "모든 비디오 테스트" "비교 테스트" "취소")
select opt in "${options[@]}"
do
    case $opt in
        "단일 비디오 테스트")
            # 비디오 목록 표시
            declare -a video_list
            i=1
            for video_file in $VIDEO_FILES; do
                video_name=$(basename "$video_file")
                video_list[$i]="$video_file"
                echo "$i) $video_name"
                ((i++))
            done
            
            echo "테스트할 비디오 번호를 선택하세요 (1-$((i-1))): "
            read -r video_num
            
            if [[ $video_num -ge 1 && $video_num -lt $i ]]; then
                selected_video="${video_list[$video_num]}"
                
                echo "선택된 비디오: $selected_video"
                
                # 추가 매개변수 설정
                echo "모델 크기를 선택하세요 (tiny, base, small, medium, large) [기본값: medium]: "
                read -r model_size
                model_size=${model_size:-medium}
                
                echo "작업자 수를 입력하세요 (최대 권장: $CPU_CORES) [기본값: 자동 설정(코어 수의 3/4)]: "
                read -r workers
                
                timestamp=$(date +%Y%m%d_%H%M%S)
                output_dir="$RESULT_DIR/$(basename "$selected_video" | cut -d. -f1)_${timestamp}"
                
                echo "테스트 실행 중..."
                if [ -n "$workers" ]; then
                    $PYTHON test_stt_parallel.py --video "$selected_video" --output "$output_dir" \
                        --model "$model_size" --test-type single --workers "$workers"
                else
                    $PYTHON test_stt_parallel.py --video "$selected_video" --output "$output_dir" \
                        --model "$model_size" --test-type single
                fi
            else
                echo "잘못된 선택입니다."
                exit 1
            fi
            break
            ;;
        "모든 비디오 테스트")
            echo "모든 비디오에 대해 테스트를 실행합니다..."
            
            # 추가 매개변수 설정
            echo "모델 크기를 선택하세요 (tiny, base, small, medium, large) [기본값: medium]: "
            read -r model_size
            model_size=${model_size:-medium}
            
            echo "작업자 수를 입력하세요 (최대 권장: $CPU_CORES) [기본값: 자동 설정(코어 수의 3/4)]: "
            read -r workers
            
            for video_file in $VIDEO_FILES; do
                video_name=$(basename "$video_file")
                timestamp=$(date +%Y%m%d_%H%M%S)
                output_dir="$RESULT_DIR/${video_name%.*}_${timestamp}"
                
                echo "비디오 테스트: $video_name"
                if [ -n "$workers" ]; then
                    $PYTHON test_stt_parallel.py --video "$video_file" --output "$output_dir" \
                        --model "$model_size" --test-type single --workers "$workers"
                else
                    $PYTHON test_stt_parallel.py --video "$video_file" --output "$output_dir" \
                        --model "$model_size" --test-type single
                fi
            done
            break
            ;;
        "비교 테스트")
            # 비디오 목록 표시
            declare -a video_list
            i=1
            for video_file in $VIDEO_FILES; do
                video_name=$(basename "$video_file")
                video_list[$i]="$video_file"
                echo "$i) $video_name"
                ((i++))
            done
            
            echo "테스트할 비디오 번호를 선택하세요 (1-$((i-1))): "
            read -r video_num
            
            if [[ $video_num -ge 1 && $video_num -lt $i ]]; then
                selected_video="${video_list[$video_num]}"
                
                echo "선택된 비디오: $selected_video"
                
                # 모델 크기 설정
                echo "모델 크기를 선택하세요 (tiny, base, small, medium, large) [기본값: small]: "
                read -r model_size
                model_size=${model_size:-small}
                
                timestamp=$(date +%Y%m%d_%H%M%S)
                output_dir="$RESULT_DIR/$(basename "$selected_video" | cut -d. -f1)_compare_${timestamp}"
                
                echo "다양한 작업자 수 비교 테스트를 실행 중..."
                echo "이 테스트는 단일 스레드, 2개 스레드, 및 CPU 코어 수에 맞는 스레드로 처리 시간을 비교합니다."
                
                $PYTHON test_stt_parallel.py --video "$selected_video" --output "$output_dir" \
                    --model "$model_size" --test-type compare
                
                echo "기존 침묵 기반 STT 프로세서와의 성능 차이는 $output_dir/best_settings.txt 파일에서 확인할 수 있습니다."
            else
                echo "잘못된 선택입니다."
                exit 1
            fi
            break
            ;;
        "취소")
            echo "테스트를 취소합니다."
            exit 0
            ;;
        *) 
            echo "잘못된 선택입니다. 1-4 사이의 숫자를 입력하세요."
            ;;
    esac
done

echo "========================================================"
echo "모든 테스트가 완료되었습니다."
echo "결과는 $RESULT_DIR 디렉토리에서 확인할 수 있습니다."
echo "========================================================" 

# 스크립트 실행 권한
chmod +x test_stt_parallel.py 