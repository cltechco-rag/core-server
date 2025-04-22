#!/bin/bash

# 필요한 환경 확인
if ! command -v python3.10 &> /dev/null; then
    echo "Python 3.10이 설치되어 있지 않습니다. 설치 후 다시 시도해주세요."
    exit 1
fi

# yt-dlp 설치 확인 및 설치
if ! python3.10 -m pip list | grep -q "yt-dlp"; then
    echo "yt-dlp 패키지를 설치합니다..."
    python3.10 -m pip install yt-dlp
fi

# 필요한 디렉토리 확인
if [ ! -d "test_videos" ]; then
    mkdir -p test_videos
fi

# 사용법 확인
if [ "$#" -ne 1 ]; then
    echo "사용법: $0 <유튜브_URL>"
    exit 1
fi

YOUTUBE_URL="$1"
DOWNLOAD_PATH="test_videos/"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VIDEO_FILENAME="youtube_${TIMESTAMP}.mp4"

echo "유튜브 영상을 다운로드 중입니다..."
python3.10 -m yt_dlp -f "best[ext=mp4]/mp4" -o "${DOWNLOAD_PATH}/${VIDEO_FILENAME}" "$YOUTUBE_URL"

if [ $? -ne 0 ]; then
    echo "비디오 다운로드에 실패했습니다."
    exit 1
fi

echo "비디오 다운로드 완료: ${DOWNLOAD_PATH}/${VIDEO_FILENAME}"

# 비디오 정보 출력
echo "비디오 정보:"
if command -v ffprobe &> /dev/null; then
    ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${DOWNLOAD_PATH}/${VIDEO_FILENAME}" 2>/dev/null | {
        read DURATION
        if [ -n "$DURATION" ]; then
            DURATION=$(printf "%.0f" "$DURATION")
            MIN=$((DURATION / 60))
            SEC=$((DURATION % 60))
            echo "- 길이: ${MIN}분 ${SEC}초"
        else
            echo "- 길이 정보를 가져올 수 없습니다"
        fi
    }
else
    echo "- ffprobe가 설치되어 있지 않아 길이 정보를 가져올 수 없습니다"
fi

echo "다운로드한 영상은 ${DOWNLOAD_PATH}/${VIDEO_FILENAME}에 저장되었습니다." 