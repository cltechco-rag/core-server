import os
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import AzureOpenAI
import httpx
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
router = APIRouter(prefix="/openai", tags=["openai"])

# 1. 회의록 텍스트 파일 읽기
def load_meeting_transcript(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 2. AzureOpenAI로 요약 요청
def summarize_with_azure_openai(content: str):
    # 환경 변수 로깅
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    logger.debug(f"Endpoint: {endpoint}")
    logger.debug(f"Deployment: {deployment}")
    logger.debug(f"API Version: {api_version}")
    logger.debug(f"API Key exists: {bool(api_key)}")

    if not all([endpoint, deployment, api_key, api_version]):
        raise ValueError("필요한 환경 변수가 설정되지 않았습니다.")

    # Azure OpenAI 클라이언트 초기화
    # 수정된 코드 (정상 동작)
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )

    try:
        logger.debug(f"API 호출 시작 - 모델: {deployment}")
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "당신은 회의록 요약 어시스턴트입니다. 내용을 요약해주세요."},
                {"role": "user", "content": content}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        logger.debug("API 호출 성공")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Azure OpenAI API 호출 중 오류 발생: {str(e)}", exc_info=True)
        raise

# 3. FastAPI 라우터
@router.post("/summarize", summary="회의록 요약", description="텍스트 파일을 업로드하면 내용을 요약합니다.")
async def summarize_meeting(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    try:
        # 파일 저장
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 파일 읽기 및 요약
        content = load_meeting_transcript(file_location)
        summary = summarize_with_azure_openai(content)
        
        return JSONResponse(content={"summary": summary})
    except ValueError as e:
        logger.error(f"환경 변수 오류: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "message": "환경 변수 설정이 잘못되었습니다."
            }
        )
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500, 
            content={
                "error": str(e),
                "message": "회의록 요약 중 오류가 발생했습니다."
            }
        )
    finally:
        # 임시 파일 삭제
        if os.path.exists(file_location):
            os.remove(file_location) 