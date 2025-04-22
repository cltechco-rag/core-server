from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from modules.upload.router import router as upload_router
from open_ai.router import router as openai_router
from app.routers.auth import router as auth_router
from core.database import Base, engine
from models.video import Video
from models.transcript import Transcript
import os
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()
app = FastAPI(
    title="Lecture QA Platform API",
    description="""
    강의 영상 업로드 및 QA 시스템 API
    
    ## 주요 기능
    * 영상 파일 업로드 (최대 700MB)
    * 음성-텍스트 변환 (STT)
    * 텍스트 임베딩 및 Vector DB 저장
    * RAG 기반 QA 시스템
    * 회의록 요약 기능
    
    ## API 문서
    * Swagger UI: `/docs`
    * ReDoc: `/redoc`
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 구체적인 origin을 지정하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터베이스 초기화
@app.on_event("startup")
async def startup():
    Base.metadata.create_all(bind=engine)

# API 라우터 등록
app.include_router(upload_router)
app.include_router(openai_router)
app.include_router(auth_router, prefix="/auth", tags=["authentication"])

@app.get("/")
async def root():
    return {
        "message": "Welcome to Lecture QA Platform API",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# 1. 회의록 텍스트 파일 읽기
def load_meeting_transcript(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 2. AzureOpenAI로 요약 요청
def summarize_with_azure_openai(content: str):
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

# 3. FastAPI 엔드포인트
@app.post("/summarize")
async def summarize_meeting(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        content = load_meeting_transcript(file_location)
        summary = summarize_with_azure_openai(content)
        return JSONResponse(content={"summary": summary})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

if __name__ == "__main__":
    import uvicorn
    import logging
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("서버 시작 중...")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        logger.error(f"서버 시작 실패: {str(e)}", exc_info=True) 