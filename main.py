from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from modules.upload.router import router as upload_router
from core.database import Base, engine
from models.video import Video
from models.transcript import Transcript

app = FastAPI(
    title="Lecture QA Platform API",
    description="""
    강의 영상 업로드 및 QA 시스템 API
    
    ## 주요 기능
    * 영상 파일 업로드 (최대 700MB)
    * 음성-텍스트 변환 (STT)
    * 텍스트 임베딩 및 Vector DB 저장
    * RAG 기반 QA 시스템
    
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

@app.get("/")
async def root():
    return {
        "message": "Welcome to Lecture QA Platform API",
        "docs": "/docs",
        "redoc": "/redoc"
    }

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
        uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
    except Exception as e:
        logger.error(f"서버 시작 실패: {str(e)}", exc_info=True) 