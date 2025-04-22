from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLALCHEMY_DATABASE_URL = "postgresql://lecture_qa:lecture_qa123@localhost:5433/lecture_qa_db"
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/lecture_qa_db" # 로컬 PostgreSQL 연결 설정

engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 