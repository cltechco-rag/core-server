from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.security import OAuth2PasswordBearer
from typing import Optional, Dict, Any
import requests
import os
import json
import base64
import uuid
from datetime import datetime, timedelta
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from core.database import get_db
from models.user import User
from utils.auth import get_password_hash, create_access_token
from schemas.notion_auth import NotionAuthCode, NotionTokenResponse, UserResponse
from config.config import settings

router = APIRouter(tags=["notion"])

# 중복 인증 요청 방지를 위한 처리된 코드 추적
processed_auth_codes = set()

# 환경 변수에서 설정을 가져옵니다
NOTION_CLIENT_ID = settings.NOTION_CLIENT_ID
# 올바른 시크릿 값으로 하드코딩합니다
NOTION_CLIENT_SECRET = settings.NOTION_CLIENT_SECRET
NOTION_REDIRECT_URI = settings.NOTION_REDIRECT_URI
NOTION_API_VERSION = "2022-06-28"

print(f"=== Notion Configuration ===")
print(f"NOTION_CLIENT_ID: {NOTION_CLIENT_ID}")
print(f"NOTION_CLIENT_SECRET: {'*' * 8 if NOTION_CLIENT_SECRET else 'Not set'}")
print(f"NOTION_REDIRECT_URI: {NOTION_REDIRECT_URI}")
print(f"NOTION_API_VERSION: {NOTION_API_VERSION}")
print(f"===========================")

if not NOTION_CLIENT_ID or not NOTION_CLIENT_SECRET:
    raise Exception("Notion credentials not properly configured. Please check your environment variables.")

# JWT 설정
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/notion/callback", response_model=UserResponse)
async def notion_callback(request: Request, db: Session = Depends(get_db)):
    try:
        # 요청 본문 파싱
        body = await request.json()
        code = body.get('code')
        
        print(f"\n=== Notion Callback Process Started ===")
        print(f"Received auth code: {code}")
        
        if not code:
            raise HTTPException(
                status_code=400,
                detail="Authorization code is required"
            )
            
        # 이미 처리된 인증 코드인지 확인
        if code in processed_auth_codes:
            print(f"Auth code already processed: {code[:8]}...")
            raise HTTPException(
                status_code=400,
                detail="This authorization code has already been used"
            )
            
        # 인증 코드를 처리된 목록에 추가
        processed_auth_codes.add(code)
        
        # 목록이 너무 커지지 않도록 관리 (옵션)
        if len(processed_auth_codes) > 100:
            # 가장 오래된 코드부터 50개 제거
            processed_auth_codes.clear()
        
        # Notion OAuth 토큰 엔드포인트
        token_url = "https://api.notion.com/v1/oauth/token"
        
        # OAuth 요청 데이터 준비
        token_request_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": NOTION_REDIRECT_URI
        }
        
        # HTTP Basic Auth 위한 인증 문자열 생성 (JavaScript 예시처럼 Base64 인코딩)
        encoded = base64.b64encode(f"{NOTION_CLIENT_ID}:{NOTION_CLIENT_SECRET}".encode('utf-8')).decode('utf-8')
        
        # Notion OAuth 요청 헤더 준비 (JavaScript 예시와 동일하게 구성)
        token_request_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Basic {encoded}",
            "Notion-Version": NOTION_API_VERSION
        }
        
        print(f"\n=== Token Request Details ===")
        print(f"URL: {token_url}")
        print(f"Headers: Accept: application/json, Content-Type: application/json, Notion-Version: {NOTION_API_VERSION}")
        print(f"Authorization Header: Basic {encoded}")  # 전체 인코딩된 값 출력
        print(f"Raw Client:Secret for encoding: {NOTION_CLIENT_ID}:{NOTION_CLIENT_SECRET}")  # 인코딩 전 값 출력
        print(f"Using Basic Auth with Client ID: {NOTION_CLIENT_ID[:8]}...")
        print(f"Request Data: {json.dumps({**token_request_data, 'code': code[:8] + '...' if code and len(code) > 8 else code})}")
        
        # Notion OAuth 토큰 요청
        print(f"\n=== Sending Request to Notion API ===")
        token_response = requests.post(
            token_url,
            headers=token_request_headers,
            json=token_request_data
        )
        
        print(f"\n=== Token Response ===")
        print(f"Status: {token_response.status_code}")
        print(f"Headers: {json.dumps(dict(token_response.headers), indent=2)}")
        
        if token_response.status_code != 200:
            error_content = token_response.json() if token_response.text else {"error": "Unknown error"}
            print(f"Error Response: {json.dumps(error_content, indent=2)}")
            print(f"Request Headers Sent: {json.dumps({k: v for k, v in token_request_headers.items()}, indent=2)}")
            print(f"Request Data Sent: {json.dumps(token_request_data, indent=2)}")
            
            error_detail = {
                "message": "Failed to get Notion access token",
                "error": error_content,
                "status_code": token_response.status_code,
                "request_data": {
                    "url": token_url,
                    "redirect_uri": NOTION_REDIRECT_URI,
                    "client_id_prefix": NOTION_CLIENT_ID[:8] if NOTION_CLIENT_ID else None
                }
            }
            
            raise HTTPException(
                status_code=token_response.status_code,
                detail=error_detail
            )
        
        # 토큰 응답 파싱
        token_data = token_response.json()
        print(f"Token Data: {json.dumps({k: v if k != 'access_token' else '****' for k, v in token_data.items()})}")
        
        access_token = token_data.get("access_token")
        
        if not access_token:
            raise HTTPException(
                status_code=400,
                detail="No access token in Notion response"
            )
        
        print(f"\n=== Successfully Retrieved Access Token ===")
        
        # Notion 사용자 정보 요청
        user_info_url = "https://api.notion.com/v1/users/me"
        user_info_headers = {
            "Authorization": f"Bearer {access_token}",
            "Notion-Version": NOTION_API_VERSION,
            "Accept": "application/json"
        }
        
        print(f"\n=== User Info Request ===")
        print(f"URL: {user_info_url}")
        
        user_response = requests.get(
            user_info_url,
            headers=user_info_headers
        )
        
        print(f"User Info Status: {user_response.status_code}")
        
        if user_response.status_code != 200:
            error_content = user_response.json() if user_response.text else {"error": "Unknown error"}
            print(f"User Info Error: {json.dumps(error_content, indent=2)}")
            raise HTTPException(
                status_code=user_response.status_code,
                detail=f"Failed to get user information: {error_content}"
            )
        
        # 사용자 정보 파싱
        user_info = user_response.json()
        print(f"User Info Raw Response: {json.dumps(user_info)}")
        
        # Notion 응답에서 사용자 정보 추출
        email = None
        name = None
        
        # owner.user 객체 또는 기타 형식의 응답 처리
        if 'bot' in user_info:
            owner = user_info.get('bot', {}).get('owner', {})
            if 'user' in owner:
                user_data = owner.get('user', {})
                email = user_data.get('person', {}).get('email')
                name = user_data.get('name')
        elif 'email' in user_info:
            # 직접 사용자 정보가 제공된 경우
            email = user_info.get('email')
            name = user_info.get('name')
        
        # 사용자 타입에 따라 정보가 다른 필드에 있을 수 있음
        if not email and 'owner' in user_info:
            owner = user_info.get('owner', {})
            if 'user' in owner:
                user_data = owner.get('user', {})
                email = user_data.get('person', {}).get('email')
                name = user_data.get('name')
        
        # 이메일이 없는 경우 대안적으로 ID를 사용할 수 있음
        if not email:
            # ID를 이메일 형식으로 변환 (임시 방법)
            user_id = user_info.get('id')
            if user_id:
                email = f"{user_id}@notion.user"
                print(f"No email found in response, using ID as email: {email}")
        
        print(f"Extracted Email: {email}")
        print(f"Extracted Name: {name or 'Not provided'}")
        
        if not email:
            raise HTTPException(
                status_code=400,
                detail="Could not extract email from user information"
            )
            
        # 사용자 확인 또는 생성
        print(f"\n=== Checking User in Database ===")
        user = db.query(User).filter(User.email == email).first()
        
        if not user:
            print(f"User not found, creating new user with email: {email}")
            # 랜덤 비밀번호 생성 (실제 로그인에는 사용되지 않음)
            random_password = str(uuid.uuid4())
            hashed_password = get_password_hash(random_password)
            
            # 사용자 이름 생성 (없는 경우 이메일 앞부분 사용)
            username = name
            if not username:
                username = email.split('@')[0]
                
            # 중복 확인
            existing_username = db.query(User).filter(User.username == username).first()
            if existing_username:
                username = f"{username}_{str(uuid.uuid4())[:8]}"
            
            # 새 사용자 생성
            user = User(
                email=email,
                username=username,
                password=hashed_password  # OAuth 사용자는 비밀번호로 로그인하지 않음
            )
            
            # 데이터베이스에 저장
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"Created new user: {user.email} with username: {user.username}")
        else:
            print(f"User found in database: {user.email}")
        
        # JWT 토큰 생성
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        jwt_token = create_access_token(
            data={"sub": user.email, "user_id": user.id},
            expires_delta=access_token_expires
        )
        
        print(f"\n=== Authentication Successful ===")
        print(f"JWT token created for user: {user.email}")
        
        # 응답 생성
        return UserResponse(
            email=user.email,
            name=user.username,
            access_token=jwt_token
        )
        
    except json.JSONDecodeError as e:
        print(f"\n=== JSON Decode Error ===")
        print(f"Error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON in request body"
        )
    except requests.exceptions.RequestException as e:
        print(f"\n=== Request Error ===")
        print(f"Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to communicate with Notion API: {str(e)}"
        )
    except Exception as e:
        print(f"\n=== Unexpected Error ===")
        import traceback
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=str(e) or "An unexpected error occurred"
        ) 