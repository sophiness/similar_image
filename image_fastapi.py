# # 맨 위에 추가 (import chromadb 전에)
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# 기존 검색 클래스 import
from image_search import ImageSearchEngine

# FastAPI 앱 생성
app = FastAPI(
    title="Image Similarity Search API",
    description="CLIP 기반 이미지 유사도 검색 서비스",
    version="1.0.0"
)

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "https://localhost:5173", 
        "http://127.0.0.1:5173",
        "https://127.0.0.1:5173", 
        "http://www.meowng.com", 
        "https://www.meowng.com", 
        "https://ds36vr51hmfa7.cloudfront.net", 
        "http://3.39.3.208", 
        "http://172.20.5.64:5173", 
        "http://testdev.meowng.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)


# 검색 엔진 인스턴스 생성 (앱 시작 시 한 번만)
search_engine = ImageSearchEngine()

# 요청 모델
class ImageSearchRequest(BaseModel):
    image_url: str
    animal: str

@app.get("/")
async def root():
    """기본 엔드포인트"""
    return {
        "message": "Image Similarity Search API",
        "status": "running",
        "docs": "/docs"
    }

@app.post("/search", response_model=List[str])
async def search_images(request: ImageSearchRequest):
    """
    이미지 유사도 검색
    
    Args:
        request: 검색할 이미지 URL과 동물 종류
        
    Returns:
        List[str]: 유사한 이미지 ID 3개
    """
    try:
        results = search_engine.search_similar_images(request.image_url, request.animal)
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"검색 실패: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """헬스체크"""
    return {
        "status": "healthy",
        "service": "image-similarity-search"
    }
