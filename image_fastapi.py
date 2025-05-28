# 맨 위에 추가 (import chromadb 전에)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# 기존 검색 함수 import
from image_search import search_similar_images

# FastAPI 앱 생성
app = FastAPI(
    title="Image Similarity Search API",
    description="CLIP 기반 이미지 유사도 검색 서비스",
    version="1.0.0"
)

# 요청 모델
class ImageSearchRequest(BaseModel):
    image_url: str

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
        request: 검색할 이미지 URL
        
    Returns:
        List[str]: 유사한 이미지 ID 3개
    """
    try:
        results = search_similar_images(request.image_url)
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

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )