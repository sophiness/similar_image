# 맨 위에 추가 (import chromadb 전에)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import chromadb

# 전역 변수로 모델과 ChromaDB 연결 (한 번만 로드)
_model = None
_processor = None
_collection = None

def initialize_models():
    """모델과 ChromaDB 초기화 (한 번만 실행)"""
    global _model, _processor, _collection
    
    if _model is None:
        # CLIP 모델 로드
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model.eval()
    
    if _collection is None:
        # ChromaDB 연결
        client = chromadb.PersistentClient(path="./image_embeddings_db")
        _collection = client.get_collection("image_embeddings")

def download_image_from_url(image_url: str) -> Image.Image:
    """웹 URL에서 이미지 다운로드"""
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception(f"이미지 다운로드 실패: HTTP {response.status_code}")
    
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return image

def extract_query_embedding(image: Image.Image) -> np.ndarray:
    """쿼리 이미지에서 CLIP 임베딩 추출"""
    global _model, _processor
    
    inputs = _processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = _model.get_image_features(**inputs)
        embedding = outputs.cpu().numpy().squeeze()
    
    return embedding

def search_chromadb(query_embedding: np.ndarray) -> list:
    """ChromaDB에서 유사도 검색"""
    global _collection
    
    results = _collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )
    
    return results['ids'][0] if results['ids'] else []

def search_similar_images(image_url: str) -> list:
    """
    메인 함수: 이미지 URL을 받아서 유사한 이미지 ID 3개 반환
    
    Args:
        image_url (str): 검색할 이미지의 웹 URL
        
    Returns:
        list: 유사한 이미지 ID 리스트 (예: ['cat_1', 'cat_5', 'cat_12'])
    """
    initialize_models()
    
    image = download_image_from_url(image_url)
    query_embedding = extract_query_embedding(image)
    similar_ids = search_chromadb(query_embedding)
    
    return similar_ids