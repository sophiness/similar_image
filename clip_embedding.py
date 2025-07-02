import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import chromadb
from typing import List, Optional, Dict
from enum import Enum

class AnimalType(Enum):
    """동물 종류 열거형"""
    CAT = "cat"
    DOG = "dog"

class MultiAnimalSearchEngine:
    """다중 동물 이미지 유사도 검색 엔진"""
    
    def __init__(self, db_base_path: str = "./image_embeddings_db"):
        """
        초기화
        
        Args:
            db_base_path: ChromaDB 기본 저장 경로
        """
        self.db_base_path = db_base_path
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.collections: Dict[str, any] = {}  # 동물별 컬렉션 저장
        self._initialize_clip_model()
        self._initialize_databases()
    
    def _initialize_clip_model(self):
        """CLIP 모델 초기화 (공통 사용)"""
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        print("✅ CLIP model loaded")
    
    def _initialize_databases(self):
        """각 동물별 ChromaDB 초기화"""
        for animal in AnimalType:
            db_path = f"{self.db_base_path}/{animal.value}_db"
            client = chromadb.PersistentClient(path=db_path)
            collection = client.get_or_create_collection(
                name=f"{animal.value}_images",
                metadata={"hnsw:space": "cosine"}
            )
            self.collections[animal.value] = collection
            print(f"✅ {animal.value.capitalize()} ChromaDB created/connected")
    
    def download_image_from_url(self, image_url: str) -> Image.Image:
        """웹 URL에서 이미지 다운로드"""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"이미지 다운로드 실패: {e}")
    
    def extract_image_embedding(self, image: Image.Image) -> np.ndarray:
        """이미지에서 CLIP 임베딩 추출"""
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            embedding = outputs.cpu().numpy().squeeze()
        
        # 임베딩 정규화
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def add_image_to_db(self, image_url: str, animal_type: str):
        """이미지를 DB에 추가"""
        collection = self.collections[animal_type]
        
        # 1. 이미지 다운로드 및 임베딩 추출
        image = self.download_image_from_url(image_url)
        embedding = self.extract_image_embedding(image)
        
        # 2. ChromaDB에 저장
        collection.add(
            ids=[image_url],
            embeddings=[embedding.tolist()]
        )
        
        print(f"✅ {image_url} added to {animal_type} DB")

# 인스턴스 생성
search_engine = MultiAnimalSearchEngine()

print("🚀 고양이 DB 구축 시작")
# cat_image_url.txt 파일 읽어서 DB에 저장
with open('cat_image_url.txt', 'r') as f:
    for i, line in enumerate(f, 1):
        url = line.strip()
        if url:  # 빈 줄 건너뛰기
            try:
                search_engine.add_image_to_db(url, "cat")
                print(f"[{i}] 완료")
            except Exception as e:
                print(f"[{i}] 실패: {url} - {e}")

print("\n🚀 강아지 DB 구축 시작")
# dog_image_url.txt 파일 읽어서 DB에 저장
with open('dog_image_url.txt', 'r') as f:
    for i, line in enumerate(f, 1):
        url = line.strip()
        if url:  # 빈 줄 건너뛰기
            try:
                search_engine.add_image_to_db(url, "dog")
                print(f"[{i}] 완료")
            except Exception as e:
                print(f"[{i}] 실패: {url} - {e}")

print("\n🎉 DB 구축 완료!")
print(f"고양이 DB: {search_engine.collections['cat'].count()}개 이미지")
print(f"강아지 DB: {search_engine.collections['dog'].count()}개 이미지")