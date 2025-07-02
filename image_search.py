import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import chromadb

class ImageSearchEngine:
    def __init__(self):
        self.model = None
        self.processor = None
        self.collections = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """모델과 ChromaDB 초기화 (한 번만 실행)"""
        # CLIP 모델 로드
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        print("✅ CLIP model loaded")
        
        # 각 동물별 ChromaDB 연결
        client_cat = chromadb.PersistentClient(path="./image_embeddings_db/cat_db")
        self.collections["cat"] = client_cat.get_collection("cat_images")
        print("✅ Cat ChromaDB connected")
        
        client_dog = chromadb.PersistentClient(path="./image_embeddings_db/dog_db")
        self.collections["dog"] = client_dog.get_collection("dog_images")
        print("✅ Dog ChromaDB connected")

    def download_image_from_url(self, image_url: str) -> Image.Image:
        """웹 URL에서 이미지 다운로드"""
        response = requests.get(image_url, timeout=10, verify=False)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image

    def extract_query_embedding(self, image: Image.Image) -> np.ndarray:
        """쿼리 이미지에서 CLIP 임베딩 추출"""
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            embedding = outputs.cpu().numpy().squeeze()
        
        # 임베딩 정규화
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding

    def search_chromadb(self, query_embedding: np.ndarray, animal_type: str) -> list:
        """ChromaDB에서 유사도 검색"""
        collection = self.collections[animal_type]
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )    
        image_urls = results['ids'][0]

        return image_urls

    def search_similar_images(self, image_url: str, animal_type: str) -> list:
        """
        메인 함수: 이미지 URL을 받아서 유사한 이미지 ID 3개 반환
        
        Args:
            image_url (str): 검색할 이미지의 웹 URL
            animal_type (str): 동물 종류 ("cat" 또는 "dog")
            
        Returns:
            list: 유사한 이미지 ID 리스트(url)
        """
        image = self.download_image_from_url(image_url)
        query_embedding = self.extract_query_embedding(image)
        similar_ids = self.search_chromadb(query_embedding, animal_type)
        
        return similar_ids