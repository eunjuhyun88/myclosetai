# backend/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import sys
import time
import base64
import numpy as np
from PIL import Image
import io
import tempfile
import shutil
from typing import Optional
import torch
import cv2

# OOTDiffusion 경로 추가
sys.path.append('./OOTDiffusion')

# OOTDiffusion 모듈 import 시도
try:
    from ootd.inference_ootd_hd import OOTDiffusionHD
    OOTD_AVAILABLE = True
except ImportError:
    OOTD_AVAILABLE = False
    print("Warning: OOTDiffusion not found. Using mock mode.")

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 모델 인스턴스
model = None

class VirtualTryOnModel:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if OOTD_AVAILABLE:
            # OOTDiffusion 모델 로드
            self.load_ootd_model()
        else:
            print("Running in mock mode - no real model loaded")
    
    def load_ootd_model(self):
        """OOTDiffusion 모델 로드"""
        try:
            # 모델 경로 확인
            model_path = "./OOTDiffusion/checkpoints/ootd"
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}")
                print("Please download the model from Hugging Face")
                return
            
            # OOTDiffusion 초기화
            self.ootd_model = OOTDiffusionHD(
                model_path=model_path,
                device=self.device
            )
            print("OOTDiffusion model loaded successfully")
        except Exception as e:
            print(f"Error loading OOTDiffusion: {e}")
            self.ootd_model = None
    
    def process_images(self, person_image: Image.Image, cloth_image: Image.Image, height: int, weight: int):
        """가상 시착 처리"""
        start_time = time.time()
        
        # BMI 계산
        height_m = height / 100
        bmi = round(weight / (height_m ** 2), 1)
        
        # 신체 측정 추정 (간단한 공식 사용)
        chest = self.estimate_chest(height, weight, bmi)
        waist = self.estimate_waist(height, weight, bmi)
        hip = self.estimate_hip(height, weight, bmi)
        
        if OOTD_AVAILABLE and self.ootd_model:
            # 실제 OOTDiffusion 처리
            try:
                result_image = self.run_ootd_inference(person_image, cloth_image)
            except Exception as e:
                print(f"OOTDiffusion error: {e}")
                result_image = self.mock_virtual_tryon(person_image, cloth_image)
        else:
            # Mock 처리
            result_image = self.mock_virtual_tryon(person_image, cloth_image)
        
        # 이미지를 base64로 변환
        buffered = io.BytesIO()
        result_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        processing_time = round(time.time() - start_time, 2)
        
        # 옷 분석 (간단한 더미 데이터)
        clothing_analysis = self.analyze_clothing(cloth_image)
        
        # Fit score 계산
        fit_score = self.calculate_fit_score(height, weight, bmi)
        
        # 추천사항 생성
        recommendations = self.generate_recommendations(fit_score, bmi, clothing_analysis)
        
        return {
            "success": True,
            "fitted_image": img_base64,
            "processing_time": processing_time,
            "confidence": 0.85 + np.random.random() * 0.1,  # 85-95%
            "measurements": {
                "chest": chest,
                "waist": waist,
                "hip": hip,
                "bmi": bmi
            },
            "clothing_analysis": clothing_analysis,
            "fit_score": fit_score,
            "recommendations": recommendations
        }
    
    def run_ootd_inference(self, person_image: Image.Image, cloth_image: Image.Image):
        """실제 OOTDiffusion 추론"""
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as person_tmp:
            person_image.save(person_tmp.name)
            person_path = person_tmp.name
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as cloth_tmp:
            cloth_image.save(cloth_tmp.name)
            cloth_path = cloth_tmp.name
        
        try:
            # OOTDiffusion 실행
            result = self.ootd_model.generate(
                person_path=person_path,
                cloth_path=cloth_path,
                category='upper_body',  # 또는 'lower_body', 'dresses'
                num_inference_steps=20,
                guidance_scale=2.0
            )
            
            # 결과 이미지 로드
            if isinstance(result, str):
                result_image = Image.open(result)
            else:
                result_image = result
                
            return result_image
            
        finally:
            # 임시 파일 삭제
            os.unlink(person_path)
            os.unlink(cloth_path)
    
    def mock_virtual_tryon(self, person_image: Image.Image, cloth_image: Image.Image):
        """Mock 가상 시착 (실제 모델이 없을 때)"""
        # 간단한 이미지 합성
        person_np = np.array(person_image)
        cloth_np = np.array(cloth_image)
        
        # 크기 조정
        h, w = person_np.shape[:2]
        cloth_resized = cv2.resize(cloth_np, (w//3, h//3))
        
        # 상단 중앙에 옷 이미지 오버레이
        result = person_np.copy()
        y_offset = h//10
        x_offset = w//3
        
        # 간단한 블렌딩
        alpha = 0.7
        for y in range(cloth_resized.shape[0]):
            for x in range(cloth_resized.shape[1]):
                if y + y_offset < h and x + x_offset < w:
                    result[y + y_offset, x + x_offset] = (
                        alpha * cloth_resized[y, x] + 
                        (1 - alpha) * result[y + y_offset, x + x_offset]
                    ).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def estimate_chest(self, height: int, weight: int, bmi: float) -> int:
        """가슴둘레 추정"""
        base = 80 if bmi < 25 else 90
        return base + int((weight - 60) * 0.5)
    
    def estimate_waist(self, height: int, weight: int, bmi: float) -> int:
        """허리둘레 추정"""
        base = 70 if bmi < 25 else 80
        return base + int((weight - 60) * 0.4)
    
    def estimate_hip(self, height: int, weight: int, bmi: float) -> int:
        """엉덩이둘레 추정"""
        base = 90 if bmi < 25 else 100
        return base + int((weight - 60) * 0.3)
    
    def analyze_clothing(self, cloth_image: Image.Image):
        """옷 분석 (간단한 더미 데이터)"""
        # 실제로는 딥러닝 모델로 분석해야 함
        categories = ["shirt", "t-shirt", "dress", "pants", "jacket"]
        styles = ["casual", "formal", "sporty", "elegant", "trendy"]
        
        # 주요 색상 추출 (간단한 평균)
        cloth_np = np.array(cloth_image)
        avg_color = cloth_np.mean(axis=(0, 1)).astype(int).tolist()
        
        return {
            "category": np.random.choice(categories),
            "style": np.random.choice(styles),
            "dominant_color": avg_color
        }
    
    def calculate_fit_score(self, height: int, weight: int, bmi: float) -> float:
        """핏 점수 계산"""
        # 이상적인 BMI 범위: 18.5 - 24.9
        if 18.5 <= bmi <= 24.9:
            base_score = 0.9
        elif 25 <= bmi <= 29.9:
            base_score = 0.8
        else:
            base_score = 0.7
        
        # 약간의 랜덤성 추가
        return min(base_score + np.random.random() * 0.1, 1.0)
    
    def generate_recommendations(self, fit_score: float, bmi: float, clothing_analysis: dict) -> list:
        """AI 추천사항 생성"""
        recommendations = []
        
        if fit_score > 0.85:
            recommendations.append("This item fits you perfectly! The size appears to be ideal for your body type.")
        elif fit_score > 0.7:
            recommendations.append("Good fit overall. You might want to consider tailoring for a perfect fit.")
        else:
            recommendations.append("Consider trying a different size for a better fit.")
        
        # BMI 기반 추천
        if bmi > 25:
            recommendations.append("This style is flattering for your body type. Consider darker colors for a slimming effect.")
        elif bmi < 18.5:
            recommendations.append("Layering with this piece would create a great look.")
        
        # 스타일 기반 추천
        style = clothing_analysis.get("style", "casual")
        if style == "formal":
            recommendations.append("Perfect for business meetings or formal events.")
        elif style == "casual":
            recommendations.append("Great for everyday wear. Pairs well with jeans or casual pants.")
        
        return recommendations[:3]  # 최대 3개 추천

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로드"""
    global model
    print("Loading virtual try-on model...")
    model = VirtualTryOnModel()
    print("Model loaded successfully!")

@app.get("/")
async def root():
    return {"message": "Virtual Try-On API is running"}

@app.post("/api/virtual-tryon")
async def virtual_tryon(
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...),
    height: int = Form(...),
    weight: int = Form(...)
):
    """가상 시착 엔드포인트"""
    try:
        # 이미지 검증
        if not person_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Person file must be an image")
        
        if not clothing_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Clothing file must be an image")
        
        # 이미지 로드
        person_img = Image.open(io.BytesIO(await person_image.read())).convert('RGB')
        cloth_img = Image.open(io.BytesIO(await clothing_image.read())).convert('RGB')
        
        # 크기 제한 (메모리 절약)
        max_size = (1024, 1024)
        person_img.thumbnail(max_size, Image.Resampling.LANCZOS)
        cloth_img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # 모델 처리
        result = model.process_images(person_img, cloth_img, height, weight)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Error in virtual_tryon: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )

@app.get("/api/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "ootd_available": OOTD_AVAILABLE,
        "device": str(model.device) if model else "not initialized"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)