# api/routes.py
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
import cv2
import numpy as np
import base64
import io
from PIL import Image

from api.schemas import VirtualTryOnResponse, ErrorResponse
from models.base_model import VirtualTryOnModel
from core.config import settings

router = APIRouter(prefix="/api/v1")
model = VirtualTryOnModel()

@router.post("/virtual-tryon", response_model=VirtualTryOnResponse)
async def virtual_tryon(
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...),
    height: float = Form(...),
    weight: float = Form(...)
):
    """가상 피팅 엔드포인트"""
    try:
        # 1. 파일 검증
        for file in [person_image, clothing_image]:
            if file.size > settings.max_upload_size:
                raise HTTPException(400, "File size exceeds 10MB limit")
            
            ext = Path(file.filename).suffix.lower()
            if ext not in settings.allowed_extensions:
                raise HTTPException(400, f"Invalid file type: {ext}")
        
        # 2. 이미지 읽기
        person_bytes = await person_image.read()
        clothing_bytes = await clothing_image.read()
        
        # PIL로 읽고 OpenCV 형식으로 변환
        person_pil = Image.open(io.BytesIO(person_bytes))
        clothing_pil = Image.open(io.BytesIO(clothing_bytes))
        
        person_np = cv2.cvtColor(np.array(person_pil), cv2.COLOR_RGB2BGR)
        clothing_np = cv2.cvtColor(np.array(clothing_pil), cv2.COLOR_RGB2BGR)
        
        # 3. 가상 피팅 수행
        result = model.virtual_fitting(person_np, clothing_np, height, weight)
        
        # 4. 결과 이미지 인코딩
        _, buffer = cv2.imencode('.jpg', result['fitted_image'])
        fitted_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 5. 응답 생성
        return VirtualTryOnResponse(
            success=True,
            fitted_image=fitted_image_base64,
            processing_time=result['processing_time'],
            confidence=result['confidence'],
            measurements=result['measurements'],
            clothing_analysis=result['clothing_analysis'],
            fit_score=result['fit_score'],
            recommendations=result['recommendations']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in virtual_tryon: {str(e)}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

@router.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(model.device) if model else "cpu"
    }