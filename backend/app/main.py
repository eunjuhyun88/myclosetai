# backend/app/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from pathlib import Path
import logging
from datetime import datetime
import uuid
import json

# 로컬 임포트
from app.core.gpu_config import gpu_config, DEVICE, MODEL_CONFIG
from app.services.image_processor import ImageProcessor
from app.services.virtual_fitter import VirtualFitter
from app.models.schemas import TryOnRequest, TryOnResponse

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="MyCloset AI Backend",
    description="AI 가상 피팅 플랫폼",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 마운트
app.mount("/static", StaticFiles(directory="static"), name="static")

# 전역 서비스 인스턴스
image_processor = ImageProcessor()
virtual_fitter = VirtualFitter()

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    logger.info("🚀 MyCloset AI Backend 시작")
    logger.info(f"🔧 GPU 디바이스: {DEVICE}")
    
    # 디렉토리 생성
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # AI 모델 초기화 (백그라운드에서)
    try:
        await virtual_fitter.initialize_models()
        logger.info("✅ AI 모델 초기화 완료")
    except Exception as e:
        logger.warning(f"⚠️ AI 모델 초기화 실패 (데모 모드): {e}")

@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "timestamp": datetime.now().isoformat(),
        "models_loaded": virtual_fitter.models_loaded
    }

@app.get("/api/status")
async def get_status():
    """시스템 상태 조회"""
    return {
        "backend_status": "running",
        "gpu_available": DEVICE != "cpu",
        "device": DEVICE,
        "models_ready": virtual_fitter.models_loaded,
        "upload_limit_mb": 50
    }

@app.post("/api/virtual-tryon")
async def virtual_tryon(
    person_image: UploadFile = File(..., description="사용자 사진"),
    clothing_image: UploadFile = File(..., description="의류 사진"),
    height: float = Form(170.0, description="키 (cm)"),
    weight: float = Form(65.0, description="몸무게 (kg)"),
    model_type: str = Form("demo", description="모델 타입")
):
    """가상 피팅 실행"""
    session_id = str(uuid.uuid4())
    
    try:
        # 1. 입력 검증
        if not person_image.content_type.startswith('image/'):
            raise HTTPException(400, "사용자 이미지 파일이 올바르지 않습니다")
        if not clothing_image.content_type.startswith('image/'):
            raise HTTPException(400, "의류 이미지 파일이 올바르지 않습니다")
        
        logger.info(f"🎯 가상 피팅 시작 - Session: {session_id}")
        
        # 2. 이미지 저장
        person_path = f"static/uploads/{session_id}_person.jpg"
        clothing_path = f"static/uploads/{session_id}_clothing.jpg"
        
        # 파일 저장
        with open(person_path, "wb") as f:
            content = await person_image.read()
            f.write(content)
        
        with open(clothing_path, "wb") as f:
            content = await clothing_image.read()
            f.write(content)
        
        # 3. 이미지 전처리
        processed_person = await image_processor.process_person_image(person_path)
        processed_clothing = await image_processor.process_clothing_image(clothing_path)
        
        # 4. 가상 피팅 실행
        start_time = datetime.now()
        
        if model_type == "demo":
            # 데모 모드: 간단한 이미지 합성
            result = await virtual_fitter.demo_fitting(
                processed_person, 
                processed_clothing,
                height, 
                weight
            )
        else:
            # 실제 AI 모델 사용
            result = await virtual_fitter.ai_fitting(
                processed_person,
                processed_clothing,
                height,
                weight,
                model_type
            )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 5. 결과 저장
        result_path = f"static/results/{session_id}_result.jpg"
        result.save(result_path)
        
        # 6. 응답 생성
        response = {
            "success": True,
            "session_id": session_id,
            "result_image_url": f"/static/results/{session_id}_result.jpg",
            "processing_time": round(processing_time, 2),
            "confidence_score": 0.85,  # 임시값
            "measurements": {
                "estimated_chest": round(height * 0.5, 1),
                "estimated_waist": round(height * 0.4, 1),
                "estimated_hip": round(height * 0.55, 1),
                "bmi": round(weight / ((height/100) ** 2), 1)
            },
            "clothing_analysis": {
                "category": "상의",
                "style": "캐주얼",
                "fit_score": 88
            },
            "recommendations": [
                "이 의류가 잘 어울립니다!",
                f"당신의 체형에 {88}% 적합합니다"
            ]
        }
        
        logger.info(f"✅ 가상 피팅 완료 - {processing_time:.2f}초")
        return response
        
    except Exception as e:
        logger.error(f"❌ 가상 피팅 오류: {str(e)}")
        raise HTTPException(500, f"처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/preprocess")
async def preprocess_images(
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...)
):
    """이미지 전처리만 수행"""
    try:
        # 임시 파일 저장
        temp_id = str(uuid.uuid4())[:8]
        person_path = f"static/uploads/temp_{temp_id}_person.jpg"
        clothing_path = f"static/uploads/temp_{temp_id}_clothing.jpg"
        
        with open(person_path, "wb") as f:
            f.write(await person_image.read())
        with open(clothing_path, "wb") as f:
            f.write(await clothing_image.read())
        
        # 전처리 실행
        person_info = await image_processor.analyze_person(person_path)
        clothing_info = await image_processor.analyze_clothing(clothing_path)
        
        return {
            "success": True,
            "person_analysis": person_info,
            "clothing_analysis": clothing_info
        }
        
    except Exception as e:
        raise HTTPException(500, f"전처리 오류: {str(e)}")

@app.get("/api/models/status")
async def get_models_status():
    """AI 모델 상태 조회"""
    return {
        "models_loaded": virtual_fitter.models_loaded,
        "available_models": ["demo", "ootd", "viton"],
        "device": DEVICE,
        "memory_usage": "측정 중..."
    }

# 개발 서버 실행
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )