# backend/app/main.py
"""
MyCloset AI Backend - 완전한 가상 피팅 API 시스템
8단계 AI 파이프라인과 mycloset-uiux.tsx 호환
"""
import os
import sys
import asyncio
import logging
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

# FastAPI 관련 임포트
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

# 이미지 처리
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO

# 기존 app 구조 임포트
try:
    # 8단계 AI 파이프라인 임포트
    from app.ai_pipeline.pipeline_manager import PipelineManager
    from app.ai_pipeline.utils.model_loader import ModelLoader
    from app.ai_pipeline.utils.memory_manager import MemoryManager
    from app.ai_pipeline.utils.data_converter import DataConverter
    
    # 코어 모듈들
    from app.core.config import get_settings
    from app.core.logging_config import setup_logging
    from app.core.gpu_config import get_device_config
    
    # 모델 스키마
    from app.models.schemas import (
        VirtualTryOnRequest, 
        VirtualTryOnResponse,
        HealthResponse,
        StatusResponse,
        ProcessingResponse
    )
    
    # API 라우터들
    from app.api.virtual_tryon import router as virtual_tryon_router
    from app.api.health import router as health_router
    
except ImportError as e:
    logging.warning(f"일부 모듈 임포트 실패: {e}")
    # 폴백 설정
    class FallbackConfig:
        APP_NAME = "MyCloset AI"
        DEBUG = True
        CORS_ORIGINS = ["*"]
        UPLOAD_MAX_SIZE = 50 * 1024 * 1024
    
    get_settings = lambda: FallbackConfig()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 설정 로드
settings = get_settings()

# FastAPI 앱 생성
app = FastAPI(
    title="MyCloset AI Backend",
    description="""
    🎯 AI 기반 가상 피팅 플랫폼 백엔드 API
    
    ## 주요 기능
    - 🤖 8단계 AI 파이프라인 가상 피팅
    - 📐 실시간 신체 측정 및 분석
    - 👔 의류 스타일 분석 및 매칭
    - 🎨 포즈 추정 및 피팅 최적화
    - 📊 품질 평가 및 개선 제안
    
    ## 지원 기능
    - MediaPipe 포즈 추정
    - TPS 기하학적 변환
    - 실시간 이미지 처리
    - M3 Max MPS 최적화
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, 'CORS_ORIGINS', [
        "http://localhost:3000",
        "http://localhost:5173", 
        "http://localhost:8080",
        "https://mycloset-ai.vercel.app"
    ]),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 정적 파일 마운트
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"정적 파일 마운트 실패: {e}")
    os.makedirs("static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")

# 전역 변수들
pipeline_manager: Optional[PipelineManager] = None
model_loader: Optional[ModelLoader] = None
memory_manager: Optional[MemoryManager] = None
data_converter: Optional[DataConverter] = None

# 세션 관리
active_sessions: Dict[str, Dict[str, Any]] = {}
processing_queue: List[Dict[str, Any]] = []

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 초기화"""
    global pipeline_manager, model_loader, memory_manager, data_converter
    
    logger.info("🚀 MyCloset AI Backend 시작...")
    
    try:
        # 디렉토리 생성
        directories = [
            "static/uploads", "static/results", "static/temp",
            "logs", "models/checkpoints", "ai_models/cache"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            # .gitkeep 파일 생성
            gitkeep_path = os.path.join(directory, ".gitkeep")
            if not os.path.exists(gitkeep_path):
                with open(gitkeep_path, "w") as f:
                    f.write("")
        
        # GPU/디바이스 설정
        try:
            device_config = get_device_config()
            logger.info(f"🔧 디바이스 설정: {device_config}")
        except:
            logger.warning("디바이스 설정 로드 실패, 기본값 사용")
        
        # 유틸리티 초기화
        try:
            model_loader = ModelLoader()
            memory_manager = MemoryManager()
            data_converter = DataConverter()
            logger.info("✅ 유틸리티 컴포넌트 초기화 완료")
        except Exception as e:
            logger.warning(f"유틸리티 초기화 실패: {e}")
        
        # 8단계 AI 파이프라인 초기화 (백그라운드)
        try:
            pipeline_manager = PipelineManager()
            
            # 비동기로 초기화 (차단하지 않음)
            asyncio.create_task(initialize_pipeline_background())
            
        except Exception as e:
            logger.warning(f"파이프라인 매니저 생성 실패: {e}")
        
        logger.info("✅ MyCloset AI Backend 시작 완료")
        
    except Exception as e:
        logger.error(f"❌ 시작 중 오류: {e}")
        logger.error(traceback.format_exc())

async def initialize_pipeline_background():
    """백그라운드에서 파이프라인 초기화"""
    global pipeline_manager
    
    try:
        logger.info("🔄 8단계 AI 파이프라인 초기화 시작...")
        
        if pipeline_manager:
            success = await pipeline_manager.initialize()
            if success:
                logger.info("✅ 8단계 AI 파이프라인 초기화 완료")
                
                # 워밍업 실행
                warmup_success = await pipeline_manager.warmup()
                if warmup_success:
                    logger.info("🔥 파이프라인 워밍업 완료")
                else:
                    logger.warning("⚠️ 파이프라인 워밍업 실패")
            else:
                logger.error("❌ 파이프라인 초기화 실패")
        
    except Exception as e:
        logger.error(f"❌ 백그라운드 파이프라인 초기화 실패: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 정리"""
    logger.info("🛑 MyCloset AI Backend 종료 중...")
    
    try:
        # 활성 세션 정리
        for session_id, session_data in active_sessions.items():
            logger.info(f"세션 {session_id} 정리 중...")
        
        active_sessions.clear()
        
        # 파이프라인 매니저 정리
        if pipeline_manager:
            await pipeline_manager.cleanup()
        
        # 메모리 정리
        if memory_manager:
            await memory_manager.cleanup()
        
        logger.info("✅ 정리 완료")
        
    except Exception as e:
        logger.error(f"❌ 종료 중 오류: {e}")

# ========================================
# 헬스체크 및 상태 엔드포인트
# ========================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """시스템 헬스체크"""
    
    # 파이프라인 상태 확인
    pipeline_status = False
    pipeline_info = {}
    
    if pipeline_manager:
        try:
            pipeline_info = await pipeline_manager.get_pipeline_status()
            pipeline_status = pipeline_info.get('initialized', False)
        except:
            pass
    
    # 메모리 상태 확인
    memory_status = "unknown"
    if memory_manager:
        try:
            memory_info = await memory_manager.get_memory_status()
            memory_status = "healthy" if memory_info.get('available_percent', 0) > 20 else "warning"
        except:
            pass
    
    return {
        "status": "healthy" if pipeline_status else "starting",
        "timestamp": datetime.now().isoformat(),
        "pipeline_ready": pipeline_status,
        "memory_status": memory_status,
        "active_sessions": len(active_sessions),
        "version": "2.0.0",
        "device": pipeline_info.get('device', 'unknown')
    }

@app.get("/api/status", response_model=StatusResponse, tags=["System"])
async def get_system_status():
    """시스템 상세 상태 조회"""
    
    status_data = {
        "backend_status": "running",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions),
        "processing_queue_length": len(processing_queue)
    }
    
    if pipeline_manager:
        try:
            pipeline_status = await pipeline_manager.get_pipeline_status()
            status_data.update({
                "pipeline_initialized": pipeline_status.get('initialized', False),
                "device": pipeline_status.get('device', 'cpu'),
                "models_loaded": pipeline_status.get('steps_loaded', 0),
                "total_steps": pipeline_status.get('total_steps', 8),
                "memory_usage": pipeline_status.get('memory_status', {}),
                "performance_stats": pipeline_status.get('stats', {})
            })
        except Exception as e:
            logger.warning(f"파이프라인 상태 조회 실패: {e}")
    
    return status_data

# ========================================
# 가상 피팅 메인 엔드포인트
# ========================================

@app.post("/api/virtual-tryon", response_model=VirtualTryOnResponse, tags=["Virtual Fitting"])
async def virtual_tryon_complete(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(..., description="사용자 사진 (최대 50MB)"),
    clothing_image: UploadFile = File(..., description="의류 사진 (최대 50MB)"),
    height: float = Form(170.0, description="키 (cm, 140-220)"),
    weight: float = Form(65.0, description="몸무게 (kg, 30-150)"),
    clothing_type: str = Form("shirt", description="의류 타입: shirt, pants, dress, jacket, skirt"),
    style_preference: str = Form("casual", description="스타일 선호도: casual, formal, sporty"),
    quality_level: str = Form("high", description="품질 레벨: fast, medium, high, ultra")
):
    """
    🔥 완전한 8단계 AI 가상 피팅
    
    mycloset-uiux.tsx와 완전 호환되는 메인 API 엔드포인트
    """
    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # 입력 검증
        validation_result = await validate_virtual_tryon_input(
            person_image, clothing_image, height, weight, clothing_type, quality_level
        )
        
        if not validation_result['valid']:
            raise HTTPException(status_code=400, detail=validation_result['error'])
        
        logger.info(f"🎯 8단계 가상 피팅 시작 - Session: {session_id}")
        logger.info(f"⚙️ 설정: {clothing_type} ({style_preference}), 품질: {quality_level}")
        
        # 세션 등록
        active_sessions[session_id] = {
            'status': 'processing',
            'start_time': start_time,
            'clothing_type': clothing_type,
            'quality_level': quality_level
        }
        
        # 파이프라인 사용 가능 여부 확인
        if not pipeline_manager or not pipeline_manager.is_initialized:
            # 폴백: 간단한 데모 모드
            logger.warning("파이프라인 미준비, 데모 모드로 처리")
            return await process_demo_mode(
                person_image, clothing_image, height, weight, 
                clothing_type, session_id, start_time
            )
        
        # 이미지 전처리
        person_pil = await load_and_validate_image(person_image, "person")
        clothing_pil = await load_and_validate_image(clothing_image, "clothing")
        
        # 8단계 AI 파이프라인 실행
        pipeline_result = await pipeline_manager.process_complete_virtual_fitting(
            person_image=person_pil,
            clothing_image=clothing_pil,
            body_measurements={
                'height': height,
                'weight': weight,
                'bmi': weight / ((height/100) ** 2)
            },
            clothing_type=clothing_type,
            fabric_type="cotton",  # 기본값
            style_preferences={
                'fit': 'regular',
                'style': style_preference,
                'color_preference': 'original'
            },
            quality_target=0.8 if quality_level == "high" else 0.7,
            progress_callback=None,  # 웹소켓으로 추후 구현
            save_intermediate=False,
            enable_auto_retry=True
        )
        
        # 결과 처리
        if pipeline_result.get('success', False):
            # 성공 결과 처리
            result_image_base64 = await process_result_image(
                pipeline_result.get('result_image'), session_id
            )
            
            processing_time = time.time() - start_time
            
            # 최종 응답 구성 (mycloset-uiux.tsx 호환)
            response = {
                "success": True,
                "session_id": session_id,
                "fitted_image": result_image_base64,
                "processing_time": processing_time,
                "confidence": pipeline_result.get('final_quality_score', 0.85),
                
                # 신체 측정 정보
                "measurements": {
                    "chest": estimate_chest_measurement(height, weight),
                    "waist": estimate_waist_measurement(height, weight),
                    "hip": estimate_hip_measurement(height, weight),
                    "bmi": weight / ((height/100) ** 2)
                },
                
                # 의류 분석
                "clothing_analysis": {
                    "category": clothing_type,
                    "style": style_preference,
                    "dominant_color": [128, 128, 128]  # 기본값
                },
                
                # 피팅 스코어 및 추천
                "fit_score": pipeline_result.get('fit_score', 0.8),
                "recommendations": pipeline_result.get('recommendations', [
                    f"✅ {clothing_type} 스타일이 잘 어울립니다!",
                    "📐 완벽한 핏을 위해 정확한 치수를 확인해보세요",
                    "🎨 다른 색상도 시도해보세요"
                ]),
                
                # 추가 분석 정보
                "quality_analysis": {
                    "overall_score": pipeline_result.get('final_quality_score', 0.85),
                    "fit_quality": pipeline_result.get('fit_analysis', {}).get('overall_fit_score', 0.8),
                    "processing_quality": min(1.0, 30.0 / processing_time) if processing_time > 0 else 1.0
                },
                
                # 처리 정보
                "processing_info": {
                    "steps_completed": len(pipeline_result.get('step_results_summary', {})),
                    "quality_level": quality_level,
                    "device_used": pipeline_result.get('processing_info', {}).get('device_used', 'cpu'),
                    "optimization": "M3_Max" if "mps" in str(pipeline_result.get('processing_info', {})) else "Standard"
                }
            }
            
            # 세션 업데이트
            active_sessions[session_id].update({
                'status': 'completed',
                'result': response,
                'processing_time': processing_time
            })
            
            logger.info(f"✅ 가상 피팅 완료 - Session: {session_id}, 시간: {processing_time:.2f}초")
            return response
            
        else:
            # 파이프라인 실패 시 폴백
            logger.warning(f"파이프라인 처리 실패: {pipeline_result.get('error', 'Unknown error')}")
            return await process_demo_mode(
                person_image, clothing_image, height, weight,
                clothing_type, session_id, start_time, 
                error_message=pipeline_result.get('error')
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 가상 피팅 처리 실패 - Session: {session_id}: {e}")
        logger.error(traceback.format_exc())
        
        # 세션 상태 업데이트
        if session_id in active_sessions:
            active_sessions[session_id].update({
                'status': 'failed',
                'error': str(e)
            })
        
        # 폴백 처리
        try:
            return await process_demo_mode(
                person_image, clothing_image, height, weight,
                clothing_type, session_id, start_time,
                error_message=str(e)
            )
        except:
            raise HTTPException(
                status_code=500, 
                detail=f"가상 피팅 처리 중 오류가 발생했습니다: {str(e)}"
            )

# ========================================
# 보조 함수들
# ========================================

async def validate_virtual_tryon_input(
    person_image: UploadFile,
    clothing_image: UploadFile, 
    height: float,
    weight: float,
    clothing_type: str,
    quality_level: str
) -> Dict[str, Any]:
    """입력 데이터 검증"""
    
    # 이미지 파일 검증
    if not person_image.content_type.startswith('image/'):
        return {'valid': False, 'error': '사용자 이미지 파일이 올바르지 않습니다'}
    
    if not clothing_image.content_type.startswith('image/'):
        return {'valid': False, 'error': '의류 이미지 파일이 올바르지 않습니다'}
    
    # 파일 크기 검증 (50MB 제한)
    max_size = 50 * 1024 * 1024
    if person_image.size > max_size:
        return {'valid': False, 'error': '사용자 이미지 파일이 너무 큽니다 (최대 50MB)'}
    
    if clothing_image.size > max_size:
        return {'valid': False, 'error': '의류 이미지 파일이 너무 큽니다 (최대 50MB)'}
    
    # 신체 치수 검증
    if not (140 <= height <= 220):
        return {'valid': False, 'error': '키는 140cm ~ 220cm 범위여야 합니다'}
    
    if not (30 <= weight <= 150):
        return {'valid': False, 'error': '몸무게는 30kg ~ 150kg 범위여야 합니다'}
    
    # 의류 타입 검증
    valid_clothing_types = ['shirt', 'pants', 'dress', 'jacket', 'skirt', 't-shirt', 'blouse']
    if clothing_type not in valid_clothing_types:
        return {'valid': False, 'error': f'지원하지 않는 의류 타입입니다: {clothing_type}'}
    
    # 품질 레벨 검증
    valid_quality_levels = ['fast', 'medium', 'high', 'ultra']
    if quality_level not in valid_quality_levels:
        return {'valid': False, 'error': f'지원하지 않는 품질 레벨입니다: {quality_level}'}
    
    return {'valid': True}

async def load_and_validate_image(upload_file: UploadFile, image_type: str) -> Image.Image:
    """이미지 로드 및 검증"""
    
    try:
        # 이미지 데이터 읽기
        image_data = await upload_file.read()
        
        # PIL 이미지로 변환
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # 크기 검증
        if image.width < 256 or image.height < 256:
            raise ValueError(f"{image_type} 이미지가 너무 작습니다 (최소 256x256)")
        
        # 크기 조정 (최대 1024x1024)
        if image.width > 1024 or image.height > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        return image
        
    except Exception as e:
        raise ValueError(f"{image_type} 이미지 처리 실패: {str(e)}")

async def process_result_image(result_image: Any, session_id: str) -> str:
    """결과 이미지 처리 및 base64 인코딩"""
    
    try:
        # PIL 이미지로 변환
        if hasattr(result_image, 'save'):
            # 이미 PIL 이미지인 경우
            pil_image = result_image
        elif isinstance(result_image, np.ndarray):
            # numpy 배열인 경우
            pil_image = Image.fromarray(result_image)
        else:
            # 기타 형식
            pil_image = Image.fromarray(np.array(result_image))
        
        # 파일로 저장
        save_path = f"static/results/{session_id}_result.jpg"
        pil_image.save(save_path, "JPEG", quality=90)
        
        # base64 인코딩
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=90)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return image_base64
        
    except Exception as e:
        logger.error(f"결과 이미지 처리 실패: {e}")
        # 기본 이미지 반환
        default_image = Image.new('RGB', (512, 512), color='lightgray')
        buffer = BytesIO()
        default_image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

async def process_demo_mode(
    person_image: UploadFile,
    clothing_image: UploadFile,
    height: float,
    weight: float,
    clothing_type: str,
    session_id: str,
    start_time: float,
    error_message: Optional[str] = None
) -> VirtualTryOnResponse:
    """데모 모드 처리 (AI 파이프라인 미준비 시)"""
    
    try:
        # 이미지 로드
        person_pil = await load_and_validate_image(person_image, "person")
        clothing_pil = await load_and_validate_image(clothing_image, "clothing")
        
        # 간단한 합성 이미지 생성
        demo_result = create_demo_composite(person_pil, clothing_pil)
        
        # base64 인코딩
        result_base64 = await process_result_image(demo_result, session_id)
        
        processing_time = time.time() - start_time
        
        # 데모 응답 생성
        response = {
            "success": True,
            "session_id": session_id,
            "fitted_image": result_base64,
            "processing_time": processing_time,
            "confidence": 0.75,  # 데모 모드 신뢰도
            
            "measurements": {
                "chest": estimate_chest_measurement(height, weight),
                "waist": estimate_waist_measurement(height, weight),
                "hip": estimate_hip_measurement(height, weight),
                "bmi": weight / ((height/100) ** 2)
            },
            
            "clothing_analysis": {
                "category": clothing_type,
                "style": "casual",
                "dominant_color": [100, 100, 150]
            },
            
            "fit_score": 0.75,
            "recommendations": [
                "🚧 데모 모드로 처리되었습니다",
                "⚡ AI 모델 로딩 완료 후 더 정확한 결과를 얻을 수 있습니다",
                f"👔 {clothing_type} 스타일 시뮬레이션"
            ],
            
            "quality_analysis": {
                "overall_score": 0.75,
                "fit_quality": 0.7,
                "processing_quality": 1.0
            },
            
            "processing_info": {
                "steps_completed": 0,
                "quality_level": "demo",
                "device_used": "cpu",
                "optimization": "Demo Mode",
                "demo_mode": True,
                "error_message": error_message
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"데모 모드 처리 실패: {e}")
        raise HTTPException(status_code=500, detail="데모 모드 처리 실패")

def create_demo_composite(person_image: Image.Image, clothing_image: Image.Image) -> Image.Image:
    """데모용 간단한 합성 이미지 생성"""
    
    try:
        # 크기 조정
        person_resized = person_image.resize((512, 512), Image.Resampling.LANCZOS)
        clothing_resized = clothing_image.resize((256, 256), Image.Resampling.LANCZOS)
        
        # 합성 이미지 생성
        result = person_resized.copy()
        
        # 의류 이미지를 우상단에 오버레이
        result.paste(clothing_resized, (256, 0), clothing_resized)
        
        # 데모 텍스트 추가
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(result)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 470), "🚧 DEMO MODE", fill=(255, 100, 100), font=font)
        draw.text((10, 490), "AI Loading...", fill=(100, 100, 255), font=font)
        
        return result
        
    except Exception as e:
        logger.error(f"데모 합성 실패: {e}")
        # 기본 이미지 반환
        return Image.new('RGB', (512, 512), color='lightblue')

# 신체 치수 추정 함수들
def estimate_chest_measurement(height: float, weight: float) -> float:
    """가슴둘레 추정"""
    bmi = weight / ((height/100) ** 2)
    base_chest = height * 0.52
    adjustment = (bmi - 22) * 2
    return round(base_chest + adjustment, 1)

def estimate_waist_measurement(height: float, weight: float) -> float:
    """허리둘레 추정"""
    bmi = weight / ((height/100) ** 2)
    base_waist = height * 0.42
    adjustment = (bmi - 22) * 2.5
    return round(base_waist + adjustment, 1)

def estimate_hip_measurement(height: float, weight: float) -> float:
    """엉덩이둘레 추정"""
    bmi = weight / ((height/100) ** 2)
    base_hip = height * 0.55
    adjustment = (bmi - 22) * 1.8
    return round(base_hip + adjustment, 1)

# ========================================
# 추가 유틸리티 엔드포인트들
# ========================================

@app.get("/api/session/{session_id}", tags=["Session"])
async def get_session_status(session_id: str):
    """세션 상태 조회"""
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    session_data = active_sessions[session_id]
    
    return {
        "session_id": session_id,
        "status": session_data.get('status', 'unknown'),
        "start_time": session_data.get('start_time'),
        "processing_time": time.time() - session_data.get('start_time', time.time()),
        "clothing_type": session_data.get('clothing_type'),
        "quality_level": session_data.get('quality_level'),
        "has_result": 'result' in session_data
    }

@app.delete("/api/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """세션 삭제"""
    
    if session_id in active_sessions:
        del active_sessions[session_id]
        
        # 관련 파일들 정리
        result_file = f"static/results/{session_id}_result.jpg"
        if os.path.exists(result_file):
            os.remove(result_file)
        
        return {"message": "세션이 삭제되었습니다"}
    else:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")

@app.get("/api/sessions", tags=["Session"])
async def list_active_sessions():
    """활성 세션 목록 조회"""
    
    return {
        "active_sessions": len(active_sessions),
        "sessions": [
            {
                "session_id": sid,
                "status": data.get('status'),
                "clothing_type": data.get('clothing_type'),
                "start_time": data.get('start_time')
            }
            for sid, data in active_sessions.items()
        ]
    }

@app.get("/api/pipeline-status", tags=["System"])
async def get_pipeline_status():
    """파이프라인 상세 상태"""
    
    if not pipeline_manager:
        return {
            "initialized": False,
            "error": "파이프라인 매니저가 초기화되지 않았습니다"
        }
    
    try:
        status = await pipeline_manager.get_pipeline_status()
        return status
    except Exception as e:
        return {
            "initialized": False,
            "error": str(e)
        }

# 개발용 엔드포인트들
if getattr(settings, 'DEBUG', False):
    
    @app.get("/api/debug/reset-pipeline", tags=["Debug"])
    async def reset_pipeline():
        """파이프라인 리셋 (개발용)"""
        global pipeline_manager
        
        try:
            if pipeline_manager:
                await pipeline_manager.cleanup()
            
            pipeline_manager = PipelineManager()
            success = await pipeline_manager.initialize()
            
            return {
                "message": "파이프라인이 리셋되었습니다",
                "success": success
            }
        except Exception as e:
            return {
                "message": "파이프라인 리셋 실패",
                "error": str(e)
            }

# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    import uvicorn
    
    # 로그 디렉토리 생성
    os.makedirs("logs", exist_ok=True)
    
    # 서버 실행
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=getattr(settings, 'DEBUG', False),
        log_level="info",
        access_log=True
    )