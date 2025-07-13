"""
MyCloset AI Virtual Try-On API
기존 ai_pipeline 구조를 활용한 완전한 가상 피팅 API 엔드포인트
"""
import os
import time
import asyncio
import uuid
import base64
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, Field
import aiofiles
from PIL import Image
import numpy as np
import logging

# 기존 ai_pipeline 구조 import
try:
    from app.ai_pipeline.pipeline_manager import PipelineManager, get_pipeline_manager
    from app.ai_pipeline.utils.memory_manager import MemoryManager
    from app.ai_pipeline.utils.image_utils import save_temp_image, load_image
    from app.ai_pipeline.utils.data_converter import DataConverter
    from app.core.config import get_settings
    from app.core.logging_config import setup_logging
    AI_PIPELINE_AVAILABLE = True
except ImportError as e:
    # 폴백: 기본 구현 사용
    AI_PIPELINE_AVAILABLE = False
    logging.warning(f"AI 파이프라인 모듈 없음: {e}")

logger = logging.getLogger(__name__)
settings = get_settings() if 'get_settings' in globals() else None

# API 라우터 초기화
router = APIRouter(prefix="/virtual-tryon", tags=["Virtual Try-On"])

# 전역 파이프라인 매니저
pipeline_manager: Optional[PipelineManager] = None

# WebSocket 연결 관리
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket 연결: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket 연결 해제: {session_id}")
    
    async def send_progress(self, session_id: str, stage: str, percentage: int, message: str = ""):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_json({
                        "type": "progress",
                        "stage": stage,
                        "percentage": percentage,
                        "message": message,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.warning(f"WebSocket 메시지 전송 실패 {session_id}: {e}")
                    self.disconnect(session_id)

manager = ConnectionManager()

# 요청/응답 모델
class VirtualTryOnRequest(BaseModel):
    height: float = Field(..., description="키 (cm)", example=170.0)
    weight: float = Field(..., description="몸무게 (kg)", example=65.0)
    chest: Optional[float] = Field(None, description="가슴둘레 (cm)", example=95.0)
    waist: Optional[float] = Field(None, description="허리둘레 (cm)", example=80.0)
    hip: Optional[float] = Field(None, description="엉덩이둘레 (cm)", example=90.0)
    clothing_type: str = Field("shirt", description="의류 타입", example="shirt")
    fabric_type: str = Field("cotton", description="천 재질", example="cotton")
    style_preference: str = Field("regular", description="핏 선호도", example="slim")
    quality_level: str = Field("high", description="품질 레벨", example="high")

class VirtualTryOnResponse(BaseModel):
    success: bool
    session_id: str
    fitted_image_url: Optional[str] = None
    fitted_image_base64: Optional[str] = None
    processing_time: float
    confidence: float = Field(..., description="전체 신뢰도")
    fit_score: float = Field(..., description="핏 점수")
    quality_score: float = Field(..., description="품질 점수")
    quality_grade: str = Field(..., description="품질 등급")
    recommendations: List[str] = Field(default_factory=list)
    measurements: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    
class ProcessingStatusResponse(BaseModel):
    session_id: str
    status: str  # "processing", "completed", "failed"
    current_stage: str
    progress_percentage: int
    estimated_remaining_time: Optional[float] = None
    error: Optional[str] = None

# 초기화 함수
async def initialize_pipeline():
    """파이프라인 매니저 초기화"""
    global pipeline_manager
    
    if not AI_PIPELINE_AVAILABLE:
        logger.warning("⚠️ AI 파이프라인 사용 불가 - 기본 모드로 실행")
        return False
    
    try:
        if pipeline_manager is None:
            logger.info("🚀 파이프라인 매니저 초기화 시작...")
            pipeline_manager = get_pipeline_manager()
            
            # 파이프라인 초기화
            success = await pipeline_manager.initialize()
            if success:
                logger.info("✅ 파이프라인 매니저 초기화 완료")
                return True
            else:
                logger.error("❌ 파이프라인 매니저 초기화 실패")
                return False
        return True
        
    except Exception as e:
        logger.error(f"❌ 파이프라인 초기화 오류: {e}")
        return False

# 유틸리티 함수들
async def save_uploaded_file(upload_file: UploadFile, session_id: str, file_type: str) -> str:
    """업로드된 파일 저장"""
    try:
        # 파일 확장자 검증
        if not upload_file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(400, "지원하지 않는 파일 형식입니다. PNG, JPG, JPEG만 지원합니다.")
        
        # 저장 경로 생성
        upload_dir = Path("static/uploads") / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일명 생성
        timestamp = int(time.time())
        file_extension = Path(upload_file.filename).suffix
        filename = f"{file_type}_{timestamp}{file_extension}"
        file_path = upload_dir / filename
        
        # 파일 저장
        async with aiofiles.open(file_path, 'wb') as f:
            content = await upload_file.read()
            await f.write(content)
        
        logger.info(f"📁 파일 저장 완료: {file_path}")
        return str(file_path)
        
    except Exception as e:
        logger.error(f"❌ 파일 저장 실패: {e}")
        raise HTTPException(500, f"파일 저장 중 오류가 발생했습니다: {str(e)}")

def validate_image_file(upload_file: UploadFile) -> bool:
    """이미지 파일 검증"""
    # 파일 크기 검증 (10MB 제한)
    if upload_file.size > 10 * 1024 * 1024:
        raise HTTPException(400, "파일 크기가 너무 큽니다. 10MB 이하로 업로드해주세요.")
    
    # 파일 형식 검증
    if not upload_file.content_type.startswith('image/'):
        raise HTTPException(400, "이미지 파일만 업로드 가능합니다.")
    
    return True

def image_to_base64(image_path: str) -> str:
    """이미지를 base64로 인코딩"""
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            encoded = base64.b64encode(img_data).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        logger.error(f"❌ Base64 인코딩 실패: {e}")
        return ""

# 메인 API 엔드포인트들

@router.post("/process", response_model=VirtualTryOnResponse)
async def virtual_tryon_process(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(..., description="사용자 사진"),
    clothing_image: UploadFile = File(..., description="의류 사진"),
    height: float = Form(..., description="키 (cm)"),
    weight: float = Form(..., description="몸무게 (kg)"),
    chest: Optional[float] = Form(None, description="가슴둘레 (cm)"),
    waist: Optional[float] = Form(None, description="허리둘레 (cm)"),
    hip: Optional[float] = Form(None, description="엉덩이둘레 (cm)"),
    clothing_type: str = Form("shirt", description="의류 타입"),
    fabric_type: str = Form("cotton", description="천 재질"),
    style_preference: str = Form("regular", description="핏 선호도"),
    quality_level: str = Form("high", description="품질 레벨")
):
    """
    🎯 메인 가상 피팅 API
    
    사용자와 의류 이미지를 업로드하여 가상 피팅을 수행합니다.
    고급 AI 모델을 사용하여 현실적인 착용 결과를 생성합니다.
    """
    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"🎯 가상 피팅 요청 시작 - 세션: {session_id}")
        
        # 파이프라인 초기화 확인
        if not await initialize_pipeline():
            logger.warning("⚠️ AI 파이프라인 없음 - 데모 모드로 실행")
            return await _demo_virtual_tryon(session_id, start_time)
        
        # 파일 검증
        validate_image_file(person_image)
        validate_image_file(clothing_image)
        
        # 파일 저장
        person_image_path = await save_uploaded_file(person_image, session_id, "person")
        clothing_image_path = await save_uploaded_file(clothing_image, session_id, "clothing")
        
        # 신체 치수 구성
        body_measurements = {
            "height": height,
            "weight": weight
        }
        if chest:
            body_measurements["chest"] = chest
        if waist:
            body_measurements["waist"] = waist
        if hip:
            body_measurements["hip"] = hip
        
        # 스타일 선호도 구성
        style_preferences = {
            "fit": style_preference,
            "color_preference": "original"
        }
        
        # 진행률 콜백 함수 정의
        async def progress_callback(stage: str, percentage: int):
            await manager.send_progress(session_id, stage, percentage)
        
        # AI 파이프라인 실행
        logger.info("🤖 AI 파이프라인 실행 중...")
        result = await pipeline_manager.process_complete_virtual_fitting(
            person_image=person_image_path,
            clothing_image=clothing_image_path,
            body_measurements=body_measurements,
            clothing_type=clothing_type,
            fabric_type=fabric_type,
            style_preferences=style_preferences,
            quality_target=0.8 if quality_level == "high" else 0.7,
            progress_callback=progress_callback,
            save_intermediate=False,
            enable_auto_retry=True
        )
        
        # 결과 처리
        if result['success']:
            # 결과 이미지 저장
            result_dir = Path("static/results") / session_id
            result_dir.mkdir(parents=True, exist_ok=True)
            
            result_image_path = result_dir / "fitted_result.jpg"
            if hasattr(result['result_image'], 'save'):
                result['result_image'].save(result_image_path)
            
            # 응답 구성
            processing_time = time.time() - start_time
            
            response = VirtualTryOnResponse(
                success=True,
                session_id=session_id,
                fitted_image_url=f"/static/results/{session_id}/fitted_result.jpg",
                fitted_image_base64=image_to_base64(str(result_image_path)),
                processing_time=processing_time,
                confidence=result.get('final_quality_score', 0.85),
                fit_score=result.get('fit_analysis', {}).get('overall_fit_score', 0.88),
                quality_score=result.get('final_quality_score', 0.85),
                quality_grade=result.get('quality_grade', 'Good'),
                recommendations=result.get('improvement_suggestions', {}).get('user_experience', [])[:3],
                measurements=body_measurements
            )
            
            # 백그라운드에서 임시 파일 정리
            background_tasks.add_task(_cleanup_session_files, session_id)
            
            logger.info(f"✅ 가상 피팅 완료 - 세션: {session_id}, 시간: {processing_time:.2f}초")
            return response
            
        else:
            # 처리 실패
            error_msg = result.get('error', '알 수 없는 오류가 발생했습니다.')
            logger.error(f"❌ 가상 피팅 실패 - 세션: {session_id}: {error_msg}")
            
            return VirtualTryOnResponse(
                success=False,
                session_id=session_id,
                processing_time=time.time() - start_time,
                confidence=0.0,
                fit_score=0.0,
                quality_score=0.0,
                quality_grade="Failed",
                error=error_msg,
                measurements=body_measurements
            )
            
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"가상 피팅 처리 중 오류가 발생했습니다: {str(e)}"
        logger.error(f"❌ 가상 피팅 오류 - 세션: {session_id}: {e}")
        
        return VirtualTryOnResponse(
            success=False,
            session_id=session_id,
            processing_time=processing_time,
            confidence=0.0,
            fit_score=0.0,
            quality_score=0.0,
            quality_grade="Error",
            error=error_msg,
            measurements={"height": height, "weight": weight}
        )

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    🔌 실시간 진행상황 WebSocket
    
    가상 피팅 처리 중 실시간으로 진행상황을 전송합니다.
    """
    await manager.connect(websocket, session_id)
    try:
        while True:
            # 클라이언트로부터 메시지 대기 (연결 유지용)
            data = await websocket.receive_text()
            
            # ping/pong 처리
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        logger.info(f"WebSocket 연결 종료: {session_id}")

@router.get("/status/{session_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(session_id: str):
    """
    📊 처리 상태 조회
    
    특정 세션의 가상 피팅 처리 상태를 조회합니다.
    """
    try:
        # 여기서는 파이프라인 매니저의 상태를 조회
        # 실제로는 Redis나 DB에서 상태를 관리해야 함
        
        # 기본 응답 (데모용)
        return ProcessingStatusResponse(
            session_id=session_id,
            status="completed",
            current_stage="완료",
            progress_percentage=100,
            estimated_remaining_time=0
        )
        
    except Exception as e:
        logger.error(f"❌ 상태 조회 실패: {e}")
        raise HTTPException(500, "상태 조회 중 오류가 발생했습니다.")

@router.get("/models/status")
async def get_models_status():
    """
    🤖 AI 모델 상태 조회
    
    현재 로드된 AI 모델들의 상태를 조회합니다.
    """
    try:
        if pipeline_manager and pipeline_manager.is_initialized:
            status = await pipeline_manager.get_pipeline_status()
            return {
                "available": True,
                "initialized": status['initialized'],
                "device": status['device'],
                "models": status.get('steps_status', {}),
                "performance": status.get('performance_metrics', {}),
                "memory_usage": status.get('memory_usage', {})
            }
        else:
            return {
                "available": False,
                "initialized": False,
                "message": "AI 파이프라인이 초기화되지 않았습니다."
            }
            
    except Exception as e:
        logger.error(f"❌ 모델 상태 조회 실패: {e}")
        return {
            "available": False,
            "error": str(e)
        }

@router.post("/analyze/body")
async def analyze_body(image: UploadFile = File(...)):
    """
    👤 신체 분석 API
    
    사용자 이미지에서 신체 정보를 분석합니다.
    """
    session_id = str(uuid.uuid4())
    
    try:
        validate_image_file(image)
        image_path = await save_uploaded_file(image, session_id, "body_analysis")
        
        # 기본 분석 결과 (실제로는 AI 모델 사용)
        analysis_result = {
            "body_type": "정상",
            "estimated_measurements": {
                "height_cm": 170,
                "chest_cm": 95,
                "waist_cm": 80,
                "hip_cm": 90
            },
            "pose_quality": 0.92,
            "clothing_recommendations": [
                "슬림핏 상의를 추천합니다",
                "허리가 잘록한 실루엣의 옷이 잘 어울립니다"
            ]
        }
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"❌ 신체 분석 실패: {e}")
        raise HTTPException(500, f"신체 분석 중 오류가 발생했습니다: {str(e)}")

@router.post("/analyze/clothing")
async def analyze_clothing(image: UploadFile = File(...)):
    """
    👕 의류 분석 API
    
    의류 이미지에서 스타일, 색상, 재질 등을 분석합니다.
    """
    session_id = str(uuid.uuid4())
    
    try:
        validate_image_file(image)
        image_path = await save_uploaded_file(image, session_id, "clothing_analysis")
        
        # 기본 분석 결과 (실제로는 AI 모델 사용)
        analysis_result = {
            "clothing_type": "셔츠",
            "colors": ["흰색", "파란색"],
            "pattern": "단색",
            "material": "면",
            "style": "캐주얼",
            "fit_type": "레귤러",
            "season": ["봄", "여름"],
            "care_instructions": ["세탁기 사용 가능", "다림질 중온"],
            "size_compatibility": {
                "small": 0.3,
                "medium": 0.8,
                "large": 0.6
            }
        }
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"❌ 의류 분석 실패: {e}")
        raise HTTPException(500, f"의류 분석 중 오류가 발생했습니다: {str(e)}")

@router.get("/supported-features")
async def get_supported_features():
    """
    🛠️ 지원 기능 목록
    
    현재 지원하는 의류 타입, 기능 등을 조회합니다.
    """
    return {
        "clothing_types": [
            {"id": "shirt", "name": "셔츠", "category": "상의"},
            {"id": "pants", "name": "바지", "category": "하의"},
            {"id": "dress", "name": "원피스", "category": "전신"},
            {"id": "jacket", "name": "재킷", "category": "상의"},
            {"id": "skirt", "name": "스커트", "category": "하의"}
        ],
        "fabric_types": [
            {"id": "cotton", "name": "면"},
            {"id": "denim", "name": "데님"},
            {"id": "silk", "name": "실크"},
            {"id": "polyester", "name": "폴리에스터"},
            {"id": "wool", "name": "울"}
        ],
        "style_preferences": [
            {"id": "slim", "name": "슬림"},
            {"id": "regular", "name": "레귤러"},
            {"id": "loose", "name": "루즈"}
        ],
        "quality_levels": [
            {"id": "fast", "name": "빠름", "description": "5초 내"},
            {"id": "balanced", "name": "균형", "description": "15초 내"},
            {"id": "high", "name": "고품질", "description": "30초 내"},
            {"id": "ultra", "name": "최고품질", "description": "60초 내"}
        ],
        "max_file_size": "10MB",
        "supported_formats": ["JPG", "JPEG", "PNG"]
    }

# 헬퍼 함수들

async def _demo_virtual_tryon(session_id: str, start_time: float) -> VirtualTryOnResponse:
    """데모 모드 가상 피팅 (AI 파이프라인 없을 때)"""
    
    # 시뮬레이션된 처리 시간
    await asyncio.sleep(2)
    processing_time = time.time() - start_time
    
    return VirtualTryOnResponse(
        success=True,
        session_id=session_id,
        fitted_image_url="/static/demo/sample_result.jpg",
        fitted_image_base64="",
        processing_time=processing_time,
        confidence=0.85,
        fit_score=0.88,
        quality_score=0.82,
        quality_grade="Demo",
        recommendations=[
            "이 색상이 당신에게 잘 어울립니다!",
            "사이즈가 적절해 보입니다.",
            "AI 모델을 설치하면 더 정확한 결과를 얻을 수 있습니다."
        ],
        measurements={}
    )

async def _cleanup_session_files(session_id: str):
    """세션 파일 정리 (백그라운드 작업)"""
    try:
        # 1시간 후 임시 파일 삭제
        await asyncio.sleep(3600)
        
        upload_dir = Path("static/uploads") / session_id
        result_dir = Path("static/results") / session_id
        
        import shutil
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        if result_dir.exists():
            shutil.rmtree(result_dir)
            
        logger.info(f"🧹 세션 파일 정리 완료: {session_id}")
        
    except Exception as e:
        logger.warning(f"⚠️ 세션 파일 정리 실패 {session_id}: {e}")

# 애플리케이션 시작시 실행될 이벤트
@router.on_event("startup")
async def startup_event():
    """API 시작시 파이프라인 초기화"""
    logger.info("🚀 Virtual Try-On API 시작...")
    
    # 필요한 디렉토리 생성
    Path("static/uploads").mkdir(parents=True, exist_ok=True)
    Path("static/results").mkdir(parents=True, exist_ok=True)
    
    # 파이프라인 초기화 (백그라운드)
    asyncio.create_task(initialize_pipeline())

@router.on_event("shutdown") 
async def shutdown_event():
    """API 종료시 리소스 정리"""
    logger.info("🛑 Virtual Try-On API 종료...")
    
    if pipeline_manager:
        await pipeline_manager.cleanup()