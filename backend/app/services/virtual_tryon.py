# app/api/virtual_tryon.py
"""
MyCloset AI Virtual Try-On API - 강화된 가상 피팅 라우터
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
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import logging
from io import BytesIO

# 기존 ai_pipeline 구조 import
try:
    from app.ai_pipeline.pipeline_manager import PipelineManager, get_pipeline_manager
    from app.ai_pipeline.utils.memory_manager import MemoryManager
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
    fitted_image: Optional[str] = None  # mycloset-uiux.tsx 호환
    processing_time: float
    confidence: float = Field(..., description="전체 신뢰도")
    fit_score: float = Field(..., description="핏 점수")
    quality_score: float = Field(..., description="품질 점수")
    quality_grade: str = Field(..., description="품질 등급")
    recommendations: List[str] = Field(default_factory=list)
    measurements: Dict[str, Any] = Field(default_factory=dict)
    clothing_analysis: Dict[str, Any] = Field(default_factory=dict)
    quality_analysis: Dict[str, Any] = Field(default_factory=dict)
    processing_info: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    
class ProcessingStatusResponse(BaseModel):
    session_id: str
    status: str  # "processing", "completed", "failed"
    current_stage: str
    progress_percentage: int
    estimated_remaining_time: Optional[float] = None
    error: Optional[str] = None

class BodyAnalysisResponse(BaseModel):
    body_type: str
    estimated_measurements: Dict[str, float]
    pose_quality: float
    clothing_recommendations: List[str]

class ClothingAnalysisResponse(BaseModel):
    clothing_type: str
    colors: List[str]
    pattern: str
    material: str
    style: str
    fit_type: str
    season: List[str]
    care_instructions: List[str]
    size_compatibility: Dict[str, float]

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
    # 파일 크기 검증 (50MB 제한)
    if upload_file.size > 50 * 1024 * 1024:
        raise HTTPException(400, "파일 크기가 너무 큽니다. 50MB 이하로 업로드해주세요.")
    
    # 파일 형식 검증
    if not upload_file.content_type.startswith('image/'):
        raise HTTPException(400, "이미지 파일만 업로드 가능합니다.")
    
    return True

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

def image_to_base64(image_path: str) -> str:
    """이미지를 base64로 인코딩"""
    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            encoded = base64.b64encode(img_data).decode('utf-8')
            return encoded
    except Exception as e:
        logger.error(f"❌ Base64 인코딩 실패: {e}")
        return ""

async def process_result_image(result_image: Any, session_id: str) -> str:
    """결과 이미지 처리 및 base64 인코딩"""
    try:
        # PIL 이미지로 변환
        if hasattr(result_image, 'save'):
            pil_image = result_image
        elif isinstance(result_image, np.ndarray):
            pil_image = Image.fromarray(result_image)
        else:
            pil_image = Image.fromarray(np.array(result_image))
        
        # 파일로 저장
        result_dir = Path("static/results")
        result_dir.mkdir(parents=True, exist_ok=True)
        save_path = result_dir / f"{session_id}_result.jpg"
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

def create_demo_composite(person_image: Image.Image, clothing_image: Image.Image) -> Image.Image:
    """데모용 간단한 합성 이미지 생성"""
    try:
        # 크기 조정
        person_resized = person_image.resize((512, 512), Image.Resampling.LANCZOS)
        clothing_resized = clothing_image.resize((256, 256), Image.Resampling.LANCZOS)
        
        # 합성 이미지 생성
        result = person_resized.copy()
        
        # 의류 이미지를 우상단에 오버레이
        result.paste(clothing_resized, (256, 0))
        
        # 데모 텍스트 추가
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
# 메인 API 엔드포인트들
# ========================================

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
    기존 8단계 AI 파이프라인을 활용한 고품질 착용 결과를 생성합니다.
    """
    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"🎯 가상 피팅 요청 시작 - 세션: {session_id}")
        
        # 파이프라인 초기화 확인
        if not await initialize_pipeline():
            logger.warning("⚠️ AI 파이프라인 없음 - 데모 모드로 실행")
            return await _demo_virtual_tryon(
                person_image, clothing_image, height, weight, 
                clothing_type, session_id, start_time
            )
        
        # 파일 검증
        validate_image_file(person_image)
        validate_image_file(clothing_image)
        
        # 파일 저장
        person_image_path = await save_uploaded_file(person_image, session_id, "person")
        clothing_image_path = await save_uploaded_file(clothing_image, session_id, "clothing")
        
        # 신체 치수 구성
        body_measurements = {
            "height": height,
            "weight": weight,
            "bmi": weight / ((height/100) ** 2)
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
        
        # 기존 8단계 AI 파이프라인 실행
        logger.info("🤖 8단계 AI 파이프라인 실행 중...")
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
            # 결과 이미지 처리
            result_image_base64 = await process_result_image(
                result.get('result_image'), session_id
            )
            
            processing_time = time.time() - start_time
            
            # mycloset-uiux.tsx 호환 응답 구성
            response = VirtualTryOnResponse(
                success=True,
                session_id=session_id,
                fitted_image_url=f"/static/results/{session_id}_result.jpg",
                fitted_image_base64=result_image_base64,
                fitted_image=result_image_base64,  # UI 호환성
                processing_time=processing_time,
                confidence=result.get('final_quality_score', 0.85),
                fit_score=result.get('fit_analysis', {}).get('overall_fit_score', 0.88),
                quality_score=result.get('final_quality_score', 0.85),
                quality_grade=result.get('quality_grade', 'Good'),
                recommendations=result.get('improvement_suggestions', {}).get('user_experience', [
                    f"✅ {clothing_type} 스타일이 잘 어울립니다!",
                    "📐 완벽한 핏을 위해 정확한 치수를 확인해보세요",
                    "🎨 다른 색상도 시도해보세요"
                ])[:3],
                measurements={
                    "chest": estimate_chest_measurement(height, weight),
                    "waist": estimate_waist_measurement(height, weight),
                    "hip": estimate_hip_measurement(height, weight),
                    "bmi": body_measurements["bmi"]
                },
                clothing_analysis={
                    "category": clothing_type,
                    "style": style_preference,
                    "fabric": fabric_type,
                    "dominant_color": [128, 128, 128]
                },
                quality_analysis={
                    "overall_score": result.get('final_quality_score', 0.85),
                    "fit_quality": result.get('fit_analysis', {}).get('overall_fit_score', 0.8),
                    "processing_quality": min(1.0, 30.0 / processing_time) if processing_time > 0 else 1.0
                },
                processing_info={
                    "steps_completed": len(result.get('step_results_summary', {})),
                    "quality_level": quality_level,
                    "device_used": result.get('processing_info', {}).get('device_used', 'cpu'),
                    "optimization": "M3_Max" if "mps" in str(result.get('processing_info', {})) else "Standard"
                }
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
                measurements=body_measurements,
                clothing_analysis={},
                quality_analysis={},
                processing_info={}
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
            measurements={"height": height, "weight": weight},
            clothing_analysis={},
            quality_analysis={},
            processing_info={}
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

@router.post("/analyze/body", response_model=BodyAnalysisResponse)
async def analyze_body(image: UploadFile = File(...)):
    """
    👤 신체 분석 API
    
    사용자 이미지에서 신체 정보를 분석합니다.
    기존 AI 파이프라인의 인체 파싱 단계를 활용합니다.
    """
    session_id = str(uuid.uuid4())
    
    try:
        validate_image_file(image)
        image_path = await save_uploaded_file(image, session_id, "body_analysis")
        
        # 실제 AI 분석 (파이프라인 사용 가능한 경우)
        if pipeline_manager and pipeline_manager.is_initialized:
            # 기존 파이프라인의 step_01_human_parsing 활용
            try:
                # TODO: pipeline_manager에 body_analysis 메서드 추가 필요
                # body_analysis_result = await pipeline_manager.analyze_body(image_path)
                pass
            except Exception as e:
                logger.warning(f"AI 신체 분석 실패, 기본값 사용: {e}")
        
        # 기본 분석 결과
        analysis_result = BodyAnalysisResponse(
            body_type="정상",
            estimated_measurements={
                "height_cm": 170,
                "chest_cm": 95,
                "waist_cm": 80,
                "hip_cm": 90
            },
            pose_quality=0.92,
            clothing_recommendations=[
                "슬림핏 상의를 추천합니다",
                "허리가 잘록한 실루엣의 옷이 잘 어울립니다"
            ]
        )
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"❌ 신체 분석 실패: {e}")
        raise HTTPException(500, f"신체 분석 중 오류가 발생했습니다: {str(e)}")

@router.post("/analyze/clothing", response_model=ClothingAnalysisResponse)
async def analyze_clothing(image: UploadFile = File(...)):
    """
    👕 의류 분석 API
    
    의류 이미지에서 스타일, 색상, 재질 등을 분석합니다.
    기존 AI 파이프라인의 의류 세그멘테이션 단계를 활용합니다.
    """
    session_id = str(uuid.uuid4())
    
    try:
        validate_image_file(image)
        image_path = await save_uploaded_file(image, session_id, "clothing_analysis")
        
        # 실제 AI 분석 (파이프라인 사용 가능한 경우)
        if pipeline_manager and pipeline_manager.is_initialized:
            # 기존 파이프라인의 step_03_cloth_segmentation 활용
            try:
                # TODO: pipeline_manager에 clothing_analysis 메서드 추가 필요
                # clothing_analysis_result = await pipeline_manager.analyze_clothing(image_path)
                pass
            except Exception as e:
                logger.warning(f"AI 의류 분석 실패, 기본값 사용: {e}")
        
        # 기본 분석 결과
        analysis_result = ClothingAnalysisResponse(
            clothing_type="셔츠",
            colors=["흰색", "파란색"],
            pattern="단색",
            material="면",
            style="캐주얼",
            fit_type="레귤러",
            season=["봄", "여름"],
            care_instructions=["세탁기 사용 가능", "다림질 중온"],
            size_compatibility={
                "small": 0.3,
                "medium": 0.8,
                "large": 0.6
            }
        )
        
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
            {"id": "skirt", "name": "스커트", "category": "하의"},
            {"id": "t-shirt", "name": "티셔츠", "category": "상의"},
            {"id": "blouse", "name": "블라우스", "category": "상의"}
        ],
        "fabric_types": [
            {"id": "cotton", "name": "면"},
            {"id": "denim", "name": "데님"},
            {"id": "silk", "name": "실크"},
            {"id": "polyester", "name": "폴리에스터"},
            {"id": "wool", "name": "울"},
            {"id": "linen", "name": "린넨"},
            {"id": "knit", "name": "니트"}
        ],
        "style_preferences": [
            {"id": "slim", "name": "슬림"},
            {"id": "regular", "name": "레귤러"},
            {"id": "loose", "name": "루즈"},
            {"id": "oversized", "name": "오버사이즈"}
        ],
        "quality_levels": [
            {"id": "fast", "name": "빠름", "description": "5초 내", "target_time": 5},
            {"id": "medium", "name": "보통", "description": "15초 내", "target_time": 15},
            {"id": "high", "name": "고품질", "description": "30초 내", "target_time": 30},
            {"id": "ultra", "name": "최고품질", "description": "60초 내", "target_time": 60}
        ],
        "max_file_size": "50MB",
        "supported_formats": ["JPG", "JPEG", "PNG"],
        "pipeline_info": {
            "available": AI_PIPELINE_AVAILABLE,
            "initialized": pipeline_manager.is_initialized if pipeline_manager else False,
            "total_steps": 8,
            "steps": [
                "Human Parsing",
                "Pose Estimation", 
                "Cloth Segmentation",
                "Geometric Matching",
                "Cloth Warping",
                "Virtual Fitting",
                "Post Processing",
                "Quality Assessment"
            ]
        }
    }

# ========================================
# 헬퍼 함수들
# ========================================

async def _demo_virtual_tryon(
    person_image: UploadFile,
    clothing_image: UploadFile,
    height: float,
    weight: float,
    clothing_type: str,
    session_id: str,
    start_time: float
) -> VirtualTryOnResponse:
    """데모 모드 가상 피팅 (AI 파이프라인 없을 때)"""
    
    try:
        # 이미지 로드
        person_pil = await load_and_validate_image(person_image, "person")
        clothing_pil = await load_and_validate_image(clothing_image, "clothing")
        
        # 간단한 합성 이미지 생성
        demo_result = create_demo_composite(person_pil, clothing_pil)
        
        # 이미지 저장 및 base64 인코딩
        result_base64 = await process_result_image(demo_result, session_id)
        
        # 시뮬레이션된 처리 시간
        await asyncio.sleep(2)
        processing_time = time.time() - start_time
        
        return VirtualTryOnResponse(
            success=True,
            session_id=session_id,
            fitted_image_url=f"/static/results/{session_id}_result.jpg",
            fitted_image_base64=result_base64,
            fitted_image=result_base64,
            processing_time=processing_time,
            confidence=0.75,  # 데모 모드 신뢰도
            fit_score=0.78,
            quality_score=0.72,
            quality_grade="Demo",
            recommendations=[
                "🚧 데모 모드로 처리되었습니다",
                "⚡ AI 모델 로딩 완료 후 더 정확한 결과를 얻을 수 있습니다",
                f"👔 {clothing_type} 스타일 시뮬레이션"
            ],
            measurements={
                "chest": estimate_chest_measurement(height, weight),
                "waist": estimate_waist_measurement(height, weight),
                "hip": estimate_hip_measurement(height, weight),
                "bmi": weight / ((height/100) ** 2)
            },
            clothing_analysis={
                "category": clothing_type,
                "style": "casual",
                "fabric": "cotton",
                "dominant_color": [100, 100, 150]
            },
            quality_analysis={
                "overall_score": 0.75,
                "fit_quality": 0.7,
                "processing_quality": 1.0
            },
            processing_info={
                "steps_completed": 0,
                "quality_level": "demo",
                "device_used": "cpu",
                "optimization": "Demo Mode",
                "demo_mode": True
            }
        )
        
    except Exception as e:
        logger.error(f"❌ 데모 모드 처리 실패: {e}")
        raise HTTPException(500, "데모 모드 처리 중 오류가 발생했습니다.")

async def _cleanup_session_files(session_id: str):
    """세션 파일 정리 (백그라운드 작업)"""
    try:
        # 1시간 후 임시 파일 삭제
        await asyncio.sleep(3600)
        
        upload_dir = Path("static/uploads") / session_id
        
        import shutil
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
            
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