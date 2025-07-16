"""
backend/app/api/step_routes.py - 완전히 분리된 API 레이어

✅ API 처리만 담당 (비즈니스 로직 없음)
✅ StepServiceManager를 통한 서비스 레이어 호출
✅ HTTP 요청/응답 처리 전담
✅ 입력 검증 및 변환
✅ 에러 처리 및 응답 포맷팅
✅ 프론트엔드 100% 호환
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

# FastAPI 필수 import
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 서비스 레이어 import (의존성 주입)
try:
    from app.services.step_service import (
        get_step_service_manager,
        StepServiceManager,
        BodyMeasurements as ServiceBodyMeasurements
    )
    STEP_SERVICE_AVAILABLE = True
except ImportError as e:
    logging.error(f"StepService import 실패: {e}")
    STEP_SERVICE_AVAILABLE = False
    raise RuntimeError("StepService가 필요합니다")

# 스키마 import (폴백 포함)
try:
    from app.models.schemas import BodyMeasurements, VirtualTryOnRequest
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False

# 로깅 설정
logger = logging.getLogger(__name__)

# ============================================================================
# 🏗️ API 스키마 정의 (폴백 포함)
# ============================================================================

if not SCHEMAS_AVAILABLE:
    class BodyMeasurements(BaseModel):
        height: float = Field(..., description="키 (cm)", ge=140, le=220)
        weight: float = Field(..., description="몸무게 (kg)", ge=40, le=150)
        chest: Optional[float] = Field(None, description="가슴둘레 (cm)", ge=70, le=130)
        waist: Optional[float] = Field(None, description="허리둘레 (cm)", ge=60, le=120)
        hips: Optional[float] = Field(None, description="엉덩이둘레 (cm)", ge=80, le=140)
        
        class Config:
            schema_extra = {
                "example": {
                    "height": 175.0,
                    "weight": 70.0,
                    "chest": 95.0,
                    "waist": 80.0,
                    "hips": 98.0
                }
            }
    
    class VirtualTryOnRequest(BaseModel):
        clothing_type: str = Field("auto_detect", description="의류 타입")
        quality_target: float = Field(0.8, description="품질 목표 (0.0-1.0)", ge=0.0, le=1.0)
        save_intermediate: bool = Field(False, description="중간 결과 저장 여부")
        
        class Config:
            schema_extra = {
                "example": {
                    "clothing_type": "shirt",
                    "quality_target": 0.8,
                    "save_intermediate": False
                }
            }

class APIResponse(BaseModel):
    """표준 API 응답 스키마"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field("", description="응답 메시지")
    step_name: Optional[str] = Field(None, description="단계 이름")
    step_id: Optional[int] = Field(None, description="단계 ID")
    session_id: Optional[str] = Field(None, description="세션 ID")
    processing_time: float = Field(0.0, description="처리 시간 (초)")
    confidence: Optional[float] = Field(None, description="신뢰도")
    device: Optional[str] = Field(None, description="처리 디바이스")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    details: Optional[Dict[str, Any]] = Field(None, description="상세 정보")
    error: Optional[str] = Field(None, description="에러 메시지")

# ============================================================================
# 🔧 API 유틸리티 함수들
# ============================================================================

def convert_body_measurements(api_measurements: BodyMeasurements) -> ServiceBodyMeasurements:
    """API BodyMeasurements를 서비스 레이어용으로 변환"""
    return ServiceBodyMeasurements(
        height=api_measurements.height,
        weight=api_measurements.weight,
        chest=api_measurements.chest,
        waist=api_measurements.waist,
        hips=api_measurements.hips
    )

def format_api_response(service_result: Dict[str, Any]) -> Dict[str, Any]:
    """서비스 결과를 API 응답 형식으로 변환"""
    return {
        "success": service_result.get("success", False),
        "message": service_result.get("message", ""),
        "step_name": service_result.get("step_name"),
        "step_id": service_result.get("step_id"),
        "session_id": service_result.get("session_id"),
        "processing_time": service_result.get("processing_time", 0.0),
        "confidence": service_result.get("confidence"),
        "device": service_result.get("device"),
        "timestamp": service_result.get("timestamp", datetime.now().isoformat()),
        "details": service_result.get("details"),
        "error": service_result.get("error")
    }

def create_error_response(
    error_message: str, 
    step_name: str = None, 
    step_id: int = None,
    processing_time: float = 0.0
) -> Dict[str, Any]:
    """에러 응답 생성"""
    return {
        "success": False,
        "message": "처리 실패",
        "step_name": step_name,
        "step_id": step_id,
        "processing_time": processing_time,
        "timestamp": datetime.now().isoformat(),
        "error": error_message
    }

# ============================================================================
# 🔥 FastAPI 라우터 및 의존성 주입
# ============================================================================

# FastAPI 라우터 초기화
router = APIRouter(prefix="/api/step", tags=["8단계 가상 피팅 API"])

# 의존성 주입: StepServiceManager
async def get_service_manager() -> StepServiceManager:
    """StepServiceManager 의존성 주입"""
    if not STEP_SERVICE_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="StepService를 사용할 수 없습니다"
        )
    
    return await get_step_service_manager()

# ============================================================================
# 🎯 8단계 개별 API 엔드포인트들
# ============================================================================

@router.post("/1/upload-validation", response_model=APIResponse)
async def step_1_upload_validation(
    person_image: UploadFile = File(..., description="사람 이미지"),
    clothing_image: UploadFile = File(..., description="의류 이미지"),
    service_manager: StepServiceManager = Depends(get_service_manager)
):
    """1단계: 이미지 업로드 검증 API"""
    start_time = time.time()
    
    try:
        # 서비스 레이어 호출
        service_result = await service_manager.process_step(1, {
            "person_image": person_image,
            "clothing_image": clothing_image
        })
        
        # API 응답 형식으로 변환
        api_response = format_api_response(service_result)
        
        return JSONResponse(
            content=api_response,
            status_code=200 if api_response["success"] else 400
        )
        
    except Exception as e:
        logger.error(f"❌ Step 1 API 오류: {e}")
        
        error_response = create_error_response(
            error_message=f"Step 1 API 처리 실패: {str(e)}",
            step_name="이미지 업로드 검증",
            step_id=1,
            processing_time=time.time() - start_time
        )
        
        return JSONResponse(
            content=error_response,
            status_code=500
        )

@router.post("/2/measurements-validation", response_model=APIResponse)
async def step_2_measurements_validation(
    measurements: BodyMeasurements,
    session_id: Optional[str] = Form(None, description="세션 ID (선택적)"),
    service_manager: StepServiceManager = Depends(get_service_manager)
):
    """2단계: 신체 측정 검증 API"""
    start_time = time.time()
    
    try:
        # API 스키마를 서비스 레이어용으로 변환
        service_measurements = convert_body_measurements(measurements)
        
        # 서비스 레이어 호출
        service_result = await service_manager.process_step(2, {
            "measurements": service_measurements,
            "session_id": session_id
        })
        
        # API 응답 형식으로 변환
        api_response = format_api_response(service_result)
        
        return JSONResponse(
            content=api_response,
            status_code=200 if api_response["success"] else 400
        )
        
    except Exception as e:
        logger.error(f"❌ Step 2 API 오류: {e}")
        
        error_response = create_error_response(
            error_message=f"Step 2 API 처리 실패: {str(e)}",
            step_name="신체 측정 검증",
            step_id=2,
            processing_time=time.time() - start_time
        )
        
        return JSONResponse(
            content=error_response,
            status_code=500
        )

@router.post("/3/human-parsing", response_model=APIResponse)
async def step_3_human_parsing(
    person_image: UploadFile = File(..., description="사람 이미지"),
    session_id: Optional[str] = Form(None, description="세션 ID (선택적)"),
    service_manager: StepServiceManager = Depends(get_service_manager)
):
    """3단계: 인간 파싱 API"""
    start_time = time.time()
    
    try:
        # 서비스 레이어 호출
        service_result = await service_manager.process_step(3, {
            "person_image": person_image,
            "session_id": session_id
        })
        
        # API 응답 형식으로 변환
        api_response = format_api_response(service_result)
        
        return JSONResponse(
            content=api_response,
            status_code=200 if api_response["success"] else 400
        )
        
    except Exception as e:
        logger.error(f"❌ Step 3 API 오류: {e}")
        
        error_response = create_error_response(
            error_message=f"Step 3 API 처리 실패: {str(e)}",
            step_name="인간 파싱",
            step_id=3,
            processing_time=time.time() - start_time
        )
        
        return JSONResponse(
            content=error_response,
            status_code=500
        )

@router.post("/4/pose-estimation", response_model=APIResponse)
async def step_4_pose_estimation(
    person_image: UploadFile = File(..., description="사람 이미지"),
    session_id: Optional[str] = Form(None, description="세션 ID (선택적)"),
    service_manager: StepServiceManager = Depends(get_service_manager)
):
    """4단계: 포즈 추정 API"""
    start_time = time.time()
    
    try:
        # 서비스 레이어 호출
        service_result = await service_manager.process_step(4, {
            "person_image": person_image,
            "session_id": session_id
        })
        
        # API 응답 형식으로 변환
        api_response = format_api_response(service_result)
        
        return JSONResponse(
            content=api_response,
            status_code=200 if api_response["success"] else 400
        )
        
    except Exception as e:
        logger.error(f"❌ Step 4 API 오류: {e}")
        
        error_response = create_error_response(
            error_message=f"Step 4 API 처리 실패: {str(e)}",
            step_name="포즈 추정",
            step_id=4,
            processing_time=time.time() - start_time
        )
        
        return JSONResponse(
            content=error_response,
            status_code=500
        )

@router.post("/5/clothing-analysis", response_model=APIResponse)
async def step_5_clothing_analysis(
    clothing_image: UploadFile = File(..., description="의류 이미지"),
    clothing_type: str = Form("auto_detect", description="의류 타입"),
    session_id: Optional[str] = Form(None, description="세션 ID (선택적)"),
    service_manager: StepServiceManager = Depends(get_service_manager)
):
    """5단계: 의류 분석 API"""
    start_time = time.time()
    
    try:
        # 서비스 레이어 호출
        service_result = await service_manager.process_step(5, {
            "clothing_image": clothing_image,
            "clothing_type": clothing_type,
            "session_id": session_id
        })
        
        # API 응답 형식으로 변환
        api_response = format_api_response(service_result)
        
        return JSONResponse(
            content=api_response,
            status_code=200 if api_response["success"] else 400
        )
        
    except Exception as e:
        logger.error(f"❌ Step 5 API 오류: {e}")
        
        error_response = create_error_response(
            error_message=f"Step 5 API 처리 실패: {str(e)}",
            step_name="의류 분석",
            step_id=5,
            processing_time=time.time() - start_time
        )
        
        return JSONResponse(
            content=error_response,
            status_code=500
        )

@router.post("/6/geometric-matching", response_model=APIResponse)
async def step_6_geometric_matching(
    person_image: UploadFile = File(..., description="사람 이미지"),
    clothing_image: UploadFile = File(..., description="의류 이미지"),
    session_id: Optional[str] = Form(None, description="세션 ID (선택적)"),
    service_manager: StepServiceManager = Depends(get_service_manager)
):
    """6단계: 기하학적 매칭 API"""
    start_time = time.time()
    
    try:
        # 서비스 레이어 호출
        service_result = await service_manager.process_step(6, {
            "person_image": person_image,
            "clothing_image": clothing_image,
            "session_id": session_id
        })
        
        # API 응답 형식으로 변환
        api_response = format_api_response(service_result)
        
        return JSONResponse(
            content=api_response,
            status_code=200 if api_response["success"] else 400
        )
        
    except Exception as e:
        logger.error(f"❌ Step 6 API 오류: {e}")
        
        error_response = create_error_response(
            error_message=f"Step 6 API 처리 실패: {str(e)}",
            step_name="기하학적 매칭",
            step_id=6,
            processing_time=time.time() - start_time
        )
        
        return JSONResponse(
            content=error_response,
            status_code=500
        )

@router.post("/7/virtual-fitting", response_model=APIResponse)
async def step_7_virtual_fitting(
    person_image: UploadFile = File(..., description="사람 이미지"),
    clothing_image: UploadFile = File(..., description="의류 이미지"),
    clothing_type: str = Form("auto_detect", description="의류 타입"),
    quality_target: float = Form(0.8, description="품질 목표", ge=0.0, le=1.0),
    session_id: Optional[str] = Form(None, description="세션 ID (선택적)"),
    service_manager: StepServiceManager = Depends(get_service_manager)
):
    """7단계: 가상 피팅 API"""
    start_time = time.time()
    
    try:
        # 서비스 레이어 호출
        service_result = await service_manager.process_step(7, {
            "person_image": person_image,
            "clothing_image": clothing_image,
            "clothing_type": clothing_type,
            "quality_target": quality_target,
            "session_id": session_id
        })
        
        # API 응답 형식으로 변환
        api_response = format_api_response(service_result)
        
        return JSONResponse(
            content=api_response,
            status_code=200 if api_response["success"] else 400
        )
        
    except Exception as e:
        logger.error(f"❌ Step 7 API 오류: {e}")
        
        error_response = create_error_response(
            error_message=f"Step 7 API 처리 실패: {str(e)}",
            step_name="가상 피팅",
            step_id=7,
            processing_time=time.time() - start_time
        )
        
        return JSONResponse(
            content=error_response,
            status_code=500
        )

@router.post("/8/result-analysis", response_model=APIResponse)
async def step_8_result_analysis(
    result_image: Optional[UploadFile] = File(None, description="결과 이미지 (선택적)"),
    session_id: Optional[str] = Form(None, description="세션 ID"),
    service_manager: StepServiceManager = Depends(get_service_manager)
):
    """8단계: 결과 분석 API"""
    start_time = time.time()
    
    try:
        # 서비스 레이어 호출
        service_result = await service_manager.process_step(8, {
            "result_image": result_image,
            "session_id": session_id
        })
        
        # API 응답 형식으로 변환
        api_response = format_api_response(service_result)
        
        return JSONResponse(
            content=api_response,
            status_code=200 if api_response["success"] else 400
        )
        
    except Exception as e:
        logger.error(f"❌ Step 8 API 오류: {e}")
        
        error_response = create_error_response(
            error_message=f"Step 8 API 처리 실패: {str(e)}",
            step_name="결과 분석",
            step_id=8,
            processing_time=time.time() - start_time
        )
        
        return JSONResponse(
            content=error_response,
            status_code=500
        )

# ============================================================================
# 🎯 통합 파이프라인 API 엔드포인트
# ============================================================================

@router.post("/complete", response_model=APIResponse)
async def complete_pipeline_processing(
    person_image: UploadFile = File(..., description="사람 이미지"),
    clothing_image: UploadFile = File(..., description="의류 이미지"),
    measurements: Optional[BodyMeasurements] = None,
    clothing_type: str = Form("auto_detect", description="의류 타입"),
    quality_target: float = Form(0.8, description="품질 목표", ge=0.0, le=1.0),
    save_intermediate: bool = Form(False, description="중간 결과 저장 여부"),
    service_manager: StepServiceManager = Depends(get_service_manager)
):
    """완전한 8단계 파이프라인 처리 API"""
    start_time = time.time()
    
    try:
        # API 스키마를 서비스 레이어용으로 변환
        service_measurements = None
        if measurements:
            service_measurements = convert_body_measurements(measurements)
        
        # 진행률 콜백 (로깅용)
        async def progress_callback(message: str, percentage: int):
            logger.info(f"🔄 진행률: {percentage}% - {message}")
        
        # 서비스 레이어 호출
        service_result = await service_manager.process_complete_pipeline({
            "person_image": person_image,
            "clothing_image": clothing_image,
            "measurements": service_measurements,
            "clothing_type": clothing_type,
            "quality_target": quality_target,
            "save_intermediate": save_intermediate,
            "progress_callback": progress_callback
        })
        
        # API 응답 형식으로 변환
        api_response = format_api_response(service_result)
        
        return JSONResponse(
            content=api_response,
            status_code=200 if api_response["success"] else 400
        )
        
    except Exception as e:
        logger.error(f"❌ 완전한 파이프라인 API 오류: {e}")
        
        error_response = create_error_response(
            error_message=f"완전한 파이프라인 API 처리 실패: {str(e)}",
            step_name="완전한 파이프라인",
            step_id=0,
            processing_time=time.time() - start_time
        )
        
        return JSONResponse(
            content=error_response,
            status_code=500
        )

# ============================================================================
# 🔍 모니터링 & 관리 API 엔드포인트들
# ============================================================================

@router.get("/health")
async def step_api_health(
    service_manager: StepServiceManager = Depends(get_service_manager)
):
    """8단계 API 헬스체크"""
    try:
        # 서비스 매니저 메트릭 조회
        metrics = service_manager.get_all_metrics()
        
        return JSONResponse(content={
            "status": "healthy",
            "message": "8단계 가상 피팅 API 정상 동작",
            "timestamp": datetime.now().isoformat(),
            "api_layer": True,
            "service_layer_connected": True,
            "available_steps": list(range(1, 9)) + [0],  # 0은 완전한 파이프라인
            "service_metrics": metrics,
            "api_version": "3.0.0-separated-layers",
            "architecture": {
                "api_layer": "step_routes.py",
                "service_layer": "step_service.py", 
                "dependency_flow": "API Layer → Service Layer → PipelineManager → AI Steps"
            },
            "features": {
                "layer_separation": True,
                "dependency_injection": True,
                "error_handling": True,
                "response_formatting": True,
                "schema_validation": True
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Health check 실패: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "api_layer": True,
                "service_layer_connected": False
            },
            status_code=503
        )

@router.get("/status")
async def step_api_status(
    service_manager: StepServiceManager = Depends(get_service_manager)
):
    """8단계 API 상태 조회"""
    try:
        # 서비스 매니저 메트릭 조회
        metrics = service_manager.get_all_metrics()
        
        return JSONResponse(content={
            "api_layer_status": "operational",
            "service_layer_connected": True,
            "total_services": metrics["total_services"],
            "device": metrics["device"],
            "service_metrics": metrics["services"],
            "available_endpoints": [
                "POST /api/step/1/upload-validation",
                "POST /api/step/2/measurements-validation",
                "POST /api/step/3/human-parsing",
                "POST /api/step/4/pose-estimation",
                "POST /api/step/5/clothing-analysis",
                "POST /api/step/6/geometric-matching",
                "POST /api/step/7/virtual-fitting",
                "POST /api/step/8/result-analysis",
                "POST /api/step/complete",
                "GET /api/step/health",
                "GET /api/step/status",
                "GET /api/step/metrics",
                "POST /api/step/cleanup"
            ],
            "layer_architecture": {
                "api_layer": "HTTP 요청/응답 처리",
                "service_layer": "비즈니스 로직 처리",
                "pipeline_layer": "AI 모델 처리",
                "separation": "완전 분리"
            },
            "api_version": "3.0.0-separated-layers",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Status check 실패: {e}")
        return JSONResponse(
            content={
                "api_layer_status": "error",
                "service_layer_connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=503
        )

@router.get("/metrics")
async def step_api_metrics(
    service_manager: StepServiceManager = Depends(get_service_manager)
):
    """API 및 서비스 메트릭 조회"""
    try:
        # 서비스 레이어 메트릭
        service_metrics = service_manager.get_all_metrics()
        
        return JSONResponse(content={
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "api_metrics": {
                "layer": "API Layer",
                "endpoints_available": 13,
                "dependency_injection": True,
                "error_handling": True,
                "response_formatting": True
            },
            "service_metrics": service_metrics,
            "performance_summary": {
                "total_services": service_metrics["total_services"],
                "device": service_metrics["device"],
                "services_performance": {
                    service_id: {
                        "success_rate": service_data["success_rate"],
                        "average_time": service_data["average_processing_time"],
                        "total_requests": service_data["total_requests"]
                    }
                    for service_id, service_data in service_metrics["services"].items()
                }
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Metrics 조회 실패: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

@router.post("/cleanup")
async def cleanup_step_services(
    service_manager: StepServiceManager = Depends(get_service_manager)
):
    """서비스 레이어 정리"""
    try:
        # 서비스 매니저 정리
        await service_manager.cleanup_all()
        
        return JSONResponse(content={
            "success": True,
            "message": "모든 서비스 정리 완료",
            "timestamp": datetime.now().isoformat(),
            "api_layer": "정상",
            "service_layer": "정리됨"
        })
        
    except Exception as e:
        logger.error(f"❌ 서비스 정리 실패: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

# ============================================================================
# 🎯 특별 엔드포인트들
# ============================================================================

@router.get("/")
async def step_api_root():
    """API 루트 엔드포인트"""
    return JSONResponse(content={
        "message": "MyCloset AI - 8단계 가상 피팅 API",
        "version": "3.0.0-separated-layers",
        "architecture": "완전히 분리된 레이어 구조",
        "api_layer": "step_routes.py - HTTP 요청/응답 처리",
        "service_layer": "step_service.py - 비즈니스 로직 처리",
        "pipeline_layer": "pipeline_manager.py - AI 모델 처리",
        "available_endpoints": {
            "individual_steps": "/api/step/{1-8}/*",
            "complete_pipeline": "/api/step/complete",
            "monitoring": ["/api/step/health", "/api/step/status", "/api/step/metrics"],
            "management": "/api/step/cleanup"
        },
        "features": [
            "완전한 레이어 분리",
            "의존성 주입",
            "스키마 검증",
            "에러 처리",
            "응답 포맷팅",
            "성능 모니터링"
        ],
        "timestamp": datetime.now().isoformat()
    })

# ============================================================================
# 🎯 EXPORT
# ============================================================================

# main.py에서 라우터 등록용
__all__ = ["router"]

# ============================================================================
# 🎉 COMPLETION MESSAGE
# ============================================================================

logger.info("🎉 완전히 분리된 API 레이어 step_routes.py 완성!")
logger.info("✅ HTTP 요청/응답 처리만 담당")
logger.info("✅ 서비스 레이어와 완전 분리")
logger.info("✅ 의존성 주입을 통한 서비스 호출")
logger.info("✅ 스키마 검증 및 데이터 변환")
logger.info("✅ 표준화된 에러 처리")
logger.info("✅ 일관된 응답 포맷팅")
logger.info("✅ 프론트엔드 100% 호환")
logger.info("🔥 완벽한 레이어 분리 구조 완성!")

"""
🎯 완전히 분리된 레이어 구조 완성!

📚 레이어별 역할:
┌─────────────────────────────────────┐
│ step_routes.py (API Layer)          │
│ - HTTP 요청/응답 처리               │
│ - 스키마 검증 및 데이터 변환        │
│ - 에러 처리 및 응답 포맷팅          │
│ - 의존성 주입                       │
└─────────────────┬───────────────────┘
                  │ 의존성 주입
┌─────────────────▼───────────────────┐
│ step_service.py (Service Layer)     │
│ - 비즈니스 로직 처리                │
│ - PipelineManager 활용              │
│ - 데이터 검증 및 변환               │
│ - 리소스 관리                       │
└─────────────────┬───────────────────┘
                  │ 의존성
┌─────────────────▼───────────────────┐
│ PipelineManager (AI Layer)          │
│ - AI 모델 처리                      │
│ - 8단계 파이프라인 실행             │
│ - 메모리 최적화                     │
└─────────────────────────────────────┘

🔥 주요 특징:
- 완전한 관심사 분리 (Separation of Concerns)
- 의존성 주입 (Dependency Injection)
- 단일 책임 원칙 (Single Responsibility Principle)
- 개방-폐쇄 원칙 (Open-Closed Principle)
- 프론트엔드 100% 호환성 유지
"""