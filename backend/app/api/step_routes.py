"""
app/api/step_routes.py - 단순화된 API 레이어 (최종 버전)

✅ API는 순수하게 요청/응답 처리만
✅ 모든 비즈니스 로직은 서비스 레이어로 위임
✅ 프론트엔드 App.tsx 100% 호환성 유지
✅ 표준화된 에러 처리
✅ 코드 중복 제거
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# 서비스 레이어 import (핵심!)
from app.services.pipeline_service import get_pipeline_service

# 스키마 import (선택적)
try:
    from app.models.schemas import BodyMeasurements
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    # 폴백 스키마
    class BodyMeasurements:
        def __init__(self, height: float, weight: float, **kwargs):
            self.height = height
            self.weight = weight
            for k, v in kwargs.items():
                setattr(self, k, v)

# 로깅 설정
logger = logging.getLogger(__name__)

# FastAPI 라우터 초기화
router = APIRouter(prefix="/api/step", tags=["AI 파이프라인 8단계"])

# ============================================================================
# 🔧 공통 응답 처리 함수들 (API 레이어의 유일한 로직)
# ============================================================================

def create_success_response(result: Dict[str, Any], status_code: int = 200) -> JSONResponse:
    """성공 응답 생성 (표준화)"""
    return JSONResponse(
        content=result,
        status_code=status_code
    )

def create_error_response(
    error_message: str, 
    step_id: Optional[int] = None,
    status_code: int = 500
) -> JSONResponse:
    """에러 응답 생성 (표준화)"""
    return JSONResponse(
        content={
            "success": False,
            "error": error_message,
            "step_id": step_id,
            "timestamp": datetime.now().isoformat(),
            "processing_time": 0,
            "device": "unknown"
        },
        status_code=status_code
    )

# ============================================================================
# 🚀 8단계 API 엔드포인트들 (단순화된 버전)
# ============================================================================

@router.post("/1/upload-validation")
async def step_1_upload_validation(
    person_image: UploadFile = File(..., description="사용자 이미지"),
    clothing_image: UploadFile = File(..., description="의류 이미지")
):
    """
    1단계: 이미지 업로드 검증
    
    API 레이어 역할: 요청 받기 → 서비스 호출 → 응답 반환
    """
    try:
        # 서비스 레이어 호출 (모든 로직은 여기서 처리)
        pipeline_service = await get_pipeline_service()
        
        # 입력 데이터 구성
        inputs = {
            "person_image": person_image,
            "clothing_image": clothing_image
        }
        
        # 서비스 레이어에서 처리 (비즈니스 로직)
        result = await pipeline_service.process_step(1, inputs)
        
        # 응답 반환 (API 레이어의 역할)
        return create_success_response(result, 200 if result["success"] else 400)
        
    except Exception as e:
        logger.error(f"❌ Step 1 API 오류: {e}")
        return create_error_response(f"Step 1 처리 실패: {str(e)}", step_id=1)

@router.post("/2/measurements-validation")
async def step_2_measurements_validation(
    measurements: BodyMeasurements
):
    """
    2단계: 신체 측정 검증
    
    API 레이어 역할: 요청 받기 → 서비스 호출 → 응답 반환
    """
    try:
        # 서비스 레이어 호출
        pipeline_service = await get_pipeline_service()
        
        # 입력 데이터 구성
        inputs = {
            "measurements": measurements
        }
        
        # 서비스 레이어에서 처리
        result = await pipeline_service.process_step(2, inputs)
        
        # 응답 반환
        return create_success_response(result, 200 if result["success"] else 400)
        
    except Exception as e:
        logger.error(f"❌ Step 2 API 오류: {e}")
        return create_error_response(f"Step 2 처리 실패: {str(e)}", step_id=2)

@router.post("/3/human-parsing")
async def step_3_human_parsing(
    person_image: UploadFile = File(..., description="사용자 이미지")
):
    """
    3단계: 인간 파싱
    
    API 레이어 역할: 요청 받기 → 서비스 호출 → 응답 반환
    """
    try:
        # 서비스 레이어 호출
        pipeline_service = await get_pipeline_service()
        
        # 입력 데이터 구성
        inputs = {
            "person_image": person_image
        }
        
        # 서비스 레이어에서 처리
        result = await pipeline_service.process_step(3, inputs)
        
        # 응답 반환
        return create_success_response(result, 200 if result["success"] else 400)
        
    except Exception as e:
        logger.error(f"❌ Step 3 API 오류: {e}")
        return create_error_response(f"Step 3 처리 실패: {str(e)}", step_id=3)

@router.post("/4/pose-estimation")
async def step_4_pose_estimation(
    person_image: UploadFile = File(..., description="사용자 이미지")
):
    """
    4단계: 포즈 추정
    
    API 레이어 역할: 요청 받기 → 서비스 호출 → 응답 반환
    """
    try:
        # 서비스 레이어 호출
        pipeline_service = await get_pipeline_service()
        
        # 입력 데이터 구성
        inputs = {
            "person_image": person_image
        }
        
        # 서비스 레이어에서 처리
        result = await pipeline_service.process_step(4, inputs)
        
        # 응답 반환
        return create_success_response(result, 200 if result["success"] else 400)
        
    except Exception as e:
        logger.error(f"❌ Step 4 API 오류: {e}")
        return create_error_response(f"Step 4 처리 실패: {str(e)}", step_id=4)

@router.post("/5/clothing-analysis")
async def step_5_clothing_analysis(
    clothing_image: UploadFile = File(..., description="의류 이미지"),
    clothing_type: str = Form("auto_detect", description="의류 타입")
):
    """
    5단계: 의류 분석
    
    API 레이어 역할: 요청 받기 → 서비스 호출 → 응답 반환
    """
    try:
        # 서비스 레이어 호출
        pipeline_service = await get_pipeline_service()
        
        # 입력 데이터 구성
        inputs = {
            "clothing_image": clothing_image,
            "clothing_type": clothing_type
        }
        
        # 서비스 레이어에서 처리
        result = await pipeline_service.process_step(5, inputs)
        
        # 응답 반환
        return create_success_response(result, 200 if result["success"] else 400)
        
    except Exception as e:
        logger.error(f"❌ Step 5 API 오류: {e}")
        return create_error_response(f"Step 5 처리 실패: {str(e)}", step_id=5)

@router.post("/6/geometric-matching")
async def step_6_geometric_matching(
    person_image: UploadFile = File(..., description="사용자 이미지"),
    clothing_image: UploadFile = File(..., description="의류 이미지")
):
    """
    6단계: 기하학적 매칭
    
    API 레이어 역할: 요청 받기 → 서비스 호출 → 응답 반환
    """
    try:
        # 서비스 레이어 호출
        pipeline_service = await get_pipeline_service()
        
        # 입력 데이터 구성
        inputs = {
            "person_image": person_image,
            "clothing_image": clothing_image
        }
        
        # 서비스 레이어에서 처리
        result = await pipeline_service.process_step(6, inputs)
        
        # 응답 반환
        return create_success_response(result, 200 if result["success"] else 400)
        
    except Exception as e:
        logger.error(f"❌ Step 6 API 오류: {e}")
        return create_error_response(f"Step 6 처리 실패: {str(e)}", step_id=6)

@router.post("/7/virtual-fitting")
async def step_7_virtual_fitting(
    person_image: UploadFile = File(..., description="사용자 이미지"),
    clothing_image: UploadFile = File(..., description="의류 이미지"),
    clothing_type: str = Form("auto_detect", description="의류 타입")
):
    """
    7단계: 가상 피팅
    
    API 레이어 역할: 요청 받기 → 서비스 호출 → 응답 반환
    """
    try:
        # 서비스 레이어 호출
        pipeline_service = await get_pipeline_service()
        
        # 입력 데이터 구성
        inputs = {
            "person_image": person_image,
            "clothing_image": clothing_image,
            "clothing_type": clothing_type
        }
        
        # 서비스 레이어에서 처리
        result = await pipeline_service.process_step(7, inputs)
        
        # 응답 반환
        return create_success_response(result, 200 if result["success"] else 400)
        
    except Exception as e:
        logger.error(f"❌ Step 7 API 오류: {e}")
        return create_error_response(f"Step 7 처리 실패: {str(e)}", step_id=7)

@router.post("/8/result-analysis")
async def step_8_result_analysis(
    result_image: UploadFile = File(..., description="결과 이미지")
):
    """
    8단계: 결과 분석
    
    API 레이어 역할: 요청 받기 → 서비스 호출 → 응답 반환
    """
    try:
        # 서비스 레이어 호출
        pipeline_service = await get_pipeline_service()
        
        # 입력 데이터 구성
        inputs = {
            "result_image": result_image
        }
        
        # 서비스 레이어에서 처리
        result = await pipeline_service.process_step(8, inputs)
        
        # 응답 반환
        return create_success_response(result, 200 if result["success"] else 400)
        
    except Exception as e:
        logger.error(f"❌ Step 8 API 오류: {e}")
        return create_error_response(f"Step 8 처리 실패: {str(e)}", step_id=8)

# ============================================================================
# 🔍 모니터링 & 헬스체크 엔드포인트들 (API 레이어 전용)
# ============================================================================

@router.get("/health")
async def step_api_health():
    """
    Step API 헬스체크
    
    API 레이어 역할: 서비스 상태 확인 → 응답 반환
    """
    try:
        # 서비스 상태 확인
        pipeline_service = await get_pipeline_service()
        service_status = pipeline_service.get_status()
        
        return JSONResponse(content={
            "status": "healthy",
            "message": "AI 파이프라인 8단계 API 정상 동작",
            "timestamp": datetime.now().isoformat(),
            "service_status": service_status,
            "available_steps": list(range(1, 9)),
            "api_version": "2.0.0-service-layer",
            "architecture": "서비스 레이어 기반",
            "layer": "API Layer (단순 요청/응답 처리)"
        })
        
    except Exception as e:
        logger.error(f"❌ Health check 실패: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "layer": "API Layer"
            },
            status_code=500
        )

@router.get("/status")
async def step_api_status():
    """
    Step API 상태 조회
    
    API 레이어 역할: 서비스 상태 조회 → 응답 반환
    """
    try:
        # 서비스 상태 확인
        pipeline_service = await get_pipeline_service()
        service_status = pipeline_service.get_status()
        
        return JSONResponse(content={
            **service_status,
            "available_endpoints": [
                "POST /api/step/1/upload-validation",
                "POST /api/step/2/measurements-validation",
                "POST /api/step/3/human-parsing",
                "POST /api/step/4/pose-estimation",
                "POST /api/step/5/clothing-analysis",
                "POST /api/step/6/geometric-matching",
                "POST /api/step/7/virtual-fitting",
                "POST /api/step/8/result-analysis",
                "GET /api/step/health",
                "GET /api/step/status"
            ],
            "api_version": "2.0.0-service-layer",
            "timestamp": datetime.now().isoformat(),
            "layer": "API Layer (단순 요청/응답 처리)",
            "architecture_notes": {
                "api_layer": "요청/응답 처리만",
                "service_layer": "비즈니스 로직 처리",
                "ai_layer": "AI 모델 실행"
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Status check 실패: {e}")
        return JSONResponse(
            content={
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "layer": "API Layer"
            },
            status_code=500
        )

# ============================================================================
# 🎯 EXPORT
# ============================================================================

# main.py에서 라우터 등록용
__all__ = ["router"]

# ============================================================================
# 🎉 COMPLETION MESSAGE
# ============================================================================

logger.info("🎉 단순화된 Step Routes API 완성!")
logger.info("✅ API 레이어: 순수하게 요청/응답 처리만")
logger.info("✅ 모든 비즈니스 로직은 서비스 레이어로 위임")
logger.info("✅ 프론트엔드 App.tsx 100% 호환성 유지")
logger.info("✅ 표준화된 에러 처리")
logger.info("🔥 완벽한 레이어 분리 완성!")

"""
🎯 단순화된 API 레이어의 특징:

📦 API 레이어 (step_routes.py):
- 역할: HTTP 요청/응답 처리만
- 하는 일: 요청 받기 → 서비스 호출 → 응답 반환
- 비즈니스 로직: 0% (모든 로직을 서비스 레이어로 위임)

🔧 서비스 레이어 (pipeline_service.py):
- 역할: 비즈니스 로직 처리
- 하는 일: 데이터 검증, AI 처리, 에러 처리, 상태 관리
- 비즈니스 로직: 100%

🎯 AI 처리 레이어 (pipeline_manager.py, steps/):
- 역할: 실제 AI 모델 실행
- 하는 일: AI 모델 로딩, 이미지 처리, 결과 생성

✅ 장점:
- 명확한 책임 분리
- 코드 재사용성 향상
- 테스트 용이성
- 유지보수성 증대
- 확장성 확보

🔄 호출 플로우:
프론트엔드 → API 레이어 → 서비스 레이어 → AI 처리 레이어
"""