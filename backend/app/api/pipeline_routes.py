"""
8단계 AI 파이프라인 API 라우터 - WebSocket 실시간 상태 통합
"""
import asyncio
import io
import logging
import time
import uuid
from typing import Dict, Any, Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from PIL import Image

from ..ai_pipeline.pipeline_manager import get_pipeline_manager
from ..core.gpu_config import GPUConfig
from ..models.schemas import (
    VirtualTryOnRequest, VirtualTryOnResponse,
    PipelineStatusResponse, ProcessingStage
)
from .websocket_routes import create_progress_callback, manager as ws_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# 전역 변수
pipeline_manager = None
gpu_config = None

@router.on_event("startup")
async def startup_pipeline():
    """파이프라인 라우터 시작 시 초기화"""
    global pipeline_manager, gpu_config
    
    try:
        # GPU 설정 초기화
        gpu_config = GPUConfig()
        gpu_config.setup_memory_optimization()
        
        # 파이프라인 매니저 초기화
        pipeline_manager = get_pipeline_manager()
        
        # 모델 초기화 (백그라운드에서)
        asyncio.create_task(initialize_pipeline_models())
        
        logger.info("✅ 파이프라인 라우터 초기화 완료")
        
    except Exception as e:
        logger.error(f"❌ 파이프라인 라우터 초기화 실패: {e}")

async def initialize_pipeline_models():
    """백그라운드에서 파이프라인 모델 초기화"""
    try:
        logger.info("🔄 백그라운드에서 AI 모델 초기화 시작...")
        
        if pipeline_manager:
            success = await pipeline_manager.initialize()
            if success:
                logger.info("✅ AI 모델 초기화 완료")
                # 웜업 실행
                await pipeline_manager.warmup()
            else:
                logger.error("❌ AI 모델 초기화 실패")
        
    except Exception as e:
        logger.error(f"❌ 백그라운드 모델 초기화 실패: {e}")

@router.post("/virtual-tryon", response_model=VirtualTryOnResponse)
async def virtual_tryon_endpoint(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(..., description="사용자 이미지"),
    clothing_image: UploadFile = File(..., description="의류 이미지"),
    height: float = Form(170.0, description="키 (cm)"),
    weight: float = Form(65.0, description="몸무게 (kg)"),
    enable_realtime: bool = Form(True, description="실시간 상태 업데이트 사용")
):
    """
    8단계 AI 파이프라인 가상 피팅 실행
    
    실시간 진행 상황은 WebSocket (/api/ws/{client_id})을 통해 전송됩니다.
    """
    if not pipeline_manager or not pipeline_manager.is_initialized:
        raise HTTPException(
            status_code=503, 
            detail="AI 파이프라인이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요."
        )
    
    # 프로세스 ID 생성
    process_id = f"tryon_{uuid.uuid4().hex[:12]}"
    start_time = time.time()
    
    try:
        # 입력 파일 검증
        await validate_upload_files(person_image, clothing_image)
        
        # 이미지 로드
        person_pil = await load_image_from_upload(person_image)
        clothing_pil = await load_image_from_upload(clothing_image)
        
        # 실시간 상태 콜백 설정
        progress_callback = None
        if enable_realtime:
            progress_callback = create_progress_callback(process_id)
            
            # 프로세스 시작 알림
            await ws_manager.broadcast_to_process({
                "type": "process_started",
                "process_id": process_id,
                "message": "가상 피팅 처리를 시작합니다...",
                "timestamp": time.time()
            }, process_id)
        
        # 8단계 파이프라인 실행
        result = await pipeline_manager.process_virtual_tryon(
            person_image=person_pil,
            clothing_image=clothing_pil,
            height=height,
            weight=weight,
            progress_callback=progress_callback
        )
        
        # 성공 시 WebSocket으로 완료 알림
        if enable_realtime and result["success"]:
            await ws_manager.broadcast_to_process({
                "type": "process_completed",
                "process_id": process_id,
                "result": {
                    "processing_time": result["processing_time"],
                    "fit_score": result["fit_score"],
                    "quality_score": result["quality_score"]
                },
                "timestamp": time.time()
            }, process_id)
        
        # 응답 구성
        response = VirtualTryOnResponse(
            success=result["success"],
            process_id=process_id,
            fitted_image=result["fitted_image"],
            processing_time=result["processing_time"],
            confidence=result["confidence"],
            fit_score=result["fit_score"],
            quality_score=result["quality_score"],
            measurements=result["measurements"],
            recommendations=result["recommendations"],
            pipeline_stages=result.get("pipeline_stages", {}),
            debug_info=result.get("debug_info", {})
        )
        
        # 백그라운드에서 통계 업데이트
        background_tasks.add_task(update_processing_stats, result)
        
        return response
        
    except Exception as e:
        error_msg = f"가상 피팅 처리 실패: {str(e)}"
        logger.error(error_msg)
        
        # 실패 시 WebSocket으로 에러 알림
        if enable_realtime:
            await ws_manager.broadcast_to_process({
                "type": "process_error",
                "process_id": process_id,
                "error": error_msg,
                "timestamp": time.time()
            }, process_id)
        
        raise HTTPException(status_code=500, detail=error_msg)

async def validate_upload_files(person_image: UploadFile, clothing_image: UploadFile):
    """업로드된 파일 검증"""
    # 파일 크기 검증 (10MB 제한)
    max_size = 10 * 1024 * 1024  # 10MB
    
    if person_image.size and person_image.size > max_size:
        raise HTTPException(status_code=413, detail="사용자 이미지가 10MB를 초과합니다.")
    
    if clothing_image.size and clothing_image.size > max_size:
        raise HTTPException(status_code=413, detail="의류 이미지가 10MB를 초과합니다.")
    
    # 파일 형식 검증
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    
    if person_image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="사용자 이미지는 JPG, PNG, WebP 형식만 지원됩니다.")
    
    if clothing_image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="의류 이미지는 JPG, PNG, WebP 형식만 지원됩니다.")

async def load_image_from_upload(upload_file: UploadFile) -> Image.Image:
    """업로드 파일에서 PIL 이미지 로드"""
    try:
        # 파일 내용 읽기
        contents = await upload_file.read()
        
        # PIL 이미지로 변환
        image = Image.open(io.BytesIO(contents))
        
        # RGB로 변환 (필요한 경우)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {str(e)}")

async def update_processing_stats(result: Dict[str, Any]):
    """처리 통계 업데이트 (백그라운드 태스크)"""
    try:
        # 여기서 데이터베이스나 로그에 통계 저장
        logger.info(f"처리 완료 - 시간: {result['processing_time']:.2f}초, 품질: {result.get('quality_score', 0):.2f}")
    except Exception as e:
        logger.error(f"통계 업데이트 실패: {e}")

@router.get("/status", response_model=PipelineStatusResponse)
async def get_pipeline_status():
    """파이프라인 현재 상태 조회"""
    try:
        if not pipeline_manager:
            return PipelineStatusResponse(
                initialized=False,
                device="unknown",
                steps_loaded=0,
                total_steps=8,
                memory_status={},
                stats={}
            )
        
        status = await pipeline_manager.get_pipeline_status()
        
        return PipelineStatusResponse(
            initialized=status["initialized"],
            device=status["device"],
            steps_loaded=status["steps_loaded"],
            total_steps=status["total_steps"],
            memory_status=status["memory_status"],
            stats=status["stats"]
        )
        
    except Exception as e:
        logger.error(f"파이프라인 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/initialize")
async def initialize_pipeline():
    """파이프라인 수동 초기화"""
    global pipeline_manager
    
    try:
        if not pipeline_manager:
            pipeline_manager = get_pipeline_manager()
        
        if pipeline_manager.is_initialized:
            return {"message": "파이프라인이 이미 초기화되었습니다.", "initialized": True}
        
        logger.info("🔄 파이프라인 수동 초기화 시작...")
        success = await pipeline_manager.initialize()
        
        if success:
            logger.info("✅ 파이프라인 수동 초기화 완료")
            return {"message": "파이프라인 초기화 완료", "initialized": True}
        else:
            raise HTTPException(status_code=500, detail="파이프라인 초기화 실패")
            
    except Exception as e:
        logger.error(f"파이프라인 수동 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/warmup")
async def warmup_pipeline():
    """파이프라인 웜업 실행"""
    if not pipeline_manager or not pipeline_manager.is_initialized:
        raise HTTPException(status_code=503, detail="파이프라인이 초기화되지 않았습니다.")
    
    try:
        logger.info("🔥 파이프라인 웜업 시작...")
        success = await pipeline_manager.warmup()
        
        if success:
            return {"message": "파이프라인 웜업 완료", "success": True}
        else:
            return {"message": "파이프라인 웜업 부분 실패", "success": False}
            
    except Exception as e:
        logger.error(f"파이프라인 웜업 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/memory")
async def get_memory_status():
    """메모리 사용량 조회"""
    try:
        if gpu_config:
            memory_info = gpu_config.get_memory_info()
            return {
                "memory_info": memory_info,
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=503, detail="GPU 설정이 초기화되지 않았습니다.")
            
    except Exception as e:
        logger.error(f"메모리 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup")
async def cleanup_memory():
    """메모리 수동 정리"""
    try:
        if gpu_config:
            gpu_config.cleanup_memory()
            return {"message": "메모리 정리 완료", "timestamp": time.time()}
        else:
            raise HTTPException(status_code=503, detail="GPU 설정이 초기화되지 않았습니다.")
            
    except Exception as e:
        logger.error(f"메모리 정리 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/info")
async def get_models_info():
    """로드된 모델 정보 조회"""
    if not pipeline_manager:
        raise HTTPException(status_code=503, detail="파이프라인이 초기화되지 않았습니다.")
    
    try:
        models_info = {}
        
        for step_num, step in pipeline_manager.steps.items():
            if hasattr(step, 'get_model_info'):
                models_info[f"step_{step_num}"] = await step.get_model_info()
            else:
                models_info[f"step_{step_num}"] = {
                    "loaded": hasattr(step, 'model') and step.model is not None,
                    "initialized": getattr(step, 'is_initialized', False)
                }
        
        return {
            "models": models_info,
            "total_steps": len(pipeline_manager.steps),
            "device": pipeline_manager.device,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"모델 정보 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def pipeline_health_check():
    """파이프라인 헬스체크"""
    health_status = {
        "pipeline_manager": pipeline_manager is not None,
        "gpu_config": gpu_config is not None,
        "initialized": pipeline_manager.is_initialized if pipeline_manager else False,
        "device": pipeline_manager.device if pipeline_manager else "unknown",
        "timestamp": time.time()
    }
    
    # 메모리 상태 추가
    if gpu_config:
        try:
            memory_info = gpu_config.get_memory_info()
            health_status["memory"] = memory_info
        except:
            health_status["memory"] = {"error": "메모리 정보 조회 실패"}
    
    # 전체 상태 판정
    if health_status["pipeline_manager"] and health_status["initialized"]:
        health_status["status"] = "healthy"
        status_code = 200
    elif health_status["pipeline_manager"]:
        health_status["status"] = "initializing"
        status_code = 202
    else:
        health_status["status"] = "unhealthy"
        status_code = 503
    
    return JSONResponse(content=health_status, status_code=status_code)

# 실시간 처리 상태 테스트 엔드포인트
@router.post("/test/realtime/{process_id}")
async def test_realtime_updates(process_id: str):
    """실시간 업데이트 테스트"""
    try:
        progress_callback = create_progress_callback(process_id)
        
        # 8단계 시뮬레이션
        steps = [
            "인체 파싱 (20개 부위)",
            "포즈 추정 (18개 키포인트)",
            "의류 세그멘테이션 (배경 제거)",
            "기하학적 매칭 (TPS 변환)",
            "옷 워핑 (신체에 맞춰 변형)",
            "가상 피팅 생성 (HR-VITON/ACGPN)",
            "후처리 (품질 향상)",
            "품질 평가 (자동 스코어링)"
        ]
        
        for i, step_name in enumerate(steps, 1):
            await progress_callback(i, f"{step_name} 처리 중...", (i / 8) * 100)
            await asyncio.sleep(1)  # 1초 대기
        
        return {"message": "실시간 업데이트 테스트 완료", "process_id": process_id}
        
    except Exception as e:
        logger.error(f"실시간 테스트 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 개발용 디버그 엔드포인트
@router.get("/debug/config")
async def get_debug_config():
    """디버그용 설정 정보"""
    debug_info = {
        "pipeline_manager_initialized": pipeline_manager is not None and pipeline_manager.is_initialized,
        "gpu_config_initialized": gpu_config is not None,
        "websocket_connections": len(ws_manager.active_connections),
        "active_processes": len(ws_manager.process_connections)
    }
    
    if gpu_config:
        debug_info["gpu_settings"] = {
            "device": gpu_config.device_type,
            "is_apple_silicon": gpu_config.is_apple_silicon,
            "memory_settings": gpu_config.memory_settings,
            "optimization_settings": gpu_config.optimization_settings
        }
    
    return debug_info