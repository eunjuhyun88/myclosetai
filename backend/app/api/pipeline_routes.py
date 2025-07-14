"""
완전 수정된 8단계 AI 파이프라인 API 라우터
- WebSocket 실시간 상태 통합
- pipeline_manager 생성자 문제 완전 해결
- M3 Max 최적화
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
import numpy as np

# 완전 수정된 import 구조
try:
    from app.ai_pipeline.pipeline_manager import (
        get_pipeline_manager, 
        create_pipeline_manager,
        PipelineMode
    )
    from app.core.gpu_config import GPUConfig
    PIPELINE_IMPORT_SUCCESS = True
except ImportError as e:
    logging.warning(f"파이프라인 import 실패: {e}")
    PIPELINE_IMPORT_SUCCESS = False
    
    # 폴백 클래스들
    class PipelineMode:
        SIMULATION = "simulation"
        PRODUCTION = "production"
    
    def get_pipeline_manager():
        return None
    
    def create_pipeline_manager(*args, **kwargs):
        return None
    
    class GPUConfig:
        def __init__(self):
            self.device_type = "mps"
        def setup_memory_optimization(self):
            pass
        def get_memory_info(self):
            return {}
        def cleanup_memory(self):
            pass

# 스키마 import (안전하게)
try:
    from app.models.schemas import (
        VirtualTryOnRequest, 
        VirtualTryOnResponse,
        PipelineStatusResponse, 
        ProcessingStage
    )
    SCHEMAS_IMPORT_SUCCESS = True
except ImportError as e:
    logging.warning(f"스키마 import 실패: {e}")
    SCHEMAS_IMPORT_SUCCESS = False
    
    # 폴백 스키마 정의
    class VirtualTryOnResponse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class PipelineStatusResponse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

# WebSocket 라우터 import (안전하게)
try:
    from app.api.websocket_routes import create_progress_callback, manager as ws_manager
    WEBSOCKET_IMPORT_SUCCESS = True
except ImportError as e:
    logging.warning(f"WebSocket import 실패: {e}")
    WEBSOCKET_IMPORT_SUCCESS = False
    
    # 폴백 함수들
    def create_progress_callback(process_id):
        async def dummy_callback(stage, percentage):
            pass
        return dummy_callback
    
    class DummyWSManager:
        def __init__(self):
            self.active_connections = []
            self.process_connections = {}
        
        async def broadcast_to_process(self, message, process_id):
            pass
    
    ws_manager = DummyWSManager()

logger = logging.getLogger(__name__)
router = APIRouter()

# 전역 변수들
pipeline_manager = None
gpu_config = None

@router.on_event("startup")
async def startup_pipeline():
    """파이프라인 라우터 시작 시 초기화 - 완전 수정"""
    global pipeline_manager, gpu_config
    
    try:
        logger.info("🚀 파이프라인 라우터 초기화 시작...")
        
        # GPU 설정 초기화
        gpu_config = GPUConfig()
        gpu_config.setup_memory_optimization()
        logger.info("✅ GPU 설정 초기화 완료")
        
        # 파이프라인 매니저 초기화 - 모든 필수 인자 포함
        if PIPELINE_IMPORT_SUCCESS:
            # 먼저 기존 매니저가 있는지 확인
            existing_manager = get_pipeline_manager()
            if existing_manager is None:
                # 새로운 매니저 생성
                pipeline_manager = create_pipeline_manager(
                    mode=PipelineMode.PRODUCTION,
                    device="mps",
                    device_type="apple_silicon",
                    memory_gb=128.0,
                    is_m3_max=True,
                    optimization_enabled=True
                )
            else:
                # 기존 매니저 사용
                pipeline_manager = existing_manager
            
            # 백그라운드에서 모델 초기화
            asyncio.create_task(initialize_pipeline_models())
            logger.info("✅ 파이프라인 매니저 생성 완료")
        else:
            logger.warning("⚠️ 파이프라인 매니저 생성 실패 - 폴백 모드")
        
        logger.info("✅ 파이프라인 라우터 초기화 완료")
        
    except Exception as e:
        logger.error(f"❌ 파이프라인 라우터 초기화 실패: {e}")
        logger.error(f"📋 상세 오류: {str(e)}")

async def initialize_pipeline_models():
    """백그라운드에서 파이프라인 모델 초기화"""
    try:
        logger.info("🔄 백그라운드에서 AI 모델 초기화 시작...")
        
        if pipeline_manager:
            success = await pipeline_manager.initialize()
            if success:
                logger.info("✅ AI 모델 초기화 완료")
                # 웜업 실행
                warmup_success = await pipeline_manager.warmup()
                logger.info(f"🔥 웜업 {'완료' if warmup_success else '부분 실패'}")
            else:
                logger.error("❌ AI 모델 초기화 실패")
        else:
            logger.warning("⚠️ 파이프라인 매니저가 없어 초기화 건너뜀")
        
    except Exception as e:
        logger.error(f"❌ 백그라운드 모델 초기화 실패: {e}")

@router.post("/virtual-tryon")
async def virtual_tryon_endpoint(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(..., description="사용자 이미지"),
    clothing_image: UploadFile = File(..., description="의류 이미지"),
    height: float = Form(170.0, description="키 (cm)"),
    weight: float = Form(65.0, description="몸무게 (kg)"),
    enable_realtime: bool = Form(True, description="실시간 상태 업데이트 사용")
):
    """
    완전 수정된 8단계 AI 파이프라인 가상 피팅 실행
    
    실시간 진행 상황은 WebSocket (/api/ws/{client_id})을 통해 전송됩니다.
    """
    # 파이프라인 매니저 상태 확인
    if not pipeline_manager:
        raise HTTPException(
            status_code=503, 
            detail="AI 파이프라인 매니저가 초기화되지 않았습니다. 잠시 후 다시 시도해주세요."
        )
    
    if not pipeline_manager.is_initialized:
        # 자동 초기화 시도
        try:
            logger.info("🔄 파이프라인 자동 초기화 시도...")
            init_success = await pipeline_manager.initialize()
            if not init_success:
                raise HTTPException(
                    status_code=503,
                    detail="AI 파이프라인 초기화에 실패했습니다. 관리자에게 문의하세요."
                )
        except Exception as e:
            logger.error(f"파이프라인 자동 초기화 실패: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"AI 파이프라인 초기화 실패: {str(e)}"
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
        if enable_realtime and WEBSOCKET_IMPORT_SUCCESS:
            progress_callback = create_progress_callback(process_id)
            
            # 프로세스 시작 알림
            await ws_manager.broadcast_to_process({
                "type": "process_started",
                "process_id": process_id,
                "message": "가상 피팅 처리를 시작합니다...",
                "timestamp": time.time()
            }, process_id)
        
        # 8단계 파이프라인 실행 - 완전 수정된 메서드 호출
        result = await pipeline_manager.process_virtual_tryon(
            person_image=person_pil,
            clothing_image=clothing_pil,
            height=height,
            weight=weight,
            progress_callback=progress_callback
        )
        
        # 성공 시 WebSocket으로 완료 알림
        if enable_realtime and WEBSOCKET_IMPORT_SUCCESS and result["success"]:
            await ws_manager.broadcast_to_process({
                "type": "process_completed",
                "process_id": process_id,
                "result": {
                    "processing_time": result["processing_time"],
                    "fit_score": result.get("fit_score", 0.8),
                    "quality_score": result.get("quality_score", 0.8)
                },
                "timestamp": time.time()
            }, process_id)
        
        # 응답 구성 - 스키마 사용 여부에 따라 분기
        if SCHEMAS_IMPORT_SUCCESS:
            response = VirtualTryOnResponse(
                success=result["success"],
                process_id=process_id,
                fitted_image=result.get("fitted_image"),
                processing_time=result["processing_time"],
                confidence=result.get("confidence", 0.85),
                fit_score=result.get("fit_score", 0.8),
                quality_score=result.get("quality_score", 0.82),
                measurements=result.get("measurements", {}),
                recommendations=result.get("recommendations", []),
                pipeline_stages=result.get("pipeline_stages", {}),
                debug_info=result.get("debug_info", {})
            )
        else:
            # 딕셔너리 형태로 반환
            response = {
                "success": result["success"],
                "process_id": process_id,
                "fitted_image": result.get("fitted_image"),
                "processing_time": result["processing_time"],
                "confidence": result.get("confidence", 0.85),
                "fit_score": result.get("fit_score", 0.8),
                "quality_score": result.get("quality_score", 0.82),
                "measurements": result.get("measurements", {}),
                "recommendations": result.get("recommendations", []),
                "pipeline_stages": result.get("pipeline_stages", {}),
                "debug_info": result.get("debug_info", {})
            }
        
        # 백그라운드에서 통계 업데이트
        background_tasks.add_task(update_processing_stats, result)
        
        return response
        
    except Exception as e:
        error_msg = f"가상 피팅 처리 실패: {str(e)}"
        logger.error(error_msg)
        logger.error(f"📋 상세 오류: {str(e)}")
        
        # 실패 시 WebSocket으로 에러 알림
        if enable_realtime and WEBSOCKET_IMPORT_SUCCESS:
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
        processing_time = result.get('processing_time', 0)
        quality_score = result.get('quality_score', 0)
        logger.info(f"📊 처리 완료 - 시간: {processing_time:.2f}초, 품질: {quality_score:.2f}")
    except Exception as e:
        logger.error(f"통계 업데이트 실패: {e}")

@router.get("/status")
async def get_pipeline_status():
    """파이프라인 현재 상태 조회"""
    try:
        if not pipeline_manager:
            status_data = {
                "initialized": False,
                "device": "unknown",
                "steps_loaded": 0,
                "total_steps": 8,
                "memory_status": {},
                "stats": {"error": "파이프라인 매니저가 없습니다"}
            }
        else:
            status = await pipeline_manager.get_pipeline_status()
            status_data = {
                "initialized": status["initialized"],
                "device": status["device"],
                "device_type": status.get("device_type", "unknown"),
                "memory_gb": status.get("memory_gb", 0),
                "is_m3_max": status.get("is_m3_max", False),
                "optimization_enabled": status.get("optimization_enabled", False),
                "steps_loaded": status["steps_loaded"],
                "total_steps": status["total_steps"],
                "memory_status": status["memory_status"],
                "stats": status["stats"],
                "performance_metrics": status.get("performance_metrics", {}),
                "pipeline_config": status.get("pipeline_config", {})
            }
        
        if SCHEMAS_IMPORT_SUCCESS:
            return PipelineStatusResponse(**status_data)
        else:
            return status_data
        
    except Exception as e:
        logger.error(f"파이프라인 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/initialize")
async def initialize_pipeline():
    """파이프라인 수동 초기화"""
    global pipeline_manager
    
    try:
        if not pipeline_manager:
            if PIPELINE_IMPORT_SUCCESS:
                pipeline_manager = create_pipeline_manager(
                    mode=PipelineMode.PRODUCTION,
                    device="mps",
                    device_type="apple_silicon",
                    memory_gb=128.0,
                    is_m3_max=True,
                    optimization_enabled=True
                )
            else:
                raise HTTPException(status_code=503, detail="파이프라인 모듈을 import할 수 없습니다")
        
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
            return {
                "memory_info": {"error": "GPU 설정이 초기화되지 않았습니다"},
                "timestamp": time.time()
            }
            
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
            return {"message": "GPU 설정이 없어 정리 생략", "timestamp": time.time()}
            
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
        
        # 파이프라인 단계들 정보 수집
        for step_name in pipeline_manager.step_order:
            if step_name in pipeline_manager.steps:
                step = pipeline_manager.steps[step_name]
                if hasattr(step, 'get_model_info'):
                    models_info[step_name] = await step.get_model_info()
                else:
                    models_info[step_name] = {
                        "loaded": hasattr(step, 'model') and step.model is not None,
                        "initialized": getattr(step, 'is_initialized', False),
                        "type": type(step).__name__
                    }
            else:
                models_info[step_name] = {
                    "loaded": False,
                    "initialized": False,
                    "type": "None"
                }
        
        return {
            "models": models_info,
            "total_steps": len(pipeline_manager.step_order),
            "loaded_steps": len(pipeline_manager.steps),
            "device": pipeline_manager.device,
            "device_type": pipeline_manager.device_type,
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
        "imports": {
            "pipeline": PIPELINE_IMPORT_SUCCESS,
            "schemas": SCHEMAS_IMPORT_SUCCESS,
            "websocket": WEBSOCKET_IMPORT_SUCCESS
        },
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
    if not WEBSOCKET_IMPORT_SUCCESS:
        return {"message": "WebSocket 기능이 비활성화되어 있습니다", "process_id": process_id}
    
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
            await progress_callback(f"{step_name} 처리 중...", (i / 8) * 100)
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
        "imports": {
            "pipeline": PIPELINE_IMPORT_SUCCESS,
            "schemas": SCHEMAS_IMPORT_SUCCESS,
            "websocket": WEBSOCKET_IMPORT_SUCCESS
        },
        "pipeline_manager": {
            "exists": pipeline_manager is not None,
            "initialized": pipeline_manager.is_initialized if pipeline_manager else False,
            "device": pipeline_manager.device if pipeline_manager else "unknown",
            "device_type": pipeline_manager.device_type if pipeline_manager else "unknown",
            "memory_gb": pipeline_manager.memory_gb if pipeline_manager else 0,
            "is_m3_max": pipeline_manager.is_m3_max if pipeline_manager else False,
            "optimization_enabled": pipeline_manager.optimization_enabled if pipeline_manager else False
        },
        "websocket_connections": len(ws_manager.active_connections) if WEBSOCKET_IMPORT_SUCCESS else 0,
        "active_processes": len(ws_manager.process_connections) if WEBSOCKET_IMPORT_SUCCESS else 0
    }
    
    if gpu_config:
        debug_info["gpu_settings"] = {
            "device_type": gpu_config.device_type,
            "initialized": True
        }
    else:
        debug_info["gpu_settings"] = {
            "device_type": "unknown",
            "initialized": False
        }
    
    return debug_info

# 개발용 파이프라인 재시작 엔드포인트
@router.post("/dev/restart")
async def restart_pipeline():
    """개발용 파이프라인 재시작"""
    global pipeline_manager
    
    try:
        # 기존 파이프라인 정리
        if pipeline_manager:
            await pipeline_manager.cleanup()
        
        # 새로운 파이프라인 생성
        if PIPELINE_IMPORT_SUCCESS:
            pipeline_manager = create_pipeline_manager(
                mode=PipelineMode.PRODUCTION,
                device="mps",
                device_type="apple_silicon",
                memory_gb=128.0,
                is_m3_max=True,
                optimization_enabled=True
            )
            
            # 초기화
            success = await pipeline_manager.initialize()
            
            return {
                "message": "파이프라인 재시작 완료",
                "success": success,
                "initialized": pipeline_manager.is_initialized
            }
        else:
            return {
                "message": "파이프라인 모듈 import 실패로 재시작 불가",
                "success": False
            }
            
    except Exception as e:
        logger.error(f"파이프라인 재시작 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 완전 수정된 라우터 종료 이벤트
@router.on_event("shutdown")
async def shutdown_pipeline():
    """파이프라인 라우터 종료 시 정리"""
    global pipeline_manager, gpu_config
    
    try:
        logger.info("🛑 파이프라인 라우터 종료 중...")
        
        if pipeline_manager:
            await pipeline_manager.cleanup()
            logger.info("✅ 파이프라인 매니저 정리 완료")
        
        if gpu_config:
            gpu_config.cleanup_memory()
            logger.info("✅ GPU 설정 정리 완료")
        
        logger.info("✅ 파이프라인 라우터 종료 완료")
        
    except Exception as e:
        logger.error(f"❌ 파이프라인 라우터 종료 중 오류: {e}")

# 라우터 정보 출력
logger.info("📡 완전 수정된 파이프라인 API 라우터 로드 완료")
logger.info(f"🔧 Pipeline Import: {'✅' if PIPELINE_IMPORT_SUCCESS else '❌'}")
logger.info(f"📋 Schemas Import: {'✅' if SCHEMAS_IMPORT_SUCCESS else '❌'}")
logger.info(f"🌐 WebSocket Import: {'✅' if WEBSOCKET_IMPORT_SUCCESS else '❌'}")