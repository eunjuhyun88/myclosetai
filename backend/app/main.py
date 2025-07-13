# app/main.py
"""
MyCloset AI 가상 피팅 시스템 메인 애플리케이션
개선된 PipelineManager와 완전히 통합된 버전
"""

import sys
import os
import asyncio
import logging
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
import uvicorn

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 실제 존재하는 모듈들 import
try:
    from app.api.unified_routes import router as unified_router
    from app.core.config import get_settings
    from app.core.logging_config import setup_logging
    from app.ai_pipeline.pipeline_manager import PipelineManager
except ImportError as e:
    # 개발 환경에서 모듈이 없을 때 대체
    print(f"Warning: {e}")
    unified_router = None
    
    class MockSettings:
        def __init__(self):
            self.debug = True
            self.cors_origins = ["*"]
    
    def get_settings():
        return MockSettings()
    
    def setup_logging():
        logging.basicConfig(level=logging.INFO)

# 설정 로드
settings = get_settings()

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)

# 글로벌 변수
pipeline_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 수명주기 관리"""
    
    # 시작 시 실행
    logger.info("🚀 MyCloset AI 시스템 시작 중...")
    
    try:
        # AI 파이프라인 초기화
        global pipeline_manager
        pipeline_manager = PipelineManager(
            config_path='config/pipeline_config.json',  # 설정 파일 경로
            device='auto'  # 최적 디바이스 자동 선택
        )
        
        # 파이프라인 초기화 (비동기)
        logger.info("🤖 AI 파이프라인 초기화 중...")
        success = await pipeline_manager.initialize()
        
        if success:
            logger.info("✅ AI 파이프라인 초기화 완료")
            
            # 파이프라인 상태 출력
            status = await pipeline_manager.get_pipeline_status()
            logger.info(f"📊 파이프라인 상태: {status['steps_status']}")
        else:
            logger.warning("⚠️ AI 파이프라인 부분 초기화 (일부 모델 누락)")
        
        # 필요한 디렉토리 생성
        os.makedirs("app/static/uploads", exist_ok=True)
        os.makedirs("app/static/results", exist_ok=True)
        os.makedirs("app/logs", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        os.makedirs("test_images", exist_ok=True)
        
        logger.info("📁 필요한 디렉토리 생성 완료")
        
        # GPU 정보 출력
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"🔥 CUDA 사용 가능: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("🍎 Apple MPS 사용 가능")
            else:
                logger.info("💻 CPU 모드로 실행")
        except ImportError:
            logger.info("💻 PyTorch 없음 - CPU 모드로 실행")
        
        logger.info("🎉 MyCloset AI 시스템 시작 완료!")
        
    except Exception as e:
        logger.error(f"❌ 시스템 초기화 실패: {e}")
        logger.error(f"📋 상세 오류: {traceback.format_exc()}")
    
    yield  # 애플리케이션 실행
    
    # 종료 시 실행
    logger.info("🛑 MyCloset AI 시스템 종료 중...")
    
    try:
        if pipeline_manager:
            await pipeline_manager.cleanup()
            logger.info("🧹 AI 파이프라인 정리 완료")
            
        # 추가 정리 작업
        logger.info("✅ 시스템 종료 완료")
        
    except Exception as e:
        logger.error(f"❌ 종료 중 오류: {e}")

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="MyCloset AI Virtual Try-On",
    description="AI 기반 8단계 가상 피팅 시스템",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if hasattr(settings, 'cors_origins') else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# 정적 파일 서빙
try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    app.mount("/output", StaticFiles(directory="output"), name="output")
except Exception as e:
    logger.warning(f"정적 파일 마운트 실패: {e}")

# API 라우터 등록
if unified_router:
    app.include_router(unified_router, prefix="", tags=["virtual-tryon"])
else:
    logger.warning("API 라우터를 로드할 수 없습니다. 기본 엔드포인트만 제공됩니다.")

# 기본 엔드포인트들
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "MyCloset AI Virtual Try-On System",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "8단계 AI 가상 피팅",
            "실시간 품질 분석",
            "자동 오류 복구",
            "M3 Max MPS 최적화"
        ],
        "docs": "/docs",
        "api_base": "/api"
    }

@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트"""
    try:
        pipeline_status = "unknown"
        device_info = "unknown"
        memory_usage = {}
        
        if pipeline_manager:
            if hasattr(pipeline_manager, 'is_initialized'):
                pipeline_status = "initialized" if pipeline_manager.is_initialized else "not_initialized"
                device_info = getattr(pipeline_manager, 'device', 'unknown')
                
                # 상세 상태 정보
                if pipeline_manager.is_initialized:
                    try:
                        status = await pipeline_manager.get_pipeline_status()
                        memory_usage = status.get('memory_usage', {})
                    except:
                        pass
            else:
                pipeline_status = "available"
        else:
            pipeline_status = "not_loaded"
        
        health_status = {
            "status": "healthy",
            "pipeline_status": pipeline_status,
            "device": device_info,
            "memory_usage": memory_usage,
            "debug_mode": getattr(settings, 'debug', False),
            "timestamp": str(asyncio.get_event_loop().time())
        }
        
        # 파이프라인이 초기화되지 않았으면 warning 상태
        if pipeline_status != "initialized":
            health_status["status"] = "warning"
            health_status["message"] = "Pipeline not fully initialized"
        
        return health_status
        
    except Exception as e:
        logger.error(f"헬스체크 오류: {e}")
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": str(asyncio.get_event_loop().time())
        }, status_code=500)

@app.get("/api/system-info")
async def system_info():
    """시스템 정보 엔드포인트"""
    try:
        import torch
        import cv2
        import numpy as np
        
        info = {
            "system": {
                "python_version": sys.version,
                "platform": sys.platform,
                "architecture": os.uname().machine if hasattr(os, 'uname') else "unknown"
            },
            "dependencies": {
                "pytorch_version": torch.__version__ if 'torch' in sys.modules else "Not installed",
                "opencv_version": cv2.__version__ if 'cv2' in sys.modules else "Not installed",
                "numpy_version": np.__version__ if 'np' in sys.modules else "Not installed"
            },
            "hardware": {
                "gpu_available": False,
                "gpu_info": "None",
                "cpu_count": os.cpu_count()
            },
            "pipeline": {
                "initialized": False,
                "device": "unknown",
                "steps_loaded": 0
            }
        }
        
        # GPU 정보
        if torch.cuda.is_available():
            info["hardware"]["gpu_available"] = True
            info["hardware"]["gpu_info"] = f"CUDA: {torch.cuda.get_device_name()}"
            info["hardware"]["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info["hardware"]["gpu_available"] = True
            info["hardware"]["gpu_info"] = "Apple MPS"
        
        # 파이프라인 정보
        if pipeline_manager and hasattr(pipeline_manager, 'is_initialized'):
            info["pipeline"]["initialized"] = pipeline_manager.is_initialized
            info["pipeline"]["device"] = getattr(pipeline_manager, 'device', 'unknown')
            
            if pipeline_manager.is_initialized:
                try:
                    status = await pipeline_manager.get_pipeline_status()
                    info["pipeline"]["steps_loaded"] = len(status.get('steps_status', {}))
                    info["pipeline"]["performance_metrics"] = status.get('performance_metrics', {})
                except:
                    pass
        
        return info
        
    except Exception as e:
        return {"error": f"시스템 정보 수집 실패: {e}"}

@app.get("/api/pipeline-status")
async def pipeline_status():
    """파이프라인 상태 상세 조회"""
    try:
        if not pipeline_manager:
            return {"error": "Pipeline manager not loaded"}
        
        if not hasattr(pipeline_manager, 'get_pipeline_status'):
            return {"error": "Pipeline status method not available"}
        
        status = await pipeline_manager.get_pipeline_status()
        return status
        
    except Exception as e:
        logger.error(f"파이프라인 상태 조회 오류: {e}")
        return {"error": str(e)}

@app.get("/api/performance-report")
async def performance_report():
    """성능 리포트 조회"""
    try:
        if not pipeline_manager or not hasattr(pipeline_manager, 'get_performance_report'):
            return {"error": "Performance report not available"}
        
        report = await pipeline_manager.get_performance_report()
        return report
        
    except Exception as e:
        logger.error(f"성능 리포트 조회 오류: {e}")
        return {"error": str(e)}

# 개선된 가상 피팅 엔드포인트
@app.post("/api/virtual-fitting")
async def virtual_fitting_endpoint(
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...),
    height: float = Form(170),
    weight: float = Form(65),
    chest: Optional[float] = Form(None),
    waist: Optional[float] = Form(None),
    hip: Optional[float] = Form(None),
    clothing_type: str = Form("shirt"),
    fabric_type: str = Form("cotton"),
    quality_target: float = Form(0.8),
    save_intermediate: bool = Form(False)
):
    """
    개선된 8단계 가상 피팅 엔드포인트
    """
    try:
        if not pipeline_manager or not pipeline_manager.is_initialized:
            raise HTTPException(500, "AI 파이프라인이 초기화되지 않았습니다.")
        
        # 이미지 파일 읽기
        person_image_data = await person_image.read()
        clothing_image_data = await clothing_image.read()
        
        # PIL 이미지로 변환
        from PIL import Image
        import io
        
        person_pil = Image.open(io.BytesIO(person_image_data)).convert('RGB')
        clothing_pil = Image.open(io.BytesIO(clothing_image_data)).convert('RGB')
        
        # 신체 치수 구성
        body_measurements = {
            'height': height,
            'weight': weight
        }
        if chest is not None:
            body_measurements['chest'] = chest
        if waist is not None:
            body_measurements['waist'] = waist
        if hip is not None:
            body_measurements['hip'] = hip
        
        # 진행률 콜백 (선택적)
        progress_updates = []
        
        async def progress_callback(stage: str, percentage: int):
            progress_updates.append({"stage": stage, "percentage": percentage})
            logger.info(f"🔄 진행률: {stage} - {percentage}%")
        
        # 개선된 가상 피팅 실행
        logger.info(f"🎯 가상 피팅 시작 - 의류: {clothing_type}, 재질: {fabric_type}")
        
        result = await pipeline_manager.process_complete_virtual_fitting(
            person_image=person_pil,
            clothing_image=clothing_pil,
            body_measurements=body_measurements,
            clothing_type=clothing_type,
            fabric_type=fabric_type,
            quality_target=quality_target,
            progress_callback=progress_callback,
            save_intermediate=save_intermediate,
            enable_auto_retry=True
        )
        
        if result['success']:
            # 결과 이미지 저장
            import uuid
            result_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
            result_path = f"output/{result_filename}"
            
            result['result_image'].save(result_path, quality=95)
            
            # 응답 구성
            response = {
                "success": True,
                "result_image_url": f"/output/{result_filename}",
                "final_quality_score": result['final_quality_score'],
                "quality_grade": result['quality_grade'],
                "processing_time": result['total_processing_time'],
                "quality_target_achieved": result['quality_target_achieved'],
                "fit_analysis": result['fit_analysis'],
                "improvement_suggestions": result['improvement_suggestions'],
                "step_results_summary": result['step_results_summary'],
                "processing_statistics": result['processing_statistics'],
                "progress_updates": progress_updates,
                "metadata": result['metadata']
            }
            
            # 중간 결과 포함 (요청된 경우)
            if save_intermediate and 'intermediate_results' in result:
                response['intermediate_results'] = result['intermediate_results']
            
            logger.info(f"✅ 가상 피팅 성공 - 품질: {result['final_quality_score']:.3f}")
            
            return response
        else:
            # 실패 응답
            logger.error(f"❌ 가상 피팅 실패: {result.get('error', 'Unknown error')}")
            
            return {
                "success": False,
                "error": result.get('error', 'Unknown error'),
                "error_type": result.get('error_type', 'processing_failure'),
                "processing_time": result.get('processing_time', 0),
                "fallback_used": result.get('fallback_used', False),
                "progress_updates": progress_updates
            }
        
    except Exception as e:
        logger.error(f"❌ 가상 피팅 엔드포인트 오류: {e}")
        logger.error(f"📋 상세: {traceback.format_exc()}")
        
        raise HTTPException(500, f"처리 중 오류 발생: {str(e)}")

# 기존 호환성을 위한 기본 가상 피팅 엔드포인트
@app.post("/api/virtual-tryon-basic")
async def virtual_tryon_basic(
    person_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...),
    height: float = Form(170),
    weight: float = Form(65)
):
    """
    기본 가상 피팅 엔드포인트 (기존 호환성)
    """
    try:
        # 이미지 파일 읽기
        person_image_data = await person_image.read()
        clothing_image_data = await clothing_image.read()
        
        # 파이프라인 매니저가 있으면 기존 메소드 사용
        if pipeline_manager and hasattr(pipeline_manager, 'process_virtual_tryon'):
            result = await pipeline_manager.process_virtual_tryon(
                person_image=person_image_data,
                clothing_image=clothing_image_data,
                height=height,
                weight=weight
            )
            
            if result['success']:
                # 결과 이미지를 base64로 인코딩
                import base64
                import io
                from PIL import Image
                
                fitted_image = result['fitted_image']
                if isinstance(fitted_image, str):
                    # 이미 base64인 경우
                    img_str = fitted_image
                else:
                    # numpy 배열이나 기타 형식인 경우
                    if hasattr(fitted_image, 'shape'):
                        # numpy 배열
                        img = Image.fromarray(fitted_image.astype('uint8'))
                    else:
                        # PIL 이미지
                        img = fitted_image
                    
                    buffered = io.BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                
                return {
                    "success": True,
                    "fitted_image": img_str,
                    "processing_time": result.get('processing_time', 0),
                    "confidence": result.get('confidence', 0.8),
                    "fit_score": result.get('fit_score', 0.8),
                    "recommendations": result.get('recommendations', []),
                    "pipeline_results": result.get('pipeline_results', {}),
                    "measurements": {
                        "chest": 95.0,
                        "waist": 80.0,
                        "hip": 98.0,
                        "bmi": round(weight / ((height/100) ** 2), 1)
                    }
                }
            else:
                return {
                    "success": False,
                    "error": result.get('error', 'Processing failed'),
                    "processing_time": result.get('processing_time', 0)
                }
        
        # 파이프라인이 없는 경우 더미 응답
        else:
            import base64
            import io
            from PIL import Image
            
            # 이미지 처리 (기본)
            person_img = Image.open(io.BytesIO(person_image_data))
            
            # 더미 결과
            buffered = io.BytesIO()
            person_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return {
                "success": True,
                "fitted_image": img_str,
                "processing_time": 1.0,
                "confidence": 0.7,
                "measurements": {
                    "chest": 95.0,
                    "waist": 80.0,
                    "hip": 98.0,
                    "bmi": round(weight / ((height/100) ** 2), 1)
                },
                "fit_score": 0.75,
                "recommendations": [
                    "이것은 데모 모드입니다.",
                    "실제 AI 모델을 설치하면 정확한 결과를 얻을 수 있습니다."
                ],
                "note": "Demo mode - AI models not loaded"
            }
        
    except Exception as e:
        logger.error(f"기본 가상 피팅 오류: {e}")
        raise HTTPException(500, f"처리 중 오류: {e}")

# 파일 다운로드 엔드포인트
@app.get("/api/download-result/{filename}")
async def download_result(filename: str):
    """결과 파일 다운로드"""
    try:
        file_path = f"output/{filename}"
        
        if not os.path.exists(file_path):
            raise HTTPException(404, "파일을 찾을 수 없습니다.")
        
        return FileResponse(
            file_path,
            media_type='image/jpeg',
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"파일 다운로드 오류: {e}")
        raise HTTPException(500, f"다운로드 중 오류: {e}")

# 예외 처리
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 예외 처리"""
    logger.error(f"HTTP 예외: {exc.status_code} - {exc.detail}")
    return JSONResponse({
        "success": False,
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": str(asyncio.get_event_loop().time())
    }, status_code=exc.status_code)

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """일반 예외 처리"""
    logger.error(f"예상치 못한 오류: {exc}")
    logger.error(f"📋 상세: {traceback.format_exc()}")
    
    return JSONResponse({
        "success": False,
        "error": "내부 서버 오류가 발생했습니다.",
        "detail": str(exc) if getattr(settings, 'debug', False) else None,
        "timestamp": str(asyncio.get_event_loop().time())
    }, status_code=500)

if __name__ == "__main__":
    # 개발 서버 실행
    logger.info("🚀 개발 서버 시작...")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )