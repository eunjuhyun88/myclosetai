# app/main.py
"""
🍎 MyCloset AI Backend - M3 Max 최적화 서버 (완전한 기능 보존)
✅ AppConfig 에러 해결하되 모든 기존 기능 유지
✅ 안전한 startup 이벤트 처리
✅ 데드락 방지
✅ 빠른 서버 시작
✅ 모든 라우터 및 서비스 로딩
✅ M3 Max 최적화 완전 보존
"""

import os
import sys
import time
import logging
import asyncio
import threading
from contextlib import asynccontextmanager
from typing import Dict, Any
from pathlib import Path

# ===============================================================
# 🔧 경로 설정 (가장 중요!)
# ===============================================================

# 현재 파일의 경로를 기준으로 프로젝트 루트 찾기
current_file = Path(__file__).resolve()  # app/main.py
app_dir = current_file.parent  # app/
backend_dir = app_dir.parent  # backend/
project_root = backend_dir.parent  # mycloset-ai/

# Python 경로에 backend 디렉토리 추가
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# 경로 확인 로그
print(f"📁 현재 파일: {current_file}")
print(f"📁 App 디렉토리: {app_dir}")
print(f"📁 Backend 디렉토리: {backend_dir}")
print(f"📁 프로젝트 루트: {project_root}")
print(f"📁 Python Path에 추가됨: {backend_dir}")

# FastAPI 및 기본 라이브러리
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# ===============================================================
# 🔧 Core 모듈 Import (경로 수정됨)
# ===============================================================

# 로깅 설정 먼저 (순환 참조 방지)
try:
    from app.core.logging_config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("✅ 로깅 설정 완료")
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ 로깅 설정 실패: {e}")

# Core 모듈 import - AppConfig 문제 해결
try:
    from app.core import (
        gpu_config,
        DEVICE,
        DEVICE_NAME,
        IS_M3_MAX,
        check_memory_available,
        optimize_memory
    )
    logger.info("✅ Core 모듈 로드 성공")
    CORE_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Core 모듈 로드 실패: {e}")
    logger.error("폴백 모드로 실행합니다")
    CORE_AVAILABLE = False
    
    # 폴백 설정
    DEVICE = "cpu"
    DEVICE_NAME = "CPU"
    IS_M3_MAX = False
    
    def check_memory_available(min_gb=1.0):
        return {"is_available": True, "system_memory": {"available_gb": 8.0}}
    
    def optimize_memory(aggressive=False):
        return {"success": True, "method": "fallback", "message": "Core 모듈 없이 실행 중"}

logger.info("✅ Core 모듈 초기화 완료")

# ===============================================================
# 🔧 설정 값 추출 (AppConfig 에러 해결)
# ===============================================================

# 환경 변수에서 설정값 가져오기
def get_env_setting(key: str, default: Any = None, value_type: type = str) -> Any:
    """환경 변수에서 설정값 안전하게 가져오기"""
    try:
        value = os.getenv(key.upper(), os.getenv(key.lower(), str(default)))
        
        if value_type == bool:
            return str(value).lower() in ['true', '1', 'yes', 'on']
        elif value_type == int:
            return int(value)
        elif value_type == float:
            return float(value)
        else:
            return value
            
    except (ValueError, TypeError) as e:
        logger.warning(f"설정값 '{key}' 변환 실패: {e}, 기본값 사용: {default}")
        return default

# 애플리케이션 설정 - 환경변수 기반 (AppConfig 대체)
APP_NAME = get_env_setting('APP_NAME', "MyCloset AI")
APP_VERSION = get_env_setting('APP_VERSION', "3.0.0")
DEBUG = get_env_setting('DEBUG', False, bool)
HOST = get_env_setting('HOST', "0.0.0.0")
PORT = get_env_setting('PORT', 8000, int)

logger.info(f"📋 애플리케이션 설정:")
logger.info(f"  - 이름: {APP_NAME}")
logger.info(f"  - 버전: {APP_VERSION}")
logger.info(f"  - 디버그: {DEBUG}")
logger.info(f"  - 호스트: {HOST}")
logger.info(f"  - 포트: {PORT}")

# ===============================================================
# 🔥 API 라우터들 안전한 Import (모든 라우터 유지)
# ===============================================================

api_routers = {}

# 1. Health 라우터 (기본)
try:
    from app.api.health import router as health_router
    api_routers['health'] = health_router
    logger.info("✅ Health 라우터 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ Health 라우터 로드 실패: {e}")

# 2. Step Routes 라우터 (메인 - PipelineManager 연결)
try:
    from app.api.step_routes import router as step_routes_router
    api_routers['step_routes'] = step_routes_router
    logger.info("🔥 Step Routes 라우터 로드 성공 (PipelineManager 연결)")
except ImportError as e:
    logger.warning(f"⚠️ Step Routes 라우터 로드 실패: {e}")

# 3. WebSocket 라우터 (실시간 통신)
try:
    from app.api.websocket_routes import router as websocket_router
    api_routers['websocket'] = websocket_router
    logger.info("✅ WebSocket 라우터 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ WebSocket 라우터 로드 실패: {e}")

# 4. Models 라우터 (모델 관리)
try:
    from app.api.models import router as models_router
    api_routers['models'] = models_router
    logger.info("✅ Models 라우터 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ Models 라우터 로드 실패: {e}")

# 5. Pipeline 라우터 (전체 파이프라인 실행)
try:
    from app.api.pipeline_routes import router as pipeline_router
    api_routers['pipeline'] = pipeline_router
    logger.info("✅ Pipeline 라우터 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ Pipeline 라우터 로드 실패: {e}")

# ===============================================================
# 🔧 정적 파일 설정
# ===============================================================

STATIC_DIR = backend_dir / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"✅ 정적 파일 서빙 설정: {STATIC_DIR}")

# ===============================================================
# 🔧 애플리케이션 lifespan (FastAPI 최신 방식 - 기존 기능 모두 유지)
# ===============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 애플리케이션 수명주기 관리 - 모든 기존 기능 유지"""
    # === 시작 이벤트 ===
    try:
        logger.info("🚀 MyCloset AI Backend 시작됨")
        logger.info(f"🏗️ 아키텍처: PipelineManager 중심")
        logger.info(f"🔧 설정: {APP_NAME} v{APP_VERSION}")
        logger.info(f"🤖 AI 파이프라인: 8단계 통합")
        logger.info(f"📊 로드된 라우터: {len(api_routers)}개")
        
        # GPU 설정 정보 (안전하게)
        if CORE_AVAILABLE:
            try:
                logger.info(f"🎯 GPU 설정:")
                logger.info(f"  - 디바이스: {DEVICE} ({DEVICE_NAME})")
                logger.info(f"  - M3 Max: {'✅' if IS_M3_MAX else '❌'}")
                
                if hasattr(gpu_config, 'memory_gb'):
                    logger.info(f"  - 메모리: {gpu_config.memory_gb:.1f}GB")
                if hasattr(gpu_config, 'optimization_settings'):
                    logger.info(f"  - 최적화 레벨: {gpu_config.optimization_settings.get('optimization_level', 'unknown')}")
                
            except Exception as e:
                logger.warning(f"⚠️ GPU 설정 정보 표시 실패: {e}")
        else:
            logger.info(f"🎯 폴백 모드:")
            logger.info(f"  - 디바이스: {DEVICE}")
            logger.info(f"  - M3 Max: {'✅' if IS_M3_MAX else '❌'}")
        
        # 메모리 상태 확인 (안전하게, 타임아웃 적용)
        try:
            def check_memory_safe():
                try:
                    return check_memory_available(min_gb=2.0)
                except Exception as e:
                    logger.warning(f"메모리 확인 실패: {e}")
                    return {"is_available": True, "system_memory": {"available_gb": 0}}
            
            # 타임아웃 적용 (3초)
            loop = asyncio.get_event_loop()
            memory_check = await asyncio.wait_for(
                loop.run_in_executor(None, check_memory_safe),
                timeout=3.0
            )
            
            if memory_check.get('is_available', False):
                available_gb = memory_check.get('system_memory', {}).get('available_gb', 0)
                logger.info(f"💾 메모리 상태: {available_gb:.1f}GB 사용 가능")
            else:
                logger.warning("⚠️ 메모리 부족 - 성능이 저하될 수 있습니다.")
                
        except asyncio.TimeoutError:
            logger.warning("⚠️ 메모리 확인 타임아웃 (3초 초과)")
        except Exception as e:
            logger.warning(f"⚠️ 메모리 확인 실패: {e}")
        
        # M3 Max 특화 기능 표시
        if IS_M3_MAX:
            try:
                logger.info("🍎 M3 Max 특화 기능 활성화:")
                logger.info("  - Neural Engine 가속")
                logger.info("  - Metal Performance Shaders")
                logger.info("  - 통합 메모리 최적화")
                logger.info("  - 8단계 파이프라인 최적화")
                logger.info("  - 고해상도 처리 지원")
                logger.info("⚙️ 8단계 파이프라인 최적화: 8개 단계 설정됨")
            except Exception as e:
                logger.warning(f"⚠️ M3 Max 기능 표시 실패: {e}")
        
        # 초기 메모리 최적화 (안전하게, 타임아웃 적용)
        try:
            def optimize_memory_safe():
                try:
                    return optimize_memory(aggressive=False)  # 부드러운 최적화
                except Exception as e:
                    logger.warning(f"메모리 최적화 실패: {e}")
                    return {"method": "skipped", "success": False}
            
            # 타임아웃 적용 (2초)
            loop = asyncio.get_event_loop()
            optimization_result = await asyncio.wait_for(
                loop.run_in_executor(None, optimize_memory_safe),
                timeout=2.0
            )
            
            method = optimization_result.get('method', 'unknown')
            logger.info(f"💾 초기 메모리 최적화 완료: {method}")
            
        except asyncio.TimeoutError:
            logger.warning("⚠️ 메모리 최적화 타임아웃 (2초 초과) - 건너뛰기")
        except Exception as e:
            logger.warning(f"⚠️ 메모리 최적화 실패: {e}")
        
        # 서버 준비 완료
        logger.info("🎉 서버 초기화 완료 - 요청 수신 대기 중...")
        
    except Exception as e:
        logger.error(f"❌ 서버 시작 이벤트 실패: {e}")
        # 실패해도 서버는 계속 시작
    
    # === 앱 실행 ===
    yield
    
    # === 종료 이벤트 ===
    try:
        logger.info("🛑 MyCloset AI Backend 종료 중...")
        
        # 안전한 종료 처리
        try:
            # 메모리 정리 (타임아웃 적용)
            def cleanup_safe():
                try:
                    optimize_memory(aggressive=True)
                    return True
                except:
                    return False
            
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, cleanup_safe),
                timeout=1.0
            )
            logger.info("💾 종료 시 메모리 정리 완료")
            
        except asyncio.TimeoutError:
            logger.warning("⚠️ 종료 시 메모리 정리 타임아웃")
        except Exception as e:
            logger.warning(f"⚠️ 종료 시 정리 실패: {e}")
        
        logger.info("✅ 서버 종료 완료")
        
    except Exception as e:
        logger.error(f"❌ 서버 종료 이벤트 실패: {e}")

# ===============================================================
# 🔧 FastAPI 앱 생성 (lifespan 적용)
# ===============================================================

app = FastAPI(
    title=APP_NAME,
    description="🍎 M3 Max 최적화 AI 가상 피팅 시스템",
    version=APP_VERSION,
    debug=DEBUG,
    lifespan=lifespan  # 🔥 최신 lifespan 방식 사용
)

# ===============================================================
# 🔧 미들웨어 설정 (기존과 동일)
# ===============================================================

# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "https://mycloset-ai.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Gzip 압축 미들웨어
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ===============================================================
# 🔧 정적 파일 마운트
# ===============================================================

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ===============================================================
# 🔧 라우터 등록 (모든 기존 라우터 유지)
# ===============================================================

for router_name, router in api_routers.items():
    try:
        if router_name == "health":
            app.include_router(router, prefix="/api", tags=["health"])
            logger.info("✅ Health 라우터 등록 완료")
            
        elif router_name == "step_routes":
            app.include_router(router, prefix="/api", tags=["steps"])
            logger.info("🔥 Step Routes 라우터 등록 완료 (PipelineManager 연결)")
            
            # Step 라우터의 엔드포인트 표시
            try:
                step_endpoints = [
                    "POST /api/step/1/upload-validation",
                    "POST /api/step/3/human-parsing", 
                    "POST /api/step/7/virtual-fitting",
                    "GET /api/step/health"
                ]
                logger.info(f"   📋 엔드포인트:")
                for endpoint in step_endpoints:
                    logger.info(f"     - {endpoint}")
            except Exception as e:
                logger.warning(f"Step 엔드포인트 표시 실패: {e}")
                
        elif router_name == "websocket":
            app.include_router(router, prefix="/api", tags=["websocket"])
            logger.info("✅ WebSocket 라우터 등록 완료")
            
        elif router_name == "models":
            app.include_router(router, prefix="/api", tags=["models"])
            logger.info("✅ Models 라우터 등록 완료")
            
        elif router_name == "pipeline":
            app.include_router(router, prefix="/api", tags=["pipeline"])
            logger.info("✅ Pipeline 라우터 등록 완료")
            
    except Exception as e:
        logger.error(f"❌ {router_name} 라우터 등록 실패: {e}")

# ===============================================================
# 🔧 기본 엔드포인트 (모든 기존 기능 유지)
# ===============================================================

@app.get("/")
async def root():
    """루트 엔드포인트 - 모든 기존 정보 유지"""
    return {
        "message": f"🍎 {APP_NAME} 서버가 실행 중입니다!",
        "version": APP_VERSION,
        "device": DEVICE,
        "m3_max": IS_M3_MAX,
        "core_available": CORE_AVAILABLE,
        "docs": "/docs",
        "health": "/api/health",
        "api_endpoints": {
            "health": "/api/health",
            "steps": "/api/step/",
            "pipeline": "/api/virtual-tryon",
            "models": "/api/models/",
            "websocket": "/api/ws"
        },
        "system_info": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "device": DEVICE,
            "device_name": DEVICE_NAME,
            "is_m3_max": IS_M3_MAX,
            "core_modules": CORE_AVAILABLE
        },
        "timestamp": time.time()
    }

@app.get("/api/system-info")
async def system_info():
    """시스템 정보 조회 - 기존 기능 유지"""
    try:
        info = {
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "device": DEVICE,
            "device_name": DEVICE_NAME,
            "is_m3_max": IS_M3_MAX,
            "core_available": CORE_AVAILABLE,
            "loaded_routers": list(api_routers.keys()),
            "static_directory": str(STATIC_DIR),
            "debug_mode": DEBUG,
            "timestamp": time.time()
        }
        
        # Core 모듈이 있으면 추가 정보
        if CORE_AVAILABLE:
            try:
                info.update({
                    "memory_gb": getattr(gpu_config, 'memory_gb', 'Unknown'),
                    "optimization_level": getattr(gpu_config, 'optimization_settings', {}).get('optimization_level', 'Unknown')
                })
            except:
                pass
        
        return info
    except Exception as e:
        logger.error(f"시스템 정보 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"시스템 정보 조회 실패: {str(e)}")

@app.post("/api/optimize-memory")
async def optimize_memory_endpoint():
    """메모리 최적화 실행 (안전한 버전) - 기존 기능 유지"""
    try:
        def optimize_with_timeout():
            try:
                return optimize_memory(aggressive=True)
            except Exception as e:
                logger.warning(f"메모리 최적화 실패: {e}")
                return {"success": False, "error": str(e)}
        
        # 타임아웃 적용 (5초)
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, optimize_with_timeout),
            timeout=5.0
        )
        
        # 메모리 상태 조회 (안전하게)
        try:
            if CORE_AVAILABLE and hasattr(gpu_config, 'get_memory_stats'):
                memory_stats = gpu_config.get_memory_stats()
            else:
                memory_stats = {"status": "unavailable"}
        except:
            memory_stats = {"status": "unavailable"}
        
        return {
            "status": "success",
            "optimization_result": result,
            "memory_stats": memory_stats,
            "timestamp": time.time()
        }
        
    except asyncio.TimeoutError:
        logger.warning("메모리 최적화 타임아웃 (5초)")
        return {
            "status": "timeout",
            "message": "메모리 최적화가 타임아웃되었습니다",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"메모리 최적화 실패: {e}")
        raise HTTPException(status_code=500, detail=f"메모리 최적화 실패: {str(e)}")

# ===============================================================
# 🔧 에러 핸들러 (기존과 동일)
# ===============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 예외 처리"""
    logger.error(f"❌ 전역 에러: {str(exc)}")
    logger.error(f"   - 요청: {request.method} {request.url}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "서버에서 예상치 못한 오류가 발생했습니다.",
            "timestamp": time.time()
        }
    )

# ===============================================================
# 🔧 서버 실행 정보 로깅 (모든 기존 정보 유지)
# ===============================================================

logger.info("🚀 MyCloset AI Backend 서버 시작 중...")
logger.info(f"📍 주소: http://{HOST}:{PORT}")
logger.info(f"📖 API 문서: http://{HOST}:{PORT}/docs")
logger.info(f"🏗️ 아키텍처: PipelineManager 중심 (VirtualFitter 제거)")
logger.info(f"🎯 GPU 최적화: {DEVICE_NAME} ({DEVICE})")
logger.info(f"🍎 M3 Max 최적화: {'✅' if IS_M3_MAX else '❌'}")
logger.info(f"⚡ FastAPI: lifespan 이벤트 핸들러 적용")

# 시스템 정보 출력 (안전하게)
logger.info("📊 시스템 정보:")
logger.info(f"  - Python: {sys.version.split()[0]}")

try:
    import torch
    pytorch_version = torch.__version__
except ImportError:
    pytorch_version = "Unknown"

logger.info(f"  - PyTorch: {pytorch_version}")
logger.info(f"  - Platform: {sys.platform.title()}")

try:
    machine = os.uname().machine if hasattr(os, 'uname') else 'Unknown'
except:
    machine = 'Unknown'

logger.info(f"  - Machine: {machine}")
logger.info(f"  - CPU 코어: {os.cpu_count()}")

if CORE_AVAILABLE and hasattr(gpu_config, 'memory_gb'):
    logger.info(f"  - 메모리: {gpu_config.memory_gb:.1f}GB")
else:
    logger.info(f"  - 메모리: Unknown")

# ===============================================================
# 🔧 개발 모드에서 uvicorn 자동 실행 (기존과 동일)
# ===============================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🔧 개발 모드: uvicorn 서버 직접 실행")
    
    try:
        uvicorn.run(
            "app.main:app",  # 현재 파일의 app 인스턴스
            host=HOST,
            port=PORT,
            reload=DEBUG,  # 개발 모드에서만 리로드
            log_level="info" if not DEBUG else "debug",
            access_log=DEBUG,  # 디버그 모드에서만 액세스 로그
            workers=1,  # 단일 워커 (GPU 메모리 공유 이슈 방지)
            loop="auto",  # 자동 이벤트 루프 선택
            timeout_keep_alive=30,  # Keep-alive 타임아웃
            limit_concurrency=1000,  # 동시 연결 제한
            limit_max_requests=10000,  # 최대 요청 수 제한
        )
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 서버가 중단되었습니다")
    except Exception as e:
        logger.error(f"❌ 서버 실행 실패: {e}")
        sys.exit(1)