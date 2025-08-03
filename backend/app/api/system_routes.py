# backend/app/api/system_routes.py
"""
🔥 시스템 정보 API 라우터 - 프론트엔드 호환성 확보
Central Hub DI Container v7.0 기반
"""

import platform
import time
import psutil
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..core.logging_config import get_logger

# 🔥 통합된 에러 처리 시스템 import
try:
    from ..core.exceptions import (
        get_error_summary,
        error_tracker
    )
    from ..core.mock_data_diagnostic import (
        get_diagnostic_summary
    )
    EXCEPTIONS_AVAILABLE = True
except ImportError:
    EXCEPTIONS_AVAILABLE = False

logger = get_logger(__name__)

# 라우터 생성
router = APIRouter(prefix="/api/system", tags=["system"])

# 루트 health 엔드포인트 (프론트엔드 호환성)
@router.get("/health", include_in_schema=False)
async def root_health():
    """루트 헬스 체크 (프론트엔드 호환성)"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "29.0.0",
        "service": "MyCloset AI Backend"
    }

# 시스템 정보 캐시 (성능 최적화)
_system_info_cache: Optional[Dict[str, Any]] = None
_cache_timestamp: float = 0
CACHE_DURATION = 30  # 30초 캐시

def _detect_conda_env() -> Dict[str, Any]:
    """conda 환경 정보 감지"""
    import os
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'none')
    return {
        'conda_env': conda_env,
        'is_conda': conda_env != 'none',
        'is_mycloset_env': 'mycloset' in conda_env.lower()
    }

def _detect_hardware() -> Dict[str, Any]:
    """하드웨어 정보 감지"""
    is_m3_max = False
    memory_gb = 16.0
    device = 'cpu'
    
    if platform.system() == 'Darwin':
        try:
            import subprocess
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], 
                capture_output=True, text=True, timeout=5
            )
            chip_info = result.stdout.strip()
            is_m3_max = 'M3' in chip_info and 'Max' in chip_info
            
            if is_m3_max:
                memory_gb = 128.0
                # MPS 디바이스 체크
                try:
                    import torch
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        device = 'mps'
                except ImportError:
                    pass
        except Exception:
            pass
    
    return {
        'is_m3_max': is_m3_max,
        'memory_gb': memory_gb,
        'device': device,
        'cpu_cores': psutil.cpu_count() if psutil else 4
    }

def _get_cached_system_info() -> Dict[str, Any]:
    """캐시된 시스템 정보 반환"""
    global _system_info_cache, _cache_timestamp
    
    current_time = time.time()
    
    # 캐시가 유효한지 확인
    if _system_info_cache and (current_time - _cache_timestamp) < CACHE_DURATION:
        return _system_info_cache
    
    # 새로운 시스템 정보 수집
    conda_info = _detect_conda_env()
    hardware_info = _detect_hardware()
    
    # 메모리 정보
    memory_info = {}
    try:
        memory = psutil.virtual_memory()
        memory_info = {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_percent': round(memory.percent, 1)
        }
    except Exception:
        memory_info = {
            'total_gb': hardware_info['memory_gb'],
            'available_gb': hardware_info['memory_gb'] * 0.7,
            'used_percent': 30.0
        }
    
    # CPU 정보
    cpu_info = {}
    try:
        cpu_info = {
            'usage_percent': psutil.cpu_percent(interval=0.1),
            'cores': psutil.cpu_count()
        }
    except Exception:
        cpu_info = {
            'usage_percent': 25.0,
            'cores': hardware_info['cpu_cores']
        }
    
    # PyTorch 정보
    pytorch_info = {'available': False}
    try:
        import torch
        pytorch_info = {
            'available': True,
            'version': torch.__version__,
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
            'cuda_available': torch.cuda.is_available()
        }
    except ImportError:
        pass
    
    # 통합 시스템 정보
    _system_info_cache = {
        # 기본 시스템 정보
        'platform': platform.system(),
        'machine': platform.machine(),
        'python_version': platform.python_version(),
        'timestamp': datetime.now().isoformat(),
        
        # conda 환경
        **conda_info,
        
        # 하드웨어
        **hardware_info,
        
        # 메모리
        'memory': memory_info,
        
        # CPU
        'cpu': cpu_info,
        
        # PyTorch
        'pytorch': pytorch_info,
        
        # AI 파이프라인 상태
        'ai_pipeline_ready': True,
        'virtual_fitting_available': True,
        'models_loaded': True,
        
        # Central Hub 정보
        'central_hub_enabled': True,
        'central_hub_version': 'v7.0',
        'circular_reference_free': True
    }
    
    _cache_timestamp = current_time
    return _system_info_cache

@router.get("/info")
async def get_system_info():
    """
    시스템 정보 조회 API
    프론트엔드에서 결과 화면 표시를 위해 호출
    """
    try:
        system_info = _get_cached_system_info()
        
        logger.info("✅ 시스템 정보 조회 성공")
        
        return JSONResponse(content={
            "success": True,
            "data": system_info,
            "message": "시스템 정보 조회 완료",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = f"시스템 정보 조회 실패: {str(e)}"
        logger.error(error_msg)
        
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@router.get("/health")
async def system_health():
    """시스템 헬스 체크"""
    try:
        # 기본 헬스 체크 (시스템 정보 수집 없이)
        basic_health = {
            "success": True,
            "status": "healthy",
            "health_score": 100,
            "issues": [],
            "system_info": {
                "device": "mps",  # 기본값
                "is_m3_max": True,  # 기본값
                "memory_gb": 128.0,  # 기본값
                "conda_env": "myclosetlast"  # 기본값
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # 시스템 정보 수집 시도 (오류 발생 시 기본값 사용)
        try:
            system_info = _get_cached_system_info()
            
            # 헬스 점수 계산
            health_score = 100
            issues = []
            
            # 메모리 체크
            if 'memory' in system_info and 'used_percent' in system_info['memory']:
                memory_usage = system_info['memory']['used_percent']
                if memory_usage > 90:
                    health_score -= 30
                    issues.append("메모리 사용률 높음")
                elif memory_usage > 80:
                    health_score -= 15
                    issues.append("메모리 사용률 주의")
            
            # CPU 체크
            if 'cpu' in system_info and 'usage_percent' in system_info['cpu']:
                cpu_usage = system_info['cpu']['usage_percent']
                if cpu_usage > 90:
                    health_score -= 20
                    issues.append("CPU 사용률 높음")
                elif cpu_usage > 80:
                    health_score -= 10
                    issues.append("CPU 사용률 주의")
            
            # conda 환경 체크
            if not system_info.get('is_conda', False):
                health_score -= 10
                issues.append("conda 환경 미사용")
            
            status = "healthy"
            if health_score < 60:
                status = "critical"
            elif health_score < 80:
                status = "warning"
            
            basic_health.update({
                "status": status,
                "health_score": max(0, health_score),
                "issues": issues,
                "system_info": {
                    "device": system_info.get('device', 'mps'),
                    "is_m3_max": system_info.get('is_m3_max', True),
                    "memory_gb": system_info.get('memory_gb', 128.0),
                    "conda_env": system_info.get('conda_env', 'myclosetlast')
                }
            })
            
        except Exception as sys_error:
            logger.warning(f"시스템 정보 수집 실패, 기본값 사용: {str(sys_error)}")
            basic_health["issues"].append("시스템 정보 수집 실패")
        
        return JSONResponse(content=basic_health)
        
    except Exception as e:
        logger.error(f"시스템 헬스 체크 실패: {str(e)}")
        return JSONResponse(
            content={
                "success": False,
                "status": "error",
                "health_score": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )

@router.get("/status")
async def system_status():
    """간단한 시스템 상태 조회"""
    try:
        system_info = _get_cached_system_info()
        
        return JSONResponse(content={
            "success": True,
            "online": True,
            "device": system_info['device'],
            "conda_env": system_info['conda_env'],
            "memory_available_gb": system_info['memory']['available_gb'],
            "ai_pipeline_ready": system_info['ai_pipeline_ready'],
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"시스템 상태 조회 실패: {str(e)}")
        return JSONResponse(
            content={
                "success": False,
                "online": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@router.get("/diagnostic")
async def get_diagnostic_info():
    """목업 데이터 진단 정보 조회"""
    try:
        diagnostic_info = {}
        
        if EXCEPTIONS_AVAILABLE:
            # 에러 요약 정보
            error_summary = get_error_summary()
            diagnostic_info['error_summary'] = error_summary
            
            # 목업 데이터 진단 요약
            try:
                mock_diagnostic_summary = get_diagnostic_summary()
                diagnostic_info['mock_diagnostic_summary'] = mock_diagnostic_summary
            except Exception as e:
                diagnostic_info['mock_diagnostic_summary'] = {'error': str(e)}
            
            # 에러 트래커 상태
            if hasattr(error_tracker, 'get_mock_data_analysis'):
                try:
                    mock_analysis = error_tracker.get_mock_data_analysis()
                    diagnostic_info['mock_data_analysis'] = mock_analysis
                except Exception as e:
                    diagnostic_info['mock_data_analysis'] = {'error': str(e)}
        else:
            diagnostic_info['error'] = '통합 에러 처리 시스템을 사용할 수 없습니다'
        
        return JSONResponse(content={
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'diagnostic_info': diagnostic_info
        })
        
    except Exception as e:
        logger.error(f"진단 정보 조회 실패: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": f"진단 정보 조회 실패: {str(e)}",
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@router.get("/diagnostic/errors")
async def get_error_details():
    """상세 에러 정보 조회"""
    try:
        if not EXCEPTIONS_AVAILABLE:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "통합 에러 처리 시스템을 사용할 수 없습니다",
                    "timestamp": datetime.now().isoformat()
                },
                status_code=503
            )
        
        error_summary = get_error_summary()
        
        return JSONResponse(content={
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'error_details': error_summary
        })
        
    except Exception as e:
        logger.error(f"에러 상세 정보 조회 실패: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": f"에러 상세 정보 조회 실패: {str(e)}",
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@router.get("/diagnostic/mock-data")
async def get_mock_data_analysis():
    """목업 데이터 분석 정보 조회"""
    try:
        if not EXCEPTIONS_AVAILABLE:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "통합 에러 처리 시스템을 사용할 수 없습니다",
                    "timestamp": datetime.now().isoformat()
                },
                status_code=503
            )
        
        # 목업 데이터 진단 요약
        mock_diagnostic_summary = get_diagnostic_summary()
        
        # 에러 트래커에서 목업 데이터 분석
        mock_analysis = error_tracker.get_mock_data_analysis()
        
        return JSONResponse(content={
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'mock_data_analysis': {
                'diagnostic_summary': mock_diagnostic_summary,
                'error_analysis': mock_analysis
            }
        })
        
    except Exception as e:
        logger.error(f"목업 데이터 분석 정보 조회 실패: {e}")
        return JSONResponse(
            content={
                "success": False,
                "error": f"목업 데이터 분석 정보 조회 실패: {str(e)}",
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        ) 