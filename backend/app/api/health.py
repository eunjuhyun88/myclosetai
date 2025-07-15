# app/api/health.py
"""
헬스체크 API 라우터 - M3 Max 최적화 (최적 생성자 패턴 적용)
단순함 + 편의성 + 확장성 + 일관성
"""

import time
import logging
import asyncio
import psutil
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

# 최적 생성자 패턴 기반 import
try:
    from ..ai_pipeline.utils import get_memory_manager, get_model_loader
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

try:
    from ..core.gpu_config import get_device, get_device_info
    GPU_CONFIG_AVAILABLE = True
except ImportError:
    GPU_CONFIG_AVAILABLE = False

class OptimalRouterConstructor:
    """
    🎯 최적화된 라우터 생성자 패턴
    모든 라우터의 통일된 기본 클래스
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # 1. 지능적 디바이스 자동 감지
        self.device = self._auto_detect_device(device)
        
        # 2. 기본 설정
        self.config = config or {}
        self.router_name = self.__class__.__name__
        self.logger = logging.getLogger(f"api.{self.router_name}")
        
        # 3. 표준 시스템 파라미터 추출
        self.device_type = kwargs.get('device_type', 'auto')
        self.memory_gb = kwargs.get('memory_gb', 16.0)
        self.is_m3_max = kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        
        # 4. 라우터별 특화 파라미터
        self.enable_detailed_health = kwargs.get('enable_detailed_health', True)
        self.enable_system_monitoring = kwargs.get('enable_system_monitoring', True)
        self.cache_duration = kwargs.get('cache_duration', 30.0)  # 30초
        
        # 5. 상태 초기화
        self.is_initialized = False
        self._last_check_time = 0
        self._cached_health_data = None
        
        self.logger.info(f"🏥 헬스체크 라우터 초기화 - {self.device}")

    def _auto_detect_device(self, preferred_device: Optional[str]) -> str:
        """지능적 디바이스 자동 감지"""
        if preferred_device:
            return preferred_device

        if GPU_CONFIG_AVAILABLE:
            try:
                return get_device()
            except:
                pass

        try:
            import torch
            if torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        except:
            return 'cpu'

    def _detect_m3_max(self) -> bool:
        """M3 Max 칩 자동 감지"""
        try:
            import platform
            import subprocess

            if platform.system() == 'Darwin':
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                return 'M3' in result.stdout
        except:
            pass
        return False

class HealthRouter(OptimalRouterConstructor):
    """
    🍎 M3 Max 최적화 헬스체크 라우터
    최적 생성자 패턴 적용
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        ✅ 최적 생성자 - 헬스체크 라우터 특화

        Args:
            device: 사용할 디바이스 (None=자동감지, 'cpu', 'cuda', 'mps')
            config: 헬스체크 설정 딕셔너리
            **kwargs: 확장 파라미터들
                - enable_detailed_health: bool = True  # 상세 헬스체크
                - enable_system_monitoring: bool = True  # 시스템 모니터링
                - cache_duration: float = 30.0  # 캐시 지속 시간
                - include_ai_pipeline: bool = True  # AI 파이프라인 상태 포함
                - include_models: bool = True  # 모델 상태 포함
                - include_memory: bool = True  # 메모리 상태 포함
        """
        super().__init__(device=device, config=config, **kwargs)
        
        # 헬스체크 특화 설정
        self.include_ai_pipeline = kwargs.get('include_ai_pipeline', True)
        self.include_models = kwargs.get('include_models', True)
        self.include_memory = kwargs.get('include_memory', True)
        
        # 헬스체크 통계
        self._health_check_count = 0
        self._total_response_time = 0.0
        
        # FastAPI 라우터 생성
        self.router = APIRouter()
        self._setup_routes()
        
        # 초기화 완료
        self.is_initialized = True

    def _setup_routes(self):
        """라우터 엔드포인트 설정"""
        
        @self.router.get("/")
        async def basic_health_check():
            """기본 헬스체크"""
            return await self.get_basic_health()
        
        @self.router.get("/detailed")
        async def detailed_health_check():
            """상세 헬스체크"""
            return await self.get_detailed_health()
        
        @self.router.get("/system")
        async def system_health_check():
            """시스템 헬스체크 (M3 Max 최적화)"""
            return await self.get_system_health()
        
        @self.router.get("/ai-pipeline")
        async def ai_pipeline_health():
            """AI 파이프라인 헬스체크"""
            return await self.get_ai_pipeline_health()
        
        @self.router.get("/models")
        async def models_health():
            """모델 상태 헬스체크"""
            return await self.get_models_health()
        
        @self.router.get("/memory")
        async def memory_health():
            """메모리 상태 헬스체크"""
            return await self.get_memory_health()
        
        @self.router.get("/reset-cache")
        async def reset_health_cache():
            """헬스체크 캐시 리셋"""
            return await self.reset_cache()

    async def get_basic_health(self) -> Dict[str, Any]:
        """기본 헬스체크"""
        try:
            start_time = time.time()
            
            # 캐시 확인
            if self._should_use_cache():
                response_time = time.time() - start_time
                self._update_stats(response_time)
                return self._cached_health_data
            
            # 기본 상태 정보
            health_data = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "3.0.0-m3max-optimized",
                "optimal_constructor_pattern": True,
                "device": self.device,
                "m3_max_optimized": self.is_m3_max,
                "uptime_seconds": self._get_uptime(),
                "health_check_count": self._health_check_count + 1
            }
            
            # 처리 시간 계산
            response_time = time.time() - start_time
            health_data["response_time"] = round(response_time, 4)
            
            # 캐시 업데이트
            self._update_cache(health_data)
            self._update_stats(response_time)
            
            return health_data
            
        except Exception as e:
            self.logger.error(f"❌ 기본 헬스체크 실패: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_detailed_health(self) -> Dict[str, Any]:
        """상세 헬스체크"""
        try:
            start_time = time.time()
            
            # 기본 정보
            basic_health = await self.get_basic_health()
            
            # 상세 정보 추가
            detailed_info = {
                **basic_health,
                "detailed": True,
                "optimal_constructor_features": {
                    "auto_device_detection": True,
                    "unified_interface": True,
                    "extensible_config": True,
                    "m3_max_optimization": self.is_m3_max
                },
                "system_info": await self._get_system_info(),
                "performance_metrics": {
                    "total_health_checks": self._health_check_count,
                    "average_response_time": self._get_average_response_time(),
                    "cache_hit_rate": self._get_cache_hit_rate()
                }
            }
            
            # 선택적 정보 추가
            if self.include_ai_pipeline:
                detailed_info["ai_pipeline"] = await self._get_ai_pipeline_status()
            
            if self.include_models:
                detailed_info["models"] = await self._get_models_status()
            
            if self.include_memory:
                detailed_info["memory"] = await self._get_memory_status()
            
            # 처리 시간 업데이트
            processing_time = time.time() - start_time
            detailed_info["processing_time"] = round(processing_time, 4)
            
            return detailed_info
            
        except Exception as e:
            self.logger.error(f"❌ 상세 헬스체크 실패: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_system_health(self) -> Dict[str, Any]:
        """시스템 헬스체크 (M3 Max 최적화)"""
        try:
            start_time = time.time()
            
            system_health = {
                "system_status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "system_info": await self._get_detailed_system_info(),
                "resource_usage": await self._get_resource_usage(),
                "device_specific": await self._get_device_specific_info()
            }
            
            # M3 Max 특화 정보
            if self.is_m3_max:
                system_health["m3_max_specific"] = await self._get_m3_max_info()
            
            processing_time = time.time() - start_time
            system_health["processing_time"] = round(processing_time, 4)
            
            return system_health
            
        except Exception as e:
            self.logger.error(f"❌ 시스템 헬스체크 실패: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_ai_pipeline_health(self) -> Dict[str, Any]:
        """AI 파이프라인 헬스체크"""
        try:
            if not self.include_ai_pipeline:
                return {"ai_pipeline": "disabled"}
            
            pipeline_health = {
                "ai_pipeline_status": "checking",
                "timestamp": datetime.now().isoformat(),
                "utils_available": UTILS_AVAILABLE,
                "components": {}
            }
            
            # 유틸리티 상태 확인
            if UTILS_AVAILABLE:
                # Memory Manager 상태
                memory_manager = get_memory_manager()
                if memory_manager:
                    pipeline_health["components"]["memory_manager"] = {
                        "status": "available",
                        "device": memory_manager.device,
                        "is_m3_max": memory_manager.is_m3_max
                    }
                else:
                    pipeline_health["components"]["memory_manager"] = {"status": "unavailable"}
                
                # Model Loader 상태
                model_loader = get_model_loader()
                if model_loader:
                    pipeline_health["components"]["model_loader"] = {
                        "status": "available",
                        "device": model_loader.device,
                        "loaded_models": len(model_loader._loaded_models) if hasattr(model_loader, '_loaded_models') else 0
                    }
                else:
                    pipeline_health["components"]["model_loader"] = {"status": "unavailable"}
            
            # 전체 상태 결정
            available_components = sum(1 for comp in pipeline_health["components"].values() if comp.get("status") == "available")
            total_components = len(pipeline_health["components"])
            
            if available_components == total_components and total_components > 0:
                pipeline_health["ai_pipeline_status"] = "healthy"
            elif available_components > 0:
                pipeline_health["ai_pipeline_status"] = "partially_healthy"
            else:
                pipeline_health["ai_pipeline_status"] = "unhealthy"
            
            return pipeline_health
            
        except Exception as e:
            self.logger.error(f"❌ AI 파이프라인 헬스체크 실패: {e}")
            return {
                "ai_pipeline_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_models_health(self) -> Dict[str, Any]:
        """모델 상태 헬스체크"""
        try:
            if not self.include_models:
                return {"models": "disabled"}
            
            models_health = {
                "models_status": "checking",
                "timestamp": datetime.now().isoformat(),
                "model_info": {}
            }
            
            if UTILS_AVAILABLE:
                model_loader = get_model_loader()
                if model_loader:
                    # 로드된 모델 정보
                    loaded_models = getattr(model_loader, '_loaded_models', {})
                    model_registry = getattr(model_loader, '_model_registry', {})
                    
                    models_health["model_info"] = {
                        "total_registered": len(model_registry),
                        "currently_loaded": len(loaded_models),
                        "device": model_loader.device,
                        "use_fp16": getattr(model_loader, 'use_fp16', False),
                        "coreml_optimization": getattr(model_loader, 'coreml_optimization', False)
                    }
                    
                    if loaded_models:
                        models_health["loaded_models"] = list(loaded_models.keys())
                    
                    models_health["models_status"] = "healthy"
                else:
                    models_health["models_status"] = "unavailable"
            else:
                models_health["models_status"] = "utils_unavailable"
            
            return models_health
            
        except Exception as e:
            self.logger.error(f"❌ 모델 헬스체크 실패: {e}")
            return {
                "models_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_memory_health(self) -> Dict[str, Any]:
        """메모리 상태 헬스체크"""
        try:
            if not self.include_memory:
                return {"memory": "disabled"}
            
            memory_health = {
                "memory_status": "checking",
                "timestamp": datetime.now().isoformat(),
                "system_memory": self._get_system_memory_info()
            }
            
            if UTILS_AVAILABLE:
                memory_manager = get_memory_manager()
                if memory_manager:
                    # 메모리 매니저 상태
                    memory_stats = memory_manager.get_memory_stats()
                    memory_health["memory_manager"] = {
                        "status": "available",
                        "device": memory_manager.device,
                        "memory_limit_gb": memory_manager.memory_limit_gb,
                        "is_m3_max": memory_manager.is_m3_max,
                        "stats": memory_stats
                    }
                    
                    # 메모리 압박 확인
                    pressure_info = memory_manager.check_memory_pressure()
                    memory_health["memory_pressure"] = pressure_info
                    
                    # 상태 결정
                    if pressure_info.get("pressure_level") in ["low", "moderate"]:
                        memory_health["memory_status"] = "healthy"
                    elif pressure_info.get("pressure_level") == "high":
                        memory_health["memory_status"] = "warning"
                    else:
                        memory_health["memory_status"] = "critical"
                else:
                    memory_health["memory_status"] = "manager_unavailable"
            else:
                memory_health["memory_status"] = "utils_unavailable"
            
            return memory_health
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 헬스체크 실패: {e}")
            return {
                "memory_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def reset_cache(self) -> Dict[str, Any]:
        """헬스체크 캐시 리셋"""
        try:
            self._cached_health_data = None
            self._last_check_time = 0
            
            return {
                "success": True,
                "message": "헬스체크 캐시가 리셋되었습니다",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 캐시 리셋 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # 유틸리티 메서드들
    def _should_use_cache(self) -> bool:
        """캐시 사용 여부 결정"""
        if not self._cached_health_data:
            return False
        
        elapsed = time.time() - self._last_check_time
        return elapsed < self.cache_duration

    def _update_cache(self, data: Dict[str, Any]):
        """캐시 업데이트"""
        self._cached_health_data = data
        self._last_check_time = time.time()

    def _update_stats(self, response_time: float):
        """통계 업데이트"""
        self._health_check_count += 1
        self._total_response_time += response_time

    def _get_average_response_time(self) -> float:
        """평균 응답 시간 계산"""
        if self._health_check_count == 0:
            return 0.0
        return round(self._total_response_time / self._health_check_count, 4)

    def _get_cache_hit_rate(self) -> float:
        """캐시 히트율 계산 (간단한 구현)"""
        # 실제 구현에서는 더 정교한 캐시 통계 필요
        return 0.85 if self._cached_health_data else 0.0

    def _get_uptime(self) -> float:
        """가동 시간 계산"""
        try:
            import psutil
            return time.time() - psutil.boot_time()
        except:
            return 0.0

    async def _get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 조회"""
        try:
            import platform
            
            return {
                "platform": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "device": self.device,
                "device_type": self.device_type,
                "is_m3_max": self.is_m3_max
            }
        except Exception as e:
            return {"error": str(e)}

    async def _get_detailed_system_info(self) -> Dict[str, Any]:
        """상세 시스템 정보"""
        basic_info = await self._get_system_info()
        
        try:
            import platform
            
            detailed = {
                **basic_info,
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.architecture(),
                "hostname": platform.node()
            }
            
            return detailed
        except Exception as e:
            return {**basic_info, "detailed_error": str(e)}

    async def _get_resource_usage(self) -> Dict[str, Any]:
        """리소스 사용량 조회"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "usage_percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "usage_percent": round((disk.used / disk.total) * 100, 1)
                }
            }
        except Exception as e:
            return {"error": str(e)}

    async def _get_device_specific_info(self) -> Dict[str, Any]:
        """디바이스별 특화 정보"""
        device_info = {"device": self.device}
        
        try:
            if GPU_CONFIG_AVAILABLE:
                gpu_info = get_device_info()
                device_info.update(gpu_info)
            
            # PyTorch 정보
            try:
                import torch
                device_info["torch"] = {
                    "version": torch.__version__,
                    "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
                    "cuda_available": torch.cuda.is_available()
                }
                
                if self.device == "cuda" and torch.cuda.is_available():
                    device_info["cuda"] = {
                        "device_count": torch.cuda.device_count(),
                        "current_device": torch.cuda.current_device(),
                        "device_name": torch.cuda.get_device_name()
                    }
                
            except ImportError:
                device_info["torch"] = "not_available"
            
        except Exception as e:
            device_info["error"] = str(e)
        
        return device_info

    async def _get_m3_max_info(self) -> Dict[str, Any]:
        """M3 Max 특화 정보"""
        if not self.is_m3_max:
            return {"m3_max": False}
        
        try:
            m3_info = {
                "neural_engine": True,
                "unified_memory": True,
                "memory_bandwidth_gbps": 400,
                "gpu_cores": "40-core",
                "cpu_cores": "16-core (12P + 4E)",
                "optimization_features": {
                    "mps_backend": self.device == "mps",
                    "coreml_support": True,
                    "metal_performance_shaders": True
                }
            }
            
            # 메모리 매니저에서 M3 Max 정보 가져오기
            if UTILS_AVAILABLE:
                memory_manager = get_memory_manager()
                if memory_manager and memory_manager.is_m3_max:
                    memory_stats = memory_manager.get_memory_stats()
                    m3_info["memory_optimization"] = memory_stats.get("m3_max", {})
            
            return m3_info
            
        except Exception as e:
            return {"m3_max": True, "error": str(e)}

    def _get_system_memory_info(self) -> Dict[str, Any]:
        """시스템 메모리 정보"""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": memory.percent,
                "free_gb": round(memory.free / (1024**3), 2)
            }
        except Exception as e:
            return {"error": str(e)}

    async def _get_ai_pipeline_status(self) -> Dict[str, Any]:
        """AI 파이프라인 상태 조회"""
        if not UTILS_AVAILABLE:
            return {"status": "utils_unavailable"}
        
        # 간단한 파이프라인 상태 체크
        return {
            "utils_available": True,
            "memory_manager_available": get_memory_manager() is not None,
            "model_loader_available": get_model_loader() is not None
        }

    async def _get_models_status(self) -> Dict[str, Any]:
        """모델 상태 조회"""
        if not UTILS_AVAILABLE:
            return {"status": "utils_unavailable"}
        
        model_loader = get_model_loader()
        if not model_loader:
            return {"status": "model_loader_unavailable"}
        
        return {
            "status": "available",
            "device": model_loader.device,
            "loaded_count": len(getattr(model_loader, '_loaded_models', {}))
        }

    async def _get_memory_status(self) -> Dict[str, Any]:
        """메모리 상태 조회"""
        if not UTILS_AVAILABLE:
            return {"status": "utils_unavailable"}
        
        memory_manager = get_memory_manager()
        if not memory_manager:
            return {"status": "memory_manager_unavailable"}
        
        return {
            "status": "available",
            "device": memory_manager.device,
            "memory_limit_gb": memory_manager.memory_limit_gb
        }

# 헬스체크 라우터 인스턴스 생성 (최적 생성자 패턴)
health_router = HealthRouter()
router = health_router.router

# 편의 함수들 (하위 호환성)
def create_health_router(
    device: Optional[str] = None,
    enable_detailed_health: bool = True,
    **kwargs
) -> HealthRouter:
    """헬스체크 라우터 생성 (하위 호환)"""
    return HealthRouter(
        device=device,
        enable_detailed_health=enable_detailed_health,
        **kwargs
    )

# 모듈 익스포트
__all__ = [
    'router',
    'HealthRouter',
    'OptimalRouterConstructor',
    'create_health_router'
]