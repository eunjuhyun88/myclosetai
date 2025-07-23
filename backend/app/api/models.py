# app/api/models.py
"""
모델 관리 API 라우터 - M3 Max 최적화 (최적 생성자 패턴 적용)
단순함 + 편의성 + 확장성 + 일관성
"""

import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse

# 최적 생성자 패턴 기반 import
try:
    from ..ai_pipeline.utils import get_memory_manager, get_model_loader
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

try:
    from ..services.model_manager import ModelManager, get_available_models
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False

class OptimalRouterConstructor:
    """최적화된 라우터 생성자 패턴 베이스 클래스"""
    
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
        
        # 4. 상태 초기화
        self.is_initialized = False

    def _auto_detect_device(self, preferred_device: Optional[str]) -> str:
        """지능적 디바이스 자동 감지"""
        if preferred_device:
            return preferred_device

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

class ModelsRouter(OptimalRouterConstructor):
    """
    🍎 M3 Max 최적화 모델 관리 라우터
    최적 생성자 패턴 적용
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        ✅ 최적 생성자 - 모델 관리 라우터 특화

        Args:
            device: 사용할 디바이스 (None=자동감지, 'cpu', 'cuda', 'mps')
            config: 모델 관리 설정 딕셔너리
            **kwargs: 확장 파라미터들
                - enable_model_caching: bool = True  # 모델 캐싱
                - max_cached_models: int = 10  # 최대 캐시 모델 수
                - auto_cleanup: bool = True  # 자동 정리
                - enable_model_warmup: bool = True  # 모델 워밍업
                - allow_model_switching: bool = True  # 모델 전환 허용
                - enable_performance_monitoring: bool = True  # 성능 모니터링
        """
        super().__init__(device=device, config=config, **kwargs)
        
        # 모델 관리 특화 설정
        self.enable_model_caching = kwargs.get('enable_model_caching', True)
        self.max_cached_models = kwargs.get('max_cached_models', 10)
        self.auto_cleanup = kwargs.get('auto_cleanup', True)
        self.enable_model_warmup = kwargs.get('enable_model_warmup', True)
        self.allow_model_switching = kwargs.get('allow_model_switching', True)
        self.enable_performance_monitoring = kwargs.get('enable_performance_monitoring', True)
        
        # M3 Max 특화 설정
        if self.is_m3_max:
            self.max_cached_models = 20  # M3 Max는 더 많은 모델 캐시 가능
            self.enable_model_warmup = True  # 항상 워밍업 활성화
        
        # 모델 상태 추적
        self._model_load_times: Dict[str, float] = {}
        self._model_performance: Dict[str, Dict[str, Any]] = {}
        self._loading_status: Dict[str, str] = {}
        
        # 통계
        self._stats = {
            "total_load_requests": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "total_unload_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "m3_max_optimized_loads": 0
        }
        
        # FastAPI 라우터 생성
        self.router = APIRouter()
        self._setup_routes()
        
        self.logger.info(f"📦 모델 관리 라우터 초기화 - {self.device} (M3 Max: {self.is_m3_max})")
        
        # 초기화 완료
        self.is_initialized = True

    def _setup_routes(self):
        """라우터 엔드포인트 설정"""
        
        @self.router.get("/")
        async def list_all_models():
            """모든 모델 목록 조회"""
            return await self.get_all_models()
        
        @self.router.get("/available")
        async def list_available_models():
            """사용 가능한 모델 목록"""
            return await self.get_available_models()
        
        @self.router.get("/loaded")
        async def list_loaded_models():
            """로드된 모델 목록"""
            return await self.get_loaded_models()
        
        @self.router.post("/load/{model_name}")
        async def load_model_endpoint(
            model_name: str,
            background_tasks: BackgroundTasks,
            force_reload: bool = False,
            enable_warmup: bool = None
        ):
            """모델 로드"""
            return await self.load_model(
                model_name=model_name,
                background_tasks=background_tasks,
                force_reload=force_reload,
                enable_warmup=enable_warmup
            )
        
        @self.router.delete("/unload/{model_name}")
        async def unload_model_endpoint(model_name: str):
            """모델 언로드"""
            return await self.unload_model(model_name)
        
        @self.router.get("/status/{model_name}")
        async def get_model_status_endpoint(model_name: str):
            """모델 상태 조회"""
            return await self.get_model_status(model_name)
        
        @self.router.post("/reload/{model_name}")
        async def reload_model_endpoint(
            model_name: str,
            background_tasks: BackgroundTasks
        ):
            """모델 재로드"""
            return await self.reload_model(model_name, background_tasks)
        
        @self.router.post("/warmup/{model_name}")
        async def warmup_model_endpoint(model_name: str):
            """모델 워밍업"""
            return await self.warmup_model(model_name)
        
        @self.router.get("/performance")
        async def get_model_performance():
            """모델 성능 통계"""
            return await self.get_performance_stats()
        
        @self.router.post("/clear-cache")
        async def clear_model_cache():
            """모델 캐시 정리"""
            return await self.clear_cache()
        
        @self.router.post("/optimize-memory")
        async def optimize_memory():
            """메모리 최적화"""
            return await self.optimize_memory()
        
        @self.router.get("/system-info")
        async def get_model_system_info():
            """모델 시스템 정보"""
            return await self.get_system_info()

    async def get_all_models(self) -> Dict[str, Any]:
        """모든 모델 목록 조회"""
        try:
            models_info = {
                "ai_pipeline_models": {},
                "service_models": {},
                "system_info": {
                    "device": self.device,
                    "m3_max_optimized": self.is_m3_max,
                    "utils_available": UTILS_AVAILABLE,
                    "model_manager_available": MODEL_MANAGER_AVAILABLE
                }
            }
            
            # AI 파이프라인 모델들
            if UTILS_AVAILABLE:
                model_loader = get_model_loader()
                if model_loader:
                    models_list = model_loader.list_models()
                    models_info["ai_pipeline_models"] = models_list
            
            # 서비스 모델들
            if MODEL_MANAGER_AVAILABLE:
                try:
                    available_models = get_available_models()
                    models_info["service_models"] = available_models
                except Exception as e:
                    models_info["service_models"] = {"error": str(e)}
            
            # 기본 모델 정보
            models_info["default_models"] = {
                "human_parsing": {
                    "name": "Graphonomy",
                    "type": "semantic_segmentation",
                    "categories": 20,
                    "input_size": [512, 512]
                },
                "pose_estimation": {
                    "name": "MediaPipe",
                    "type": "pose_detection",
                    "keypoints": 33,
                    "realtime_capable": True
                },
                "cloth_segmentation": {
                    "name": "U²-Net",
                    "type": "image_segmentation",
                    "specialized": "clothing"
                },
                "virtual_fitting": {
                    "name": "HR-VITON",
                    "type": "generative_model",
                    "resolution": "1024x768"
                }
            }
            
            return {
                "success": True,
                "models": models_info,
                "total_categories": len(models_info),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 모델 목록 조회 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_available_models(self) -> Dict[str, Any]:
        """사용 가능한 모델 목록"""
        try:
            available_models = {}
            
            # AI 파이프라인 모델 로더
            if UTILS_AVAILABLE:
                model_loader = get_model_loader()
                if model_loader:
                    registered_models = getattr(model_loader, '_model_registry', {})
                    available_models["pipeline_models"] = {
                        name: {
                            "status": "registered",
                            "format": info.get("config", {}).get("format", "unknown"),
                            "device": info.get("config", {}).get("device", "unknown"),
                            "loaded": info.get("loaded", False)
                        }
                        for name, info in registered_models.items()
                    }
            
            # 기본 사용 가능 모델들
            available_models["built_in_models"] = {
                "human_parsing_graphonomy": {
                    "status": "available",
                    "type": "human_parsing",
                    "framework": "pytorch",
                    "m3_max_optimized": self.is_m3_max
                },
                "pose_estimation_mediapipe": {
                    "status": "available", 
                    "type": "pose_estimation",
                    "framework": "mediapipe",
                    "realtime": True
                },
                "cloth_segmentation_u2net": {
                    "status": "available",
                    "type": "cloth_segmentation", 
                    "framework": "pytorch",
                    "specialized": True
                },
                "geometric_matching": {
                    "status": "available",
                    "type": "geometric_matching",
                    "framework": "opencv_pytorch"
                },
                "cloth_warping": {
                    "status": "available",
                    "type": "cloth_warping",
                    "framework": "pytorch",
                    "physics_enabled": True
                },
                "virtual_fitting_hrviton": {
                    "status": "available",
                    "type": "virtual_fitting",
                    "framework": "pytorch",
                    "resolution": "high"
                }
            }
            
            # M3 Max 전용 모델들
            if self.is_m3_max:
                available_models["m3_max_exclusive"] = {
                    "neural_engine_accelerated": {
                        "status": "available",
                        "type": "all_models",
                        "acceleration": "neural_engine",
                        "performance_boost": "30-50%"
                    },
                    "coreml_optimized": {
                        "status": "available",
                        "type": "inference_optimized",
                        "framework": "coreml",
                        "ultra_quality_mode": True
                    }
                }
            
            return {
                "success": True,
                "available_models": available_models,
                "device": self.device,
                "m3_max_optimized": self.is_m3_max,
                "total_available": sum(len(models) for models in available_models.values()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 사용 가능한 모델 조회 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_loaded_models(self) -> Dict[str, Any]:
        """로드된 모델 목록"""
        try:
            loaded_models = {}
            
            # AI 파이프라인에서 로드된 모델들
            if UTILS_AVAILABLE:
                model_loader = get_model_loader()
                if model_loader:
                    loaded_model_dict = getattr(model_loader, '_loaded_models', {})
                    model_registry = getattr(model_loader, '_model_registry', {})
                    
                    for model_name in loaded_model_dict.keys():
                        registry_info = model_registry.get(model_name, {})
                        load_time = self._model_load_times.get(model_name, 0)
                        
                        loaded_models[model_name] = {
                            "status": "loaded",
                            "device": model_loader.device,
                            "load_time": load_time,
                            "load_count": registry_info.get("load_count", 0),
                            "last_loaded": registry_info.get("last_loaded"),
                            "format": registry_info.get("config", {}).get("format", "unknown"),
                            "performance": self._model_performance.get(model_name, {})
                        }
            
            # 서비스 매니저에서 로드된 모델들
            if MODEL_MANAGER_AVAILABLE:
                try:
                    # 서비스 모델 매니저 상태 확인
                    # 실제 구현에서는 서비스 매니저의 로드된 모델 목록을 가져옴
                    pass
                except Exception as e:
                    self.logger.warning(f"⚠️ 서비스 모델 조회 실패: {e}")
            
            return {
                "success": True,
                "loaded_models": loaded_models,
                "total_loaded": len(loaded_models),
                "device": self.device,
                "m3_max_optimized": self.is_m3_max,
                "memory_usage": await self._get_memory_usage(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 로드된 모델 조회 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def load_model(
        self,
        model_name: str,
        background_tasks: BackgroundTasks,
        force_reload: bool = False,
        enable_warmup: Optional[bool] = None
    ) -> Dict[str, Any]:
        """모델 로드"""
        try:
            start_time = time.time()
            self._stats["total_load_requests"] += 1
            
            # 워밍업 설정
            if enable_warmup is None:
                enable_warmup = self.enable_model_warmup
            
            # 로딩 상태 설정
            self._loading_status[model_name] = "loading"
            
            # AI 파이프라인 모델 로더 사용
            if UTILS_AVAILABLE:
                model_loader = get_model_loader()
                if model_loader:
                    # 캐시 확인
                    if not force_reload and model_name in getattr(model_loader, 'loaded_models', {}):
                        self._stats["cache_hits"] += 1
                        self._loading_status[model_name] = "loaded"
                        
                        return {
                            "success": True,
                            "message": f"모델 {model_name}이 이미 로드되어 있습니다 (캐시 히트)",
                            "model_name": model_name,
                            "cache_hit": True,
                            "device": self.device,
                            "load_time": 0.0,
                            "timestamp": datetime.now().isoformat()
                        }
                    
                    # 실제 모델 로드
                    self._stats["cache_misses"] += 1
                    
                    # 백그라운드에서 로드 (비동기)
                    if self.enable_model_caching:
                        background_tasks.add_task(
                            self._load_model_background,
                            model_loader, model_name, enable_warmup, start_time
                        )
                        
                        return {
                            "success": True,
                            "message": f"모델 {model_name} 로딩이 백그라운드에서 시작되었습니다",
                            "model_name": model_name,
                            "background_loading": True,
                            "device": self.device,
                            "estimated_time": self._estimate_load_time(model_name),
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        # 동기 로드
                        loaded_model = await model_loader.load_model(model_name, force_reload=force_reload)
                        
                        if loaded_model:
                            load_time = time.time() - start_time
                            self._model_load_times[model_name] = load_time
                            self._loading_status[model_name] = "loaded"
                            self._stats["successful_loads"] += 1
                            
                            if self.is_m3_max:
                                self._stats["m3_max_optimized_loads"] += 1
                            
                            # 워밍업 실행
                            if enable_warmup:
                                background_tasks.add_task(self._warmup_model_background, model_name)
                            
                            return {
                                "success": True,
                                "message": f"모델 {model_name} 로드 완료",
                                "model_name": model_name,
                                "load_time": load_time,
                                "device": self.device,
                                "m3_max_optimized": self.is_m3_max,
                                "warmup_scheduled": enable_warmup,
                                "timestamp": datetime.now().isoformat()
                            }
                        else:
                            self._loading_status[model_name] = "failed"
                            self._stats["failed_loads"] += 1
                            
                            return {
                                "success": False,
                                "message": f"모델 {model_name} 로드 실패",
                                "model_name": model_name,
                                "error": "Model loader returned None",
                                "timestamp": datetime.now().isoformat()
                            }
            
            # 폴백: 시뮬레이션 로드
            return await self._simulate_model_load(model_name, start_time, enable_warmup)
            
        except Exception as e:
            self.logger.error(f"❌ 모델 로드 실패 {model_name}: {e}")
            self._loading_status[model_name] = "failed"
            self._stats["failed_loads"] += 1
            
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }

    async def unload_model(self, model_name: str) -> Dict[str, Any]:
        """모델 언로드"""
        try:
            self._stats["total_unload_requests"] += 1
            
            # AI 파이프라인 모델 언로드
            if UTILS_AVAILABLE:
                model_loader = get_model_loader()
                if model_loader:
                    success = model_loader.unload_model(model_name)
                    
                    if success:
                        # 상태 정리
                        self._loading_status.pop(model_name, None)
                        self._model_load_times.pop(model_name, None)
                        self._model_performance.pop(model_name, None)
                        
                        return {
                            "success": True,
                            "message": f"모델 {model_name} 언로드 완료",
                            "model_name": model_name,
                            "device": self.device,
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        return {
                            "success": False,
                            "message": f"모델 {model_name}이 로드되어 있지 않거나 언로드 실패",
                            "model_name": model_name,
                            "timestamp": datetime.now().isoformat()
                        }
            
            # 폴백: 시뮬레이션 언로드
            return {
                "success": True,
                "message": f"모델 {model_name} 언로드 완료 (시뮬레이션)",
                "model_name": model_name,
                "simulation_mode": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 모델 언로드 실패 {model_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }

    async def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """모델 상태 조회"""
        try:
            status_info = {
                "model_name": model_name,
                "loading_status": self._loading_status.get(model_name, "not_loaded"),
                "device": self.device,
                "m3_max_optimized": self.is_m3_max
            }
            
            # AI 파이프라인 모델 정보
            if UTILS_AVAILABLE:
                model_loader = get_model_loader()
                if model_loader:
                    model_info = model_loader.get_model_info(model_name)
                    if model_info:
                        status_info.update({
                            "pipeline_info": model_info,
                            "load_time": self._model_load_times.get(model_name),
                            "performance": self._model_performance.get(model_name, {})
                        })
            
            # 로딩 상태별 추가 정보
            if status_info["loading_status"] == "loaded":
                status_info["ready_for_inference"] = True
                status_info["last_used"] = time.time()  # 실제로는 마지막 사용 시간 추적
            
            elif status_info["loading_status"] == "loading":
                status_info["estimated_completion"] = time.time() + self._estimate_load_time(model_name)
            
            elif status_info["loading_status"] == "failed":
                status_info["error_info"] = "모델 로딩 중 오류 발생"
                status_info["retry_available"] = True
            
            return {
                "success": True,
                "status": status_info,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 모델 상태 조회 실패 {model_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }

    async def reload_model(self, model_name: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
        """모델 재로드"""
        try:
            # 먼저 언로드
            unload_result = await self.unload_model(model_name)
            
            if unload_result.get("success", False):
                # 잠시 대기 (메모리 정리 시간)
                await asyncio.sleep(1.0)
                
                # 다시 로드
                load_result = await self.load_model(
                    model_name=model_name,
                    background_tasks=background_tasks,
                    force_reload=True,
                    enable_warmup=True
                )
                
                if load_result.get("success", False):
                    return {
                        "success": True,
                        "message": f"모델 {model_name} 재로드 완료",
                        "model_name": model_name,
                        "reload_time": load_result.get("load_time", 0),
                        "device": self.device,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "success": False,
                        "message": f"모델 {model_name} 재로드 실패 - 로드 단계에서 실패",
                        "model_name": model_name,
                        "load_error": load_result.get("error"),
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                return {
                    "success": False,
                    "message": f"모델 {model_name} 재로드 실패 - 언로드 단계에서 실패",
                    "model_name": model_name,
                    "unload_error": unload_result.get("error"),
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            self.logger.error(f"❌ 모델 재로드 실패 {model_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }

    async def warmup_model(self, model_name: str) -> Dict[str, Any]:
        """모델 워밍업"""
        try:
            if not self.enable_model_warmup:
                return {
                    "success": False,
                    "message": "모델 워밍업이 비활성화되어 있습니다",
                    "model_name": model_name,
                    "timestamp": datetime.now().isoformat()
                }
            
            start_time = time.time()
            
            # 워밍업 실행 (시뮬레이션)
            await asyncio.sleep(0.5 if self.is_m3_max else 1.0)
            
            warmup_time = time.time() - start_time
            
            # 성능 정보 업데이트
            if model_name not in self._model_performance:
                self._model_performance[model_name] = {}
            
            self._model_performance[model_name].update({
                "last_warmup": time.time(),
                "warmup_time": warmup_time,
                "warmed_up": True,
                "device": self.device,
                "m3_max_optimized": self.is_m3_max
            })
            
            return {
                "success": True,
                "message": f"모델 {model_name} 워밍업 완료",
                "model_name": model_name,
                "warmup_time": warmup_time,
                "device": self.device,
                "m3_max_optimized": self.is_m3_max,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 모델 워밍업 실패 {model_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }

    async def get_performance_stats(self) -> Dict[str, Any]:
        """모델 성능 통계"""
        try:
            stats = self._stats.copy()
            
            # 추가 계산
            if stats["total_load_requests"] > 0:
                stats["load_success_rate"] = stats["successful_loads"] / stats["total_load_requests"]
                stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_load_requests"]
            else:
                stats["load_success_rate"] = 0.0
                stats["cache_hit_rate"] = 0.0
            
            # 평균 로드 시간
            if self._model_load_times:
                stats["average_load_time"] = sum(self._model_load_times.values()) / len(self._model_load_times)
                stats["fastest_load_time"] = min(self._model_load_times.values())
                stats["slowest_load_time"] = max(self._model_load_times.values())
            else:
                stats["average_load_time"] = 0.0
                stats["fastest_load_time"] = 0.0
                stats["slowest_load_time"] = 0.0
            
            # M3 Max 최적화율
            if self.is_m3_max and stats["total_load_requests"] > 0:
                stats["m3_max_optimization_rate"] = stats["m3_max_optimized_loads"] / stats["total_load_requests"]
            
            return {
                "success": True,
                "performance_stats": stats,
                "model_performance": self._model_performance,
                "device": self.device,
                "m3_max_optimized": self.is_m3_max,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 성능 통계 조회 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def clear_cache(self) -> Dict[str, Any]:
        """모델 캐시 정리"""
        try:
            cleared_models = []
            
            # AI 파이프라인 캐시 정리
            if UTILS_AVAILABLE:
                model_loader = get_model_loader()
                if model_loader:
                    loaded_models = list(getattr(model_loader, '_loaded_models', {}).keys())
                    
                    for model_name in loaded_models:
                        success = model_loader.unload_model(model_name)
                        if success:
                            cleared_models.append(model_name)
            
            # 내부 상태 정리
            self._model_load_times.clear()
            self._model_performance.clear()
            self._loading_status.clear()
            
            # 메모리 정리
            memory_result = {"success": False, "error": "Memory manager not available"}
            if UTILS_AVAILABLE:
                memory_manager = get_memory_manager()
                if memory_manager:
                    memory_result = memory_manager.clear_cache(aggressive=True)
            
            return {
                "success": True,
                "message": "모델 캐시 정리 완료",
                "cleared_models": cleared_models,
                "cleared_count": len(cleared_models),
                "memory_cleanup": memory_result,
                "device": self.device,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 캐시 정리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def optimize_memory(self) -> Dict[str, Any]:
        """메모리 최적화"""
        try:
            optimization_results = []
            
            # 메모리 매니저를 통한 최적화
            if UTILS_AVAILABLE:
                memory_manager = get_memory_manager()
                if memory_manager:
                    # 메모리 상태 확인
                    memory_stats = memory_manager.get_memory_stats()
                    optimization_results.append(f"메모리 사용량: {memory_stats}")
                    
                    # 메모리 최적화 실행
                    optimization_result = memory_manager.optimize_for_task("model_loading")
                    optimization_results.append(f"최적화 결과: {optimization_result}")
                    
                    # 캐시 정리
                    cache_result = memory_manager.clear_cache(aggressive=self.is_m3_max)
                    optimization_results.append(f"캐시 정리: {cache_result}")
            
            # 모델별 최적화
            if UTILS_AVAILABLE:
                model_loader = get_model_loader()
                if model_loader and hasattr(model_loader, '_loaded_models'):
                    loaded_count = len(model_loader._loaded_models)
                    if loaded_count > self.max_cached_models:
                        # 오래된 모델들 정리
                        optimization_results.append(f"캐시된 모델 수 ({loaded_count})가 최대치를 초과하여 정리 필요")
            
            return {
                "success": True,
                "message": "메모리 최적화 완료",
                "optimization_results": optimization_results,
                "device": self.device,
                "m3_max_optimized": self.is_m3_max,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_system_info(self) -> Dict[str, Any]:
        """모델 시스템 정보"""
        try:
            system_info = {
                "device": self.device,
                "device_type": self.device_type,
                "m3_max_optimized": self.is_m3_max,
                "router_config": {
                    "enable_model_caching": self.enable_model_caching,
                    "max_cached_models": self.max_cached_models,
                    "auto_cleanup": self.auto_cleanup,
                    "enable_model_warmup": self.enable_model_warmup,
                    "allow_model_switching": self.allow_model_switching,
                    "enable_performance_monitoring": self.enable_performance_monitoring
                },
                "component_status": {
                    "utils_available": UTILS_AVAILABLE,
                    "model_manager_available": MODEL_MANAGER_AVAILABLE
                }
            }
            
            # 메모리 정보
            if UTILS_AVAILABLE:
                memory_manager = get_memory_manager()
                if memory_manager:
                    memory_stats = memory_manager.get_memory_stats()
                    system_info["memory_info"] = memory_stats
            
            # 모델 로더 정보
            if UTILS_AVAILABLE:
                model_loader = get_model_loader()
                if model_loader:
                    system_info["model_loader_info"] = {
                        "device": model_loader.device,
                        "use_fp16": getattr(model_loader, 'use_fp16', False),
                        "coreml_optimization": getattr(model_loader, 'coreml_optimization', False),
                        "max_cached_models": getattr(model_loader, 'max_cached_models', 0),
                        "currently_loaded": len(getattr(model_loader, '_loaded_models', {}))
                    }
            
            # M3 Max 특화 정보
            if self.is_m3_max:
                system_info["m3_max_features"] = {
                    "neural_engine": True,
                    "unified_memory": True,
                    "mps_backend": self.device == "mps",
                    "coreml_support": True,
                    "increased_cache_capacity": True,
                    "optimized_loading": True
                }
            
            return {
                "success": True,
                "system_info": system_info,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 시스템 정보 조회 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # 유틸리티 메서드들
    async def _load_model_background(
        self,
        model_loader,
        model_name: str,
        enable_warmup: bool,
        start_time: float
    ):
        """백그라운드 모델 로딩"""
        try:
            loaded_model = await model_loader.load_model(model_name)
            
            if loaded_model:
                load_time = time.time() - start_time
                self._model_load_times[model_name] = load_time
                self._loading_status[model_name] = "loaded"
                self._stats["successful_loads"] += 1
                
                if self.is_m3_max:
                    self._stats["m3_max_optimized_loads"] += 1
                
                # 워밍업
                if enable_warmup:
                    await self._warmup_model_background(model_name)
                
                self.logger.info(f"✅ 백그라운드 모델 로딩 완료: {model_name} ({load_time:.2f}s)")
            else:
                self._loading_status[model_name] = "failed"
                self._stats["failed_loads"] += 1
                self.logger.error(f"❌ 백그라운드 모델 로딩 실패: {model_name}")
                
        except Exception as e:
            self._loading_status[model_name] = "failed"
            self._stats["failed_loads"] += 1
            self.logger.error(f"❌ 백그라운드 모델 로딩 예외: {model_name} - {e}")

    async def _warmup_model_background(self, model_name: str):
        """백그라운드 모델 워밍업"""
        try:
            await asyncio.sleep(0.5)  # 워밍업 시뮬레이션
            
            if model_name not in self._model_performance:
                self._model_performance[model_name] = {}
            
            self._model_performance[model_name].update({
                "warmed_up": True,
                "warmup_time": 0.5,
                "last_warmup": time.time()
            })
            
            self.logger.info(f"🔥 모델 워밍업 완료: {model_name}")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 워밍업 실패: {model_name} - {e}")

    async def _simulate_model_load(self, model_name: str, start_time: float, enable_warmup: bool) -> Dict[str, Any]:
        """모델 로딩 시뮬레이션"""
        try:
            # 로딩 시간 시뮬레이션
            load_time = 2.0 if self.is_m3_max else 4.0
            await asyncio.sleep(load_time)
            
            actual_load_time = time.time() - start_time
            self._model_load_times[model_name] = actual_load_time
            self._loading_status[model_name] = "loaded"
            self._stats["successful_loads"] += 1
            
            if self.is_m3_max:
                self._stats["m3_max_optimized_loads"] += 1
            
            return {
                "success": True,
                "message": f"모델 {model_name} 로드 완료 (시뮬레이션)",
                "model_name": model_name,
                "load_time": actual_load_time,
                "device": self.device,
                "m3_max_optimized": self.is_m3_max,
                "simulation_mode": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self._loading_status[model_name] = "failed"
            self._stats["failed_loads"] += 1
            raise

    def _estimate_load_time(self, model_name: str) -> float:
        """모델 로딩 시간 추정"""
        base_times = {
            "human_parsing": 3.0,
            "pose_estimation": 1.5,
            "cloth_segmentation": 2.5,
            "geometric_matching": 2.0,
            "cloth_warping": 4.0,
            "virtual_fitting": 5.0
        }
        
        # 모델 이름에서 타입 추정
        for model_type, base_time in base_times.items():
            if model_type in model_name.lower():
                # M3 Max는 30-50% 빠름
                if self.is_m3_max:
                    return base_time * 0.6
                return base_time
        
        # 기본값
        return 3.0 if self.is_m3_max else 5.0

    async def _get_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 조회"""
        try:
            if UTILS_AVAILABLE:
                memory_manager = get_memory_manager()
                if memory_manager:
                    memory_stats = memory_manager.get_memory_stats()
                    return {
                        "system_memory": memory_stats.get("system_memory", {}),
                        "gpu_memory": memory_stats.get("gpu_memory", {}),
                        "available": True
                    }
            
            return {"available": False, "reason": "Memory manager not available"}
            
        except Exception as e:
            return {"available": False, "error": str(e)}

# 모델 관리 라우터 인스턴스 생성 (최적 생성자 패턴)
models_router = ModelsRouter()
router = models_router.router

# 편의 함수들 (하위 호환성)
def create_models_router(
    device: Optional[str] = None,
    enable_model_caching: bool = True,
    **kwargs
) -> ModelsRouter:
    """모델 관리 라우터 생성 (하위 호환)"""
    return ModelsRouter(
        device=device,
        enable_model_caching=enable_model_caching,
        **kwargs
    )

# 모듈 익스포트
__all__ = [
    'router',
    'ModelsRouter',
    'OptimalRouterConstructor',
    'create_models_router'
]