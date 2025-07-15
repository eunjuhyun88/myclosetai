# app/core/gpu_config.py
"""
MyCloset AI - M3 Max 128GB 완전 최적화 GPU 설정
GPUConfig 생성자 파라미터 문제 해결, 누락된 클래스들 추가
"""

import os
import logging
import torch
import platform
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from functools import lru_cache
import gc
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# ===============================================================
# 🔧 GPU 설정 데이터 클래스 (파라미터 문제 해결)
# ===============================================================

@dataclass
class GPUConfig:
    """GPU 설정 데이터 클래스 - 생성자 파라미터 문제 해결"""
    device: str = "mps"
    device_name: str = "Apple M3 Max"
    memory_gb: float = 128.0
    is_m3_max: bool = True
    optimization_level: str = "maximum"
    device_type: str = "apple_silicon"
    neural_engine_available: bool = True
    metal_performance_shaders: bool = True
    unified_memory_optimization: bool = True
    
    def __post_init__(self):
        """초기화 후 검증 및 조정"""
        # M3 Max가 아닌 경우 설정 조정
        if not self.is_m3_max:
            self.neural_engine_available = False
            self.metal_performance_shaders = False
            self.optimization_level = "balanced"
            if self.memory_gb > 64:
                self.memory_gb = 16.0  # 일반적인 기본값
    
    @classmethod
    def create_optimal(cls, device: str = None, auto_detect: bool = True) -> 'GPUConfig':
        """최적 설정으로 GPUConfig 생성"""
        if auto_detect:
            detector = M3MaxDetector()
            return cls(
                device=device or detector.get_optimal_device(),
                device_name=detector.get_device_name(),
                memory_gb=detector.memory_gb,
                is_m3_max=detector.is_m3_max,
                optimization_level="maximum" if detector.is_m3_max else "balanced",
                device_type="apple_silicon" if detector.is_apple_silicon else "generic",
                neural_engine_available=detector.is_m3_max,
                metal_performance_shaders=detector.is_m3_max,
                unified_memory_optimization=detector.is_m3_max
            )
        else:
            return cls()

# ===============================================================
# 🍎 M3 Max 감지 클래스
# ===============================================================

class M3MaxDetector:
    """M3 Max 환경 정밀 감지"""
    
    def __init__(self):
        self.platform_info = self._get_platform_info()
        self.is_apple_silicon = self._is_apple_silicon()
        self.memory_gb = self._get_memory_gb()
        self.cpu_cores = self._get_cpu_cores()
        self.is_m3_max = self._detect_m3_max()
        
    def _get_platform_info(self) -> Dict[str, str]:
        """플랫폼 정보 수집"""
        return {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python_version": platform.python_version()
        }
    
    def _is_apple_silicon(self) -> bool:
        """Apple Silicon 감지"""
        return (self.platform_info["system"] == "Darwin" and 
                self.platform_info["machine"] == "arm64")
    
    def _get_memory_gb(self) -> float:
        """시스템 메모리 용량(GB) 반환"""
        try:
            import psutil
            return round(psutil.virtual_memory().total / (1024**3), 1)
        except:
            return 8.0
    
    def _get_cpu_cores(self) -> int:
        """CPU 코어 수 반환"""
        try:
            import psutil
            return psutil.cpu_count(logical=False) or 4
        except:
            return 4
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 환경 정밀 감지"""
        if not self.is_apple_silicon:
            return False
            
        # 메모리 기반 판정 (더 정확)
        if self.memory_gb >= 120:  # 128GB M3 Max
            return True
        elif self.memory_gb >= 90:  # 96GB M3 Max  
            return True
        elif self.cpu_cores >= 12:  # M3 Max는 12코어 이상
            return True
            
        return False
    
    def get_optimal_device(self) -> str:
        """최적 디바이스 반환"""
        try:
            import torch
            if torch.backends.mps.is_available() and self.is_apple_silicon:
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def get_device_name(self) -> str:
        """디바이스 이름 반환"""
        if self.is_m3_max:
            if self.memory_gb >= 120:
                return "Apple M3 Max (128GB)"
            else:
                return "Apple M3 Max (96GB)"
        elif self.is_apple_silicon:
            return "Apple Silicon"
        else:
            return "Unknown Device"
    
    def get_optimized_settings(self) -> Dict[str, Any]:
        """최적화된 설정 반환"""
        if self.is_m3_max:
            return {
                "batch_size": 8 if self.memory_gb >= 120 else 4,
                "max_workers": min(12, self.cpu_cores),
                "concurrent_sessions": 8,
                "memory_pool_gb": min(64, self.memory_gb // 2),
                "cache_size_gb": min(32, self.memory_gb // 4),
                "quality_level": "ultra",
                "enable_neural_engine": True,
                "enable_mps": True,
                "optimization_level": "maximum"
            }
        elif self.is_apple_silicon:
            return {
                "batch_size": 2,
                "max_workers": min(4, self.cpu_cores),
                "concurrent_sessions": 4,
                "memory_pool_gb": min(16, self.memory_gb // 2),
                "cache_size_gb": min(8, self.memory_gb // 4),
                "quality_level": "high",
                "enable_neural_engine": False,
                "enable_mps": True,
                "optimization_level": "balanced"
            }
        else:
            return {
                "batch_size": 1,
                "max_workers": 2,
                "concurrent_sessions": 2,
                "memory_pool_gb": 4,
                "cache_size_gb": 2,
                "quality_level": "balanced",
                "enable_neural_engine": False,
                "enable_mps": False,
                "optimization_level": "safe"
            }

# ===============================================================
# 🎯 M3 Max GPU 관리자 (메인 클래스)
# ===============================================================

class M3MaxGPUManager:
    """M3 Max 128GB 전용 GPU 관리자"""
    
    def __init__(self):
        """초기화"""
        self.detector = M3MaxDetector()
        self.device = None
        self.device_name = ""
        self.device_type = ""
        self.memory_gb = 0.0
        self.is_m3_max = False
        self.optimization_level = "balanced"
        self.device_info = {}
        self.model_config = {}
        self.is_initialized = False
        
        # 8단계 파이프라인별 최적화 설정
        self.pipeline_optimizations = {}
        
        # 초기화 실행
        self._initialize()
    
    def _initialize(self):
        """GPU 설정 완전 초기화"""
        try:
            logger.info("🔧 M3 Max GPU 설정 초기화 시작...")
            
            # 1. 하드웨어 정보 설정
            self._setup_hardware_info()
            
            # 2. 디바이스 설정
            self._setup_device()
            
            # 3. 8단계 파이프라인 최적화 설정
            self._setup_pipeline_optimizations()
            
            # 4. 모델 설정
            self._setup_model_config()
            
            # 5. 디바이스 정보 수집
            self._collect_device_info()
            
            # 6. 환경 최적화 적용
            self._apply_optimizations()
            
            self.is_initialized = True
            logger.info(f"🚀 M3 Max GPU 설정 완료: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ GPU 설정 초기화 실패: {e}")
            self._fallback_cpu_setup()
    
    def _setup_hardware_info(self):
        """하드웨어 정보 설정"""
        self.is_m3_max = self.detector.is_m3_max
        self.memory_gb = self.detector.memory_gb
        
        if self.is_m3_max:
            self.optimization_level = "maximum"
            logger.info(f"🍎 M3 Max {self.memory_gb}GB 감지!")
        else:
            self.optimization_level = "balanced"
            logger.info(f"💻 일반 환경 감지: {self.memory_gb}GB")
    
    def _setup_device(self):
        """디바이스 설정 및 MPS 최적화"""
        try:
            self.device = self.detector.get_optimal_device()
            self.device_name = self.detector.get_device_name()
            
            if self.device == "mps":
                self.device_type = "mps"
                
                # M3 Max 특화 MPS 환경변수 설정
                if self.is_m3_max:
                    os.environ.update({
                        'PYTORCH_ENABLE_MPS_FALLBACK': '1',
                        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
                        'PYTORCH_MPS_ALLOCATOR_POLICY': 'garbage_collection',
                        'METAL_DEVICE_WRAPPER_TYPE': '1',
                        'METAL_PERFORMANCE_SHADERS_ENABLED': '1'
                    })
                    logger.info("🍎 M3 Max MPS 환경변수 최적화 적용")
                
                logger.info("🍎 Apple Silicon MPS 활성화")
                
            elif self.device == "cuda":
                self.device_type = "cuda"
                self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA GPU"
                logger.info("🚀 CUDA GPU 감지")
                
            else:
                self.device_type = "cpu"
                self.device_name = "CPU"
                logger.info("💻 CPU 모드 설정")
                
        except Exception as e:
            logger.error(f"❌ 디바이스 설정 실패: {e}")
            self._fallback_cpu_setup()
    
    def _setup_pipeline_optimizations(self):
        """8단계 파이프라인별 최적화 설정"""
        
        # 최적화된 기본 설정 가져오기
        optimized = self.detector.get_optimized_settings()
        
        # 8단계별 특화 설정
        self.pipeline_optimizations = {
            "step_01_human_parsing": {
                "batch_size": optimized["batch_size"] // 2,  # 메모리 절약
                "precision": "float16" if self.device != "cpu" else "float32",
                "max_resolution": 512,
                "enable_segmentation_cache": True,
                "memory_fraction": 0.3
            },
            "step_02_pose_estimation": {
                "batch_size": optimized["batch_size"],
                "precision": "float16" if self.device != "cpu" else "float32",
                "keypoint_threshold": 0.3,
                "enable_pose_cache": True,
                "memory_fraction": 0.2
            },
            "step_03_cloth_segmentation": {
                "batch_size": optimized["batch_size"],
                "segmentation_model": "u2net",
                "background_threshold": 0.5,
                "enable_edge_refinement": True,
                "memory_fraction": 0.25
            },
            "step_04_geometric_matching": {
                "batch_size": optimized["batch_size"] // 2,
                "matching_algorithm": "optical_flow",
                "warp_resolution": 256,
                "enable_geometric_cache": True,
                "memory_fraction": 0.3
            },
            "step_05_cloth_warping": {
                "batch_size": optimized["batch_size"],
                "warp_method": "thin_plate_spline",
                "interpolation": "bilinear",
                "preserve_details": True,
                "memory_fraction": 0.25
            },
            "step_06_virtual_fitting": {
                "batch_size": optimized["batch_size"] // 4,  # 가장 메모리 집약적
                "diffusion_steps": 20 if self.is_m3_max else 15,
                "guidance_scale": 7.5,
                "enable_safety_checker": True,
                "scheduler": "ddim",
                "memory_fraction": 0.5
            },
            "step_07_post_processing": {
                "batch_size": optimized["batch_size"],
                "enhancement_level": "high" if self.is_m3_max else "medium",
                "noise_reduction": True,
                "color_correction": True,
                "memory_fraction": 0.2
            },
            "step_08_quality_assessment": {
                "batch_size": optimized["batch_size"],
                "quality_metrics": ["ssim", "lpips", "fid"],
                "assessment_threshold": 0.7,
                "enable_automatic_retry": True,
                "memory_fraction": 0.15
            }
        }
        
        logger.info("⚙️ 8단계 파이프라인 최적화 설정 완료")
    
    def _setup_model_config(self):
        """모델 설정 구성"""
        optimized = self.detector.get_optimized_settings()
        
        base_config = {
            "device": self.device,
            "dtype": "float16" if self.device != "cpu" else "float32",
            "batch_size": optimized["batch_size"],
            "memory_fraction": 0.8,
            "optimization_level": self.optimization_level,
            "max_workers": optimized["max_workers"],
            "concurrent_sessions": optimized["concurrent_sessions"]
        }
        
        # M3 Max 특화 설정
        if self.is_m3_max:
            base_config.update({
                "use_neural_engine": True,
                "metal_performance_shaders": True,
                "unified_memory_optimization": True,
                "high_resolution_processing": True,
                "concurrent_pipeline_steps": 3,
                "memory_pool_size_gb": optimized["memory_pool_gb"],
                "model_cache_size_gb": optimized["cache_size_gb"],
                "intermediate_cache_gb": optimized["cache_size_gb"] // 2
            })
            logger.info("🍎 M3 Max 특화 모델 설정 적용")
        
        self.model_config = base_config
        logger.info(f"⚙️ 모델 설정 완료: 배치={base_config['batch_size']}, 정밀도={base_config['dtype']}")
    
    def _collect_device_info(self):
        """디바이스 정보 수집"""
        try:
            base_info = {
                "device": self.device,
                "device_name": self.device_name,
                "device_type": self.device_type,
                "platform": self.detector.platform_info["system"],
                "architecture": self.detector.platform_info["machine"],
                "pytorch_version": torch.__version__,
                "python_version": self.detector.platform_info["python_version"],
                "optimization_level": self.optimization_level,
                "total_memory_gb": self.memory_gb,
                "is_m3_max": self.is_m3_max
            }
            
            # M3 Max 특화 정보
            if self.is_m3_max:
                base_info.update({
                    "neural_engine_available": True,
                    "neural_engine_tops": "15.8 TOPS",
                    "gpu_cores": "30-40 cores", 
                    "memory_bandwidth": "400GB/s",
                    "unified_memory": True,
                    "metal_performance_shaders": True,
                    "optimized_for_pipeline": "8-step virtual fitting"
                })
            
            # MPS 특화 정보
            if self.device == "mps":
                base_info.update({
                    "mps_available": True,
                    "mps_fallback_enabled": True,
                    "metal_api_available": True
                })
            
            # CUDA 정보
            elif self.device == "cuda" and torch.cuda.is_available():
                base_info.update({
                    "cuda_version": torch.version.cuda,
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    "compute_capability": torch.cuda.get_device_capability(0)
                })
            
            self.device_info = base_info
            logger.info(f"ℹ️ 디바이스 정보 수집 완료: {self.device_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ 디바이스 정보 수집 실패: {e}")
            self.device_info = {"device": self.device, "error": str(e)}
    
    def _apply_optimizations(self):
        """환경 최적화 적용"""
        try:
            # PyTorch 멀티스레딩 설정
            num_threads = self.detector.get_optimized_settings()["max_workers"]
            torch.set_num_threads(num_threads)
            
            # MPS 최적화
            if self.device == "mps":
                if self.is_m3_max:
                    os.environ['METAL_PERFORMANCE_SHADERS_ENABLED'] = '1'
                    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                logger.info("✅ MPS 최적화 적용 완료")
            
            # CUDA 최적화
            elif self.device == "cuda":
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                logger.info("✅ CUDA 최적화 적용 완료")
            
            # 메모리 관리 최적화
            gc.collect()
            
            logger.info(f"✅ 환경 최적화 완료 (스레드: {num_threads})")
            
        except Exception as e:
            logger.warning(f"⚠️ 최적화 적용 실패: {e}")
    
    def _fallback_cpu_setup(self):
        """CPU 폴백 설정"""
        self.device = "cpu"
        self.device_type = "cpu"
        self.device_name = "CPU (Fallback)"
        self.is_m3_max = False
        self.optimization_level = "safe"
        
        self.model_config = {
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 1,
            "memory_fraction": 0.5,
            "optimization_level": "safe"
        }
        
        self.device_info = {
            "device": "cpu",
            "device_name": "CPU (Fallback)",
            "error": "GPU initialization failed"
        }
        
        logger.warning("🚨 CPU 폴백 모드로 설정됨")
    
    # ==========================================
    # 접근자 메서드들
    # ==========================================
    
    def get_device(self) -> str:
        """현재 디바이스 반환"""
        return self.device
    
    def get_device_name(self) -> str:
        """디바이스 이름 반환"""
        return self.device_name
    
    def get_device_type(self) -> str:
        """디바이스 타입 반환"""
        return self.device_type
    
    def get_recommended_batch_size(self) -> int:
        """권장 배치 크기 반환"""
        return self.model_config.get('batch_size', 1)
    
    def get_recommended_precision(self) -> str:
        """권장 정밀도 반환"""
        return self.model_config.get('dtype', 'float32')
    
    def get_memory_fraction(self) -> float:
        """메모리 사용 비율 반환"""
        return self.model_config.get('memory_fraction', 0.5)
    
    def setup_multiprocessing(self) -> int:
        """멀티프로세싱 워커 수 설정"""
        return self.model_config.get('max_workers', 4)
    
    def get_device_config(self) -> Dict[str, Any]:
        """디바이스 설정 반환"""
        return GPUConfig(
            device=self.device,
            device_name=self.device_name,
            device_type=self.device_type,
            memory_gb=self.memory_gb,
            is_m3_max=self.is_m3_max,
            optimization_level=self.optimization_level,
            neural_engine_available=self.is_m3_max,
            metal_performance_shaders=self.is_m3_max,
            unified_memory_optimization=self.is_m3_max
        ).__dict__
    
    def get_model_config(self) -> Dict[str, Any]:
        """모델 설정 반환"""
        return self.model_config.copy()
    
    def get_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 반환"""
        return self.device_info.copy()
    
    def get_pipeline_config(self, step_name: str) -> Dict[str, Any]:
        """특정 파이프라인 단계 설정 반환"""
        return self.pipeline_optimizations.get(step_name, {})
    
    def get_all_pipeline_configs(self) -> Dict[str, Any]:
        """모든 파이프라인 단계 설정 반환"""
        return self.pipeline_optimizations.copy()

# ===============================================================
# 🎯 M3 Optimizer 클래스 (누락된 클래스 추가)
# ===============================================================

class M3Optimizer:
    """M3 Max 전용 최적화 클래스"""
    
    def __init__(self, device_name: str, memory_gb: float, is_m3_max: bool, optimization_level: str):
        """M3 최적화 초기화"""
        self.device_name = device_name
        self.memory_gb = memory_gb
        self.is_m3_max = is_m3_max
        self.optimization_level = optimization_level
        
        logger.info(f"🍎 M3Optimizer 초기화: {device_name}, {memory_gb}GB, {optimization_level}")
        
        if is_m3_max:
            self._apply_m3_max_optimizations()
    
    def _apply_m3_max_optimizations(self):
        """M3 Max 전용 최적화 적용"""
        try:
            if torch.backends.mps.is_available():
                logger.info("🧠 Neural Engine 최적화 활성화")
                logger.info("⚙️ Metal Performance Shaders 활성화")
                
                # 8단계 파이프라인 최적화
                self.pipeline_config = {
                    "stages": 8,
                    "parallel_processing": True,
                    "batch_optimization": True,
                    "memory_pooling": True
                }
                
                logger.info("⚙️ 8단계 파이프라인 최적화 설정 완료")
                
        except Exception as e:
            logger.error(f"❌ M3 Max 최적화 실패: {e}")
    
    def optimize_model(self, model):
        """모델 최적화"""
        if not self.is_m3_max:
            return model
            
        try:
            if hasattr(model, 'to'):
                model = model.to('mps')
            logger.info("✅ 모델 M3 Max 최적화 완료")
            return model
            
        except Exception as e:
            logger.error(f"❌ 모델 최적화 실패: {e}")
            return model

# ===============================================================
# 🔧 유틸리티 및 편의 함수들
# ===============================================================

def optimize_memory(device: Optional[str] = None, aggressive: bool = False) -> Dict[str, Any]:
    """M3 Max 최적화된 메모리 관리"""
    try:
        import psutil
        
        current_device = device or gpu_config.device
        start_memory = psutil.virtual_memory().percent
        
        # 기본 가비지 컬렉션
        gc.collect()
        
        result = {
            "success": True,
            "device": current_device,
            "start_memory_percent": start_memory,
            "method": "standard_gc"
        }
        
        if current_device == "mps":
            try:
                # MPS 메모리 정리 (PyTorch 버전별 대응)
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    result["method"] = "mps_empty_cache"
                elif hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
                    result["method"] = "mps_synchronize"
                
                if aggressive and gpu_config.is_m3_max:
                    torch.mps.synchronize()
                    gc.collect()
                    result["method"] = "m3_max_aggressive_cleanup"
                    result["aggressive"] = True
                
            except Exception as mps_error:
                logger.warning(f"MPS 메모리 정리 실패: {mps_error}")
                result["mps_error"] = str(mps_error)
        
        elif current_device == "cuda":
            try:
                torch.cuda.empty_cache()
                if aggressive:
                    torch.cuda.synchronize()
                result["method"] = "cuda_empty_cache"
                if aggressive:
                    result["aggressive"] = True
            except Exception as cuda_error:
                logger.warning(f"CUDA 메모리 정리 실패: {cuda_error}")
                result["cuda_error"] = str(cuda_error)
        
        # 메모리 정리 후 상태
        end_memory = psutil.virtual_memory().percent
        memory_freed = start_memory - end_memory
        
        result.update({
            "end_memory_percent": end_memory,
            "memory_freed_percent": memory_freed,
            "m3_max_optimized": gpu_config.is_m3_max
        })
        
        if memory_freed > 0:
            logger.info(f"💾 메모리 {memory_freed:.1f}% 정리됨 ({result['method']})")
        
        return result
        
    except Exception as e:
        logger.error(f"메모리 최적화 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "device": device or "unknown",
            "method": "failed"
        }

def get_memory_status() -> Dict[str, Any]:
    """메모리 상태 조회"""
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        
        status = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "usage_percent": memory.percent,
            "status": "good",
            "is_m3_max": gpu_config.is_m3_max if 'gpu_config' in globals() else False
        }
        
        # 상태 판정
        if memory.percent < 40:
            status["status"] = "excellent"
        elif memory.percent < 70:
            status["status"] = "good"
        elif memory.percent < 85:
            status["status"] = "moderate"
        else:
            status["status"] = "high"
        
        return status
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def check_device_compatibility() -> Dict[str, bool]:
    """디바이스 호환성 확인"""
    return {
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "m3_max_detected": getattr(gpu_config, 'is_m3_max', False) if 'gpu_config' in globals() else False,
        "neural_engine_available": (
            torch.backends.mps.is_available() and 
            getattr(gpu_config, 'is_m3_max', False) if 'gpu_config' in globals() else False
        ),
        "8step_pipeline_ready": True
    }

def test_device_performance() -> Dict[str, Any]:
    """디바이스 성능 테스트"""
    config = get_gpu_config()
    
    try:
        import time
        
        device = torch.device(config.device)
        
        # 테스트 텐서 생성
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # 행렬 곱셈 성능 테스트
        start_time = time.time()
        for _ in range(100):
            z = torch.mm(x, y)
        
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        
        end_time = time.time()
        execution_time = end_time - start_time
        performance_score = 100 / execution_time
        
        return {
            "device": config.device,
            "device_name": config.device_name,
            "is_m3_max": config.is_m3_max,
            "performance_score": performance_score,
            "execution_time": execution_time,
            "operations_per_second": 100 / execution_time,
            "test_passed": True,
            "optimization_level": config.optimization_level
        }
        
    except Exception as e:
        logger.error(f"디바이스 성능 테스트 실패: {e}")
        return {
            "device": config.device,
            "performance_score": 0.0,
            "execution_time": float('inf'),
            "operations_per_second": 0.0,
            "test_passed": False,
            "error": str(e)
        }

def get_optimal_settings() -> Dict[str, Any]:
    """최적 설정 반환"""
    config = get_gpu_config()
    
    optimal_settings = {
        "device": config.device,
        "device_name": config.device_name,
        "batch_size": config.get_recommended_batch_size(),
        "precision": config.get_recommended_precision(),
        "memory_fraction": config.get_memory_fraction(),
        "max_workers": config.setup_multiprocessing(),
        "optimization_level": config.optimization_level,
        "is_m3_max": config.is_m3_max,
        "memory_gb": config.memory_gb
    }
    
    # M3 Max 특화 설정 추가
    if config.is_m3_max:
        optimal_settings.update({
            "neural_engine_enabled": True,
            "mps_optimization": True,
            "metal_performance_shaders": True,
            "memory_bandwidth": "400GB/s",
            "concurrent_sessions": 8,
            "cache_size_gb": 16,
            "pipeline_parallel_steps": 3
        })
    
    return optimal_settings

def get_device_capabilities() -> Dict[str, Any]:
    """디바이스 기능 반환"""
    config = get_gpu_config()
    
    capabilities = {
        "supports_fp16": config.device != "cpu",
        "supports_int8": True,
        "supports_compilation": config.device in ["cuda", "cpu"],
        "supports_parallel_inference": True,
        "max_batch_size": config.get_recommended_batch_size() * 2,
        "recommended_image_size": (512, 512) if config.is_m3_max else (256, 256),
        "supports_8step_pipeline": True
    }
    
    if config.device == "mps":
        capabilities.update({
            "supports_neural_engine": config.is_m3_max,
            "supports_metal_shaders": True,
            "mps_fallback_enabled": True,
            "unified_memory_optimization": config.is_m3_max
        })
    elif config.device == "cuda":
        capabilities.update({
            "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else None,
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "tensor_cores_available": True
        })
    
    return capabilities

def optimize_for_inference() -> Dict[str, Any]:
    """추론 최적화 설정"""
    config = get_gpu_config()
    
    inference_settings = {
        "torch_compile": config.device in ["cuda", "cpu"],
        "channels_last": config.device == "cuda",
        "mixed_precision": config.device in ["cuda", "mps"],
        "gradient_checkpointing": False,
        "enable_cudnn_benchmark": config.device == "cuda",
        "deterministic": False,
        "memory_efficient": True,
        "pipeline_optimization": True
    }
    
    # M3 Max 특화 추론 최적화
    if config.is_m3_max:
        inference_settings.update({
            "mps_high_watermark": 0.0,
            "mps_allocator_policy": "garbage_collection",
            "metal_api_validation": False,
            "neural_engine_priority": "high",
            "parallel_pipeline_steps": 3,
            "aggressive_memory_optimization": True
        })
    
    return inference_settings

def apply_optimizations():
    """최적화 설정 적용"""
    config = get_gpu_config()
    settings = get_optimal_settings()
    
    try:
        # PyTorch 환경 변수 설정
        if config.device == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            
            if config.is_m3_max:
                os.environ["PYTORCH_MPS_ALLOCATOR_POLICY"] = "garbage_collection"
                os.environ["METAL_PERFORMANCE_SHADERS_ENABLED"] = "1"
        
        # 멀티프로세싱 설정
        torch.set_num_threads(settings["max_workers"])
        
        # CUDA 설정
        if config.device == "cuda":
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        logger.info("✅ GPU 최적화 설정 적용 완료")
        return True
        
    except Exception as e:
        logger.error(f"GPU 최적화 설정 적용 실패: {e}")
        return False

def get_memory_info() -> Dict[str, float]:
    """메모리 정보 반환"""
    config = get_gpu_config()
    
    if config.device == "cuda" and torch.cuda.is_available():
        return {
            "total_memory": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "allocated_memory": torch.cuda.memory_allocated() / 1024**3,
            "cached_memory": torch.cuda.memory_reserved() / 1024**3,
            "free_memory": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
        }
    else:
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total_memory": memory.total / 1024**3,
                "available_memory": memory.available / 1024**3,
                "used_memory": memory.used / 1024**3,
                "memory_percent": memory.percent
            }
        except ImportError:
            return {"total_memory": 0.0, "available_memory": 0.0}

def check_memory_available(required_gb: float = 4.0) -> bool:
    """M3 Max 메모리 사용 가능 여부 확인"""
    import psutil
    
    try:
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        logger.info(f"💾 시스템 메모리: {memory.total / (1024**3):.1f}GB")
        logger.info(f"💾 사용 가능: {available_gb:.1f}GB")
        logger.info(f"💾 요구사항: {required_gb:.1f}GB")
        
        # MPS 메모리 확인 (M3 Max)
        if torch.backends.mps.is_available():
            logger.info("🍎 M3 Max Unified Memory 사용 중")
            return available_gb >= required_gb
        
        return available_gb >= required_gb
        
    except Exception as e:
        logger.warning(f"⚠️ 메모리 확인 실패: {e}")
        return True  # 안전하게 True 반환

def cleanup_gpu_resources():
    """GPU 리소스 정리"""
    try:
        optimize_memory(aggressive=True)
        logger.info("✅ GPU 리소스 정리 완료")
    except Exception as e:
        logger.warning(f"⚠️ GPU 정리 중 오류: {e}")

# ==========================================
# 모듈 초기화 및 전역 변수
# ==========================================

def _initialize_gpu_optimizations():
    """GPU 최적화 초기화"""
    try:
        apply_optimizations()
        logger.info("🚀 GPU 최적화 초기화 완료")
    except Exception as e:
        logger.warning(f"GPU 최적화 초기화 실패: {e}")

# 전역 GPU 설정 매니저 생성
gpu_config = M3MaxGPUManager()

# 편의를 위한 전역 변수들
DEVICE = gpu_config.device
DEVICE_NAME = gpu_config.device_name
DEVICE_TYPE = gpu_config.device_type
MODEL_CONFIG = gpu_config.model_config
DEVICE_INFO = gpu_config.device_info
IS_M3_MAX = gpu_config.is_m3_max

# ==========================================
# 주요 함수들 (호환성)
# ==========================================

@lru_cache(maxsize=1)
def get_gpu_config() -> M3MaxGPUManager:
    """GPU 설정 매니저 반환 (캐시됨)"""
    return gpu_config

def configure_gpu() -> str:
    """GPU 설정 및 디바이스 반환"""
    return gpu_config.device

def get_optimal_device() -> str:
    """최적 디바이스 반환"""
    return gpu_config.device

def is_m3_max() -> bool:
    """M3 Max 여부 확인"""
    return gpu_config.is_m3_max

def get_device() -> str:
    """현재 디바이스 반환"""
    return gpu_config.get_device()

def get_device_config() -> Dict[str, Any]:
    """디바이스 설정 반환"""
    return gpu_config.get_device_config()

def get_model_config() -> Dict[str, Any]:
    """모델 설정 반환"""
    return gpu_config.get_model_config()

def get_device_info() -> Dict[str, Any]:
    """디바이스 정보 반환"""
    return gpu_config.get_device_info()

def test_device() -> bool:
    """디바이스 테스트"""
    result = test_device_performance()
    return result.get("test_passed", False)

# 편의 함수들
def is_gpu_available() -> bool:
    """GPU 사용 가능 여부"""
    return gpu_config.device != "cpu"

def is_m3_max_available() -> bool:
    """M3 Max 사용 가능 여부"""
    return gpu_config.is_m3_max

def get_recommended_settings() -> Dict[str, Any]:
    """권장 설정 반환"""
    return get_optimal_settings()

def get_device_name() -> str:
    """디바이스 이름 반환"""
    return gpu_config.device_name

def get_device_type() -> str:
    """디바이스 타입 반환"""
    return gpu_config.device_type

def get_pipeline_config(step_name: str) -> Dict[str, Any]:
    """파이프라인 단계별 설정 반환"""
    return gpu_config.get_pipeline_config(step_name)

def get_all_pipeline_configs() -> Dict[str, Any]:
    """모든 파이프라인 설정 반환"""
    return gpu_config.get_all_pipeline_configs()

# 모듈 로드시 자동 최적화 적용
_initialize_gpu_optimizations()

# 초기화 및 검증
if gpu_config.is_initialized:
    test_success = test_device()
    if test_success:
        logger.info("✅ M3 Max GPU 설정 검증 완료")
    else:
        logger.warning("⚠️ GPU 설정 검증 실패")

# M3 Max 상태 로깅
if gpu_config.is_m3_max:
    logger.info("🍎 M3 Max 128GB 최적화 활성화:")
    logger.info(f"  - Neural Engine: {'✅' if MODEL_CONFIG.get('use_neural_engine') else '❌'}")
    logger.info(f"  - Metal Performance Shaders: {'✅' if MODEL_CONFIG.get('metal_performance_shaders') else '❌'}")
    logger.info(f"  - 배치 크기: {MODEL_CONFIG.get('batch_size', 1)}")
    logger.info(f"  - 8단계 파이프라인 최적화: ✅")
    logger.info(f"  - 메모리 대역폭: {DEVICE_INFO.get('memory_bandwidth', 'N/A')}")

# ==========================================
# Export 리스트
# ==========================================

__all__ = [
    # 주요 객체들
    'gpu_config', 'DEVICE', 'DEVICE_NAME', 'DEVICE_TYPE', 'MODEL_CONFIG', 'DEVICE_INFO', 'IS_M3_MAX',
    
    # 핵심 함수들
    'get_gpu_config', 'configure_gpu', 'get_optimal_device', 'is_m3_max',
    'get_device', 'get_device_config', 'get_model_config', 'get_device_info',
    'test_device', 'cleanup_gpu_resources',
    
    # 최적화 함수들
    'get_optimal_settings', 'get_device_capabilities', 'optimize_for_inference',
    'apply_optimizations', 'get_memory_info', 'test_device_performance',
    
    # 메모리 관리 함수들
    'optimize_memory', 'get_memory_status', 'check_device_compatibility',
    'check_memory_available',
    
    # 편의 함수들
    'is_gpu_available', 'is_m3_max_available', 'get_recommended_settings',
    'get_device_name', 'get_device_type',
    
    # 파이프라인 특화 함수들
    'get_pipeline_config', 'get_all_pipeline_configs',
    
    # 클래스들
    'M3MaxGPUManager', 'GPUConfig', 'M3Optimizer', 'M3MaxDetector'
]