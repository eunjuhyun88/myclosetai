# backend/app/core/gpu_config.py
"""
🍎 MyCloset AI - 완전한 GPU 설정 매니저 (M3 Max 최적화) - GPUConfig 클래스 추가
✅ GPUConfig 클래스 정의 추가 (import 오류 해결)
✅ PyTorch 2.6+ MPS 호환성 완전 해결
✅ torch.mps.empty_cache() 오류 완전 수정
✅ Float16 호환성 문제 해결
✅ 메모리 관리 최적화
✅ 안전한 폴백 메커니즘
"""

import os
import gc
import logging
import platform
import subprocess
from typing import Dict, Any, Optional, Union, List
from functools import lru_cache
import time

# 조건부 import (안전한 처리)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 로깅 최적화
logger = logging.getLogger(__name__)

# ===============================================================
# 🔥 GPUConfig 클래스 정의 (import 오류 해결)
# ===============================================================

class GPUConfig:
    """🔥 GPUConfig 클래스 - import 오류 완전 해결"""
    
    def __init__(self, device: str = "auto", optimization_level: str = "balanced"):
        """GPUConfig 초기화"""
        self.device = self._detect_device(device)
        self.optimization_level = optimization_level
        self.is_m3_max = self._detect_m3_max()
        self.memory_gb = self._get_memory_gb()
        self.float_compatibility_mode = True
        
        # 기본 설정
        self.batch_size = self._calculate_batch_size()
        self.max_workers = self._calculate_max_workers()
        self.memory_fraction = 0.7 if self.is_m3_max else 0.6
        
        # M3 Max 최적화 적용
        if self.is_m3_max:
            self._apply_m3_max_optimizations()
    
    def _detect_device(self, device: str) -> str:
        """디바이스 자동 감지"""
        if device != "auto":
            return device
            
        if not TORCH_AVAILABLE:
            return "cpu"
            
        # M3 Max MPS 우선
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 감지"""
        try:
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                return False
                
            # 메모리 기반 감지
            if PSUTIL_AVAILABLE:
                memory_gb = psutil.virtual_memory().total / (1024**3)
                if memory_gb >= 90:  # 90GB 이상이면 M3 Max
                    return True
            
            return False
        except:
            return False
    
    def _get_memory_gb(self) -> float:
        """시스템 메모리 감지"""
        try:
            if PSUTIL_AVAILABLE:
                return round(psutil.virtual_memory().total / (1024**3), 1)
            return 16.0
        except:
            return 16.0
    
    def _calculate_batch_size(self) -> int:
        """최적 배치 크기 계산"""
        if self.is_m3_max and self.memory_gb >= 128:
            return 6  # 안정성 우선
        elif self.memory_gb >= 64:
            return 4
        elif self.memory_gb >= 32:
            return 2
        else:
            return 1
    
    def _calculate_max_workers(self) -> int:
        """최적 워커 수 계산"""
        try:
            cpu_count = os.cpu_count() or 8
            if self.is_m3_max:
                return min(12, cpu_count)
            else:
                return min(6, cpu_count)
        except:
            return 4
    
    def _apply_m3_max_optimizations(self):
        """M3 Max 최적화 적용"""
        try:
            if TORCH_AVAILABLE:
                # PyTorch 환경 변수 설정
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                os.environ['OMP_NUM_THREADS'] = '16'
                
                # M3 Max 특화 설정
                if self.is_m3_max:
                    os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
                    os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'
        except:
            pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """딕셔너리 스타일 접근"""
        return getattr(self, key, default)
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """🚀 메모리 정리 - MPS 호환성 완전 수정"""
        try:
            start_time = time.time()
            
            # 기본 가비지 컬렉션
            gc.collect()
            
            result = {
                "success": True,
                "device": self.device,
                "method": "standard_gc",
                "aggressive": aggressive,
                "duration": 0.0
            }
            
            if not TORCH_AVAILABLE:
                result["duration"] = time.time() - start_time
                return result
            
            # MPS 메모리 정리 (PyTorch 2.6+ 호환)
            if self.device == "mps":
                try:
                    mps_cleaned = False
                    
                    # 방법 1: torch.mps.empty_cache() (최신)
                    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                        try:
                            torch.mps.empty_cache()
                            result["method"] = "mps_empty_cache_v2"
                            mps_cleaned = True
                        except:
                            pass
                    
                    # 방법 2: torch.mps.synchronize() (대안)
                    if not mps_cleaned and hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
                        try:
                            torch.mps.synchronize()
                            result["method"] = "mps_synchronize"
                            mps_cleaned = True
                        except:
                            pass
                    
                    # 방법 3: torch.backends.mps.empty_cache() (이전 버전)
                    if not mps_cleaned and hasattr(torch.backends, 'mps'):
                        if hasattr(torch.backends.mps, 'empty_cache'):
                            try:
                                torch.backends.mps.empty_cache()
                                result["method"] = "mps_backends_empty_cache"
                                mps_cleaned = True
                            except:
                                pass
                    
                    if not mps_cleaned:
                        result["method"] = "mps_gc_only"
                
                except Exception as e:
                    result["warning"] = f"MPS 메모리 정리 오류: {str(e)[:100]}"
                    result["method"] = "mps_error_fallback"
            
            # CUDA 메모리 정리
            elif self.device == "cuda":
                try:
                    if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                        result["method"] = "cuda_empty_cache"
                        
                        if aggressive and hasattr(torch.cuda, 'synchronize'):
                            torch.cuda.synchronize()
                            result["method"] = "cuda_aggressive_cleanup"
                except:
                    result["method"] = "cuda_gc_only"
            
            result["duration"] = time.time() - start_time
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)[:100],
                "device": self.device,
                "duration": time.time() - start_time if 'start_time' in locals() else 0.0
            }

# ===============================================================
# 🍎 M3 Max 감지 및 하드웨어 정보
# ===============================================================

class HardwareDetector:
    """하드웨어 정보 감지 클래스"""
    
    def __init__(self):
        self._cache = {}
        self.system_info = self._get_system_info()
        self.is_m3_max = self._detect_m3_max()
        self.memory_gb = self._get_memory_gb()
        self.cpu_cores = self._get_cpu_cores()
        self.gpu_info = self._get_gpu_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 수집"""
        if 'system_info' in self._cache:
            return self._cache['system_info']
            
        info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__ if TORCH_AVAILABLE else "not_available"
        }
        self._cache['system_info'] = info
        return info
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 정밀 감지"""
        if 'm3_max' in self._cache:
            return self._cache['m3_max']
            
        try:
            # macOS ARM64만 체크
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                self._cache['m3_max'] = False
                return False
            
            # 메모리 기반 감지 (M3 Max는 96GB 또는 128GB)
            if PSUTIL_AVAILABLE:
                total_memory = psutil.virtual_memory().total / (1024**3)
                if total_memory >= 90:
                    self._cache['m3_max'] = True
                    return True
            
            # CPU 코어 수 기반 감지
            if PSUTIL_AVAILABLE:
                cpu_count = psutil.cpu_count(logical=False)
                if cpu_count and cpu_count >= 12:
                    self._cache['m3_max'] = True
                    return True
                
            self._cache['m3_max'] = False
            return False
            
        except:
            self._cache['m3_max'] = False
            return False
    
    def _get_memory_gb(self) -> float:
        """메모리 용량 감지"""
        if 'memory_gb' in self._cache:
            return self._cache['memory_gb']
            
        try:
            if PSUTIL_AVAILABLE:
                memory = round(psutil.virtual_memory().total / (1024**3), 1)
            else:
                memory = 16.0
            self._cache['memory_gb'] = memory
            return memory
        except:
            self._cache['memory_gb'] = 16.0
            return 16.0
    
    def _get_cpu_cores(self) -> int:
        """CPU 코어 수 감지"""
        if 'cpu_cores' in self._cache:
            return self._cache['cpu_cores']
            
        try:
            if PSUTIL_AVAILABLE:
                cores = psutil.cpu_count(logical=True) or 8
            else:
                cores = os.cpu_count() or 8
            self._cache['cpu_cores'] = cores
            return cores
        except:
            self._cache['cpu_cores'] = 8
            return 8
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """GPU 정보 수집"""
        if 'gpu_info' in self._cache:
            return self._cache['gpu_info']
            
        gpu_info = {
            "device": "cpu",
            "name": "CPU",
            "memory_gb": self.memory_gb,
            "available": True,
            "backend": "CPU"
        }
        
        if not TORCH_AVAILABLE:
            self._cache['gpu_info'] = gpu_info
            return gpu_info
        
        try:
            # MPS 지원 확인
            if (hasattr(torch.backends, 'mps') and 
                hasattr(torch.backends.mps, 'is_available') and 
                torch.backends.mps.is_available()):
                
                gpu_info.update({
                    "device": "mps",
                    "name": "Apple M3 Max" if self.is_m3_max else "Apple Silicon",
                    "memory_gb": self.memory_gb,
                    "available": True,
                    "backend": "Metal Performance Shaders"
                })
            
            # CUDA 지원 확인
            elif hasattr(torch, 'cuda') and torch.cuda.is_available():
                try:
                    gpu_props = torch.cuda.get_device_properties(0)
                    gpu_info.update({
                        "device": "cuda",
                        "name": gpu_props.name,
                        "memory_gb": gpu_props.total_memory / (1024**3),
                        "available": True,
                        "backend": "CUDA"
                    })
                except:
                    pass
        
        except:
            pass
        
        self._cache['gpu_info'] = gpu_info
        return gpu_info

# ===============================================================
# 🎯 완전한 GPU 설정 매니저
# ===============================================================

class GPUManager:
    """완전한 GPU 설정 매니저"""
    
    def __init__(self):
        """GPU 매니저 초기화"""
        # 하드웨어 감지
        self.hardware = HardwareDetector()
        
        # 기본 속성 설정
        self.device = self.hardware.gpu_info["device"]
        self.device_name = self.hardware.gpu_info["name"]
        self.device_type = self.device
        self.memory_gb = self.hardware.memory_gb
        self.is_m3_max = self.hardware.is_m3_max
        self.is_initialized = False
        
        # 최적화 설정
        self.optimization_settings = self._calculate_optimization_settings()
        self.model_config = self._create_model_config()
        self.device_info = self._collect_device_info()
        self.pipeline_optimizations = self._setup_pipeline_optimizations()
        
        # 환경 최적화 적용
        self._apply_optimizations()
        
        self.is_initialized = True
    
    def _calculate_optimization_settings(self) -> Dict[str, Any]:
        """최적화 설정 계산"""
        if self.is_m3_max:
            return {
                "batch_size": 6 if self.memory_gb >= 120 else 4,
                "max_workers": min(12, self.hardware.cpu_cores),
                "concurrent_sessions": 8 if self.memory_gb >= 120 else 6,
                "memory_pool_gb": min(48, self.memory_gb // 3),
                "cache_size_gb": min(24, self.memory_gb // 5),
                "quality_level": "high",
                "enable_neural_engine": True,
                "enable_mps": True,
                "optimization_level": "balanced",
                "fp16_enabled": False,  # Float32 사용 (안정성)
                "memory_fraction": 0.75,
                "high_resolution_processing": True,
                "unified_memory_optimization": True,
                "metal_performance_shaders": True,
                "pipeline_parallelism": True,
                "step_caching": True,
                "model_preloading": False
            }
        elif self.hardware.system_info["machine"] == "arm64":
            return {
                "batch_size": 3,
                "max_workers": min(6, self.hardware.cpu_cores),
                "concurrent_sessions": 4,
                "memory_pool_gb": min(12, self.memory_gb // 3),
                "cache_size_gb": min(6, self.memory_gb // 5),
                "quality_level": "balanced",
                "enable_neural_engine": False,
                "enable_mps": True,
                "optimization_level": "balanced",
                "fp16_enabled": False,
                "memory_fraction": 0.65,
                "high_resolution_processing": False,
                "unified_memory_optimization": True,
                "metal_performance_shaders": True,
                "pipeline_parallelism": False,
                "step_caching": True,
                "model_preloading": False
            }
        else:
            return {
                "batch_size": 2,
                "max_workers": min(4, self.hardware.cpu_cores),
                "concurrent_sessions": 3,
                "memory_pool_gb": min(6, self.memory_gb // 3),
                "cache_size_gb": min(3, self.memory_gb // 5),
                "quality_level": "balanced",
                "enable_neural_engine": False,
                "enable_mps": False,
                "optimization_level": "safe",
                "fp16_enabled": False,
                "memory_fraction": 0.6,
                "high_resolution_processing": False,
                "unified_memory_optimization": False,
                "metal_performance_shaders": False,
                "pipeline_parallelism": False,
                "step_caching": False,
                "model_preloading": False
            }
    
    def _create_model_config(self) -> Dict[str, Any]:
        """모델 설정 생성"""
        return {
            "device": self.device,
            "dtype": "float32",  # 항상 float32 사용
            "batch_size": self.optimization_settings["batch_size"],
            "max_workers": self.optimization_settings["max_workers"],
            "concurrent_sessions": self.optimization_settings["concurrent_sessions"],
            "memory_fraction": self.optimization_settings["memory_fraction"],
            "optimization_level": self.optimization_settings["optimization_level"],
            "quality_level": self.optimization_settings["quality_level"],
            "enable_caching": self.optimization_settings["step_caching"],
            "enable_preloading": self.optimization_settings["model_preloading"],
            "use_neural_engine": self.optimization_settings["enable_neural_engine"],
            "metal_performance_shaders": self.optimization_settings["metal_performance_shaders"],
            "unified_memory_optimization": self.optimization_settings["unified_memory_optimization"],
            "high_resolution_processing": self.optimization_settings["high_resolution_processing"],
            "memory_pool_size_gb": self.optimization_settings["memory_pool_gb"],
            "model_cache_size_gb": self.optimization_settings["cache_size_gb"],
            "m3_max_optimized": self.is_m3_max,
            "float_compatibility_mode": True
        }
    
    def _collect_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 수집"""
        device_info = {
            "device": self.device,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_level": self.optimization_settings["optimization_level"],
            "pytorch_version": torch.__version__ if TORCH_AVAILABLE else "not_available",
            "system_info": self.hardware.system_info,
            "gpu_info": self.hardware.gpu_info,
            "float_compatibility_mode": True
        }
        
        if self.is_m3_max:
            device_info["m3_max_features"] = {
                "neural_engine_available": True,
                "neural_engine_tops": "15.8 TOPS",
                "gpu_cores": "30-40 cores",
                "memory_bandwidth": "400GB/s",
                "unified_memory": True,
                "metal_performance_shaders": True,
                "optimized_for_ai": True,
                "pipeline_acceleration": True,
                "real_time_processing": True,
                "high_resolution_support": True,
                "float32_optimized": True
            }
        
        return device_info
    
    def _setup_pipeline_optimizations(self) -> Dict[str, Any]:
        """8단계 파이프라인 최적화 설정"""
        base_batch = self.optimization_settings["batch_size"]
        precision = "float32"
        
        if self.is_m3_max:
            return {
                "step_01_human_parsing": {
                    "batch_size": max(1, base_batch // 3),
                    "precision": precision,
                    "max_resolution": 640,
                    "memory_fraction": 0.2,
                    "enable_caching": True,
                    "neural_engine_boost": True,
                    "metal_shader_acceleration": True,
                    "float_compatibility": True
                },
                "step_02_pose_estimation": {
                    "batch_size": max(1, base_batch // 2),
                    "precision": precision,
                    "keypoint_threshold": 0.3,
                    "memory_fraction": 0.18,
                    "enable_caching": True,
                    "high_precision_mode": True,
                    "batch_optimization": True,
                    "float_compatibility": True
                },
                "step_03_cloth_segmentation": {
                    "batch_size": max(1, base_batch // 2),
                    "precision": precision,
                    "background_threshold": 0.5,
                    "memory_fraction": 0.22,
                    "enable_edge_refinement": True,
                    "unified_memory_optimization": True,
                    "parallel_processing": True,
                    "float_compatibility": True
                },
                "step_04_geometric_matching": {
                    "batch_size": max(1, base_batch // 4),
                    "precision": precision,
                    "warp_resolution": 448,
                    "memory_fraction": 0.25,
                    "enable_caching": True,
                    "high_accuracy_mode": True,
                    "gpu_acceleration": True,
                    "float_compatibility": True
                },
                "step_05_cloth_warping": {
                    "batch_size": max(1, base_batch // 2),
                    "precision": precision,
                    "interpolation": "bicubic",
                    "memory_fraction": 0.22,
                    "preserve_details": True,
                    "texture_enhancement": True,
                    "anti_aliasing": True,
                    "float_compatibility": True
                },
                "step_06_virtual_fitting": {
                    "batch_size": 1,
                    "precision": precision,
                    "diffusion_steps": 20,
                    "memory_fraction": 0.4,
                    "scheduler": "ddim",
                    "guidance_scale": 7.5,
                    "high_quality_mode": True,
                    "neural_engine_diffusion": True,
                    "float_compatibility": True
                },
                "step_07_post_processing": {
                    "batch_size": base_batch,
                    "precision": precision,
                    "enhancement_level": "high",
                    "memory_fraction": 0.18,
                    "noise_reduction": True,
                    "detail_preservation": True,
                    "color_correction": True,
                    "float_compatibility": True
                },
                "step_08_quality_assessment": {
                    "batch_size": base_batch,
                    "precision": precision,
                    "quality_metrics": ["ssim", "lpips", "clip_score"],
                    "memory_fraction": 0.12,
                    "assessment_threshold": 0.75,
                    "comprehensive_analysis": True,
                    "real_time_feedback": True,
                    "float_compatibility": True
                }
            }
        else:
            return {
                "step_01_human_parsing": {
                    "batch_size": 1,
                    "precision": precision,
                    "max_resolution": 512,
                    "memory_fraction": 0.3,
                    "enable_caching": False,
                    "float_compatibility": True
                },
                "step_02_pose_estimation": {
                    "batch_size": 1,
                    "precision": precision,
                    "keypoint_threshold": 0.35,
                    "memory_fraction": 0.25,
                    "enable_caching": False,
                    "float_compatibility": True
                },
                "step_03_cloth_segmentation": {
                    "batch_size": 1,
                    "precision": precision,
                    "background_threshold": 0.5,
                    "memory_fraction": 0.3,
                    "enable_edge_refinement": False,
                    "float_compatibility": True
                },
                "step_04_geometric_matching": {
                    "batch_size": 1,
                    "precision": precision,
                    "warp_resolution": 256,
                    "memory_fraction": 0.35,
                    "enable_caching": False,
                    "float_compatibility": True
                },
                "step_05_cloth_warping": {
                    "batch_size": 1,
                    "precision": precision,
                    "interpolation": "bilinear",
                    "memory_fraction": 0.3,
                    "preserve_details": False,
                    "float_compatibility": True
                },
                "step_06_virtual_fitting": {
                    "batch_size": 1,
                    "precision": precision,
                    "diffusion_steps": 15,
                    "memory_fraction": 0.6,
                    "scheduler": "ddim",
                    "guidance_scale": 7.5,
                    "float_compatibility": True
                },
                "step_07_post_processing": {
                    "batch_size": 1,
                    "precision": precision,
                    "enhancement_level": "medium",
                    "memory_fraction": 0.25,
                    "noise_reduction": False,
                    "float_compatibility": True
                },
                "step_08_quality_assessment": {
                    "batch_size": 1,
                    "precision": precision,
                    "quality_metrics": ["ssim", "lpips"],
                    "memory_fraction": 0.2,
                    "assessment_threshold": 0.6,
                    "float_compatibility": True
                }
            }
    
    def _apply_optimizations(self):
        """환경 최적화 적용"""
        try:
            if not TORCH_AVAILABLE:
                return
                
            # PyTorch 스레드 설정
            torch.set_num_threads(self.optimization_settings["max_workers"])
            
            if self.device == "mps":
                # MPS 환경 변수 설정
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                
                # M3 Max 특화 환경 변수
                if self.is_m3_max:
                    try:
                        os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
                        os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'
                        os.environ['METAL_PERFORMANCE_SHADERS_ENABLED'] = '1'
                        os.environ['PYTORCH_MPS_PREFER_METAL'] = '1'
                    except:
                        pass
                
            elif self.device == "cuda":
                try:
                    if hasattr(torch.backends, 'cudnn'):
                        torch.backends.cudnn.enabled = True
                        torch.backends.cudnn.benchmark = True
                        torch.backends.cudnn.deterministic = False
                    
                    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
                    os.environ['CUDA_CACHE_DISABLE'] = '0'
                except:
                    pass
            
            gc.collect()
            
        except:
            pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """딕셔너리 스타일 접근 메서드"""
        attribute_mapping = {
            'device': self.device,
            'device_name': self.device_name,
            'device_type': self.device_type,
            'memory_gb': self.memory_gb,
            'is_m3_max': self.is_m3_max,
            'optimization_level': self.optimization_settings["optimization_level"],
            'is_initialized': self.is_initialized,
            'device_info': self.device_info,
            'model_config': self.model_config,
            'pipeline_optimizations': self.pipeline_optimizations,
            'optimization_settings': self.optimization_settings,
            'pytorch_version': torch.__version__ if TORCH_AVAILABLE else "not_available",
            'batch_size': self.optimization_settings["batch_size"],
            'max_workers': self.optimization_settings["max_workers"],
            'memory_fraction': self.optimization_settings["memory_fraction"],
            'quality_level': self.optimization_settings["quality_level"],
            'float_compatibility_mode': True
        }
        
        if key in attribute_mapping:
            return attribute_mapping[key]
        
        if hasattr(self, 'model_config') and key in self.model_config:
            return self.model_config[key]
        
        if hasattr(self, 'device_info') and key in self.device_info:
            return self.device_info[key]
        
        if hasattr(self, 'pipeline_optimizations') and key in self.pipeline_optimizations:
            return self.pipeline_optimizations[key]
        
        if hasattr(self, 'optimization_settings') and key in self.optimization_settings:
            return self.optimization_settings[key]
        
        if hasattr(self, key):
            return getattr(self, key)
        
        return default
    
    def keys(self) -> List[str]:
        """사용 가능한 키 목록"""
        return [
            'device', 'device_name', 'device_type', 'memory_gb',
            'is_m3_max', 'optimization_level', 'is_initialized',
            'device_info', 'model_config', 'pipeline_optimizations',
            'optimization_settings', 'pytorch_version', 'batch_size',
            'max_workers', 'memory_fraction', 'quality_level',
            'float_compatibility_mode'
        ]
    
    def __getitem__(self, key: str) -> Any:
        """[] 접근자 지원"""
        result = self.get(key)
        if result is None:
            raise KeyError(f"Key '{key}' not found")
        return result
    
    def __contains__(self, key: str) -> bool:
        """in 연산자 지원"""
        return self.get(key) is not None
    
    def get_device(self) -> str:
        """현재 디바이스 반환"""
        return self.device
    
    def get_device_name(self) -> str:
        """디바이스 이름 반환"""
        return self.device_name
    
    def get_device_config(self) -> Dict[str, Any]:
        """디바이스 설정 반환"""
        return {
            "device": self.device,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_level": self.optimization_settings["optimization_level"],
            "neural_engine_available": self.is_m3_max,
            "metal_performance_shaders": self.is_m3_max,
            "unified_memory_optimization": self.optimization_settings["unified_memory_optimization"],
            "high_resolution_processing": self.optimization_settings["high_resolution_processing"],
            "pipeline_parallelism": self.optimization_settings["pipeline_parallelism"],
            "float_compatibility_mode": True
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """모델 설정 반환"""
        return self.model_config.copy()
    
    def get_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 반환"""
        return self.device_info.copy()
    
    def get_pipeline_config(self, step_name: str) -> Dict[str, Any]:
        """특정 파이프라인 단계 설정 반환"""
        return self.pipeline_optimizations.get(step_name, {})
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 정리 - PyTorch 2.6+ MPS 호환성 완전 수정"""
        try:
            start_time = time.time()
            
            gc.collect()
            
            result = {
                "success": True,
                "device": self.device,
                "method": "standard_gc",
                "aggressive": aggressive,
                "duration": 0.0,
                "pytorch_available": TORCH_AVAILABLE
            }
            
            if not TORCH_AVAILABLE:
                result["duration"] = time.time() - start_time
                return result
            
            if self.device == "mps":
                try:
                    mps_cleaned = False
                    
                    # torch.mps.empty_cache() (최신 버전)
                    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                        try:
                            torch.mps.empty_cache()
                            result["method"] = "mps_empty_cache_v2"
                            mps_cleaned = True
                        except:
                            pass
                    
                    # torch.mps.synchronize() (대안)
                    if not mps_cleaned and hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
                        try:
                            torch.mps.synchronize()
                            result["method"] = "mps_synchronize"
                            mps_cleaned = True
                        except:
                            pass
                    
                    # torch.backends.mps.empty_cache() (이전 버전)
                    if not mps_cleaned and hasattr(torch.backends, 'mps'):
                        if hasattr(torch.backends.mps, 'empty_cache'):
                            try:
                                torch.backends.mps.empty_cache()
                                result["method"] = "mps_backends_empty_cache"
                                mps_cleaned = True
                            except:
                                pass
                    
                    if not mps_cleaned:
                        result["method"] = "mps_gc_only"
                        result["info"] = "MPS 메모리 정리 함수를 찾을 수 없어 GC만 실행"
                
                except Exception as e:
                    result["warning"] = f"MPS 메모리 정리 중 오류: {str(e)[:100]}"
                    result["method"] = "mps_error_fallback"
            
            elif self.device == "cuda":
                try:
                    if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                        result["method"] = "cuda_empty_cache"
                        
                        if aggressive and hasattr(torch.cuda, 'synchronize'):
                            torch.cuda.synchronize()
                            result["method"] = "cuda_aggressive_cleanup"
                except:
                    result["method"] = "cuda_gc_only"
            
            if aggressive:
                try:
                    for _ in range(3):
                        gc.collect()
                    result["method"] = f"{result['method']}_aggressive"
                except:
                    pass
            
            result["duration"] = time.time() - start_time
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)[:200],
                "device": self.device,
                "pytorch_available": TORCH_AVAILABLE,
                "duration": time.time() - start_time if 'start_time' in locals() else 0.0
            }

# ===============================================================
# 🔧 유틸리티 함수들
# ===============================================================

def check_memory_available(device: Optional[str] = None, min_gb: float = 1.0) -> Dict[str, Any]:
    """메모리 사용 가능 상태 확인"""
    try:
        if 'gpu_config' not in globals():
            return {
                "device": device or "unknown",
                "error": "GPU config not initialized",
                "is_available": False,
                "min_required_gb": min_gb,
                "timestamp": time.time()
            }
        
        current_device = device or gpu_config.device
        
        result = {
            "device": current_device,
            "min_required_gb": min_gb,
            "timestamp": time.time(),
            "pytorch_version": torch.__version__ if TORCH_AVAILABLE else "not_available",
            "is_m3_max": getattr(gpu_config, 'is_m3_max', False),
            "psutil_available": PSUTIL_AVAILABLE
        }
        
        if PSUTIL_AVAILABLE:
            try:
                vm = psutil.virtual_memory()
                system_memory = {
                    "total_gb": round(vm.total / (1024**3), 2),
                    "available_gb": round(vm.available / (1024**3), 2),
                    "used_gb": round(vm.used / (1024**3), 2),
                    "percent_used": round(vm.percent, 1)
                }
                result["system_memory"] = system_memory
                result["is_available"] = system_memory["available_gb"] >= min_gb
            except Exception as e:
                result["system_memory_error"] = str(e)[:100]
                result["is_available"] = False
        else:
            result["system_memory"] = {"error": "psutil not available"}
            result["is_available"] = False
        
        return result
        
    except Exception as e:
        return {
            "device": device or "unknown",
            "error": str(e)[:200],
            "is_available": False,
            "min_required_gb": min_gb,
            "timestamp": time.time(),
            "psutil_available": PSUTIL_AVAILABLE,
            "torch_available": TORCH_AVAILABLE
        }

def optimize_memory(device: Optional[str] = None, aggressive: bool = False) -> Dict[str, Any]:
    """메모리 최적화"""
    try:
        if 'gpu_config' not in globals():
            return {
                "success": False,
                "error": "GPU config not initialized",
                "device": device or "unknown"
            }
        return gpu_config.cleanup_memory(aggressive)
    except Exception as e:
        return {
            "success": False,
            "error": str(e)[:200],
            "device": device or "unknown"
        }

@lru_cache(maxsize=1)
def get_gpu_config():
    """GPU 설정 매니저 반환"""
    try:
        if 'gpu_config' in globals():
            return gpu_config
        return None
    except:
        return None

def get_device_config() -> Dict[str, Any]:
    """디바이스 설정 반환"""
    try:
        if 'gpu_config' not in globals():
            return {"error": "GPU config not initialized"}
        return gpu_config.get_device_config()
    except Exception as e:
        return {"error": str(e)[:200]}

def get_model_config() -> Dict[str, Any]:
    """모델 설정 반환"""
    try:
        if 'gpu_config' not in globals():
            return {"error": "GPU config not initialized"}
        return gpu_config.get_model_config()
    except Exception as e:
        return {"error": str(e)[:200]}

def get_device_info() -> Dict[str, Any]:
    """디바이스 정보 반환"""
    try:
        if 'gpu_config' not in globals():
            return {"error": "GPU config not initialized"}
        return gpu_config.get_device_info()
    except Exception as e:
        return {"error": str(e)[:200]}

def get_device() -> str:
    """현재 디바이스 반환"""
    try:
        if 'gpu_config' not in globals():
            return "cpu"
        return gpu_config.get_device()
    except:
        return "cpu"

def is_m3_max() -> bool:
    """M3 Max 여부 확인"""
    try:
        if 'gpu_config' not in globals():
            return False
        return gpu_config.is_m3_max
    except:
        return False

def get_device_name() -> str:
    """디바이스 이름 반환"""
    try:
        if 'gpu_config' not in globals():
            return "Unknown"
        return gpu_config.get_device_name()
    except:
        return "Unknown"

def apply_optimizations() -> bool:
    """최적화 설정 적용"""
    try:
        if 'gpu_config' not in globals():
            return False
        return gpu_config.is_initialized
    except:
        return False

def get_optimal_settings() -> Dict[str, Any]:
    """최적 설정 반환"""
    try:
        if 'gpu_config' not in globals():
            return {"error": "GPU config not initialized"}
        return gpu_config.optimization_settings.copy()
    except Exception as e:
        return {"error": str(e)[:200]}

def get_device_capabilities() -> Dict[str, Any]:
    """디바이스 기능 반환"""
    try:
        if 'gpu_config' not in globals():
            return {"error": "GPU config not initialized"}
            
        return {
            "device": gpu_config.device,
            "device_name": gpu_config.device_name,
            "supports_fp16": False,  # 항상 False (호환성)
            "supports_fp32": True,   # 항상 True (호환성)
            "max_batch_size": gpu_config.optimization_settings["batch_size"] * 2,
            "recommended_image_size": (640, 640) if gpu_config.is_m3_max else (512, 512),
            "supports_8step_pipeline": True,
            "optimization_level": gpu_config.optimization_settings["optimization_level"],
            "memory_gb": gpu_config.memory_gb,
            "pytorch_version": torch.__version__ if TORCH_AVAILABLE else "not_available",
            "is_m3_max": gpu_config.is_m3_max,
            "supports_neural_engine": gpu_config.is_m3_max,
            "supports_metal_shaders": gpu_config.device == "mps",
            "unified_memory_optimization": gpu_config.optimization_settings["unified_memory_optimization"],
            "high_resolution_processing": gpu_config.optimization_settings["high_resolution_processing"],
            "pipeline_parallelism": gpu_config.optimization_settings["pipeline_parallelism"],
            "float_compatibility_mode": True,
            "stable_operation_mode": True
        }
    except Exception as e:
        return {"error": str(e)[:200]}

def get_memory_info(device: Optional[str] = None) -> Dict[str, Any]:
    """메모리 정보 반환"""
    try:
        if 'gpu_config' not in globals():
            return {"error": "GPU config not initialized", "device": device or "unknown"}
        return gpu_config.get_memory_stats()
    except Exception as e:
        return {"error": str(e)[:200], "device": device or "unknown"}

# ===============================================================
# 🔧 전역 GPU 설정 매니저 생성
# ===============================================================

try:
    gpu_config = GPUManager()
    
    # 편의를 위한 전역 변수들
    DEVICE = gpu_config.device
    DEVICE_NAME = gpu_config.device_name
    DEVICE_TYPE = gpu_config.device_type
    MODEL_CONFIG = gpu_config.model_config
    DEVICE_INFO = gpu_config.device_info
    IS_M3_MAX = gpu_config.is_m3_max
    
    # GPUConfig 인스턴스도 생성 (하위 호환성)
    GPUConfig = GPUConfig(device=DEVICE, optimization_level=gpu_config.optimization_settings["optimization_level"])
    
    # 초기화 성공 로그
    if IS_M3_MAX:
        print(f"🍎 M3 Max ({DEVICE}) 최적화 모드 활성화 - Float32 안정성 우선")
    else:
        print(f"✅ GPU 설정 모듈 로드 완료 - 안정성 우선 모드")

except Exception as e:
    print(f"⚠️ GPU 설정 초기화 실패: {str(e)[:100]}")
    
    # 폴백 설정
    DEVICE = "cpu"
    DEVICE_NAME = "CPU (Fallback)"
    DEVICE_TYPE = "cpu"
    MODEL_CONFIG = {
        "device": "cpu",
        "dtype": "float32",
        "batch_size": 1,
        "optimization_level": "safe",
        "float_compatibility_mode": True
    }
    DEVICE_INFO = {
        "device": "cpu",
        "error": "GPU config initialization failed",
        "fallback_mode": True
    }
    IS_M3_MAX = False
    
    # 더미 GPU 설정 객체 생성
    class DummyGPUConfig:
        def __init__(self):
            self.device = "cpu"
            self.device_name = "CPU (Fallback)"
            self.is_m3_max = False
            self.is_initialized = False
            self.optimization_settings = {"optimization_level": "safe"}
        
        def get(self, key, default=None):
            return getattr(self, key, default)
        
        def cleanup_memory(self, aggressive=False):
            return {"success": True, "method": "fallback_gc", "device": "cpu"}
        
        def get_device(self):
            return self.device
        
        def get_device_name(self):
            return self.device_name
    
    gpu_config = DummyGPUConfig()
    GPUConfig = DummyGPUConfig()

# ===============================================================
# 🔧 Export 리스트
# ===============================================================

__all__ = [
    # 주요 객체들
    'gpu_config', 'GPUConfig', 'DEVICE', 'DEVICE_NAME', 'DEVICE_TYPE', 
    'MODEL_CONFIG', 'DEVICE_INFO', 'IS_M3_MAX',
    
    # 핵심 함수들
    'get_gpu_config', 'get_device_config', 'get_model_config', 'get_device_info',
    'get_device', 'get_device_name', 'is_m3_max', 'get_optimal_settings', 'get_device_capabilities',
    'apply_optimizations',
    
    # 메모리 관리 함수들
    'check_memory_available', 'optimize_memory', 'get_memory_info',
    
    # 클래스들
    'GPUManager', 'HardwareDetector'
]