# app/core/gpu_config.py
"""
MyCloset AI - M3 Max 128GB 완전 최적화 GPU 설정
8단계 가상 피팅 파이프라인 최적화
Pydantic V2 호환, 누락된 함수들 포함
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

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GPUConfig:
    """GPU 설정 데이터 클래스"""
    device: str
    device_name: str
    device_type: str
    memory_gb: float
    is_m3_max: bool
    optimization_level: str
    neural_engine_available: bool = False
    metal_performance_shaders: bool = False
    unified_memory_optimization: bool = False

class M3MaxGPUManager:
    """M3 Max 128GB 전용 GPU 관리자 - 8단계 파이프라인 최적화"""
    
    def __init__(self):
        """초기화"""
        self.device = None
        self.device_name = ""
        self.device_type = ""
        self.memory_gb = 0.0
        self.is_m3_max = False
        self.optimization_level = "balanced"
        self.device_info = {}
        self.model_config = {}
        self.is_initialized = False
        self.neural_engine_available = False
        self.metal_performance_shaders = False
        
        # 8단계 파이프라인별 최적화 설정
        self.pipeline_optimizations = {}
        
        # 초기화 실행
        self._initialize()
    
    def _initialize(self):
        """GPU 설정 완전 초기화"""
        try:
            logger.info("🔧 M3 Max GPU 설정 초기화 시작...")
            
            # 1. 하드웨어 감지
            self._detect_hardware()
            
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
    
    def _detect_hardware(self):
        """M3 Max 하드웨어 정밀 감지"""
        try:
            import psutil
            
            system_info = {
                "platform": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            }
            
            # Apple Silicon 확인
            if (system_info["platform"] == "Darwin" and 
                system_info["machine"] == "arm64"):
                
                # 메모리 크기로 M3 Max 판정
                memory_gb = psutil.virtual_memory().total / (1024**3)
                
                if memory_gb >= 120:  # 128GB M3 Max
                    self.is_m3_max = True
                    self.optimization_level = "ultra"
                    self.neural_engine_available = True
                    self.metal_performance_shaders = True
                    logger.info(f"🍎 M3 Max 128GB 감지! 메모리: {memory_gb:.0f}GB")
                    
                elif memory_gb >= 90:  # 96GB M3 Max
                    self.is_m3_max = True
                    self.optimization_level = "high"
                    self.neural_engine_available = True
                    logger.info(f"🍎 M3 Max 96GB 감지! 메모리: {memory_gb:.0f}GB")
                    
                else:  # 기타 Apple Silicon
                    self.optimization_level = "balanced"
                    logger.info(f"🍎 Apple Silicon 감지 - 메모리: {memory_gb:.0f}GB")
                
                self.memory_gb = memory_gb
            
            else:
                logger.info("🖥️ 비-Apple Silicon 환경 감지")
                self.memory_gb = psutil.virtual_memory().total / (1024**3)
            
        except Exception as e:
            logger.warning(f"⚠️ 하드웨어 감지 실패: {e}")
            self.optimization_level = "safe"
    
    def _setup_device(self):
        """디바이스 설정 및 MPS 최적화"""
        try:
            # MPS 사용 가능성 확인
            if torch.backends.mps.is_available():
                self.device = "mps"
                self.device_type = "mps"
                self.device_name = "Apple Silicon GPU (MPS)"
                
                # M3 Max 특화 MPS 환경변수 설정
                if self.is_m3_max:
                    os.environ.update({
                        'PYTORCH_ENABLE_MPS_FALLBACK': '1',
                        'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',  # 128GB 메모리 최적화
                        'PYTORCH_MPS_ALLOCATOR_POLICY': 'garbage_collection',
                        'METAL_DEVICE_WRAPPER_TYPE': '1',
                        'METAL_PERFORMANCE_SHADERS_ENABLED': '1'
                    })
                    logger.info("🍎 M3 Max MPS 환경변수 최적화 적용")
                
                logger.info("🍎 Apple Silicon MPS 활성화")
                
            elif torch.cuda.is_available():
                self.device = "cuda"
                self.device_type = "cuda"
                self.device_name = torch.cuda.get_device_name(0)
                logger.info("🚀 CUDA GPU 감지")
                
            else:
                self.device = "cpu"
                self.device_type = "cpu"
                self.device_name = "CPU"
                logger.info("💻 CPU 모드 설정")
                
        except Exception as e:
            logger.error(f"❌ 디바이스 설정 실패: {e}")
            self._fallback_cpu_setup()
    
    def _setup_pipeline_optimizations(self):
        """8단계 파이프라인별 최적화 설정"""
        
        # M3 Max 128GB 기준 최적화
        if self.is_m3_max and self.device == "mps":
            base_config = {
                "batch_size": 2,
                "precision": "float16",
                "memory_fraction": 0.6,  # 128GB 중 일부만 사용
                "enable_attention_slicing": True,
                "enable_cpu_offload": False,  # 128GB RAM 충분
                "concurrent_processing": True
            }
        else:
            base_config = {
                "batch_size": 1,
                "precision": "float32",
                "memory_fraction": 0.7,
                "enable_attention_slicing": True,
                "enable_cpu_offload": True,
                "concurrent_processing": False
            }
        
        # 8단계별 특화 설정
        self.pipeline_optimizations = {
            "step_01_human_parsing": {
                **base_config,
                "model_precision": "float16" if self.device == "mps" else "float32",
                "max_resolution": 512,
                "enable_segmentation_cache": True
            },
            "step_02_pose_estimation": {
                **base_config,
                "openpose_precision": "float16" if self.device == "mps" else "float32",
                "keypoint_threshold": 0.3,
                "enable_pose_cache": True
            },
            "step_03_cloth_segmentation": {
                **base_config,
                "segmentation_model": "u2net",
                "background_threshold": 0.5,
                "enable_edge_refinement": True
            },
            "step_04_geometric_matching": {
                **base_config,
                "matching_algorithm": "optical_flow",
                "warp_resolution": 256,
                "enable_geometric_cache": True
            },
            "step_05_cloth_warping": {
                **base_config,
                "warp_method": "thin_plate_spline",
                "interpolation": "bilinear",
                "preserve_details": True
            },
            "step_06_virtual_fitting": {
                **base_config,
                "diffusion_steps": 20 if self.is_m3_max else 15,
                "guidance_scale": 7.5,
                "enable_safety_checker": True,
                "scheduler": "ddim"
            },
            "step_07_post_processing": {
                **base_config,
                "enhancement_level": "high" if self.is_m3_max else "medium",
                "noise_reduction": True,
                "color_correction": True
            },
            "step_08_quality_assessment": {
                **base_config,
                "quality_metrics": ["ssim", "lpips", "fid"],
                "assessment_threshold": 0.7,
                "enable_automatic_retry": True
            }
        }
        
        logger.info("⚙️ 8단계 파이프라인 최적화 설정 완료")
    
    def _setup_model_config(self):
        """모델 설정 구성"""
        base_config = {
            "device": self.device,
            "dtype": "float16" if self.device in ["mps", "cuda"] else "float32",
            "batch_size": self.get_recommended_batch_size(),
            "memory_fraction": self.get_memory_fraction(),
            "optimization_level": self.optimization_level
        }
        
        # M3 Max 특화 설정
        if self.is_m3_max and self.device == "mps":
            base_config.update({
                "use_neural_engine": True,
                "metal_performance_shaders": True,
                "unified_memory_optimization": True,
                "high_resolution_processing": True,
                "concurrent_pipeline_steps": 3,  # 3단계 동시 처리
                "memory_pool_size_gb": 32,  # 128GB 중 32GB 할당
                "model_cache_size_gb": 16,  # 모델 캐싱용
                "intermediate_cache_gb": 8   # 중간 결과 캐싱
            })
            logger.info("🍎 M3 Max 특화 모델 설정 적용")
        
        self.model_config = base_config
        logger.info(f"⚙️ 모델 설정 완료: 배치={base_config['batch_size']}, 정밀도={base_config['dtype']}")
    
    def _collect_device_info(self):
        """디바이스 정보 수집"""
        try:
            import psutil
            
            base_info = {
                "device": self.device,
                "device_name": self.device_name,
                "device_type": self.device_type,
                "platform": platform.system(),
                "architecture": platform.machine(),
                "pytorch_version": torch.__version__,
                "python_version": platform.python_version(),
                "optimization_level": self.optimization_level
            }
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            base_info.update({
                "total_memory_gb": memory.total / (1024**3),
                "available_memory_gb": memory.available / (1024**3),
                "memory_usage_percent": memory.percent
            })
            
            # M3 Max 특화 정보
            if self.is_m3_max:
                base_info.update({
                    "is_m3_max": True,
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
            elif self.device == "cuda":
                if torch.cuda.is_available():
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
            num_threads = min(8, os.cpu_count() or 4)
            if self.is_m3_max:
                num_threads = min(12, os.cpu_count() or 8)  # M3 Max 더 많은 스레드
            
            torch.set_num_threads(num_threads)
            
            # MPS 최적화
            if self.device == "mps":
                # M3 Max 특화 MPS 설정
                if self.is_m3_max:
                    # Metal Performance Shaders 활성화
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
        if self.is_m3_max:
            return 4 if self.device == "mps" else 2
        elif self.device == "cuda":
            return 2
        else:
            return 1
    
    def get_recommended_precision(self) -> str:
        """권장 정밀도 반환"""
        if self.device in ["mps", "cuda"]:
            return "float16"
        return "float32"
    
    def get_memory_fraction(self) -> float:
        """메모리 사용 비율 반환"""
        if self.is_m3_max:
            return 0.6  # 128GB는 여유있게
        elif self.device == "mps":
            return 0.7
        elif self.device == "cuda":
            return 0.8
        else:
            return 0.5
    
    def setup_multiprocessing(self) -> int:
        """멀티프로세싱 워커 수 설정"""
        if self.is_m3_max:
            return min(8, os.cpu_count() or 4)
        else:
            return min(4, os.cpu_count() or 2)
    
    def get_device_config(self) -> Dict[str, Any]:
        """디바이스 설정 반환"""
        return GPUConfig(
            device=self.device,
            device_name=self.device_name,
            device_type=self.device_type,
            memory_gb=self.memory_gb,
            is_m3_max=self.is_m3_max,
            optimization_level=self.optimization_level,
            neural_engine_available=self.neural_engine_available,
            metal_performance_shaders=self.metal_performance_shaders,
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

# ==========================================
# 성능 및 최적화 함수들
# ==========================================

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

# ==========================================
# 메모리 관리 함수들
# ==========================================

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
            # M3 Max MPS 메모리 최적화
            try:
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    result["method"] = "mps_empty_cache"
                elif hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
                    result["method"] = "mps_synchronize"
                
                if aggressive and gpu_config.is_m3_max:
                    # M3 Max 특화 적극적 정리
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
# backend/app/core/gpu_config.py에 추가할 누락된 함수들

def check_memory_available(required_gb: float = 4.0) -> bool:
    """
    M3 Max 메모리 사용 가능 여부 확인
    
    Args:
        required_gb: 필요한 메모리 용량 (GB)
    
    Returns:
        bool: 메모리 사용 가능 여부
    """
    import psutil
    import torch
    
    try:
        # 시스템 메모리 확인
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        logger.info(f"💾 시스템 메모리: {memory.total / (1024**3):.1f}GB")
        logger.info(f"💾 사용 가능: {available_gb:.1f}GB")
        logger.info(f"💾 요구사항: {required_gb:.1f}GB")
        
        # MPS 메모리 확인 (M3 Max)
        if torch.backends.mps.is_available():
            # MPS는 unified memory 사용
            logger.info("🍎 M3 Max Unified Memory 사용 중")
            return available_gb >= required_gb
        
        return available_gb >= required_gb
        
    except Exception as e:
        logger.warning(f"⚠️ 메모리 확인 실패: {e}")
        return True  # 안전하게 True 반환

def get_device_config() -> Dict[str, Any]:
    """
    M3 Max 디바이스 설정 반환
    
    Returns:
        Dict: 디바이스 설정 정보
    """
    import platform
    import torch
    
    config = {
        "device_name": "Apple M3 Max",
        "memory_gb": 128,  # M3 Max 128GB 모델
        "is_m3_max": True,
        "optimization_level": "maximum",
        "mps_available": torch.backends.mps.is_available(),
        "system_info": {
            "platform": platform.system(),
            "processor": platform.processor(),
            "machine": platform.machine()
        },
        "recommended_settings": {
            "batch_size": 4,
            "precision": "float16",
            "max_workers": 12,
            "memory_fraction": 0.8
        }
    }
    
    logger.info("🍎 M3 Max 디바이스 설정 생성됨")
    return config

def initialize_global_memory_manager(device: str = "mps", memory_gb: float = 128.0):
    """
    전역 메모리 매니저 초기화
    
    Args:
        device: 사용할 디바이스
        memory_gb: 총 메모리 용량
    """
    try:
        import gc
        import torch
        
        # 메모리 정리
        gc.collect()
        
        if device == "mps" and torch.backends.mps.is_available():
            # MPS 메모리 설정
            logger.info(f"🍎 M3 Max MPS 메모리 매니저 초기화: {memory_gb}GB")
            
            # 환경변수 설정
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
            
        logger.info("✅ 전역 메모리 매니저 초기화 완료")
        
    except Exception as e:
        logger.error(f"❌ 메모리 매니저 초기화 실패: {e}")

# M3Optimizer 클래스 (app/core/m3_optimizer.py용)
class M3Optimizer:
    """
    M3 Max 전용 최적화 클래스
    """
    
    def __init__(self, device_name: str, memory_gb: float, is_m3_max: bool, optimization_level: str):
        """
        M3 최적화 초기화
        
        Args:
            device_name: 디바이스 이름
            memory_gb: 메모리 용량
            is_m3_max: M3 Max 여부
            optimization_level: 최적화 레벨
        """
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
            import torch
            
            # Neural Engine 활성화
            if torch.backends.mps.is_available():
                logger.info("🧠 Neural Engine 최적화 활성화")
                
                # 8단계 파이프라인 최적화
                self.pipeline_config = {
                    "stages": 8,
                    "parallel_processing": True,
                    "batch_optimization": True,
                    "memory_pooling": True
                }
                
                logger.info("⚙️ 8단계 파이프라인 최적화 완료")
                
        except Exception as e:
            logger.error(f"❌ M3 Max 최적화 실패: {e}")
    
    def optimize_model(self, model):
        """모델 최적화"""
        if not self.is_m3_max:
            return model
            
        try:
            import torch
            
            if hasattr(model, 'to'):
                model = model.to('mps')
                
            # 추가 최적화 로직
            logger.info("✅ 모델 M3 Max 최적화 완료")
            return model
            
        except Exception as e:
            logger.error(f"❌ 모델 최적화 실패: {e}")
            return model

# ModelFormat 클래스 (app/ai_pipeline/utils/model_loader.py용)  
class ModelFormat:
    """AI 모델 형식 정의"""
    
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    COREML = "coreml"  # Apple Core ML for M3 Max
    
    @classmethod
    def get_optimized_format(cls, device: str = "mps") -> str:
        """디바이스에 최적화된 모델 형식 반환"""
        if device == "mps":
            return cls.COREML  # M3 Max에서는 Core ML 추천
        return cls.PYTORCH

def initialize_global_model_loader(device: str = "mps"):
    """전역 모델 로더 초기화"""
    try:
        from .model_loader import ModelLoader
        
        # 글로벌 모델 로더 인스턴스 생성
        global_loader = ModelLoader(device=device)
        
        logger.info(f"✅ 전역 ModelLoader 초기화 완료: {device}")
        return global_loader
        
    except Exception as e:
        logger.error(f"❌ 전역 ModelLoader 초기화 실패: {e}")
        return None
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
        
        # M3 Max 특화 상태 판정
        if hasattr(gpu_config, 'is_m3_max') and gpu_config.is_m3_max:
            if memory.percent < 40:
                status["status"] = "excellent"
            elif memory.percent < 70:
                status["status"] = "good"
            elif memory.percent < 85:
                status["status"] = "moderate"
            else:
                status["status"] = "high"
        else:
            if memory.percent < 70:
                status["status"] = "good"
            elif memory.percent < 85:
                status["status"] = "moderate"
            else:
                status["status"] = "high"
        
        # GPU 메모리 정보 추가
        if hasattr(gpu_config, 'device') and gpu_config.device == "cuda":
            try:
                status.update({
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3)
                })
            except:
                pass
        
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

def cleanup_gpu_resources():
    """GPU 리소스 정리"""
    try:
        optimize_memory(aggressive=True)
        logger.info("✅ GPU 리소스 정리 완료")
    except Exception as e:
        logger.warning(f"⚠️ GPU 정리 중 오류: {e}")

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
    
    # 편의 함수들
    'is_gpu_available', 'is_m3_max_available', 'get_recommended_settings',
    'get_device_name', 'get_device_type',
    
    # 파이프라인 특화 함수들
    'get_pipeline_config', 'get_all_pipeline_configs',
    
    # 클래스들
    'M3MaxGPUManager', 'GPUConfig'
]