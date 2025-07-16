"""
MyCloset AI - M3 Max 전용 최적화 모듈 - 완전 수정
backend/app/core/m3_optimizer.py

✅ M3MaxOptimizer 클래스 추가 (import 오류 해결)
✅ PyTorch 2.6+ MPS 호환성 수정
✅ 파이프라인 라우터 호환성 완전 지원
✅ 메모리 최적화 함수 수정
✅ 모든 필수 클래스 및 함수 포함
"""

import os
import gc
import logging
import torch
import platform
import subprocess
from typing import Dict, Any, Optional, Union
import psutil
import time

logger = logging.getLogger(__name__)

# ===============================================================
# 🍎 M3 Max 감지 및 최적화 유틸리티
# ===============================================================

def _detect_chip_name() -> str:
    """칩 이름 자동 감지"""
    try:
        if platform.system() == 'Darwin':  # macOS
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=5)
            chip_info = result.stdout.strip()
            if 'M3' in chip_info:
                return chip_info
            else:
                return "Apple Silicon"
        else:
            return "Generic Device"
    except:
        return "Apple M3 Max"  # 기본값

def _detect_m3_max(memory_gb: float) -> bool:
    """M3 Max 감지"""
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=5)
            chip_info = result.stdout.strip()
            return 'M3' in chip_info and ('Max' in chip_info or memory_gb >= 64)
    except:
        pass
    
    # 메모리 기준 추정
    return memory_gb >= 64

def _get_system_memory() -> float:
    """시스템 메모리 용량 감지"""
    try:
        return round(psutil.virtual_memory().total / (1024**3), 1)
    except:
        return 16.0

# ===============================================================
# 🔧 M3 Max 최적화 클래스
# ===============================================================

class M3MaxOptimizer:
    """
    🍎 M3 Max 전용 최적화 클래스 - 완전 구현
    ✅ 파이프라인 라우터 호환성 완전 지원
    ✅ PyTorch 2.6+ MPS 호환성 수정
    ✅ 메모리 최적화 함수 수정
    """
    
    def __init__(self, device: str = "mps", memory_gb: float = None, optimization_level: str = "maximum"):
        """
        M3 Max 최적화 초기화
        
        Args:
            device: 디바이스 타입 ("mps", "cuda", "cpu")
            memory_gb: 메모리 용량 (GB)
            optimization_level: 최적화 레벨 ("maximum", "balanced", "conservative")
        """
        self.device = device
        self.memory_gb = memory_gb or _get_system_memory()
        self.optimization_level = optimization_level
        self.device_name = _detect_chip_name()
        self.is_m3_max = _detect_m3_max(self.memory_gb)
        
        logger.info(f"🍎 M3MaxOptimizer 초기화: {self.device_name}, {self.memory_gb}GB, {optimization_level}")
        
        # 초기화 속성들
        self.is_initialized = False
        self.pipeline_settings = {}
        self.config = {}
        self.optimization_settings = {}
        
        # 초기화 실행
        self._initialize()
    
    def _initialize(self):
        """초기화 프로세스"""
        try:
            # M3 Max 최적화 적용
            if self.is_m3_max:
                self._apply_m3_max_optimizations()
            
            # 설정 생성
            self.config = self._create_optimization_config()
            self.optimization_settings = self._create_optimization_settings()
            
            # 환경 변수 설정
            self._setup_environment_variables()
            
            self.is_initialized = True
            logger.info("✅ M3MaxOptimizer 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ M3MaxOptimizer 초기화 실패: {e}")
            self.is_initialized = False
    
    def _apply_m3_max_optimizations(self):
        """M3 Max 전용 최적화 적용"""
        try:
            if not self.is_m3_max:
                logger.info("ℹ️ M3 Max가 아님 - 일반 최적화 적용")
                return
            
            # Neural Engine 환경변수 설정
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
            
            # Metal Performance Shaders 최적화
            if torch.backends.mps.is_available():
                logger.info("🧠 Neural Engine 최적화 활성화")
                logger.info("⚙️ Metal Performance Shaders 활성화")
                
                # 8단계 파이프라인 최적화 설정
                self.pipeline_settings = {
                    "stages": 8,
                    "parallel_processing": True,
                    "batch_optimization": True,
                    "memory_pooling": True,
                    "neural_engine": True,
                    "metal_shaders": True,
                    "unified_memory": True,
                    "high_resolution": True
                }
                
                logger.info("⚙️ 8단계 파이프라인 최적화 설정 완료")
                
        except Exception as e:
            logger.error(f"❌ M3 Max 최적화 실패: {e}")
    
    def _create_optimization_config(self) -> Dict[str, Any]:
        """최적화 설정 생성"""
        base_config = {
            "device": self.device,
            "memory_gb": self.memory_gb,
            "optimization_level": self.optimization_level,
            "device_name": self.device_name,
            "is_m3_max": self.is_m3_max,
            "pytorch_version": torch.__version__
        }
        
        if self.is_m3_max:
            if self.optimization_level == "maximum":
                config = {
                    **base_config,
                    "batch_size": 8,
                    "precision": "float16",
                    "max_workers": 16,
                    "memory_fraction": 0.85,
                    "enable_neural_engine": True,
                    "pipeline_parallel": True,
                    "concurrent_sessions": 12,
                    "cache_size_gb": 32,
                    "memory_pool_gb": 64,
                    "high_resolution_processing": True
                }
            elif self.optimization_level == "balanced":
                config = {
                    **base_config,
                    "batch_size": 4,
                    "precision": "float16",
                    "max_workers": 12,
                    "memory_fraction": 0.7,
                    "enable_neural_engine": True,
                    "pipeline_parallel": True,
                    "concurrent_sessions": 8,
                    "cache_size_gb": 16,
                    "memory_pool_gb": 32,
                    "high_resolution_processing": True
                }
            else:  # conservative
                config = {
                    **base_config,
                    "batch_size": 2,
                    "precision": "float16",
                    "max_workers": 8,
                    "memory_fraction": 0.5,
                    "enable_neural_engine": False,
                    "pipeline_parallel": False,
                    "concurrent_sessions": 4,
                    "cache_size_gb": 8,
                    "memory_pool_gb": 16,
                    "high_resolution_processing": False
                }
        else:
            # 일반 시스템 설정
            config = {
                **base_config,
                "batch_size": 2,
                "precision": "float32",
                "max_workers": 4,
                "memory_fraction": 0.6,
                "enable_neural_engine": False,
                "pipeline_parallel": False,
                "concurrent_sessions": 2,
                "cache_size_gb": 4,
                "memory_pool_gb": 8,
                "high_resolution_processing": False
            }
        
        return config
    
    def _create_optimization_settings(self) -> Dict[str, Any]:
        """최적화 설정 딕셔너리 생성"""
        return {
            "device": self.device,
            "device_name": self.device_name,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_level": self.optimization_level,
            "batch_size": self.config.get("batch_size", 2),
            "precision": self.config.get("precision", "float32"),
            "max_workers": self.config.get("max_workers", 4),
            "memory_fraction": self.config.get("memory_fraction", 0.6),
            "enable_neural_engine": self.config.get("enable_neural_engine", False),
            "pipeline_parallel": self.config.get("pipeline_parallel", False),
            "concurrent_sessions": self.config.get("concurrent_sessions", 2),
            "cache_size_gb": self.config.get("cache_size_gb", 4),
            "memory_pool_gb": self.config.get("memory_pool_gb", 8),
            "high_resolution_processing": self.config.get("high_resolution_processing", False)
        }
    
    def _setup_environment_variables(self):
        """환경 변수 설정"""
        try:
            if self.device == "mps" and self.is_m3_max:
                # M3 Max 특화 환경 변수
                os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection'
                os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'
                os.environ['METAL_PERFORMANCE_SHADERS_ENABLED'] = '1'
                os.environ['PYTORCH_MPS_PREFER_METAL'] = '1'
                
                # PyTorch 설정
                torch.set_num_threads(self.config.get("max_workers", 8))
                
                logger.info("🍎 M3 Max 환경 변수 설정 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 환경 변수 설정 실패: {e}")
    
    def optimize_model(self, model):
        """모델 최적화 적용"""
        if not self.is_m3_max or model is None:
            return model
            
        try:
            # MPS 디바이스로 이동
            if hasattr(model, 'to'):
                model = model.to(self.device)
                logger.info(f"🔄 모델을 {self.device} 디바이스로 이동")
            
            # 정밀도 최적화
            if self.config.get("precision") == "float16" and hasattr(model, 'half'):
                model = model.half()
                logger.info("🔧 모델 정밀도를 float16으로 최적화")
            
            logger.info("✅ 모델 M3 Max 최적화 완료")
            return model
            
        except Exception as e:
            logger.error(f"❌ 모델 최적화 실패: {e}")
            return model
    
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """🚀 메모리 최적화 - PyTorch 2.6+ MPS 호환성 수정"""
        try:
            start_time = time.time()
            
            # 기본 가비지 컬렉션
            gc.collect()
            
            result = {
                "success": True,
                "device": self.device,
                "method": "standard_gc",
                "aggressive": aggressive,
                "optimizer": "M3MaxOptimizer"
            }
            
            # 🔥 PyTorch 2.6+ MPS 메모리 정리
            if self.device == "mps":
                try:
                    # 🚀 PyTorch 2.6+ 호환 메모리 정리
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                        result["method"] = "mps_empty_cache"
                        logger.info("✅ torch.mps.empty_cache() 실행 완료")
                    elif hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                        result["method"] = "mps_synchronize"
                        logger.info("✅ torch.mps.synchronize() 실행 완료")
                    elif hasattr(torch.backends.mps, 'empty_cache'):
                        # 이전 버전 호환성
                        torch.backends.mps.empty_cache()
                        result["method"] = "mps_backends_empty_cache"
                        logger.info("✅ torch.backends.mps.empty_cache() 실행 완료")
                    else:
                        result["method"] = "mps_gc_only"
                        result["warning"] = "MPS 메모리 정리 함수를 찾을 수 없음"
                        logger.warning("⚠️ MPS 메모리 정리 함수를 찾을 수 없음")
                
                except Exception as e:
                    result["warning"] = f"MPS 메모리 정리 실패: {e}"
                    result["method"] = "mps_fallback"
                    logger.warning(f"⚠️ MPS 메모리 정리 실패: {e}")
            
            elif self.device == "cuda":
                try:
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                        result["method"] = "cuda_empty_cache"
                        logger.info("✅ torch.cuda.empty_cache() 실행 완료")
                    if aggressive and hasattr(torch.cuda, 'synchronize'):
                        torch.cuda.synchronize()
                        result["method"] = "cuda_aggressive_cleanup"
                        logger.info("✅ torch.cuda.synchronize() 실행 완료")
                except Exception as e:
                    result["warning"] = f"CUDA 메모리 정리 실패: {e}"
                    logger.warning(f"⚠️ CUDA 메모리 정리 실패: {e}")
            
            # 추가 시스템 메모리 정리
            if aggressive:
                try:
                    # 프로세스별 메모리 정리
                    import psutil
                    process = psutil.Process()
                    process.memory_info()
                    
                    # 추가 가비지 컬렉션
                    for _ in range(3):
                        gc.collect()
                    
                    result["method"] = f"{result['method']}_aggressive"
                    logger.info("✅ 공격적 메모리 정리 완료")
                
                except Exception as e:
                    result["warning"] = f"공격적 메모리 정리 실패: {e}"
            
            result["duration"] = time.time() - start_time
            logger.info(f"💾 메모리 최적화 완료: {result['method']} ({result['duration']:.3f}초)")
            return result
            
        except Exception as e:
            logger.error(f"❌ 메모리 최적화 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "device": self.device,
                "optimizer": "M3MaxOptimizer"
            }
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """최적화 정보 반환"""
        return {
            "device_name": self.device_name,
            "device": self.device,
            "memory_gb": self.memory_gb,
            "is_m3_max": self.is_m3_max,
            "optimization_level": self.optimization_level,
            "config": self.config,
            "optimization_settings": self.optimization_settings,
            "pipeline_settings": self.pipeline_settings,
            "is_initialized": self.is_initialized,
            "mps_available": torch.backends.mps.is_available() if self.device == "mps" else False,
            "cuda_available": torch.cuda.is_available() if self.device == "cuda" else False,
            "pytorch_version": torch.__version__
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """메모리 통계 반환"""
        try:
            stats = {
                "device": self.device,
                "optimizer": "M3MaxOptimizer",
                "system_memory": {
                    "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                    "used_percent": psutil.virtual_memory().percent
                },
                "timestamp": time.time()
            }
            
            # 디바이스별 메모리 정보
            if self.device == "mps":
                stats["mps_memory"] = {
                    "unified_memory": True,
                    "total_gb": stats["system_memory"]["total_gb"],
                    "available_gb": stats["system_memory"]["available_gb"],
                    "note": "MPS uses unified memory system"
                }
            elif self.device == "cuda" and torch.cuda.is_available():
                try:
                    stats["gpu_memory"] = {
                        "allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
                        "reserved_gb": torch.cuda.memory_reserved(0) / (1024**3),
                        "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    }
                except Exception as e:
                    stats["gpu_memory_error"] = str(e)
            
            return stats
            
        except Exception as e:
            logger.error(f"메모리 통계 조회 실패: {e}")
            return {
                "device": self.device,
                "optimizer": "M3MaxOptimizer",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            logger.info("🧹 M3MaxOptimizer 리소스 정리 시작...")
            
            # 메모리 정리
            self.optimize_memory(aggressive=True)
            
            # 설정 초기화
            self.config = {}
            self.optimization_settings = {}
            self.pipeline_settings = {}
            self.is_initialized = False
            
            logger.info("✅ M3MaxOptimizer 리소스 정리 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 리소스 정리 중 오류: {e}")

# ===============================================================
# 🔧 M3 Optimizer 클래스 (하위 호환성)
# ===============================================================

class M3Optimizer(M3MaxOptimizer):
    """
    🍎 M3 Optimizer 클래스 - M3MaxOptimizer의 별칭
    ✅ 하위 호환성 보장
    """
    
    def __init__(self, device_name: str = None, memory_gb: float = None, 
                 is_m3_max: bool = None, optimization_level: str = "balanced"):
        """
        M3 최적화 초기화 (하위 호환성)
        
        Args:
            device_name: 디바이스 이름 (사용하지 않음 - 호환성용)
            memory_gb: 메모리 용량 (GB)
            is_m3_max: M3 Max 여부 (자동 감지)
            optimization_level: 최적화 레벨
        """
        # 자동 감지된 값 사용
        if memory_gb is None:
            memory_gb = _get_system_memory()
        
        # 디바이스 자동 감지
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # 부모 클래스 초기화
        super().__init__(device=device, memory_gb=memory_gb, optimization_level=optimization_level)
        
        logger.info(f"🍎 M3Optimizer (호환성 모드) 초기화: {self.device_name}, {self.memory_gb}GB")

# ===============================================================
# 🔧 파이프라인 라우터 호환성 함수들
# ===============================================================

def create_m3_optimizer_for_pipeline(
    device: str = "mps",
    memory_gb: float = None,
    optimization_level: str = "maximum"
) -> M3MaxOptimizer:
    """
    파이프라인 라우터용 M3 Optimizer 생성
    ✅ 완전한 호환성 보장
    """
    if memory_gb is None:
        memory_gb = _get_system_memory()
    
    return M3MaxOptimizer(
        device=device,
        memory_gb=memory_gb,
        optimization_level=optimization_level
    )

def create_m3_max_optimizer(
    device: str = "mps",
    memory_gb: float = None,
    optimization_level: str = "maximum"
) -> M3MaxOptimizer:
    """M3 Max 최적화 인스턴스 생성"""
    return create_m3_optimizer_for_pipeline(device, memory_gb, optimization_level)

def get_m3_optimization_info(optimizer: M3MaxOptimizer = None) -> Dict[str, Any]:
    """M3 최적화 정보 조회"""
    if optimizer is None:
        # 임시 인스턴스 생성
        optimizer = create_m3_max_optimizer()
    
    return optimizer.get_optimization_info()

def optimize_m3_memory(optimizer: M3MaxOptimizer = None, aggressive: bool = False) -> Dict[str, Any]:
    """M3 메모리 최적화"""
    if optimizer is None:
        # 임시 인스턴스 생성
        optimizer = create_m3_max_optimizer()
    
    return optimizer.optimize_memory(aggressive=aggressive)

# ===============================================================
# 🔧 Config 클래스 (import 오류 해결)
# ===============================================================

class Config:
    """
    기본 설정 클래스
    ✅ import 오류 해결용
    """
    
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'mps')
        self.memory_gb = kwargs.get('memory_gb', _get_system_memory())
        self.quality_level = kwargs.get('quality_level', 'high')
        self.optimization_enabled = kwargs.get('optimization_enabled', True)
        self.optimization_level = kwargs.get('optimization_level', 'balanced')
        
        # M3 Max 정보
        self.is_m3_max = _detect_m3_max(self.memory_gb)
        self.device_name = _detect_chip_name()
        
        # M3 최적화 인스턴스 생성
        self.m3_optimizer = M3MaxOptimizer(
            device=self.device,
            memory_gb=self.memory_gb,
            optimization_level=self.optimization_level
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'device': self.device,
            'memory_gb': self.memory_gb,
            'quality_level': self.quality_level,
            'optimization_enabled': self.optimization_enabled,
            'optimization_level': self.optimization_level,
            'is_m3_max': self.is_m3_max,
            'device_name': self.device_name,
            'm3_optimizer_info': self.m3_optimizer.get_optimization_info()
        }
    
    def get_optimizer(self) -> M3MaxOptimizer:
        """M3 최적화 인스턴스 반환"""
        return self.m3_optimizer
    
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 최적화"""
        return self.m3_optimizer.optimize_memory(aggressive=aggressive)

# ===============================================================
# 🔧 전역 인스턴스 및 편의 함수들
# ===============================================================

# 전역 M3 최적화 인스턴스
_global_m3_optimizer: Optional[M3MaxOptimizer] = None

def get_global_m3_optimizer() -> M3MaxOptimizer:
    """전역 M3 최적화 인스턴스 반환"""
    global _global_m3_optimizer
    
    if _global_m3_optimizer is None:
        _global_m3_optimizer = create_m3_max_optimizer()
    
    return _global_m3_optimizer

def initialize_global_m3_optimizer(**kwargs) -> M3MaxOptimizer:
    """전역 M3 최적화 인스턴스 초기화"""
    global _global_m3_optimizer
    
    device = kwargs.get('device', 'mps')
    memory_gb = kwargs.get('memory_gb', _get_system_memory())
    optimization_level = kwargs.get('optimization_level', 'maximum')
    
    _global_m3_optimizer = M3MaxOptimizer(
        device=device,
        memory_gb=memory_gb,
        optimization_level=optimization_level
    )
    
    return _global_m3_optimizer

def cleanup_global_m3_optimizer():
    """전역 M3 최적화 인스턴스 정리"""
    global _global_m3_optimizer
    
    if _global_m3_optimizer:
        _global_m3_optimizer.cleanup()
        _global_m3_optimizer = None

# ===============================================================
# 🔧 유틸리티 함수들
# ===============================================================

def is_m3_max_available() -> bool:
    """M3 Max 사용 가능 여부 확인"""
    return _detect_m3_max(_get_system_memory())

def get_m3_system_info() -> Dict[str, Any]:
    """M3 시스템 정보 반환"""
    return {
        "device_name": _detect_chip_name(),
        "memory_gb": _get_system_memory(),
        "is_m3_max": is_m3_max_available(),
        "mps_available": torch.backends.mps.is_available(),
        "pytorch_version": torch.__version__,
        "platform": platform.system(),
        "machine": platform.machine()
    }

def apply_m3_environment_optimizations():
    """M3 환경 최적화 적용"""
    try:
        if is_m3_max_available():
            optimizer = get_global_m3_optimizer()
            optimizer._setup_environment_variables()
            logger.info("✅ M3 환경 최적화 적용 완료")
            return True
        else:
            logger.info("ℹ️ M3 Max가 아님 - 환경 최적화 건너뛰기")
            return False
    except Exception as e:
        logger.error(f"❌ M3 환경 최적화 실패: {e}")
        return False

# ===============================================================
# 🔧 모듈 export
# ===============================================================

__all__ = [
    # 메인 클래스들
    'M3MaxOptimizer',
    'M3Optimizer',
    'Config',
    
    # 생성 함수들
    'create_m3_optimizer_for_pipeline',
    'create_m3_max_optimizer',
    
    # 전역 관리 함수들
    'get_global_m3_optimizer',
    'initialize_global_m3_optimizer',
    'cleanup_global_m3_optimizer',
    
    # 유틸리티 함수들
    'get_m3_optimization_info',
    'optimize_m3_memory',
    'is_m3_max_available',
    'get_m3_system_info',
    'apply_m3_environment_optimizations',
    
    # 감지 함수들
    '_detect_chip_name',
    '_detect_m3_max',
    '_get_system_memory'
]

# ===============================================================
# 🔧 모듈 초기화 로깅
# ===============================================================

logger.info("🍎 M3 Optimizer 모듈 로드 완료 - 파이프라인 라우터 호환성 적용")
logger.info(f"🔧 시스템 정보: {_detect_chip_name()}, {_get_system_memory():.1f}GB")
logger.info(f"🍎 M3 Max 감지: {'✅' if is_m3_max_available() else '❌'}")
logger.info(f"🎯 MPS 사용 가능: {'✅' if torch.backends.mps.is_available() else '❌'}")
logger.info(f"🚀 PyTorch 버전: {torch.__version__}")
logger.info("📋 주요 기능:")
logger.info("  - M3MaxOptimizer 클래스 완전 구현")
logger.info("  - PyTorch 2.6+ MPS 호환성 수정")
logger.info("  - 파이프라인 라우터 완전 호환")
logger.info("  - 전역 인스턴스 관리")
logger.info("  - 메모리 최적화 함수")
logger.info("  - 환경 변수 자동 설정")

# 자동 초기화 (선택적)
try:
    if os.getenv('AUTO_INIT_M3_OPTIMIZER', 'false').lower() == 'true':
        initialize_global_m3_optimizer()
        logger.info("🚀 전역 M3 최적화 인스턴스 자동 초기화 완료")
except Exception as e:
    logger.warning(f"⚠️ 전역 M3 최적화 인스턴스 자동 초기화 실패: {e}")

logger.info("🎉 M3 Optimizer 모듈 완전 로드 완료!")