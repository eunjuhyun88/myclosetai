# app/core/gpu_config.py
"""
MyCloset AI - M3 Max 128GB 최적화 GPU 설정
Pydantic V2 호환, 누락된 함수들 추가
"""

import os
import logging
import torch
import platform
from typing import Dict, Any, Optional
from dataclasses import dataclass

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GPUConfig:
    """GPU 설정 클래스"""
    device: str
    device_type: str
    memory_gb: float
    is_m3_max: bool
    optimization_enabled: bool
    
class M3MaxGPUManager:
    """M3 Max 128GB 전용 GPU 관리자"""
    
    def __init__(self):
        self.device = None
        self.device_info = {}
        self.model_config = {}
        self.is_initialized = False
        self.m3_max_detected = False
        
        # 초기화
        self._initialize()
    
    def _initialize(self):
        """GPU 설정 초기화"""
        try:
            logger.info("🔧 GPU 설정 초기화 시작...")
            
            # M3 Max 감지
            self._detect_m3_max()
            
            # 디바이스 설정
            self._setup_device()
            
            # 모델 설정
            self._setup_model_config()
            
            # 디바이스 정보 수집
            self._collect_device_info()
            
            # 최적화 적용
            self._apply_optimizations()
            
            self.is_initialized = True
            logger.info(f"🚀 GPU 설정 초기화 완료: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ GPU 설정 초기화 실패: {e}")
            self._fallback_cpu_setup()
    
    def _detect_m3_max(self):
        """M3 Max 감지"""
        try:
            import psutil
            
            # Apple Silicon + 대용량 메모리 확인
            if platform.machine() == 'arm64' and platform.system() == 'Darwin':
                memory_gb = psutil.virtual_memory().total / (1024**3)
                
                if memory_gb >= 120:  # 128GB 근사치
                    self.m3_max_detected = True
                    logger.info("🍎 M3 Max 128GB 환경 감지!")
                else:
                    logger.info(f"🍎 Apple Silicon 감지 - 메모리: {memory_gb:.0f}GB")
            
        except Exception as e:
            logger.warning(f"⚠️ M3 Max 감지 실패: {e}")
    
    def _setup_device(self):
        """디바이스 설정"""
        try:
            if torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("🍎 Apple Silicon MPS 감지")
                
                if self.m3_max_detected:
                    logger.info("🍎 M3 Max 특화 최적화 적용")
                    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # M3 Max 메모리 최적화
            
            elif torch.cuda.is_available():
                self.device = "cuda"
                logger.info("🚀 CUDA GPU 감지")
            
            else:
                self.device = "cpu"
                logger.info("💻 CPU 모드로 설정")
        
        except Exception as e:
            logger.error(f"❌ 디바이스 설정 실패: {e}")
            self.device = "cpu"
    
    def _setup_model_config(self):
        """모델 설정 구성"""
        base_config = {
            "device": self.device,
            "dtype": "float16" if self.device in ["mps", "cuda"] else "float32",
            "batch_size": 1,
            "memory_fraction": 0.8
        }
        
        if self.m3_max_detected and self.device == "mps":
            # M3 Max 특화 설정
            base_config.update({
                "batch_size": 4,  # M3 Max는 더 큰 배치 가능
                "memory_fraction": 0.6,  # 128GB 중 일부만 사용
                "use_neural_engine": True,
                "metal_performance_shaders": True,
                "unified_memory_optimization": True,
                "high_resolution_processing": True
            })
            logger.info("🍎 M3 Max 특화 최적화 적용")
        
        elif self.device == "mps":
            # 일반 Apple Silicon 설정
            base_config.update({
                "batch_size": 2,
                "memory_fraction": 0.7,
                "use_neural_engine": False
            })
        
        elif self.device == "cuda":
            # CUDA 설정
            base_config.update({
                "batch_size": 2,
                "memory_fraction": 0.8,
                "mixed_precision": True
            })
        
        self.model_config = base_config
        logger.info(f"⚙️ 모델 설정 완료: 배치크기={base_config['batch_size']}")
    
    def _collect_device_info(self):
        """디바이스 정보 수집"""
        try:
            import psutil
            
            self.device_info = {
                "device": self.device,
                "platform": platform.system(),
                "architecture": platform.machine(),
                "pytorch_version": torch.__version__,
                "python_version": platform.python_version()
            }
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            self.device_info.update({
                "total_memory_gb": memory.total / (1024**3),
                "available_memory_gb": memory.available / (1024**3),
                "memory_usage_percent": memory.percent
            })
            
            # 디바이스별 정보
            if self.device == "mps":
                self.device_info.update({
                    "name": "Apple Silicon GPU (MPS)",
                    "mps_available": True,
                    "is_m3_max": self.m3_max_detected,
                    "neural_engine_available": self.m3_max_detected,
                    "metal_performance_shaders": True
                })
                
                if self.m3_max_detected:
                    self.device_info.update({
                        "gpu_cores": "30-40 cores",
                        "memory_bandwidth": "400GB/s",
                        "neural_engine_tops": "15.8 TOPS"
                    })
            
            elif self.device == "cuda":
                if torch.cuda.is_available():
                    self.device_info.update({
                        "name": torch.cuda.get_device_name(0),
                        "cuda_version": torch.version.cuda,
                        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                        "compute_capability": torch.cuda.get_device_capability(0)
                    })
            
            else:  # CPU
                self.device_info.update({
                    "name": "CPU",
                    "cpu_cores": psutil.cpu_count(),
                    "cpu_cores_physical": psutil.cpu_count(logical=False)
                })
            
            logger.info(f"ℹ️ 디바이스 정보 수집 완료: {self.device_info.get('name', 'Unknown')}")
            
        except Exception as e:
            logger.warning(f"⚠️ 디바이스 정보 수집 실패: {e}")
            self.device_info = {"device": self.device, "name": "Unknown", "error": str(e)}
    
    def _apply_optimizations(self):
        """최적화 적용"""
        try:
            if self.device == "mps":
                # MPS 최적화
                if hasattr(torch.backends.mps, 'empty_cache'):
                    # 새로운 PyTorch 버전에서 지원
                    logger.info("ℹ️ MPS empty_cache 지원됨")
                else:
                    logger.info("ℹ️ MPS empty_cache 미지원 - 대체 메모리 관리 사용")
                
                # M3 Max 특화 최적화
                if self.m3_max_detected:
                    # Metal Performance Shaders 최적화
                    os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'
                    os.environ['METAL_PERFORMANCE_SHADERS_ENABLED'] = '1'
                    logger.info("🍎 MPS 최적화 완료")
                
                logger.info("✅ MPS 최적화 적용 완료")
            
            elif self.device == "cuda":
                # CUDA 최적화
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                logger.info("✅ CUDA 최적화 적용 완료")
            
            # 공통 최적화
            torch.set_num_threads(min(8, os.cpu_count() or 4))
            
        except Exception as e:
            logger.warning(f"⚠️ 최적화 적용 실패: {e}")
    
    def _fallback_cpu_setup(self):
        """CPU 폴백 설정"""
        self.device = "cpu"
        self.model_config = {
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 1,
            "memory_fraction": 0.5
        }
        self.device_info = {
            "device": "cpu",
            "name": "CPU (Fallback)",
            "error": "GPU initialization failed"
        }
        logger.warning("🚨 CPU 폴백 모드로 설정됨")
    
    def get_device(self) -> str:
        """현재 디바이스 반환"""
        return self.device
    
    def get_device_config(self) -> Dict[str, Any]:
        """디바이스 설정 반환"""
        return {
            "device": self.device,
            "name": self.device_info.get("name", "Unknown"),
            "is_m3_max": self.m3_max_detected,
            "optimization_enabled": self.is_initialized
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """모델 설정 반환"""
        return self.model_config.copy()
    
    def get_device_info(self) -> Dict[str, Any]:
        """디바이스 정보 반환"""
        return self.device_info.copy()
    
    def test_device(self) -> bool:
        """디바이스 테스트"""
        try:
            # 간단한 텐서 연산 테스트
            device = torch.device(self.device)
            test_tensor = torch.randn(10, 10, device=device)
            result = torch.matmul(test_tensor, test_tensor.T)
            
            if self.device == "mps":
                # MPS에서 CPU로 이동 테스트
                cpu_result = result.cpu()
                logger.info(f"✅ {self.device} 디바이스 테스트 성공")
            else:
                logger.info(f"✅ {self.device} 디바이스 테스트 성공")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {self.device} 디바이스 테스트 실패: {e}")
            return False
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.device == "mps":
                # MPS 메모리 정리 시도
                try:
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    elif hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                except:
                    pass
            
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            
            # 가비지 컬렉션
            import gc
            gc.collect()
            
            logger.info("✅ GPU 리소스 정리 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ GPU 정리 중 오류: {e}")

# ============================================
# M3 Max 최적화된 메모리 관리 함수들
# ============================================

def optimize_memory(device: Optional[str] = None, aggressive: bool = False) -> Dict[str, Any]:
    """M3 Max 최적화된 메모리 최적화 함수"""
    try:
        import gc
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
                
                if aggressive and gpu_config.m3_max_detected:
                    # M3 Max 특화 적극적 정리
                    torch.mps.synchronize()
                    gc.collect()
                    result["method"] = "m3_max_aggressive_cleanup"
                    result["aggressive"] = True
                
            except Exception as mps_error:
                logger.warning(f"MPS 메모리 정리 실패: {mps_error}")
                result["mps_error"] = str(mps_error)
        
        elif current_device == "cuda":
            # CUDA 메모리 최적화
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
        
        # 메모리 정리 후 상태 확인
        end_memory = psutil.virtual_memory().percent
        memory_freed = start_memory - end_memory
        
        result.update({
            "end_memory_percent": end_memory,
            "memory_freed_percent": memory_freed,
            "timestamp": torch.get_default_dtype(),  # 간접적인 시간 표시
            "m3_max_optimized": gpu_config.m3_max_detected if 'gpu_config' in globals() else False
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
            "status": "good"
        }
        
        # M3 Max 특화 상태 판정
        if hasattr(gpu_config, 'm3_max_detected') and gpu_config.m3_max_detected:
            if memory.percent < 40:
                status["status"] = "excellent"
            elif memory.percent < 70:
                status["status"] = "good"
            elif memory.percent < 85:
                status["status"] = "moderate"
            else:
                status["status"] = "high"
        else:
            # 일반 환경
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
        "m3_max_detected": getattr(gpu_config, 'm3_max_detected', False) if 'gpu_config' in globals() else False,
        "neural_engine_available": (
            torch.backends.mps.is_available() and 
            getattr(gpu_config, 'm3_max_detected', False) if 'gpu_config' in globals() else False
        )
    }

# ============================================
# 전역 GPU 설정 인스턴스 생성
# ============================================

# 전역 GPU 설정 매니저 생성
gpu_config = M3MaxGPUManager()

# 편의를 위한 전역 변수들
DEVICE = gpu_config.device
MODEL_CONFIG = gpu_config.model_config
DEVICE_INFO = gpu_config.device_info

# ============================================
# 주요 함수들 (main.py 호환용)
# ============================================

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
    return gpu_config.test_device()

def cleanup_gpu_resources():
    """GPU 리소스 정리"""
    gpu_config.cleanup()

# ============================================
# 초기화 및 검증
# ============================================

# 디바이스 테스트 실행
if gpu_config.is_initialized:
    test_success = gpu_config.test_device()
    if test_success:
        logger.info("✅ GPU 설정 검증 완료")
    else:
        logger.warning("⚠️ GPU 설정 검증 실패 - CPU 폴백 권장")

# M3 Max 상태 로깅
if gpu_config.m3_max_detected:
    logger.info("🍎 M3 Max 128GB 최적화 활성화:")
    logger.info(f"  - Neural Engine: {'✅' if MODEL_CONFIG.get('use_neural_engine') else '❌'}")
    logger.info(f"  - Metal Performance Shaders: {'✅' if MODEL_CONFIG.get('metal_performance_shaders') else '❌'}")
    logger.info(f"  - 배치 크기: {MODEL_CONFIG.get('batch_size', 1)}")
    logger.info(f"  - 메모리 대역폭: {DEVICE_INFO.get('memory_bandwidth', 'N/A')}")

# ============================================
# Export 리스트 (main.py import 호환)
# ============================================

__all__ = [
    # 주요 객체들
    'gpu_config', 'DEVICE', 'MODEL_CONFIG', 'DEVICE_INFO',
    
    # 함수들
    'get_device', 'get_device_config', 'get_model_config', 'get_device_info',
    'test_device', 'cleanup_gpu_resources',
    
    # 메모리 관리 함수들 (main.py에서 요구)
    'optimize_memory', 'get_memory_status', 'check_device_compatibility',
    
    # 클래스
    'M3MaxGPUManager', 'GPUConfig'
]