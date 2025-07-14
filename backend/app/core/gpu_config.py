"""
GPU/MPS 최적화 설정
Apple Silicon M3 Max 및 CUDA 환경에서의 최적 성능 제공
"""
import os
import logging
import platform
from typing import Dict, Any, Optional, Tuple
import torch
import gc
import psutil

logger = logging.getLogger(__name__)

class GPUConfig:
    """GPU/MPS 최적화 설정 관리자"""
    
    def __init__(self):
        """GPU 설정 초기화"""
        self.device_type = self._detect_device()
        self.device = torch.device(self.device_type)
        self.is_apple_silicon = self._is_apple_silicon()
        
        # 메모리 설정
        self.memory_settings = self._configure_memory()
        
        # 최적화 설정
        self.optimization_settings = self._configure_optimization()
        
        logger.info(f"🔧 GPU 설정 완료 - 디바이스: {self.device_type}")
        self._log_system_info()
    
    def _detect_device(self) -> str:
        """최적 디바이스 감지"""
        # 환경 변수 우선 확인
        forced_device = os.environ.get('FORCE_DEVICE', '').lower()
        if forced_device in ['cpu', 'cuda', 'mps']:
            logger.info(f"🎯 강제 디바이스 설정: {forced_device}")
            return forced_device
        
        # Apple Silicon MPS 확인
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if torch.backends.mps.is_built():
                logger.info("🍎 Apple Silicon MPS 감지됨")
                return "mps"
        
        # CUDA 확인
        if torch.cuda.is_available():
            cuda_device_count = torch.cuda.device_count()
            logger.info(f"🚀 CUDA 감지됨 - GPU 개수: {cuda_device_count}")
            return "cuda"
        
        # CPU 폴백
        logger.info("💻 CPU 모드로 설정")
        return "cpu"
    
    def _is_apple_silicon(self) -> bool:
        """Apple Silicon 여부 확인"""
        system = platform.system()
        machine = platform.machine()
        
        if system == "Darwin" and machine == "arm64":
            return True
        
        # M 시리즈 칩 직접 확인
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2
            )
            if "Apple" in result.stdout:
                return True
        except:
            pass
        
        return False
    
    def _configure_memory(self) -> Dict[str, Any]:
        """메모리 설정"""
        # 시스템 메모리 정보
        system_memory = psutil.virtual_memory()
        total_memory_gb = system_memory.total / (1024**3)
        
        settings = {
            "total_system_memory_gb": total_memory_gb,
            "reserved_system_memory_gb": 4.0,  # 시스템용 예약
            "max_model_memory_gb": min(total_memory_gb * 0.6, 32.0)  # 모델용 최대
        }
        
        if self.device_type == "mps":
            # Apple Silicon MPS 설정
            # M3 Max는 보통 36GB 또는 128GB unified memory
            if total_memory_gb >= 64:
                settings.update({
                    "mps_memory_fraction": 0.7,  # 70% 사용
                    "batch_size_multiplier": 1.5,
                    "enable_memory_mapping": True
                })
            else:
                settings.update({
                    "mps_memory_fraction": 0.6,  # 60% 사용
                    "batch_size_multiplier": 1.0,
                    "enable_memory_mapping": True
                })
            
            # MPS 특화 설정
            settings.update({
                "enable_mixed_precision": True,
                "optimize_for_inference": True,
                "enable_graph_optimization": True
            })
            
        elif self.device_type == "cuda":
            # CUDA 설정
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            settings.update({
                "gpu_memory_gb": gpu_memory,
                "cuda_memory_fraction": min(0.8, (gpu_memory - 2) / gpu_memory),
                "enable_cudnn_benchmark": True,
                "enable_amp": True  # Automatic Mixed Precision
            })
            
            # CUDA 메모리 설정
            try:
                torch.cuda.set_per_process_memory_fraction(settings["cuda_memory_fraction"])
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            except Exception as e:
                logger.warning(f"CUDA 메모리 설정 실패: {e}")
        
        else:  # CPU
            # CPU 설정
            cpu_count = psutil.cpu_count(logical=False)
            settings.update({
                "cpu_cores": cpu_count,
                "thread_count": min(cpu_count, 8),  # 최대 8 스레드
                "enable_mkldnn": True
            })
            
            # CPU 최적화
            torch.set_num_threads(settings["thread_count"])
        
        return settings
    
    def _configure_optimization(self) -> Dict[str, Any]:
        """최적화 설정"""
        settings = {
            "compile_models": True,  # PyTorch 2.0+ 컴파일
            "use_channels_last": True,  # 메모리 효율성
            "enable_jit": True,  # JIT 컴파일
            "gradient_accumulation": True,
            "model_quantization": {
                "enabled": True,
                "mode": "dynamic"  # dynamic, static, qat
            }
        }
        
        if self.device_type == "mps":
            # Apple Silicon 특화 최적화
            settings.update({
                "mps_optimizations": {
                    "enable_fusion": True,
                    "optimize_memory_layout": True,
                    "use_metal_performance_shaders": True,
                    "enable_graph_capture": True
                },
                "batch_processing": {
                    "optimal_batch_size": 4 if self.memory_settings.get("mps_memory_fraction", 0.6) > 0.65 else 2,
                    "dynamic_batching": True
                }
            })
            
        elif self.device_type == "cuda":
            # CUDA 특화 최적화
            settings.update({
                "cuda_optimizations": {
                    "enable_tensor_core": True,
                    "use_half_precision": True,
                    "optimize_attention": True,
                    "enable_flash_attention": True
                },
                "batch_processing": {
                    "optimal_batch_size": 8,
                    "gradient_checkpointing": True
                }
            })
        
        return settings
    
    def get_optimal_device(self) -> str:
        """최적 디바이스 반환"""
        return self.device_type
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """모델별 최적 설정 반환"""
        base_config = {
            "device": self.device,
            "dtype": torch.float32
        }
        
        if self.device_type == "mps":
            # MPS는 float16 지원 제한적
            base_config.update({
                "dtype": torch.float32,
                "use_compile": True,
                "memory_format": torch.channels_last if self.optimization_settings["use_channels_last"] else torch.contiguous_format
            })
            
        elif self.device_type == "cuda":
            # CUDA 최적화
            base_config.update({
                "dtype": torch.float16 if self.optimization_settings["cuda_optimizations"]["use_half_precision"] else torch.float32,
                "use_compile": self.optimization_settings["compile_models"],
                "memory_format": torch.channels_last
            })
        
        # 모델 타입별 특화 설정
        model_specific = self._get_model_specific_config(model_type)
        base_config.update(model_specific)
        
        return base_config
    
    def _get_model_specific_config(self, model_type: str) -> Dict[str, Any]:
        """모델 타입별 특화 설정"""
        configs = {
            "human_parsing": {
                "batch_size": 1,
                "input_size": (512, 512),
                "enable_optimization": True
            },
            "pose_estimation": {
                "batch_size": 1,
                "input_size": (256, 192),
                "keypoint_threshold": 0.3
            },
            "cloth_segmentation": {
                "batch_size": 1,
                "input_size": (320, 320),
                "enable_postprocessing": True
            },
            "virtual_fitting": {
                "batch_size": 1,
                "input_size": (512, 512),
                "num_inference_steps": 20,
                "guidance_scale": 7.5
            }
        }
        
        return configs.get(model_type, {})
    
    def optimize_model(self, model: torch.nn.Module, model_type: str) -> torch.nn.Module:
        """모델 최적화 적용"""
        try:
            # 디바이스로 이동
            model = model.to(self.device)
            
            # 평가 모드
            model.eval()
            
            # 메모리 포맷 최적화
            if self.optimization_settings["use_channels_last"]:
                model = model.to(memory_format=torch.channels_last)
            
            # 컴파일 최적화 (PyTorch 2.0+)
            if self.optimization_settings["compile_models"]:
                try:
                    if hasattr(torch, 'compile'):
                        if self.device_type == "mps":
                            # MPS는 일부 compile 기능 제한
                            model = torch.compile(model, mode="reduce-overhead")
                        else:
                            model = torch.compile(model, mode="max-autotune")
                        logger.info(f"✅ {model_type} 모델 컴파일 완료")
                except Exception as e:
                    logger.warning(f"모델 컴파일 실패 (계속 진행): {e}")
            
            # 양자화 (추론 전용)
            if self.optimization_settings["model_quantization"]["enabled"]:
                try:
                    if self.device_type == "cpu":
                        # CPU에서만 양자화 적용
                        model = torch.quantization.quantize_dynamic(
                            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                        )
                        logger.info(f"✅ {model_type} 모델 양자화 완료")
                except Exception as e:
                    logger.warning(f"모델 양자화 실패 (계속 진행): {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"모델 최적화 실패: {e}")
            return model
    
    def setup_memory_optimization(self):
        """메모리 최적화 설정"""
        try:
            if self.device_type == "mps":
                # MPS 메모리 최적화
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
                
                # Metal 성능 최적화
                if self.is_apple_silicon:
                    os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'
                    os.environ['METAL_DEBUG_ERROR_MODE'] = '0'
                
            elif self.device_type == "cuda":
                # CUDA 메모리 최적화
                torch.cuda.empty_cache()
                
                # CUDA 그래프 최적화
                if hasattr(torch.cuda, 'memory_stats'):
                    logger.info("CUDA 메모리 통계 활성화")
            
            # 공통 최적화
            torch.backends.opt_einsum.enabled = True
            
            # 가비지 컬렉션 최적화
            gc.collect()
            
            logger.info("✅ 메모리 최적화 설정 완료")
            
        except Exception as e:
            logger.warning(f"메모리 최적화 설정 실패: {e}")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """현재 메모리 사용량 정보"""
        info = {
            "device": self.device_type,
            "system_memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "used_percent": psutil.virtual_memory().percent
            }
        }
        
        if self.device_type == "cuda":
            info["gpu_memory"] = {
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3)
            }
        elif self.device_type == "mps":
            info["mps_memory"] = {
                "current_allocated_gb": torch.mps.current_allocated_memory() / (1024**3),
                "driver_allocated_gb": torch.mps.driver_allocated_memory() / (1024**3)
            }
        
        return info
    
    def cleanup_memory(self):
        """메모리 정리"""
        try:
            # PyTorch 캐시 정리
            if self.device_type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif self.device_type == "mps":
                torch.mps.empty_cache()
                torch.mps.synchronize()
            
            # 가비지 컬렉션
            gc.collect()
            
        except Exception as e:
            logger.warning(f"메모리 정리 중 오류: {e}")
    
    def _log_system_info(self):
        """시스템 정보 로깅"""
        logger.info("=" * 50)
        logger.info("🖥️  시스템 정보")
        logger.info(f"플랫폼: {platform.platform()}")
        logger.info(f"프로세서: {platform.processor()}")
        logger.info(f"아키텍처: {platform.machine()}")
        
        if self.is_apple_silicon:
            logger.info("🍎 Apple Silicon 감지됨")
        
        # PyTorch 정보
        logger.info(f"PyTorch 버전: {torch.__version__}")
        logger.info(f"디바이스: {self.device_type}")
        
        if self.device_type == "mps":
            logger.info(f"MPS 사용 가능: {torch.backends.mps.is_available()}")
            logger.info(f"MPS 빌드됨: {torch.backends.mps.is_built()}")
        elif self.device_type == "cuda":
            logger.info(f"CUDA 버전: {torch.version.cuda}")
            logger.info(f"GPU 이름: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
        
        # 메모리 정보
        memory_info = self.get_memory_info()
        logger.info(f"시스템 메모리: {memory_info['system_memory']['total_gb']:.1f}GB")
        logger.info("=" * 50)


# 전역 함수들 추가 (호환성 유지)
def gpu_config():
    """GPU 설정 함수 - 호환성을 위한 래퍼"""
    return GPUConfig()

def get_optimal_device():
    """최적 디바이스 반환"""
    config = GPUConfig()
    return config.get_optimal_device()

def get_device_config():
    """디바이스 설정 반환"""
    config = GPUConfig()
    return {
        'device': config.device_type,
        'memory': f"{config.memory_settings['total_system_memory_gb']:.0f}GB"
    }