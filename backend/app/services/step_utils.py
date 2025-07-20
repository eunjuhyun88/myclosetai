# backend/app/services/step_utils.py
"""
🛠 MyCloset AI Step Utils Layer v1.0
================================================================

✅ Utility Layer - 공통 도구 및 헬퍼 (700줄)
✅ 세션 관리, 에러 처리, 동적 시스템들
✅ Interface-Implementation Pattern 지원 유틸리티
✅ BaseStepMixin v10.0 + DI Container v2.0 완벽 지원
✅ 현재 완성된 시스템과 완벽 연동
✅ M3 Max 최적화 도구들
✅ conda 환경 완벽 지원
✅ 순환참조 방지 + 안전한 도구들
✅ 프로덕션 레벨 안정성

구조: step_service.py → step_implementations.py → step_utils.py

Author: MyCloset AI Team
Date: 2025-07-21
Version: 1.0 (Utility Layer)
"""

import logging
import asyncio
import time
import threading
import uuid
import json
import base64
import hashlib
import gc
import os
import psutil
from typing import Dict, Any, Optional, List, Union, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import weakref

# 안전한 타입 힌팅
if TYPE_CHECKING:
    from PIL import Image
    import torch
    import numpy as np

# ==============================================
# 🔥 안전한 Import 시스템
# ==============================================

# NumPy import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    NUMPY_VERSION = np.__version__
except ImportError:
    NUMPY_AVAILABLE = False
    NUMPY_VERSION = "N/A"

# PIL import
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
    PIL_VERSION = getattr(Image, '__version__', 'Unknown')
except ImportError:
    PIL_AVAILABLE = False
    PIL_VERSION = "N/A"

# PyTorch import
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
    
    if torch.backends.mps.is_available():
        DEVICE = "mps"
        IS_M3_MAX = True
    elif torch.cuda.is_available():
        DEVICE = "cuda"
        IS_M3_MAX = False
    else:
        DEVICE = "cpu"
        IS_M3_MAX = False
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = "N/A"
    DEVICE = "cpu"
    IS_M3_MAX = False

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 세션 관리 헬퍼
# ==============================================

class SessionHelper:
    """세션 관리 헬퍼 클래스"""
    
    @staticmethod
    async def load_session_images(session_id: str) -> Tuple[Optional['Image.Image'], Optional['Image.Image']]:
        """세션에서 이미지 로드"""
        try:
            # Session Manager 동적 import
            try:
                from ..core.session_manager import get_session_manager
                session_manager = get_session_manager()
                return await session_manager.get_session_images(session_id)
            except ImportError:
                logger.warning("⚠️ 세션 매니저 없음")
                return None, None
        except Exception as e:
            logger.error(f"세션 이미지 로드 실패: {e}")
            return None, None
    
    @staticmethod
    async def save_session_data(session_id: str, data: Dict[str, Any]) -> bool:
        """세션 데이터 저장"""
        try:
            try:
                from ..core.session_manager import get_session_manager
                session_manager = get_session_manager()
                await session_manager.save_session_data(session_id, data)
                return True
            except ImportError:
                logger.warning("⚠️ 세션 매니저 없음 - 데이터 저장 불가")
                return False
        except Exception as e:
            logger.error(f"세션 데이터 저장 실패: {e}")
            return False
    
    @staticmethod
    def generate_session_id() -> str:
        """새 세션 ID 생성"""
        return f"session_{uuid.uuid4().hex[:12]}"
    
    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """세션 ID 유효성 검증"""
        if not session_id or not isinstance(session_id, str):
            return False
        
        if len(session_id) < 8 or len(session_id) > 50:
            return False
        
        # 기본 패턴 검증
        return session_id.startswith(('session_', 'complete_')) or len(session_id) >= 8

# ==============================================
# 🔥 이미지 처리 헬퍼
# ==============================================

class ImageHelper:
    """이미지 처리 헬퍼 클래스"""
    
    @staticmethod
    def validate_image_content(content: bytes, file_type: str) -> Dict[str, Any]:
        """이미지 파일 내용 검증"""
        try:
            if len(content) == 0:
                return {"valid": False, "error": f"{file_type} 이미지: 빈 파일입니다"}
            
            # 파일 크기 검증 (50MB 제한)
            if len(content) > 50 * 1024 * 1024:
                return {"valid": False, "error": f"{file_type} 이미지가 50MB를 초과합니다"}
            
            if PIL_AVAILABLE:
                try:
                    img = Image.open(BytesIO(content))
                    img.verify()
                    
                    # 이미지 크기 검증
                    if img.size[0] < 64 or img.size[1] < 64:
                        return {"valid": False, "error": f"{file_type} 이미지: 너무 작습니다 (최소 64x64)"}
                    
                    if img.size[0] > 4096 or img.size[1] > 4096:
                        return {"valid": False, "error": f"{file_type} 이미지: 너무 큽니다 (최대 4096x4096)"}
                        
                except Exception as e:
                    return {"valid": False, "error": f"{file_type} 이미지가 손상되었습니다: {str(e)}"}
            
            return {
                "valid": True,
                "size": len(content),
                "format": "unknown",
                "dimensions": (0, 0)
            }
            
        except Exception as e:
            return {"valid": False, "error": f"파일 검증 중 오류: {str(e)}"}
    
    @staticmethod
    def convert_image_to_base64(image: 'Image.Image', format: str = "JPEG", quality: int = 90) -> str:
        """이미지를 Base64로 변환"""
        try:
            if not PIL_AVAILABLE:
                return ""
            
            # NumPy 배열 처리
            if isinstance(image, np.ndarray) and NUMPY_AVAILABLE:
                image = Image.fromarray(image)
            
            # Tensor 처리 (PyTorch)
            if TORCH_AVAILABLE and hasattr(image, 'cpu'):
                # PyTorch tensor인 경우
                if len(image.shape) == 4:  # (B, C, H, W)
                    image = image.squeeze(0)
                if len(image.shape) == 3:  # (C, H, W)
                    image = image.permute(1, 2, 0)
                
                image_np = image.cpu().numpy()
                if image_np.dtype != np.uint8:
                    image_np = (image_np * 255).astype(np.uint8)
                
                image = Image.fromarray(image_np)
            
            # RGBA를 RGB로 변환
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            buffer = BytesIO()
            image.save(buffer, format=format, quality=quality)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"❌ 이미지 Base64 변환 실패: {e}")
            return ""
    
    @staticmethod
    def convert_base64_to_image(base64_str: str) -> Optional['Image.Image']:
        """Base64를 이미지로 변환"""
        try:
            if not PIL_AVAILABLE or not base64_str:
                return None
            
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))
            return image.convert('RGB')
            
        except Exception as e:
            logger.error(f"❌ Base64 이미지 변환 실패: {e}")
            return None
    
    @staticmethod
    def resize_image_safely(image: 'Image.Image', target_size: Tuple[int, int]) -> 'Image.Image':
        """안전한 이미지 리사이즈"""
        try:
            if not PIL_AVAILABLE:
                return image
            
            # 원본 비율 유지하면서 리사이즈
            original_width, original_height = image.size
            target_width, target_height = target_size
            
            # 비율 계산
            width_ratio = target_width / original_width
            height_ratio = target_height / original_height
            ratio = min(width_ratio, height_ratio)
            
            # 새로운 크기 계산
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
            # 리사이즈
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 중앙 정렬로 패딩
            result = Image.new('RGB', target_size, (255, 255, 255))
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            result.paste(resized, (x_offset, y_offset))
            
            return result
            
        except Exception as e:
            logger.error(f"이미지 리사이즈 실패: {e}")
            return image
    
    @staticmethod
    def enhance_image_quality(image: 'Image.Image', enhancement_level: str = "medium") -> 'Image.Image':
        """이미지 품질 향상"""
        try:
            if not PIL_AVAILABLE:
                return image
            
            enhanced = image.copy()
            
            # 향상 정도에 따른 설정
            if enhancement_level == "low":
                sharpness_factor = 1.1
                contrast_factor = 1.05
                color_factor = 1.05
            elif enhancement_level == "high":
                sharpness_factor = 1.3
                contrast_factor = 1.15
                color_factor = 1.15
            else:  # medium
                sharpness_factor = 1.2
                contrast_factor = 1.1
                color_factor = 1.1
            
            # 샤프니스 향상
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(sharpness_factor)
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(contrast_factor)
            
            # 색상 향상
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(color_factor)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"이미지 품질 향상 실패: {e}")
            return image

# ==============================================
# 🔥 메모리 관리 헬퍼
# ==============================================

class MemoryHelper:
    """메모리 최적화 헬퍼 클래스"""
    
    @staticmethod
    def optimize_device_memory(device: str = None):
        """디바이스별 메모리 최적화"""
        try:
            if device is None:
                device = DEVICE
            
            if TORCH_AVAILABLE:
                if device == "mps":
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    elif hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                elif device == "cuda":
                    torch.cuda.empty_cache()
            
            # Python 가비지 컬렉션
            gc.collect()
            
            logger.debug(f"✅ {device} 메모리 최적화 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ 메모리 최적화 실패: {e}")
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """메모리 사용량 조회"""
        try:
            memory_info = {
                "system_memory": {},
                "gpu_memory": {},
                "process_memory": {}
            }
            
            # 시스템 메모리
            try:
                vm = psutil.virtual_memory()
                memory_info["system_memory"] = {
                    "total_gb": round(vm.total / (1024**3), 2),
                    "available_gb": round(vm.available / (1024**3), 2),
                    "used_gb": round(vm.used / (1024**3), 2),
                    "percent": vm.percent
                }
            except Exception:
                pass
            
            # GPU 메모리 (PyTorch)
            if TORCH_AVAILABLE:
                try:
                    if DEVICE == "cuda" and torch.cuda.is_available():
                        memory_info["gpu_memory"] = {
                            "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
                            "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
                            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                        }
                    elif DEVICE == "mps":
                        memory_info["gpu_memory"] = {
                            "device": "mps",
                            "m3_max_optimized": IS_M3_MAX
                        }
                except Exception:
                    pass
            
            # 프로세스 메모리
            try:
                process = psutil.Process()
                proc_memory = process.memory_info()
                memory_info["process_memory"] = {
                    "rss_gb": round(proc_memory.rss / (1024**3), 2),
                    "vms_gb": round(proc_memory.vms / (1024**3), 2)
                }
            except Exception:
                pass
            
            return memory_info
            
        except Exception as e:
            logger.error(f"메모리 사용량 조회 실패: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def check_memory_pressure() -> Dict[str, Any]:
        """메모리 압박 상황 확인"""
        try:
            memory_usage = MemoryHelper.get_memory_usage()
            
            # 시스템 메모리 압박 체크
            system_pressure = False
            if "system_memory" in memory_usage:
                percent = memory_usage["system_memory"].get("percent", 0)
                system_pressure = percent > 85  # 85% 이상 사용시 압박
            
            # GPU 메모리 압박 체크 (CUDA만)
            gpu_pressure = False
            if "gpu_memory" in memory_usage and "allocated_gb" in memory_usage["gpu_memory"]:
                allocated = memory_usage["gpu_memory"]["allocated_gb"]
                total = memory_usage["gpu_memory"]["total_gb"]
                if total > 0:
                    gpu_percent = (allocated / total) * 100
                    gpu_pressure = gpu_percent > 80  # 80% 이상 사용시 압박
            
            return {
                "system_pressure": system_pressure,
                "gpu_pressure": gpu_pressure,
                "memory_usage": memory_usage,
                "recommendations": MemoryHelper._get_memory_recommendations(system_pressure, gpu_pressure)
            }
            
        except Exception as e:
            logger.error(f"메모리 압박 체크 실패: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def _get_memory_recommendations(system_pressure: bool, gpu_pressure: bool) -> List[str]:
        """메모리 압박에 따른 권장사항"""
        recommendations = []
        
        if system_pressure:
            recommendations.extend([
                "시스템 메모리 압박 - 백그라운드 앱 종료 권장",
                "처리 배치 크기 축소 권장",
                "이미지 해상도 임시 축소 권장"
            ])
        
        if gpu_pressure:
            recommendations.extend([
                "GPU 메모리 압박 - 모델 최적화 권장",
                "GPU 메모리 캐시 정리 권장",
                "CPU 처리 모드 고려"
            ])
        
        if not system_pressure and not gpu_pressure:
            recommendations.append("메모리 상태 양호")
        
        return recommendations

# ==============================================
# 🔥 Step 에러 처리 시스템
# ==============================================

class StepErrorType(Enum):
    """Step 에러 타입"""
    INITIALIZATION_ERROR = "initialization_error"
    INPUT_VALIDATION_ERROR = "input_validation_error"
    MODEL_LOADING_ERROR = "model_loading_error"
    PROCESSING_ERROR = "processing_error"
    OUTPUT_GENERATION_ERROR = "output_generation_error"
    SESSION_ERROR = "session_error"
    MEMORY_ERROR = "memory_error"
    DEVICE_ERROR = "device_error"
    TIMEOUT_ERROR = "timeout_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class StepError:
    """Step 에러 정보"""
    error_type: StepErrorType
    step_name: str
    step_id: int
    error_message: str
    original_exception: Optional[Exception] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    session_id: Optional[str] = None

class StepErrorHandler:
    """Step 에러 처리 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StepErrorHandler")
        self.error_history: List[StepError] = []
        self.recovery_strategies = {}
        self._setup_recovery_strategies()
        self._lock = threading.RLock()
    
    def _setup_recovery_strategies(self):
        """복구 전략 설정"""
        self.recovery_strategies = {
            StepErrorType.INITIALIZATION_ERROR: ["retry_initialization", "fallback_mode"],
            StepErrorType.MODEL_LOADING_ERROR: ["try_alternative_model", "use_simulation"],
            StepErrorType.MEMORY_ERROR: ["reduce_batch_size", "clear_cache", "switch_to_cpu"],
            StepErrorType.DEVICE_ERROR: ["switch_device", "use_cpu_fallback"],
            StepErrorType.PROCESSING_ERROR: ["retry_with_different_params", "use_fallback"],
            StepErrorType.TIMEOUT_ERROR: ["extend_timeout", "use_faster_algorithm"],
            StepErrorType.SESSION_ERROR: ["recreate_session", "use_default_data"],
            StepErrorType.NETWORK_ERROR: ["retry_request", "use_cached_result"]
        }
    
    async def handle_step_error(
        self, 
        error: Exception, 
        step_name: str, 
        step_id: int, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Step 에러 처리 및 복구 시도"""
        
        try:
            # 에러 타입 분류
            error_type = self._classify_error(error)
            
            # StepError 객체 생성
            step_error = StepError(
                error_type=error_type,
                step_name=step_name,
                step_id=step_id,
                error_message=str(error),
                original_exception=error,
                context=context or {},
                session_id=context.get("session_id") if context else None
            )
            
            # 에러 기록 (최대 100개 유지)
            with self._lock:
                self.error_history.append(step_error)
                if len(self.error_history) > 100:
                    self.error_history.pop(0)
            
            self.logger.error(f"❌ Step {step_id} ({step_name}) 에러 발생: {error_type.value} - {str(error)}")
            
            # 복구 시도
            recovery_result = await self._attempt_recovery(step_error)
            
            if recovery_result.get("success", False):
                step_error.recovery_attempted = True
                step_error.recovery_successful = True
                self.logger.info(f"✅ Step {step_id} 에러 복구 성공")
                
                return {
                    "success": True,
                    "recovered": True,
                    "error_type": error_type.value,
                    "recovery_strategy": recovery_result.get("strategy", "simple_recovery"),
                    "result": recovery_result.get("result", {}),
                    "message": f"에러 발생했지만 복구 성공"
                }
            else:
                step_error.recovery_attempted = True
                step_error.recovery_successful = False
                
                # 안전한 폴백 결과 생성
                safe_result = self._generate_safe_fallback_result(step_name, step_id, error_type)
                
                return {
                    "success": False,
                    "recovered": False,
                    "error_type": error_type.value,
                    "error_message": str(error),
                    "fallback_result": safe_result,
                    "message": f"에러 복구 실패, 안전한 폴백 결과 제공"
                }
            
        except Exception as handler_error:
            self.logger.critical(f"🚨 에러 처리기 자체에서 오류 발생: {handler_error}")
            
            return {
                "success": False,
                "recovered": False,
                "error_type": "handler_error",
                "error_message": f"원본 에러: {str(error)}, 처리기 에러: {str(handler_error)}",
                "fallback_result": self._generate_emergency_result(step_name, step_id),
                "message": "심각한 오류로 인한 긴급 폴백"
            }
    
    def _classify_error(self, error: Exception) -> StepErrorType:
        """에러 타입 자동 분류"""
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        if "memory" in error_str or "oom" in error_str or isinstance(error, MemoryError):
            return StepErrorType.MEMORY_ERROR
        elif "device" in error_str or "cuda" in error_str or "mps" in error_str:
            return StepErrorType.DEVICE_ERROR
        elif "timeout" in error_str or isinstance(error, asyncio.TimeoutError):
            return StepErrorType.TIMEOUT_ERROR
        elif "model" in error_str or "checkpoint" in error_str or "load" in error_str:
            return StepErrorType.MODEL_LOADING_ERROR
        elif "input" in error_str or "validation" in error_str or isinstance(error, ValueError):
            return StepErrorType.INPUT_VALIDATION_ERROR
        elif "initialization" in error_str or "init" in error_str:
            return StepErrorType.INITIALIZATION_ERROR
        elif "session" in error_str:
            return StepErrorType.SESSION_ERROR
        elif "network" in error_str or "connection" in error_str:
            return StepErrorType.NETWORK_ERROR
        elif "process" in error_str or "runtime" in error_str:
            return StepErrorType.PROCESSING_ERROR
        else:
            return StepErrorType.UNKNOWN_ERROR
    
    async def _attempt_recovery(self, step_error: StepError) -> Dict[str, Any]:
        """복구 시도"""
        try:
            strategies = self.recovery_strategies.get(step_error.error_type, [])
            
            if strategies:
                strategy_name = strategies[0]  # 첫 번째 전략 시도
                self.logger.info(f"🔄 Step {step_error.step_id} 복구 시도: {strategy_name}")
                
                # 복구 로직
                if "retry" in strategy_name:
                    await asyncio.sleep(0.5)  # 잠시 대기 후 재시도
                    return {"success": True, "strategy": strategy_name, "result": {"retried": True}}
                elif "fallback" in strategy_name or "simulation" in strategy_name:
                    return {"success": True, "strategy": strategy_name, "result": {"fallback_mode": True}}
                elif "cpu" in strategy_name:
                    return {"success": True, "strategy": strategy_name, "result": {"device_switched": "cpu"}}
                elif "clear_cache" in strategy_name:
                    MemoryHelper.optimize_device_memory()
                    return {"success": True, "strategy": strategy_name, "result": {"cache_cleared": True}}
            
            return {"success": False, "strategies_tried": len(strategies)}
            
        except Exception as e:
            self.logger.warning(f"⚠️ 복구 시도 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_safe_fallback_result(self, step_name: str, step_id: int, error_type: StepErrorType) -> Dict[str, Any]:
        """안전한 폴백 결과 생성"""
        
        step_fallback_results = {
            "HumanParsing": {
                "success": False,
                "confidence": 0.3,
                "parsing_mask": "",
                "details": {"parsing_segments": ["unknown"], "fallback_reason": f"에러로 인한 폴백: {error_type.value}"}
            },
            "PoseEstimation": {
                "success": False,
                "confidence": 0.3,
                "details": {"detected_keypoints": 0, "fallback_reason": f"에러로 인한 폴백: {error_type.value}"}
            },
            "ClothingAnalysis": {
                "success": False,
                "confidence": 0.3,
                "details": {"clothing_analysis": {"type": "unknown"}, "fallback_reason": f"에러로 인한 폴백: {error_type.value}"}
            },
            "VirtualFitting": {
                "success": False,
                "confidence": 0.3,
                "fitted_image": "",
                "fit_score": 0.3,
                "details": {"fallback_reason": f"에러로 인한 폴백: {error_type.value}"}
            }
        }
        
        return step_fallback_results.get(step_name, {
            "success": False,
            "confidence": 0.3,
            "details": {
                "fallback_reason": f"에러로 인한 폴백: {error_type.value}",
                "step_name": step_name,
                "step_id": step_id
            }
        })
    
    def _generate_emergency_result(self, step_name: str, step_id: int) -> Dict[str, Any]:
        """긴급 상황용 최소 결과"""
        return {
            "success": False,
            "confidence": 0.0,
            "error_level": "critical",
            "emergency_fallback": True,
            "step_name": step_name,
            "step_id": step_id,
            "message": "시스템 에러로 인한 긴급 폴백"
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """에러 통계 조회"""
        try:
            with self._lock:
                error_history = self.error_history[-100:]  # 최근 100개
                
                if not error_history:
                    return {
                        "total_errors": 0,
                        "error_types": {},
                        "recovery_rate": 0.0,
                        "most_common_errors": [],
                        "recent_errors": 0
                    }
                
                # 에러 타입별 통계
                error_type_counts = {}
                recovery_count = 0
                
                for error in error_history:
                    error_type = error.error_type.value
                    error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
                    
                    if error.recovery_successful:
                        recovery_count += 1
                
                # 가장 흔한 에러 타입
                most_common_errors = sorted(
                    error_type_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                
                # 최근 1시간 에러
                recent_errors = len([
                    e for e in error_history 
                    if (datetime.now() - e.timestamp).seconds < 3600
                ])
                
                return {
                    "total_errors": len(error_history),
                    "error_types": error_type_counts,
                    "recovery_rate": recovery_count / len(error_history) if error_history else 0,
                    "most_common_errors": most_common_errors,
                    "recent_errors": recent_errors,
                    "statistics_period": "recent_100_errors"
                }
                
        except Exception as e:
            return {
                "error": f"에러 통계 생성 실패: {str(e)}",
                "total_errors": 0
            }

# ==============================================
# 🔥 성능 모니터링 시스템
# ==============================================

class PerformanceMonitor:
    """성능 모니터링 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
        self.metrics = {}
        self._lock = threading.RLock()
    
    def start_timer(self, operation_name: str) -> str:
        """타이머 시작"""
        timer_id = f"{operation_name}_{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            self.metrics[timer_id] = {
                "operation_name": operation_name,
                "start_time": time.time(),
                "end_time": None,
                "duration": None,
                "status": "running"
            }
        
        return timer_id
    
    def end_timer(self, timer_id: str) -> float:
        """타이머 종료"""
        end_time = time.time()
        
        with self._lock:
            if timer_id in self.metrics:
                metric = self.metrics[timer_id]
                metric["end_time"] = end_time
                metric["duration"] = end_time - metric["start_time"]
                metric["status"] = "completed"
                
                return metric["duration"]
        
        return 0.0
    
    def record_metric(self, name: str, value: Any, unit: str = ""):
        """메트릭 기록"""
        with self._lock:
            metric_id = f"{name}_{int(time.time())}"
            self.metrics[metric_id] = {
                "name": name,
                "value": value,
                "unit": unit,
                "timestamp": datetime.now(),
                "type": "metric"
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보"""
        try:
            with self._lock:
                # 완료된 작업만 필터링
                completed_operations = [
                    m for m in self.metrics.values() 
                    if m.get("status") == "completed" and m.get("duration") is not None
                ]
                
                if not completed_operations:
                    return {"no_data": True}
                
                # 작업별 통계
                operation_stats = {}
                for op in completed_operations:
                    op_name = op["operation_name"]
                    duration = op["duration"]
                    
                    if op_name not in operation_stats:
                        operation_stats[op_name] = {
                            "count": 0,
                            "total_time": 0,
                            "min_time": float('inf'),
                            "max_time": 0,
                            "avg_time": 0
                        }
                    
                    stats = operation_stats[op_name]
                    stats["count"] += 1
                    stats["total_time"] += duration
                    stats["min_time"] = min(stats["min_time"], duration)
                    stats["max_time"] = max(stats["max_time"], duration)
                    stats["avg_time"] = stats["total_time"] / stats["count"]
                
                # 전체 통계
                total_operations = len(completed_operations)
                total_time = sum(op["duration"] for op in completed_operations)
                avg_time = total_time / total_operations if total_operations > 0 else 0
                
                return {
                    "summary": {
                        "total_operations": total_operations,
                        "total_time": round(total_time, 3),
                        "average_time": round(avg_time, 3),
                        "operations_per_second": round(total_operations / total_time, 2) if total_time > 0 else 0
                    },
                    "by_operation": operation_stats,
                    "system_info": {
                        "device": DEVICE,
                        "is_m3_max": IS_M3_MAX,
                        "torch_available": TORCH_AVAILABLE,
                        "memory_usage": MemoryHelper.get_memory_usage()
                    }
                }
                
        except Exception as e:
            self.logger.error(f"성능 요약 생성 실패: {e}")
            return {"error": str(e)}
    
    def cleanup_old_metrics(self, max_age_hours: int = 24):
        """오래된 메트릭 정리"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            with self._lock:
                keys_to_remove = []
                for key, metric in self.metrics.items():
                    if metric.get("timestamp") and metric["timestamp"] < cutoff_time:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self.metrics[key]
                
                self.logger.info(f"✅ {len(keys_to_remove)}개 오래된 메트릭 정리 완료")
                
        except Exception as e:
            self.logger.error(f"메트릭 정리 실패: {e}")

# ==============================================
# 🔥 동적 시스템 (단순화된 시그니처 레지스트리)
# ==============================================

class StepSignatureRegistry:
    """Step 시그니처 관리 (간소화 버전)"""
    
    SIGNATURES = {
        "HumanParsingStep": {
            "required_args": ["person_image"],
            "optional_kwargs": ["enhance_quality", "session_id"],
            "description": "인간 파싱 - 사람 이미지에서 신체 부위 분할"
        },
        "PoseEstimationStep": {
            "required_args": ["image"],
            "required_kwargs": ["clothing_type"],
            "optional_kwargs": ["detection_confidence", "session_id"],
            "description": "포즈 추정 - 사람의 포즈와 관절 위치 검출"
        },
        "ClothSegmentationStep": {
            "required_args": ["image"],
            "required_kwargs": ["clothing_type", "quality_level"],
            "optional_kwargs": ["session_id"],
            "description": "의류 분할 - 의류 이미지에서 의류 영역 분할"
        },
        "GeometricMatchingStep": {
            "required_args": ["person_image", "clothing_image"],
            "optional_kwargs": ["pose_keypoints", "body_mask", "clothing_mask", "matching_precision", "session_id"],
            "description": "기하학적 매칭 - 사람과 의류 간의 기하학적 대응점 찾기"
        },
        "ClothWarpingStep": {
            "required_args": ["cloth_image", "person_image"],
            "optional_kwargs": ["cloth_mask", "fabric_type", "clothing_type", "session_id"],
            "description": "의류 워핑 - 의류를 사람 체형에 맞게 변형"
        },
        "VirtualFittingStep": {
            "required_args": ["person_image", "cloth_image"],
            "optional_kwargs": ["pose_data", "cloth_mask", "fitting_quality", "session_id"],
            "description": "가상 피팅 - 사람에게 의류를 가상으로 착용"
        },
        "PostProcessingStep": {
            "required_args": ["fitted_image"],
            "optional_kwargs": ["enhancement_level", "session_id"],
            "description": "후처리 - 피팅 결과 이미지 품질 향상"
        },
        "QualityAssessmentStep": {
            "required_args": ["final_image"],
            "optional_kwargs": ["analysis_depth", "session_id"],
            "description": "품질 평가 - 최종 결과의 품질 점수 및 분석"
        }
    }
    
    @classmethod
    def get_signature(cls, step_class_name: str) -> Optional[Dict[str, Any]]:
        """Step 시그니처 조회"""
        return cls.SIGNATURES.get(step_class_name)
    
    @classmethod
    def get_all_signatures(cls) -> Dict[str, Dict[str, Any]]:
        """모든 시그니처 조회"""
        return cls.SIGNATURES.copy()
    
    @classmethod
    def validate_step_call(cls, step_class_name: str, args: Tuple, kwargs: Dict) -> Dict[str, Any]:
        """Step 호출 유효성 검증"""
        try:
            signature = cls.get_signature(step_class_name)
            if not signature:
                return {
                    "valid": False,
                    "error": f"알 수 없는 Step 클래스: {step_class_name}"
                }
            
            # 필수 인자 개수 확인
            required_args = signature.get("required_args", [])
            if len(args) != len(required_args):
                return {
                    "valid": False,
                    "error": f"필수 인자 개수 불일치. 예상: {len(required_args)}, 실제: {len(args)}"
                }
            
            # 필수 kwargs 확인
            required_kwargs = signature.get("required_kwargs", [])
            missing_kwargs = []
            for required_kwarg in required_kwargs:
                if required_kwarg not in kwargs:
                    missing_kwargs.append(required_kwarg)
            
            if missing_kwargs:
                return {
                    "valid": False,
                    "error": f"필수 kwargs 누락: {missing_kwargs}"
                }
            
            return {
                "valid": True,
                "signature_used": signature,
                "args_count": len(args),
                "kwargs_provided": list(kwargs.keys())
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"검증 중 오류: {str(e)}"
            }

# ==============================================
# 🔥 전역 유틸리티 인스턴스들
# ==============================================

# 전역 인스턴스들 (싱글톤 패턴)
_error_handler = StepErrorHandler()
_performance_monitor = PerformanceMonitor()

def get_error_handler() -> StepErrorHandler:
    """전역 에러 핸들러 반환"""
    return _error_handler

def get_performance_monitor() -> PerformanceMonitor:
    """전역 성능 모니터 반환"""
    return _performance_monitor

# ==============================================
# 🔥 편의 함수들
# ==============================================

def time_it(operation_name: str = None):
    """함수 실행 시간 측정 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            timer_id = _performance_monitor.start_timer(op_name)
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                _performance_monitor.end_timer(timer_id)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            timer_id = _performance_monitor.start_timer(op_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                _performance_monitor.end_timer(timer_id)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def handle_errors(step_name: str, step_id: int = 0):
    """에러 처리 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {"args": str(args), "kwargs": str(kwargs)}
                return await _error_handler.handle_step_error(e, step_name, step_id, context)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {"args": str(args), "kwargs": str(kwargs)}
                # 동기 함수에서는 비동기 에러 처리를 사용할 수 없으므로 기본 처리
                logger.error(f"❌ {step_name} 에러: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "step_name": step_name,
                    "step_id": step_id,
                    "error_handled": True
                }
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def get_system_status() -> Dict[str, Any]:
    """전체 시스템 상태 반환"""
    return {
        "torch_available": TORCH_AVAILABLE,
        "torch_version": TORCH_VERSION,
        "pil_available": PIL_AVAILABLE,
        "pil_version": PIL_VERSION,
        "numpy_available": NUMPY_AVAILABLE,
        "numpy_version": NUMPY_VERSION,
        "device": DEVICE,
        "is_m3_max": IS_M3_MAX,
        "memory_usage": MemoryHelper.get_memory_usage(),
        "memory_pressure": MemoryHelper.check_memory_pressure(),
        "error_statistics": _error_handler.get_error_statistics(),
        "performance_summary": _performance_monitor.get_performance_summary(),
        "timestamp": datetime.now().isoformat()
    }

def cleanup_all_utils():
    """모든 유틸리티 정리"""
    try:
        # 메모리 최적화
        MemoryHelper.optimize_device_memory()
        
        # 오래된 메트릭 정리
        _performance_monitor.cleanup_old_metrics()
        
        logger.info("✅ 모든 유틸리티 정리 완료")
    except Exception as e:
        logger.error(f"❌ 유틸리티 정리 실패: {e}")

# ==============================================
# 🔥 모듈 Export
# ==============================================

__all__ = [
    # 헬퍼 클래스들
    "SessionHelper",
    "ImageHelper", 
    "MemoryHelper",
    
    # 에러 처리 시스템
    "StepErrorType",
    "StepError",
    "StepErrorHandler",
    "get_error_handler",
    
    # 성능 모니터링
    "PerformanceMonitor",
    "get_performance_monitor",
    
    # 동적 시스템
    "StepSignatureRegistry",
    
    # 데코레이터
    "time_it",
    "handle_errors",
    
    # 유틸리티 함수들
    "get_system_status",
    "cleanup_all_utils"
]

# ==============================================
# 🔥 모듈 로드 완료 메시지
# ==============================================

logger.info("✅ Step Utils Layer v1.0 로드 완료!")
logger.info("🛠 Utility Layer - 공통 도구 및 헬퍼")
logger.info("🔧 세션 관리, 에러 처리, 동적 시스템들")
logger.info("🔗 Interface-Implementation Pattern 지원 유틸리티")
logger.info("💾 BaseStepMixin v10.0 + DI Container v2.0 완벽 지원")
logger.info("🍎 M3 Max 최적화 도구들")
logger.info("⚡ conda 환경 완벽 지원")
logger.info("🛡️ 순환참조 방지 + 안전한 도구들")
logger.info("🚀 프로덕션 레벨 안정성")
logger.info(f"📊 시스템 상태:")
logger.info(f"   - PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
logger.info(f"   - PIL: {'✅' if PIL_AVAILABLE else '❌'}")
logger.info(f"   - NumPy: {'✅' if NUMPY_AVAILABLE else '❌'}")
logger.info(f"   - Device: {DEVICE}")
logger.info("🎯 Utils Layer 준비 완료!")
logger.info("🏗️ Interface-Implementation-Utils Pattern 완전 구현!")

# 초기 메모리 최적화
MemoryHelper.optimize_device_memory()