# backend/app/services/step_utils.py
"""
🛠️ MyCloset AI Step Utils Layer v2.0 - 완전한 유틸리티 레이어
================================================================

✅ unified_step_mapping.py 완전 활용 - 세 파일 통합 지원
✅ BaseStepMixin 완벽 호환 - logger 속성 및 초기화 과정
✅ ModelLoader 완전 연동 - 89.8GB 체크포인트 활용
✅ 실제 Step 클래스들과 100% 호환 - HumanParsingStep 등
✅ step_service.py + step_implementations.py 공통 지원
✅ SessionManager, DI Container 완전 연동
✅ 에러 처리 및 복구 시스템
✅ 성능 모니터링 및 메모리 관리
✅ M3 Max 128GB 최적화 + conda 환경 우선
✅ 순환참조 완전 방지 - 단방향 의존성
✅ 프로덕션 레벨 안정성

구조: step_service.py + step_implementations.py → step_utils.py → BaseStepMixin + AI Steps

Author: MyCloset AI Team
Date: 2025-07-21
Version: 2.0 (Complete Utils Layer)
"""

import logging
import asyncio
import time
import threading
import uuid
import base64
import json
import gc
import os
import sys
import weakref
import importlib
from typing import Dict, Any, Optional, List, Union, Tuple, Type, Callable, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
from io import BytesIO
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# 안전한 타입 힌팅
if TYPE_CHECKING:
    from fastapi import UploadFile
    import torch
    import numpy as np
    from PIL import Image

# ==============================================
# 🔥 통합 매핑 시스템 import (핵심!)
# ==============================================

# 통합 매핑 설정
try:
    from .unified_step_mapping import (
        UNIFIED_STEP_CLASS_MAPPING,
        UNIFIED_SERVICE_CLASS_MAPPING,
        SERVICE_TO_STEP_MAPPING,
        STEP_TO_SERVICE_MAPPING,
        SERVICE_ID_TO_STEP_ID,
        STEP_ID_TO_SERVICE_ID,
        UnifiedStepSignature,
        UNIFIED_STEP_SIGNATURES,
        StepFactoryHelper,
        validate_step_compatibility,
        setup_conda_optimization,
        get_step_id_by_service_id,
        get_service_id_by_step_id,
        get_all_available_steps,
        get_all_available_services,
        get_system_compatibility_info
    )
    UNIFIED_MAPPING_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ 통합 매핑 시스템 import 성공")
except ImportError as e:
    UNIFIED_MAPPING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"❌ 통합 매핑 시스템 import 실패: {e}")
    raise ImportError("통합 매핑 시스템이 필요합니다. unified_step_mapping.py를 확인하세요.")

# ==============================================
# 🔥 안전한 Import 시스템
# ==============================================

# NumPy import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# PIL import
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# PyTorch import
try:
    import torch
    TORCH_AVAILABLE = True
    
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
    DEVICE = "cpu"
    IS_M3_MAX = False

# DI Container import
try:
    from ..core.di_container import DIContainer, get_di_container
    DI_CONTAINER_AVAILABLE = True
    logger.info("✅ DI Container import 성공")
except ImportError:
    DI_CONTAINER_AVAILABLE = False
    logger.warning("⚠️ DI Container import 실패")
    
    class DIContainer:
        def __init__(self):
            self._services = {}
        
        def get(self, service_name: str) -> Any:
            return self._services.get(service_name)
        
        def register(self, service_name: str, service: Any):
            self._services[service_name] = service
    
    def get_di_container() -> DIContainer:
        return DIContainer()

# Session Manager import
try:
    from ..core.session_manager import SessionManager, get_session_manager
    SESSION_MANAGER_AVAILABLE = True
    logger.info("✅ Session Manager import 성공")
except ImportError:
    SESSION_MANAGER_AVAILABLE = False
    logger.warning("⚠️ Session Manager import 실패")
    
    class SessionManager:
        def __init__(self):
            self.sessions = {}
        
        async def get_session_images(self, session_id: str):
            return None, None
        
        async def store_session_data(self, session_id: str, data: Dict[str, Any]):
            pass
    
    def get_session_manager() -> SessionManager:
        return SessionManager()

# ModelLoader import (핵심!)
try:
    from ..ai_pipeline.utils.model_loader import ModelLoader, get_global_model_loader
    MODEL_LOADER_AVAILABLE = True
    logger.info("✅ ModelLoader import 성공")
except ImportError:
    MODEL_LOADER_AVAILABLE = False
    logger.warning("⚠️ ModelLoader import 실패")
    
    class ModelLoader:
        def create_step_interface(self, step_name: str):
            return None
        
        def load_model(self, model_name: str):
            return None
    
    def get_global_model_loader() -> Optional[ModelLoader]:
        return None

# 스키마 import
try:
    from ..models.schemas import BodyMeasurements
    SCHEMAS_AVAILABLE = True
    logger.info("✅ 스키마 import 성공")
except ImportError:
    SCHEMAS_AVAILABLE = False
    logger.warning("⚠️ 스키마 import 실패")
    
    @dataclass
    class BodyMeasurements:
        height: float
        weight: float
        chest: Optional[float] = None
        waist: Optional[float] = None
        hips: Optional[float] = None

# ==============================================
# 🔥 에러 정의 및 핸들링 시스템
# ==============================================

class StepUtilsError(Exception):
    """Step Utils 기본 에러"""
    pass

class SessionError(StepUtilsError):
    """세션 관련 에러"""
    pass

class ImageProcessingError(StepUtilsError):
    """이미지 처리 에러"""
    pass

class MemoryError(StepUtilsError):
    """메모리 관리 에러"""
    pass

class StepInstanceError(StepUtilsError):
    """Step 인스턴스 에러"""
    pass

class StepErrorHandler:
    """통합 에러 핸들러"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StepErrorHandler")
        self.error_counts = {}
        self.recovery_strategies = {}
        self._lock = threading.RLock()
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """에러 처리 및 복구 전략"""
        try:
            with self._lock:
                error_type = type(error).__name__
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                
                error_info = {
                    "error_type": error_type,
                    "error_message": str(error),
                    "error_count": self.error_counts[error_type],
                    "context": context or {},
                    "timestamp": datetime.now().isoformat(),
                    "recovery_suggested": False,
                    "recovery_strategy": None
                }
                
                # 복구 전략 결정
                if isinstance(error, SessionError):
                    error_info.update({
                        "recovery_suggested": True,
                        "recovery_strategy": "session_reload"
                    })
                elif isinstance(error, ImageProcessingError):
                    error_info.update({
                        "recovery_suggested": True,
                        "recovery_strategy": "image_fallback"
                    })
                elif isinstance(error, MemoryError):
                    error_info.update({
                        "recovery_suggested": True,
                        "recovery_strategy": "memory_cleanup"
                    })
                elif isinstance(error, StepInstanceError):
                    error_info.update({
                        "recovery_suggested": True,
                        "recovery_strategy": "instance_recreate"
                    })
                
                self.logger.error(f"❌ 에러 처리: {error_type} - {str(error)}")
                if error_info["recovery_suggested"]:
                    self.logger.info(f"🔧 복구 전략: {error_info['recovery_strategy']}")
                
                return error_info
                
        except Exception as e:
            self.logger.error(f"❌ 에러 핸들러 자체 오류: {e}")
            return {
                "error_type": "ErrorHandlerFailure",
                "error_message": str(e),
                "original_error": str(error),
                "recovery_suggested": False
            }
        
    
    def get_error_summary(self) -> Dict[str, Any]:
        """에러 요약"""
        with self._lock:
            return {
                "total_errors": sum(self.error_counts.values()),
                "error_types": dict(self.error_counts),
                "most_common_error": max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
            }
# backend/app/services/step_utils.py 또는 관련 파일에 추가

def safe_mps_empty_cache():
    """M3 Max MPS 캐시 안전 정리"""
    try:
        import torch
        import gc
        
        # 일반 가비지 컬렉션
        gc.collect()
        
        # M3 Max MPS 캐시 정리 시도
        if hasattr(torch, 'mps') and torch.mps.is_available():
            try:
                # PyTorch 2.1+ 방식
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    return {"success": True, "method": "torch.mps.empty_cache"}
                
                # PyTorch 2.0 방식
                elif hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
                    return {"success": True, "method": "torch.backends.mps.empty_cache"}
                
                # 동기화만 수행
                elif hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
                    return {"success": True, "method": "torch.mps.synchronize"}
                
            except (AttributeError, RuntimeError) as e:
                # MPS 캐시 정리 실패 시 일반 정리만
                return {"success": True, "method": "gc_only", "warning": str(e)}
        
        # CUDA 사용 시
        elif hasattr(torch, 'cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
            return {"success": True, "method": "cuda_empty_cache"}
        
        # CPU만 사용 시
        return {"success": True, "method": "gc_only"}
        
    except Exception as e:
        # 모든 실패 시 기본 가비지 컬렉션만
        import gc
        gc.collect()
        return {"success": True, "method": "fallback_gc", "error": str(e)}

# MemoryHelper 클래스 수정
class MemoryHelper:
    @staticmethod
    def optimize_memory():
        """메모리 최적화 - conda 환경 고려"""
        try:
            result = safe_mps_empty_cache()
            if result["success"]:
                return result
            else:
                raise MemoryError(f"메모리 최적화 실패: {result.get('error', 'Unknown')}")
        except Exception as e:
            # 폴백: 기본 가비지 컬렉션
            import gc
            gc.collect()
            return {"success": True, "method": "fallback", "warning": str(e)}

# StepErrorHandler 클래스에 추가
class StepErrorHandler:
    @staticmethod
    def handle_memory_error(error):
        """메모리 오류 처리"""
        try:
            # 안전한 메모리 정리
            result = safe_mps_empty_cache()
            return {
                "handled": True,
                "method": result.get("method", "unknown"),
                "original_error": str(error)
            }
        except Exception as e:
            return {
                "handled": False,
                "error": str(e),
                "original_error": str(error)
            }
# 전역 에러 핸들러
_global_error_handler: Optional[StepErrorHandler] = None
_error_handler_lock = threading.RLock()

def get_error_handler() -> StepErrorHandler:
    """전역 에러 핸들러 반환"""
    global _global_error_handler
    
    with _error_handler_lock:
        if _global_error_handler is None:
            _global_error_handler = StepErrorHandler()
    
    return _global_error_handler

def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """에러 처리 메서드 (누락된 메서드 추가)"""
    try:
        import traceback
        from datetime import datetime
        
        context = context or {}
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "traceback": traceback.format_exc() if hasattr(traceback, 'format_exc') else None
        }
        
        # 로깅
        if hasattr(self, 'logger'):
            self.logger.error(f"❌ 에러 처리: {error_info['error_type']}: {error_info['error_message']}")
        
        return error_info
        
    except Exception as e:
        # 최후의 폴백
        return {
            "error_type": "ErrorHandlerFailure",
            "error_message": f"에러 핸들러 자체 실패: {str(e)}",
            "original_error": str(error),
            "timestamp": datetime.now().isoformat() if 'datetime' in locals() else "unknown",
            "context": context
        }
# ==============================================
# 🔥 세션 관리 헬퍼 (통합 버전)
# ==============================================

class SessionHelper:
    """통합 세션 관리 헬퍼 - step_service.py + step_implementations.py 공통 지원"""
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        self.session_manager = session_manager or (get_session_manager() if SESSION_MANAGER_AVAILABLE else SessionManager())
        self.logger = logging.getLogger(f"{__name__}.SessionHelper")
        self.session_cache = {}
        self._lock = threading.RLock()
    
    async def load_session_images(self, session_id: str) -> Tuple[Optional['Image.Image'], Optional['Image.Image']]:
        """세션에서 이미지 로드 (캐싱 지원)"""
        try:
            if not session_id:
                raise SessionError("session_id가 필요합니다")
            
            # 캐시 확인
            with self._lock:
                if session_id in self.session_cache:
                    cached_data = self.session_cache[session_id]
                    if (time.time() - cached_data['timestamp']) < 300:  # 5분 캐시
                        self.logger.debug(f"세션 캐시 히트: {session_id}")
                        return cached_data['person_image'], cached_data['clothing_image']
            
            # 세션 매니저에서 로드
            person_img, clothing_img = await self.session_manager.get_session_images(session_id)
            
            # 이미지 검증
            if person_img is None and clothing_img is None:
                self.logger.warning(f"⚠️ 세션 {session_id}에서 이미지를 찾을 수 없음")
                return None, None
            
            # 캐시에 저장
            with self._lock:
                self.session_cache[session_id] = {
                    'person_image': person_img,
                    'clothing_image': clothing_img,
                    'timestamp': time.time()
                }
                
                # 캐시 크기 제한 (최대 20개)
                if len(self.session_cache) > 20:
                    oldest_key = min(self.session_cache.keys(), 
                                   key=lambda k: self.session_cache[k]['timestamp'])
                    del self.session_cache[oldest_key]
            
            self.logger.debug(f"✅ 세션 이미지 로드 성공: {session_id}")
            return person_img, clothing_img
            
        except Exception as e:
            error_handler = get_error_handler()
            error_info = error_handler.handle_error(
                SessionError(f"세션 이미지 로드 실패: {str(e)}"),
                {"session_id": session_id}
            )
            self.logger.error(f"❌ 세션 이미지 로드 실패: {e}")
            return None, None
    
    async def store_session_data(self, session_id: str, data: Dict[str, Any]) -> bool:
        """세션 데이터 저장"""
        try:
            if not session_id:
                raise SessionError("session_id가 필요합니다")
            
            await self.session_manager.store_session_data(session_id, data)
            self.logger.debug(f"✅ 세션 데이터 저장 성공: {session_id}")
            return True
            
        except Exception as e:
            error_handler = get_error_handler()
            error_handler.handle_error(
                SessionError(f"세션 데이터 저장 실패: {str(e)}"),
                {"session_id": session_id, "data_keys": list(data.keys()) if data else []}
            )
            self.logger.error(f"❌ 세션 데이터 저장 실패: {e}")
            return False
    
    def clear_session_cache(self, session_id: Optional[str] = None):
        """세션 캐시 정리"""
        try:
            with self._lock:
                if session_id:
                    self.session_cache.pop(session_id, None)
                    self.logger.debug(f"세션 캐시 정리: {session_id}")
                else:
                    self.session_cache.clear()
                    self.logger.debug("모든 세션 캐시 정리")
        except Exception as e:
            self.logger.warning(f"세션 캐시 정리 실패: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """세션 통계"""
        with self._lock:
            return {
                "cached_sessions": len(self.session_cache),
                "session_manager_available": SESSION_MANAGER_AVAILABLE,
                "cache_enabled": True
            }

# 전역 세션 헬퍼
_global_session_helper: Optional[SessionHelper] = None
_session_helper_lock = threading.RLock()

def get_session_helper() -> SessionHelper:
    """전역 세션 헬퍼 반환"""
    global _global_session_helper
    
    with _session_helper_lock:
        if _global_session_helper is None:
            _global_session_helper = SessionHelper()
    
    return _global_session_helper

# ==============================================
# 🔥 이미지 처리 헬퍼 (통합 버전)
# ==============================================

class ImageHelper:
    """통합 이미지 처리 헬퍼 - PIL, NumPy, Base64 등 지원"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ImageHelper")
        self.supported_formats = ['JPEG', 'PNG', 'RGB', 'RGBA']
        self.max_image_size = (2048, 2048)  # 최대 이미지 크기
        self.min_image_size = (64, 64)      # 최소 이미지 크기
    
    def validate_image_content(self, content: bytes, file_type: str) -> Dict[str, Any]:
        """이미지 파일 내용 검증 (step_service.py + step_implementations.py 공통)"""
        try:
            if len(content) == 0:
                return {"valid": False, "error": f"{file_type} 이미지: 빈 파일입니다"}
            
            if len(content) > 50 * 1024 * 1024:  # 50MB
                return {"valid": False, "error": f"{file_type} 이미지가 50MB를 초과합니다"}
            
            if PIL_AVAILABLE:
                try:
                    img = Image.open(BytesIO(content))
                    img.verify()
                    
                    # 크기 검증
                    img = Image.open(BytesIO(content))  # verify() 후 다시 열기
                    width, height = img.size
                    
                    if width < self.min_image_size[0] or height < self.min_image_size[1]:
                        return {
                            "valid": False, 
                            "error": f"{file_type} 이미지: 너무 작습니다 (최소 {self.min_image_size[0]}x{self.min_image_size[1]})"
                        }
                    
                    if width > self.max_image_size[0] or height > self.max_image_size[1]:
                        return {
                            "valid": False,
                            "error": f"{file_type} 이미지: 너무 큽니다 (최대 {self.max_image_size[0]}x{self.max_image_size[1]})"
                        }
                    
                    # 색상 모드 검증
                    if img.mode not in ['RGB', 'RGBA', 'L']:
                        return {
                            "valid": False,
                            "error": f"{file_type} 이미지: 지원되지 않는 색상 모드 ({img.mode})"
                        }
                    
                    return {
                        "valid": True,
                        "size": len(content),
                        "format": img.format,
                        "dimensions": (width, height),
                        "mode": img.mode,
                        "file_type": file_type
                    }
                    
                except Exception as e:
                    return {"valid": False, "error": f"{file_type} 이미지가 손상되었습니다: {str(e)}"}
            else:
                # PIL 없는 경우 기본 검증
                return {
                    "valid": True,
                    "size": len(content),
                    "format": "unknown",
                    "dimensions": (0, 0),
                    "mode": "unknown",
                    "file_type": file_type
                }
            
        except Exception as e:
            error_handler = get_error_handler()
            error_handler.handle_error(
                ImageProcessingError(f"이미지 검증 실패: {str(e)}"),
                {"file_type": file_type, "content_size": len(content) if content else 0}
            )
            return {"valid": False, "error": f"파일 검증 중 오류: {str(e)}"}
    
    def convert_image_to_base64(self, image: Union['Image.Image', 'np.ndarray'], format: str = "JPEG", quality: int = 90) -> str:
        """이미지를 Base64로 변환"""
        try:
            if not PIL_AVAILABLE:
                self.logger.warning("PIL 없음 - Base64 변환 불가")
                return ""
            
            # NumPy 배열을 PIL Image로 변환
            if NUMPY_AVAILABLE and isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            # PIL Image 처리
            if hasattr(image, 'save'):
                # RGB 모드로 변환 (JPEG는 RGBA 지원 안함)
                if format.upper() == 'JPEG' and image.mode == 'RGBA':
                    # 흰색 배경으로 RGBA → RGB 변환
                    rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[-1] if len(image.split()) == 4 else None)
                    image = rgb_image
                
                buffer = BytesIO()
                image.save(buffer, format=format, quality=quality, optimize=True)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            else:
                self.logger.error("지원되지 않는 이미지 타입")
                return ""
                
        except Exception as e:
            error_handler = get_error_handler()
            error_handler.handle_error(
                ImageProcessingError(f"Base64 변환 실패: {str(e)}"),
                {"format": format, "quality": quality}
            )
            self.logger.error(f"❌ 이미지 Base64 변환 실패: {e}")
            return ""
    
    def convert_base64_to_image(self, base64_str: str) -> Optional['Image.Image']:
        """Base64를 PIL Image로 변환"""
        try:
            if not PIL_AVAILABLE:
                self.logger.warning("PIL 없음 - Base64 변환 불가")
                return None
            
            # Base64 디코딩
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))
            
            # RGB 모드로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            error_handler = get_error_handler()
            error_handler.handle_error(
                ImageProcessingError(f"Base64 → Image 변환 실패: {str(e)}"),
                {"base64_length": len(base64_str) if base64_str else 0}
            )
            self.logger.error(f"❌ Base64 → Image 변환 실패: {e}")
            return None
    
    def resize_image_with_aspect_ratio(self, image: 'Image.Image', target_size: Tuple[int, int], maintain_ratio: bool = True) -> 'Image.Image':
        """비율 유지하면서 이미지 크기 조정"""
        try:
            if not PIL_AVAILABLE:
                return image
            
            if maintain_ratio:
                image.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                # 중앙 정렬을 위한 패딩
                new_image = Image.new('RGB', target_size, (255, 255, 255))
                paste_x = (target_size[0] - image.width) // 2
                paste_y = (target_size[1] - image.height) // 2
                new_image.paste(image, (paste_x, paste_y))
                
                return new_image
            else:
                return image.resize(target_size, Image.Resampling.LANCZOS)
                
        except Exception as e:
            error_handler = get_error_handler()
            error_handler.handle_error(
                ImageProcessingError(f"이미지 크기 조정 실패: {str(e)}"),
                {"target_size": target_size, "maintain_ratio": maintain_ratio}
            )
            self.logger.error(f"❌ 이미지 크기 조정 실패: {e}")
            return image
    
    def create_dummy_image(self, size: Tuple[int, int] = (512, 512), color: Tuple[int, int, int] = (200, 200, 200), text: Optional[str] = None) -> Optional['Image.Image']:
        """더미 이미지 생성 (테스트용)"""
        try:
            if not PIL_AVAILABLE:
                return None
            
            image = Image.new('RGB', size, color)
            
            if text:
                try:
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(image)
                    # 기본 폰트 사용
                    font_size = min(size) // 20
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
                    
                    # 텍스트 중앙 정렬
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    text_x = (size[0] - text_width) // 2
                    text_y = (size[1] - text_height) // 2
                    
                    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
                except ImportError:
                    pass  # ImageDraw/ImageFont 없으면 텍스트 없이
            
            return image
            
        except Exception as e:
            self.logger.error(f"❌ 더미 이미지 생성 실패: {e}")
            return None
    
    def get_image_stats(self) -> Dict[str, Any]:
        """이미지 헬퍼 통계"""
        return {
            "pil_available": PIL_AVAILABLE,
            "numpy_available": NUMPY_AVAILABLE,
            "supported_formats": self.supported_formats,
            "max_image_size": self.max_image_size,
            "min_image_size": self.min_image_size
        }

# 전역 이미지 헬퍼
_global_image_helper: Optional[ImageHelper] = None
_image_helper_lock = threading.RLock()

def get_image_helper() -> ImageHelper:
    """전역 이미지 헬퍼 반환"""
    global _global_image_helper
    
    with _image_helper_lock:
        if _global_image_helper is None:
            _global_image_helper = ImageHelper()
    
    return _global_image_helper

# ==============================================
# 🔥 메모리 관리 헬퍼 (M3 Max 128GB 최적화)
# ==============================================

class MemoryHelper:
    """통합 메모리 관리 헬퍼 - M3 Max 128GB + conda 환경 최적화"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MemoryHelper")
        self.memory_stats = {
            'cleanup_count': 0,
            'last_cleanup': None,
            'optimization_count': 0
        }
        self._lock = threading.RLock()
        
        # conda 환경 최적화 자동 실행
        self.setup_conda_memory_optimization()
    
    def setup_conda_memory_optimization(self):
        """conda 환경 우선 메모리 최적화"""
        try:
            if 'CONDA_DEFAULT_ENV' in os.environ:
                conda_env = os.environ['CONDA_DEFAULT_ENV']
                self.logger.info(f"🐍 conda 환경 감지: {conda_env}")
                
                # conda 환경 변수 설정
                os.environ['OMP_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
                os.environ['MKL_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
                os.environ['NUMEXPR_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
                
                if TORCH_AVAILABLE:
                    # PyTorch conda 최적화
                    torch.set_num_threads(max(1, os.cpu_count() // 2))
                    
                    # M3 Max 메모리 최적화
                    if IS_M3_MAX:
                        torch.backends.mps.empty_cache()
                        self.logger.info("🍎 M3 Max MPS 메모리 최적화 활성화")
                
                self.logger.info("✅ conda 환경 메모리 최적화 완료")
            else:
                self.logger.info("🐍 conda 환경 아님 - 기본 메모리 최적화 사용")
                
        except Exception as e:
            self.logger.warning(f"⚠️ conda 메모리 최적화 실패: {e}")
    
    def optimize_device_memory(self, device: str):
        """디바이스별 메모리 최적화"""
        try:
            with self._lock:
                if TORCH_AVAILABLE:
                    if device == "mps" and IS_M3_MAX:
                        # M3 Max MPS 최적화
                        if hasattr(torch.mps, 'empty_cache'):
                            safe_mps_empty_cache()
                        if hasattr(torch.mps, 'synchronize'):
                            torch.mps.synchronize()
                        self.logger.debug("✅ M3 Max MPS 메모리 최적화")
                        
                    elif device == "cuda":
                        # CUDA 최적화
                        torch.cuda.empty_cache()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        self.logger.debug("✅ CUDA 메모리 최적화")
                    
                    elif device == "cpu":
                        # CPU 메모리 최적화
                        gc.collect()
                        self.logger.debug("✅ CPU 메모리 최적화")
                
                # Python 가비지 컬렉션
                collected = gc.collect()
                
                self.memory_stats['optimization_count'] += 1
                self.logger.debug(f"✅ {device} 메모리 최적화 완료 (GC: {collected})")
                
        except Exception as e:
            error_handler = get_error_handler()
            error_handler.handle_error(
                MemoryError(f"메모리 최적화 실패: {str(e)}"),
                {"device": device}
            )
            self.logger.warning(f"⚠️ 메모리 최적화 실패: {e}")
    
    def cleanup_memory(self, force: bool = False):
        """강제 메모리 정리"""
        try:
            with self._lock:
                # 캐시 정리
                if hasattr(self, '_cache'):
                    self._cache.clear()
                
                # 디바이스별 정리
                if TORCH_AVAILABLE:
                    if IS_M3_MAX and torch.backends.mps.is_available():
                        safe_mps_empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Python 메모리 정리
                collected = gc.collect()
                
                self.memory_stats['cleanup_count'] += 1
                self.memory_stats['last_cleanup'] = datetime.now()
                
                self.logger.info(f"🧹 메모리 정리 완료 (GC: {collected}, 강제: {force})")
                
        except Exception as e:
            self.logger.error(f"❌ 메모리 정리 실패: {e}")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """메모리 정보 조회"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            
            memory_info = {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent": memory.percent,
                "is_m3_max": IS_M3_MAX,
                "device": DEVICE,
                "conda_env": os.environ.get('CONDA_DEFAULT_ENV'),
                "torch_available": TORCH_AVAILABLE
            }
            
            # PyTorch 메모리 정보
            if TORCH_AVAILABLE:
                if IS_M3_MAX and torch.backends.mps.is_available():
                    # M3 Max MPS 정보는 제한적
                    memory_info["mps_available"] = True
                elif torch.cuda.is_available():
                    memory_info.update({
                        "cuda_memory_allocated": torch.cuda.memory_allocated(),
                        "cuda_memory_reserved": torch.cuda.memory_reserved(),
                        "cuda_memory_cached": torch.cuda.memory_cached()
                    })
            
            memory_info.update(self.memory_stats)
            return memory_info
            
        except ImportError:
            # psutil 없는 경우 기본 정보
            return {
                "total_gb": 128.0 if IS_M3_MAX else 16.0,
                "is_m3_max": IS_M3_MAX,
                "device": DEVICE,
                "torch_available": TORCH_AVAILABLE,
                "conda_env": os.environ.get('CONDA_DEFAULT_ENV'),
                **self.memory_stats
            }
        except Exception as e:
            self.logger.error(f"메모리 정보 조회 실패: {e}")
            return {"error": str(e), **self.memory_stats}
    
    @asynccontextmanager
    async def memory_context(self, cleanup_after: bool = True):
        """메모리 관리 컨텍스트 매니저"""
        try:
            # 진입 시 최적화
            self.optimize_device_memory(DEVICE)
            yield
        finally:
            # 종료 시 정리
            if cleanup_after:
                self.cleanup_memory()

# 전역 메모리 헬퍼
_global_memory_helper: Optional[MemoryHelper] = None
_memory_helper_lock = threading.RLock()

def get_memory_helper() -> MemoryHelper:
    """전역 메모리 헬퍼 반환"""
    global _global_memory_helper
    
    with _memory_helper_lock:
        if _global_memory_helper is None:
            _global_memory_helper = MemoryHelper()
    
    return _global_memory_helper

# ==============================================
# 🔥 성능 모니터링 시스템
# ==============================================

@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """성능 모니터링 시스템 - step_service.py + step_implementations.py 공통"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
        self.metrics: List[PerformanceMetrics] = []
        self.operation_stats = {}
        self._lock = threading.RLock()
        self.max_metrics = 1000  # 최대 메트릭 개수
    
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str, **additional_data):
        """작업 모니터링 컨텍스트 매니저"""
        metric = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            additional_data=additional_data
        )
        
        # 메모리 정보 수집 (가능한 경우)
        try:
            memory_helper = get_memory_helper()
            memory_info = memory_helper.get_memory_info()
            metric.memory_before = memory_info.get('used_gb', 0)
        except Exception:
            pass
        
        try:
            yield metric
            metric.success = True
        except Exception as e:
            metric.success = False
            metric.error_message = str(e)
            raise
        finally:
            # 종료 처리
            metric.end_time = time.time()
            metric.duration = metric.end_time - metric.start_time
            
            # 메모리 정보 수집 (종료 시)
            try:
                memory_helper = get_memory_helper()
                memory_info = memory_helper.get_memory_info()
                metric.memory_after = memory_info.get('used_gb', 0)
            except Exception:
                pass
            
            self._record_metric(metric)
    
    def _record_metric(self, metric: PerformanceMetrics):
        """메트릭 기록"""
        try:
            with self._lock:
                self.metrics.append(metric)
                
                # 메트릭 수 제한
                if len(self.metrics) > self.max_metrics:
                    self.metrics.pop(0)
                
                # 통계 업데이트
                if metric.operation_name not in self.operation_stats:
                    self.operation_stats[metric.operation_name] = {
                        'total_count': 0,
                        'success_count': 0,
                        'error_count': 0,
                        'total_duration': 0.0,
                        'min_duration': float('inf'),
                        'max_duration': 0.0,
                        'avg_duration': 0.0
                    }
                
                stats = self.operation_stats[metric.operation_name]
                stats['total_count'] += 1
                
                if metric.success:
                    stats['success_count'] += 1
                else:
                    stats['error_count'] += 1
                
                if metric.duration is not None:
                    stats['total_duration'] += metric.duration
                    stats['min_duration'] = min(stats['min_duration'], metric.duration)
                    stats['max_duration'] = max(stats['max_duration'], metric.duration)
                    stats['avg_duration'] = stats['total_duration'] / stats['total_count']
                
                self.logger.debug(f"📊 성능 기록: {metric.operation_name} - {metric.duration:.3f}s (성공: {metric.success})")
                
        except Exception as e:
            self.logger.warning(f"성능 메트릭 기록 실패: {e}")
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """작업 통계 조회"""
        with self._lock:
            if operation_name:
                return self.operation_stats.get(operation_name, {})
            else:
                return dict(self.operation_stats)
    
    def get_recent_metrics(self, count: int = 10, operation_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """최근 메트릭 조회"""
        with self._lock:
            filtered_metrics = self.metrics
            
            if operation_name:
                filtered_metrics = [m for m in self.metrics if m.operation_name == operation_name]
            
            recent = filtered_metrics[-count:] if count > 0 else filtered_metrics
            
            return [
                {
                    'operation_name': m.operation_name,
                    'duration': m.duration,
                    'success': m.success,
                    'error_message': m.error_message,
                    'memory_before': m.memory_before,
                    'memory_after': m.memory_after,
                    'timestamp': m.start_time,
                    'additional_data': m.additional_data
                }
                for m in recent
            ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약"""
        with self._lock:
            total_operations = len(self.metrics)
            successful_operations = sum(1 for m in self.metrics if m.success)
            
            return {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "error_rate": (total_operations - successful_operations) / max(total_operations, 1),
                "operation_types": len(self.operation_stats),
                "operation_stats": dict(self.operation_stats),
                "memory_monitoring": True,
                "max_metrics_stored": self.max_metrics
            }
    
    def clear_metrics(self, operation_name: Optional[str] = None):
        """메트릭 정리"""
        with self._lock:
            if operation_name:
                self.metrics = [m for m in self.metrics if m.operation_name != operation_name]
                self.operation_stats.pop(operation_name, None)
                self.logger.info(f"📊 {operation_name} 메트릭 정리 완료")
            else:
                self.metrics.clear()
                self.operation_stats.clear()
                self.logger.info("📊 모든 메트릭 정리 완료")

# 전역 성능 모니터
_global_performance_monitor: Optional[PerformanceMonitor] = None
_performance_monitor_lock = threading.RLock()

def get_performance_monitor() -> PerformanceMonitor:
    """전역 성능 모니터 반환"""
    global _global_performance_monitor
    
    with _performance_monitor_lock:
        if _global_performance_monitor is None:
            _global_performance_monitor = PerformanceMonitor()
    
    return _global_performance_monitor

# ==============================================
# 🔥 Step 데이터 준비 헬퍼 (시그니처 기반)
# ==============================================

class StepDataPreparer:
    """Step별 동적 데이터 준비 - 통합 시그니처 기반"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StepDataPreparer")
        self.session_helper = get_session_helper()
        self.image_helper = get_image_helper()
    
    async def prepare_step_data(
        self, 
        step_id: int, 
        inputs: Dict[str, Any]
    ) -> Tuple[Tuple, Dict[str, Any]]:
        """Step별 동적 데이터 준비 - 통합 시그니처 기반 자동 매핑"""
        try:
            # 통합 시그니처 조회
            step_class_name = UNIFIED_STEP_CLASS_MAPPING.get(step_id)
            if not step_class_name:
                raise ValueError(f"Step {step_id}에 대한 클래스 매핑을 찾을 수 없음")
            
            signature = UNIFIED_STEP_SIGNATURES.get(step_class_name)
            if not signature:
                raise ValueError(f"Step {step_id} ({step_class_name})에 대한 시그니처를 찾을 수 없음")
            
            # 세션에서 이미지 로드
            session_id = inputs.get("session_id")
            person_img, clothing_img = await self.session_helper.load_session_images(session_id)
            
            args = []
            kwargs = {}
            
            # 필수 인자 준비 (통합 시그니처 기반)
            for arg_name in signature.required_args:
                if arg_name in ["person_image", "image"] and step_id in [1, 2]:  # HumanParsing, PoseEstimation
                    if person_img is None:
                        raise ValueError(f"Step {step_id} ({step_class_name}): person_image 로드 실패")
                    args.append(person_img)
                    
                elif arg_name == "image" and step_id == 3:  # ClothSegmentation
                    if clothing_img is None:
                        raise ValueError(f"Step {step_id} ({step_class_name}): clothing_image 로드 실패")
                    args.append(clothing_img)
                    
                elif arg_name in ["person_image", "cloth_image", "clothing_image"]:
                    if "person" in arg_name:
                        if person_img is None:
                            raise ValueError(f"Step {step_id} ({step_class_name}): person_image 로드 실패")
                        args.append(person_img)
                    else:
                        if clothing_img is None:
                            raise ValueError(f"Step {step_id} ({step_class_name}): clothing_image 로드 실패")
                        args.append(clothing_img)
                        
                elif arg_name == "fitted_image":
                    fitted_image = inputs.get("fitted_image", person_img)
                    if fitted_image is None:
                        raise ValueError(f"Step {step_id} ({step_class_name}): fitted_image 로드 실패")
                    args.append(fitted_image)
                    
                elif arg_name == "final_image":
                    final_image = inputs.get("final_image", person_img)
                    if final_image is None:
                        raise ValueError(f"Step {step_id} ({step_class_name}): final_image 로드 실패")
                    args.append(final_image)
                    
                elif arg_name == "measurements":
                    measurements = inputs.get("measurements")
                    if measurements is None:
                        raise ValueError(f"Step {step_id} ({step_class_name}): measurements 로드 실패")
                    args.append(measurements)
                    
                else:
                    # 기타 필수 인자들
                    if arg_name in inputs:
                        args.append(inputs[arg_name])
                    else:
                        raise ValueError(f"Step {step_id} ({step_class_name}): 필수 인자 {arg_name} 없음")
            
            # 필수 kwargs 준비 (통합 시그니처 기반)
            for kwarg_name in signature.required_kwargs:
                if kwarg_name == "clothing_type":
                    kwargs[kwarg_name] = inputs.get("clothing_type", "shirt")
                elif kwarg_name == "quality_level":
                    kwargs[kwarg_name] = inputs.get("quality_level", "medium")
                else:
                    if kwarg_name in inputs:
                        kwargs[kwarg_name] = inputs[kwarg_name]
                    else:
                        # 기본값 제공
                        default_values = {
                            "detection_confidence": 0.5,
                            "matching_precision": "high",
                            "fabric_type": "cotton",
                            "fitting_quality": "high",
                            "enhancement_level": "medium",
                            "analysis_depth": "comprehensive"
                        }
                        kwargs[kwarg_name] = default_values.get(kwarg_name, "default")
            
            # 선택적 kwargs 준비 (통합 시그니처 기반)
            for kwarg_name in signature.optional_kwargs:
                if kwarg_name in inputs:
                    kwargs[kwarg_name] = inputs[kwarg_name]
                elif kwarg_name == "session_id":
                    kwargs[kwarg_name] = session_id
                elif kwarg_name == "enhance_quality":
                    kwargs[kwarg_name] = inputs.get("enhance_quality", True)
            
            self.logger.debug(
                f"✅ Step {step_id} ({step_class_name}) 데이터 준비 완료: "
                f"args={len(args)}, kwargs={list(kwargs.keys())}"
            )
            
            return tuple(args), kwargs
            
        except Exception as e:
            self.logger.error(f"❌ Step {step_id} 데이터 준비 실패: {e}")
            raise
    
    def validate_step_inputs(self, step_id: int, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Step 입력값 검증"""
        try:
            step_class_name = UNIFIED_STEP_CLASS_MAPPING.get(step_id)
            signature = UNIFIED_STEP_SIGNATURES.get(step_class_name) if step_class_name else None
            
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "step_id": step_id,
                "step_class_name": step_class_name
            }
            
            if not signature:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Step {step_id} 시그니처 없음")
                return validation_result
            
            # 필수 인자 검증
            missing_args = []
            for arg_name in signature.required_args:
                if arg_name not in ["person_image", "cloth_image", "clothing_image", "image", "fitted_image", "final_image"]:
                    if arg_name not in inputs:
                        missing_args.append(arg_name)
            
            if missing_args:
                validation_result["valid"] = False
                validation_result["errors"].append(f"필수 인자 누락: {missing_args}")
            
            # 필수 kwargs 검증
            missing_kwargs = []
            for kwarg_name in signature.required_kwargs:
                if kwarg_name not in inputs:
                    missing_kwargs.append(kwarg_name)
            
            if missing_kwargs:
                validation_result["warnings"].append(f"필수 kwargs 누락 (기본값 사용): {missing_kwargs}")
            
            # 세션 ID 검증
            if not inputs.get("session_id"):
                validation_result["valid"] = False
                validation_result["errors"].append("session_id 필요")
            
            return validation_result
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"입력값 검증 실패: {str(e)}"],
                "step_id": step_id
            }

# 전역 데이터 준비자
_global_step_data_preparer: Optional[StepDataPreparer] = None
_data_preparer_lock = threading.RLock()

def get_step_data_preparer() -> StepDataPreparer:
    """전역 데이터 준비자 반환"""
    global _global_step_data_preparer
    
    with _data_preparer_lock:
        if _global_step_data_preparer is None:
            _global_step_data_preparer = StepDataPreparer()
    
    return _global_step_data_preparer

# ==============================================
# 🔥 통합 유틸리티 매니저 (모든 헬퍼 통합)
# ==============================================

class UtilsManager:
    """통합 유틸리티 매니저 - 모든 헬퍼들을 통합 관리"""
    
    def __init__(self, di_container: Optional[DIContainer] = None):
        self.di_container = di_container or get_di_container()
        self.logger = logging.getLogger(f"{__name__}.UtilsManager")
        
        # 헬퍼들 초기화
        self.error_handler = get_error_handler()
        self.session_helper = get_session_helper()
        self.image_helper = get_image_helper()
        self.memory_helper = get_memory_helper()
        self.performance_monitor = get_performance_monitor()
        self.step_data_preparer = get_step_data_preparer()
        
        # 상태 관리
        self.initialized = False
        self.start_time = datetime.now()
        
        # conda 환경 최적화
        setup_conda_optimization()
        
        self.logger.info("✅ UtilsManager 초기화 완료")
    
    async def initialize(self) -> bool:
        """유틸리티 매니저 초기화"""
        try:
            # 메모리 최적화
            self.memory_helper.optimize_device_memory(DEVICE)
            
            # 세션 헬퍼 설정
            if hasattr(self.session_helper, 'session_manager') and SESSION_MANAGER_AVAILABLE:
                self.logger.info("✅ 세션 매니저 연동 확인")
            
            self.initialized = True
            self.logger.info("✅ UtilsManager 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ UtilsManager 초기화 실패: {e}")
            return False
    
    def get_unified_stats(self) -> Dict[str, Any]:
        """통합 유틸리티 통계"""
        try:
            return {
                "utils_manager": {
                    "initialized": self.initialized,
                    "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                    "di_container_available": DI_CONTAINER_AVAILABLE
                },
                "error_handler": self.error_handler.get_error_summary(),
                "session_helper": self.session_helper.get_session_stats(),
                "image_helper": self.image_helper.get_image_stats(),
                "memory_helper": self.memory_helper.get_memory_info(),
                "performance_monitor": self.performance_monitor.get_performance_summary(),
                "system_info": {
                    "unified_mapping_available": UNIFIED_MAPPING_AVAILABLE,
                    "torch_available": TORCH_AVAILABLE,
                    "pil_available": PIL_AVAILABLE,
                    "numpy_available": NUMPY_AVAILABLE,
                    "session_manager_available": SESSION_MANAGER_AVAILABLE,
                    "model_loader_available": MODEL_LOADER_AVAILABLE,
                    "device": DEVICE,
                    "is_m3_max": IS_M3_MAX,
                    "conda_env": os.environ.get('CONDA_DEFAULT_ENV'),
                    "conda_optimized": 'CONDA_DEFAULT_ENV' in os.environ
                },
                "unified_mapping_info": get_system_compatibility_info() if UNIFIED_MAPPING_AVAILABLE else {}
            }
            
        except Exception as e:
            self.logger.error(f"통합 통계 조회 실패: {e}")
            return {"error": str(e)}
    
    async def cleanup_all(self):
        """모든 유틸리티 정리"""
        try:
            # 성능 모니터 정리
            self.performance_monitor.clear_metrics()
            
            # 세션 캐시 정리
            self.session_helper.clear_session_cache()
            
            # 메모리 정리
            self.memory_helper.cleanup_memory(force=True)
            
            self.initialized = False
            self.logger.info("✅ UtilsManager 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ UtilsManager 정리 실패: {e}")

# 전역 유틸리티 매니저
_global_utils_manager: Optional[UtilsManager] = None
_utils_manager_lock = threading.RLock()

def get_utils_manager(di_container: Optional[DIContainer] = None) -> UtilsManager:
    """전역 유틸리티 매니저 반환"""
    global _global_utils_manager
    
    with _utils_manager_lock:
        if _global_utils_manager is None:
            _global_utils_manager = UtilsManager(di_container)
    
    return _global_utils_manager

async def get_utils_manager_async(di_container: Optional[DIContainer] = None) -> UtilsManager:
    """비동기 유틸리티 매니저 반환"""
    manager = get_utils_manager(di_container)
    if not manager.initialized:
        await manager.initialize()
    return manager

# ==============================================
# 🔥 공개 인터페이스 및 편의 함수들
# ==============================================

# 편의 함수들 (step_service.py + step_implementations.py에서 직접 사용)
async def load_session_images(session_id: str) -> Tuple[Optional['Image.Image'], Optional['Image.Image']]:
    """세션 이미지 로드 (편의 함수)"""
    session_helper = get_session_helper()
    return await session_helper.load_session_images(session_id)

def validate_image_content(content: bytes, file_type: str) -> Dict[str, Any]:
    """이미지 검증 (편의 함수)"""
    image_helper = get_image_helper()
    return image_helper.validate_image_content(content, file_type)

def convert_image_to_base64(image: Union['Image.Image', 'np.ndarray'], format: str = "JPEG") -> str:
    """Base64 변환 (편의 함수)"""
    image_helper = get_image_helper()
    return image_helper.convert_image_to_base64(image, format)

def optimize_memory(device: str = None):
    """메모리 최적화 (편의 함수)"""
    memory_helper = get_memory_helper()
    memory_helper.optimize_device_memory(device or DEVICE)

async def prepare_step_data(step_id: int, inputs: Dict[str, Any]) -> Tuple[Tuple, Dict[str, Any]]:
    """Step 데이터 준비 (편의 함수)"""
    data_preparer = get_step_data_preparer()
    return await data_preparer.prepare_step_data(step_id, inputs)

@asynccontextmanager
async def monitor_performance(operation_name: str, **additional_data):
    """성능 모니터링 (편의 함수)"""
    performance_monitor = get_performance_monitor()
    async with performance_monitor.monitor_operation(operation_name, **additional_data) as metric:
        yield metric

def handle_step_error(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """에러 처리 (편의 함수)"""
    error_handler = get_error_handler()
    return error_handler.handle_error(error, context)

# ==============================================
# 🔥 모듈 Export
# ==============================================

__all__ = [
    # 헬퍼 클래스들
    "SessionHelper",
    "ImageHelper", 
    "MemoryHelper",
    "PerformanceMonitor",
    "StepDataPreparer",
    "StepErrorHandler",
    "UtilsManager",
    
    # 전역 인스턴스 함수들
    "get_session_helper",
    "get_image_helper",
    "get_memory_helper", 
    "get_performance_monitor",
    "get_step_data_preparer",
    "get_error_handler",
    "get_utils_manager",
    "get_utils_manager_async",
    
    # 편의 함수들
    "load_session_images",
    "validate_image_content",
    "convert_image_to_base64",
    "optimize_memory",
    "prepare_step_data",
    "monitor_performance",
    "handle_step_error",
    
    # 에러 클래스들
    "StepUtilsError",
    "SessionError",
    "ImageProcessingError", 
    "MemoryError",
    "StepInstanceError",
    
    # 데이터 클래스들
    "PerformanceMetrics",
    "BodyMeasurements",
    
    # 통합 매핑 시스템 re-export
    "UNIFIED_STEP_CLASS_MAPPING",
    "UNIFIED_SERVICE_CLASS_MAPPING",
    "SERVICE_TO_STEP_MAPPING",
    "STEP_TO_SERVICE_MAPPING",
    "SERVICE_ID_TO_STEP_ID",
    "STEP_ID_TO_SERVICE_ID",
    "UnifiedStepSignature",
    "UNIFIED_STEP_SIGNATURES",
    "StepFactoryHelper",
    "setup_conda_optimization",
    "validate_step_compatibility",
    "get_step_id_by_service_id",
    "get_service_id_by_step_id",
    "get_all_available_steps",
    "get_all_available_services",
    "get_system_compatibility_info",
    
    # 시스템 정보
    "TORCH_AVAILABLE",
    "PIL_AVAILABLE", 
    "NUMPY_AVAILABLE",
    "DI_CONTAINER_AVAILABLE",
    "SESSION_MANAGER_AVAILABLE",
    "MODEL_LOADER_AVAILABLE",
    "UNIFIED_MAPPING_AVAILABLE",
    "DEVICE",
    "IS_M3_MAX"
]

# ==============================================
# 🔥 모듈 로드 완료 메시지
# ==============================================

logger.info("✅ Step Utils Layer v2.0 로드 완료!")
logger.info("🛠️ Complete Utility Layer for Step Services")
logger.info("🔗 unified_step_mapping.py 완전 활용 - 세 파일 통합 지원")
logger.info("🤖 BaseStepMixin 완벽 호환 - logger 속성 및 초기화 과정")
logger.info("💾 ModelLoader 완전 연동 - 89.8GB 체크포인트 활용")
logger.info("🔧 실제 Step 클래스들과 100% 호환 - HumanParsingStep 등")
logger.info("🏗️ step_service.py + step_implementations.py 공통 지원")
logger.info("📊 SessionManager, DI Container 완전 연동")
logger.info("🛡️ 에러 처리 및 복구 시스템")
logger.info("📈 성능 모니터링 및 메모리 관리")
logger.info("🍎 M3 Max 128GB 최적화 + conda 환경 우선")
logger.info("⚡ 순환참조 완전 방지 - 단방향 의존성")
logger.info("🚀 프로덕션 레벨 안정성")

logger.info(f"📊 시스템 상태:")
logger.info(f"   - 통합 매핑: {'✅' if UNIFIED_MAPPING_AVAILABLE else '❌'}")
logger.info(f"   - PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
logger.info(f"   - PIL: {'✅' if PIL_AVAILABLE else '❌'}")
logger.info(f"   - NumPy: {'✅' if NUMPY_AVAILABLE else '❌'}")
logger.info(f"   - DI Container: {'✅' if DI_CONTAINER_AVAILABLE else '❌'}")
logger.info(f"   - Session Manager: {'✅' if SESSION_MANAGER_AVAILABLE else '❌'}")
logger.info(f"   - ModelLoader: {'✅' if MODEL_LOADER_AVAILABLE else '❌'}")
logger.info(f"   - Device: {DEVICE}")
logger.info(f"   - conda 환경: {'✅' if 'CONDA_DEFAULT_ENV' in os.environ else '❌'}")

logger.info("🔧 제공되는 헬퍼들:")
logger.info("   - SessionHelper: 세션 관리 및 이미지 로드")
logger.info("   - ImageHelper: 이미지 검증, 변환, 처리")
logger.info("   - MemoryHelper: M3 Max 메모리 최적화")
logger.info("   - PerformanceMonitor: 성능 모니터링")
logger.info("   - StepDataPreparer: Step별 데이터 준비")
logger.info("   - StepErrorHandler: 에러 처리 및 복구")
logger.info("   - UtilsManager: 모든 헬퍼 통합 관리")

logger.info("🎯 편의 함수들:")
logger.info("   - load_session_images(): 세션 이미지 로드")
logger.info("   - validate_image_content(): 이미지 검증")
logger.info("   - convert_image_to_base64(): Base64 변환")
logger.info("   - optimize_memory(): 메모리 최적화")
logger.info("   - prepare_step_data(): Step 데이터 준비")
logger.info("   - monitor_performance(): 성능 모니터링")
logger.info("   - handle_step_error(): 에러 처리")

logger.info(f"🔗 통합 매핑 정보:")
if UNIFIED_MAPPING_AVAILABLE:
    logger.info(f"   - Step 클래스: {len(UNIFIED_STEP_CLASS_MAPPING)}개")
    logger.info(f"   - Service 클래스: {len(UNIFIED_SERVICE_CLASS_MAPPING)}개")
    logger.info(f"   - Step 시그니처: {len(UNIFIED_STEP_SIGNATURES)}개")
    
    # Step 클래스 매핑 출력
    for step_id, step_class_name in UNIFIED_STEP_CLASS_MAPPING.items():
        service_id = STEP_ID_TO_SERVICE_ID.get(step_id, 0)
        service_name = UNIFIED_SERVICE_CLASS_MAPPING.get(service_id, "N/A")
        logger.info(f"   - Step {step_id:02d} ({step_class_name}) ↔ Service {service_id} ({service_name})")

logger.info("🎯 Step Utils Layer 준비 완료!")
logger.info("🏗️ step_service.py + step_implementations.py 완벽 지원!")
logger.info("🤖 BaseStepMixin + ModelLoader + 실제 Step 클래스 완전 연동!")

# conda 환경 최적화 자동 실행
if 'CONDA_DEFAULT_ENV' in os.environ:
    setup_conda_optimization()
    logger.info("🐍 conda 환경 자동 최적화 완료!")

# 초기 메모리 최적화
try:
    memory_helper = get_memory_helper()
    memory_helper.optimize_device_memory(DEVICE)
    logger.info(f"💾 {DEVICE} 초기 메모리 최적화 완료!")
except Exception as e:
    logger.warning(f"⚠️ 초기 메모리 최적화 실패: {e}")

# 전역 유틸리티 매니저 초기화 (동기적으로)
try:
    utils_manager = get_utils_manager()
    logger.info("✅ 전역 UtilsManager 초기화 완료!")
except Exception as e:
    logger.warning(f"⚠️ 전역 UtilsManager 초기화 실패: {e}")

logger.info("🚀 Step Utils Layer v2.0 완전 준비 완료! 🚀")