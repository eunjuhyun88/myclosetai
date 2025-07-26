# backend/app/services/step_implementations.py
"""
🔥 MyCloset AI Step Implementations - 실제 Step 클래스 브릿지 v6.0
================================================================================

✅ 올바른 역할: ai_pipeline/steps/step_XX.py 클래스들과의 브릿지
✅ 실제 AI 모델 연동: 각 Step 클래스가 담당 (ModelLoader + UnifiedDependencyManager)
✅ step_implementations.py: Step 인스턴스 관리 및 호출만 담당
✅ BaseStepMixin 완전 호환성
✅ 기존 API 100% 유지
✅ 순환참조 완전 방지

올바른 구조:
step_routes.py → step_service.py → step_implementations.py → ai_pipeline/steps/step_XX.py
                                                              ↓
                                                         ModelLoader + 실제 AI 모델

Author: MyCloset AI Team  
Date: 2025-07-26
Version: 6.0 (올바른 브릿지 구조)
"""

import logging
import asyncio
import time
import threading
import uuid
import base64
import json
import gc
import importlib
import traceback
import weakref
import os
import sys
from typing import Dict, Any, Optional, List, Union, Tuple, Type, TYPE_CHECKING
from datetime import datetime
from pathlib import Path
from io import BytesIO
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

# 안전한 타입 힌팅
if TYPE_CHECKING:
    from fastapi import UploadFile
    import torch
    import numpy as np
    from PIL import Image

# ==============================================
# 🔥 로깅 설정
# ==============================================
logger = logging.getLogger(__name__)

# ==============================================
# 🔥 실제 Step 클래스 매핑 import
# ==============================================

try:
    from .unified_step_mapping import (
        REAL_STEP_CLASS_MAPPING,
        SERVICE_CLASS_MAPPING,
        SERVICE_TO_STEP_MAPPING,
        STEP_TO_SERVICE_MAPPING,
        SERVICE_NAME_TO_STEP_CLASS,
        STEP_CLASS_TO_SERVICE_NAME,
        RealStepSignature,
        REAL_STEP_SIGNATURES,
        StepFactory,
        setup_conda_optimization,
        validate_step_compatibility,
        get_all_available_steps,
        get_all_available_services,
        get_system_compatibility_info,
        create_step_data_mapper
    )
    REAL_MAPPING_AVAILABLE = True
    logger.info("✅ 실제 Step 클래스 매핑 import 성공")
except ImportError as e:
    REAL_MAPPING_AVAILABLE = False
    logger.error(f"❌ 실제 Step 클래스 매핑 import 실패: {e}")
    # 폴백용 더미 데이터
    REAL_STEP_CLASS_MAPPING = {
        1: "Step01HumanParsing", 2: "Step02PoseEstimation", 3: "Step03ClothSegmentation", 
        4: "Step04GeometricMatching", 5: "Step05ClothWarping", 6: "Step06VirtualFitting",
        7: "Step07PostProcessing", 8: "Step08QualityAssessment"
    }
    REAL_STEP_SIGNATURES = {}

# ==============================================
# 🔥 안전한 Import 시스템
# ==============================================

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

# BaseStepMixin import (핵심!)
try:
    from ..ai_pipeline.steps.base_step_mixin import BaseStepMixin
    BASE_STEP_MIXIN_AVAILABLE = True
    logger.info("✅ BaseStepMixin import 성공")
except ImportError:
    BASE_STEP_MIXIN_AVAILABLE = False
    logger.warning("⚠️ BaseStepMixin import 실패")
    
    class BaseStepMixin:
        def __init__(self, **kwargs):
            self.logger = logging.getLogger(self.__class__.__name__)
            self.device = kwargs.get('device', 'cpu')
            self.is_initialized = False
        
        def initialize(self):
            self.is_initialized = True
            return True
        
        def cleanup(self):
            pass

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
# 🔥 실제 Step 클래스 동적 로딩 시스템
# ==============================================

class RealStepClassLoader:
    """실제 ai_pipeline/steps/step_XX.py 클래스 동적 로딩"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RealStepClassLoader")
        self.loaded_classes: Dict[int, Type] = {}
        self.import_cache: Dict[int, str] = {}
        self._lock = threading.RLock()
        
        # Step별 import 경로 매핑
        self.step_import_paths = {
            1: "app.ai_pipeline.steps.step_01_human_parsing",
            2: "app.ai_pipeline.steps.step_02_pose_estimation", 
            3: "app.ai_pipeline.steps.step_03_cloth_segmentation",
            4: "app.ai_pipeline.steps.step_04_geometric_matching",
            5: "app.ai_pipeline.steps.step_05_cloth_warping",
            6: "app.ai_pipeline.steps.step_06_virtual_fitting",
            7: "app.ai_pipeline.steps.step_07_post_processing",
            8: "app.ai_pipeline.steps.step_08_quality_assessment"
        }
        
        # Step별 클래스명 매핑
        self.step_class_names = {
            1: "Step01HumanParsing",
            2: "Step02PoseEstimation",
            3: "Step03ClothSegmentation", 
            4: "Step04GeometricMatching",
            5: "Step05ClothWarping",
            6: "Step06VirtualFitting",
            7: "Step07PostProcessing",
            8: "Step08QualityAssessment"
        }
    
    def load_step_class(self, step_id: int) -> Optional[Type]:
        """실제 Step 클래스 동적 로딩"""
        try:
            with self._lock:
                # 캐시 확인
                if step_id in self.loaded_classes:
                    return self.loaded_classes[step_id]
                
                # import 경로 확인
                import_path = self.step_import_paths.get(step_id)
                class_name = self.step_class_names.get(step_id)
                
                if not import_path or not class_name:
                    self.logger.error(f"Step {step_id}의 import 정보 없음")
                    return None
                
                # 동적 import 시도
                step_class = self._try_import_step_class(import_path, class_name, step_id)
                
                if step_class:
                    self.loaded_classes[step_id] = step_class
                    self.import_cache[step_id] = import_path
                    self.logger.info(f"✅ Step {step_id} 클래스 로딩 성공: {class_name}")
                    return step_class
                else:
                    self.logger.error(f"❌ Step {step_id} 클래스 로딩 실패: {class_name}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ Step {step_id} 클래스 로딩 예외: {e}")
            return None
    
    def _try_import_step_class(self, import_path: str, class_name: str, step_id: int) -> Optional[Type]:
        """다양한 import 경로로 Step 클래스 시도"""
        import_attempts = [
            import_path,  # 기본 경로
            import_path.replace('app.', ''),  # app. 제거
            f"ai_pipeline.steps.step_{step_id:02d}",  # 간단한 경로
            f"backend.app.ai_pipeline.steps.step_{step_id:02d}_{class_name.lower().replace('step0', '').replace('step', '')}"  # 풀 경로
        ]
        
        for attempt_path in import_attempts:
            try:
                self.logger.debug(f"Step {step_id} import 시도: {attempt_path}")
                
                # 모듈 import
                module = importlib.import_module(attempt_path)
                
                # 클래스 조회
                if hasattr(module, class_name):
                    step_class = getattr(module, class_name)
                    
                    # 클래스 검증
                    if self._validate_step_class(step_class, step_id, class_name):
                        self.logger.info(f"✅ Step {step_id} 클래스 import 성공: {attempt_path}.{class_name}")
                        return step_class
                    else:
                        self.logger.warning(f"⚠️ Step {step_id} 클래스 검증 실패: {class_name}")
                        continue
                else:
                    self.logger.debug(f"Step {step_id} 클래스 {class_name}를 {attempt_path}에서 찾을 수 없음")
                    continue
                    
            except ImportError as e:
                self.logger.debug(f"Step {step_id} import 실패 ({attempt_path}): {e}")
                continue
            except Exception as e:
                self.logger.warning(f"Step {step_id} import 예외 ({attempt_path}): {e}")
                continue
        
        return None
    
    def _validate_step_class(self, step_class: Type, step_id: int, class_name: str) -> bool:
        """Step 클래스 검증"""
        try:
            # 기본 검사
            if not step_class:
                return False
            
            # 필수 메서드 확인
            required_methods = ['process']
            for method in required_methods:
                if not hasattr(step_class, method):
                    self.logger.warning(f"⚠️ {class_name}에 필수 메서드 {method} 없음")
                    return False
            
            # BaseStepMixin 상속 확인 (선택적)
            try:
                mro = [cls.__name__ for cls in step_class.__mro__]
                if 'BaseStepMixin' in mro:
                    self.logger.debug(f"✅ {class_name} BaseStepMixin 상속 확인")
                else:
                    self.logger.debug(f"ℹ️ {class_name} BaseStepMixin 미상속 (선택적)")
            except:
                pass
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {class_name} 클래스 검증 실패: {e}")
            return False
    
    def get_loaded_classes_info(self) -> Dict[str, Any]:
        """로딩된 클래스 정보"""
        with self._lock:
            return {
                "loaded_classes": {
                    step_id: {
                        "class_name": cls.__name__,
                        "import_path": self.import_cache.get(step_id, "unknown"),
                        "module": cls.__module__
                    }
                    for step_id, cls in self.loaded_classes.items()
                },
                "total_loaded": len(self.loaded_classes),
                "available_steps": list(self.step_import_paths.keys())
            }

# ==============================================
# 🔥 Step 구현체 브릿지 클래스
# ==============================================

class StepImplementationBridge:
    """개별 Step 클래스와의 브릿지"""
    
    def __init__(self, step_id: int, step_class: Type, **config):
        self.step_id = step_id
        self.step_class = step_class
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.StepBridge.{step_id}")
        
        # Step 인스턴스 생성
        self.step_instance = None
        self.is_initialized = False
        self._lock = threading.RLock()
        
        # 성능 메트릭
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.processing_times = []
        
        self.logger.info(f"✅ Step {step_id} 브릿지 생성: {step_class.__name__}")
    
    def initialize(self) -> bool:
        """Step 인스턴스 초기화"""
        try:
            if self.is_initialized and self.step_instance:
                return True
            
            with self._lock:
                # Step 인스턴스 생성
                step_config = {
                    'step_id': self.step_id,
                    'step_name': f"Step{self.step_id:02d}",
                    'device': self.config.get('device', DEVICE),
                    **self.config
                }
                
                self.logger.info(f"🔄 Step {self.step_id} 인스턴스 생성 중...")
                self.step_instance = self.step_class(**step_config)
                
                # Step 초기화 호출
                if hasattr(self.step_instance, 'initialize'):
                    success = self.step_instance.initialize()
                    if not success:
                        self.logger.error(f"❌ Step {self.step_id} 초기화 실패")
                        return False
                
                self.is_initialized = True
                self.logger.info(f"✅ Step {self.step_id} 브릿지 초기화 완료")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Step {self.step_id} 브릿지 초기화 실패: {e}")
            return False
    
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Step 처리 (실제 Step 클래스의 process 호출)"""
        start_time = time.time()
        
        try:
            with self._lock:
                self.total_requests += 1
            
            # 초기화 확인
            if not self.is_initialized:
                if not self.initialize():
                    raise Exception(f"Step {self.step_id} 초기화 실패")
            
            if not self.step_instance:
                raise Exception(f"Step {self.step_id} 인스턴스 없음")
            
            # 실제 Step 클래스의 process 메서드 호출
            self.logger.debug(f"🔄 Step {self.step_id} 실제 처리 시작...")
            
            if asyncio.iscoroutinefunction(self.step_instance.process):
                result = await self.step_instance.process(*args, **kwargs)
            else:
                result = self.step_instance.process(*args, **kwargs)
            
            # 처리 시간 기록
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # 결과 검증 및 포맷
            if isinstance(result, dict):
                if 'success' not in result:
                    result['success'] = True
                
                if 'details' not in result:
                    result['details'] = {}
                
                # 브릿지 메타데이터 추가
                result['details'].update({
                    'step_id': self.step_id,
                    'step_class': self.step_class.__name__,
                    'processing_time': processing_time,
                    'bridge_mode': True,
                    'real_ai_processing': True,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                # dict가 아닌 결과를 dict로 변환
                result = {
                    'success': True,
                    'result': result,
                    'details': {
                        'step_id': self.step_id,
                        'step_class': self.step_class.__name__,
                        'processing_time': processing_time,
                        'bridge_mode': True,
                        'real_ai_processing': True,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            
            with self._lock:
                self.successful_requests += 1
            
            self.logger.info(f"✅ Step {self.step_id} 처리 완료 ({processing_time:.2f}초)")
            return result
            
        except Exception as e:
            with self._lock:
                self.failed_requests += 1
            
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Step {self.step_id} 처리 실패: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'details': {
                    'step_id': self.step_id,
                    'step_class': self.step_class.__name__ if self.step_class else 'Unknown',
                    'processing_time': processing_time,
                    'bridge_mode': True,
                    'error_type': type(e).__name__,
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def cleanup(self):
        """Step 인스턴스 정리"""
        try:
            if self.step_instance and hasattr(self.step_instance, 'cleanup'):
                self.step_instance.cleanup()
            
            self.step_instance = None
            self.is_initialized = False
            
            self.logger.info(f"✅ Step {self.step_id} 브릿지 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ Step {self.step_id} 브릿지 정리 실패: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """브릿지 메트릭"""
        with self._lock:
            avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
            
            return {
                'step_id': self.step_id,
                'step_class': self.step_class.__name__ if self.step_class else 'Unknown',
                'is_initialized': self.is_initialized,
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': self.successful_requests / max(self.total_requests, 1),
                'average_processing_time': avg_time,
                'has_step_instance': self.step_instance is not None
            }

# ==============================================
# 🔥 실제 Step 구현체 관리자 (브릿지 버전)
# ==============================================

class RealStepImplementationManager:
    """실제 Step 클래스들과의 브릿지 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RealStepImplementationManager")
        self.class_loader = RealStepClassLoader()
        self.step_bridges: Dict[int, StepImplementationBridge] = {}
        self._lock = threading.RLock()
        
        # 시스템 상태
        if REAL_MAPPING_AVAILABLE:
            self.system_info = get_system_compatibility_info()
        else:
            self.system_info = {"total_steps": 8, "total_services": 8}
        
        # 전체 매니저 메트릭
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = datetime.now()
        
        # conda 환경 최적화
        if REAL_MAPPING_AVAILABLE:
            setup_conda_optimization()
        
        self.logger.info("✅ RealStepImplementationManager 초기화 완료 (브릿지 모드)")
        self.logger.info(f"📊 지원 Step: {self.system_info.get('total_steps', 8)}개")
    
    def get_step_bridge(self, step_id: int, **config) -> Optional[StepImplementationBridge]:
        """Step 브릿지 인스턴스 반환"""
        with self._lock:
            # 캐시 확인
            if step_id in self.step_bridges:
                return self.step_bridges[step_id]
            
            # Step 클래스 로딩
            step_class = self.class_loader.load_step_class(step_id)
            if not step_class:
                self.logger.error(f"❌ Step {step_id} 클래스 로딩 실패")
                return None
            
            # 브릿지 생성
            try:
                bridge = StepImplementationBridge(step_id, step_class, **config)
                if bridge.initialize():
                    self.step_bridges[step_id] = bridge
                    self.logger.info(f"✅ Step {step_id} 브릿지 생성 완료")
                    return bridge
                else:
                    self.logger.error(f"❌ Step {step_id} 브릿지 초기화 실패")
                    return None
                    
            except Exception as e:
                self.logger.error(f"❌ Step {step_id} 브릿지 생성 실패: {e}")
                return None
    
    async def process_step(self, step_id: int, *args, **kwargs) -> Dict[str, Any]:
        """Step 처리 (브릿지를 통한 실제 Step 클래스 호출)"""
        try:
            with self._lock:
                self.total_requests += 1
            
            # Step 브릿지 조회
            bridge = self.get_step_bridge(step_id)
            if not bridge:
                with self._lock:
                    self.failed_requests += 1
                return {
                    "success": False,
                    "error": f"Step {step_id} 브릿지를 찾을 수 없음",
                    "step_id": step_id,
                    "bridge_mode": True,
                    "timestamp": datetime.now().isoformat()
                }
            
            # 실제 Step 처리
            result = await bridge.process(*args, **kwargs)
            
            # 메트릭 업데이트
            with self._lock:
                if result.get("success", False):
                    self.successful_requests += 1
                else:
                    self.failed_requests += 1
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failed_requests += 1
            
            self.logger.error(f"❌ Step {step_id} 처리 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "step_id": step_id,
                "bridge_mode": True,
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """모든 브릿지 메트릭 반환"""
        with self._lock:
            bridge_metrics = {}
            for step_id, bridge in self.step_bridges.items():
                bridge_metrics[f"step_{step_id}"] = bridge.get_metrics()
            
            return {
                "manager_version": "6.0_bridge_mode",
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": self.successful_requests / max(self.total_requests, 1),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "real_mapping_available": REAL_MAPPING_AVAILABLE,
                "system_compatibility": self.system_info,
                "architecture": "Step 클래스 브릿지 (ai_pipeline/steps/step_XX.py 연동)",
                "ai_model_responsibility": "각 Step 클래스가 ModelLoader + AI 모델 담당",
                "bridge_responsibility": "Step 인스턴스 관리 및 호출만 담당",
                "basestepmixin_integration": BASE_STEP_MIXIN_AVAILABLE,
                "conda_optimization": 'CONDA_DEFAULT_ENV' in os.environ,
                "loaded_classes": self.class_loader.get_loaded_classes_info(),
                "active_bridges": len(self.step_bridges),
                "bridge_metrics": bridge_metrics
            }
    
    def cleanup_all_bridges(self):
        """모든 브릿지 정리"""
        try:
            with self._lock:
                for step_id, bridge in self.step_bridges.items():
                    try:
                        bridge.cleanup()
                        self.logger.info(f"✅ Step {step_id} 브릿지 정리 완료")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Step {step_id} 브릿지 정리 실패: {e}")
                
                self.step_bridges.clear()
            
            # 메모리 정리
            if TORCH_AVAILABLE:
                if DEVICE == "mps" and IS_M3_MAX:
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                elif DEVICE == "cuda":
                    torch.cuda.empty_cache()
            
            gc.collect()
            
            self.logger.info("✅ 모든 Step 브릿지 정리 완료")
        except Exception as e:
            self.logger.error(f"❌ Step 브릿지 정리 실패: {e}")

# ==============================================
# 🔥 싱글톤 관리자 인스턴스
# ==============================================

_real_step_implementation_manager_instance: Optional[RealStepImplementationManager] = None
_manager_lock = threading.RLock()

def get_step_implementation_manager() -> RealStepImplementationManager:
    """RealStepImplementationManager 싱글톤 인스턴스 반환"""
    global _real_step_implementation_manager_instance
    
    with _manager_lock:
        if _real_step_implementation_manager_instance is None:
            _real_step_implementation_manager_instance = RealStepImplementationManager()
            logger.info("✅ RealStepImplementationManager 싱글톤 인스턴스 생성 완료")
    
    return _real_step_implementation_manager_instance

async def get_step_implementation_manager_async() -> RealStepImplementationManager:
    """RealStepImplementationManager 싱글톤 인스턴스 반환 - 비동기 버전"""
    return get_step_implementation_manager()

def cleanup_step_implementation_manager():
    """RealStepImplementationManager 정리"""
    global _real_step_implementation_manager_instance
    
    with _manager_lock:
        if _real_step_implementation_manager_instance:
            _real_step_implementation_manager_instance.cleanup_all_bridges()
            _real_step_implementation_manager_instance = None
            logger.info("🧹 RealStepImplementationManager 정리 완료")

# ==============================================
# 🔥 편의 함수들 (기존 API 100% 호환)
# ==============================================

async def process_human_parsing_implementation(
    person_image,
    enhance_quality: bool = True,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """인간 파싱 구현체 처리 - ai_pipeline/steps/step_01_human_parsing.py 호출"""
    manager = get_step_implementation_manager()
    return await manager.process_step(
        1, person_image=person_image, enhance_quality=enhance_quality, session_id=session_id, **kwargs
    )

async def process_pose_estimation_implementation(
    image,
    clothing_type: str = "shirt",
    detection_confidence: float = 0.5,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """포즈 추정 구현체 처리 - ai_pipeline/steps/step_02_pose_estimation.py 호출"""
    manager = get_step_implementation_manager()
    return await manager.process_step(
        2, image=image, clothing_type=clothing_type, detection_confidence=detection_confidence, session_id=session_id, **kwargs
    )

async def process_cloth_segmentation_implementation(
    image,
    clothing_type: str = "shirt",
    quality_level: str = "medium",
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """의류 분할 구현체 처리 - ai_pipeline/steps/step_03_cloth_segmentation.py 호출"""
    manager = get_step_implementation_manager()
    return await manager.process_step(
        3, image=image, clothing_type=clothing_type, quality_level=quality_level, session_id=session_id, **kwargs
    )

async def process_geometric_matching_implementation(
    person_image,
    clothing_image,
    pose_keypoints=None,
    body_mask=None,
    clothing_mask=None,
    matching_precision: str = "high",
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """기하학적 매칭 구현체 처리 - ai_pipeline/steps/step_04_geometric_matching.py 호출"""
    manager = get_step_implementation_manager()
    return await manager.process_step(
        4, person_image=person_image, clothing_image=clothing_image, pose_keypoints=pose_keypoints, 
        body_mask=body_mask, clothing_mask=clothing_mask, matching_precision=matching_precision, 
        session_id=session_id, **kwargs
    )

async def process_cloth_warping_implementation(
    cloth_image,
    person_image,
    cloth_mask=None,
    fabric_type: str = "cotton",
    clothing_type: str = "shirt",
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """의류 워핑 구현체 처리 - ai_pipeline/steps/step_05_cloth_warping.py 호출"""
    manager = get_step_implementation_manager()
    return await manager.process_step(
        5, cloth_image=cloth_image, person_image=person_image, cloth_mask=cloth_mask, 
        fabric_type=fabric_type, clothing_type=clothing_type, session_id=session_id, **kwargs
    )

async def process_virtual_fitting_implementation(
    person_image,
    cloth_image,
    pose_data=None,
    cloth_mask=None,
    fitting_quality: str = "high",
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """가상 피팅 구현체 처리 - ai_pipeline/steps/step_06_virtual_fitting.py 호출 (핵심!)"""
    manager = get_step_implementation_manager()
    return await manager.process_step(
        6, person_image=person_image, cloth_image=cloth_image, pose_data=pose_data, 
        cloth_mask=cloth_mask, fitting_quality=fitting_quality, session_id=session_id, **kwargs
    )

async def process_post_processing_implementation(
    fitted_image,
    enhancement_level: str = "medium",
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """후처리 구현체 처리 - ai_pipeline/steps/step_07_post_processing.py 호출"""
    manager = get_step_implementation_manager()
    return await manager.process_step(
        7, fitted_image=fitted_image, enhancement_level=enhancement_level, session_id=session_id, **kwargs
    )

async def process_quality_assessment_implementation(
    final_image,
    analysis_depth: str = "comprehensive",
    session_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """품질 평가 구현체 처리 - ai_pipeline/steps/step_08_quality_assessment.py 호출"""
    manager = get_step_implementation_manager()
    return await manager.process_step(
        8, final_image=final_image, analysis_depth=analysis_depth, session_id=session_id, **kwargs
    )

# ==============================================
# 🔥 상태 및 가용성 정보
# ==============================================

STEP_IMPLEMENTATIONS_AVAILABLE = True

def get_implementation_availability_info() -> Dict[str, Any]:
    """실제 Step 구현체 가용성 정보 반환"""
    return {
        "step_implementations_available": STEP_IMPLEMENTATIONS_AVAILABLE,
        "architecture": "Step 클래스 브릿지 (ai_pipeline/steps/step_XX.py 연동)",
        "version": "6.0_bridge_mode",
        "api_compatibility": "100%",
        "real_mapping_available": REAL_MAPPING_AVAILABLE,
        "real_step_implementation": True,
        "basestepmixin_integration": BASE_STEP_MIXIN_AVAILABLE,
        "step_class_mappings": REAL_STEP_CLASS_MAPPING,
        "total_steps_supported": len(REAL_STEP_CLASS_MAPPING),
        "real_step_classes_integrated": True,
        "ai_model_responsibility": "각 Step 클래스가 ModelLoader + AI 모델 담당",
        "bridge_responsibility": "Step 인스턴스 관리 및 호출만 담당",
        "conda_optimization": 'CONDA_DEFAULT_ENV' in os.environ,
        "device_optimization": f"{DEVICE}_optimized",
        "production_ready": True,
        "correct_architecture": True,
        "step_classes_location": "ai_pipeline/steps/step_XX.py",
        "ai_models_location": "각 Step 클래스 내부 (ModelLoader 사용)",
        "bridge_pattern": {
            "step_routes.py": "API 엔드포인트",
            "step_service.py": "DI 기반 서비스 매니저",
            "step_implementations.py": "Step 클래스 브릿지 (이 파일)",
            "ai_pipeline/steps/step_XX.py": "실제 AI 모델 + 처리 로직"
        }
    }

# ==============================================
# 🔥 conda 환경 최적화 함수들
# ==============================================

def setup_conda_step_implementations():
    """conda 환경에서 Step 구현체 최적화 설정"""
    try:
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            logger.info(f"🐍 conda 환경 감지: {conda_env}")
            
            # PyTorch conda 최적화
            if TORCH_AVAILABLE:
                # MPS 최적화 (M3 Max)
                if DEVICE == "mps":
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                    logger.info("🍎 M3 Max MPS 최적화 활성화")
                
                # CPU 스레드 최적화
                cpu_count = os.cpu_count()
                torch.set_num_threads(max(1, cpu_count // 2))
                logger.info(f"🧵 PyTorch 스레드 최적화: {torch.get_num_threads()}/{cpu_count}")
            
            # 환경 변수 설정
            os.environ['OMP_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
            os.environ['MKL_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
            
            return True
    except Exception as e:
        logger.warning(f"⚠️ conda 최적화 설정 실패: {e}")
        return False

def validate_conda_environment():
    """conda 환경 검증"""
    try:
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if not conda_env:
            logger.warning("⚠️ conda 환경이 활성화되지 않음")
            return False
        
        # 필수 패키지 확인
        required_packages = ['numpy', 'pillow']
        missing_packages = []
        
        if not NUMPY_AVAILABLE:
            missing_packages.append('numpy')
        if not PIL_AVAILABLE:
            missing_packages.append('pillow')
        
        if missing_packages:
            logger.warning(f"⚠️ conda 환경에 누락된 패키지: {missing_packages}")
            return False
        
        logger.info(f"✅ conda 환경 검증 완료: {conda_env}")
        return True
        
    except Exception as e:
        logger.error(f"❌ conda 환경 검증 실패: {e}")
        return False

# ==============================================
# 🔥 모듈 Export
# ==============================================

__all__ = [
    # 메인 클래스들
    "RealStepImplementationManager",
    "StepImplementationBridge",
    "RealStepClassLoader",
    
    # 관리자 함수들
    "get_step_implementation_manager",
    "get_step_implementation_manager_async",
    "cleanup_step_implementation_manager",
    
    # 편의 함수들 (ai_pipeline/steps/step_XX.py 호출)
    "process_human_parsing_implementation",
    "process_pose_estimation_implementation",
    "process_cloth_segmentation_implementation",
    "process_geometric_matching_implementation",
    "process_cloth_warping_implementation",
    "process_virtual_fitting_implementation",
    "process_post_processing_implementation",
    "process_quality_assessment_implementation",
    
    # 유틸리티
    "get_implementation_availability_info",
    "setup_conda_step_implementations",
    "validate_conda_environment",
    
    # 스키마
    "BodyMeasurements",
    
    # 상수
    "STEP_IMPLEMENTATIONS_AVAILABLE"
]

# 호환성을 위한 별칭
StepImplementationManager = RealStepImplementationManager

# ==============================================
# 🔥 모듈 로드 완료 메시지
# ==============================================

logger.info("🔥 Step Implementations v6.0 로드 완료 (올바른 브릿지 구조)!")
logger.info("✅ 올바른 아키텍처:")
logger.info("   step_routes.py → step_service.py → step_implementations.py → ai_pipeline/steps/step_XX.py")
logger.info("✅ step_implementations.py 역할: Step 클래스 브릿지 (인스턴스 관리 + 호출)")
logger.info("✅ AI 모델 처리: 각 ai_pipeline/steps/step_XX.py에서 ModelLoader 사용")
logger.info("✅ 순환참조 완전 방지: 단방향 의존성 구조")
logger.info("✅ 기존 API 100% 호환: 모든 함수명/시그니처 유지")

logger.info(f"📊 시스템 상태:")
logger.info(f"   - 실제 매핑: {'✅' if REAL_MAPPING_AVAILABLE else '❌'}")
logger.info(f"   - PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
logger.info(f"   - BaseStepMixin: {'✅' if BASE_STEP_MIXIN_AVAILABLE else '❌'}")
logger.info(f"   - Device: {DEVICE}")
logger.info(f"   - conda 환경: {'✅' if 'CONDA_DEFAULT_ENV' in os.environ else '❌'}")

logger.info("🎯 Step 클래스 로딩:")
for step_id, class_name in REAL_STEP_CLASS_MAPPING.items():
    logger.info(f"   - Step {step_id}: {class_name}")

# conda 환경 최적화 자동 실행
if 'CONDA_DEFAULT_ENV' in os.environ:
    setup_conda_step_implementations()
    if validate_conda_environment():
        logger.info("🐍 conda 환경 자동 최적화 및 검증 완료!")
    else:
        logger.warning("⚠️ conda 환경 검증 실패!")

# 초기 메모리 최적화
try:
    if TORCH_AVAILABLE:
        if DEVICE == "mps" and IS_M3_MAX:
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        elif DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    gc.collect()
    logger.info(f"💾 {DEVICE} 초기 메모리 최적화 완료!")
except Exception as e:
    logger.warning(f"⚠️ 초기 메모리 최적화 실패: {e}")

logger.info("🎉 Step Implementations v6.0 완전 준비 완료!")
logger.info("🚀 올바른 브릿지 구조로 실제 AI Step 클래스들과 연동!")
logger.info("💯 ai_pipeline/steps/step_XX.py에서 실제 AI 모델 처리!")