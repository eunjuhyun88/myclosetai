#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Step Core
=====================================================================

ClothSegmentationStep 클래스의 핵심 기능들

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import threading
import gc
import time
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor

try:
    import numpy as np
    import cv2
    NUMPY_AVAILABLE = True
    CV2_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    CV2_AVAILABLE = False
    np = None
    cv2 = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..base import BaseStepMixin
from ..config import ClothSegmentationConfig, SegmentationMethod, ClothCategory, QualityLevel

logger = logging.getLogger(__name__)

class ClothSegmentationStepCore(BaseStepMixin):
    """
    🔥 의류 세그멘테이션 Step - 핵심 기능들
    
    분리된 핵심 기능들:
    - 초기화 및 설정 관리
    - 모델 로딩 및 관리
    - AI 추론 실행
    - 결과 처리 및 검증
    """
    
    def __init__(self, **kwargs):
        """핵심 초기화"""
        try:
            # 🔥 1. 필수 속성들 우선 초기화 (에러 방지)
            self._initialize_critical_attributes()
            
            # 🔥 2. BaseStepMixin 초기화 (안전한 호출)
            try:
                super().__init__(step_name="ClothSegmentationStep", **kwargs)
            except Exception as e:
                logger.warning(f"⚠️ BaseStepMixin 초기화 실패, 폴백 모드: {e}")
                self._fallback_initialization(**kwargs)
            
            # 🔥 3. Cloth Segmentation 특화 초기화
            self._initialize_cloth_segmentation_specifics()
            
            logger.info(f"✅ {self.step_name} 핵심 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ ClothSegmentationStepCore 초기화 실패: {e}")
            self._emergency_setup(**kwargs)

    def _initialize_critical_attributes(self):
        """중요 속성들 우선 초기화"""
        # Logger 먼저 설정
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 필수 속성들
        self.step_name = "ClothSegmentationStep"
        self.step_id = 3
        self.device = "cpu"
        self.is_initialized = False
        self.is_ready = False
        
        # 🔥 누락되었던 속성들 추가 (오류 해결)
        self.segmentation_models = {}
        self.segmentation_ready = False
        self.cloth_cache = {}
        
        # 핵심 컨테이너들
        self.ai_models = {}
        self.model_paths = {}
        self.loaded_models = {}
        self.models_loading_status = {
            'deeplabv3plus': False,
            'maskrcnn': False,
            'sam_huge': False,
            'u2net_cloth': False,
            'total_loaded': 0,
            'loading_errors': []
        }
        
        # 의류 카테고리 정의 (추가)
        self.cloth_categories = {
            0: 'background',
            1: 'shirt', 2: 't_shirt', 3: 'sweater', 4: 'hoodie',
            5: 'jacket', 6: 'coat', 7: 'dress', 8: 'skirt',
            9: 'pants', 10: 'jeans', 11: 'shorts',
            12: 'shoes', 13: 'boots', 14: 'sneakers',
            15: 'bag', 16: 'hat', 17: 'glasses', 18: 'scarf', 19: 'belt'
        }
        
        # 통계 (추가)
        self.ai_stats = {
            'total_processed': 0,
            'deeplabv3_calls': 0,
            'sam_calls': 0,
            'u2net_calls': 0,
            'average_confidence': 0.0
        }
        
        # 의존성 주입 관련
        self.model_loader = None
        self.model_interface = None
        
    def _fallback_initialization(self, **kwargs):
        """BaseStepMixin 초기화 실패시 폴백"""
        self.logger.warning("⚠️ 폴백 초기화 모드")
        self.step_name = kwargs.get('step_name', 'ClothSegmentationStep')
        self.step_id = kwargs.get('step_id', 3)
        self.device = kwargs.get('device', 'cpu')

    def _initialize_step_attributes(self):
        """Step 필수 속성들 초기화 (BaseStepMixin 호환)"""
        self.ai_models = {}
        self.models_loading_status = {
            'deeplabv3plus': False,
            'maskrcnn': False,
            'sam_huge': False,
            'u2net_cloth': False,
            'total_loaded': 0,
            'loading_errors': []
        }
        self.model_interface = None
        self.loaded_models = {}
        
        # Cloth Segmentation 특화 속성들
        self.segmentation_models = {}
        self.segmentation_ready = False
        self.cloth_cache = {}
        
        # 의류 카테고리 정의
        self.cloth_categories = {category.value: category.name.lower() 
                                for category in ClothCategory}
        
        # 통계
        self.ai_stats = {
            'total_processed': 0,
            'deeplabv3_calls': 0,
            'sam_calls': 0,
            'u2net_calls': 0,
            'average_confidence': 0.0
        }
    
    def _initialize_cloth_segmentation_specifics(self):
        """Cloth Segmentation 특화 초기화"""
        try:
            # 설정
            self.config = ClothSegmentationConfig()
            
            # 🔧 핵심 속성들 안전 초기화
            if not hasattr(self, 'model_paths'):
                self.model_paths = {}
            if not hasattr(self, 'ai_models'):
                self.ai_models = {}
            
            # 시스템 최적화
            self.is_m3_max = self._detect_m3_max()
            self.memory_gb = 16.0
            
            # 성능 및 캐싱
            try:
                self.executor = ThreadPoolExecutor(
                    max_workers=4 if self.is_m3_max else 2,
                    thread_name_prefix="cloth_seg"
                )
            except Exception as e:
                logger.warning(f"ThreadPoolExecutor 생성 실패: {e}")
                self.executor = None
            
            self.segmentation_cache = {}
            self.cache_lock = threading.RLock()
            
            # 사용 가능한 방법 초기화
            self.available_methods = []
            
            logger.debug(f"✅ {self.step_name} 특화 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ Cloth Segmentation 특화 초기화 실패: {e}")
            # 🔧 최소한의 속성들 보장
            self.model_paths = {}
            self.ai_models = {}
            self.available_methods = []
    
    def _emergency_setup(self, **kwargs):
        """긴급 설정"""
        try:
            self.logger.warning("⚠️ 긴급 설정 모드")
            self.step_name = kwargs.get('step_name', 'ClothSegmentationStep')
            self.step_id = kwargs.get('step_id', 3)
            self.device = kwargs.get('device', 'cpu')
            self.is_initialized = False
            self.is_ready = False
            self.ai_models = {}
            self.model_paths = {}
            self.ai_stats = {'total_processed': 0}
            self.config = ClothSegmentationConfig()
            self.cache_lock = threading.RLock()
            self.cloth_categories = {category.value: category.name.lower() 
                                    for category in ClothCategory}
        except Exception as e:
            logger.error(f"❌ 긴급 설정도 실패: {e}")
            self.model_paths = {}
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 감지"""
        try:
            import platform
            import subprocess
            if platform.system() == 'Darwin':
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True, timeout=5
                )
                return 'M3' in result.stdout
        except:
            pass
        return False

    def initialize(self) -> bool:
        """AI 모델 초기화 + 메모리 안전성 강화"""
        try:
            if self.is_initialized:
                return True
            
            # 메모리 정리
            gc.collect()
            
            # 메모리 안전성 체크
            try:
                import psutil
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > 90:
                    logger.warning(f"⚠️ 메모리 사용량이 높습니다: {memory_usage}% - 안전 모드로 전환")
                    return self._fallback_initialization()
            except ImportError:
                pass
            
            logger.info(f"🔄 {self.step_name} AI 모델 초기화 시작...")
            
            # 🔥 1. 모델 로딩 (메모리 안전 모드)
            try:
                logger.info("🔄 AI 모델 로딩 시작...")
                self._load_segmentation_models_via_central_hub()
                logger.info("✅ AI 모델 로딩 완료")
            except Exception as e:
                logger.error(f"❌ AI 모델 로딩 실패: {e}")
                return self._fallback_initialization()
            
            # 2. 사용 가능한 방법 감지
            self.available_methods = self._detect_available_methods()
            
            # 3. BaseStepMixin 초기화
            super_initialized = super().initialize() if hasattr(super(), 'initialize') else True
            
            self.is_initialized = True
            self.is_ready = True
            self.segmentation_ready = len(self.ai_models) > 0
            
            logger.info(f"✅ {self.step_name} 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 초기화 실패: {e}")
            return False

    def _load_segmentation_models_via_central_hub(self):
        """Central Hub를 통한 세그멘테이션 모델 로딩"""
        # 이 메서드는 model_loader_service.py로 이동될 예정
        pass

    def _detect_available_methods(self) -> List[SegmentationMethod]:
        """사용 가능한 세그멘테이션 방법 감지"""
        available_methods = []
        
        # U2Net 체크
        if 'u2net_cloth' in self.ai_models:
            available_methods.append(SegmentationMethod.U2NET_CLOTH)
        
        # SAM 체크
        if 'sam_huge' in self.ai_models:
            available_methods.append(SegmentationMethod.SAM_HUGE)
        
        # DeepLabV3+ 체크
        if 'deeplabv3plus' in self.ai_models:
            available_methods.append(SegmentationMethod.DEEPLABV3_PLUS)
        
        # 하이브리드 AI (여러 모델이 있을 때)
        if len(available_methods) > 1:
            available_methods.append(SegmentationMethod.HYBRID_AI)
        
        return available_methods

    def cleanup(self):
        """리소스 정리"""
        try:
            # 캐시 정리
            if hasattr(self, 'segmentation_cache'):
                self.segmentation_cache.clear()
            
            # 모델 정리
            if hasattr(self, 'ai_models'):
                for model in self.ai_models.values():
                    if hasattr(model, 'cleanup'):
                        model.cleanup()
                self.ai_models.clear()
            
            # Executor 정리
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=True)
            
            # 메모리 정리
            gc.collect()
            if TORCH_AVAILABLE and hasattr(torch, 'mps') and torch.mps.is_available():
                torch.mps.empty_cache()
            
            logger.info(f"✅ {self.step_name} 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ 리소스 정리 실패: {e}")

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            'step_name': self.step_name,
            'step_id': self.step_id,
            'is_initialized': self.is_initialized,
            'is_ready': self.is_ready,
            'segmentation_ready': self.segmentation_ready,
            'available_methods': [method.value for method in self.available_methods],
            'models_loaded': len(self.ai_models),
            'total_processed': self.ai_stats.get('total_processed', 0),
            'memory_usage': self._get_memory_usage()
        }

    def _get_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 정보"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_percent': memory.percent
            }
        except ImportError:
            return {'error': 'psutil not available'}
