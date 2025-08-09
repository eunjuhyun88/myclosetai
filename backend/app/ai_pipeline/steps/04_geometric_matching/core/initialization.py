"""
Initialization utilities for geometric matching step.
"""

import torch
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class GeometricMatchingInitializer:
    """기하학적 매칭 초기화 유틸리티"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def initialize_step_attributes(self, step_instance) -> None:
        """스텝 기본 속성 초기화"""
        step_instance.step_name = "geometric_matching"
        step_instance.step_version = "v1.0_modularized"
        step_instance.step_description = "기하학적 매칭 - 모듈화된 버전"
        
        # 성능 통계
        step_instance.processing_stats = {
            'total_processing_time': 0.0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'average_processing_time': 0.0,
            'last_processing_time': 0.0
        }
        
        # 캐시 시스템
        step_instance.cache = {}
        step_instance.cache_enabled = True
        
        # 에러 추적
        step_instance.error_history = []
        step_instance.last_error = None
        
        self.logger.info("✅ 스텝 속성 초기화 완료")
    
    def initialize_geometric_matching_specifics(self, step_instance, **kwargs) -> None:
        """기하학적 매칭 특화 속성 초기화"""
        # 디바이스 설정
        step_instance.device = kwargs.get('device', 'auto')
        if step_instance.device == 'auto':
            step_instance.device = self.detect_optimal_device()
        
        # 모델 경로 매퍼
        from ..utils.model_path_mapper import EnhancedModelPathMapper
        step_instance.model_path_mapper = EnhancedModelPathMapper()
        
        # 입력 크기 설정
        step_instance.input_size = kwargs.get('input_size', (256, 192))
        
        # 신뢰도 임계값
        step_instance.confidence_threshold = kwargs.get('confidence_threshold', 0.7)
        
        # 시각화 설정
        step_instance.enable_visualization = kwargs.get('enable_visualization', True)
        
        self.logger.info(f"🔧 기하학적 매칭 특화 설정 완료 - Device: {step_instance.device}, Input Size: {step_instance.input_size}")
    
    def detect_optimal_device(self) -> str:
        """최적 디바이스 감지"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def detect_m3_max(self) -> bool:
        """M3 Max 디바이스 탐지"""
        try:
            import platform
            if platform.system() == "Darwin":
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                if 'M3 Max' in result.stdout:
                    return True
        except:
            pass
        return False
    
    def apply_m3_max_optimization(self, step_instance) -> None:
        """M3 Max 최적화 적용"""
        try:
            if torch.backends.mps.is_available():
                torch.backends.mps.empty_cache()
                self.logger.info("✅ M3 Max 최적화 적용됨")
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 적용 실패: {e}")
    
    def emergency_setup(self, step_instance, **kwargs) -> None:
        """긴급 설정"""
        self.logger.warning("🚨 긴급 설정 모드로 전환")
        
        # 기본 설정으로 초기화
        step_instance.device = "cpu"
        
        # 모델들을 None으로 설정
        step_instance.geometric_matching_models = {}
        step_instance.advanced_ai_models = {}
        
        self.logger.info("✅ 긴급 설정 완료")
    
    def initialize(self, step_instance) -> bool:
        """비동기 초기화"""
        try:
            self.logger.info("🔄 GeometricMatchingStep 비동기 초기화 시작...")
            
            # 모델 로딩
            from .geometric_matching_model_loader import GeometricMatchingModelLoader
            model_loader = GeometricMatchingModelLoader()
            model_loader.load_geometric_matching_models(step_instance)
            
            # 초기화 완료
            if hasattr(step_instance, 'processing_status'):
                step_instance.processing_status.update_status(initialization_complete=True)
            
            self.logger.info("✅ GeometricMatchingStep 비동기 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 비동기 초기화 실패: {e}")
            return False
