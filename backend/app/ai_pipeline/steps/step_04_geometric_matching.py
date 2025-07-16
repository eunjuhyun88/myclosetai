# app/ai_pipeline/steps/step_04_geometric_matching.py
"""
4단계: 기하학적 매칭 (Geometric Matching) - 최적 생성자 패턴 적용
M3 Max 최적화 + 견고한 에러 처리 + 기존 기능 100% 유지
"""
import os
import logging
import time
import asyncio
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from PIL import Image
import cv2

# PyTorch 선택적 임포트
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# SciPy 선택적 임포트
try:
    from scipy.spatial.distance import cdist
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    cdist = None

logger = logging.getLogger(__name__)
# backend/app/ai_pipeline/steps/step_04_geometric_matching.py 생성자 수정

class GeometricMatchingStep:
    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device_type: Optional[str] = None,
        memory_gb: Optional[float] = None,
        is_m3_max: Optional[bool] = None,
        optimization_enabled: Optional[bool] = None,
        quality_level: Optional[str] = None,  # ✅ 추가됨
        **kwargs
    ):
        """🔧 완전 호환 생성자 - 모든 파라미터 지원"""
        
        # 기본값 설정
        self.device = self._auto_detect_device(device)
        self.config = config or {}
        self.step_name = self.__class__.__name__
        self.logger = logging.getLogger(f"pipeline.{self.step_name}")
        
        # 파라미터 처리 (None 체크 추가)
        self.device_type = device_type or kwargs.get('device_type', 'auto')
        self.memory_gb = memory_gb or kwargs.get('memory_gb', 16.0)
        self.is_m3_max = is_m3_max if is_m3_max is not None else kwargs.get('is_m3_max', self._detect_m3_max())
        self.optimization_enabled = optimization_enabled if optimization_enabled is not None else kwargs.get('optimization_enabled', True)
        self.quality_level = quality_level or kwargs.get('quality_level', 'balanced')  # ✅ 수정됨
        
        # 기존 초기화 로직...
        self._merge_step_specific_config(kwargs)
        self.is_initialized = False
        self.initialization_error = None
        
        # ModelLoader 인터페이스 설정
        try:
            from app.ai_pipeline.utils.model_loader import BaseStepMixin
            if hasattr(BaseStepMixin, '_setup_model_interface'):
                BaseStepMixin._setup_model_interface(self)
        except Exception as e:
            self.logger.warning(f"ModelLoader 인터페이스 설정 실패: {e}")
        
        # 스텝 특화 초기화
        self._initialize_step_specific()
        self.logger.info(f"🎯 {self.step_name} 초기화 - 디바이스: {self.device}")
        
        # 나머지 초기화 로직은 동일...
        # (matching_config, tps_config, optimization_config 등)
    
    def _initialize_step_specific(self):
        """스텝별 특화 초기화"""
        try:
            # 매칭 설정 (quality_level 반영)
            base_config = {
                'method': 'auto',
                'max_iterations': 1000,
                'convergence_threshold': 1e-6,
                'outlier_threshold': 0.15,
                'use_pose_guidance': True,
                'adaptive_weights': True,
                'quality_threshold': 0.7
            }
            
            # quality_level에 따른 조정
            if self.quality_level == 'high':
                base_config.update({
                    'max_iterations': 1500,
                    'quality_threshold': 0.8,
                    'convergence_threshold': 1e-7
                })
            elif self.quality_level == 'ultra':
                base_config.update({
                    'max_iterations': 2000,
                    'quality_threshold': 0.9,
                    'convergence_threshold': 1e-8
                })
            elif self.quality_level == 'fast':
                base_config.update({
                    'max_iterations': 500,
                    'quality_threshold': 0.6,
                    'convergence_threshold': 1e-5
                })
            
            self.matching_config = self.config.get('matching', base_config)
            
            # TPS 설정 (M3 Max 최적화)
            self.tps_config = self.config.get('tps', {
                'regularization': 0.1,
                'grid_size': 30 if self.is_m3_max else 20,
                'boundary_padding': 0.1
            })
            
            # 최적화 설정
            learning_rate_base = 0.01
            if self.is_m3_max and self.optimization_enabled:
                learning_rate_base *= 1.2
            
            self.optimization_config = self.config.get('optimization', {
                'learning_rate': learning_rate_base,
                'momentum': 0.9,
                'weight_decay': 1e-4,
                'scheduler_step': 100
            })
            
            # 매칭 통계
            self.matching_stats = {
                'total_matches': 0,
                'successful_matches': 0,
                'average_accuracy': 0.0,
                'method_performance': {}
            }
            
            # 매칭 컴포넌트들
            self.tps_grid = None
            self.ransac_params = None
            self.optimizer_config = None
            
            self.logger.debug("✅ 스텝별 특화 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 스텝별 특화 초기화 실패: {e}")
            # 최소한의 기본값 설정
            self.matching_config = {'method': 'similarity', 'quality_threshold': 0.5}
            self.tps_config = {'regularization': 0.1, 'grid_size': 20}
            self.optimization_config = {'learning_rate': 0.01}

# ===============================================================
# 🔄 하위 호환성 지원 (기존 코드 100% 지원)
# ===============================================================

def create_geometric_matching_step(
    device: str = "mps", 
    config: Optional[Dict[str, Any]] = None
) -> GeometricMatchingStep:
    """🔄 기존 방식 100% 호환 생성자"""
    return GeometricMatchingStep(device=device, config=config)

# M3 Max 최적화 전용 생성자도 지원
def create_m3_max_geometric_matching_step(
    device: Optional[str] = None,
    memory_gb: float = 128.0,
    optimization_level: str = "ultra",
    **kwargs
) -> GeometricMatchingStep:
    """🍎 M3 Max 최적화 전용 생성자"""
    return GeometricMatchingStep(
        device=device,
        memory_gb=memory_gb,
        quality_level=optimization_level,
        is_m3_max=True,
        optimization_enabled=True,
        **kwargs
    )