#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Factories
=====================================================================

팩토리 함수들을 분리한 모듈

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def create_cloth_segmentation_step(**kwargs) -> 'ClothSegmentationStep':
    """
    ClothSegmentationStep 인스턴스 생성
    
    Args:
        **kwargs: 초기화 매개변수들
        
    Returns:
        ClothSegmentationStep 인스턴스
    """
    try:
        from ..models.step import ClothSegmentationStep
        return ClothSegmentationStep(**kwargs)
    except ImportError:
        logger.error("ClothSegmentationStep import 실패")
        return None


def create_m3_max_segmentation_step(**kwargs) -> 'ClothSegmentationStep':
    """
    M3 Max 최적화된 ClothSegmentationStep 인스턴스 생성
    
    Args:
        **kwargs: 초기화 매개변수들
        
    Returns:
        ClothSegmentationStep 인스턴스
    """
    try:
        from ..models.step import ClothSegmentationStep
        
        # M3 Max 최적화 설정
        m3_max_kwargs = {
            'device': 'mps' if kwargs.get('device') == 'auto' else kwargs.get('device', 'cpu'),
            'memory_limit': '8GB',
            'optimization_level': 'high'
        }
        m3_max_kwargs.update(kwargs)
        
        return ClothSegmentationStep(**m3_max_kwargs)
    except ImportError:
        logger.error("M3 Max ClothSegmentationStep import 실패")
        return None


def create_cloth_segmentation_step_integrated(**kwargs) -> 'ClothSegmentationStepIntegrated':
    """
    통합된 ClothSegmentationStep 인스턴스 생성
    
    Args:
        **kwargs: 초기화 매개변수들
        
    Returns:
        ClothSegmentationStepIntegrated 인스턴스
    """
    try:
        from ..models.step import ClothSegmentationStep
        
        # 통합 설정
        integrated_kwargs = {
            'enable_ensemble': True,
            'enable_quality_assessment': True,
            'enable_postprocessing': True
        }
        integrated_kwargs.update(kwargs)
        
        return ClothSegmentationStep(**integrated_kwargs)
    except ImportError:
        logger.error("통합 ClothSegmentationStep import 실패")
        return None
