#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Integrated Step
=====================================================================

기존 step.py의 모든 기능을 import해서 사용하는 통합 파일
기존 step.py는 수정하지 않고, 여기서 모든 기능을 통합하여 사용

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import time
from typing import Dict, Any, List, Tuple, Optional

# 기존 step.py에서 모든 기능을 import
from .step import (
    # 기본 클래스들
    BaseStepMixin, ClothSegmentationStep,
    
    # Enum과 Config
    SegmentationMethod, ClothCategory, QualityLevel, ClothSegmentationConfig,
    
    # AI 모델들
    ASPPModule, SelfCorrectionModule, MultiHeadSelfAttention, PositionalEncoding2D,
    DeepLabV3PlusBackbone, DeepLabV3PlusDecoder, DeepLabV3PlusModel,
    RealDeepLabV3PlusModel, RealU2NETModel, RealSAMModel,
    
    # 앙상블 기능들
    _run_hybrid_ensemble_sync, _combine_ensemble_results,
    _calculate_adaptive_threshold, _apply_ensemble_postprocessing,
    
    # 유틸리티 함수들
    _get_central_hub_container, _inject_dependencies_safe, _get_service_from_central_hub,
    
    # 팩토리 함수들
    create_cloth_segmentation_step, create_m3_max_segmentation_step
)

# 새로 분리된 모듈들 import
from .postprocessing.quality_enhancement import (
    _fill_holes_and_remove_noise_advanced,
    _evaluate_segmentation_quality,
    _create_segmentation_visualizations,
    _assess_image_quality,
    _normalize_lighting,
    _correct_colors
)

from .utils.feature_extraction import (
    _extract_cloth_features,
    _calculate_centroid,
    _calculate_bounding_box,
    _extract_cloth_contours,
    _get_cloth_bounding_boxes,
    _get_cloth_centroids,
    _get_cloth_areas,
    _get_cloth_contours_dict,
    _detect_cloth_categories
)

logger = logging.getLogger(__name__)

class ClothSegmentationStepIntegrated(ClothSegmentationStep):
    """
    통합된 의류 세그멘테이션 스텝 클래스
    기존 step.py의 모든 기능을 상속받고, 새로 분리된 모듈들의 기능도 통합
    """
    
    def __init__(self, **kwargs):
        # 기존 클래스 초기화
        super().__init__(**kwargs)
        
        # 새로 분리된 모듈들의 기능들을 클래스 메서드로 바인딩
        self._fill_holes_and_remove_noise_advanced = _fill_holes_and_remove_noise_advanced.__get__(self, self.__class__)
        self._evaluate_segmentation_quality = _evaluate_segmentation_quality.__get__(self, self.__class__)
        self._create_segmentation_visualizations = _create_segmentation_visualizations.__get__(self, self.__class__)
        self._assess_image_quality = _assess_image_quality.__get__(self, self.__class__)
        self._normalize_lighting = _normalize_lighting.__get__(self, self.__class__)
        self._correct_colors = _correct_colors.__get__(self, self.__class__)
        
        self._extract_cloth_features = _extract_cloth_features.__get__(self, self.__class__)
        self._calculate_centroid = _calculate_centroid.__get__(self, self.__class__)
        self._calculate_bounding_box = _calculate_bounding_box.__get__(self, self.__class__)
        self._extract_cloth_contours = _extract_cloth_contours.__get__(self, self.__class__)
        self._get_cloth_bounding_boxes = _get_cloth_bounding_boxes.__get__(self, self.__class__)
        self._get_cloth_centroids = _get_cloth_centroids.__get__(self, self.__class__)
        self._get_cloth_areas = _get_cloth_areas.__get__(self, self.__class__)
        self._get_cloth_contours_dict = _get_cloth_contours_dict.__get__(self, self.__class__)
        self._detect_cloth_categories = _detect_cloth_categories.__get__(self, self.__class__)
        
        logger.info("🔥 ClothSegmentationStepIntegrated 초기화 완료")
    
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        통합된 처리 메서드
        기존 기능 + 새로 분리된 모듈들의 기능을 모두 사용
        """
        try:
            # 기존 process 메서드 호출
            result = super().process(**kwargs)
            
            # 새로 분리된 모듈들의 기능들을 추가로 적용
            if 'masks' in result and result['masks']:
                # 품질 향상 후처리 적용
                enhanced_masks = self._fill_holes_and_remove_noise_advanced(result['masks'])
                result['masks'] = enhanced_masks
                
                # 품질 평가 추가
                if 'image' in kwargs:
                    quality_metrics = self._evaluate_segmentation_quality(enhanced_masks, kwargs['image'])
                    result['quality_metrics'] = quality_metrics
                
                # 특성 추출 추가
                if 'image' in kwargs:
                    features = self._extract_cloth_features(enhanced_masks, kwargs['image'])
                    result['features'] = features
                
                # 시각화 추가
                if 'image' in kwargs:
                    visualizations = self._create_segmentation_visualizations(kwargs['image'], enhanced_masks)
                    result['visualizations'] = visualizations
            
            return result
            
        except Exception as e:
            logger.error(f"통합 처리 중 오류 발생: {e}")
            return self._create_error_response(str(e))
    
    def _enhanced_postprocessing(self, masks: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
        """
        향상된 후처리 메서드
        새로 분리된 모듈들의 기능들을 모두 적용
        """
        try:
            # 기존 후처리
            processed_masks = self._postprocess_masks(masks)
            
            # 새로 분리된 모듈들의 기능들 적용
            # 1. 품질 향상
            enhanced_masks = self._fill_holes_and_remove_noise_advanced(processed_masks)
            
            # 2. 품질 평가
            quality_metrics = self._evaluate_segmentation_quality(enhanced_masks, image)
            
            # 3. 특성 추출
            features = self._extract_cloth_features(enhanced_masks, image)
            
            # 4. 시각화
            visualizations = self._create_segmentation_visualizations(image, enhanced_masks)
            
            return {
                'masks': enhanced_masks,
                'quality_metrics': quality_metrics,
                'features': features,
                'visualizations': visualizations
            }
            
        except Exception as e:
            logger.error(f"향상된 후처리 중 오류 발생: {e}")
            return {'masks': masks}

def create_cloth_segmentation_step_integrated(**kwargs) -> ClothSegmentationStepIntegrated:
    """
    통합된 의류 세그멘테이션 스텝 생성 팩토리 함수
    """
    try:
        step = ClothSegmentationStepIntegrated(**kwargs)
        logger.info("🔥 통합된 의류 세그멘테이션 스텝 생성 완료")
        return step
    except Exception as e:
        logger.error(f"통합된 의류 세그멘테이션 스텝 생성 실패: {e}")
        raise

def create_m3_max_segmentation_step_integrated(**kwargs) -> ClothSegmentationStepIntegrated:
    """
    M3 Max용 통합된 의류 세그멘테이션 스텝 생성 팩토리 함수
    """
    try:
        # M3 Max 최적화 설정 추가
        m3_max_kwargs = {
            'device': 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu',
            'memory_efficient': True,
            'batch_size': 1,
            **kwargs
        }
        
        step = ClothSegmentationStepIntegrated(**m3_max_kwargs)
        logger.info("🍎 M3 Max용 통합된 의류 세그멘테이션 스텝 생성 완료")
        return step
    except Exception as e:
        logger.error(f"M3 Max용 통합된 의류 세그멘테이션 스텝 생성 실패: {e}")
        raise
