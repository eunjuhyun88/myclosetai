#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Config
=====================================================================

설정 및 타입 정의들을 분리한 모듈

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union

class SegmentationMethod(Enum):
    """세그멘테이션 방법"""
    U2NET_CLOTH = "u2net_cloth"         # U2Net 의류 특화 (168.1MB) - 우선순위 1 (M3 Max 안전)
    SAM_HUGE = "sam_huge"               # SAM ViT-Huge (2445.7MB) - 우선순위 2 (메모리 여유시)
    DEEPLABV3_PLUS = "deeplabv3_plus"   # DeepLabV3+ (233.3MB) - 우선순위 3 (나중에)
    MASK_RCNN = "mask_rcnn"             # Mask R-CNN (폴백)
    HYBRID_AI = "hybrid_ai"             # 하이브리드 앙상블

class ClothCategory(Enum):
    """의류 카테고리 (다중 클래스)"""
    BACKGROUND = 0
    SHIRT = 1           # 셔츠/블라우스
    T_SHIRT = 2         # 티셔츠
    SWEATER = 3         # 스웨터/니트
    HOODIE = 4          # 후드티
    JACKET = 5          # 재킷/아우터
    COAT = 6            # 코트
    DRESS = 7           # 원피스
    SKIRT = 8           # 스커트
    PANTS = 9           # 바지
    JEANS = 10          # 청바지
    SHORTS = 11         # 반바지
    SHOES = 12          # 신발
    BOOTS = 13          # 부츠
    SNEAKERS = 14       # 운동화
    BAG = 15            # 가방
    HAT = 16            # 모자
    GLASSES = 17        # 안경
    SCARF = 18          # 스카프
    BELT = 19           # 벨트
    ACCESSORY = 20      # 액세서리

class QualityLevel(Enum):
    """품질 레벨"""
    FAST = "fast"           # 빠른 처리
    BALANCED = "balanced"   # 균형
    HIGH = "high"          # 고품질
    ULTRA = "ultra"        # 최고품질

@dataclass
class ClothSegmentationConfig:
    """의류 세그멘테이션 설정"""
    method: SegmentationMethod = SegmentationMethod.U2NET_CLOTH  # M3 Max 안전 모드
    quality_level: QualityLevel = QualityLevel.HIGH
    input_size: Tuple[int, int] = (512, 512)
    
    # 전처리 설정
    enable_quality_assessment: bool = True
    enable_lighting_normalization: bool = True
    enable_color_correction: bool = True
    
    # 의류 분류 설정
    enable_clothing_classification: bool = True
    classification_confidence_threshold: float = 0.8
    
    # 후처리 설정
    enable_crf_postprocessing: bool = True  # 🔥 CRF 후처리 복원
    enable_edge_refinement: bool = True
    enable_hole_filling: bool = True
    enable_multiscale_processing: bool = True  # 🔥 멀티스케일 처리 복원
    
    # 품질 검증 설정
    enable_quality_validation: bool = True
    quality_threshold: float = 0.7
    enable_auto_retry: bool = True
    max_retry_attempts: int = 3
    
    # 기본 설정
    confidence_threshold: float = 0.5
    enable_visualization: bool = True
    
    # 자동 전처리 설정
    auto_preprocessing: bool = True
    
    # 자동 후처리 설정
    auto_postprocessing: bool = True
    
    # 데이터 검증 설정
    strict_data_validation: bool = True

# 의류 카테고리 매핑
CLOTH_CATEGORIES = {
    0: 'background',
    1: 'shirt', 2: 't_shirt', 3: 'sweater', 4: 'hoodie',
    5: 'jacket', 6: 'coat', 7: 'dress', 8: 'skirt',
    9: 'pants', 10: 'jeans', 11: 'shorts',
    12: 'shoes', 13: 'boots', 14: 'sneakers',
    15: 'bag', 16: 'hat', 17: 'glasses', 18: 'scarf', 19: 'belt',
    20: 'accessory'
}

# 의류 카테고리 그룹
CLOTH_CATEGORY_GROUPS = {
    'upper_body': [1, 2, 3, 4, 5, 6],  # 상의
    'lower_body': [9, 10, 11],          # 하의
    'full_body': [7],                   # 전신
    'accessories': [15, 16, 17, 18, 19, 20],  # 액세서리
    'footwear': [12, 13, 14]            # 신발
}

# 품질 레벨별 설정
QUALITY_LEVEL_CONFIGS = {
    QualityLevel.FAST: {
        'input_size': (256, 256),
        'enable_multiscale_processing': False,
        'enable_crf_postprocessing': False,
        'max_retry_attempts': 1
    },
    QualityLevel.BALANCED: {
        'input_size': (384, 384),
        'enable_multiscale_processing': True,
        'enable_crf_postprocessing': True,
        'max_retry_attempts': 2
    },
    QualityLevel.HIGH: {
        'input_size': (512, 512),
        'enable_multiscale_processing': True,
        'enable_crf_postprocessing': True,
        'max_retry_attempts': 3
    },
    QualityLevel.ULTRA: {
        'input_size': (1024, 1024),
        'enable_multiscale_processing': True,
        'enable_crf_postprocessing': True,
        'max_retry_attempts': 5
    }
}

# 모델별 설정
MODEL_CONFIGS = {
    SegmentationMethod.U2NET_CLOTH: {
        'model_size': '168.1MB',
        'priority': 1,
        'memory_required': '2GB',
        'device': 'cpu',
        'input_size': (512, 512)
    },
    SegmentationMethod.SAM_HUGE: {
        'model_size': '2445.7MB',
        'priority': 2,
        'memory_required': '8GB',
        'device': 'cpu',
        'input_size': (1024, 1024)
    },
    SegmentationMethod.DEEPLABV3_PLUS: {
        'model_size': '233.3MB',
        'priority': 3,
        'memory_required': '4GB',
        'device': 'cpu',
        'input_size': (512, 512)
    },
    SegmentationMethod.MASK_RCNN: {
        'model_size': '500MB',
        'priority': 4,
        'memory_required': '6GB',
        'device': 'cpu',
        'input_size': (800, 800)
    }
}

def get_quality_config(quality_level: QualityLevel) -> Dict[str, Any]:
    """품질 레벨별 설정 조회"""
    return QUALITY_LEVEL_CONFIGS.get(quality_level, QUALITY_LEVEL_CONFIGS[QualityLevel.BALANCED])

def get_model_config(method: SegmentationMethod) -> Dict[str, Any]:
    """모델별 설정 조회"""
    return MODEL_CONFIGS.get(method, MODEL_CONFIGS[SegmentationMethod.U2NET_CLOTH])

def get_cloth_category_name(category_id: int) -> str:
    """의류 카테고리 ID로 이름 조회"""
    return CLOTH_CATEGORIES.get(category_id, 'unknown')

def get_cloth_category_group(category_id: int) -> str:
    """의류 카테고리 ID로 그룹 조회"""
    for group_name, category_ids in CLOTH_CATEGORY_GROUPS.items():
        if category_id in category_ids:
            return group_name
    return 'other'
