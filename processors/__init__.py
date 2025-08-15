#!/usr/bin/env python3
"""
🔥 MyCloset AI - Processors Package for Cloth Warping
======================================================

🎯 의류 워핑 데이터 처리 프로세서들
✅ 고급 후처리
✅ 고해상도 처리
✅ 품질 향상
✅ 특수 케이스 처리
✅ M3 Max 최적화
"""

# 고급 후처리 프로세서
from .advanced_post_processor import (
    AdvancedPostProcessor,
    AdvancedPostProcessorConfig,
    create_advanced_post_processor
)

# 고해상도 처리 프로세서
from .high_resolution_processor import (
    HighResolutionProcessor,
    HighResolutionProcessorConfig,
    create_high_resolution_processor
)

# 품질 향상 프로세서
from .quality_enhancer import (
    QualityEnhancer,
    QualityEnhancerConfig,
    create_quality_enhancer
)

# 특수 케이스 처리 프로세서
from .special_case_processor import (
    SpecialCaseProcessor,
    SpecialCaseProcessorConfig,
    create_special_case_processor
)

# 기존 프로세서들
from .cloth_warping_preprocessor import (
    ClothWarpingPreprocessor,
    ClothWarpingPreprocessorConfig,
    create_cloth_warping_preprocessor
)

__all__ = [
    # 고급 후처리
    'AdvancedPostProcessor',
    'AdvancedPostProcessorConfig',
    'create_advanced_post_processor',
    
    # 고해상도 처리
    'HighResolutionProcessor',
    'HighResolutionProcessorConfig',
    'create_high_resolution_processor',
    
    # 품질 향상
    'QualityEnhancer',
    'QualityEnhancerConfig',
    'create_quality_enhancer',
    
    # 특수 케이스 처리
    'SpecialCaseProcessor',
    'SpecialCaseProcessorConfig',
    'create_special_case_processor',
    
    # 기존 프로세서
    'ClothWarpingPreprocessor',
    'ClothWarpingPreprocessorConfig',
    'create_cloth_warping_preprocessor'
]

# 프로세서 팩토리 함수
def create_processor(processor_type: str, config: dict = None):
    """
    프로세서 타입에 따른 프로세서 생성
    
    Args:
        processor_type: 프로세서 타입
        config: 설정 딕셔너리
    
    Returns:
        생성된 프로세서 인스턴스
    """
    processor_factories = {
        'advanced_post': create_advanced_post_processor,
        'high_resolution': create_high_resolution_processor,
        'quality_enhancer': create_quality_enhancer,
        'special_case': create_special_case_processor,
        'preprocessor': create_cloth_warping_preprocessor
    }
    
    if processor_type not in processor_factories:
        raise ValueError(f"지원하지 않는 프로세서 타입: {processor_type}")
    
    factory = processor_factories[processor_type]
    
    if config:
        # 설정 딕셔너리를 적절한 설정 클래스로 변환
        if processor_type == 'advanced_post':
            config_obj = AdvancedPostProcessorConfig(**config)
        elif processor_type == 'high_resolution':
            config_obj = HighResolutionProcessorConfig(**config)
        elif processor_type == 'quality_enhancer':
            config_obj = QualityEnhancerConfig(**config)
        elif processor_type == 'special_case':
            config_obj = SpecialCaseProcessorConfig(**config)
        elif processor_type == 'preprocessor':
            config_obj = ClothWarpingPreprocessorConfig(**config)
        else:
            config_obj = None
        
        return factory(config_obj)
    else:
        return factory()

# 프로세서 체인 생성 함수
def create_processor_chain(processor_types: list, configs: list = None):
    """
    여러 프로세서를 연결한 체인 생성
    
    Args:
        processor_types: 프로세서 타입 리스트
        configs: 설정 리스트
    
    Returns:
        프로세서 체인
    """
    processors = []
    
    for i, processor_type in enumerate(processor_types):
        config = configs[i] if configs and i < len(configs) else None
        processor = create_processor(processor_type, config)
        processors.append(processor)
    
    return processors

# 프로세서 정보 조회 함수
def get_processor_info(processor_type: str = None):
    """
    프로세서 정보 조회
    
    Args:
        processor_type: 특정 프로세서 타입 (None이면 모든 정보)
    
    Returns:
        프로세서 정보 딕셔너리
    """
    processor_info = {
        'advanced_post': {
            'name': 'Advanced Post Processor',
            'description': '고급 후처리 프로세서',
            'capabilities': ['엣지 정제', '아티팩트 제거', '텍스처 향상', '품질 향상'],
            'config_class': 'AdvancedPostProcessorConfig'
        },
        'high_resolution': {
            'name': 'High Resolution Processor',
            'description': '고해상도 처리 프로세서',
            'capabilities': ['멀티스케일 처리', '슈퍼해상도', '적응형 처리'],
            'config_class': 'HighResolutionProcessorConfig'
        },
        'quality_enhancer': {
            'name': 'Quality Enhancer',
            'description': '품질 향상 프로세서',
            'capabilities': ['노이즈 제거', '선명도 향상', '텍스처 보존', '색상 향상'],
            'config_class': 'QualityEnhancerConfig'
        },
        'special_case': {
            'name': 'Special Case Processor',
            'description': '특수 케이스 처리 프로세서',
            'capabilities': ['복잡한 패턴 처리', '투명도 처리', '특수 소재 처리'],
            'config_class': 'SpecialCaseProcessorConfig'
        },
        'preprocessor': {
            'name': 'Cloth Warping Preprocessor',
            'description': '의류 워핑 전처리 프로세서',
            'capabilities': ['입력 검증', '전처리', '정규화'],
            'config_class': 'ClothWarpingPreprocessorConfig'
        }
    }
    
    if processor_type:
        return processor_info.get(processor_type, {})
    else:
        return processor_info
