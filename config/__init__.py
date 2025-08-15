#!/usr/bin/env python3
"""
🔥 MyCloset AI - Configuration Package for Cloth Warping
========================================================

🎯 의류 워핑 설정 관리
✅ 기본 설정
✅ 상수 정의
✅ 타입 정의
✅ 워핑 전용 설정
✅ M3 Max 최적화
"""

# 기본 설정
from .config import (
    ClothWarpingConfig as BaseConfig,
    create_config,
    load_config_from_file,
    save_config_to_file
)

# 상수 정의
from .constants import (
    SUPPORTED_MODELS,
    ENSEMBLE_METHODS,
    QUALITY_METRICS,
    PROCESSING_MODES,
    DEVICE_TYPES,
    DEFAULT_VALUES
)

# 타입 정의
from .types import (
    ModelType,
    EnsembleMethod,
    QualityMetric,
    ProcessingMode,
    DeviceType,
    ConfigDict
)

# 워핑 전용 설정
from .warping_config import (
    ClothWarpingConfig,
    TPSWarpingConfig,
    GeometricFlowConfig,
    NeuralWarpingConfig,
    ClothDeformationConfig,
    QualityEnhancementConfig,
    HighResolutionConfig,
    ProcessingConfig,
    DeviceConfig,
    create_default_warping_config,
    create_warping_config
)

__all__ = [
    # 기본 설정
    'BaseConfig',
    'create_config',
    'load_config_from_file',
    'save_config_to_file',
    
    # 상수
    'SUPPORTED_MODELS',
    'ENSEMBLE_METHODS',
    'QUALITY_METRICS',
    'PROCESSING_MODES',
    'DEVICE_TYPES',
    'DEFAULT_VALUES',
    
    # 타입
    'ModelType',
    'EnsembleMethod',
    'QualityMetric',
    'ProcessingMode',
    'DeviceType',
    'ConfigDict',
    
    # 워핑 설정
    'ClothWarpingConfig',
    'TPSWarpingConfig',
    'GeometricFlowConfig',
    'NeuralWarpingConfig',
    'ClothDeformationConfig',
    'QualityEnhancementConfig',
    'HighResolutionConfig',
    'ProcessingConfig',
    'DeviceConfig',
    'create_default_warping_config',
    'create_warping_config'
]

# 설정 팩토리 함수
def create_config_factory(config_type: str = "default", **kwargs):
    """
    설정 타입에 따른 설정 객체 생성
    
    Args:
        config_type: 설정 타입 (default, high_quality, fast, memory_efficient, custom)
        **kwargs: 추가 설정
    
    Returns:
        설정 객체
    """
    if config_type == "default":
        return create_default_warping_config()
    elif config_type == "high_quality":
        return create_warping_config("high_quality", **kwargs)
    elif config_type == "fast":
        return create_warping_config("fast", **kwargs)
    elif config_type == "memory_efficient":
        return create_warping_config("memory_efficient", **kwargs)
    elif config_type == "custom":
        # 사용자 정의 설정
        config = create_default_warping_config()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    else:
        raise ValueError(f"지원하지 않는 설정 타입: {config_type}")

# 설정 검증 함수
def validate_config(config: ClothWarpingConfig) -> bool:
    """
    설정 객체 검증
    
    Args:
        config: 검증할 설정 객체
    
    Returns:
        검증 결과 (True: 유효, False: 무효)
    """
    try:
        # 기본 검증
        if not hasattr(config, 'model_name') or not config.model_name:
            return False
        
        if not hasattr(config, 'input_size') or not config.input_size:
            return False
        
        if not hasattr(config, 'output_size') or not config.output_size:
            return False
        
        # 워핑 알고리즘 검증
        if not hasattr(config, 'tps_warping'):
            return False
        
        if not hasattr(config, 'geometric_flow'):
            return False
        
        if not hasattr(config, 'neural_warping'):
            return False
        
        if not hasattr(config, 'cloth_deformation'):
            return False
        
        # 품질 향상 검증
        if not hasattr(config, 'quality_enhancement'):
            return False
        
        # 고해상도 처리 검증
        if not hasattr(config, 'high_resolution'):
            return False
        
        # 처리 설정 검증
        if not hasattr(config, 'processing'):
            return False
        
        # 디바이스 설정 검증
        if not hasattr(config, 'device'):
            return False
        
        return True
        
    except Exception:
        return False

# 설정 병합 함수
def merge_configs(base_config: ClothWarpingConfig, 
                  override_config: dict) -> ClothWarpingConfig:
    """
    기본 설정과 오버라이드 설정을 병합
    
    Args:
        base_config: 기본 설정 객체
        override_config: 오버라이드할 설정 딕셔너리
    
    Returns:
        병합된 설정 객체
    """
    try:
        # 설정 복사
        merged_config = ClothWarpingConfig()
        
        # 기본 설정 복사
        for key, value in base_config.__dict__.items():
            if hasattr(merged_config, key):
                setattr(merged_config, key, value)
        
        # 오버라이드 설정 적용
        for key, value in override_config.items():
            if hasattr(merged_config, key):
                if isinstance(value, dict) and hasattr(getattr(merged_config, key), '__dict__'):
                    # 중첩된 설정 객체 업데이트
                    for sub_key, sub_value in value.items():
                        if hasattr(getattr(merged_config, key), sub_key):
                            setattr(getattr(merged_config, key), sub_key, sub_value)
                else:
                    setattr(merged_config, key, value)
        
        # 설정 검증 및 디바이스 설정
        merged_config._validate_config()
        merged_config._setup_device()
        merged_config._setup_logging()
        
        return merged_config
        
    except Exception as e:
        raise RuntimeError(f"설정 병합 실패: {e}")

# 설정 정보 조회 함수
def get_config_info(config_type: str = None):
    """
    설정 정보 조회
    
    Args:
        config_type: 특정 설정 타입 (None이면 모든 정보)
    
    Returns:
        설정 정보 딕셔너리
    """
    config_info = {
        'default': {
            'name': 'Default Configuration',
            'description': '기본 워핑 설정',
            'features': ['균형잡힌 성능', '표준 품질', '적당한 메모리 사용량']
        },
        'high_quality': {
            'name': 'High Quality Configuration',
            'description': '고품질 워핑 설정',
            'features': ['최고 품질', '세밀한 처리', '높은 메모리 사용량']
        },
        'fast': {
            'name': 'Fast Configuration',
            'description': '고속 워핑 설정',
            'features': ['빠른 처리', '기본 품질', '낮은 메모리 사용량']
        },
        'memory_efficient': {
            'name': 'Memory Efficient Configuration',
            'description': '메모리 효율적 워핑 설정',
            'features': ['메모리 절약', '적당한 품질', '최적화된 처리']
        }
    }
    
    if config_type:
        return config_info.get(config_type, {})
    else:
        return config_info

# 설정 템플릿 생성 함수
def create_config_template(template_type: str = "basic") -> dict:
    """
    설정 템플릿 생성
    
    Args:
        template_type: 템플릿 타입 (basic, advanced, minimal)
    
    Returns:
        설정 템플릿 딕셔너리
    """
    if template_type == "basic":
        return {
            "model_name": "cloth_warping_model",
            "version": "1.0",
            "input_size": [256, 256],
            "output_size": [256, 256],
            "enable_ensemble": True,
            "device_type": "auto"
        }
    elif template_type == "advanced":
        return {
            "model_name": "advanced_cloth_warping",
            "version": "2.0",
            "input_size": [512, 512],
            "output_size": [1024, 1024],
            "enable_ensemble": True,
            "quality_enhancement": {
                "enable_edge_refinement": True,
                "enable_artifact_removal": True,
                "refinement_iterations": 5
            },
            "high_resolution": {
                "enable_super_resolution": True,
                "target_resolutions": [[512, 512], [1024, 1024], [2048, 2048]]
            }
        }
    elif template_type == "minimal":
        return {
            "model_name": "minimal_cloth_warping",
            "version": "1.0",
            "input_size": [128, 128],
            "output_size": [256, 256],
            "enable_ensemble": False,
            "quality_enhancement": {
                "enable_edge_refinement": False,
                "enable_artifact_removal": False
            }
        }
    else:
        raise ValueError(f"지원하지 않는 템플릿 타입: {template_type}")

if __name__ == "__main__":
    # 테스트 코드
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 설정 팩토리 테스트
    config = create_config_factory("high_quality")
    print("✅ 고품질 설정 생성 완료")
    
    # 설정 검증 테스트
    is_valid = validate_config(config)
    print(f"✅ 설정 검증 결과: {is_valid}")
    
    # 설정 정보 조회 테스트
    config_info = get_config_info("high_quality")
    print(f"✅ 설정 정보: {config_info}")
    
    # 설정 템플릿 테스트
    template = create_config_template("advanced")
    print(f"✅ 고급 템플릿 생성 완료: {template}")
