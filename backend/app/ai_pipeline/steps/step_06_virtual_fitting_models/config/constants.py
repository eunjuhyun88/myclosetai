"""
Virtual Fitting 상수 정의
가상 피팅에 필요한 모든 상수와 매직 넘버를 정의합니다.
"""

import math
from typing import Dict, Any

# 이미지 처리 상수
IMAGE_CONSTANTS = {
    "MAX_DIMENSION": 4096,
    "MIN_DIMENSION": 256,
    "DEFAULT_CHANNELS": 3,
    "ALPHA_CHANNEL": 4,
    "GRAYSCALE_CHANNELS": 1,
    "SUPPORTED_FORMATS": [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
    "DEFAULT_DPI": 72,
    "HIGH_DPI": 300
}

# 색상 공간 상수
COLOR_CONSTANTS = {
    "RGB_MAX": 255,
    "RGB_MIN": 0,
    "NORMALIZED_MAX": 1.0,
    "NORMALIZED_MIN": 0.0,
    "ALPHA_MAX": 1.0,
    "ALPHA_MIN": 0.0,
    "GAMMA_CORRECTION": 2.2,
    "SRGB_GAMMA": 2.4
}

# 블렌딩 모드 상수
BLENDING_MODES = {
    "NORMAL": "normal",
    "MULTIPLY": "multiply",
    "SCREEN": "screen",
    "OVERLAY": "overlay",
    "SOFT_LIGHT": "soft_light",
    "HARD_LIGHT": "hard_light",
    "COLOR_DODGE": "color_dodge",
    "COLOR_BURN": "color_burn",
    "DARKEN": "darken",
    "LIGHTEN": "lighten",
    "DIFFERENCE": "difference",
    "EXCLUSION": "exclusion"
}

# 워핑 상수
WARPING_CONSTANTS = {
    "MIN_WARP_STRENGTH": 0.1,
    "MAX_WARP_STRENGTH": 2.0,
    "DEFAULT_WARP_STRENGTH": 1.0,
    "WARP_ITERATIONS": 3,
    "WARP_TOLERANCE": 1e-6,
    "MAX_WARP_DISTANCE": 100.0,
    "MIN_WARP_DISTANCE": 0.1
}

# 품질 평가 상수
QUALITY_CONSTANTS = {
    "SSIM_WINDOW_SIZE": 11,
    "SSIM_GAUSSIAN_SIGMA": 1.5,
    "LPIPS_NETWORK": "alex",
    "FID_FEATURE_DIM": 2048,
    "PSNR_MAX_VALUE": 255.0,
    "PSNR_MIN_VALUE": 0.0,
    "PSNR_REFERENCE": 255.0
}

# 신경망 상수
NEURAL_NETWORK_CONSTANTS = {
    "DEFAULT_BATCH_SIZE": 1,
    "MAX_BATCH_SIZE": 8,
    "MIN_BATCH_SIZE": 1,
    "DEFAULT_LEARNING_RATE": 1e-4,
    "MIN_LEARNING_RATE": 1e-6,
    "MAX_LEARNING_RATE": 1e-2,
    "DEFAULT_EPOCHS": 100,
    "MIN_EPOCHS": 10,
    "MAX_EPOCHS": 1000,
    "DEFAULT_PATIENCE": 10,
    "MIN_PATIENCE": 5,
    "MAX_PATIENCE": 50
}

# 확산 모델 상수
DIFFUSION_CONSTANTS = {
    "DEFAULT_STEPS": 50,
    "MIN_STEPS": 10,
    "MAX_STEPS": 1000,
    "DEFAULT_GUIDANCE_SCALE": 7.5,
    "MIN_GUIDANCE_SCALE": 1.0,
    "MAX_GUIDANCE_SCALE": 20.0,
    "DEFAULT_ETA": 0.0,
    "MIN_ETA": -1.0,
    "MAX_ETA": 1.0,
    "DEFAULT_SEED": 42,
    "MIN_SEED": 0,
    "MAX_SEED": 2**32 - 1
}

# HR-VITON 모델 상수
HR_VITON_CONSTANTS = {
    "INPUT_HEIGHT": 1024,
    "INPUT_WIDTH": 768,
    "OUTPUT_HEIGHT": 1024,
    "OUTPUT_WIDTH": 768,
    "FEATURE_CHANNELS": 256,
    "ATTENTION_HEADS": 8,
    "TRANSFORMER_LAYERS": 6,
    "DROPOUT_RATE": 0.1,
    "LAYER_NORM_EPS": 1e-6,
    "POSITIONAL_ENCODING_DIM": 512
}

# OOTD 모델 상수
OOTD_CONSTANTS = {
    "INPUT_HEIGHT": 1024,
    "INPUT_WIDTH": 1024,
    "OUTPUT_HEIGHT": 1024,
    "OUTPUT_WIDTH": 1024,
    "LATENT_DIM": 4,
    "LATENT_HEIGHT": 128,
    "LATENT_WIDTH": 128,
    "UNET_BLOCK_OUT_CHANNELS": [320, 640, 1280, 1280],
    "UNET_ATTENTION_HEADS": [5, 10, 20, 20],
    "UNET_CROSS_ATTENTION_DIM": 1024,
    "UNET_ADD_EMBEDDING_PROJECTION_DIM": 1280
}

# VITON-HD 모델 상수
VITON_HD_CONSTANTS = {
    "INPUT_HEIGHT": 1024,
    "INPUT_WIDTH": 768,
    "OUTPUT_HEIGHT": 1024,
    "OUTPUT_WIDTH": 768,
    "FEATURE_CHANNELS": 128,
    "ATTENTION_HEADS": 4,
    "TRANSFORMER_LAYERS": 4,
    "DROPOUT_RATE": 0.1,
    "LAYER_NORM_EPS": 1e-6
}

# 메모리 최적화 상수
MEMORY_CONSTANTS = {
    "MAX_MEMORY_USAGE": 0.8,  # 80%
    "MIN_MEMORY_USAGE": 0.1,  # 10%
    "DEFAULT_MEMORY_USAGE": 0.6,  # 60%
    "GRADIENT_CHECKPOINTING": True,
    "MIXED_PRECISION": True,
    "MODEL_SHARDING": False,
    "ATTENTION_OPTIMIZATION": True
}

# 성능 최적화 상수
PERFORMANCE_CONSTANTS = {
    "DEFAULT_NUM_WORKERS": 4,
    "MIN_NUM_WORKERS": 1,
    "MAX_NUM_WORKERS": 16,
    "DEFAULT_PREFETCH_FACTOR": 2,
    "MIN_PREFETCH_FACTOR": 1,
    "MAX_PREFETCH_FACTOR": 4,
    "PIN_MEMORY": True,
    "NON_BLOCKING": True
}

# 오류 처리 상수
ERROR_CONSTANTS = {
    "MAX_RETRY_ATTEMPTS": 3,
    "RETRY_DELAY": 1.0,  # seconds
    "MAX_RETRY_DELAY": 10.0,  # seconds
    "BACKOFF_MULTIPLIER": 2.0,
    "TIMEOUT_SECONDS": 300,  # 5 minutes
    "MAX_TIMEOUT_SECONDS": 1800,  # 30 minutes
    "MIN_TIMEOUT_SECONDS": 60  # 1 minute
}

# 로깅 상수
LOGGING_CONSTANTS = {
    "DEFAULT_LOG_LEVEL": "INFO",
    "MIN_LOG_LEVEL": "DEBUG",
    "MAX_LOG_LEVEL": "CRITICAL",
    "LOG_FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "LOG_DATE_FORMAT": "%Y-%m-%d %H:%M:%S",
    "MAX_LOG_FILE_SIZE": 10 * 1024 * 1024,  # 10MB
    "MAX_LOG_FILES": 5
}

# 파일 경로 상수
PATH_CONSTANTS = {
    "TEMP_DIR": "temp",
    "CACHE_DIR": "cache",
    "LOG_DIR": "logs",
    "MODEL_DIR": "models",
    "CHECKPOINT_DIR": "checkpoints",
    "OUTPUT_DIR": "output",
    "DEBUG_DIR": "debug"
}

# 수학 상수
MATH_CONSTANTS = {
    "PI": math.pi,
    "E": math.e,
    "GOLDEN_RATIO": 1.618033988749895,
    "SQRT_2": math.sqrt(2),
    "SQRT_3": math.sqrt(3),
    "LN_2": math.log(2),
    "LN_10": math.log(10)
}

# 모든 상수를 하나의 딕셔너리로 통합
ALL_CONSTANTS = {
    "IMAGE": IMAGE_CONSTANTS,
    "COLOR": COLOR_CONSTANTS,
    "BLENDING": BLENDING_MODES,
    "WARPING": WARPING_CONSTANTS,
    "QUALITY": QUALITY_CONSTANTS,
    "NEURAL_NETWORK": NEURAL_NETWORK_CONSTANTS,
    "DIFFUSION": DIFFUSION_CONSTANTS,
    "HR_VITON": HR_VITON_CONSTANTS,
    "OOTD": OOTD_CONSTANTS,
    "VITON_HD": VITON_HD_CONSTANTS,
    "MEMORY": MEMORY_CONSTANTS,
    "PERFORMANCE": PERFORMANCE_CONSTANTS,
    "ERROR": ERROR_CONSTANTS,
    "LOGGING": LOGGING_CONSTANTS,
    "PATH": PATH_CONSTANTS,
    "MATH": MATH_CONSTANTS
}

def get_constant(category: str, key: str, default: Any = None) -> Any:
    """특정 카테고리의 상수를 가져옵니다."""
    return ALL_CONSTANTS.get(category, {}).get(key, default)

def get_constants_by_category(category: str) -> Dict[str, Any]:
    """특정 카테고리의 모든 상수를 가져옵니다."""
    return ALL_CONSTANTS.get(category, {})

def validate_constant(category: str, key: str, value: Any) -> bool:
    """상수 값의 유효성을 검증합니다."""
    if category not in ALL_CONSTANTS:
        return False
    
    if key not in ALL_CONSTANTS[category]:
        return False
    
    expected_type = type(ALL_CONSTANTS[category][key])
    return isinstance(value, expected_type)
