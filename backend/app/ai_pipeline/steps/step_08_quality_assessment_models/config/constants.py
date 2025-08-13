#!/usr/bin/env python3
"""
Quality Assessment Constants - 품질 평가를 위한 상수 정의
"""

# 품질 평가 임계값
QUALITY_THRESHOLDS = {
    'excellent': 0.9,
    'good': 0.7,
    'fair': 0.5,
    'poor': 0.3,
    'very_poor': 0.1
}

# 이미지 품질 지표 가중치
QUALITY_WEIGHTS = {
    'sharpness': 0.25,
    'contrast': 0.20,
    'brightness': 0.15,
    'noise_level': 0.20,
    'color_balance': 0.15,
    'composition': 0.05
}

# 이미지 처리 상수
DEFAULT_IMAGE_SIZE = (512, 512)
MAX_IMAGE_SIZE = (4096, 4096)
MIN_IMAGE_SIZE = (64, 64)

# 품질 평가 모델 상수
DEFAULT_BATCH_SIZE = 1
MAX_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 0.001

# 파일 포맷 상수
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

# 에러 메시지 상수
ERROR_MESSAGES = {
    'invalid_image': '유효하지 않은 이미지입니다',
    'unsupported_format': '지원하지 않는 이미지 포맷입니다',
    'model_loading_failed': '품질 평가 모델 로딩에 실패했습니다',
    'assessment_failed': '품질 평가에 실패했습니다'
}

# 성공 메시지 상수
SUCCESS_MESSAGES = {
    'model_loaded': '품질 평가 모델 로딩 완료',
    'assessment_complete': '품질 평가 완료',
    'quality_analyzed': '이미지 품질 분석 완료'
}

# 로깅 상수
LOG_LEVELS = {
    'debug': 10,
    'info': 20,
    'warning': 30,
    'error': 40,
    'critical': 50
}

# 성능 상수
MEMORY_LIMIT_MB = 4096  # 4GB
PROCESSING_TIMEOUT_SECONDS = 180  # 3분
CACHE_SIZE_LIMIT = 500  # 캐시된 이미지 수 제한

# 품질 평가 알고리즘 상수
SHARPNESS_KERNEL_SIZE = 3
CONTRAST_HISTOGRAM_BINS = 256
NOISE_ESTIMATION_WINDOW = 8
COLOR_BALANCE_TARGET = 128
