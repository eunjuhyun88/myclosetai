#!/usr/bin/env python3
"""
Cloth Warping Constants - 의류 변형을 위한 상수 정의
"""

# 이미지 처리 상수
DEFAULT_IMAGE_SIZE = (512, 512)
MAX_IMAGE_SIZE = (2048, 2048)
MIN_IMAGE_SIZE = (64, 64)

# 품질 설정 상수
QUALITY_LEVELS = {
    'low': {'resolution': 256, 'iterations': 100},
    'medium': {'resolution': 512, 'iterations': 200},
    'high': {'resolution': 1024, 'iterations': 500},
    'ultra': {'resolution': 2048, 'iterations': 1000}
}

# 변형 파라미터 상수
MAX_SCALE = 3.0
MIN_SCALE = 0.1
MAX_ROTATION = 180.0  # 도
MAX_TRANSLATION = 100.0  # 픽셀
MAX_SHEAR = 45.0  # 도

# 모델 상수
DEFAULT_BATCH_SIZE = 1
MAX_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 0.001

# 파일 포맷 상수
SUPPORTED_INPUT_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
SUPPORTED_OUTPUT_FORMATS = ['.png', '.jpg', '.tiff']

# 에러 메시지 상수
ERROR_MESSAGES = {
    'invalid_image_size': '이미지 크기가 지원 범위를 벗어났습니다',
    'unsupported_format': '지원하지 않는 이미지 포맷입니다',
    'model_loading_failed': '모델 로딩에 실패했습니다',
    'processing_failed': '이미지 처리에 실패했습니다'
}

# 성공 메시지 상수
SUCCESS_MESSAGES = {
    'model_loaded': '모델 로딩 완료',
    'processing_complete': '이미지 처리 완료',
    'warping_successful': '의류 변형 성공'
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
MEMORY_LIMIT_MB = 8192  # 8GB
PROCESSING_TIMEOUT_SECONDS = 300  # 5분
CACHE_SIZE_LIMIT = 1000  # 캐시된 이미지 수 제한
