#!/usr/bin/env python3
"""
🔥 Final Output 상수 파일
================================================================================

✅ 시스템 상수
✅ 메트릭 기준
✅ 품질 임계값
✅ 출력 상수

Author: MyCloset AI Team
Date: 2025-08-13
Version: 1.0
"""

# ==============================================
# 🔥 시스템 상수
# ==============================================

# 모델 상수
DEFAULT_D_MODEL = 512
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FF = 2048
DEFAULT_DROPOUT = 0.1

# 출력 상수
DEFAULT_OUTPUT_SIZE = 64
DEFAULT_NUM_CHANNELS = 3
DEFAULT_BATCH_SIZE = 1

# ==============================================
# 🔥 품질 임계값
# ==============================================

# 품질 점수 임계값
QUALITY_THRESHOLDS = {
    'excellent': 0.9,
    'good': 0.8,
    'acceptable': 0.7,
    'poor': 0.6,
    'unacceptable': 0.5
}

# 신뢰도 임계값
CONFIDENCE_THRESHOLDS = {
    'high': 0.8,
    'medium': 0.6,
    'low': 0.4
}

# ==============================================
# 🔥 메트릭 기준
# ==============================================

# PSNR 기준 (dB)
PSNR_THRESHOLDS = {
    'excellent': 40.0,
    'good': 35.0,
    'acceptable': 30.0,
    'poor': 25.0
}

# SSIM 기준
SSIM_THRESHOLDS = {
    'excellent': 0.95,
    'good': 0.90,
    'acceptable': 0.85,
    'poor': 0.80
}

# LPIPS 기준 (낮을수록 좋음)
LPIPS_THRESHOLDS = {
    'excellent': 0.05,
    'good': 0.10,
    'acceptable': 0.15,
    'poor': 0.20
}

# ==============================================
# 🔥 출력 상수
# ==============================================

# 해상도 옵션
RESOLUTION_OPTIONS = {
    'low': (256, 256),
    'medium': (512, 512),
    'high': (1024, 1024),
    'ultra': (2048, 2048)
}

# 압축 품질 옵션
COMPRESSION_QUALITY_OPTIONS = {
    'low': 70,
    'medium': 85,
    'high': 95,
    'ultra': 100
}

# ==============================================
# 🔥 성능 상수
# ==============================================

# 메모리 사용량 제한
MEMORY_LIMITS = {
    'low': "4GB",
    'medium': "8GB",
    'high': "16GB",
    'ultra': "32GB"
}

# 배치 크기 옵션
BATCH_SIZE_OPTIONS = {
    'low': 1,
    'medium': 2,
    'high': 4,
    'ultra': 8
}

# ==============================================
# 🔥 파일 형식 상수
# ==============================================

# 지원 이미지 형식
SUPPORTED_IMAGE_FORMATS = [
    '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'
]

# 지원 메타데이터 형식
SUPPORTED_METADATA_FORMATS = [
    '.json', '.xml', '.yaml', '.yml'
]

# ==============================================
# 🔥 로깅 상수
# ==========================================

# 로그 레벨
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

# 로그 포맷
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ==============================================
# 🔥 에러 코드
# ==========================================

# 에러 코드
ERROR_CODES = {
    'SUCCESS': 0,
    'MODEL_LOAD_ERROR': 1001,
    'INFERENCE_ERROR': 1002,
    'QUALITY_ASSESSMENT_ERROR': 1003,
    'OUTPUT_GENERATION_ERROR': 1004,
    'INTEGRATION_ERROR': 1005,
    'VALIDATION_ERROR': 1006
}

# 에러 메시지
ERROR_MESSAGES = {
    'MODEL_LOAD_ERROR': '모델 로드 실패',
    'INFERENCE_ERROR': '추론 실행 실패',
    'QUALITY_ASSESSMENT_ERROR': '품질 평가 실패',
    'OUTPUT_GENERATION_ERROR': '출력 생성 실패',
    'INTEGRATION_ERROR': '통합 처리 실패',
    'VALIDATION_ERROR': '데이터 검증 실패'
}
