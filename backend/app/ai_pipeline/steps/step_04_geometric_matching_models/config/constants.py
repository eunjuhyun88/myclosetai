"""
Geometric Matching Constants
기하학적 매칭에 필요한 상수들을 정의합니다.
"""

# GMM (Geometric Matching Model) 상수
GMM_CONSTANTS = {
    "DEFAULT_GRID_SIZE": 8,
    "MIN_GRID_SIZE": 4,
    "MAX_GRID_SIZE": 16,
    "DEFAULT_ITERATIONS": 100,
    "MIN_ITERATIONS": 50,
    "MAX_ITERATIONS": 500,
    "DEFAULT_LEARNING_RATE": 0.01,
    "MIN_LEARNING_RATE": 0.001,
    "MAX_LEARNING_RATE": 0.1,
    "DEFAULT_REGULARIZATION": 0.1,
    "MIN_REGULARIZATION": 0.01,
    "MAX_REGULARIZATION": 1.0
}

# TPS (Thin Plate Spline) 상수
TPS_CONSTANTS = {
    "DEFAULT_CONTROL_POINTS": 16,
    "MIN_CONTROL_POINTS": 9,
    "MAX_CONTROL_POINTS": 64,
    "DEFAULT_LAMBDA": 0.1,
    "MIN_LAMBDA": 0.01,
    "MAX_LAMBDA": 1.0,
    "DEFAULT_ITERATIONS": 200,
    "MIN_ITERATIONS": 100,
    "MAX_ITERATIONS": 1000
}

# 매칭 품질 임계값
MATCHING_QUALITY_THRESHOLDS = {
    "EXCELLENT": 0.9,
    "GOOD": 0.7,
    "ACCEPTABLE": 0.5,
    "POOR": 0.3,
    "MIN_ACCEPTABLE": 0.4
}

# 그리드 매칭 상수
GRID_MATCHING_CONSTANTS = {
    "DEFAULT_CELL_SIZE": 32,
    "MIN_CELL_SIZE": 16,
    "MAX_CELL_SIZE": 128,
    "DEFAULT_OVERLAP": 0.5,
    "MIN_OVERLAP": 0.1,
    "MAX_OVERLAP": 0.9
}

# 특징점 매칭 상수
FEATURE_MATCHING_CONSTANTS = {
    "DEFAULT_MAX_FEATURES": 1000,
    "MIN_MAX_FEATURES": 100,
    "MAX_MAX_FEATURES": 10000,
    "DEFAULT_MATCH_RATIO": 0.75,
    "MIN_MATCH_RATIO": 0.5,
    "MAX_MATCH_RATIO": 0.95,
    "DEFAULT_MIN_MATCHES": 10,
    "MIN_MIN_MATCHES": 4,
    "MAX_MIN_MATCHES": 100
}

# 변형 모델 상수
TRANSFORMATION_MODELS = {
    "AFFINE": "affine",
    "PERSPECTIVE": "perspective",
    "TPS": "tps",
    "GMM": "gmm",
    "HYBRID": "hybrid"
}

# 품질 평가 메트릭
QUALITY_METRICS = {
    "SSIM": "ssim",
    "PSNR": "psnr",
    "MSE": "mse",
    "MAE": "mae",
    "CORRELATION": "correlation"
}

# 기본 설정
DEFAULT_CONFIG = {
    "matching_method": "gmm",
    "grid_size": GMM_CONSTANTS["DEFAULT_GRID_SIZE"],
    "iterations": GMM_CONSTANTS["DEFAULT_ITERATIONS"],
    "learning_rate": GMM_CONSTANTS["DEFAULT_LEARNING_RATE"],
    "regularization": GMM_CONSTANTS["DEFAULT_REGULARIZATION"],
    "quality_threshold": MATCHING_QUALITY_THRESHOLDS["ACCEPTABLE"],
    "cell_size": GRID_MATCHING_CONSTANTS["DEFAULT_CELL_SIZE"],
    "overlap": GRID_MATCHING_CONSTANTS["DEFAULT_OVERLAP"],
    "max_features": FEATURE_MATCHING_CONSTANTS["DEFAULT_MAX_FEATURES"],
    "match_ratio": FEATURE_MATCHING_CONSTANTS["DEFAULT_MATCH_RATIO"],
    "min_matches": FEATURE_MATCHING_CONSTANTS["DEFAULT_MIN_MATCHES"]
}
