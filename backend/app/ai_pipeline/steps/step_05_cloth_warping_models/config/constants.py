#!/usr/bin/env python3
"""
Cloth Warping Constants
의류 워핑 상수들
"""

# RealVisXL 모델 상수
REALVISXL_CONSTANTS = {
    "MODEL_TYPE": "realvisxl_v4",
    "CHECKPOINT_PATH": "realvisxl_v4.0.safetensors",
    "DEVICE": "mps",
    "MAX_RESOLUTION": 1024,
    "BATCH_SIZE": 1,
    "NUM_INFERENCE_STEPS": 20,
    "GUIDANCE_SCALE": 7.5,
    "STRENGTH": 0.8
}

# 워핑 품질 임계값
WARPING_QUALITY_THRESHOLDS = {
    "MIN_CONFIDENCE": 0.7,
    "MIN_WARP_QUALITY": 0.6,
    "MAX_DISTORTION": 0.3,
    "MIN_EDGE_PRESERVATION": 0.5
}

# 의류 타입별 워핑 설정
CLOTHING_TYPE_WARPING = {
    "shirt": {
        "warp_strength": 0.8,
        "preserve_patterns": True,
        "maintain_fit": True
    },
    "pants": {
        "warp_strength": 0.7,
        "preserve_patterns": True,
        "maintain_fit": True
    },
    "dress": {
        "warp_strength": 0.9,
        "preserve_patterns": True,
        "maintain_fit": True
    },
    "jacket": {
        "warp_strength": 0.6,
        "preserve_patterns": True,
        "maintain_fit": True
    }
}

# 워핑 알고리즘 설정
WARPING_ALGORITHMS = {
    "thin_plate_spline": {
        "control_points": 20,
        "regularization": 0.1
    },
    "affine": {
        "preserve_aspect": True,
        "maintain_scale": True
    },
    "perspective": {
        "corner_points": 4,
        "interpolation": "bilinear"
    }
}

# 출력 품질 설정
OUTPUT_QUALITY = {
    "resolution": 1024,
    "format": "png",
    "compression": 95,
    "metadata": True
}

# 성능 최적화 설정
PERFORMANCE_OPTIONS = {
    "use_amp": True,
    "memory_efficient": True,
    "gradient_checkpointing": False,
    "batch_processing": True
}
