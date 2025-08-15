"""
Virtual Fitting Constants
가상 피팅 상수들
"""

# HR-VITON 모델 상수
HR_VITON_CONSTANTS = {
    "MODEL_TYPE": "hr_viton",
    "CHECKPOINT_PATH": "hr_viton.pth",
    "DEVICE": "mps",
    "MAX_RESOLUTION": 1024,
    "BATCH_SIZE": 1,
    "NUM_INFERENCE_STEPS": 30,
    "GUIDANCE_SCALE": 5.0,
    "STRENGTH": 0.7
}

# VITON-HD 모델 상수
VITON_HD_CONSTANTS = {
    "MODEL_TYPE": "viton_hd",
    "CHECKPOINT_PATH": "viton_hd.pth",
    "DEVICE": "mps",
    "MAX_RESOLUTION": 1024,
    "BATCH_SIZE": 1,
    "NUM_INFERENCE_STEPS": 40,
    "GUIDANCE_SCALE": 6.0,
    "STRENGTH": 0.75
}

# OOTD 모델 상수
OOTD_CONSTANTS = {
    "MODEL_TYPE": "ootd_v2",
    "CHECKPOINT_PATH": "ootd_v2.0.pth",
    "DEVICE": "mps",
    "MAX_RESOLUTION": 1024,
    "BATCH_SIZE": 1,
    "NUM_INFERENCE_STEPS": 50,
    "GUIDANCE_SCALE": 7.5,
    "STRENGTH": 0.8
}

# 피팅 품질 임계값
FITTING_QUALITY_THRESHOLDS = {
    "MIN_CONFIDENCE": 0.7,
    "MIN_FITTING_QUALITY": 0.6,
    "MAX_DISTORTION": 0.3,
    "MIN_EDGE_PRESERVATION": 0.5
}

# 의류 타입별 피팅 설정
CLOTHING_TYPE_FITTING = {
    "shirt": {
        "fitting_strength": 0.8,
        "preserve_patterns": True,
        "maintain_fit": True
    },
    "pants": {
        "fitting_strength": 0.7,
        "preserve_patterns": True,
        "maintain_fit": True
    },
    "dress": {
        "fitting_strength": 0.9,
        "preserve_patterns": True,
        "maintain_fit": True
    },
    "jacket": {
        "fitting_strength": 0.6,
        "preserve_patterns": True,
        "maintain_fit": True
    }
}

# 피팅 알고리즘 설정
FITTING_ALGORITHMS = {
    "diffusion": {
        "num_steps": 50,
        "guidance_scale": 7.5,
        "strength": 0.8
    },
    "gan": {
        "discriminator_steps": 5,
        "generator_steps": 1,
        "learning_rate": 0.0002
    },
    "hybrid": {
        "diffusion_weight": 0.7,
        "gan_weight": 0.3,
        "ensemble_method": "weighted"
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

# 통합 피팅 상수 (FITTING_CONSTANTS)
FITTING_CONSTANTS = {
    "DEFAULT_MODEL": "hr_viton",
    "DEFAULT_DEVICE": "mps",
    "DEFAULT_RESOLUTION": 1024,
    "DEFAULT_BATCH_SIZE": 1,
    "DEFAULT_STRENGTH": 0.7,
    "DEFAULT_GUIDANCE_SCALE": 5.0,
    "DEFAULT_STEPS": 30,
    "SUPPORTED_MODELS": ["hr_viton", "viton_hd", "ootd_v2"],
    "SUPPORTED_DEVICES": ["cpu", "cuda", "mps"],
    "MAX_RESOLUTION": 2048,
    "MIN_RESOLUTION": 256,
    "QUALITY_PRESETS": {
        "fast": {"steps": 20, "guidance": 3.0},
        "balanced": {"steps": 30, "guidance": 5.0},
        "high": {"steps": 50, "guidance": 7.5}
    }
}

# 품질 임계값 (QUALITY_THRESHOLDS)
QUALITY_THRESHOLDS = {
    "EXCELLENT": 0.9,
    "GOOD": 0.7,
    "ACCEPTABLE": 0.5,
    "POOR": 0.3,
    "MIN_ACCEPTABLE": 0.4,
    "FITTING_QUALITY": {
        "PERFECT": 0.95,
        "EXCELLENT": 0.85,
        "GOOD": 0.75,
        "ACCEPTABLE": 0.65,
        "POOR": 0.55,
        "UNACCEPTABLE": 0.45
    },
    "TEXTURE_PRESERVATION": {
        "PERFECT": 0.95,
        "EXCELLENT": 0.85,
        "GOOD": 0.75,
        "ACCEPTABLE": 0.65,
        "POOR": 0.55
    },
    "LIGHTING_CONSISTENCY": {
        "PERFECT": 0.95,
        "EXCELLENT": 0.85,
        "GOOD": 0.75,
        "ACCEPTABLE": 0.65,
        "POOR": 0.55
    }
}
