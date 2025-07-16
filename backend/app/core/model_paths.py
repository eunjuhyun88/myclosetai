# app/core/model_paths.py
"""
AI 모델 경로 설정 - 자동 생성됨
"""

from pathlib import Path
from typing import Dict, Optional

# 기본 경로
AI_MODELS_ROOT = Path(__file__).parent.parent.parent / "ai_models"
CHECKPOINTS_ROOT = AI_MODELS_ROOT / "checkpoints"

# 다운로드된 모델들
DOWNLOADED_MODELS = {
    "ootdiffusion": {
        "name": "OOTDiffusion",
        "path": CHECKPOINTS_ROOT / "ootdiffusion_hf",
        "step": "step_06_virtual_fitting",
        "priority": 1,
        "size_gb": 8.0,
        "enabled": True
    },
    "human_parsing": {
        "name": "Human Parsing (Graphonomy)",
        "path": CHECKPOINTS_ROOT / "human_parsing",
        "step": "step_01_human_parsing",
        "priority": 2,
        "size_gb": 0.5,
        "enabled": True
    },
    "u2net": {
        "name": "U²-Net Background Removal",
        "path": CHECKPOINTS_ROOT / "u2net",
        "step": "step_03_cloth_segmentation",
        "priority": 3,
        "size_gb": 0.2,
        "enabled": True
    },
    "stable_diffusion": {
        "name": "Stable Diffusion v1.5",
        "path": CHECKPOINTS_ROOT / "stable-diffusion-v1-5",
        "step": "step_06_virtual_fitting",
        "priority": 4,
        "size_gb": 4.0,
        "enabled": True
    },
    "clip_vit_base": {
        "name": "CLIP ViT-B/32",
        "path": CHECKPOINTS_ROOT / "clip-vit-base-patch32",
        "step": "auxiliary",
        "priority": 5,
        "size_gb": 0.6,
        "enabled": True
    },
    "clip_vit_large": {
        "name": "CLIP ViT-L/14",
        "path": CHECKPOINTS_ROOT / "clip-vit-large-patch14",
        "step": "auxiliary",
        "priority": 6,
        "size_gb": 1.6,
        "enabled": True
    },
}

def get_model_path(model_key: str) -> Optional[Path]:
    """모델 경로 반환"""
    model_info = DOWNLOADED_MODELS.get(model_key)
    if model_info:
        return model_info["path"]
    return None

def is_model_available(model_key: str) -> bool:
    """모델 사용 가능 여부 확인"""
    model_path = get_model_path(model_key)
    return model_path and model_path.exists()

def get_step_model(step_name: str) -> Optional[str]:
    """특정 단계의 모델 반환"""
    for model_key, model_info in DOWNLOADED_MODELS.items():
        if model_info["step"] == step_name:
            return model_key
    return None

def get_all_available_models() -> Dict[str, Dict]:
    """사용 가능한 모든 모델 반환"""
    return {
        key: info for key, info in DOWNLOADED_MODELS.items()
        if is_model_available(key)
    }
