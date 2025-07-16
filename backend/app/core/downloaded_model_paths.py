# app/core/downloaded_model_paths.py
"""
다운로드된 AI 모델 경로 설정 - 자동 생성됨
"""

from pathlib import Path
from typing import Dict, Optional

# 기본 경로
AI_MODELS_ROOT = Path(__file__).parent.parent.parent / "ai_models"
CHECKPOINTS_ROOT = AI_MODELS_ROOT / "checkpoints"

# 다운로드된 모델들
DOWNLOADED_MODELS = {
    "human_parsing": {
        "name": "Human Parsing ATR",
        "path": CHECKPOINTS_ROOT / "human_parsing",
        "step": "step_01_human_parsing",
        "type": "human_parsing",
        "priority": 1,
        "size_gb": 0.5,
        "downloaded": True,
        "repo_id": "mattmdjaga/segformer_b2_clothes"
    },
    "clip-vit-base-patch32": {
        "name": "CLIP ViT Base",
        "path": CHECKPOINTS_ROOT / "clip-vit-base-patch32",
        "step": "auxiliary",
        "type": "text_image",
        "priority": 1,
        "size_gb": 0.6,
        "downloaded": True,
        "repo_id": "openai/clip-vit-base-patch32"
    },
    "openpose": {
        "name": "OpenPose Model",
        "path": CHECKPOINTS_ROOT / "openpose",
        "step": "step_02_pose_estimation",
        "type": "pose_estimation",
        "priority": 1,
        "size_gb": 0.8,
        "downloaded": True,
        "repo_id": "lllyasviel/Annotators"
    },
    "controlnet_openpose": {
        "name": "ControlNet OpenPose",
        "path": CHECKPOINTS_ROOT / "controlnet_openpose",
        "step": "auxiliary",
        "type": "controlnet",
        "priority": 4,
        "size_gb": 1.4,
        "downloaded": True,
        "repo_id": "lllyasviel/sd-controlnet-openpose"
    },
    "sam_vit_h": {
        "name": "SAM ViT-H",
        "path": CHECKPOINTS_ROOT / "sam_vit_h",
        "step": "auxiliary",
        "type": "segmentation",
        "priority": 3,
        "size_gb": 2.4,
        "downloaded": True,
        "repo_id": "facebook/sam-vit-huge"
    },
    "stable-diffusion-v1-5": {
        "name": "Stable Diffusion v1.5",
        "path": CHECKPOINTS_ROOT / "stable-diffusion-v1-5",
        "step": "step_06_virtual_fitting",
        "type": "diffusion",
        "priority": 3,
        "size_gb": 4.0,
        "downloaded": True,
        "repo_id": "runwayml/stable-diffusion-v1-5"
    },
    "clip-vit-large-patch14": {
        "name": "CLIP ViT Large",
        "path": CHECKPOINTS_ROOT / "clip-vit-large-patch14",
        "step": "auxiliary",
        "type": "text_image",
        "priority": 2,
        "size_gb": 6.5,
        "downloaded": True,
        "repo_id": "openai/clip-vit-large-patch14"
    },
    "ootdiffusion": {
        "name": "OOTDiffusion Main",
        "path": CHECKPOINTS_ROOT / "ootdiffusion",
        "step": "step_06_virtual_fitting",
        "type": "diffusion",
        "priority": 1,
        "size_gb": 15.0,
        "downloaded": True,
        "repo_id": "levihsu/OOTDiffusion"
    },
    "ootdiffusion_hf": {
        "name": "OOTDiffusion HF",
        "path": CHECKPOINTS_ROOT / "ootdiffusion_hf",
        "step": "step_06_virtual_fitting",
        "type": "diffusion",
        "priority": 2,
        "size_gb": 20.0,
        "downloaded": True,
        "repo_id": "yisol/IDM-VTON"
    },
}

def get_model_path(model_key: str) -> Optional[Path]:
    """모델 경로 반환"""
    model_info = DOWNLOADED_MODELS.get(model_key)
    if model_info and model_info["downloaded"]:
        return model_info["path"]
    return None

def get_models_by_step(step: str) -> Dict[str, Dict]:
    """단계별 모델들 반환"""
    return {
        key: info for key, info in DOWNLOADED_MODELS.items()
        if info["step"] == step and info["downloaded"]
    }

def get_required_models() -> Dict[str, Dict]:
    """필수 모델들만 반환"""
    return {
        key: info for key, info in DOWNLOADED_MODELS.items()
        if info["priority"] <= 2 and info["downloaded"]
    }
