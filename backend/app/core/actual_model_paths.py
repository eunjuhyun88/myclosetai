# Auto-generated model paths configuration
"""
자동 생성된 모델 경로 설정
Generated at: 2025-07-17 19:20:00
"""

from pathlib import Path

# 베이스 경로
BACKEND_DIR = Path(__file__).parent.parent
AI_MODELS_DIR = BACKEND_DIR / "ai_models"

# 실제 탐지된 모델 경로들
ACTUAL_MODEL_PATHS = {
    "human_parsing_graphonomy": "/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/step_01_human_parsing/graphonomy.pth",
    "virtual_fitting_hrviton": "/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/hrviton_final.pth",
    "pose_estimation_openpose": "/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/step_02_pose_estimation/openpose.pth",
    "stable_diffusion": "/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/diffusion/stable-diffusion-v1-5",
    "clip_vit_base": "/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/clip-vit-base-patch32/pytorch_model.bin",
    "geometric_matching_gmm": "/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/gmm_final.pth",

}

# 모델 가용성 체크
MODEL_AVAILABILITY = {
    "human_parsing_graphonomy": True,
    "pose_estimation_openpose": True,
    "cloth_segmentation_u2net": True,
    "geometric_matching_gmm": True,
    "cloth_warping_tom": False,
    "virtual_fitting_hrviton": True,
    "stable_diffusion": True,
    "clip_vit_base": True,

}

def get_model_path(model_key: str) -> str:
    """모델 경로 반환"""
    return ACTUAL_MODEL_PATHS.get(model_key, "")

def is_model_available(model_key: str) -> bool:
    """모델 사용 가능 여부 확인"""
    return MODEL_AVAILABILITY.get(model_key, False)

def get_available_models() -> list:
    """사용 가능한 모델 목록 반환"""
    return [k for k, v in MODEL_AVAILABILITY.items() if v]
