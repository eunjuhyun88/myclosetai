"""
자동 생성된 모델 경로 파일
다운로드된 AI 모델들의 경로 정의
"""

from pathlib import Path

# 기본 경로
MODELS_DIR = Path(__file__).parent.parent.parent / "ai_models"

# 모델 경로들
OOTDIFFUSION_PATH = MODELS_DIR / "checkpoints/ootdiffusion"
STABLE_DIFFUSION_INPAINT_PATH = MODELS_DIR / "checkpoints/stable_diffusion_inpaint"
HUMAN_PARSING_PATH = MODELS_DIR / "checkpoints/human_parsing"
POSE_ESTIMATION_PATH = MODELS_DIR / "checkpoints/openpose/ckpts"
SAM_SEGMENTATION_PATH = MODELS_DIR / "checkpoints/sam_vit_h"
CLIP_VIT_BASE_PATH = MODELS_DIR / "checkpoints/clip-vit-base-patch32"
CONTROLNET_OPENPOSE_PATH = MODELS_DIR / "checkpoints/controlnet_openpose"
CLOTH_SEGMENTATION_PATH = MODELS_DIR / "checkpoints/cloth_segmentation"

# 모델 경로 딕셔너리
MODEL_PATHS = {
    "ootdiffusion": OOTDIFFUSION_PATH,
    "stable_diffusion_inpaint": STABLE_DIFFUSION_INPAINT_PATH,
    "human_parsing": HUMAN_PARSING_PATH,
    "pose_estimation": POSE_ESTIMATION_PATH,
    "sam_segmentation": SAM_SEGMENTATION_PATH,
    "clip_vit_base": CLIP_VIT_BASE_PATH,
    "controlnet_openpose": CONTROLNET_OPENPOSE_PATH,
    "cloth_segmentation": CLOTH_SEGMENTATION_PATH,
}

def get_model_path(model_name: str) -> Path:
    """모델 경로 가져오기"""
    return MODEL_PATHS.get(model_name)

def is_model_available(model_name: str) -> bool:
    """모델 사용 가능 여부 확인"""
    path = get_model_path(model_name)
    return path and path.exists() and any(path.iterdir())
