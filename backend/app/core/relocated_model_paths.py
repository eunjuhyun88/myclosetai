# app/core/relocated_model_paths.py
"""
재배치된 AI 모델 경로 설정
자동 생성됨: 2025-07-17 19:32:31
"""

from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 재배치된 모델 경로들
RELOCATED_MODEL_PATHS = {
    "human_parsing_graphonomy": PROJECT_ROOT / "ai_models/checkpoints/step_01_human_parsing/graphonomy.pth",
    "pose_estimation_openpose": PROJECT_ROOT / "ai_models/checkpoints/step_02_pose_estimation/openpose.pth",
    "cloth_segmentation_u2net": PROJECT_ROOT / "ai_models/checkpoints/step_03_cloth_segmentation/u2net.pth",
    "cloth_warping_tom": PROJECT_ROOT / "ai_models/checkpoints/tom_final.pth",
    "virtual_fitting_hrviton": PROJECT_ROOT / "ai_models/checkpoints/hrviton_final.pth",

}

# 모델 타입별 매핑
MODEL_TYPE_MAPPING = {
    "human_parsing_graphonomy": "step_01_human_parsing",
    "pose_estimation_openpose": "step_02_pose_estimation", 
    "cloth_segmentation_u2net": "step_03_cloth_segmentation",
    "geometric_matching_gmm": "step_04_geometric_matching",
    "cloth_warping_tom": "step_05_cloth_warping",
    "virtual_fitting_hrviton": "step_06_virtual_fitting"
}

def get_model_path(model_type: str) -> Path:
    """모델 타입으로 경로 반환"""
    return RELOCATED_MODEL_PATHS.get(model_type, None)

def is_model_available(model_type: str) -> bool:
    """모델 사용 가능 여부 확인"""
    path = get_model_path(model_type)
    return path is not None and path.exists()
