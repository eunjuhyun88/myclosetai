# app/core/corrected_model_paths.py
"""
수정된 AI 모델 경로 설정 - 정확한 매칭 기반
자동 생성됨: 2025-07-18 02:05:34
"""

from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 정확히 매칭된 모델 경로들
CORRECTED_MODEL_PATHS = {
    "human_parsing_graphonomy": PROJECT_ROOT / "ai_models/checkpoints/step_01_human_parsing/graphonomy.pth",
    "pose_estimation_openpose": PROJECT_ROOT / "ai_models/checkpoints/step_02_pose_estimation/openpose.pth",
    "geometric_matching_gmm": PROJECT_ROOT / "ai_models/checkpoints/step_04_geometric_matching/gmm_final.pth",
    "cloth_warping_tom": PROJECT_ROOT / "ai_models/checkpoints/step_05_cloth_warping/tom_final.pth",
    "virtual_fitting_hrviton": PROJECT_ROOT / "ai_models/checkpoints/step_06_virtual_fitting/hrviton_final.pth",

}

# ModelLoader 호환 경로 매핑
MODEL_LOADER_PATHS = {
    # Step 01: Human Parsing
    "human_parsing_graphonomy": {
        "primary": CORRECTED_MODEL_PATHS.get("human_parsing_graphonomy"),
        "alternatives": []
    },
    
    # Step 02: Pose Estimation  
    "pose_estimation_openpose": {
        "primary": CORRECTED_MODEL_PATHS.get("pose_estimation_openpose"),
        "alternatives": []
    },
    
    # Step 03: Cloth Segmentation
    "cloth_segmentation_u2net": {
        "primary": CORRECTED_MODEL_PATHS.get("cloth_segmentation_u2net"),
        "alternatives": []
    },
    
    # Step 04: Geometric Matching
    "geometric_matching_gmm": {
        "primary": CORRECTED_MODEL_PATHS.get("geometric_matching_gmm"),
        "alternatives": []
    },
    
    # Step 05: Cloth Warping
    "cloth_warping_tom": {
        "primary": CORRECTED_MODEL_PATHS.get("cloth_warping_tom"),
        "alternatives": []
    },
    
    # Step 06: Virtual Fitting
    "virtual_fitting_hrviton": {
        "primary": CORRECTED_MODEL_PATHS.get("virtual_fitting_hrviton"),
        "alternatives": []
    }
}

def get_model_path(model_type: str) -> Path:
    """모델 타입으로 경로 반환"""
    return CORRECTED_MODEL_PATHS.get(model_type, None)

def is_model_available(model_type: str) -> bool:
    """모델 사용 가능 여부 확인"""
    path = get_model_path(model_type)
    return path is not None and path.exists()

def get_all_available_models() -> Dict[str, str]:
    """사용 가능한 모든 모델 반환"""
    available = {}
    for model_type, path in CORRECTED_MODEL_PATHS.items():
        if path.exists():
            available[model_type] = str(path)
    return available

# 총 재배치 정보
RELOCATE_SUMMARY = {
    "total_models": 5,
    "total_size_mb": 6183.5,
    "generation_time": "2025-07-18 02:05:34",
    "corrected_issues": [
        "SAM 모델 분리",
        "정확한 경로 매칭",
        "실제 존재 파일만 사용"
    ]
}
