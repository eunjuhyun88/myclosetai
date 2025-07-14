# app/core/model_paths.py
"""
AI 모델 경로 설정 - 자동 생성됨
기존 다운로드된 모델들의 실제 경로 매핑
"""

from pathlib import Path
from typing import Dict, Optional, List

# 기본 경로
AI_MODELS_ROOT = Path(__file__).parent.parent.parent / "ai_models"

# 스캔된 모델 정보
SCANNED_MODELS = {
    "ootdiffusion": {
        "name": "OOTDiffusion",
        "type": "diffusion",
        "step": "step_06_virtual_fitting",
        "path": AI_MODELS_ROOT / "OOTDiffusion",
        "ready": False,
        "size_mb": 51.8,
        "priority": 1
    },
    "hr_viton": {
        "name": "HR-VITON",
        "type": "virtual_tryon",
        "step": "step_05_cloth_warping",
        "path": AI_MODELS_ROOT / "HR-VITON",
        "ready": False,
        "size_mb": 17.4,
        "priority": 2
    },
    "graphonomy": {
        "name": "Graphonomy",
        "type": "human_parsing",
        "step": "step_01_human_parsing",
        "path": AI_MODELS_ROOT / "Graphonomy",
        "ready": False,
        "size_mb": 0.9,
        "priority": 3
    },
    "openpose": {
        "name": "OpenPose",
        "type": "pose_estimation",
        "step": "step_02_pose_estimation",
        "path": AI_MODELS_ROOT / "openpose",
        "ready": True,
        "size_mb": 148.9,
        "priority": 4
    },
    "detectron2": {
        "name": "Detectron2",
        "type": "detection_segmentation",
        "step": "auxiliary",
        "path": AI_MODELS_ROOT / "detectron2",
        "ready": False,
        "size_mb": 11.0,
        "priority": 5
    },
    "self_correction_parsing": {
        "name": "Self-Correction Human Parsing",
        "type": "human_parsing",
        "step": "step_01_human_parsing",
        "path": AI_MODELS_ROOT / "Self-Correction-Human-Parsing",
        "ready": False,
        "size_mb": 11.3,
        "priority": 6
    },
    "checkpoints": {
        "name": "Additional Checkpoints",
        "type": "mixed",
        "step": "auxiliary",
        "path": AI_MODELS_ROOT / "checkpoints",
        "ready": True,
        "size_mb": 80363.6,
        "priority": 7
    },
}

# 단계별 모델 매핑
STEP_TO_MODELS = {
    "step_01_human_parsing": ["graphonomy", "self_correction_parsing"],
    "step_02_pose_estimation": ["openpose"],
    "step_03_cloth_segmentation": [],  # U2Net 등 추가 필요
    "step_04_geometric_matching": [],  # HR-VITON GMM
    "step_05_cloth_warping": ["hr_viton"],  # HR-VITON TOM
    "step_06_virtual_fitting": ["ootdiffusion", "hr_viton"],
    "step_07_post_processing": [],
    "step_08_quality_assessment": []
}

def get_model_path(model_key: str) -> Optional[Path]:
    """모델 경로 반환"""
    if model_key in SCANNED_MODELS:
        return SCANNED_MODELS[model_key]["path"]
    return None

def is_model_ready(model_key: str) -> bool:
    """모델 사용 가능 여부 확인"""
    if model_key in SCANNED_MODELS:
        model_info = SCANNED_MODELS[model_key]
        return model_info["ready"] and model_info["path"].exists()
    return False

def get_ready_models() -> List[str]:
    """사용 가능한 모델 목록"""
    return [key for key, info in SCANNED_MODELS.items() if info["ready"]]

def get_models_for_step(step: str) -> List[str]:
    """특정 단계에 사용 가능한 모델들"""
    available_models = []
    for model_key in STEP_TO_MODELS.get(step, []):
        if is_model_ready(model_key):
            available_models.append(model_key)
    return available_models

def get_primary_model_for_step(step: str) -> Optional[str]:
    """단계별 주요 모델 반환 (우선순위 기준)"""
    models = get_models_for_step(step)
    if not models:
        return None
    
    # 우선순위로 정렬
    models_with_priority = [(model, SCANNED_MODELS[model]["priority"]) for model in models]
    models_with_priority.sort(key=lambda x: x[1])
    
    return models_with_priority[0][0] if models_with_priority else None

def get_ootdiffusion_path() -> Optional[Path]:
    """OOTDiffusion 경로 반환"""
    return get_model_path("ootdiffusion")

def get_hr_viton_path() -> Optional[Path]:
    """HR-VITON 경로 반환"""
    return get_model_path("hr_viton")

def get_graphonomy_path() -> Optional[Path]:
    """Graphonomy 경로 반환"""
    return get_model_path("graphonomy")

def get_openpose_path() -> Optional[Path]:
    """OpenPose 경로 반환"""
    return get_model_path("openpose")

def get_model_info(model_key: str) -> Optional[Dict]:
    """모델 상세 정보 반환"""
    return SCANNED_MODELS.get(model_key)

def list_all_models() -> Dict[str, Dict]:
    """모든 모델 정보 반환"""
    return SCANNED_MODELS.copy()
