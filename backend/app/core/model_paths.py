# backend/app/core/model_paths.py
"""
AI 모델 경로 설정 - 자동 생성됨
기존 다운로드된 모델들의 실제 경로 매핑
"""

from pathlib import Path
from typing import Dict, Optional, List

# 기본 경로
AI_MODELS_ROOT = Path(__file__).parent.parent.parent / "ai_models"

# 발견된 모델 경로 매핑
DETECTED_MODELS = {
    "ootdiffusion_additional": {
        "name": "OOTDiffusion (Checkpoints)",
        "path": Path("/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/ootdiffusion"),
        "type": "virtual_tryon_additional",
        "ready": True,
        "priority": 99
    },
    "stable_diffusion": {
        "name": "Stable Diffusion v1.5",
        "path": Path("/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/stable-diffusion-v1-5"),
        "type": "base_diffusion",
        "ready": True,
        "priority": 2
    },
    "graphonomy": {
        "name": "Graphonomy (Human Parsing)",
        "path": Path("/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/Graphonomy"),
        "type": "human_parsing",
        "ready": True,
        "priority": 4
    },
    "schp": {
        "name": "Self-Correction Human Parsing",
        "path": Path("/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/Self-Correction-Human-Parsing"),
        "type": "human_parsing",
        "ready": True,
        "priority": 4
    },
    "openpose": {
        "name": "OpenPose",
        "path": Path("/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/openpose"),
        "type": "pose_estimation",
        "ready": True,
        "priority": 4
    },
    "clip": {
        "name": "CLIP ViT-Large",
        "path": Path("/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/checkpoints/clip-vit-large-patch14"),
        "type": "vision_language",
        "ready": True,
        "priority": 5
    },
    "hr_viton": {
        "name": "HR-VITON",
        "path": Path("/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models/HR-VITON"),
        "type": "virtual_tryon",
        "ready": True,
        "priority": 8
    },
}}

# 타입별 모델 그룹핑
def get_models_by_type(model_type: str) -> List[str]:
    """타입별 모델 목록 반환"""
    return [key for key, info in DETECTED_MODELS.items() 
            if info["type"] == model_type and info["ready"]]

def get_virtual_tryon_models() -> List[str]:
    """가상 피팅 모델 목록"""
    return get_models_by_type("virtual_tryon")

def get_primary_ootd_path() -> Path:
    """메인 OOTDiffusion 경로 반환"""
    if "ootdiffusion" in DETECTED_MODELS:
        return DETECTED_MODELS["ootdiffusion"]["path"]
    raise FileNotFoundError("OOTDiffusion 모델을 찾을 수 없습니다")

def get_stable_diffusion_path() -> Path:
    """Stable Diffusion 경로 반환"""
    if "stable_diffusion" in DETECTED_MODELS:
        return DETECTED_MODELS["stable_diffusion"]["path"]
    raise FileNotFoundError("Stable Diffusion 모델을 찾을 수 없습니다")

def get_sam_path(model_size: str = "vit_h") -> Path:
    """SAM 모델 경로 반환"""
    if "sam" in DETECTED_MODELS:
        base_path = DETECTED_MODELS["sam"]["path"]
        if model_size == "vit_h":
            return Path(base_path) / "sam_vit_h_4b8939.pth"
        elif model_size == "vit_b":
            return Path(base_path) / "sam_vit_b_01ec64.pth"
    raise FileNotFoundError(f"SAM {model_size} 모델을 찾을 수 없습니다")

def is_model_available(model_key: str) -> bool:
    """모델 사용 가능 여부 확인"""
    if model_key in DETECTED_MODELS:
        model_path = DETECTED_MODELS[model_key]["path"]
        return Path(model_path).exists()
    return False

def get_all_available_models() -> List[str]:
    """사용 가능한 모든 모델 목록"""
    available = []
    for key, info in DETECTED_MODELS.items():
        if info["ready"] and Path(info["path"]).exists():
            available.append(key)
    return sorted(available, key=lambda x: DETECTED_MODELS[x]["priority"])

def get_model_info(model_key: str) -> Optional[Dict]:
    """모델 정보 반환"""
    return DETECTED_MODELS.get(model_key)

# 빠른 경로 접근
class ModelPaths:
    """모델 경로 빠른 접근 클래스"""
    
    @property
    def ootd_hf(self) -> Path:
        return get_primary_ootd_path()
    
    @property
    def stable_diffusion(self) -> Path:
        return get_stable_diffusion_path()
    
    @property
    def sam_large(self) -> Path:
        return get_sam_path("vit_h")
    
    @property
    def sam_base(self) -> Path:
        return get_sam_path("vit_b")

# 전역 인스턴스
model_paths = ModelPaths()
