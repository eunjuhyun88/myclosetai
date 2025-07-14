# app/core/optimized_model_paths.py
"""
최적화된 AI 모델 경로 설정 - 체크포인트 분석 기반
실제 사용 가능한 체크포인트들로만 구성
"""

from pathlib import Path
from typing import Dict, Optional, List, Any

# 기본 경로
AI_MODELS_ROOT = Path(__file__).parent.parent.parent / "ai_models"
CHECKPOINTS_ROOT = AI_MODELS_ROOT / "checkpoints"

# 분석된 체크포인트 모델들
ANALYZED_MODELS = {
    "clip-vit-large-patch14": {
        "name": "clip-vit-large-patch14",
        "type": "text_image",
        "step": "auxiliary",
        "path": CHECKPOINTS_ROOT / "clip-vit-large-patch14",
        "ready": True,
        "size_mb": 6527.1,
        "priority": 9,
        "checkpoints": [{'name': 'pytorch_model.bin', 'path': 'pytorch_model.bin', 'size_mb': 1631.4, 'type': '.bin'}],  # 상위 3개만
        "total_checkpoints": 1
    },
    "human_parsing": {
        "name": "human_parsing",
        "type": "human_parsing",
        "step": "step_01_human_parsing",
        "path": CHECKPOINTS_ROOT / "human_parsing",
        "ready": True,
        "size_mb": 510.1,
        "priority": 5,
        "checkpoints": [{'name': 'atr_model.pth', 'path': 'atr_model.pth', 'size_mb': 255.1, 'type': '.pth'}, {'name': 'lip_model.pth', 'path': 'lip_model.pth', 'size_mb': 255.1, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 2
    },
    "ootdiffusion_hf": {
        "name": "ootdiffusion_hf",
        "type": "diffusion",
        "step": "step_06_virtual_fitting",
        "path": CHECKPOINTS_ROOT / "ootdiffusion_hf",
        "ready": True,
        "size_mb": 15129.3,
        "priority": 1,
        "checkpoints": [{'name': 'pytorch_model.bin', 'path': 'checkpoints/ootd/text_encoder/pytorch_model.bin', 'size_mb': 469.5, 'type': '.bin'}, {'name': 'diffusion_pytorch_model.bin', 'path': 'checkpoints/ootd/vae/diffusion_pytorch_model.bin', 'size_mb': 319.2, 'type': '.bin'}, {'name': 'body_pose_model.pth', 'path': 'checkpoints/openpose/ckpts/body_pose_model.pth', 'size_mb': 199.6, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 5
    },
    "stable-diffusion-v1-5": {
        "name": "stable-diffusion-v1-5",
        "type": "diffusion",
        "step": "step_06_virtual_fitting",
        "path": CHECKPOINTS_ROOT / "stable-diffusion-v1-5",
        "ready": True,
        "size_mb": 45070.6,
        "priority": 1,
        "checkpoints": [{'name': 'v1-5-pruned.ckpt', 'path': 'v1-5-pruned.ckpt', 'size_mb': 7346.9, 'type': '.ckpt'}, {'name': 'v1-5-pruned-emaonly.ckpt', 'path': 'v1-5-pruned-emaonly.ckpt', 'size_mb': 4067.8, 'type': '.ckpt'}, {'name': 'pytorch_model.fp16.bin', 'path': 'text_encoder/pytorch_model.fp16.bin', 'size_mb': 234.8, 'type': '.bin'}],  # 상위 3개만
        "total_checkpoints": 11
    },
}

# 단계별 최적 모델 매핑
STEP_OPTIMAL_MODELS = {
    "auxiliary": "clip-vit-large-patch14",
    "step_01_human_parsing": "human_parsing",
    "step_06_virtual_fitting": "ootdiffusion_hf",
}

def get_optimal_model_for_step(step: str) -> Optional[str]:
    """단계별 최적 모델 반환"""
    return STEP_OPTIMAL_MODELS.get(step)

def get_model_checkpoints(model_name: str) -> List[Dict]:
    """모델의 체크포인트 목록 반환"""
    if model_name in ANALYZED_MODELS:
        return ANALYZED_MODELS[model_name]["checkpoints"]
    return []

def get_largest_checkpoint(model_name: str) -> Optional[str]:
    """모델의 가장 큰 체크포인트 반환 (보통 메인 모델)"""
    checkpoints = get_model_checkpoints(model_name)
    if not checkpoints:
        return None
    
    largest = max(checkpoints, key=lambda x: x['size_mb'])
    return largest['path']

def get_ready_models_by_type(model_type: str) -> List[str]:
    """타입별 사용 가능한 모델들"""
    return [name for name, info in ANALYZED_MODELS.items() 
            if info["type"] == model_type and info["ready"]]

def get_diffusion_models() -> List[str]:
    """Diffusion 모델들 (OOTD 등)"""
    return get_ready_models_by_type("diffusion")

def get_virtual_tryon_models() -> List[str]:
    """가상 피팅 모델들 (HR-VITON 등)"""
    return get_ready_models_by_type("virtual_tryon")

def get_human_parsing_models() -> List[str]:
    """인체 파싱 모델들"""
    return get_ready_models_by_type("human_parsing")

def get_model_info(model_name: str) -> Optional[Dict]:
    """모델 상세 정보 반환"""
    return ANALYZED_MODELS.get(model_name)

def list_all_ready_models() -> Dict[str, Dict]:
    """모든 사용 가능한 모델 정보"""
    return ANALYZED_MODELS.copy()

# 빠른 접근 함수들
def get_best_diffusion_model() -> Optional[str]:
    """최고 성능 Diffusion 모델"""
    return get_optimal_model_for_step("step_06_virtual_fitting")

def get_best_human_parsing_model() -> Optional[str]:
    """최고 성능 인체 파싱 모델"""  
    return get_optimal_model_for_step("step_01_human_parsing")

def get_model_path(model_name: str) -> Optional[Path]:
    """모델 디렉토리 경로 반환"""
    if model_name in ANALYZED_MODELS:
        return ANALYZED_MODELS[model_name]["path"]
    return None

def get_checkpoint_path(model_name: str, checkpoint_name: Optional[str] = None) -> Optional[Path]:
    """특정 체크포인트 파일 경로 반환"""
    model_path = get_model_path(model_name)
    if not model_path:
        return None
    
    if checkpoint_name:
        return model_path / checkpoint_name
    else:
        # 가장 큰 체크포인트 반환
        largest_ckpt = get_largest_checkpoint(model_name)
        return model_path / largest_ckpt if largest_ckpt else None

# 사용 통계
ANALYSIS_STATS = {
    "total_models": len(ANALYZED_MODELS),
    "total_size_gb": sum(info["size_mb"] for info in ANALYZED_MODELS.values()) / 1024,
    "models_by_step": {step: len([m for m in ANALYZED_MODELS.values() if m["step"] == step]) 
                      for step in set(info["step"] for info in ANALYZED_MODELS.values())},
    "largest_model": max(ANALYZED_MODELS.items(), key=lambda x: x[1]["size_mb"])[0] if ANALYZED_MODELS else None
}
