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
    "auxiliary": {
        "name": "auxiliary",
        "type": "diffusion",
        "step": "step_06_virtual_fitting",
        "path": CHECKPOINTS_ROOT / "auxiliary",
        "ready": True,
        "size_mb": 3135.1,
        "priority": 2,
        "checkpoints": [{'name': 'pytorch_model.bin', 'path': 'pytorch_model.bin', 'size_mb': 577.2, 'type': '.bin'}, {'name': 'resnet50_features.pth', 'path': 'resnet50_features/resnet50_features.pth', 'size_mb': 97.8, 'type': '.pth'}, {'name': 'pytorch_model.bin', 'path': 'clip-vit-base-patch32/pytorch_model.bin', 'size_mb': 577.2, 'type': '.bin'}],  # 상위 3개만
        "total_checkpoints": 7
    },
    "background_removal": {
        "name": "background_removal",
        "type": "cloth_segmentation",
        "step": "step_03_cloth_segmentation",
        "path": CHECKPOINTS_ROOT / "background_removal",
        "ready": True,
        "size_mb": 803.2,
        "priority": 7,
        "checkpoints": [{'name': 'model.pth', 'path': 'model.pth', 'size_mb': 168.5, 'type': '.pth'}, {'name': 'pytorch_model.bin', 'path': 'pytorch_model.bin', 'size_mb': 168.4, 'type': '.bin'}, {'name': 'u2net.pth', 'path': 'u2net.pth', 'size_mb': 0.0, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 3
    },
    "clip-vit-base-patch32": {
        "name": "clip-vit-base-patch32",
        "type": "text_image",
        "step": "auxiliary",
        "path": CHECKPOINTS_ROOT / "clip-vit-base-patch32",
        "ready": True,
        "size_mb": 580.7,
        "priority": 11,
        "checkpoints": [{'name': 'pytorch_model.bin', 'path': 'pytorch_model.bin', 'size_mb': 577.2, 'type': '.bin'}],  # 상위 3개만
        "total_checkpoints": 1
    },
    "clip-vit-large-patch14": {
        "name": "clip-vit-large-patch14",
        "type": "text_image",
        "step": "auxiliary",
        "path": CHECKPOINTS_ROOT / "clip-vit-large-patch14",
        "ready": True,
        "size_mb": 6529.4,
        "priority": 9,
        "checkpoints": [{'name': 'pytorch_model.bin', 'path': 'pytorch_model.bin', 'size_mb': 1631.4, 'type': '.bin'}],  # 상위 3개만
        "total_checkpoints": 1
    },
    "clip_vit_base": {
        "name": "clip_vit_base",
        "type": "text_image",
        "step": "auxiliary",
        "path": CHECKPOINTS_ROOT / "clip_vit_base",
        "ready": True,
        "size_mb": 1735.3,
        "priority": 10,
        "checkpoints": [{'name': 'pytorch_model.bin', 'path': 'pytorch_model.bin', 'size_mb': 577.2, 'type': '.bin'}],  # 상위 3개만
        "total_checkpoints": 1
    },
    "cloth_segmentation": {
        "name": "cloth_segmentation",
        "type": "human_parsing",
        "step": "step_01_human_parsing",
        "path": CHECKPOINTS_ROOT / "cloth_segmentation",
        "ready": True,
        "size_mb": 803.2,
        "priority": 5,
        "checkpoints": [{'name': 'model.pth', 'path': 'model.pth', 'size_mb': 168.5, 'type': '.pth'}, {'name': 'pytorch_model.bin', 'path': 'pytorch_model.bin', 'size_mb': 168.4, 'type': '.bin'}],  # 상위 3개만
        "total_checkpoints": 2
    },
    "grounding_dino": {
        "name": "grounding_dino",
        "type": "virtual_tryon",
        "step": "step_06_virtual_fitting",
        "path": CHECKPOINTS_ROOT / "grounding_dino",
        "ready": True,
        "size_mb": 1318.2,
        "priority": 3,
        "checkpoints": [{'name': 'pytorch_model.bin', 'path': 'pytorch_model.bin', 'size_mb': 659.9, 'type': '.bin'}],  # 상위 3개만
        "total_checkpoints": 1
    },
    "human_parsing": {
        "name": "human_parsing",
        "type": "human_parsing",
        "step": "step_01_human_parsing",
        "path": CHECKPOINTS_ROOT / "human_parsing",
        "ready": True,
        "size_mb": 1288.3,
        "priority": 4,
        "checkpoints": [{'name': 'schp_atr.pth', 'path': 'schp_atr.pth', 'size_mb': 255.1, 'type': '.pth'}, {'name': 'rng_state.pth', 'path': 'rng_state.pth', 'size_mb': 0.0, 'type': '.pth'}, {'name': 'optimizer.pt', 'path': 'optimizer.pt', 'size_mb': 209.0, 'type': '.pt'}],  # 상위 3개만
        "total_checkpoints": 8
    },
    "ootdiffusion": {
        "name": "ootdiffusion",
        "type": "diffusion",
        "step": "step_06_virtual_fitting",
        "path": CHECKPOINTS_ROOT / "ootdiffusion",
        "ready": True,
        "size_mb": 15129.3,
        "priority": 1,
        "checkpoints": [{'name': 'pytorch_model.bin', 'path': 'checkpoints/ootd/text_encoder/pytorch_model.bin', 'size_mb': 469.5, 'type': '.bin'}, {'name': 'diffusion_pytorch_model.bin', 'path': 'checkpoints/ootd/vae/diffusion_pytorch_model.bin', 'size_mb': 319.2, 'type': '.bin'}, {'name': 'body_pose_model.pth', 'path': 'checkpoints/openpose/ckpts/body_pose_model.pth', 'size_mb': 199.6, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 5
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
    "openpose": {
        "name": "openpose",
        "type": "pose_estimation",
        "step": "step_02_pose_estimation",
        "path": CHECKPOINTS_ROOT / "openpose",
        "ready": True,
        "size_mb": 539.7,
        "priority": 6,
        "checkpoints": [{'name': 'body_pose_model.pth', 'path': 'body_pose_model.pth', 'size_mb': 199.6, 'type': '.pth'}, {'name': 'hand_pose_model.pth', 'path': 'hand_pose_model.pth', 'size_mb': 140.5, 'type': '.pth'}, {'name': 'body_pose_model.pth', 'path': 'ckpts/body_pose_model.pth', 'size_mb': 199.6, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 3
    },
    "pose_estimation": {
        "name": "pose_estimation",
        "type": "pose_estimation",
        "step": "step_02_pose_estimation",
        "path": CHECKPOINTS_ROOT / "pose_estimation",
        "ready": True,
        "size_mb": 10095.6,
        "priority": 4,
        "checkpoints": [{'name': 'sk_model.pth', 'path': 'sk_model.pth', 'size_mb': 16.4, 'type': '.pth'}, {'name': 'upernet_global_small.pth', 'path': 'upernet_global_small.pth', 'size_mb': 196.8, 'type': '.pth'}, {'name': 'latest_net_G.pth', 'path': 'latest_net_G.pth', 'size_mb': 303.5, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 23
    },
    "sam": {
        "name": "sam",
        "type": "virtual_tryon",
        "step": "step_06_virtual_fitting",
        "path": CHECKPOINTS_ROOT / "sam",
        "ready": True,
        "size_mb": 2445.7,
        "priority": 3,
        "checkpoints": [{'name': 'sam_vit_h_4b8939.pth', 'path': 'sam_vit_h_4b8939.pth', 'size_mb': 2445.7, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 1
    },
    "shared_encoder": {
        "name": "shared_encoder",
        "type": "virtual_tryon",
        "step": "step_06_virtual_fitting",
        "path": CHECKPOINTS_ROOT / "shared_encoder",
        "ready": True,
        "size_mb": 1735.3,
        "priority": 3,
        "checkpoints": [{'name': 'pytorch_model.bin', 'path': 'clip-vit-base-patch32/pytorch_model.bin', 'size_mb': 577.2, 'type': '.bin'}],  # 상위 3개만
        "total_checkpoints": 1
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
    "step_01": {
        "name": "step_01",
        "type": "human_parsing",
        "step": "step_01_human_parsing",
        "path": CHECKPOINTS_ROOT / "step_01",
        "ready": True,
        "size_mb": 255.1,
        "priority": 5,
        "checkpoints": [{'name': 'exp-schp-201908301523-atr.pth', 'path': 'schp_human_parsing/exp-schp-201908301523-atr.pth', 'size_mb': 255.1, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 1
    },
    "step_01_human_parsing": {
        "name": "step_01_human_parsing",
        "type": "human_parsing",
        "step": "step_01_human_parsing",
        "path": CHECKPOINTS_ROOT / "step_01_human_parsing",
        "ready": True,
        "size_mb": 1787.7,
        "priority": 4,
        "checkpoints": [{'name': 'densepose_rcnn_R_50_FPN_s1x.pkl', 'path': 'densepose_rcnn_R_50_FPN_s1x.pkl', 'size_mb': 243.9, 'type': '.pkl'}, {'name': 'lightweight_parsing.pth', 'path': 'lightweight_parsing.pth', 'size_mb': 0.5, 'type': '.pth'}, {'name': 'graphonomy_lip.pth', 'path': 'graphonomy_lip.pth', 'size_mb': 255.1, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 11
    },
    "step_02_pose_estimation": {
        "name": "step_02_pose_estimation",
        "type": "pose_estimation",
        "step": "step_02_pose_estimation",
        "path": CHECKPOINTS_ROOT / "step_02_pose_estimation",
        "ready": True,
        "size_mb": 273.6,
        "priority": 6,
        "checkpoints": [{'name': 'openpose.pth', 'path': 'openpose.pth', 'size_mb': 199.6, 'type': '.pth'}, {'name': 'yolov8n-pose.pt', 'path': 'yolov8n-pose.pt', 'size_mb': 6.5, 'type': '.pt'}],  # 상위 3개만
        "total_checkpoints": 2
    },
    "step_03": {
        "name": "step_03",
        "type": "human_parsing",
        "step": "step_01_human_parsing",
        "path": CHECKPOINTS_ROOT / "step_03",
        "ready": True,
        "size_mb": 168.1,
        "priority": 5,
        "checkpoints": [{'name': 'u2net.pth', 'path': 'u2net_segmentation/u2net.pth', 'size_mb': 168.1, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 1
    },
    "step_03_cloth_segmentation": {
        "name": "step_03_cloth_segmentation",
        "type": "human_parsing",
        "step": "step_01_human_parsing",
        "path": CHECKPOINTS_ROOT / "step_03_cloth_segmentation",
        "ready": True,
        "size_mb": 374.8,
        "priority": 5,
        "checkpoints": [{'name': 'mobile_sam.pt', 'path': 'mobile_sam.pt', 'size_mb': 38.8, 'type': '.pt'}, {'name': 'u2net.pth', 'path': 'u2net.pth', 'size_mb': 168.1, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 2
    },
    "step_04": {
        "name": "step_04",
        "type": "auxiliary",
        "step": "auxiliary",
        "path": CHECKPOINTS_ROOT / "step_04",
        "ready": True,
        "size_mb": 20.9,
        "priority": 102,
        "checkpoints": [{'name': 'geometric_matching_base.pth', 'path': 'step_04_geometric_matching_base/geometric_matching_base.pth', 'size_mb': 18.7, 'type': '.pth'}, {'name': 'tps_network.pth', 'path': 'step_04_tps_network/tps_network.pth', 'size_mb': 2.1, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 2
    },
    "step_04_geometric_matching": {
        "name": "step_04_geometric_matching",
        "type": "geometric_matching",
        "step": "step_04_geometric_matching",
        "path": CHECKPOINTS_ROOT / "step_04_geometric_matching",
        "ready": True,
        "size_mb": 33.2,
        "priority": 9,
        "checkpoints": [{'name': 'gmm_final.pth', 'path': 'gmm_final.pth', 'size_mb': 4.1, 'type': '.pth'}, {'name': 'lightweight_gmm.pth', 'path': 'lightweight_gmm.pth', 'size_mb': 4.1, 'type': '.pth'}, {'name': 'tps_network.pth', 'path': 'tps_transformation_model/tps_network.pth', 'size_mb': 2.1, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 4
    },
    "step_05_cloth_warping": {
        "name": "step_05_cloth_warping",
        "type": "cloth_warping",
        "step": "step_05_cloth_warping",
        "path": CHECKPOINTS_ROOT / "step_05_cloth_warping",
        "ready": True,
        "size_mb": 3279.2,
        "priority": 8,
        "checkpoints": [{'name': 'tom_final.pth', 'path': 'tom_final.pth', 'size_mb': 3279.1, 'type': '.pth'}, {'name': 'lightweight_warping.pth', 'path': 'lightweight_warping.pth', 'size_mb': 0.1, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 2
    },
    "step_06_virtual_fitting": {
        "name": "step_06_virtual_fitting",
        "type": "virtual_tryon",
        "step": "step_06_virtual_fitting",
        "path": CHECKPOINTS_ROOT / "step_06_virtual_fitting",
        "ready": True,
        "size_mb": 20854.2,
        "priority": 2,
        "checkpoints": [{'name': 'hrviton_final.pth', 'path': 'hrviton_final.pth', 'size_mb': 2445.7, 'type': '.pth'}, {'name': 'diffusion_pytorch_model.bin', 'path': 'diffusion_pytorch_model.bin', 'size_mb': 3279.1, 'type': '.bin'}, {'name': 'pytorch_model.bin', 'path': 'ootdiffusion/checkpoints/ootd/text_encoder/pytorch_model.bin', 'size_mb': 469.5, 'type': '.bin'}],  # 상위 3개만
        "total_checkpoints": 7
    },
    "step_07_post_processing": {
        "name": "step_07_post_processing",
        "type": "auxiliary",
        "step": "auxiliary",
        "path": CHECKPOINTS_ROOT / "step_07_post_processing",
        "ready": True,
        "size_mb": 63.9,
        "priority": 102,
        "checkpoints": [{'name': 'RealESRGAN_x4plus.pth', 'path': 'RealESRGAN_x4plus.pth', 'size_mb': 63.9, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 1
    },
    "u2net": {
        "name": "u2net",
        "type": "cloth_segmentation",
        "step": "step_03_cloth_segmentation",
        "path": CHECKPOINTS_ROOT / "u2net",
        "ready": True,
        "size_mb": 168.1,
        "priority": 7,
        "checkpoints": [{'name': 'u2net.pth', 'path': 'u2net.pth', 'size_mb': 168.1, 'type': '.pth'}],  # 상위 3개만
        "total_checkpoints": 1
    },
}

# 단계별 최적 모델 매핑
STEP_OPTIMAL_MODELS = {
    "step_06_virtual_fitting": "ootdiffusion",
    "step_03_cloth_segmentation": "background_removal",
    "auxiliary": "clip-vit-large-patch14",
    "step_01_human_parsing": "human_parsing",
    "step_02_pose_estimation": "pose_estimation",
    "step_04_geometric_matching": "step_04_geometric_matching",
    "step_05_cloth_warping": "step_05_cloth_warping",
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
