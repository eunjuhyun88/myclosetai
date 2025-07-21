# backend/app/core/optimized_model_paths.py
"""
최적화된 AI 모델 경로 설정 - 체크포인트 분석 기반 (SyntaxError 완전 수정)
실제 사용 가능한 체크포인트들로만 구성
생성일: 2025-07-21
분석된 모델: 17개
총 크기: 116.1GB
"""

from pathlib import Path
from typing import Dict, Optional, List, Any

# 기본 경로
AI_MODELS_ROOT = Path(__file__).parent.parent.parent / "ai_models"
CHECKPOINTS_ROOT = AI_MODELS_ROOT / "checkpoints"

# 분석된 체크포인트 모델들
ANALYZED_MODELS = {
    "ootdiffusion": {
        "name": "OOTDiffusion",
        "type": "diffusion",
        "step": "step_06_virtual_fitting",
        "path": CHECKPOINTS_ROOT / "ootdiffusion",
        "ready": True,
        "size_mb": 15129.3,
        "total_size_mb": 15129.3,
        "priority": 1,
        "checkpoints": [
            {'name': 'pytorch_model.bin', 'path': 'pytorch_model.bin', 'size_mb': 469.5}, 
            {'name': 'diffusion_pytorch_model.bin', 'path': 'diffusion_pytorch_model.bin', 'size_mb': 319.2}, 
            {'name': 'body_pose_model.pth', 'path': 'body_pose_model.pth', 'size_mb': 199.6}
        ],
        "total_checkpoints": 5
    },
    "ootdiffusion_hf": {
        "name": "OOTDiffusion HF",
        "type": "diffusion",
        "step": "step_06_virtual_fitting",
        "path": CHECKPOINTS_ROOT / "ootdiffusion_hf",
        "ready": True,
        "size_mb": 15129.3,
        "total_size_mb": 15129.3,
        "priority": 1,
        "checkpoints": [
            {'name': 'pytorch_model.bin', 'path': 'pytorch_model.bin', 'size_mb': 469.5}, 
            {'name': 'diffusion_pytorch_model.bin', 'path': 'diffusion_pytorch_model.bin', 'size_mb': 319.2}, 
            {'name': 'body_pose_model.pth', 'path': 'body_pose_model.pth', 'size_mb': 199.6}
        ],
        "total_checkpoints": 5
    },
    "stable-diffusion-v1-5": {
        "name": "Stable Diffusion v1.5",
        "type": "diffusion",
        "step": "step_06_virtual_fitting",
        "path": CHECKPOINTS_ROOT / "stable-diffusion-v1-5",
        "ready": True,
        "size_mb": 45070.6,
        "total_size_mb": 45070.6,
        "priority": 2,
        "checkpoints": [
            {'name': 'v1-5-pruned.ckpt', 'path': 'v1-5-pruned.ckpt', 'size_mb': 7346.9}, 
            {'name': 'v1-5-pruned-emaonly.ckpt', 'path': 'v1-5-pruned-emaonly.ckpt', 'size_mb': 4067.8}, 
            {'name': 'pytorch_model.fp16.bin', 'path': 'pytorch_model.fp16.bin', 'size_mb': 234.8}
        ],
        "total_checkpoints": 11
    },
    "human_parsing": {
        "name": "Human Parsing",
        "type": "human_parsing",
        "step": "step_01_human_parsing",
        "path": CHECKPOINTS_ROOT / "human_parsing",
        "ready": True,
        "size_mb": 1288.3,
        "total_size_mb": 1288.3,
        "priority": 3,
        "checkpoints": [
            {'name': 'schp_atr.pth', 'path': 'schp_atr.pth', 'size_mb': 255.1}, 
            {'name': 'optimizer.pt', 'path': 'optimizer.pt', 'size_mb': 209.0}, 
            {'name': 'rng_state.pth', 'path': 'rng_state.pth', 'size_mb': 0.0}
        ],
        "total_checkpoints": 8
    },
    "step_01_human_parsing": {
        "name": "Step 01 Human Parsing",
        "type": "human_parsing",
        "step": "step_01_human_parsing",
        "path": CHECKPOINTS_ROOT / "step_01_human_parsing",
        "ready": True,
        "size_mb": 1787.7,
        "total_size_mb": 1787.7,
        "priority": 3,
        "checkpoints": [
            {'name': 'densepose_rcnn_R_50_FPN_s1x.pkl', 'path': 'densepose_rcnn_R_50_FPN_s1x.pkl', 'size_mb': 243.9}, 
            {'name': 'graphonomy_lip.pth', 'path': 'graphonomy_lip.pth', 'size_mb': 255.1}, 
            {'name': 'lightweight_parsing.pth', 'path': 'lightweight_parsing.pth', 'size_mb': 0.5}
        ],
        "total_checkpoints": 11
    },
    "pose_estimation": {
        "name": "Pose Estimation",
        "type": "pose_estimation",
        "step": "step_02_pose_estimation",
        "path": CHECKPOINTS_ROOT / "pose_estimation",
        "ready": True,
        "size_mb": 10095.6,
        "total_size_mb": 10095.6,
        "priority": 4,
        "checkpoints": [
            {'name': 'sk_model.pth', 'path': 'sk_model.pth', 'size_mb': 16.4}, 
            {'name': 'upernet_global_small.pth', 'path': 'upernet_global_small.pth', 'size_mb': 196.8}, 
            {'name': 'latest_net_G.pth', 'path': 'latest_net_G.pth', 'size_mb': 303.5}
        ],
        "total_checkpoints": 23
    },
    "step_02_pose_estimation": {
        "name": "Step 02 Pose Estimation",
        "type": "pose_estimation",
        "step": "step_02_pose_estimation",
        "path": CHECKPOINTS_ROOT / "step_02_pose_estimation",
        "ready": True,
        "size_mb": 273.6,
        "total_size_mb": 273.6,
        "priority": 4,
        "checkpoints": [
            {'name': 'openpose.pth', 'path': 'openpose.pth', 'size_mb': 199.6}, 
            {'name': 'yolov8n-pose.pt', 'path': 'yolov8n-pose.pt', 'size_mb': 6.5}
        ],
        "total_checkpoints": 2
    },
    "openpose": {
        "name": "OpenPose",
        "type": "pose_estimation",
        "step": "step_02_pose_estimation",
        "path": CHECKPOINTS_ROOT / "openpose",
        "ready": True,
        "size_mb": 539.7,
        "total_size_mb": 539.7,
        "priority": 4,
        "checkpoints": [
            {'name': 'body_pose_model.pth', 'path': 'body_pose_model.pth', 'size_mb': 199.6}, 
            {'name': 'hand_pose_model.pth', 'path': 'hand_pose_model.pth', 'size_mb': 140.5}
        ],
        "total_checkpoints": 3
    },
    "cloth_segmentation": {
        "name": "Cloth Segmentation",
        "type": "cloth_segmentation",
        "step": "step_03_cloth_segmentation",
        "path": CHECKPOINTS_ROOT / "cloth_segmentation",
        "ready": True,
        "size_mb": 803.2,
        "total_size_mb": 803.2,
        "priority": 5,
        "checkpoints": [
            {'name': 'model.pth', 'path': 'model.pth', 'size_mb': 168.5}, 
            {'name': 'pytorch_model.bin', 'path': 'pytorch_model.bin', 'size_mb': 168.4}
        ],
        "total_checkpoints": 2
    },
    "step_03_cloth_segmentation": {
        "name": "Step 03 Cloth Segmentation",
        "type": "cloth_segmentation",
        "step": "step_03_cloth_segmentation",
        "path": CHECKPOINTS_ROOT / "step_03_cloth_segmentation",
        "ready": True,
        "size_mb": 206.7,
        "total_size_mb": 206.7,
        "priority": 5,
        "checkpoints": [
            {'name': 'mobile_sam.pt', 'path': 'mobile_sam.pt', 'size_mb': 38.8}
        ],
        "total_checkpoints": 1
    },
    "step_04_geometric_matching": {
        "name": "Step 04 Geometric Matching",
        "type": "geometric_matching",
        "step": "step_04_geometric_matching",
        "path": CHECKPOINTS_ROOT / "step_04_geometric_matching",
        "ready": True,
        "size_mb": 33.2,
        "total_size_mb": 33.2,
        "priority": 6,
        "checkpoints": [
            {'name': 'gmm_final.pth', 'path': 'gmm_final.pth', 'size_mb': 4.1}, 
            {'name': 'lightweight_gmm.pth', 'path': 'lightweight_gmm.pth', 'size_mb': 4.1}, 
            {'name': 'tps_network.pth', 'path': 'tps_network.pth', 'size_mb': 2.1}
        ],
        "total_checkpoints": 4
    },
    "step_05_cloth_warping": {
        "name": "Step 05 Cloth Warping",
        "type": "cloth_warping",
        "step": "step_05_cloth_warping",
        "path": CHECKPOINTS_ROOT / "step_05_cloth_warping",
        "ready": True,
        "size_mb": 3279.2,
        "total_size_mb": 3279.2,
        "priority": 7,
        "checkpoints": [
            {'name': 'tom_final.pth', 'path': 'tom_final.pth', 'size_mb': 3279.1}, 
            {'name': 'lightweight_warping.pth', 'path': 'lightweight_warping.pth', 'size_mb': 0.1}
        ],
        "total_checkpoints": 2
    },
    "step_06_virtual_fitting": {
        "name": "Step 06 Virtual Fitting",
        "type": "virtual_tryon",
        "step": "step_06_virtual_fitting",
        "path": CHECKPOINTS_ROOT / "step_06_virtual_fitting",
        "ready": True,
        "size_mb": 20854.2,
        "total_size_mb": 20854.2,
        "priority": 1,
        "checkpoints": [
            {'name': 'hrviton_final.pth', 'path': 'hrviton_final.pth', 'size_mb': 2445.7}, 
            {'name': 'diffusion_pytorch_model.bin', 'path': 'diffusion_pytorch_model.bin', 'size_mb': 3279.1}, 
            {'name': 'pytorch_model.bin', 'path': 'pytorch_model.bin', 'size_mb': 469.5}
        ],
        "total_checkpoints": 7
    },
    "step_07_post_processing": {
        "name": "Step 07 Post Processing",
        "type": "auxiliary",
        "step": "step_07_post_processing",
        "path": CHECKPOINTS_ROOT / "step_07_post_processing",
        "ready": True,
        "size_mb": 63.9,
        "total_size_mb": 63.9,
        "priority": 8,
        "checkpoints": [
            {'name': 'RealESRGAN_x4plus.pth', 'path': 'RealESRGAN_x4plus.pth', 'size_mb': 63.9}
        ],
        "total_checkpoints": 1
    },
    "sam": {
        "name": "SAM (Segment Anything Model)",
        "type": "auxiliary",
        "step": "auxiliary",
        "path": CHECKPOINTS_ROOT / "sam",
        "ready": True,
        "size_mb": 2445.7,
        "total_size_mb": 2445.7,
        "priority": 8,
        "checkpoints": [
            {'name': 'sam_vit_h_4b8939.pth', 'path': 'sam_vit_h_4b8939.pth', 'size_mb': 2445.7}
        ],
        "total_checkpoints": 1
    },
    "clip-vit-base-patch32": {
        "name": "CLIP ViT Base",
        "type": "text_image",
        "step": "auxiliary",
        "path": CHECKPOINTS_ROOT / "clip-vit-base-patch32",
        "ready": True,
        "size_mb": 580.7,
        "total_size_mb": 580.7,
        "priority": 9,
        "checkpoints": [
            {'name': 'pytorch_model.bin', 'path': 'pytorch_model.bin', 'size_mb': 577.2}
        ],
        "total_checkpoints": 1
    },
    "grounding_dino": {
        "name": "Grounding DINO",
        "type": "auxiliary",
        "step": "auxiliary",
        "path": CHECKPOINTS_ROOT / "grounding_dino",
        "ready": True,
        "size_mb": 1318.2,
        "total_size_mb": 1318.2,
        "priority": 9,
        "checkpoints": [
            {'name': 'pytorch_model.bin', 'path': 'pytorch_model.bin', 'size_mb': 659.9}
        ],
        "total_checkpoints": 1
    }
}

# 단계별 최적 모델 매핑
STEP_OPTIMAL_MODELS = {
    "step_01_human_parsing": "step_01_human_parsing",
    "step_02_pose_estimation": "step_02_pose_estimation",
    "step_03_cloth_segmentation": "step_03_cloth_segmentation",
    "step_04_geometric_matching": "step_04_geometric_matching",
    "step_05_cloth_warping": "step_05_cloth_warping",
    "step_06_virtual_fitting": "step_06_virtual_fitting",
    "step_07_post_processing": "step_07_post_processing",
    "auxiliary": "sam",
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
    
    largest = max(checkpoints, key=lambda x: x.get('size_mb', 0))
    return largest.get('path', largest.get('name'))

def get_ready_models_by_type(model_type: str) -> List[str]:
    """타입별 사용 가능한 모델들"""
    return [name for name, info in ANALYZED_MODELS.items() 
            if info["type"] == model_type and info["ready"]]

def get_diffusion_models() -> List[str]:
    """Diffusion 모델들 (OOTDiffusion 등)"""
    return get_ready_models_by_type("diffusion")

def get_virtual_tryon_models() -> List[str]:
    """가상 피팅 모델들"""
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

# ==============================================
# 🔥 사용 통계 - SyntaxError 완전 수정
# ==============================================

def get_analysis_stats() -> Dict[str, Any]:
    """분석 통계 동적 생성 - SyntaxError 해결"""
    try:
        total_models = len(ANALYZED_MODELS)
        ready_models = len([m for m in ANALYZED_MODELS.values() if m.get("ready", False)])
        
        # total_size_mb 키 확인 후 계산
        total_size_gb = 0.0
        for model_info in ANALYZED_MODELS.values():
            if "total_size_mb" in model_info:
                total_size_gb += model_info["total_size_mb"]
            elif "size_mb" in model_info:
                total_size_gb += model_info["size_mb"]
        
        total_size_gb = round(total_size_gb / 1024, 1)
        
        # Step별 모델 수 계산
        steps = set(m.get("step", "unknown") for m in ANALYZED_MODELS.values())
        models_by_step = {}
        for step in steps:
            models_by_step[step] = len([
                m for m in ANALYZED_MODELS.values() 
                if m.get("step") == step
            ])
        
        # 가장 큰 모델 찾기
        largest_model = "unknown"
        try:
            largest_model_item = max(
                ANALYZED_MODELS.items(), 
                key=lambda x: x[1].get("total_size_mb", x[1].get("size_mb", 0))
            )
            largest_model = largest_model_item[0]
        except (ValueError, KeyError):
            pass
        
        return {
            "total_models": total_models,
            "ready_models": ready_models,
            "total_size_gb": total_size_gb,
            "models_by_step": models_by_step,
            "largest_model": largest_model
        }
        
    except Exception as e:
        # 오류 발생 시 기본값 반환
        return {
            "total_models": len(ANALYZED_MODELS),
            "ready_models": len(ANALYZED_MODELS),
            "total_size_gb": 116.1,
            "models_by_step": {},
            "largest_model": "stable-diffusion-v1-5",
            "error": str(e)
        }

# 정적 분석 통계 (호환성을 위해 유지)
ANALYSIS_STATS = get_analysis_stats()

# 빠른 접근 함수들
def get_best_diffusion_model() -> Optional[str]:
    """최고 성능 Diffusion 모델"""
    return get_optimal_model_for_step("step_06_virtual_fitting")

def get_best_human_parsing_model() -> Optional[str]:
    """최고 성능 인체 파싱 모델"""  
    return get_optimal_model_for_step("step_01_human_parsing")

def get_best_pose_model() -> Optional[str]:
    """최고 성능 포즈 추정 모델"""
    return get_optimal_model_for_step("step_02_pose_estimation")

def get_best_cloth_segmentation_model() -> Optional[str]:
    """최고 성능 의류 분할 모델"""
    return get_optimal_model_for_step("step_03_cloth_segmentation")

# ==============================================
# 🔥 추가 유틸리티 함수들
# ==============================================

def validate_model_exists(model_name: str) -> bool:
    """모델 경로 존재 여부 확인"""
    try:
        path = get_model_path(model_name)
        return path is not None and path.exists()
    except Exception:
        return False

def get_available_models() -> List[str]:
    """실제 존재하는 모델들만 반환"""
    available = []
    for model_name in ANALYZED_MODELS.keys():
        if validate_model_exists(model_name):
            available.append(model_name)
    return available

def get_total_size_gb() -> float:
    """전체 모델 크기 (GB)"""
    try:
        total_mb = sum(
            m.get("total_size_mb", m.get("size_mb", 0)) 
            for m in ANALYZED_MODELS.values()
        )
        return round(total_mb / 1024, 1)
    except Exception:
        return 116.1

def debug_model_paths() -> Dict[str, Any]:
    """모델 경로 디버깅 정보"""
    debug_info = {
        "ai_models_root": str(AI_MODELS_ROOT),
        "checkpoints_root": str(CHECKPOINTS_ROOT),
        "ai_models_exists": AI_MODELS_ROOT.exists(),
        "checkpoints_exists": CHECKPOINTS_ROOT.exists(),
        "model_paths": {}
    }
    
    for model_name, model_info in ANALYZED_MODELS.items():
        path = model_info.get("path")
        debug_info["model_paths"][model_name] = {
            "path": str(path) if path else None,
            "exists": path.exists() if path else False,
            "ready": model_info.get("ready", False)
        }
    
    return debug_info

# 모듈 로드 확인
if __name__ == "__main__":
    print("✅ optimized_model_paths.py 로드 완료")
    print(f"📊 분석 통계: {get_analysis_stats()}")
    print(f"💾 전체 크기: {get_total_size_gb()}GB")
    print(f"📁 사용 가능한 모델: {len(get_available_models())}개")