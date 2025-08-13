#!/usr/bin/env python3
"""
🔥 2단계 Pose Estimation 체크포인트 경로 설정
================================================

✅ YOLOv8 Pose 체크포인트
✅ OpenPose 체크포인트  
✅ HRNet 체크포인트
✅ MediaPipe 체크포인트
"""

import os
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path("/Users/gimdudeul/MVP/mycloset-ai")
BACKEND_ROOT = PROJECT_ROOT / "backend"

# 체크포인트 경로들
CHECKPOINT_PATHS = {
    # YOLOv8 Pose 체크포인트들
    'yolov8': {
        'yolov8n-pose': BACKEND_ROOT / "yolov8n-pose.pt",
        'yolov8m-pose': BACKEND_ROOT / "ai_models" / "step_02_pose_estimation" / "yolov8m-pose.pt",
        'yolov8s-pose': None,  # 아직 없음
        'yolov8l-pose': None,  # 아직 없음
        'yolov8x-pose': None,  # 아직 없음
    },
    
    # OpenPose 체크포인트들
    'openpose': {
        'openpose': BACKEND_ROOT / "ai_models" / "openpose.pth",
        'body_pose_model': BACKEND_ROOT / "ai_models" / "step_02" / "body_pose_model.pth",
        'openpose_step02': BACKEND_ROOT / "ai_models" / "step_02" / "openpose.pth",
        'body_pose_virtual_fitting': BACKEND_ROOT / "ai_models" / "step_06_virtual_fitting" / "ootdiffusion" / "checkpoints" / "openpose" / "ckpts" / "body_pose_model.pth",
    },
    
    # HRNet 체크포인트들
    'hrnet': {
        'hrnet': BACKEND_ROOT / "ai_models" / "step_02" / "hrnet.pth",
        'hrnet_w48_coco': BACKEND_ROOT / "ai_models" / "step_02" / "hrnet_w48_coco_256x192.pth",
        'hrnet_w32_coco': None,  # 아직 없음
        'hrnet_w64_coco': None,  # 아직 없음
    },
    
    # MediaPipe 체크포인트들 (MediaPipe는 보통 사전 훈련된 모델을 다운로드)
    'mediapipe': {
        'blazepose': None,  # MediaPipe는 자동 다운로드
        'pose_landmarker': None,  # MediaPipe는 자동 다운로드
    },
    
    # 추가 포즈 추정 모델들
    'additional': {
        'dpt_large': BACKEND_ROOT / "ai_models" / "checkpoints" / "pose_estimation" / "dpt_large-501f0c75.pt",
        'dpt_hybrid_midas': BACKEND_ROOT / "ai_models" / "checkpoints" / "pose_estimation" / "dpt_hybrid-midas-501f0c75.pt",
    }
}

def get_checkpoint_path(model_type: str, model_name: str = None) -> str:
    """체크포인트 경로 반환"""
    if model_type not in CHECKPOINT_PATHS:
        return None
    
    if model_name is None:
        # 첫 번째 사용 가능한 체크포인트 반환
        for name, path in CHECKPOINT_PATHS[model_type].items():
            if path and path.exists():
                return str(path)
        return None
    
    if model_name not in CHECKPOINT_PATHS[model_type]:
        return None
    
    path = CHECKPOINT_PATHS[model_type][model_name]
    if path and path.exists():
        return str(path)
    return None

def get_available_checkpoints() -> dict:
    """사용 가능한 체크포인트 목록 반환"""
    available = {}
    
    for model_type, models in CHECKPOINT_PATHS.items():
        available[model_type] = {}
        for model_name, path in models.items():
            if path and path.exists():
                available[model_type][model_name] = str(path)
    
    return available

def check_checkpoint_exists(model_type: str, model_name: str) -> bool:
    """체크포인트 존재 여부 확인"""
    path = get_checkpoint_path(model_type, model_name)
    return path is not None

if __name__ == "__main__":
    print("🔍 사용 가능한 체크포인트들:")
    available = get_available_checkpoints()
    
    for model_type, models in available.items():
        print(f"\n📁 {model_type.upper()}:")
        for model_name, path in models.items():
            print(f"   ✅ {model_name}: {path}")
    
    print(f"\n📊 총 체크포인트 수: {sum(len(models) for models in available.values())}")
