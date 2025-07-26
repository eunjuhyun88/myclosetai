#!/usr/bin/env python3
"""
올바른 경로 설정 (자동 생성)
"""
import os
from pathlib import Path

# 올바른 경로들
PROJECT_ROOT = Path("/Users/gimdudeul/MVP/mycloset-ai")
BACKEND_ROOT = PROJECT_ROOT / "backend"  
AI_MODELS_ROOT = BACKEND_ROOT / "ai_models"

# Step 04 모델 경로들  
STEP04_MODELS = {
    "gmm_final": AI_MODELS_ROOT / "step_04_geometric_matching" / "gmm_final.pth",
    "tps_network": AI_MODELS_ROOT / "step_04_geometric_matching" / "tps_network.pth",
    "sam_shared": AI_MODELS_ROOT / "step_03_cloth_segmentation" / "sam_vit_h_4b8939.pth"
}

def get_step04_model_path(model_name: str):
    """Step 04 모델 경로 반환"""
    return str(STEP04_MODELS.get(model_name, ""))

def patch_geometric_matching_step():
    """GeometricMatchingStep 경로 패치"""
    import sys
    sys.path.insert(0, str(BACKEND_ROOT))
    
    # 환경 변수로 올바른 경로 설정
    os.environ["AI_MODELS_ROOT"] = str(AI_MODELS_ROOT)
    os.environ["STEP04_GMM_PATH"] = str(STEP04_MODELS["gmm_final"])
    os.environ["STEP04_TPS_PATH"] = str(STEP04_MODELS["tps_network"])
    
    print(f"✅ AI_MODELS_ROOT: {AI_MODELS_ROOT}")
    print(f"✅ GMM 경로: {STEP04_MODELS['gmm_final']}")
    
if __name__ == "__main__":
    patch_geometric_matching_step()
    
    # 파일 존재 확인
    print("\n📊 모델 파일 확인:")
    for name, path in STEP04_MODELS.items():
        exists = Path(path).exists()
        print(f"  {'✅' if exists else '❌'} {name}: {path}")
