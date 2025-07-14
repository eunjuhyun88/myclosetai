# app/ai_pipeline/utils/model_registry_update.py
"""
스캔된 모델들을 ModelLoader에 자동 등록
"""

from app.ai_pipeline.utils.model_loader import ModelConfig, ModelType
from pathlib import Path

def update_model_registry(model_loader):
    """스캔된 모델들을 ModelLoader에 등록"""
    
    # 기본 경로
    ai_models_root = Path("ai_models")
    
    # 스캔된 모델들 등록

    # OpenPose
    model_loader.register_model(
        "openpose",
        ModelConfig(
            name="OpenPose",
            model_type=ModelType.POSE_ESTIMATION,
            model_class="OpenPoseModel",
            checkpoint_path=str(ai_models_root / "openpose"),
            input_size=(512, 512),
            device="mps"
        )
    )

def get_available_models():
    """사용 가능한 모델 목록 반환"""
    return ["openpose", "checkpoints", ]

# 사용 예시:
# from app.ai_pipeline.utils.model_loader import ModelLoader
# from app.ai_pipeline.utils.model_registry_update import update_model_registry
# 
# loader = ModelLoader()
# update_model_registry(loader)
# model = await loader.load_model("ootdiffusion")
