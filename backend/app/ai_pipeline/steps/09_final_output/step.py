# 모델 로더 import (새로운 models 폴더에서)
try:
    from app.ai_pipeline.models.model_loader import ModelLoader
    MODEL_LOADER_AVAILABLE = True
except ImportError:
    try:
        from ..models.model_loader import ModelLoader
        MODEL_LOADER_AVAILABLE = True
    except ImportError:
        MODEL_LOADER_AVAILABLE = False
        ModelLoader = None
