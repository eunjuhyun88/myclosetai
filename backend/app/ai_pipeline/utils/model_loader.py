"""모델 로딩 유틸리티"""
import torch
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, device="mps", use_fp16=True):
        self.device = torch.device(device)
        self.use_fp16 = use_fp16
        self.loaded_models = {}
    
    def load_model(self, model_name, model_path=None):
        """더미 모델 로드"""
        logger.info(f"더미 모델 로드: {model_name}")
        
        class DummyModel:
            def __init__(self, name):
                self.name = name
            
            def __call__(self, *args, **kwargs):
                return {"result": f"dummy_{self.name}", "success": True}
        
        model = DummyModel(model_name)
        self.loaded_models[model_name] = model
        return model
