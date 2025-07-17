"""
MyCloset AI - 최적화된 통합 모델 로더
M3 Max 128GB 전용 최적화 버전
"""

import json
import torch
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedModelLoader:
    """M3 Max 최적화 모델 로더"""
    
    def __init__(self, config_path: str = "model_scan_results.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.loaded_models = {}
        self.load_times = {}
        
        logger.info(f"🍎 M3 Max 모델 로더 초기화: {self.device}")
        logger.info(f"📦 관리 대상 모델: {len(self.config['path_mappings'])}개")
    
    async def load_model_async(self, model_name: str) -> Optional[torch.nn.Module]:
        """비동기 모델 로딩"""
        if model_name in self.loaded_models:
            logger.info(f"🔄 캐시된 모델 반환: {model_name}")
            return self.loaded_models[model_name]
        
        if model_name not in self.config['path_mappings']:
            logger.error(f"❌ 모델 경로 없음: {model_name}")
            return None
        
        model_path = self.config['path_mappings'][model_name]
        
        try:
            start_time = time.time()
            logger.info(f"🔄 모델 로딩 시작: {model_name}")
            
            # M3 Max MPS 최적화 로딩
            if self.device.type == 'mps':
                # CPU에서 로드 후 MPS로 이동 (안전한 방법)
                model = torch.load(model_path, map_location='cpu', weights_only=False)
                if hasattr(model, 'to'):
                    model = model.to(self.device)
            else:
                model = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 평가 모드 설정
            if hasattr(model, 'eval'):
                model.eval()
            
            # 메모리 최적화
            if hasattr(model, 'half') and self.device.type == 'mps':
                # MPS에서 float16 사용 (메모리 절약)
                try:
                    model = model.half()
                except:
                    logger.warning(f"⚠️ float16 변환 실패, float32 유지: {model_name}")
            
            load_time = time.time() - start_time
            self.loaded_models[model_name] = model
            self.load_times[model_name] = load_time
            
            logger.info(f"✅ 모델 로딩 완료: {model_name} ({load_time:.2f}초)")
            return model
            
        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패 {model_name}: {e}")
            return None
    
    def unload_model(self, model_name: str):
        """모델 언로드 (메모리 절약)"""
        if model_name in self.loaded_models:
            model = self.loaded_models[model_name]
            if hasattr(model, 'cpu'):
                model.cpu()
            del self.loaded_models[model_name]
            logger.info(f"🗑️ 모델 언로드: {model_name}")
    
    def get_step_model(self, step: str) -> Optional[str]:
        """단계별 권장 모델 반환"""
        for model_name, model_info in self.config['selected_models'].items():
            if model_info['step'] == step:
                return model_name
        return None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """메모리 사용량 조회"""
        if self.device.type == 'mps':
            try:
                total_memory = torch.mps.current_allocated_memory() / (1024**2)  # MB
                return {"mps_memory_mb": total_memory}
            except:
                return {"mps_memory_mb": 0}
        return {"cpu_memory": "N/A"}
    
    def get_statistics(self) -> Dict[str, Any]:
        """로더 통계"""
        return {
            "device": str(self.device),
            "loaded_models": len(self.loaded_models),
            "available_models": len(self.config['path_mappings']),
            "load_times": self.load_times,
            "memory_usage": self.get_memory_usage()
        }

# 전역 로더 인스턴스
_global_loader = None

def get_model_loader() -> OptimizedModelLoader:
    """전역 모델 로더 반환"""
    global _global_loader
    if _global_loader is None:
        _global_loader = OptimizedModelLoader()
    return _global_loader

async def load_essential_models():
    """필수 모델들 사전 로딩"""
    loader = get_model_loader()
    
    essential_steps = [
        'step_01_human_parsing',
        'step_02_pose_estimation', 
        'step_03_cloth_segmentation',
        'step_06_virtual_fitting'
    ]
    
    loaded_count = 0
    for step in essential_steps:
        model_name = loader.get_step_model(step)
        if model_name:
            model = await loader.load_model_async(model_name)
            if model:
                loaded_count += 1
    
    logger.info(f"✅ 필수 모델 로딩 완료: {loaded_count}/{len(essential_steps)}")
    return loaded_count

if __name__ == "__main__":
    # 테스트 실행
    async def test_loader():
        loader = get_model_loader()
        print("📊 모델 로더 통계:")
        stats = loader.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n🔄 필수 모델 로딩 테스트:")
        loaded_count = await load_essential_models()
        
        print(f"\n✅ 테스트 완료: {loaded_count}개 모델 로딩 성공")
    
    asyncio.run(test_loader())
