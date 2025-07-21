# app/ai_pipeline/utils/checkpoint_model_loader.py
"""
체크포인트 분석 기반 ModelLoader 완전 연동
실제 다운로드된 127.2GB 체크포인트들 활용
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

try:
    from app.core.optimized_model_paths import (
        ANALYZED_MODELS, get_optimal_model_for_step, 
        get_checkpoint_path, get_largest_checkpoint
    )
    OPTIMIZED_PATHS_AVAILABLE = True
except ImportError:
    OPTIMIZED_PATHS_AVAILABLE = False

logger = logging.getLogger(__name__)

class CheckpointModelLoader:
    """체크포인트 분석 기반 모델 로더"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.models = {}
        self.loaded_models = {}
        
        if OPTIMIZED_PATHS_AVAILABLE:
            self._register_analyzed_models()
        else:
            logger.warning("⚠️ 최적화된 모델 경로가 없습니다")
    
    def _setup_device(self, device: str) -> str:
        """디바이스 설정"""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _register_analyzed_models(self):
        """분석된 체크포인트 모델들 등록"""
        if not OPTIMIZED_PATHS_AVAILABLE:
            return
            
        registered_count = 0
        
        for model_name, model_info in ANALYZED_MODELS.items():
            if not model_info["ready"]:
                continue
            
            try:
                # 모델 정보 등록
                self.models[model_name] = {
                    "name": model_info["name"],
                    "type": model_info["type"],
                    "step": model_info["step"],
                    "path": model_info["path"],
                    "checkpoints": model_info["checkpoints"],
                    "size_mb": model_info["size_mb"],
                    "priority": model_info["priority"]
                }
                
                registered_count += 1
                
            except Exception as e:
                logger.warning(f"   ⚠️ {model_name} 등록 실패: {e}")
        
        logger.info(f"📦 {registered_count}개 체크포인트 모델 등록 완료")
    
    async def load_model(self, model_name: str, **kwargs) -> Optional[Any]:
        """모델 로드"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        if model_name not in self.models:
            logger.warning(f"⚠️ 등록되지 않은 모델: {model_name}")
            return None
        
        try:
            model_info = self.models[model_name]
            
            # 가장 큰 체크포인트 경로 찾기
            largest_checkpoint = get_largest_checkpoint(model_name)
            if not largest_checkpoint:
                logger.warning(f"⚠️ {model_name}의 체크포인트를 찾을 수 없습니다")
                return None
            
            checkpoint_path = get_checkpoint_path(model_name, largest_checkpoint)
            
            if not checkpoint_path or not checkpoint_path.exists():
                logger.warning(f"⚠️ {model_name}의 체크포인트 파일이 없습니다: {checkpoint_path}")
                return None
            
            # PyTorch 모델 로드
            logger.info(f"🔧 {model_name} 로딩 중... ({checkpoint_path})")
            
            # 안전한 로드
            try:
                model = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            except:
                # weights_only가 지원되지 않는 경우 폴백
                model = torch.load(checkpoint_path, map_location=self.device)
            
            # 모델 정리 및 디바이스 이동
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif 'state_dict' in model:
                    model = model['state_dict']
            
            # 캐시에 저장
            self.loaded_models[model_name] = model
            
            logger.info(f"✅ {model_name} 로딩 완료")
            return model
            
        except Exception as e:
            logger.error(f"❌ {model_name} 로딩 실패: {e}")
            return None
    
    async def load_optimal_model_for_step(self, step: str, **kwargs) -> Optional[Any]:
        """단계별 최적 모델 로드"""
        optimal_model = get_optimal_model_for_step(step)
        if not optimal_model:
            logger.warning(f"⚠️ {step}에 대한 최적 모델을 찾을 수 없음")
            return None
        
        logger.info(f"🎯 {step} 최적 모델 로드: {optimal_model}")
        return await self.load_model(optimal_model, **kwargs)
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """모델 정보 반환"""
        return self.models.get(model_name)
    
    def list_models(self) -> Dict[str, Dict]:
        """등록된 모델 목록"""
        return self.models.copy()
    
    def clear_cache(self):
        """모델 캐시 정리"""
        self.loaded_models.clear()
        
        if self.device == "mps" and torch.backends.mps.is_available():
            safe_mps_empty_cache()
        elif self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🧹 모델 캐시 정리 완료")

# 전역 모델 로더
_global_checkpoint_loader: Optional[CheckpointModelLoader] = None

def get_checkpoint_model_loader(**kwargs) -> CheckpointModelLoader:
    """전역 체크포인트 모델 로더 반환"""
    global _global_checkpoint_loader
    if _global_checkpoint_loader is None:
        _global_checkpoint_loader = CheckpointModelLoader(**kwargs)
    return _global_checkpoint_loader

async def load_best_model_for_step(step: str, **kwargs) -> Optional[Any]:
    """단계별 최고 성능 모델 로드"""
    loader = get_checkpoint_model_loader()
    return await loader.load_optimal_model_for_step(step, **kwargs)

# 빠른 접근 함수들
async def load_best_diffusion_model(**kwargs) -> Optional[Any]:
    """최고 성능 Diffusion 모델 로드"""
    return await load_best_model_for_step("step_06_virtual_fitting", **kwargs)

async def load_best_human_parsing_model(**kwargs) -> Optional[Any]:
    """최고 성능 인체 파싱 모델 로드"""
    return await load_best_model_for_step("step_01_human_parsing", **kwargs)

async def load_best_pose_model(**kwargs) -> Optional[Any]:
    """최고 성능 포즈 추정 모델 로드"""
    return await load_best_model_for_step("step_02_pose_estimation", **kwargs)

async def load_best_cloth_segmentation_model(**kwargs) -> Optional[Any]:
    """최고 성능 의류 분할 모델 로드"""
    return await load_best_model_for_step("step_03_cloth_segmentation", **kwargs)
