# app/ai_pipeline/utils/checkpoint_model_loader.py
"""
체크포인트 분석 기반 ModelLoader 완전 연동
실제 다운로드된 80GB 체크포인트들 활용
"""

from app.ai_pipeline.utils.model_loader import ModelLoader, ModelConfig, ModelType
from app.core.optimized_model_paths import (
    ANALYZED_MODELS, get_optimal_model_for_step, 
    get_checkpoint_path, get_largest_checkpoint
)
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CheckpointModelLoader(ModelLoader):
    """체크포인트 분석 기반 확장 ModelLoader"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._register_analyzed_models()
    
    def _register_analyzed_models(self):
        """분석된 체크포인트 모델들 자동 등록"""
        logger.info("📦 분석된 체크포인트 모델들 등록 중...")
        
        registered_count = 0
        
        for model_name, model_info in ANALYZED_MODELS.items():
            if not model_info["ready"]:
                continue
                
            try:
                # ModelType 매핑
                model_type = self._map_to_model_type(model_info["type"])
                if not model_type:
                    continue
                
                # 가장 큰 체크포인트 경로
                main_checkpoint = get_largest_checkpoint(model_name)
                checkpoint_path = get_checkpoint_path(model_name, main_checkpoint) if main_checkpoint else None
                
                # 모델 설정 생성
                model_config = ModelConfig(
                    name=model_info["name"],
                    model_type=model_type,
                    model_class=self._get_model_class(model_info["type"]),
                    checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                    input_size=(512, 512),
                    device=self.device
                )
                
                # 모델 등록
                self.register_model(model_name, model_config)
                registered_count += 1
                
                logger.info(f"   ✅ {model_name}: {model_info['name']}")
                
            except Exception as e:
                logger.warning(f"   ⚠️ {model_name} 등록 실패: {e}")
        
        logger.info(f"📦 총 {registered_count}개 체크포인트 모델 등록 완료")
    
    def _map_to_model_type(self, analysis_type: str) -> Optional[ModelType]:
        """분석 타입을 ModelType으로 매핑"""
        mapping = {
            'diffusion': ModelType.DIFFUSION,
            'virtual_tryon': ModelType.VIRTUAL_FITTING,
            'human_parsing': ModelType.HUMAN_PARSING,
            'pose_estimation': ModelType.POSE_ESTIMATION,
            'cloth_segmentation': ModelType.CLOTH_SEGMENTATION,
            'geometric_matching': ModelType.GEOMETRIC_MATCHING,
            'cloth_warping': ModelType.CLOTH_WARPING,
            'detection': ModelType.SEGMENTATION,
            'text_image': ModelType.DIFFUSION
        }
        return mapping.get(analysis_type)
    
    def _get_model_class(self, analysis_type: str) -> str:
        """분석 타입에서 모델 클래스명 추출"""
        mapping = {
            'diffusion': 'StableDiffusionPipeline',
            'virtual_tryon': 'HRVITONModel',
            'human_parsing': 'GraphonomyModel',
            'pose_estimation': 'OpenPoseModel',
            'cloth_segmentation': 'U2NetModel',
            'geometric_matching': 'GeometricMatchingModel',
            'cloth_warping': 'HRVITONModel',
            'detection': 'DetectronModel',
            'text_image': 'CLIPModel'
        }
        return mapping.get(analysis_type, 'BaseModel')
    
    async def load_optimal_model_for_step(self, step: str, **kwargs):
        """단계별 최적 모델 로드"""
        optimal_model = get_optimal_model_for_step(step)
        if not optimal_model:
            logger.warning(f"⚠️ {step}에 대한 최적 모델을 찾을 수 없음")
            return None
        
        logger.info(f"🎯 {step} 최적 모델 로드: {optimal_model}")
        return await self.load_model(optimal_model, **kwargs)

# 전역 체크포인트 모델 로더
_global_checkpoint_loader: Optional[CheckpointModelLoader] = None

def get_checkpoint_model_loader(**kwargs) -> CheckpointModelLoader:
    """전역 체크포인트 모델 로더 반환"""
    global _global_checkpoint_loader
    if _global_checkpoint_loader is None:
        _global_checkpoint_loader = CheckpointModelLoader(**kwargs)
    return _global_checkpoint_loader

async def load_best_model_for_step(step: str, **kwargs):
    """단계별 최고 성능 모델 로드"""
    loader = get_checkpoint_model_loader()
    return await loader.load_optimal_model_for_step(step, **kwargs)

# 빠른 접근 함수들
async def load_best_diffusion_model(**kwargs):
    """최고 성능 Diffusion 모델 로드"""
    return await load_best_model_for_step("step_06_virtual_fitting", **kwargs)

async def load_best_human_parsing_model(**kwargs):
    """최고 성능 인체 파싱 모델 로드"""
    return await load_best_model_for_step("step_01_human_parsing", **kwargs)

async def load_best_pose_model(**kwargs):
    """최고 성능 포즈 추정 모델 로드"""
    return await load_best_model_for_step("step_02_pose_estimation", **kwargs)
