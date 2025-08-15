"""
Post Processing Model Loader

후처리 모델들을 로드하고 관리하는 클래스입니다.
논문 기반의 AI 모델 구조에 맞춰 구현되었습니다.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
import logging
from pathlib import Path

# 프로젝트 로깅 설정 import
from backend.app.ai_pipeline.utils.logging_config import get_logger

# 체크포인트 로더 import
from .checkpoints.post_processing_checkpoint_loader import PostProcessingCheckpointLoader
from .checkpoints.post_processing_weight_mapper import PostProcessingWeightMapper

logger = get_logger(__name__)


class PostProcessingModelLoader:
    """
    후처리 모델들을 로드하고 관리하는 클래스
    
    지원 모델:
    - SwinIR (Super-Resolution)
    - Real-ESRGAN (Enhancement)
    - GFPGAN (Face Restoration)
    - CodeFormer (Face Restoration)
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Args:
            checkpoint_dir: 체크포인트 파일들이 저장된 디렉토리 경로
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 체크포인트 로더와 가중치 매퍼 초기화
        self.checkpoint_loader = PostProcessingCheckpointLoader(checkpoint_dir)
        self.weight_mapper = PostProcessingWeightMapper()
        
        # 로드된 모델들을 저장
        self.loaded_models = {}
        
        # 지원하는 모델 타입들
        self.supported_models = {
            'swinir': 'SwinIR',
            'realesrgan': 'RealESRGAN',
            'gfpgan': 'GFPGAN',
            'codeformer': 'CodeFormer'
        }
        
        logger.info(f"PostProcessingModelLoader initialized on device: {self.device}")
    
    def load_model(self, model_type: str, **kwargs) -> nn.Module:
        """
        지정된 타입의 모델을 로드합니다.
        
        Args:
            model_type: 모델 타입 ('swinir', 'realesrgan', 'gfpgan', 'codeformer')
            **kwargs: 모델 생성에 필요한 추가 파라미터
            
        Returns:
            로드된 모델
            
        Raises:
            ValueError: 지원하지 않는 모델 타입인 경우
            RuntimeError: 모델 로드 중 오류가 발생한 경우
        """
        if model_type not in self.supported_models:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(self.supported_models.keys())}")
        
        try:
            logger.info(f"Loading {model_type} model...")
            
            # 이미 로드된 모델이 있는지 확인
            if model_type in self.loaded_models:
                logger.info(f"{model_type} model already loaded, returning cached version")
                return self.loaded_models[model_type]
            
            # 모델 생성
            model = self._create_model(model_type, **kwargs)
            
            # 체크포인트 로드
            model = self.checkpoint_loader.load_checkpoint(model_type, model)
            
            # 로드된 모델 저장
            self.loaded_models[model_type] = model
            
            logger.info(f"Successfully loaded {model_type} model")
            return model
            
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {str(e)}")
            raise RuntimeError(f"Failed to load {model_type} model: {str(e)}")
    
    def _create_model(self, model_type: str, **kwargs) -> nn.Module:
        """
        지정된 타입의 모델을 생성합니다.
        
        Args:
            model_type: 모델 타입
            **kwargs: 모델 생성에 필요한 추가 파라미터
            
        Returns:
            생성된 모델
        """
        if model_type == 'swinir':
            return self._create_swinir_model(**kwargs)
        elif model_type == 'realesrgan':
            return self._create_realesrgan_model(**kwargs)
        elif model_type == 'gfpgan':
            return self._create_gfpgan_model(**kwargs)
        elif model_type == 'codeformer':
            return self._create_codeformer_model(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_swinir_model(self, **kwargs) -> nn.Module:
        """SwinIR 모델 생성"""
        from .models.swinir_model import SwinIRModel
        
        # 기본 파라미터 설정
        default_params = {
            'img_size': 64,
            'patch_size': 1,
            'in_chans': 3,
            'embed_dim': 96,
            'depths': [6, 6, 6, 6, 6, 6],
            'num_heads': [6, 6, 6, 6, 6, 6],
            'window_size': 7,
            'mlp_ratio': 4.,
            'qkv_bias': True,
            'qk_scale': None,
            'drop_rate': 0.,
            'attn_drop_rate': 0.,
            'drop_path_rate': 0.1,
            'norm_layer': nn.LayerNorm,
            'ape': False,
            'patch_norm': True,
            'use_checkpoint': False,
            'upscale': 4,
            'img_range': 1.,
            'upsampler': 'nearest+conv'
        }
        
        # 사용자 파라미터로 기본값 업데이트
        default_params.update(kwargs)
        
        return SwinIRModel(**default_params)
    
    def _create_realesrgan_model(self, **kwargs) -> nn.Module:
        """Real-ESRGAN 모델 생성"""
        from .models.realesrgan_model import RealESRGANModel
        
        # 기본 파라미터 설정
        default_params = {
            'num_feat': 64,
            'num_block': 23,
            'upscale': 4,
            'num_in_ch': 3,
            'num_out_ch': 3,
            'task': 'realesrgan'
        }
        
        # 사용자 파라미터로 기본값 업데이트
        default_params.update(kwargs)
        
        return RealESRGANModel(**default_params)
    
    def _create_gfpgan_model(self, **kwargs) -> nn.Module:
        """GFPGAN 모델 생성"""
        from .models.gfpgan_model import GFPGANModel
        
        # 기본 파라미터 설정
        default_params = {
            'out_size': 512,
            'num_style_feat': 512,
            'channel_multiplier': 2,
            'decoder_load_path': None,
            'fix_decoder': True,
            'num_mlp': 8,
            'input_is_latent': True,
            'different_w': True,
            'narrow': 1,
            'sft_half': True
        }
        
        # 사용자 파라미터로 기본값 업데이트
        default_params.update(kwargs)
        
        return GFPGANModel(**default_params)
    
    def _create_codeformer_model(self, **kwargs) -> nn.Module:
        """CodeFormer 모델 생성"""
        from .models.codeformer_model import CodeFormerModel
        
        # 기본 파라미터 설정
        default_params = {
            'dim_embd': 512,
            'n_head': 8,
            'n_layers': 9,
            'channels': [64, 128, 256, 512],
            'codebook_size': 1024,
            'latent_dim': 256
        }
        
        # 사용자 파라미터로 기본값 업데이트
        default_params.update(kwargs)
        
        return CodeFormerModel(**default_params)
    
    def get_model(self, model_type: str) -> Optional[nn.Module]:
        """
        이미 로드된 모델을 반환합니다.
        
        Args:
            model_type: 모델 타입
            
        Returns:
            로드된 모델 또는 None
        """
        return self.loaded_models.get(model_type)
    
    def unload_model(self, model_type: str) -> bool:
        """
        지정된 모델을 메모리에서 해제합니다.
        
        Args:
            model_type: 모델 타입
            
        Returns:
            해제 성공 여부
        """
        if model_type in self.loaded_models:
            del self.loaded_models[model_type]
            torch.cuda.empty_cache()  # GPU 메모리 정리
            logger.info(f"Unloaded model: {model_type}")
            return True
        return False
    
    def unload_all_models(self) -> int:
        """
        모든 로드된 모델을 메모리에서 해제합니다.
        
        Returns:
            해제된 모델 수
        """
        count = len(self.loaded_models)
        for model_type in list(self.loaded_models.keys()):
            self.unload_model(model_type)
        
        logger.info(f"Unloaded all {count} models")
        return count
    
    def get_loaded_model_types(self) -> List[str]:
        """
        현재 로드된 모델 타입들을 반환합니다.
        
        Returns:
            로드된 모델 타입 리스트
        """
        return list(self.loaded_models.keys())
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        모델의 정보를 반환합니다.
        
        Args:
            model_type: 모델 타입
            
        Returns:
            모델 정보 딕셔너리
        """
        if model_type not in self.loaded_models:
            return {}
        
        model = self.loaded_models[model_type]
        
        # 모델 정보 수집
        info = {
            'model_type': model_type,
            'model_class': model.__class__.__name__,
            'device': str(next(model.parameters()).device),
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        }
        
        return info
    
    def validate_model(self, model_type: str) -> Dict[str, Any]:
        """
        모델의 유효성을 검증합니다.
        
        Args:
            model_type: 모델 타입
            
        Returns:
            검증 결과 딕셔너리
        """
        if model_type not in self.loaded_models:
            return {'valid': False, 'error': 'Model not loaded'}
        
        try:
            model = self.loaded_models[model_type]
            
            # 기본 검증
            validation_result = {
                'valid': True,
                'model_type': model_type,
                'device': str(next(model.parameters()).device),
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'model_state': 'eval' if model.training == False else 'training'
            }
            
            # 체크포인트 검증
            checkpoint_valid = self.checkpoint_loader.validate_checkpoint(model_type)
            validation_result['checkpoint_valid'] = checkpoint_valid
            
            return validation_result
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        현재 메모리 사용량을 반환합니다.
        
        Returns:
            메모리 사용량 정보
        """
        if torch.cuda.is_available():
            memory_info = {
                'gpu_allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
                'gpu_reserved': torch.cuda.memory_reserved() / (1024**3),   # GB
                'gpu_max_allocated': torch.cuda.max_memory_allocated() / (1024**3),  # GB
                'gpu_max_reserved': torch.cuda.max_memory_reserved() / (1024**3)     # GB
            }
        else:
            memory_info = {'gpu_available': False}
        
        # CPU 메모리 정보 (간단한 추정)
        total_params = sum(
            sum(p.numel() for p in model.parameters()) 
            for model in self.loaded_models.values()
        )
        
        memory_info.update({
            'total_models': len(self.loaded_models),
            'total_parameters': total_params,
            'estimated_cpu_memory_mb': total_params * 4 / (1024 * 1024)  # float32 기준
        })
        
        return memory_info
