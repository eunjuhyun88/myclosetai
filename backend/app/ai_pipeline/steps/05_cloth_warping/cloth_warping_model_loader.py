# backend/app/ai_pipeline/steps/cloth_warping_integrated_loader.py
"""
🔥 ClothWarpingStep 통합 모델 로딩 시스템
================================================================================

✅ Central Hub 통합
✅ 체크포인트 분석 시스템 연동
✅ 모델 아키텍처 기반 생성
✅ 단계적 폴백 시스템
✅ BaseStepMixin 완전 호환

Author: MyCloset AI Team
Date: 2025-01-27
Version: 1.0 (통합 모델 로딩 시스템)
"""

import os
import sys
import gc
import time
import json
import logging
import traceback
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass

# PyTorch 안전 import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# NumPy 안전 import
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# MPS 지원 확인
MPS_AVAILABLE = TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

# 기본 디바이스 설정
DEFAULT_DEVICE = "mps" if MPS_AVAILABLE else ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)

@dataclass
class ModelLoadingResult:
    """모델 로딩 결과"""
    success: bool
    model: Optional[Any] = None
    model_name: str = ""
    loading_method: str = ""
    error_message: str = ""
    processing_time: float = 0.0

class ClothWarpingModelLoader:
    """ClothWarpingStep 전용 모델 로딩 시스템"""
    
    def __init__(self, device: str = DEFAULT_DEVICE, logger=None):
        self.device = self._setup_device(device)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.models = {}
        self.loaded_models = {}
        
    def _setup_device(self, device: str) -> str:
        """디바이스 설정"""
        if device == "auto":
            if MPS_AVAILABLE:
                return "mps"
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_models_integrated(self) -> bool:
        """통합된 모델 로딩 시스템 - 메인 메서드"""
        try:
            self.logger.info("🚀 ClothWarping 통합 모델 로딩 시스템 시작")
            start_time = time.time()
            
            # 1단계: Central Hub 시도
            if self._load_via_central_hub():
                self.logger.info("✅ Central Hub를 통한 모델 로딩 성공")
                return True
            
            # 2단계: 체크포인트 분석 기반 로딩
            models_loaded = 0
            for model_name in ['tps_model', 'viton_checkpoint', 'dpt_model', 'raft_model']:
                result = self._load_with_checkpoint_analysis(model_name)
                if result.success:
                    self.models[model_name] = result.model
                    models_loaded += 1
            
            if models_loaded > 0:
                self.logger.info(f"✅ 체크포인트 분석 기반 모델 로딩 성공: {models_loaded}개")
                return True
            
            # 3단계: 아키텍처 기반 생성
            fallback_models = {
                'tps_model': {'num_control_points': 25, 'architecture_type': 'tps'},
                'viton_checkpoint': {'input_channels': 6, 'architecture_type': 'viton'},
                'dpt_model': {'architecture_type': 'dpt'},
                'raft_model': {'architecture_type': 'raft'}
            }
            
            for model_name, config in fallback_models.items():
                result = self._create_with_architecture(model_name, config)
                if result.success:
                    self.models[model_name] = result.model
                    models_loaded += 1
            
            if models_loaded > 0:
                self.logger.info(f"✅ 아키텍처 기반 모델 생성 성공: {models_loaded}개")
                return True
            
            self.logger.error("❌ 모든 모델 로딩 방법 실패")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 통합 모델 로딩 실패: {e}")
            return False
    
    def _load_via_central_hub(self) -> bool:
        """Central Hub를 통한 모델 로딩"""
        try:
            # Central Hub에서 모델 로더 서비스 가져오기
            model_loader_service = self._get_service_from_central_hub('model_loader')
            if not model_loader_service:
                self.logger.warning("⚠️ Central Hub에서 모델 로더 서비스를 찾을 수 없습니다")
                return False
            
            # Step별 최적 모델 로드
            models_to_load = {
                'tps_model': 'cloth_warping',
                'viton_checkpoint': 'cloth_warping',
                'dpt_model': 'cloth_warping',
                'raft_model': 'cloth_warping'
            }
            
            models_loaded = 0
            for model_name, step_type in models_to_load.items():
                try:
                    model = model_loader_service.load_model_for_step(
                        step_type=step_type,
                        model_name=model_name,
                        device=self.device
                    )
                    if model:
                        self.models[model_name] = model
                        models_loaded += 1
                        self.logger.info(f"✅ {model_name} Central Hub 로드 완료")
                    else:
                        self.logger.warning(f"⚠️ {model_name} Central Hub 로드 실패")
                except Exception as e:
                    self.logger.error(f"❌ {model_name} Central Hub 로드 중 오류: {e}")
            
            return models_loaded > 0
            
        except Exception as e:
            self.logger.error(f"❌ Central Hub 모델 로딩 실패: {e}")
            return False
    
    def _load_with_checkpoint_analysis(self, model_name: str) -> ModelLoadingResult:
        """체크포인트 분석 기반 모델 로딩"""
        start_time = time.time()
        try:
            from app.ai_pipeline.models.model_loader import DynamicModelCreator
            from app.ai_pipeline.models.checkpoint_model_loader import get_checkpoint_model_loader
            
            # 체크포인트 모델 로더 가져오기
            checkpoint_loader = get_checkpoint_model_loader(device=self.device)
            
            # 동적 모델 생성기
            model_creator = DynamicModelCreator()
            
            # 체크포인트 경로 가져오기
            checkpoint_path = checkpoint_loader.get_checkpoint_path(model_name)
            if not checkpoint_path:
                return ModelLoadingResult(
                    success=False,
                    model_name=model_name,
                    loading_method="checkpoint_analysis",
                    error_message=f"체크포인트 경로를 찾을 수 없습니다: {model_name}",
                    processing_time=time.time() - start_time
                )
            
            # 체크포인트에서 모델 생성
            model = model_creator.create_model_from_checkpoint(
                checkpoint_path=checkpoint_path,
                step_type='cloth_warping',
                device=self.device
            )
            
            if model:
                self.logger.info(f"✅ {model_name} 체크포인트 분석 기반 로드 완료")
                return ModelLoadingResult(
                    success=True,
                    model=model,
                    model_name=model_name,
                    loading_method="checkpoint_analysis",
                    processing_time=time.time() - start_time
                )
            else:
                return ModelLoadingResult(
                    success=False,
                    model_name=model_name,
                    loading_method="checkpoint_analysis",
                    error_message="체크포인트 분석 실패",
                    processing_time=time.time() - start_time
                )
                
        except Exception as e:
            return ModelLoadingResult(
                success=False,
                model_name=model_name,
                loading_method="checkpoint_analysis",
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _create_with_architecture(self, model_type: str, config: Dict[str, Any]) -> ModelLoadingResult:
        """모델 아키텍처 기반 모델 생성"""
        start_time = time.time()
        try:
            from app.ai_pipeline.models.model_loader import ClothWarpingArchitecture
            
            # Cloth Warping 특화 아키텍처 생성
            architecture = ClothWarpingArchitecture(
                step_type='cloth_warping',
                device=self.device
            )
            
            # 아키텍처 기반 모델 생성
            model = architecture.create_model(config)
            
            # 모델 검증
            if architecture.validate_model(model):
                self.logger.info(f"✅ {model_type} 아키텍처 기반 모델 생성 완료")
                return ModelLoadingResult(
                    success=True,
                    model=model,
                    model_name=model_type,
                    loading_method="architecture_based",
                    processing_time=time.time() - start_time
                )
            else:
                return ModelLoadingResult(
                    success=False,
                    model_name=model_type,
                    loading_method="architecture_based",
                    error_message="모델 검증 실패",
                    processing_time=time.time() - start_time
                )
                
        except Exception as e:
            return ModelLoadingResult(
                success=False,
                model_name=model_type,
                loading_method="architecture_based",
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _get_service_from_central_hub(self, service_key: str):
        """Central Hub에서 서비스 가져오기"""
        try:
            # Central Hub import 시도
            from app.api.central_hub import get_service
            return get_service(service_key)
        except ImportError:
            try:
                # 대체 경로 시도
                from app.core.di_container import get_service
                return get_service(service_key)
            except ImportError:
                self.logger.warning(f"⚠️ Central Hub 서비스 가져오기 실패: {service_key}")
                return None
    
    def get_loaded_models(self) -> Dict[str, Any]:
        """로드된 모델들 반환"""
        return self.models.copy()
    
    def cleanup_resources(self):
        """리소스 정리"""
        try:
            # AI 모델 정리
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'cpu'):
                        model.cpu()
                    del model
                except:
                    pass
            
            self.models.clear()
            self.loaded_models.clear()
            
            # 메모리 정리
            for _ in range(3):
                gc.collect()
            if TORCH_AVAILABLE and MPS_AVAILABLE:
                try:
                    torch.mps.empty_cache()
                    if hasattr(torch.mps, 'synchronize'):
                        torch.mps.synchronize()
                except Exception as e:
                    self.logger.warning(f"⚠️ MPS 메모리 정리 실패: {e}")
            
            self.logger.info("✅ ClothWarpingIntegratedLoader 리소스 정리 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ 리소스 정리 실패: {e}")

# 전역 로더 인스턴스
_global_integrated_loader: Optional[ClothWarpingModelLoader] = None

def get_integrated_loader(device: str = DEFAULT_DEVICE, logger=None) -> ClothWarpingModelLoader:
    """전역 통합 로더 반환"""
    global _global_integrated_loader
    if _global_integrated_loader is None:
        _global_integrated_loader = ClothWarpingModelLoader(device=device, logger=logger)
    return _global_integrated_loader

# 모듈 내보내기
__all__ = [
    "ClothWarpingModelLoader",
    "ModelLoadingResult", 
    "get_integrated_loader"
]
