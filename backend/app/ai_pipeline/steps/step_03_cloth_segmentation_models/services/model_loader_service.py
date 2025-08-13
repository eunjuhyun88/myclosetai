#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Model Loader Service
=====================================================================

모델 로딩 및 관리 서비스

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import os
import gc
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..config import SegmentationMethod, ClothCategory, QualityLevel
from ..models import RealDeepLabV3PlusModel, RealU2NETModel, RealSAMModel

logger = logging.getLogger(__name__)

class ModelLoaderService:
    """
    🔥 모델 로딩 및 관리 서비스
    
    분리된 기능들:
    - 모델 경로 감지 및 관리
    - 모델 로딩 및 초기화
    - 모델 상태 관리
    - 메모리 안전성 보장
    """
    
    def __init__(self, device: str = "cpu"):
        """초기화"""
        self.device = device
        self.logger = logging.getLogger(f"{__name__}.ModelLoaderService")
        self.model_paths = {}
        self.loaded_models = {}
        self.models_loading_status = {
            'deeplabv3plus': False,
            'maskrcnn': False,
            'sam_huge': False,
            'u2net_cloth': False,
            'total_loaded': 0,
            'loading_errors': []
        }
        
    def load_segmentation_models(self, model_paths: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """세그멘테이션 모델들을 로딩"""
        try:
            self.logger.info("🔄 세그멘테이션 모델 로딩 시작...")
            
            # 모델 경로 설정
            if model_paths:
                self.model_paths = model_paths
            else:
                self._detect_model_paths()
            
            # 각 모델 로딩
            models_loaded = {}
            
            # U2Net 모델 로딩
            if 'u2net_cloth' in self.model_paths:
                try:
                    u2net_model = self._load_u2net_model()
                    if u2net_model:
                        models_loaded['u2net_cloth'] = u2net_model
                        self.models_loading_status['u2net_cloth'] = True
                        self.models_loading_status['total_loaded'] += 1
                        self.logger.info("✅ U2Net 모델 로딩 완료")
                except Exception as e:
                    self.logger.error(f"❌ U2Net 모델 로딩 실패: {e}")
                    self.models_loading_status['loading_errors'].append(f"u2net_cloth: {e}")
            
            # SAM 모델 로딩
            if 'sam_huge' in self.model_paths:
                try:
                    sam_model = self._load_sam_model()
                    if sam_model:
                        models_loaded['sam_huge'] = sam_model
                        self.models_loading_status['sam_huge'] = True
                        self.models_loading_status['total_loaded'] += 1
                        self.logger.info("✅ SAM 모델 로딩 완료")
                except Exception as e:
                    self.logger.error(f"❌ SAM 모델 로딩 실패: {e}")
                    self.models_loading_status['loading_errors'].append(f"sam_huge: {e}")
            
            # DeepLabV3+ 모델 로딩
            if 'deeplabv3plus' in self.model_paths:
                try:
                    deeplabv3_model = self._load_deeplabv3plus_model()
                    if deeplabv3_model:
                        models_loaded['deeplabv3plus'] = deeplabv3_model
                        self.models_loading_status['deeplabv3plus'] = True
                        self.models_loading_status['total_loaded'] += 1
                        self.logger.info("✅ DeepLabV3+ 모델 로딩 완료")
                except Exception as e:
                    self.logger.error(f"❌ DeepLabV3+ 모델 로딩 실패: {e}")
                    self.models_loading_status['loading_errors'].append(f"deeplabv3plus: {e}")
            
            self.loaded_models = models_loaded
            self.logger.info(f"✅ 세그멘테이션 모델 로딩 완료: {len(models_loaded)}개")
            
            return models_loaded
            
        except Exception as e:
            self.logger.error(f"❌ 세그멘테이션 모델 로딩 실패: {e}")
            return {}

    def _detect_model_paths(self):
        """모델 경로 감지"""
        try:
            # 기본 모델 경로들
            base_paths = [
                "models/",
                "backend/models/",
                "backend/app/ai_pipeline/models/",
                "backend/app/ai_pipeline/steps/03_cloth_segmentation/models/"
            ]
            
            # 각 모델별 경로 감지
            model_paths = {}
            
            # U2Net 경로 감지
            u2net_paths = [
                "u2net_cloth.pth",
                "u2net_cloth_model.pth",
                "u2net_cloth_weights.pth"
            ]
            
            for base_path in base_paths:
                for u2net_path in u2net_paths:
                    full_path = os.path.join(base_path, u2net_path)
                    if os.path.exists(full_path):
                        model_paths['u2net_cloth'] = full_path
                        break
                if 'u2net_cloth' in model_paths:
                    break
            
            # SAM 경로 감지
            sam_paths = [
                "sam_huge.pth",
                "sam_huge_model.pth",
                "sam_huge_weights.pth"
            ]
            
            for base_path in base_paths:
                for sam_path in sam_paths:
                    full_path = os.path.join(base_path, sam_path)
                    if os.path.exists(full_path):
                        model_paths['sam_huge'] = full_path
                        break
                if 'sam_huge' in model_paths:
                    break
            
            # DeepLabV3+ 경로 감지
            deeplabv3_paths = [
                "deeplabv3plus.pth",
                "deeplabv3plus_model.pth",
                "deeplabv3plus_weights.pth"
            ]
            
            for base_path in base_paths:
                for deeplabv3_path in deeplabv3_paths:
                    full_path = os.path.join(base_path, deeplabv3_path)
                    if os.path.exists(full_path):
                        model_paths['deeplabv3plus'] = full_path
                        break
                if 'deeplabv3plus' in model_paths:
                    break
            
            self.model_paths = model_paths
            self.logger.info(f"✅ 모델 경로 감지 완료: {len(model_paths)}개")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 경로 감지 실패: {e}")
            self.model_paths = {}

    def _load_u2net_model(self) -> Optional[RealU2NETModel]:
        """U2Net 모델 로딩"""
        try:
            if 'u2net_cloth' not in self.model_paths:
                self.logger.warning("⚠️ U2Net 모델 경로가 없습니다")
                return None
            
            model_path = self.model_paths['u2net_cloth']
            self.logger.info(f"🔄 U2Net 모델 로딩 중: {model_path}")
            
            model = RealU2NETModel(model_path=model_path, device=self.device)
            if model.load():
                self.logger.info("✅ U2Net 모델 로딩 성공")
                return model
            else:
                self.logger.error("❌ U2Net 모델 로딩 실패")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ U2Net 모델 로딩 중 오류: {e}")
            return None

    def _load_sam_model(self) -> Optional[RealSAMModel]:
        """SAM 모델 로딩"""
        try:
            if 'sam_huge' not in self.model_paths:
                self.logger.warning("⚠️ SAM 모델 경로가 없습니다")
                return None
            
            model_path = self.model_paths['sam_huge']
            self.logger.info(f"🔄 SAM 모델 로딩 중: {model_path}")
            
            model = RealSAMModel(model_path=model_path, device=self.device)
            if model.load():
                self.logger.info("✅ SAM 모델 로딩 성공")
                return model
            else:
                self.logger.error("❌ SAM 모델 로딩 실패")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ SAM 모델 로딩 중 오류: {e}")
            return None

    def _load_deeplabv3plus_model(self) -> Optional[RealDeepLabV3PlusModel]:
        """DeepLabV3+ 모델 로딩"""
        try:
            if 'deeplabv3plus' not in self.model_paths:
                self.logger.warning("⚠️ DeepLabV3+ 모델 경로가 없습니다")
                return None
            
            model_path = self.model_paths['deeplabv3plus']
            self.logger.info(f"🔄 DeepLabV3+ 모델 로딩 중: {model_path}")
            
            model = RealDeepLabV3PlusModel(model_path=model_path, device=self.device)
            if model.load():
                self.logger.info("✅ DeepLabV3+ 모델 로딩 성공")
                return model
            else:
                self.logger.error("❌ DeepLabV3+ 모델 로딩 실패")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ DeepLabV3+ 모델 로딩 중 오류: {e}")
            return None

    def get_loaded_models(self) -> Dict[str, Any]:
        """로딩된 모델들 반환"""
        return self.loaded_models

    def get_loading_status(self) -> Dict[str, Any]:
        """로딩 상태 반환"""
        return self.models_loading_status

    def get_model_paths(self) -> Dict[str, str]:
        """모델 경로들 반환"""
        return self.model_paths

    def reload_models(self) -> bool:
        """모델들 재로딩"""
        try:
            self.logger.info("🔄 모델 재로딩 시작...")
            
            # 기존 모델들 정리
            self.cleanup_models()
            
            # 모델들 다시 로딩
            models_loaded = self.load_segmentation_models()
            
            success = len(models_loaded) > 0
            if success:
                self.logger.info("✅ 모델 재로딩 완료")
            else:
                self.logger.warning("⚠️ 모델 재로딩 실패")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ 모델 재로딩 실패: {e}")
            return False

    def cleanup_models(self):
        """모델들 정리"""
        try:
            # 로딩된 모델들 정리
            for model_name, model in self.loaded_models.items():
                if hasattr(model, 'cleanup'):
                    model.cleanup()
                elif hasattr(model, 'cpu'):
                    model.cpu()
            
            self.loaded_models.clear()
            
            # 메모리 정리
            gc.collect()
            if TORCH_AVAILABLE and hasattr(torch, 'mps') and torch.mps.is_available():
                torch.mps.empty_cache()
            
            self.logger.info("✅ 모델 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 모델 정리 실패: {e}")

    def get_available_models(self) -> List[str]:
        """사용 가능한 모델들 반환"""
        return list(self.loaded_models.keys())

    def is_model_loaded(self, model_name: str) -> bool:
        """특정 모델이 로딩되었는지 확인"""
        return model_name in self.loaded_models

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """특정 모델 정보 반환"""
        if model_name not in self.loaded_models:
            return {}
        
        model = self.loaded_models[model_name]
        info = {
            'name': model_name,
            'loaded': True,
            'device': getattr(model, 'device', self.device),
            'model_type': type(model).__name__
        }
        
        # 모델별 추가 정보
        if hasattr(model, 'get_model_info'):
            info.update(model.get_model_info())
        
        return info
