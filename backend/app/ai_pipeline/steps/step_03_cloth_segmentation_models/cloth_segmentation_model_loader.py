#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Model Loader
==========================================================

의류 세그멘테이션을 위한 통합 모델 로더
- models/ 폴더의 완전한 신경망 구조 로딩 (체크포인트 없어도 동작)
- checkpoints/ 폴더의 사전 훈련된 가중치 로딩 (성능 향상)
- 두 가지를 조합하여 최적의 모델 제공

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from abc import ABC, abstractmethod

# PyTorch import 시도
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    # torch가 없을 때는 기본 타입 사용
    class MockNNModule:
        """Mock nn.Module (torch 없음)"""
        pass
    # nn.Module을 MockNNModule으로 대체
    class nn:
        Module = MockNNModule

# 체크포인트 관련 모듈들
try:
    from .checkpoints.cloth_segmentation_checkpoint_loader import ClothSegmentationCheckpointLoader
    from .checkpoints.cloth_segmentation_weight_mapper import ClothSegmentationWeightMapper
    from .checkpoints.cloth_segmentation_checkpoint_utils import ClothSegmentationCheckpointValidator
    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False
    ClothSegmentationCheckpointLoader = None
    ClothSegmentationWeightMapper = None
    ClothSegmentationCheckpointValidator = None

class BaseModelLoader(ABC):
    """모델 로더 기본 클래스"""
    
    def __init__(self, models_dir: str = None, checkpoints_dir: str = None):
        self.models_dir = Path(models_dir) if models_dir else None
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else 'cpu'
        
    @abstractmethod
    def load_model(self, model_name: str, use_checkpoint: bool = True) -> Optional[nn.Module]:
        """모델 로딩 (구현 필요)"""
        pass
        
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환 (구현 필요)"""
        pass
        
    def validate_model(self, model: nn.Module) -> bool:
        """모델 유효성 검증"""
        try:
            if not TORCH_AVAILABLE:
                return False
                
            # 기본 모델 구조 확인
            if not isinstance(model, nn.Module):
                return False
                
            # 더미 입력으로 forward pass 테스트
            try:
                with torch.no_grad():
                    if hasattr(model, 'input_channels'):
                        input_channels = model.input_channels
                    else:
                        input_channels = 3  # 기본값
                        
                    dummy_input = torch.randn(1, input_channels, 64, 64).to(self.device)
                    output = model(dummy_input)
                    
                    if output is not None:
                        self.logger.info(f"✅ 모델 유효성 검증 성공: {model.__class__.__name__}")
                        return True
                    else:
                        self.logger.warning(f"⚠️ 모델 출력이 None입니다: {model.__class__.__name__}")
                        return False
                        
            except Exception as e:
                self.logger.warning(f"⚠️ 모델 forward pass 실패: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"모델 유효성 검증 실패: {e}")
            return False
            
        return False

class ClothSegmentationU2NetLoader(BaseModelLoader):
    """U2Net 모델 로더"""
    
    def __init__(self, models_dir: str = None, checkpoints_dir: str = None):
        super().__init__(models_dir, checkpoints_dir)
        
    def load_model(self, model_name: str = "u2net", use_checkpoint: bool = True) -> Optional[nn.Module]:
        """U2Net 모델 로딩"""
        try:
            if not TORCH_AVAILABLE:
                self.logger.error("PyTorch를 사용할 수 없습니다")
                return None
                
            # 1. 기본 모델 구조 로딩 (체크포인트 없어도 동작)
            model = self._load_u2net_structure()
            if model is None:
                self.logger.error("U2Net 모델 구조 로딩 실패")
                return None
                
            # 2. 체크포인트가 있으면 가중치 로딩 (성능 향상)
            if use_checkpoint and CHECKPOINT_AVAILABLE and self.checkpoints_dir:
                checkpoint_loader = ClothSegmentationCheckpointLoader(str(self.checkpoints_dir))
                available_checkpoints = checkpoint_loader.get_available_checkpoints()
                
                if 'u2net' in available_checkpoints and available_checkpoints['u2net']:
                    checkpoint_path = available_checkpoints['u2net'][0]  # 첫 번째 체크포인트 사용
                    if checkpoint_loader.load_u2net_checkpoint(model, checkpoint_path):
                        self.logger.info(f"✅ U2Net 체크포인트 로딩 성공: {checkpoint_path}")
                    else:
                        self.logger.warning(f"⚠️ U2Net 체크포인트 로딩 실패, 기본 모델 사용")
                else:
                    self.logger.info("ℹ️ U2Net 체크포인트가 없습니다. 기본 모델 사용")
            else:
                self.logger.info("ℹ️ 체크포인트 사용하지 않음. 기본 모델 사용")
                
            # 3. 모델 유효성 검증
            if self.validate_model(model):
                self.logger.info(f"✅ U2Net 모델 로딩 완료: {model_name}")
                return model
            else:
                self.logger.error("U2Net 모델 유효성 검증 실패")
                return None
                
        except Exception as e:
            self.logger.error(f"U2Net 모델 로딩 실패: {e}")
            return None
            
    def _load_u2net_structure(self) -> Optional[nn.Module]:
        """U2Net 모델 구조 로딩"""
        try:
            # models/ 폴더에서 U2Net 모델 구조 로딩
            if self.models_dir and (self.models_dir / "u2net.py").exists():
                # 동적 import 시도
                sys.path.insert(0, str(self.models_dir))
                try:
                    from u2net import U2Net
                    model = U2Net()
                    model.to(self.device)
                    return model
                except ImportError as e:
                    self.logger.warning(f"U2Net 모델 구조 import 실패: {e}")
                    
            # 폴백: 기본 U2Net 구조 생성
            return self._create_basic_u2net()
            
        except Exception as e:
            self.logger.error(f"U2Net 모델 구조 로딩 실패: {e}")
            return None
            
    def _create_basic_u2net(self) -> nn.Module:
        """기본 U2Net 구조 생성 (체크포인트 없어도 동작)"""
        try:
            # 간단한 U2Net 구조 생성
            class BasicU2Net(nn.Module):
                def __init__(self, in_ch=3, out_ch=1):
                    super().__init__()
                    self.input_channels = in_ch
                    
                    # 인코더
                    self.en_1 = nn.Sequential(
                        nn.Conv2d(in_ch, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
                    )
                    self.en_2 = nn.Sequential(
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True)
                    )
                    
                    # 디코더
                    self.de_1 = nn.Sequential(
                        nn.Conv2d(128, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
                    )
                    
                    # 최종 출력
                    self.final = nn.Conv2d(64, out_ch, 1)
                    
                def forward(self, x):
                    # 인코더
                    en1 = self.en_1(x)
                    en2 = self.en_2(en1)
                    
                    # 디코더
                    de1 = self.de_1(en2)
                    
                    # 최종 출력
                    output = self.final(de1)
                    return torch.sigmoid(output)
                    
            model = BasicU2Net()
            model.to(self.device)
            self.logger.info("✅ 기본 U2Net 구조 생성 완료 (체크포인트 없어도 동작)")
            return model
            
        except Exception as e:
            self.logger.error(f"기본 U2Net 구조 생성 실패: {e}")
            return None
            
    def get_available_models(self) -> List[str]:
        """사용 가능한 U2Net 모델 목록"""
        models = []
        
        if self.models_dir and (self.models_dir / "u2net.py").exists():
            models.append("u2net")
            
        # 기본 모델은 항상 사용 가능
        models.append("u2net_basic")
        
        return models

class ClothSegmentationDeepLabV3PlusLoader(BaseModelLoader):
    """DeepLabV3+ 모델 로더"""
    
    def __init__(self, models_dir: str = None, checkpoints_dir: str = None):
        super().__init__(models_dir, checkpoints_dir)
        
    def load_model(self, model_name: str = "deeplabv3plus", use_checkpoint: bool = True) -> Optional[nn.Module]:
        """DeepLabV3+ 모델 로딩"""
        try:
            if not TORCH_AVAILABLE:
                self.logger.error("PyTorch를 사용할 수 없습니다")
                return None
                
            # 1. 기본 모델 구조 로딩 (체크포인트 없어도 동작)
            model = self._load_deeplabv3plus_structure()
            if model is None:
                self.logger.error("DeepLabV3+ 모델 구조 로딩 실패")
                return None
                
            # 2. 체크포인트가 있으면 가중치 로딩 (성능 향상)
            if use_checkpoint and CHECKPOINT_AVAILABLE and self.checkpoints_dir:
                checkpoint_loader = ClothSegmentationCheckpointLoader(str(self.checkpoints_dir))
                available_checkpoints = checkpoint_loader.get_available_checkpoints()
                
                if 'deeplabv3plus' in available_checkpoints and available_checkpoints['deeplabv3plus']:
                    checkpoint_path = available_checkpoints['deeplabv3plus'][0]
                    if checkpoint_loader.load_deeplabv3plus_checkpoint(model, checkpoint_path):
                        self.logger.info(f"✅ DeepLabV3+ 체크포인트 로딩 성공: {checkpoint_path}")
                    else:
                        self.logger.warning(f"⚠️ DeepLabV3+ 체크포인트 로딩 실패, 기본 모델 사용")
                else:
                    self.logger.info("ℹ️ DeepLabV3+ 체크포인트가 없습니다. 기본 모델 사용")
            else:
                self.logger.info("ℹ️ 체크포인트 사용하지 않음. 기본 모델 사용")
                
            # 3. 모델 유효성 검증
            if self.validate_model(model):
                self.logger.info(f"✅ DeepLabV3+ 모델 로딩 완료: {model_name}")
                return model
            else:
                self.logger.error("DeepLabV3+ 모델 유효성 검증 실패")
                return None
                
        except Exception as e:
            self.logger.error(f"DeepLabV3+ 모델 로딩 실패: {e}")
            return None
            
    def _load_deeplabv3plus_structure(self) -> Optional[nn.Module]:
        """DeepLabV3+ 모델 구조 로딩"""
        try:
            # models/ 폴더에서 DeepLabV3+ 모델 구조 로딩
            if self.models_dir and (self.models_dir / "deeplabv3plus.py").exists():
                # 동적 import 시도
                sys.path.insert(0, str(self.models_dir))
                try:
                    from deeplabv3plus import DeepLabV3Plus
                    model = DeepLabV3Plus()
                    model.to(self.device)
                    return model
                except ImportError as e:
                    self.logger.warning(f"DeepLabV3+ 모델 구조 import 실패: {e}")
                    
            # 폴백: 기본 DeepLabV3+ 구조 생성
            return self._create_basic_deeplabv3plus()
            
        except Exception as e:
            self.logger.error(f"DeepLabV3+ 모델 구조 로딩 실패: {e}")
            return None
            
    def _create_basic_deeplabv3plus(self) -> nn.Module:
        """기본 DeepLabV3+ 구조 생성 (체크포인트 없어도 동작)"""
        try:
            # 간단한 DeepLabV3+ 구조 생성
            class BasicDeepLabV3Plus(nn.Module):
                def __init__(self, in_ch=3, out_ch=1):
                    super().__init__()
                    self.input_channels = in_ch
                    
                    # 백본 (간단한 ResNet 블록)
                    self.backbone = nn.Sequential(
                        nn.Conv2d(in_ch, 64, 7, stride=2, padding=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2, padding=1)
                    )
                    
                    # ASPP 모듈
                    self.aspp = nn.Sequential(
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True)
                    )
                    
                    # 디코더
                    self.decoder = nn.Sequential(
                        nn.Conv2d(128, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, out_ch, 1)
                    )
                    
                def forward(self, x):
                    # 백본
                    backbone_out = self.backbone(x)
                    
                    # ASPP
                    aspp_out = self.aspp(backbone_out)
                    
                    # 디코더
                    decoder_out = self.decoder(aspp_out)
                    
                    # 업샘플링
                    output = F.interpolate(decoder_out, size=x.shape[2:], mode='bilinear', align_corners=False)
                    return torch.sigmoid(output)
                    
            model = BasicDeepLabV3Plus()
            model.to(self.device)
            self.logger.info("✅ 기본 DeepLabV3+ 구조 생성 완료 (체크포인트 없어도 동작)")
            return model
            
        except Exception as e:
            self.logger.error(f"기본 DeepLabV3+ 구조 생성 실패: {e}")
            return None
            
    def get_available_models(self) -> List[str]:
        """사용 가능한 DeepLabV3+ 모델 목록"""
        models = []
        
        if self.models_dir and (self.models_dir / "deeplabv3plus.py").exists():
            models.append("deeplabv3plus")
            
        # 기본 모델은 항상 사용 가능
        models.append("deeplabv3plus_basic")
        
        return models

class ClothSegmentationSAMLoader(BaseModelLoader):
    """SAM 모델 로더"""
    
    def __init__(self, models_dir: str = None, checkpoints_dir: str = None):
        super().__init__(models_dir, checkpoints_dir)
        
    def load_model(self, model_name: str = "sam", use_checkpoint: bool = True) -> Optional[nn.Module]:
        """SAM 모델 로딩"""
        try:
            if not TORCH_AVAILABLE:
                self.logger.error("PyTorch를 사용할 수 없습니다")
                return None
                
            # 1. 기본 모델 구조 로딩 (체크포인트 없어도 동작)
            model = self._load_sam_structure()
            if model is None:
                self.logger.error("SAM 모델 구조 로딩 실패")
                return None
                
            # 2. 체크포인트가 있으면 가중치 로딩 (성능 향상)
            if use_checkpoint and CHECKPOINT_AVAILABLE and self.checkpoints_dir:
                checkpoint_loader = ClothSegmentationCheckpointLoader(str(self.checkpoints_dir))
                available_checkpoints = checkpoint_loader.get_available_checkpoints()
                
                if 'sam' in available_checkpoints and available_checkpoints['sam']:
                    checkpoint_path = available_checkpoints['sam'][0]
                    if checkpoint_loader.load_sam_checkpoint(model, checkpoint_path):
                        self.logger.info(f"✅ SAM 체크포인트 로딩 성공: {checkpoint_path}")
                    else:
                        self.logger.warning(f"⚠️ SAM 체크포인트 로딩 실패, 기본 모델 사용")
                else:
                    self.logger.info("ℹ️ SAM 체크포인트가 없습니다. 기본 모델 사용")
            else:
                self.logger.info("ℹ️ 체크포인트 사용하지 않음. 기본 모델 사용")
                
            # 3. 모델 유효성 검증
            if self.validate_model(model):
                self.logger.info(f"✅ SAM 모델 로딩 완료: {model_name}")
                return model
            else:
                self.logger.error("SAM 모델 유효성 검증 실패")
                return None
                
        except Exception as e:
            self.logger.error(f"SAM 모델 로딩 실패: {e}")
            return None
            
    def _load_sam_structure(self) -> Optional[nn.Module]:
        """SAM 모델 구조 로딩"""
        try:
            # models/ 폴더에서 SAM 모델 구조 로딩
            if self.models_dir and (self.models_dir / "sam.py").exists():
                # 동적 import 시도
                sys.path.insert(0, str(self.models_dir))
                try:
                    from sam import SAM
                    model = SAM()
                    model.to(self.device)
                    return model
                except ImportError as e:
                    self.logger.warning(f"SAM 모델 구조 import 실패: {e}")
                    
            # 폴백: 기본 SAM 구조 생성
            return self._create_basic_sam()
            
        except Exception as e:
            self.logger.error(f"SAM 모델 구조 로딩 실패: {e}")
            return None
            
    def _create_basic_sam(self) -> nn.Module:
        """기본 SAM 구조 생성 (체크포인트 없어도 동작)"""
        try:
            # 간단한 SAM 구조 생성
            class BasicSAM(nn.Module):
                def __init__(self, in_ch=3, out_ch=1):
                    super().__init__()
                    self.input_channels = in_ch
                    
                    # 이미지 인코더
                    self.image_encoder = nn.Sequential(
                        nn.Conv2d(in_ch, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True)
                    )
                    
                    # 프롬프트 인코더
                    self.prompt_encoder = nn.Sequential(
                        nn.Linear(2, 64),  # 2D 좌표
                        nn.ReLU(inplace=True)
                    )
                    
                    # 마스크 디코더
                    self.mask_decoder = nn.Sequential(
                        nn.Conv2d(64, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, out_ch, 1)
                    )
                    
                def forward(self, x, points=None):
                    # 이미지 인코딩
                    image_features = self.image_encoder(x)
                    
                    # 프롬프트 인코딩 (기본값)
                    if points is None:
                        points = torch.zeros(1, 1, 2).to(x.device)
                    prompt_features = self.prompt_encoder(points)
                    
                    # 마스크 디코딩
                    mask = self.mask_decoder(image_features)
                    return torch.sigmoid(mask)
                    
            model = BasicSAM()
            model.to(self.device)
            self.logger.info("✅ 기본 SAM 구조 생성 완료 (체크포인트 없어도 동작)")
            return model
            
        except Exception as e:
            self.logger.error(f"기본 SAM 구조 생성 실패: {e}")
            return None
            
    def get_available_models(self) -> List[str]:
        """사용 가능한 SAM 모델 목록"""
        models = []
        
        if self.models_dir and (self.models_dir / "sam.py").exists():
            models.append("sam")
            
        # 기본 모델은 항상 사용 가능
        models.append("sam_basic")
        
        return models

class ClothSegmentationModelLoader:
    """의류 세그멘테이션 통합 모델 로더"""
    
    def __init__(self, models_dir: str = None, checkpoints_dir: str = None):
        self.models_dir = Path(models_dir) if models_dir else None
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 개별 모델 로더들
        self.u2net_loader = ClothSegmentationU2NetLoader(models_dir, checkpoints_dir)
        self.deeplabv3plus_loader = ClothSegmentationDeepLabV3PlusLoader(models_dir, checkpoints_dir)
        self.sam_loader = ClothSegmentationSAMLoader(models_dir, checkpoints_dir)
        
        # 로딩된 모델들 캐시
        self.loaded_models = {}
        
    def load_u2net(self, use_checkpoint: bool = True) -> Optional[nn.Module]:
        """U2Net 모델 로딩"""
        if 'u2net' not in self.loaded_models:
            self.loaded_models['u2net'] = self.u2net_loader.load_model("u2net", use_checkpoint)
        return self.loaded_models['u2net']
        
    def load_deeplabv3plus(self, use_checkpoint: bool = True) -> Optional[nn.Module]:
        """DeepLabV3+ 모델 로딩"""
        if 'deeplabv3plus' not in self.loaded_models:
            self.loaded_models['deeplabv3plus'] = self.deeplabv3plus_loader.load_model("deeplabv3plus", use_checkpoint)
        return self.loaded_models['deeplabv3plus']
        
    def load_sam(self, use_checkpoint: bool = True) -> Optional[nn.Module]:
        """SAM 모델 로딩"""
        if 'sam' not in self.loaded_models:
            self.loaded_models['sam'] = self.sam_loader.load_model("sam", use_checkpoint)
        return self.loaded_models['sam']
        
    def load_all_models(self, use_checkpoint: bool = True) -> Dict[str, nn.Module]:
        """모든 모델 로딩"""
        models = {}
        
        models['u2net'] = self.load_u2net(use_checkpoint)
        models['deeplabv3plus'] = self.load_deeplabv3plus(use_checkpoint)
        models['sam'] = self.load_sam(use_checkpoint)
        
        # None이 아닌 모델만 반환
        return {k: v for k, v in models.items() if v is not None}
        
    def get_available_models(self) -> Dict[str, List[str]]:
        """사용 가능한 모든 모델 목록"""
        return {
            'u2net': self.u2net_loader.get_available_models(),
            'deeplabv3plus': self.deeplabv3plus_loader.get_available_models(),
            'sam': self.sam_loader.get_available_models()
        }
        
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        info = {
            'models_dir': str(self.models_dir) if self.models_dir else None,
            'checkpoints_dir': str(self.checkpoints_dir) if self.checkpoints_dir else None,
            'checkpoint_available': CHECKPOINT_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'available_models': self.get_available_models(),
            'loaded_models': list(self.loaded_models.keys()),
            'device': str(self.u2net_loader.device) if TORCH_AVAILABLE else 'N/A'
        }
        
        return info
        
    def clear_cache(self):
        """모델 캐시 정리"""
        self.loaded_models.clear()
        self.logger.info("✅ 모델 캐시 정리 완료")
        
    def reload_model(self, model_name: str, use_checkpoint: bool = True) -> Optional[nn.Module]:
        """특정 모델 재로딩"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            
        if model_name == 'u2net':
            return self.load_u2net(use_checkpoint)
        elif model_name == 'deeplabv3plus':
            return self.load_deeplabv3plus(use_checkpoint)
        elif model_name == 'sam':
            return self.load_sam(use_checkpoint)
        else:
            self.logger.error(f"알 수 없는 모델 이름: {model_name}")
            return None
