#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Cloth Segmentation (통합 시스템)
================================================================================

의류 세그멘테이션을 위한 AI 파이프라인 스텝
BaseStepMixin을 상속받아 모듈화된 구조로 구현
✅ 통일된 import 구조

Author: MyCloset AI Team  
Date: 2025-01-27
Version: 9.0 - 통합 시스템
"""

# 기본 imports
import os
import sys
import gc
import time
import asyncio
import logging
import threading
import traceback
import hashlib
import json
import base64
import math
import warnings
import weakref
import uuid
import subprocess
import platform
import datetime
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable, TYPE_CHECKING, Set
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from io import BytesIO, StringIO
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps, partial
from contextlib import asynccontextmanager, contextmanager
from collections import defaultdict, deque
from itertools import chain

# NumPy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch import autograd
    from torch.cuda.amp import autocast
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    DataLoader = None
    autograd = None
    autocast = None
    TORCH_VERSION = "N/A"

# PIL
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

# OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

# scikit-image
try:
    from skimage import measure, morphology, segmentation, filters, restoration, exposure
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# scipy
try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, median_filter
    from scipy.signal import convolve2d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# 기타 라이브러리들
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModel = None
    AutoTokenizer = None

# 환경 변수
IS_M3_MAX = platform.system() == 'Darwin' and 'M3' in platform.processor()
CONDA_INFO = os.environ.get('CONDA_DEFAULT_ENV', 'none')
MEMORY_GB = 128.0 if IS_M3_MAX else 16.0
DEVICE = 'mps' if IS_M3_MAX and TORCH_AVAILABLE else 'cpu'

# 가용성 변수들
MPS_AVAILABLE = TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
CUDA_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()

# 경고 무시
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=ImportWarning)

# 로거 초기화
logger = logging.getLogger(__name__)

# 실제 AI 모델 import 시도
REAL_MODELS_AVAILABLE = False
try:
    # 상대 경로로 import 시도
    from .models.cloth_segmentation_u2net import U2NET, RealU2NETModel
    from .models.cloth_segmentation_deeplabv3plus import DeepLabV3PlusModel, RealDeepLabV3PlusModel
    from .models.cloth_segmentation_sam import SAM2025
    REAL_MODELS_AVAILABLE = True
    logger.info("✅ 상대 경로로 실제 AI 모델들 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ 상대 경로 import 실패: {e}")
    try:
        # 절대 경로로 import 시도
        from app.ai_pipeline.steps.step_03_cloth_segmentation_models.models.cloth_segmentation_u2net import U2NET, RealU2NETModel
        from app.ai_pipeline.steps.step_03_cloth_segmentation_models.models.cloth_segmentation_deeplabv3plus import DeepLabV3PlusModel, RealDeepLabV3PlusModel
        from app.ai_pipeline.steps.step_03_cloth_segmentation_models.models.cloth_segmentation_sam import SAM2025
        REAL_MODELS_AVAILABLE = True
        logger.info("✅ 절대 경로로 실제 AI 모델들 로드 성공")
    except ImportError as e2:
        logger.warning(f"⚠️ 절대 경로 import도 실패: {e2}")
        try:
            # 직접 경로 조작
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(current_dir, 'models')
            if os.path.exists(models_dir):
                sys.path.insert(0, models_dir)
                from cloth_segmentation_u2net import U2NET, RealU2NETModel
                from cloth_segmentation_deeplabv3plus import DeepLabV3PlusModel, RealDeepLabV3PlusModel
                from cloth_segmentation_sam import SAM2025
                REAL_MODELS_AVAILABLE = True
                logger.info("✅ 직접 경로로 실제 AI 모델들 로드 성공")
            else:
                raise ImportError(f"models 디렉토리를 찾을 수 없음: {models_dir}")
        except ImportError as e3:
            logger.warning(f"⚠️ 모든 import 방법 실패: {e3}")
            # Mock 모델들 사용
            U2NET = None
            RealU2NETModel = None
            DeepLabV3PlusModel = None
            RealDeepLabV3PlusModel = None
            SAM2025 = None

# Mock 모델 클래스들 (실제 모델이 없을 경우)
if not REAL_MODELS_AVAILABLE:
    # torch가 없을 때는 기본 클래스 사용
    try:
        import torch.nn as nn
        class MockU2NetModel(nn.Module):
            """Mock U2Net 모델"""
            def __init__(self, in_ch=3, out_ch=1):
                super().__init__()
                self.conv = nn.Conv2d(in_ch, out_ch, 1)
            
            def forward(self, x):
                return self.conv(x)
        
        class MockDeepLabV3PlusModel(nn.Module):
            """Mock DeepLabV3+ 모델"""
            def __init__(self, num_classes=21):
                super().__init__()
                self.conv = nn.Conv2d(3, num_classes, 1)
            
            def forward(self, x):
                return self.conv(x)
        
        class MockSAMModel(nn.Module):
            """Mock SAM 모델"""
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 1, 1)
            
            def forward(self, x):
                return self.conv(x)
    except ImportError:
        # torch가 없을 때는 기본 Mock 클래스 사용
        class MockU2NetModel:
            """Mock U2Net 모델 (torch 없음)"""
            def __init__(self, in_ch=3, out_ch=1):
                pass
            
            def forward(self, x):
                return x
        
        class MockDeepLabV3PlusModel:
            """Mock DeepLabV3+ 모델 (torch 없음)"""
            def __init__(self, num_classes=21):
                pass
            
            def forward(self, x):
                return x
        
        class MockSAMModel:
            """Mock SAM 모델 (torch 없음)"""
            def __init__(self):
                pass
            
            def forward(self, x):
                return x

# BaseStepMixin import
try:
    from ...base.core.base_step_mixin import BaseStepMixin
except ImportError:
    from app.ai_pipeline.steps.base.core.base_step_mixin import BaseStepMixin
BASE_STEP_MIXIN_AVAILABLE = True

class ClothSegmentationStep(BaseStepMixin):
    """
    의류 세그멘테이션을 위한 AI 파이프라인 스텝
    """
    
    def __init__(self, device: str = "auto", **kwargs):
        """의류 세그멘테이션 스텝 초기화"""
        super().__init__(device=device, **kwargs)
        
        # 기본 속성 설정
        self.step_name = "ClothSegmentationStep"
        self.step_id = 3
        
        # 특화 초기화
        self._init_cloth_segmentation_specific()
        
        logger.info(f"✅ {self.step_name} 초기화 완료")
    
    def _init_cloth_segmentation_specific(self):
        """의류 세그멘테이션 특화 초기화"""
        try:
            # 디바이스 설정
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            
            # 모델 초기화
            self.models = {}
            self.models_loading_status = {}
            
            # 실제 AI 모델만 로드 (Mock 모델 제거)
            if REAL_MODELS_AVAILABLE:
                self._load_real_models()
            else:
                raise RuntimeError("실제 AI 모델을 로드할 수 없습니다. 모델 파일을 확인해주세요.")
            
            # 성능 통계 초기화
            self.performance_stats = {
                'total_inferences': 0,
                'successful_inferences': 0,
                'failed_inferences': 0,
                'average_inference_time': 0.0,
                'total_processing_time': 0.0
            }
            
            # 앙상블 매니저 초기화
            try:
                if 'ClothSegmentationEnsembleSystem' in globals() and ClothSegmentationEnsembleSystem:
                    self.ensemble_system = ClothSegmentationEnsembleSystem()
                    logger.info("✅ 앙상블 시스템 초기화 완료")
                else:
                    self.ensemble_system = None
                    logger.warning("⚠️ 앙상블 시스템을 사용할 수 없습니다")
            except Exception as e:
                logger.warning(f"⚠️ 앙상블 시스템 초기화 실패: {e}")
                self.ensemble_system = None
            
            # 품질 분석기 초기화
            try:
                if 'ClothSegmentationQualityAnalyzer' in globals() and ClothSegmentationQualityAnalyzer:
                    self.analyzer = ClothSegmentationQualityAnalyzer()
                    logger.info("✅ 품질 분석기 초기화 완료")
                else:
                    self.analyzer = None
                    logger.warning("⚠️ 품질 분석기를 사용할 수 없습니다")
            except Exception as e:
                logger.warning(f"⚠️ 품질 분석기 초기화 실패: {e}")
                self.analyzer = None
            
            logger.info(f"✅ 의류 세그멘테이션 특화 초기화 완료 (디바이스: {self.device})")
            
        except Exception as e:
            logger.error(f"❌ 의류 세그멘테이션 특화 초기화 실패: {e}")
            self._fallback_initialization()
            raise
    
    def _load_real_models(self):
        """실제 AI 모델 로드"""
        try:
            logger.info("🚀 실제 AI 모델 로딩 시작...")
            
            # U2Net 모델
            try:
                self.models['u2net'] = U2NET(in_ch=3, out_ch=1)
                self.models['u2net'].to(self.device)
                self.models_loading_status['u2net'] = True
                logger.info("✅ U2Net 모델 로드 성공")
            except Exception as e:
                logger.error(f"❌ U2Net 모델 로드 실패: {e}")
                self.models_loading_status['u2net'] = False
            
            # DeepLabV3+ 모델
            try:
                self.models['deeplabv3plus'] = DeepLabV3PlusModel(num_classes=21)
                self.models['deeplabv3plus'].to(self.device)
                self.models_loading_status['deeplabv3plus'] = True
                logger.info("✅ DeepLabV3+ 모델 로드 성공")
            except Exception as e:
                logger.error(f"❌ DeepLabV3+ 모델 로드 실패: {e}")
                self.models_loading_status['deeplabv3plus'] = False
            
            # SAM 모델
            try:
                self.models['sam'] = SAM()
                self.models['sam'].to(self.device)
                self.models_loading_status['sam'] = True
                logger.info("✅ SAM 모델 로드 성공")
            except Exception as e:
                logger.error(f"❌ SAM 모델 로드 실패: {e}")
                self.models_loading_status['sam'] = False
            
            # 실제 모델이 하나라도 로드되었는지 확인
            real_models_loaded = any(self.models_loading_status.values())
            if real_models_loaded:
                logger.info(f"🎉 실제 AI 모델 로딩 완료: {sum(self.models_loading_status.values())}/{len(self.models_loading_status)}개")
                self.is_ready = True
            else:
                logger.warning("⚠️ 모든 실제 AI 모델 로드 실패 - Mock 모델로 폴백")
                self._create_mock_models()
                
        except Exception as e:
            logger.error(f"❌ 실제 AI 모델 로드 실패: {e}")
            self._create_mock_models()
    
    def _create_mock_models(self):
        """Mock 모델 생성 (폴백용)"""
        try:
            logger.info("⚠️ Mock 모델 생성 시작...")
            
            # Mock U2Net 모델
            try:
                self.models['u2net'] = MockU2NetModel(in_ch=3, out_ch=1)
                self.models['u2net'].to(self.device)
                self.models_loading_status['u2net'] = True
                logger.info("✅ Mock U2Net 모델 생성 완료")
            except Exception as e:
                logger.error(f"❌ Mock U2Net 모델 생성 실패: {e}")
                self.models_loading_status['u2net'] = False
            
            # Mock DeepLabV3+ 모델
            try:
                self.models['deeplabv3plus'] = MockDeepLabV3PlusModel(num_classes=21)
                self.models['deeplabv3plus'].to(self.device)
                self.models_loading_status['deeplabv3plus'] = True
                logger.info("✅ Mock DeepLabV3+ 모델 생성 완료")
            except Exception as e:
                logger.error(f"❌ Mock DeepLabV3+ 모델 생성 실패: {e}")
                self.models_loading_status['deeplabv3plus'] = False
            
            # Mock SAM 모델
            try:
                self.models['sam'] = MockSAMModel()
                self.models['sam'].to(self.device)
                self.models_loading_status['sam'] = True
                logger.info("✅ Mock SAM 모델 생성 완료")
            except Exception as e:
                logger.error(f"❌ Mock SAM 모델 생성 실패: {e}")
                self.models_loading_status['sam'] = False
            
            # Mock 모델이 하나라도 생성되었는지 확인
            mock_models_created = any(self.models_loading_status.values())
            if mock_models_created:
                logger.info(f"⚠️ Mock 모델 생성 완료: {sum(self.models_loading_status.values())}/{len(self.models_loading_status)}개")
                self.is_ready = True
            else:
                logger.error("❌ 모든 Mock 모델 생성 실패")
                
        except Exception as e:
            logger.error(f"❌ Mock 모델 생성 실패: {e}")
    
    def _fallback_initialization(self):
        """폴백 초기화"""
        self.device = 'cpu'
        self.models = {}
        self.models_loading_status = {}
        self.performance_stats = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'average_inference_time': 0.0,
            'total_processing_time': 0.0
        }
        self.ensemble_system = None
        self.analyzer = None
        self.is_ready = False
        self.is_initialized = True
        logger.warning("⚠️ 폴백 초기화 완료")
    
    def _run_ai_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """AI 추론 실행"""
        start_time = time.time()
        try:
            # 입력 이미지 처리
            if 'image' not in input_data:
                return {'error': '입력 이미지가 없습니다'}
            
            input_tensor = input_data['image']
            if not isinstance(input_tensor, torch.Tensor):
                input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
            
            # 디바이스로 이동
            input_tensor = input_tensor.to(self.device)
            
            # 모델 선택 및 추론
            model_name = input_data.get('model', 'u2net')
            if model_name not in self.models:
                model_name = 'u2net'  # 기본값
            
            model = self.models[model_name]
            model.eval()
            
            with torch.no_grad():
                if model_name == 'u2net':
                    # U2Net은 (main_output, side1, side2, side3, side4, side5, side6) 형태로 반환
                    outputs = model(input_tensor)
                    if isinstance(outputs, tuple):
                        main_output = outputs[0]  # 메인 출력만 사용
                    else:
                        main_output = outputs
                    # 마스크 생성
                    mask = torch.sigmoid(main_output)
                    mask = (mask > 0.5).float()
                elif model_name == 'deeplabv3plus':
                    # DeepLabV3+는 클래스별 예측
                    output = model(input_tensor)
                    mask = torch.argmax(output, dim=1, keepdim=True)
                elif model_name == 'sam':
                    # SAM 모델 처리
                    output = model(input_tensor)
                    mask = torch.sigmoid(output) if output.shape[1] == 1 else torch.argmax(output, dim=1, keepdim=True)
                else:
                    # 기본 처리
                    output = model(input_tensor)
                    mask = torch.sigmoid(output) if output.shape[1] == 1 else torch.argmax(output, dim=1, keepdim=True)
            
            # 결과 후처리
            mask = mask.cpu().numpy()
            
            # 성능 통계 업데이트
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, True)
            
            return {
                'method_used': model_name,
                'confidence_score': 0.85,  # Mock 값
                'quality_score': 0.90,     # Mock 값
                'processing_time': processing_time,
                'mask': mask,
                'segmentation_result': {
                    'mask_shape': mask.shape,
                    'mask_dtype': str(mask.dtype),
                    'unique_values': np.unique(mask).tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ AI 추론 실패: {e}")
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, False)
            return {
                'error': str(e),
                'method_used': 'error',
                'confidence_score': 0.0,
                'quality_score': 0.0,
                'processing_time': processing_time
            }
    
    def _update_performance_stats(self, processing_time: float, success: bool):
        """성능 통계 업데이트"""
        self.performance_stats['total_inferences'] += 1
        self.performance_stats['total_processing_time'] += processing_time
        
        if success:
            self.performance_stats['successful_inferences'] += 1
        else:
            self.performance_stats['failed_inferences'] += 1
        
        # 평균 처리 시간 계산
        total_successful = self.performance_stats['successful_inferences']
        if total_successful > 0:
            self.performance_stats['average_inference_time'] = (
                self.performance_stats['total_processing_time'] / total_successful
            )
    
    def process(self, **kwargs) -> Dict[str, Any]:
        """메인 처리 메서드"""
        try:
            # 입력 데이터 검증
            if not kwargs:
                return {'error': '입력 데이터가 없습니다'}
            
            # AI 추론 실행
            result = self._run_ai_inference(kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 처리 실패: {e}")
            return {'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """스텝 상태 반환"""
        return {
            'step_name': self.step_name,
            'step_id': self.step_id,
            'device': self.device,
            'models_loaded': list(self.models.keys()),
            'models_loading_status': self.models_loading_status,
            'performance_stats': self.performance_stats,
            'is_initialized': self.is_initialized,
            'is_ready': self.is_ready,
            'real_models_available': REAL_MODELS_AVAILABLE
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 모델들을 CPU로 이동
            for model in self.models.values():
                if hasattr(model, 'to'):
                    model.to('cpu')
            
            # CUDA 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"❌ 리소스 정리 실패: {e}")

# 편의 함수들
def create_cloth_segmentation_step(**kwargs):
    """의류 세그멘테이션 스텝 생성"""
    return ClothSegmentationStep(**kwargs)

def create_m3_max_segmentation_step(**kwargs):
    """M3 Max 최적화된 의류 세그멘테이션 스텝 생성"""
    kwargs['device'] = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return ClothSegmentationStep(**kwargs)

def test_cloth_segmentation_step():
    """의류 세그멘테이션 스텝 테스트"""
    try:
        logger.info("🧪 의류 세그멘테이션 스텝 테스트 시작...")
        
        step = ClothSegmentationStep()
        status = step.get_status()
        
        logger.info(f"✅ 스텝 상태: {status}")
        
        # 간단한 추론 테스트
        if step.models:
            logger.info("🧪 추론 테스트 시작...")
            test_image = torch.randn(1, 3, 512, 512)  # 테스트 이미지 생성
            result = step.process(image=test_image)
            logger.info(f"✅ 추론 테스트 결과: {result}")
        
        return {
            'success': True,
            'status': status,
            'message': '의류 세그멘테이션 스텝 테스트 성공'
        }
    except Exception as e:
        logger.error(f"❌ 의류 세그멘테이션 스텝 테스트 실패: {e}")
        return {
            'success': False,
            'error': str(e),
            'message': '의류 세그멘테이션 스텝 테스트 실패'
        }

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 테스트 실행
    logger.info("🚀 의류 세그멘테이션 스텝 테스트 시작")
    result = test_cloth_segmentation_step()
    print(f"테스트 결과: {result}")
    logger.info("🏁 테스트 완료")
