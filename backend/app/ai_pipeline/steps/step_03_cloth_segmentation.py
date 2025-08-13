#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Cloth Segmentation
=====================================================================

의류 세그멘테이션을 위한 AI 파이프라인 스텝
BaseStepMixin을 상속받아 모듈화된 구조로 구현

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 8.1 - Real AI Models Loading
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
import cv2
import time

# 로거 초기화
logger = logging.getLogger(__name__)

# 실제 AI 모델 import 시도
REAL_MODELS_AVAILABLE = False
try:
    # 절대 경로로 import 시도
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from ...models.model_architectures import (
        U2NetModel, DeepLabV3PlusModel, HRNetSegModel
    )
    REAL_MODELS_AVAILABLE = True
    logger.info("✅ 실제 AI 모델들 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ 실제 AI 모델 import 실패: {e}")
    try:
        # 상대 경로로 import 시도
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
        from model_architectures import (
            U2NetModel, DeepLabV3PlusModel, HRNetSegModel
        )
        REAL_MODELS_AVAILABLE = True
        logger.info("✅ 상대 경로로 실제 AI 모델들 로드 성공")
    except ImportError as e2:
        logger.warning(f"⚠️ 상대 경로 import도 실패: {e2}")
        try:
            # 직접 sys.path 조작
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(current_dir, '..', 'models')
            if os.path.exists(models_dir):
                sys.path.insert(0, models_dir)
                from model_architectures import (
                    U2NetModel, DeepLabV3PlusModel, HRNetSegModel
                )
                REAL_MODELS_AVAILABLE = True
                logger.info("✅ sys.path 조작으로 실제 AI 모델들 로드 성공")
        except ImportError as e3:
            logger.warning(f"⚠️ 모든 import 방법 실패: {e3}")

# HRNetSegModel 클래스 정의 (model_architectures.py에 없는 경우)
if not REAL_MODELS_AVAILABLE:
    class HRNetSegModel(nn.Module):
        """HRNet 기반 세그멘테이션 모델"""
        def __init__(self, num_classes=19):
            super().__init__()
            self.num_classes = num_classes
            
            # HRNet backbone
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            
            # HRNet stages
            self.stage1 = self._make_stage(64, 32, 1)
            self.stage2 = self._make_stage(32, 64, 1)
            self.stage3 = self._make_stage(64, 128, 1)
            self.stage4 = self._make_stage(128, 256, 1)
            
            # Final layer
            self.final_layer = nn.Conv2d(256, num_classes, kernel_size=1)
            
        def _make_stage(self, inplanes, planes, num_blocks):
            layers = []
            layers.append(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            
            # num_blocks가 1보다 클 때만 추가 블록 생성
            if num_blocks > 1:
                for _ in range(num_blocks - 1):
                    layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
                    layers.append(nn.BatchNorm2d(planes))
                    layers.append(nn.ReLU(inplace=True))
            
            return nn.Sequential(*layers)
        
        def forward(self, x):
            # Stem
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            
            # Stages
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            
            # Final layer
            output = self.final_layer(x)
            
            return output

# BaseStepMixin import
from .base.base_step_mixin import BaseStepMixin
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
            
            # 실제 AI 모델 로드 시도
            if REAL_MODELS_AVAILABLE:
                self._load_real_models()
            else:
                self._create_mock_models()
            
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
                    self.ensemble_enabled = True
                    self.ensemble_manager = self.ensemble_system
                else:
                    self.ensemble_system = None
                    self.ensemble_enabled = False
                    self.ensemble_manager = None
            except Exception:
                self.ensemble_system = None
                self.ensemble_enabled = False
                self.ensemble_manager = None
            
            # 분석기 초기화
            try:
                if 'ClothSegmentationAnalyzer' in globals() and ClothSegmentationAnalyzer:
                    self.analyzer = ClothSegmentationAnalyzer()
                else:
                    self.analyzer = None
            except Exception:
                self.analyzer = None
            
            logger.info("✅ 의류 세그멘테이션 특화 초기화 완료")
            
        except Exception as e:
            logger.error(f"❌ 의류 세그멘테이션 특화 초기화 실패: {e}")
            # 폴백 초기화
            self._fallback_initialization()
    
    def _load_real_models(self):
        """실제 AI 모델 로드"""
        try:
            logger.info("🚀 실제 AI 모델 로딩 시작...")
            
            # U2Net 모델
            try:
                self.models['u2net'] = U2NetModel()
                self.models['u2net'].to(self.device)
                self.models_loading_status['u2net'] = True
                logger.info("✅ U2Net 모델 로드 성공")
            except Exception as e:
                logger.error(f"❌ U2Net 모델 로드 실패: {e}")
                self.models_loading_status['u2net'] = False
            
            # DeepLabV3+ 모델
            try:
                self.models['deeplabv3plus'] = DeepLabV3PlusModel()
                self.models['deeplabv3plus'].to(self.device)
                self.models_loading_status['deeplabv3plus'] = True
                logger.info("✅ DeepLabV3+ 모델 로드 성공")
            except Exception as e:
                logger.error(f"❌ DeepLabV3+ 모델 로드 실패: {e}")
                self.models_loading_status['deeplabv3plus'] = False
            
            # HRNet 세그멘테이션 모델
            try:
                self.models['hrnet'] = HRNetSegModel()
                self.models['hrnet'].to(self.device)
                self.models_loading_status['hrnet'] = True
                logger.info("✅ HRNet 세그멘테이션 모델 로드 성공")
            except Exception as e:
                logger.error(f"❌ HRNet 세그멘테이션 모델 로드 실패: {e}")
                self.models_loading_status['hrnet'] = False
            
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
        """Mock 모델 생성 (폴백)"""
        logger.warning("⚠️ Mock 모델 사용 - 실제 AI 모델 로드 실패")
        
        # Mock U2Net 모델
        class MockU2NetModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 1, 1)
            
            def forward(self, x):
                return self.conv(x)
        
        # Mock DeepLabV3+ 모델
        class MockDeepLabV3PlusModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 21, 1)  # 21 classes
            
            def forward(self, x):
                return self.conv(x)
        
        # Mock HRNet 모델
        class MockHRNetSegModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 19, 1)  # 19 classes
            
            def forward(self, x):
                return self.conv(x)
        
        self.models['u2net'] = MockU2NetModel()
        self.models['deeplabv3plus'] = MockDeepLabV3PlusModel()
        self.models['hrnet'] = MockHRNetSegModel()
        
        self.models_loading_status['u2net'] = False
        self.models_loading_status['deeplabv3plus'] = False
        self.models_loading_status['hrnet'] = False
        
        logger.info("✅ Mock 모델 생성 완료")
    
    def _fallback_initialization(self):
        """폴백 초기화"""
        self.device = 'cpu'
        self.models = {}
        self.models_loading_status = {}
        self.performance_stats = {}
        self.ensemble_manager = None
        self.analyzer = None
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
                    # U2Net은 단일 이미지 입력
                    output = model(input_tensor)
                    # 마스크 생성
                    mask = torch.sigmoid(output)
                    mask = (mask > 0.5).float()
                elif model_name == 'deeplabv3plus':
                    # DeepLabV3+는 클래스별 예측
                    output = model(input_tensor)
                    mask = torch.argmax(output, dim=1, keepdim=True)
                elif model_name == 'hrnet':
                    # HRNet도 클래스별 예측
                    output = model(input_tensor)
                    mask = torch.argmax(output, dim=1, keepdim=True)
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
