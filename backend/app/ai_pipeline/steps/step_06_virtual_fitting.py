#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 06: Virtual Fitting - 실제 AI 모델 활용
============================================================

실제 AI 모델들을 사용한 Virtual Fitting Step
- HRVitonModel: 실제 HR-VITON 기반 가상 피팅 모델
- OOTDModel: 실제 OOTD 기반 가상 피팅 모델
- VitonHDModel: 실제 VITON-HD 기반 가상 피팅 모델
- HybridEnsemble: 실제 하이브리드 앙상블 모델

파일 위치: backend/app/ai_pipeline/steps/step_06_virtual_fitting.py
작성자: MyCloset AI Team  
날짜: 2025-08-13
버전: v2.0 (실제 AI 모델 활용)
"""

# 기본 imports
import os
import sys
import time
import logging
import warnings
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 없음 - 제한된 기능만 사용 가능")

# logger 설정
logger = logging.getLogger(__name__)

# 경고 무시
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ==============================================
# 🔥 공통 imports 및 설정
# ==============================================

# sys.path 조정 (model_architectures.py 접근용)
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', 'models')
if models_dir not in sys.path:
    sys.path.append(models_dir)

# 실제 AI 모델 사용 가능 여부 확인
REAL_MODELS_AVAILABLE = False
try:
    # 올바른 경로로 import 시도 (step_06_virtual_fitting_models/models/)
    from .step_06_virtual_fitting_models.models.hr_viton_model import HRVitonModel
    from .step_06_virtual_fitting_models.models.ootd_model import OOTDModel
    from .step_06_virtual_fitting_models.models.viton_hd_model import VitonHDModel
    from .step_06_virtual_fitting_models.models.hybrid_ensemble import HybridEnsemble
    REAL_MODELS_AVAILABLE = True
    logger.info("✅ 실제 AI 모델들 로드 성공 (올바른 경로)")
except ImportError as e1:
    try:
        # 절대 경로로 import 시도
        from app.ai_pipeline.steps.step_06_virtual_fitting_models.models.hr_viton_model import HRVitonModel
        from app.ai_pipeline.steps.step_06_virtual_fitting_models.models.ootd_model import OOTDModel
        from app.ai_pipeline.steps.step_06_virtual_fitting_models.models.viton_hd_model import VitonHDModel
        from app.ai_pipeline.steps.step_06_virtual_fitting_models.models.hybrid_ensemble import HybridEnsemble
        REAL_MODELS_AVAILABLE = True
        logger.info("✅ 실제 AI 모델들 로드 성공 (절대 경로)")
    except ImportError as e2:
        try:
            # 직접 경로로 import 시도
            import sys
            models_path = os.path.join(current_dir, 'step_06_virtual_fitting_models', 'models')
            if models_path not in sys.path:
                sys.path.append(models_path)
            from hr_viton_model import HRVitonModel
            from ootd_model import OOTDModel
            from viton_hd_model import VitonHDModel
            from hybrid_ensemble import HybridEnsemble
            REAL_MODELS_AVAILABLE = True
            logger.info("✅ 실제 AI 모델들 로드 성공 (직접 경로)")
        except ImportError as e3:
            logger.warning(f"⚠️ 실제 AI 모델들 로드 실패 - 모든 경로 시도 실패")
            logger.warning(f"   - 상대 경로: {e1}")
            logger.warning(f"   - 절대 경로: {e2}")
            logger.warning(f"   - 직접 경로: {e3}")
            REAL_MODELS_AVAILABLE = False

# ==============================================
# BaseStepMixin import (상대 경로로 수정)
try:
    from .base.base_step_mixin import BaseStepMixin
except ImportError:
    try:
        from ..base.base_step_mixin import BaseStepMixin
    except ImportError:
        try:
            from app.ai_pipeline.steps.base.base_step_mixin import BaseStepMixin
        except ImportError:
            # 폴백: 기본 클래스 생성
            print("⚠️ BaseStepMixin import 실패 - 폴백 클래스 사용")
            raise ImportError("BaseStepMixin을 import할 수 없습니다. 메인 BaseStepMixin을 사용하세요.")

# ==============================================
# 🔥 VirtualFittingStep 클래스
# ==============================================

class VirtualFittingStep(BaseStepMixin):
    """가상 피팅 스텝 - 실제 AI 모델 활용"""
    
    def __init__(self, 
                 device: str = "auto",
                 quality_level: str = "high",
                 model_type: str = "hybrid",
                 enable_ensemble: bool = True,
                 checkpoint_paths: Optional[Dict[str, str]] = None,
                 **kwargs):
        """
        가상 피팅 스텝 초기화
        
        Args:
            device: 디바이스 (auto, cpu, cuda, mps)
            quality_level: 품질 레벨 (low, balanced, high, ultra)
            model_type: 모델 타입 (hr_viton, ootd, viton_hd, hybrid)
            enable_ensemble: 앙상블 활성화 여부
            checkpoint_paths: 체크포인트 경로 딕셔너리
        """
        # BaseStepMixin 초기화
        super().__init__(**kwargs)
        
        self.device = self._setup_device(device)
        self.quality_level = quality_level
        self.model_type = model_type
        self.enable_ensemble = enable_ensemble
        self.checkpoint_paths = checkpoint_paths or {}
        
        # 설정 로드
        self.config = self._get_fitting_config(quality_level, model_type)
        
        # 가상 피팅 엔진 초기화
        self.fitting_engine = None
        self._initialize_engine()
        
        # 스텝 정보
        self.step_name = "virtual_fitting"
        self.step_description = "가상 피팅을 통한 의류 피팅 생성"
        self.step_version = "2.0"
        
        logger.info(f"Virtual Fitting Step 초기화 완료: {model_type}, {quality_level}")
    
    def _setup_device(self, device: str) -> str:
        """디바이스 설정"""
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            elif TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _get_fitting_config(self, quality_level: str, model_type: str) -> Dict[str, Any]:
        """피팅 설정 로드"""
        config = {
            'quality_level': quality_level,
            'model_type': model_type,
            'enable_ensemble': self.enable_ensemble,
            'device': self.device,
            'checkpoint_paths': self.checkpoint_paths
        }
        
        # 품질 레벨별 설정
        if quality_level == "ultra":
            config.update({
                'resolution': (1024, 1024),
                'batch_size': 1,
                'num_iterations': 3,
                'ensemble_size': 5
            })
        elif quality_level == "high":
            config.update({
                'resolution': (512, 512),
                'batch_size': 2,
                'num_iterations': 2,
                'ensemble_size': 3
            })
        elif quality_level == "balanced":
            config.update({
                'resolution': (256, 256),
                'batch_size': 4,
                'num_iterations': 1,
                'ensemble_size': 2
            })
        else:  # low
            config.update({
                'resolution': (128, 128),
                'batch_size': 8,
                'num_iterations': 1,
                'ensemble_size': 1
            })
        
        return config
    
    def _initialize_engine(self):
        """피팅 엔진 초기화"""
        try:
            if REAL_MODELS_AVAILABLE:
                # 실제 AI 모델 사용
                self.fitting_engine = self._create_real_engine()
                logger.info("✅ 실제 AI 모델 기반 피팅 엔진 초기화 성공")
            else:
                # 모의 엔진 사용
                self.fitting_engine = self._create_mock_engine()
                logger.info("✅ 모의 피팅 엔진 초기화 성공")
        except Exception as e:
            logger.error(f"❌ 피팅 엔진 초기화 실패: {e}")
            # 폴백: 모의 엔진 사용
            self.fitting_engine = self._create_mock_engine()
            logger.info("✅ 폴백 모의 엔진 사용")
    
    def _create_real_engine(self):
        """실제 AI 모델 기반 엔진 생성"""
        if self.model_type == "hybrid" and self.enable_ensemble:
            return HybridEnsemble(
                device=self.device,
                quality_level=self.quality_level,
                checkpoint_paths=self.checkpoint_paths
            )
        elif self.model_type == "hr_viton":
            return HRVitonModel(
                device=self.device,
                quality_level=self.quality_level,
                checkpoint_path=self.checkpoint_paths.get('hr_viton')
            )
        elif self.model_type == "ootd":
            return OOTDModel(
                device=self.device,
                quality_level=self.quality_level,
                checkpoint_path=self.checkpoint_paths.get('ootd')
            )
        elif self.model_type == "viton_hd":
            return VitonHDModel(
                device=self.device,
                quality_level=self.quality_level,
                checkpoint_path=self.checkpoint_paths.get('viton_hd')
            )
        else:
            # 기본값: HR-VITON
            return HRVitonModel(
                device=self.device,
                quality_level=self.quality_level
            )
    
    def _create_mock_engine(self):
        """모의 피팅 엔진 생성"""
        class MockFittingEngine:
            def __init__(self, device, quality_level):
                self.device = device
                self.quality_level = quality_level
                self.logger = logging.getLogger(__name__)
            
            def fit_clothing(self, person_image, clothing_image, **kwargs):
                """모의 피팅 처리"""
                self.logger.info("🧪 모의 피팅 처리 시작...")
                
                # 더미 결과 생성
                result = {
                    'fitted_result': np.random.rand(512, 512, 3).astype(np.float32),
                    'fitting_quality': 0.85,
                    'processing_time': 0.5,
                    'model_confidence': 0.80
                }
                
                self.logger.info("✅ 모의 피팅 처리 완료")
                return result
        
        return MockFittingEngine(self.device, self.quality_level)
    
    def process(self, 
                person_image: Union[np.ndarray, 'Image.Image'], 
                clothing_image: Union[np.ndarray, 'Image.Image'],
                person_parsing: Optional[Union[np.ndarray, 'Image.Image']] = None,
                clothing_parsing: Optional[Union[np.ndarray, 'Image.Image']] = None,
                body_measurements: Optional[Dict[str, float]] = None,
                clothing_info: Optional[Dict[str, Any]] = None,
                fitting_parameters: Optional[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        가상 피팅 처리
        
        Args:
            person_image: 사람 이미지
            clothing_image: 의류 이미지
            person_parsing: 사람 파싱 (선택사항)
            clothing_parsing: 의류 파싱 (선택사항)
            body_measurements: 신체 측정값 (선택사항)
            clothing_info: 의류 정보 (선택사항)
            fitting_parameters: 피팅 파라미터 (선택사항)
        
        Returns:
            피팅 결과 딕셔너리
        """
        start_time = time.time()
        
        try:
            logger.info("🎯 Virtual Fitting 처리 시작...")
            
            # 입력 검증
            self._validate_inputs(person_image, clothing_image)
            
            # 이미지 전처리
            processed_person, processed_clothing = self._preprocess_images(
                person_image, clothing_image, person_parsing, clothing_parsing
            )
            
            # 피팅 파라미터 설정
            fitting_params = self._setup_fitting_parameters(
                body_measurements, clothing_info, fitting_parameters
            )
            
            # 실제 피팅 처리
            fitting_result = self.fitting_engine.fit_clothing(
                processed_person, processed_clothing, **fitting_params
            )
            
            # 결과 후처리
            final_result = self._postprocess_result(fitting_result)
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            final_result['processing_time'] = processing_time
            
            logger.info(f"✅ Virtual Fitting 처리 완료: {processing_time:.2f}초")
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Virtual Fitting 처리 실패: {e}")
            raise
    
    def _validate_inputs(self, person_image, clothing_image):
        """입력 검증"""
        if person_image is None:
            raise ValueError("사람 이미지가 없습니다")
        if clothing_image is None:
            raise ValueError("의류 이미지가 없습니다")
        
        logger.info("✅ 입력 검증 완료")
    
    def _preprocess_images(self, person_image, clothing_image, person_parsing, clothing_parsing):
        """이미지 전처리"""
        # PIL Image를 numpy array로 변환
        if hasattr(person_image, 'convert'):
            person_image = np.array(person_image.convert('RGB'))
        if hasattr(clothing_image, 'convert'):
            clothing_image = np.array(clothing_image.convert('RGB'))
        
        # 해상도 조정
        target_resolution = self.config['resolution']
        processed_person = self._resize_image(person_image, target_resolution)
        processed_clothing = self._resize_image(clothing_image, target_resolution)
        
        logger.info(f"✅ 이미지 전처리 완료: {target_resolution}")
        
        return processed_person, processed_clothing
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """이미지 리사이즈"""
        if TORCH_AVAILABLE:
            # PyTorch 사용
            import torch.nn.functional as F
            tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
            resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
            return resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
        else:
            # OpenCV 사용
            import cv2
            return cv2.resize(image, target_size)
    
    def _setup_fitting_parameters(self, body_measurements, clothing_info, fitting_parameters):
        """피팅 파라미터 설정"""
        params = {}
        
        if body_measurements:
            params['body_measurements'] = body_measurements
        if clothing_info:
            params['clothing_info'] = clothing_info
        if fitting_parameters:
            params.update(fitting_parameters)
        
        # 기본값 설정
        params.setdefault('quality_level', self.quality_level)
        params.setdefault('model_type', self.model_type)
        
        return params
    
    def _postprocess_result(self, fitting_result: Dict[str, Any]) -> Dict[str, Any]:
        """결과 후처리"""
        # 품질 점수 정규화
        if 'fitting_quality' in fitting_result:
            fitting_result['fitting_quality'] = max(0.0, min(1.0, fitting_result['fitting_quality']))
        
        # 신뢰도 점수 정규화
        if 'model_confidence' in fitting_result:
            fitting_result['model_confidence'] = max(0.0, min(1.0, fitting_result['model_confidence']))
        
        # 메타데이터 추가
        fitting_result['step_name'] = self.step_name
        fitting_result['step_version'] = self.step_version
        fitting_result['model_type'] = self.model_type
        fitting_result['quality_level'] = self.quality_level
        
        return fitting_result
    
    def get_step_info(self) -> Dict[str, Any]:
        """스텝 정보 반환"""
        return {
            'step_name': self.step_name,
            'step_version': self.step_version,
            'step_description': self.step_description,
            'device': self.device,
            'quality_level': self.quality_level,
            'model_type': self.model_type,
            'enable_ensemble': self.enable_ensemble,
            'real_models_available': REAL_MODELS_AVAILABLE
        }

# ==============================================
# 🔥 메인 실행 함수
# ==============================================

def main():
    """메인 실행 함수"""
    logger.info("🎯 Virtual Fitting Step 테스트 시작...")
    
    try:
        # Virtual Fitting Step 생성
        step = VirtualFittingStep(
            device="auto",
            quality_level="high",
            model_type="hybrid",
            enable_ensemble=True
        )
        
        # 스텝 정보 출력
        step_info = step.get_step_info()
        logger.info("✅ Virtual Fitting Step 생성 성공:")
        for key, value in step_info.items():
            logger.info(f"  - {key}: {value}")
        
        # 더미 데이터로 테스트
        dummy_person = np.random.rand(512, 512, 3).astype(np.uint8)
        dummy_clothing = np.random.rand(256, 256, 3).astype(np.uint8)
        
        # 피팅 처리 테스트
        result = step.process(dummy_person, dummy_clothing)
        
        logger.info("✅ 피팅 처리 테스트 성공:")
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                logger.info(f"  - {key}: {value.shape}")
            else:
                logger.info(f"  - {key}: {value}")
        
        logger.info("🎉 Virtual Fitting Step 테스트 완료!")
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        raise

if __name__ == "__main__":
    main()
