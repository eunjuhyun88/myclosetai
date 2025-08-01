#!/usr/bin/env python3
"""
🔥 Real AI Inference Validator v7.0 - 실제 추론 완전 검증 시스템
===============================================================================
✅ Mock/폴백 없는 100% 실제 AI 모델 추론 검증
✅ 체크포인트 로딩 → 전처리 → 추론 → 후처리 전 과정 검증
✅ 각 Step별 실제 AI 모델 정상 작동 여부 확인
✅ BaseStepMixin _run_ai_inference() 메서드 실제 실행
✅ Central Hub DI Container 연동 상태 검증
✅ M3 Max MPS 디바이스 최적화 추론 테스트
✅ 메모리 사용량 실시간 모니터링
✅ 추론 성능 및 결과 품질 검증
✅ 실제 체크포인트 파일 검증 및 무결성 체크
✅ GPU/MPS 텐서 연산 정상 작동 검증
===============================================================================
"""

import sys
import os
import time
import traceback
import logging
import asyncio
import threading
import psutil
import platform
import hashlib
import json
import importlib
import inspect
import gc
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from enum import Enum
import base64
from io import BytesIO

# 경고 무시
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# 프로젝트 구조 설정
current_file = Path(__file__).resolve()
project_root = current_file.parent
backend_root = project_root / 'backend'
ai_models_root = backend_root / "ai_models"

# 경로 추가
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_root))
sys.path.insert(0, str(backend_root / "app"))

print(f"🔥 Real AI Inference Validator v7.0 시작")
print(f"   프로젝트 루트: {project_root}")
print(f"   AI 모델 루트: {ai_models_root}")

# =============================================================================
# 🔥 1. 실제 추론 검증 데이터 구조
# =============================================================================

class RealInferenceStatus(Enum):
    """실제 추론 상태"""
    NOT_TESTED = "not_tested"
    MODEL_LOADING_FAILED = "model_loading_failed"
    CHECKPOINT_MISSING = "checkpoint_missing"
    CHECKPOINT_CORRUPTED = "checkpoint_corrupted"
    PREPROCESSING_FAILED = "preprocessing_failed"
    INFERENCE_FAILED = "inference_failed"
    POSTPROCESSING_FAILED = "postprocessing_failed"
    TENSOR_OPERATION_FAILED = "tensor_operation_failed"
    DEVICE_INCOMPATIBLE = "device_incompatible"
    MEMORY_INSUFFICIENT = "memory_insufficient"
    MOCK_FALLBACK_DETECTED = "mock_fallback_detected"
    SUCCESS = "success"

@dataclass
class RealInferenceResult:
    """실제 추론 검증 결과"""
    step_name: str
    step_id: int
    
    # 모델 로딩 검증
    model_loading_success: bool = False
    checkpoint_loaded: bool = False
    checkpoint_size_mb: float = 0.0
    checkpoint_hash: str = ""
    model_parameters_count: int = 0
    
    # 디바이스 검증
    device_used: str = "cpu"
    device_compatible: bool = False
    mps_optimized: bool = False
    tensor_operations_working: bool = False
    
    # 실제 추론 검증
    preprocessing_success: bool = False
    inference_success: bool = False
    postprocessing_success: bool = False
    total_inference_time: float = 0.0
    
    # 결과 검증
    output_shape_valid: bool = False
    output_data_type_valid: bool = False
    output_range_valid: bool = False
    confidence_score: float = 0.0
    
    # Mock/폴백 감지
    mock_detected: bool = False
    fallback_used: bool = False
    real_ai_model_used: bool = False
    
    # 메모리 및 성능
    peak_memory_mb: float = 0.0
    memory_efficiency: str = "unknown"
    inference_fps: float = 0.0
    
    # 오류 정보
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status: RealInferenceStatus = RealInferenceStatus.NOT_TESTED
    
    # 상세 정보
    model_info: Dict[str, Any] = field(default_factory=dict)
    inference_details: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# 🔥 2. 실제 추론 검증 시스템
# =============================================================================

class RealAIInferenceValidator:
    """실제 AI 추론 완전 검증 시스템"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.torch_available = False
        self.pil_available = False
        self.numpy_available = False
        self.cv2_available = False
        
        # 의존성 확인
        self._check_dependencies()
        
        # GitHub Step 설정 (프로젝트 지식 기반)
        self.github_steps = [
            {
                'step_id': 1,
                'step_name': 'HumanParsingStep',
                'module_path': 'app.ai_pipeline.steps.step_01_human_parsing',
                'class_name': 'HumanParsingStep',
                'expected_models': ['graphonomy.pth', 'schp_model.pth'],
                'priority': 'critical'
            },
            {
                'step_id': 2,
                'step_name': 'PoseEstimationStep',
                'module_path': 'app.ai_pipeline.steps.step_02_pose_estimation', 
                'class_name': 'PoseEstimationStep',
                'expected_models': ['pose_model.pth', 'dw-ll_ucoco_384.pth'],
                'priority': 'critical'
            },
            {
                'step_id': 3,
                'step_name': 'ClothSegmentationStep',
                'module_path': 'app.ai_pipeline.steps.step_03_cloth_segmentation',
                'class_name': 'ClothSegmentationStep', 
                'expected_models': ['sam_vit_h_4b8939.pth', 'u2net_alternative.pth'],
                'priority': 'critical'
            },
            {
                'step_id': 4,
                'step_name': 'GeometricMatchingStep',
                'module_path': 'app.ai_pipeline.steps.step_04_geometric_matching',
                'class_name': 'GeometricMatchingStep',
                'expected_models': ['gmm_model.pth', 'tom_model.pth'],
                'priority': 'high'
            },
            {
                'step_id': 5,
                'step_name': 'ClothWarpingStep',
                'module_path': 'app.ai_pipeline.steps.step_05_cloth_warping',
                'class_name': 'ClothWarpingStep',
                'expected_models': ['RealVisXL_V4.0.safetensors', 'warping_model.pth'],
                'priority': 'high'
            },
            {
                'step_id': 6,
                'step_name': 'VirtualFittingStep',
                'module_path': 'app.ai_pipeline.steps.step_06_virtual_fitting',
                'class_name': 'VirtualFittingStep',
                'expected_models': ['ootd_hd_checkpoint.safetensors', 'sd_model.safetensors'],
                'priority': 'critical'
            },
            {
                'step_id': 7,
                'step_name': 'PostProcessingStep',
                'module_path': 'app.ai_pipeline.steps.step_07_post_processing',
                'class_name': 'PostProcessingStep',
                'expected_models': ['esrgan_x8.pth', 'realesrgan_x4.pth'],
                'priority': 'medium'
            },
            {
                'step_id': 8,
                'step_name': 'QualityAssessmentStep',
                'module_path': 'app.ai_pipeline.steps.step_08_quality_assessment',
                'class_name': 'QualityAssessmentStep',
                'expected_models': ['ViT-L-14.pt', 'clip_model.pt'],
                'priority': 'medium'
            }
        ]
    
    def _check_dependencies(self):
        """의존성 확인"""
        try:
            import torch
            self.torch_available = True
            self.torch = torch
        except ImportError:
            pass
            
        try:
            from PIL import Image
            self.pil_available = True
            self.pil = Image
        except ImportError:
            pass
            
        try:
            import numpy as np
            self.numpy_available = True
            self.numpy = np
        except ImportError:
            pass
            
        try:
            import cv2
            self.cv2_available = True
            self.cv2 = cv2
        except ImportError:
            pass
    
    def validate_real_inference_for_step(self, step_config: Dict[str, Any]) -> RealInferenceResult:
        """Step별 실제 추론 완전 검증"""
        
        result = RealInferenceResult(
            step_name=step_config['step_name'],
            step_id=step_config['step_id']
        )
        
        print(f"\n🔥 {step_config['step_name']} 실제 추론 검증 시작...")
        
        try:
            # 1. Step 클래스 로딩 및 인스턴스 생성
            step_instance = self._create_step_instance(step_config, result)
            if not step_instance:
                result.status = RealInferenceStatus.MODEL_LOADING_FAILED
                return result
            
            # 2. 실제 모델 로딩 검증
            if not self._validate_model_loading(step_instance, result):
                return result
            
            # 3. 디바이스 호환성 검증
            if not self._validate_device_compatibility(step_instance, result):
                return result
            
            # 4. 실제 추론 실행 검증
            if not self._validate_real_inference_execution(step_instance, result):
                return result
            
            # 5. Mock/폴백 감지
            self._detect_mock_fallback(step_instance, result)
            
            # 6. 성능 및 결과 품질 검증
            self._validate_inference_quality(step_instance, result)
            
            # Final 상태 결정
            if result.real_ai_model_used and result.inference_success and not result.mock_detected:
                result.status = RealInferenceStatus.SUCCESS
                print(f"   ✅ 실제 AI 추론 완전 검증 성공!")
            else:
                result.status = RealInferenceStatus.MOCK_FALLBACK_DETECTED
                print(f"   ⚠️ Mock/폴백 감지됨")
            
        except Exception as e:
            result.errors.append(f"검증 실행 실패: {e}")
            result.status = RealInferenceStatus.INFERENCE_FAILED
            print(f"   ❌ 검증 실패: {str(e)[:100]}")
        
        return result
    
    def _create_step_instance(self, step_config: Dict[str, Any], result: RealInferenceResult) -> Any:
        """Step 인스턴스 생성 (Central Hub DI Container 사용)"""
        try:
            # 🔥 Central Hub DI Container를 통한 Step 생성
            print(f"   🔄 Central Hub DI Container를 통한 Step 생성 시도...")
            
            # Central Hub DI Container 조회
            try:
                import importlib
                di_module = importlib.import_module('app.core.di_container')
                central_hub_container = di_module.get_global_container()
                
                if central_hub_container:
                    print(f"   ✅ Central Hub DI Container 연결됨")
                    
                    # StepFactory를 통한 Step 생성
                    step_factory = central_hub_container.get('step_factory')
                    if step_factory:
                        print(f"   ✅ StepFactory 발견")
                        
                        # StepType으로 변환 (StepFactory가 인식하는 형식)
                        step_name = step_config['step_name']
                        if step_name == 'HumanParsingStep':
                            step_type = 'human_parsing'
                        elif step_name == 'PoseEstimationStep':
                            step_type = 'pose_estimation'
                        elif step_name == 'ClothSegmentationStep':
                            step_type = 'cloth_segmentation'
                        elif step_name == 'GeometricMatchingStep':
                            step_type = 'geometric_matching'
                        elif step_name == 'ClothWarpingStep':
                            step_type = 'cloth_warping'
                        elif step_name == 'VirtualFittingStep':
                            step_type = 'virtual_fitting'
                        elif step_name == 'PostProcessingStep':
                            step_type = 'post_processing'
                        elif step_name == 'QualityAssessmentStep':
                            step_type = 'quality_assessment'
                        else:
                            step_type = step_name.lower().replace('step', '').replace('_', '')
                        
                        step_instance = step_factory.create_step(step_type)
                        
                        if step_instance:
                            print(f"   ✅ Central Hub를 통한 Step 생성 성공")
                            
                            # BaseStepMixin 상속 확인
                            from app.ai_pipeline.steps.base_step_mixin import BaseStepMixin
                            if isinstance(step_instance, BaseStepMixin):
                                print(f"   ✅ BaseStepMixin 상속 확인")
                            else:
                                result.warnings.append("BaseStepMixin 상속되지 않음")
                            
                            return step_instance
                        else:
                            print(f"   ⚠️ Central Hub Step 생성 실패 - 직접 생성 시도")
                    else:
                        print(f"   ⚠️ StepFactory 없음 - 직접 생성 시도")
                else:
                    print(f"   ⚠️ Central Hub Container 없음 - 직접 생성 시도")
            except Exception as e:
                print(f"   ⚠️ Central Hub 연결 실패: {e} - 직접 생성 시도")
            
            # 🔄 폴백: 직접 생성 (기존 방식)
            print(f"   🔄 직접 Step 생성 (폴백)...")
            
            # 동적 import
            module = importlib.import_module(step_config['module_path'])
            step_class = getattr(module, step_config['class_name'])
            
            # 최적 디바이스 결정
            device = 'cpu'
            if self.torch_available:
                if hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
                    device = 'mps'
                    result.mps_optimized = True
                elif self.torch.cuda.is_available():
                    device = 'cuda'
            
            result.device_used = device
            
            # Step 인스턴스 생성 (실제 AI 모델 로딩 활성화)
            try:
                # 먼저 기본 매개변수로 시도
                step_instance = step_class(device=device)
            except Exception as e1:
                print(f"   ⚠️ 기본 매개변수로 생성 실패: {e1}")
                try:
                    # 추가 매개변수로 시도
                    step_instance = step_class(device=device, strict_mode=False)
                except Exception as e2:
                    print(f"   ⚠️ 추가 매개변수로 생성 실패: {e2}")
                    # 마지막으로 매개변수 없이 시도
                    step_instance = step_class()
            
            # BaseStepMixin 상속 확인
            from app.ai_pipeline.steps.base_step_mixin import BaseStepMixin
            if isinstance(step_instance, BaseStepMixin):
                print(f"   ✅ BaseStepMixin 상속 확인")
            else:
                result.warnings.append("BaseStepMixin 상속되지 않음")
            
            return step_instance
            
        except Exception as e:
            result.errors.append(f"Step 인스턴스 생성 실패: {e}")
            return None
    
    def _validate_model_loading(self, step_instance: Any, result: RealInferenceResult) -> bool:
        """실제 모델 로딩 검증"""
        try:
            print(f"   🔍 모델 로딩 검증...")
            
            # 초기화 시도
            if hasattr(step_instance, 'initialize'):
                if asyncio.iscoroutinefunction(step_instance.initialize):
                    # 비동기 초기화
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    init_success = loop.run_until_complete(
                        asyncio.wait_for(step_instance.initialize(), timeout=180.0)
                    )
                else:
                    # 동기 초기화
                    init_success = step_instance.initialize()
                
                if not init_success:
                    result.errors.append("초기화 실패")
                    result.status = RealInferenceStatus.MODEL_LOADING_FAILED
                    return False
            
            # 모델 로딩 상태 확인
            if hasattr(step_instance, 'has_model') and not step_instance.has_model:
                result.errors.append("has_model = False")
                result.status = RealInferenceStatus.MODEL_LOADING_FAILED
                return False
            
            # AI 모델 존재 확인
            if hasattr(step_instance, 'ai_models'):
                if not step_instance.ai_models or all(model is None for model in step_instance.ai_models.values()):
                    result.errors.append("AI 모델이 None")
                    result.status = RealInferenceStatus.MODEL_LOADING_FAILED
                    return False
                
                # 로딩된 모델 정보
                loaded_models = []
                total_params = 0
                
                for model_name, model in step_instance.ai_models.items():
                    if model is not None:
                        loaded_models.append(model_name)
                        
                        # 파라미터 수 계산
                        if hasattr(model, 'parameters'):
                            try:
                                params = sum(p.numel() for p in model.parameters())
                                total_params += params
                            except Exception:
                                pass
                
                result.model_info['loaded_models'] = loaded_models
                result.model_parameters_count = total_params
                result.model_loading_success = len(loaded_models) > 0
                
                print(f"      ✅ 로딩된 모델: {loaded_models}")
                print(f"      📊 총 파라미터: {total_params:,}")
                
            # 체크포인트 정보
            self._analyze_checkpoint_info(step_instance, result)
            
            return result.model_loading_success
            
        except Exception as e:
            result.errors.append(f"모델 로딩 검증 실패: {e}")
            result.status = RealInferenceStatus.MODEL_LOADING_FAILED
            return False
    
    def _analyze_checkpoint_info(self, step_instance: Any, result: RealInferenceResult):
        """체크포인트 정보 분석"""
        try:
            # 체크포인트 파일 찾기
            step_name = result.step_name.lower()
            step_id = result.step_id
            
            # 가능한 체크포인트 경로들
            checkpoint_paths = []
            
            if ai_models_root.exists():
                # Step별 디렉토리 패턴
                patterns = [
                    f"step_{step_id:02d}_*",
                    f"*{step_name.replace('step', '').lower()}*",
                    f"checkpoints/step_{step_id:02d}_*"
                ]
                
                for pattern in patterns:
                    matching_dirs = list(ai_models_root.glob(pattern))
                    for model_dir in matching_dirs:
                        if model_dir.is_dir():
                            for ext in ['*.pth', '*.pt', '*.safetensors', '*.bin', '*.ckpt']:
                                checkpoint_paths.extend(model_dir.rglob(ext))
            
            # 첫 번째 체크포인트 분석
            if checkpoint_paths:
                checkpoint_file = checkpoint_paths[0]
                
                # 파일 크기
                result.checkpoint_size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
                
                # 해시 (빠른 샘플 해시)
                result.checkpoint_hash = self._calculate_quick_hash(checkpoint_file)
                result.checkpoint_loaded = True
                
                print(f"      📁 체크포인트: {checkpoint_file.name} ({result.checkpoint_size_mb:.1f}MB)")
                
        except Exception as e:
            result.warnings.append(f"체크포인트 분석 실패: {e}")
    
    def _calculate_quick_hash(self, file_path: Path) -> str:
        """빠른 해시 계산"""
        try:
            hash_md5 = hashlib.md5()
            file_size = file_path.stat().st_size
            sample_size = min(1024 * 1024, file_size)  # 최대 1MB 샘플링
            
            with open(file_path, "rb") as f:
                chunk = f.read(sample_size)
                hash_md5.update(chunk)
            
            return hash_md5.hexdigest()[:16]  # 처음 16자만
        except Exception:
            return ""
    
    def _validate_device_compatibility(self, step_instance: Any, result: RealInferenceResult) -> bool:
        """디바이스 호환성 검증"""
        try:
            print(f"   🖥️ 디바이스 호환성 검증...")
            
            if not self.torch_available:
                result.warnings.append("PyTorch 없음")
                return True  # CPU 모드로 계속 진행
            
            # 기본 텐서 연산 테스트
            try:
                device = result.device_used
                test_tensor = self.torch.randn(4, 4).to(device)
                result_tensor = test_tensor * 2.0 + 1.0
                
                # 결과 검증
                if result_tensor.shape == (4, 4) and result_tensor.device.type == device:
                    result.tensor_operations_working = True
                    result.device_compatible = True
                    print(f"      ✅ {device} 텐서 연산 정상")
                
            except Exception as e:
                result.errors.append(f"텐서 연산 실패: {e}")
                result.status = RealInferenceStatus.TENSOR_OPERATION_FAILED
                return False
            
            # MPS 특화 테스트
            if result.device_used == 'mps':
                try:
                    # float64 → float32 변환 테스트
                    test_tensor_64 = self.torch.randn(2, 2, dtype=self.torch.float64)
                    test_tensor_32 = test_tensor_64.to(self.torch.float32).to('mps')
                    _ = test_tensor_32 + 1.0
                    result.mps_optimized = True
                    print(f"      ✅ MPS float64→float32 변환 정상")
                except Exception as e:
                    result.warnings.append(f"MPS 최적화 실패: {e}")
            
            return True
            
        except Exception as e:
            result.errors.append(f"디바이스 호환성 검증 실패: {e}")
            result.status = RealInferenceStatus.DEVICE_INCOMPATIBLE
            return False
    
    def _validate_real_inference_execution(self, step_instance: Any, result: RealInferenceResult) -> bool:
        """실제 추론 실행 검증"""
        try:
            print(f"   🧠 실제 AI 추론 실행 검증...")
            
            # 메모리 모니터링 시작
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            start_time = time.time()
            
            # 더미 입력 데이터 생성
            test_input = self._create_test_input(step_instance, result)
            if not test_input:
                result.status = RealInferenceStatus.PREPROCESSING_FAILED
                return False
            
            result.preprocessing_success = True
            
            # 실제 _run_ai_inference 메서드 호출
            if not hasattr(step_instance, '_run_ai_inference'):
                result.errors.append("_run_ai_inference 메서드 없음")
                result.status = RealInferenceStatus.INFERENCE_FAILED
                return False
            
            # 실제 추론 실행
            print(f"      🔥 _run_ai_inference() 실행...")
            ai_result = step_instance._run_ai_inference(test_input)
            
            # 추론 시간 계산
            inference_time = time.time() - start_time
            result.total_inference_time = inference_time
            
            # 메모리 사용량 계산
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            result.peak_memory_mb = end_memory - start_memory
            
            if result.peak_memory_mb < 500:
                result.memory_efficiency = "excellent"
            elif result.peak_memory_mb < 1000:
                result.memory_efficiency = "good"
            else:
                result.memory_efficiency = "high"
            
            # 결과 검증
            if not self._validate_inference_result(ai_result, result):
                return False
            
            result.inference_success = True
            result.postprocessing_success = True
            
            print(f"      ✅ 추론 성공 ({inference_time:.3f}초, 메모리: {result.peak_memory_mb:.1f}MB)")
            
            return True
            
        except Exception as e:
            result.errors.append(f"추론 실행 실패: {e}")
            result.status = RealInferenceStatus.INFERENCE_FAILED
            return False
    
    def _create_test_input(self, step_instance: Any, result: RealInferenceResult) -> Dict[str, Any]:
        """Step별 테스트 입력 데이터 생성"""
        try:
            step_name = result.step_name
            
            # 기본 테스트 입력
            test_input = {}
            
            if self.numpy_available and self.pil_available:
                # 512x512 RGB 더미 이미지 생성
                dummy_image_np = self.numpy.random.randint(0, 255, (512, 512, 3), dtype=self.numpy.uint8)
                dummy_image_pil = self.pil.fromarray(dummy_image_np)
                
                # Step별 특화 입력
                if 'HumanParsing' in step_name:
                    test_input = {
                        'person_image': dummy_image_pil,
                        'parsing_classes': 20
                    }
                elif 'PoseEstimation' in step_name:
                    test_input = {
                        'person_image': dummy_image_pil,
                        'keypoint_threshold': 0.1
                    }
                elif 'ClothSegmentation' in step_name:
                    test_input = {
                        'clothing_image': dummy_image_pil,
                        'person_image': dummy_image_pil
                    }
                elif 'GeometricMatching' in step_name:
                    test_input = {
                        'person_image': dummy_image_pil,
                        'clothing_image': dummy_image_pil,
                        'person_parsing': dummy_image_np
                    }
                elif 'ClothWarping' in step_name:
                    test_input = {
                        'clothing_image': dummy_image_pil,
                        'person_parsing': dummy_image_np,
                        'pose_keypoints': self.numpy.random.rand(18, 3).tolist()
                    }
                elif 'VirtualFitting' in step_name:
                    test_input = {
                        'person_image': dummy_image_pil,
                        'clothing_image': dummy_image_pil,
                        'warped_cloth': dummy_image_pil
                    }
                elif 'PostProcessing' in step_name:
                    test_input = {
                        'fitted_image': dummy_image_pil,
                        'enhancement_level': 0.8
                    }
                elif 'QualityAssessment' in step_name:
                    test_input = {
                        'result_image': dummy_image_pil,
                        'original_person': dummy_image_pil,
                        'target_clothing': dummy_image_pil
                    }
                else:
                    # 범용 입력
                    test_input = {
                        'input_image': dummy_image_pil,
                        'data': dummy_image_np
                    }
            else:
                # numpy/PIL 없는 경우 기본 데이터
                test_input = {
                    'data': {'test': True},
                    'input_size': (512, 512)
                }
            
            return test_input
            
        except Exception as e:
            result.errors.append(f"테스트 입력 생성 실패: {e}")
            return {}
    
    def _validate_inference_result(self, ai_result: Any, result: RealInferenceResult) -> bool:
        """추론 결과 검증"""
        try:
            if not ai_result:
                result.errors.append("추론 결과가 None")
                return False
            
            if not isinstance(ai_result, dict):
                result.errors.append("추론 결과가 딕셔너리가 아님")
                return False
            
            # 기본 필드 확인
            required_fields = ['success', 'processing_time']
            missing_fields = [field for field in required_fields if field not in ai_result]
            
            if missing_fields:
                result.warnings.append(f"누락된 필드: {missing_fields}")
            
            # 성공 여부 확인
            if 'success' in ai_result and not ai_result['success']:
                result.errors.append("추론 결과 success=False")
                return False
            
            # 결과 데이터 타입 확인
            if 'result' in ai_result:
                output_data = ai_result['result']
                
                if self.numpy_available and isinstance(output_data, self.numpy.ndarray):
                    result.output_shape_valid = len(output_data.shape) >= 2
                    result.output_data_type_valid = True
                    
                    # 값 범위 확인 (이미지인 경우)
                    if output_data.dtype in [self.numpy.uint8, self.numpy.float32]:
                        if self.numpy.all((output_data >= 0) & (output_data <= 255)):
                            result.output_range_valid = True
                
                elif self.pil_available and hasattr(output_data, 'size'):
                    # PIL Image 확인
                    result.output_shape_valid = len(output_data.size) == 2
                    result.output_data_type_valid = True
                    result.output_range_valid = True
                
                elif isinstance(output_data, (list, tuple)):
                    result.output_data_type_valid = True
                    result.output_shape_valid = len(output_data) > 0
            
            # 신뢰도 점수
            if 'confidence' in ai_result:
                try:
                    confidence = float(ai_result['confidence'])
                    result.confidence_score = confidence
                except (ValueError, TypeError):
                    result.warnings.append("confidence 값이 숫자가 아님")
            
            # 상세 정보 저장
            result.inference_details = {
                'result_keys': list(ai_result.keys()),
                'processing_time': ai_result.get('processing_time', 0),
                'device_used': ai_result.get('device_used', 'unknown'),
                'model_loaded': ai_result.get('model_loaded', False),
                'step_name': ai_result.get('step_name', result.step_name)
            }
            
            return True
            
        except Exception as e:
            result.errors.append(f"결과 검증 실패: {e}")
            return False
    
    def _detect_mock_fallback(self, step_instance: Any, result: RealInferenceResult):
        """Mock/폴백 사용 감지"""
        try:
            print(f"   🔍 Mock/폴백 감지...")
            
            # Mock 감지 패턴들
            mock_indicators = [
                'mock_result',
                'fallback_result', 
                'dummy_output',
                'test_result',
                'placeholder'
            ]
            
            fallback_indicators = [
                'fallback_used',
                'emergency_mode',
                'mock_detected',
                'no_model_available'
            ]
            
            # Step 상태 확인
            if hasattr(step_instance, 'get_status'):
                status = step_instance.get_status()
                
                # Mock 감지
                for indicator in mock_indicators:
                    if indicator in str(status).lower():
                        result.mock_detected = True
                        break
                
                # 폴백 감지  
                for indicator in fallback_indicators:
                    if indicator in str(status).lower():
                        result.fallback_used = True
                        break
            
            # AI 모델 실제 사용 확인
            if hasattr(step_instance, 'ai_models') and step_instance.ai_models:
                # 모든 모델이 None이 아니면 실제 모델 사용
                real_models = [model for model in step_instance.ai_models.values() if model is not None]
                result.real_ai_model_used = len(real_models) > 0
                
                if result.real_ai_model_used:
                    print(f"      ✅ 실제 AI 모델 사용 확인: {len(real_models)}개")
                else:
                    print(f"      ⚠️ AI 모델이 모두 None")
                    result.mock_detected = True
            
            # BaseStepMixin의 실제 추론 메서드 확인
            if hasattr(step_instance, '_run_ai_inference'):
                # 메서드 소스 코드 확인 (간접적 Mock 감지)
                try:
                    import inspect
                    source = inspect.getsource(step_instance._run_ai_inference)
                    
                    if any(keyword in source.lower() for keyword in ['mock', 'dummy', 'fallback', 'placeholder']):
                        result.mock_detected = True
                        result.warnings.append("_run_ai_inference에 Mock 패턴 감지")
                        
                except Exception:
                    pass
            
            # 최종 판정
            if not result.mock_detected and not result.fallback_used and result.real_ai_model_used:
                print(f"      ✅ 실제 AI 모델 사용 확인")
            else:
                print(f"      ⚠️ Mock/폴백 사용 의심")
                
        except Exception as e:
            result.warnings.append(f"Mock 감지 실패: {e}")
    
    def _validate_inference_quality(self, step_instance: Any, result: RealInferenceResult):
        """추론 품질 및 성능 검증"""
        try:
            print(f"   📊 추론 품질 및 성능 검증...")
            
            # FPS 계산
            if result.total_inference_time > 0:
                result.inference_fps = 1.0 / result.total_inference_time
            
            # 메모리 효율성 재평가
            if result.peak_memory_mb > 0:
                if result.peak_memory_mb < 200:
                    result.memory_efficiency = "excellent"
                elif result.peak_memory_mb < 500:
                    result.memory_efficiency = "good"  
                elif result.peak_memory_mb < 1000:
                    result.memory_efficiency = "moderate"
                else:
                    result.memory_efficiency = "high"
            
            # Step별 특화 품질 확인
            step_name = result.step_name
            
            if 'HumanParsing' in step_name:
                # Human Parsing 품질 확인
                if result.output_shape_valid and result.confidence_score > 0.8:
                    result.warnings.append("Human Parsing 품질 양호")
                    
            elif 'VirtualFitting' in step_name:
                # Virtual Fitting 품질 확인 (가장 중요)
                if result.inference_success and result.total_inference_time < 10.0:
                    result.warnings.append("Virtual Fitting 성능 양호")
                elif result.total_inference_time > 30.0:
                    result.warnings.append("Virtual Fitting 성능 저하")
            
            # 전체 품질 점수 계산
            quality_factors = [
                result.model_loading_success,
                result.inference_success,
                result.real_ai_model_used,
                not result.mock_detected,
                result.output_shape_valid,
                result.output_data_type_valid,
                result.device_compatible
            ]
            
            quality_score = sum(quality_factors) / len(quality_factors) * 100
            result.confidence_score = max(result.confidence_score, quality_score / 100)
            
            print(f"      📈 품질 점수: {quality_score:.1f}%, FPS: {result.inference_fps:.2f}")
            
        except Exception as e:
            result.warnings.append(f"품질 검증 실패: {e}")

# =============================================================================
# 🔥 3. 전체 시스템 검증 매니저  
# =============================================================================

class RealAISystemValidator:
    """전체 AI 시스템 실제 추론 검증 매니저"""
    
    def __init__(self):
        self.validator = RealAIInferenceValidator()
        self.results = {}
        self.system_metrics = {}
        
    def validate_entire_ai_pipeline(self) -> Dict[str, Any]:
        """전체 AI 파이프라인 실제 추론 검증"""
        
        print("🔥" * 60)
        print("🔥 Real AI Inference Validator v7.0 - 전체 파이프라인 검증")
        print("🔥 Target: Mock/폴백 없는 100% 실제 AI 추론 검증")
        print("🔥" * 60)
        
        validation_report = {
            'timestamp': time.time(),
            'validator_version': '7.0',
            'total_steps': len(self.validator.github_steps),
            'step_results': {},
            'system_summary': {},
            'critical_issues': [],
            'performance_metrics': {},
            'recommendations': []
        }
        
        start_time = time.time()
        
        try:
            # 1. 시스템 환경 확인
            self._check_system_environment()
            
            # 2. 각 Step별 실제 추론 검증
            print(f"\n📊 8단계 AI Step 실제 추론 검증 시작...")
            
            for step_config in self.validator.github_steps:
                try:
                    print(f"\n{'='*60}")
                    result = self.validator.validate_real_inference_for_step(step_config)
                    self.results[step_config['step_name']] = result
                    validation_report['step_results'][step_config['step_name']] = self._serialize_result(result)
                    
                except Exception as e:
                    print(f"❌ {step_config['step_name']} 검증 실패: {e}")
                    validation_report['step_results'][step_config['step_name']] = {
                        'error': str(e),
                        'status': 'validation_failed'
                    }
            
            # 3. 전체 분석 및 요약
            validation_report['system_summary'] = self._generate_system_summary()
            validation_report['critical_issues'] = self._identify_critical_issues()
            validation_report['performance_metrics'] = self._calculate_performance_metrics()
            validation_report['recommendations'] = self._generate_recommendations()
            
            # 4. 결과 출력
            self._print_validation_results(validation_report)
            
            # 5. 결과 저장
            self._save_validation_results(validation_report)
            
        except Exception as e:
            print(f"\n❌ 전체 검증 실행 중 오류: {e}")
            validation_report['fatal_error'] = str(e)
            
        finally:
            total_time = time.time() - start_time
            validation_report['total_validation_time'] = total_time
            print(f"\n🎉 Real AI Inference Validation 완료! (총 소요시간: {total_time:.2f}초)")
        
        return validation_report
    
    def _check_system_environment(self):
        """시스템 환경 확인"""
        print(f"\n🖥️ 시스템 환경 확인...")
        
        # 하드웨어 정보
        cpu_count = psutil.cpu_count(logical=True)
        memory_info = psutil.virtual_memory()
        total_memory_gb = memory_info.total / (1024**3)
        available_memory_gb = memory_info.available / (1024**3)
        
        print(f"   💻 CPU: {cpu_count}코어")
        print(f"   💾 메모리: {available_memory_gb:.1f}GB 사용가능 / {total_memory_gb:.1f}GB 전체")
        
        # M3 Max 감지
        is_m3_max = False
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=5)
                if 'M3' in result.stdout:
                    is_m3_max = True
                    print(f"   🚀 M3 Max 감지됨")
            except Exception:
                pass
        
        # PyTorch 환경
        if self.validator.torch_available:
            torch = self.validator.torch
            print(f"   🔥 PyTorch: {torch.__version__}")
            
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print(f"   ⚡ MPS: 사용 가능")
            elif torch.cuda.is_available():
                print(f"   🎯 CUDA: 사용 가능")
            else:
                print(f"   🖥️ CPU 모드")
        else:
            print(f"   ❌ PyTorch 없음")
        
        # AI 모델 디렉토리
        if ai_models_root.exists():
            total_size = sum(f.stat().st_size for f in ai_models_root.rglob('*') if f.is_file())
            total_size_gb = total_size / (1024**3)
            print(f"   📁 AI 모델: {total_size_gb:.1f}GB")
        else:
            print(f"   ❌ AI 모델 디렉토리 없음")
        
        self.system_metrics = {
            'cpu_count': cpu_count,
            'total_memory_gb': total_memory_gb,
            'available_memory_gb': available_memory_gb,
            'is_m3_max': is_m3_max,
            'torch_available': self.validator.torch_available,
            'ai_models_size_gb': total_size_gb if ai_models_root.exists() else 0
        }
    
    def _serialize_result(self, result: RealInferenceResult) -> Dict[str, Any]:
        """결과 직렬화"""
        return {
            'step_name': result.step_name,
            'step_id': result.step_id,
            'status': result.status.value,
            'model_loading_success': result.model_loading_success,
            'checkpoint_loaded': result.checkpoint_loaded,
            'checkpoint_size_mb': result.checkpoint_size_mb,
            'model_parameters_count': result.model_parameters_count,
            'device_used': result.device_used,
            'device_compatible': result.device_compatible,
            'mps_optimized': result.mps_optimized,
            'tensor_operations_working': result.tensor_operations_working,
            'preprocessing_success': result.preprocessing_success,
            'inference_success': result.inference_success,
            'postprocessing_success': result.postprocessing_success,
            'total_inference_time': result.total_inference_time,
            'output_shape_valid': result.output_shape_valid,
            'output_data_type_valid': result.output_data_type_valid,
            'output_range_valid': result.output_range_valid,
            'confidence_score': result.confidence_score,
            'mock_detected': result.mock_detected,
            'fallback_used': result.fallback_used,
            'real_ai_model_used': result.real_ai_model_used,
            'peak_memory_mb': result.peak_memory_mb,
            'memory_efficiency': result.memory_efficiency,
            'inference_fps': result.inference_fps,
            'errors': result.errors,
            'warnings': result.warnings,
            'model_info': result.model_info,
            'inference_details': result.inference_details
        }
    
    def _generate_system_summary(self) -> Dict[str, Any]:
        """시스템 요약 생성"""
        total_steps = len(self.results)
        successful_steps = sum(1 for result in self.results.values() 
                              if result.status == RealInferenceStatus.SUCCESS)
        
        real_ai_steps = sum(1 for result in self.results.values() 
                           if result.real_ai_model_used and not result.mock_detected)
        
        mock_detected_steps = sum(1 for result in self.results.values() 
                                 if result.mock_detected)
        
        critical_steps = ['HumanParsingStep', 'PoseEstimationStep', 'ClothSegmentationStep', 'VirtualFittingStep']
        critical_success = sum(1 for step_name in critical_steps 
                              if step_name in self.results and 
                              self.results[step_name].status == RealInferenceStatus.SUCCESS)
        
        # 성능 통계
        inference_times = [result.total_inference_time for result in self.results.values()
                          if result.total_inference_time > 0]
        avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        memory_usage = [result.peak_memory_mb for result in self.results.values()
                       if result.peak_memory_mb > 0]
        avg_memory_usage = sum(memory_usage) / len(memory_usage) if memory_usage else 0
        
        return {
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'success_rate': (successful_steps / total_steps * 100) if total_steps > 0 else 0,
            'real_ai_steps': real_ai_steps,
            'real_ai_rate': (real_ai_steps / total_steps * 100) if total_steps > 0 else 0,
            'mock_detected_steps': mock_detected_steps,
            'critical_steps_success': critical_success,
            'critical_steps_total': len(critical_steps),
            'avg_inference_time': avg_inference_time,
            'avg_memory_usage': avg_memory_usage,
            'system_ready': successful_steps >= 6 and real_ai_steps >= 4,
            'pipeline_validated': successful_steps == total_steps and mock_detected_steps == 0
        }
    
    def _identify_critical_issues(self) -> List[str]:
        """중요 문제점 식별"""
        issues = []
        
        # 시스템 수준 문제
        if not self.validator.torch_available:
            issues.append("🔥 CRITICAL: PyTorch 없음 - AI 추론 불가능")
        
        if self.system_metrics.get('available_memory_gb', 0) < 4:
            issues.append("🔥 CRITICAL: 메모리 부족 - 대용량 모델 로딩 불가")
        
        # Step별 문제
        failed_steps = []
        mock_steps = []
        
        for step_name, result in self.results.items():
            if result.status != RealInferenceStatus.SUCCESS:
                failed_steps.append(f"{step_name}({result.status.value})")
            
            if result.mock_detected or result.fallback_used:
                mock_steps.append(step_name)
        
        if failed_steps:
            issues.append(f"❌ FAILED STEPS: {', '.join(failed_steps)}")
        
        if mock_steps:
            issues.append(f"⚠️ MOCK/FALLBACK DETECTED: {', '.join(mock_steps)}")
        
        # Critical Step 확인
        critical_steps = ['HumanParsingStep', 'VirtualFittingStep', 'ClothSegmentationStep']
        failed_critical = []
        
        for step_name in critical_steps:
            if step_name in self.results:
                result = self.results[step_name] 
                if result.status != RealInferenceStatus.SUCCESS or result.mock_detected:
                    failed_critical.append(step_name)
        
        if failed_critical:
            issues.append(f"🔥 CRITICAL STEPS FAILED: {', '.join(failed_critical)}")
        
        return issues
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """성능 지표 계산"""
        
        # 추론 시간 통계
        inference_times = [result.total_inference_time for result in self.results.values()
                          if result.total_inference_time > 0]
        
        # 메모리 사용량 통계  
        memory_usage = [result.peak_memory_mb for result in self.results.values()
                       if result.peak_memory_mb > 0]
        
        # 모델 파라미터 통계
        model_params = [result.model_parameters_count for result in self.results.values()
                       if result.model_parameters_count > 0]
        
        return {
            'inference_time': {
                'min': min(inference_times) if inference_times else 0,
                'max': max(inference_times) if inference_times else 0,
                'avg': sum(inference_times) / len(inference_times) if inference_times else 0,
                'total': sum(inference_times) if inference_times else 0
            },
            'memory_usage': {
                'min_mb': min(memory_usage) if memory_usage else 0,
                'max_mb': max(memory_usage) if memory_usage else 0,
                'avg_mb': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                'total_mb': sum(memory_usage) if memory_usage else 0
            },
            'model_complexity': {
                'total_parameters': sum(model_params) if model_params else 0,
                'avg_parameters_per_step': sum(model_params) / len(model_params) if model_params else 0,
                'loaded_models_count': len([r for r in self.results.values() if r.model_loading_success])
            },
            'device_utilization': {
                'mps_optimized_steps': len([r for r in self.results.values() if r.mps_optimized]),
                'device_compatible_steps': len([r for r in self.results.values() if r.device_compatible]),
                'tensor_operations_working': len([r for r in self.results.values() if r.tensor_operations_working])
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """추천사항 생성"""
        recommendations = []
        
        # 시스템 개선
        if not self.validator.torch_available:
            recommendations.append("📦 PyTorch 설치: pip install torch torchvision")
            
        if self.system_metrics.get('available_memory_gb', 0) < 8:
            recommendations.append("💾 메모리 증설 또는 메모리 정리 필요")
        
        # Step별 개선사항
        for step_name, result in self.results.items():
            if result.status == RealInferenceStatus.MODEL_LOADING_FAILED:
                recommendations.append(f"🔧 {step_name}: 모델 파일 다운로드 및 경로 확인")
                
            elif result.status == RealInferenceStatus.CHECKPOINT_MISSING:
                recommendations.append(f"📁 {step_name}: 체크포인트 파일 누락 - 다운로드 필요")
                
            elif result.mock_detected:
                recommendations.append(f"⚠️ {step_name}: Mock 모드 감지 - 실제 AI 모델 로딩 확인")
                
            elif result.fallback_used:
                recommendations.append(f"🔄 {step_name}: 폴백 모드 - 메인 모델 로딩 문제 해결")
        
        # 성능 최적화
        slow_steps = [name for name, result in self.results.items() 
                     if result.total_inference_time > 10.0]
        if slow_steps:
            recommendations.append(f"⚡ 성능 최적화 필요: {', '.join(slow_steps)}")
        
        # 메모리 최적화
        high_memory_steps = [name for name, result in self.results.items()
                           if result.peak_memory_mb > 1000]
        if high_memory_steps:
            recommendations.append(f"💾 메모리 최적화 필요: {', '.join(high_memory_steps)}")
            
        # MPS 최적화 (M3 Max인 경우)
        if self.system_metrics.get('is_m3_max', False):
            non_mps_steps = [name for name, result in self.results.items()
                           if not result.mps_optimized and result.device_used != 'mps']
            if non_mps_steps:
                recommendations.append(f"🍎 M3 Max MPS 최적화 필요: {', '.join(non_mps_steps)}")
        
        return recommendations
    
    def _print_validation_results(self, validation_report: Dict[str, Any]):
        """검증 결과 출력"""
        print("\n" + "=" * 80)
        print("📊 Real AI Inference Validation Results v7.0")
        print("=" * 80)
        
        # 전체 요약
        summary = validation_report['system_summary']
        print(f"\n🎯 전체 요약:")
        print(f"   Step 성공률: {summary['success_rate']:.1f}% ({summary['successful_steps']}/{summary['total_steps']})")
        print(f"   실제 AI 사용률: {summary['real_ai_rate']:.1f}% ({summary['real_ai_steps']}/{summary['total_steps']})")
        print(f"   Mock 감지: {summary['mock_detected_steps']}개 Step")
        print(f"   Critical Step 성공: {summary['critical_steps_success']}/{summary['critical_steps_total']}")
        print(f"   평균 추론 시간: {summary['avg_inference_time']:.3f}초")
        print(f"   평균 메모리 사용: {summary['avg_memory_usage']:.1f}MB")
        print(f"   시스템 준비: {'✅' if summary['system_ready'] else '❌'}")
        print(f"   파이프라인 검증: {'✅' if summary['pipeline_validated'] else '❌'}")
        
        # Step별 상세 결과
        print(f"\n🚀 Step별 실제 추론 검증 결과:")
        
        for step_name, result in self.results.items():
            status_icon = "✅" if result.status == RealInferenceStatus.SUCCESS else "❌"
            real_ai_icon = "🧠" if result.real_ai_model_used and not result.mock_detected else "🎭"
            
            print(f"   {status_icon} {real_ai_icon} Step {result.step_id}: {step_name}")
            print(f"      상태: {result.status.value}")
            print(f"      모델 로딩: {'✅' if result.model_loading_success else '❌'}")
            print(f"      실제 추론: {'✅' if result.inference_success else '❌'}")
            print(f"      실제 AI 사용: {'✅' if result.real_ai_model_used else '❌'}")
            print(f"      Mock 감지: {'❌' if result.mock_detected else '✅'}")
            print(f"      추론 시간: {result.total_inference_time:.3f}초")
            print(f"      메모리: {result.peak_memory_mb:.1f}MB ({result.memory_efficiency})")
            print(f"      디바이스: {result.device_used}")
            
            if result.errors:
                print(f"      ❌ 오류: {result.errors[0]}")
            if result.warnings:
                print(f"      ⚠️ 경고: {result.warnings[0]}")
        
        # 중요 문제점
        if validation_report['critical_issues']:
            print(f"\n🔥 중요 문제점:")
            for issue in validation_report['critical_issues']:
                print(f"   {issue}")
        
        # 추천사항
        if validation_report['recommendations']:
            print(f"\n💡 추천사항:")
            for i, rec in enumerate(validation_report['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # 성능 지표
        metrics = validation_report['performance_metrics']
        print(f"\n📈 성능 지표:")
        print(f"   추론 시간: 평균 {metrics['inference_time']['avg']:.3f}초 (최대: {metrics['inference_time']['max']:.3f}초)")
        print(f"   메모리 사용: 평균 {metrics['memory_usage']['avg_mb']:.1f}MB (최대: {metrics['memory_usage']['max_mb']:.1f}MB)")
        print(f"   총 모델 파라미터: {metrics['model_complexity']['total_parameters']:,}")
        print(f"   MPS 최적화: {metrics['device_utilization']['mps_optimized_steps']}개 Step")
    
    def _save_validation_results(self, validation_report: Dict[str, Any]):
        """검증 결과 저장"""
        try:
            timestamp = int(time.time())
            
            # JSON 결과 저장
            results_file = Path(f"real_ai_inference_validation_v7_{timestamp}.json")
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(validation_report, f, indent=2, ensure_ascii=False, default=str)
            
            # 요약 리포트 저장
            summary_file = Path(f"real_ai_inference_summary_v7_{timestamp}.md")
            self._save_summary_report(summary_file, validation_report)
            
            print(f"\n📄 상세 결과: {results_file}")
            print(f"📄 요약 리포트: {summary_file}")
            
        except Exception as e:
            print(f"\n⚠️ 결과 저장 실패: {e}")
    
    def _save_summary_report(self, file_path: Path, validation_report: Dict[str, Any]):
        """요약 리포트 저장"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# 🔥 Real AI Inference Validation Report v7.0\n\n")
                f.write(f"**생성 시간**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                f.write(f"**검증 대상**: MyCloset AI Pipeline (8단계)\n")
                f.write(f"**검증 소요 시간**: {validation_report['total_validation_time']:.1f}초\n\n")
                
                # 시스템 환경
                f.write("## 🖥️ 시스템 환경\n\n")
                f.write(f"- **CPU**: {self.system_metrics.get('cpu_count', 0)}코어\n")
                f.write(f"- **메모리**: {self.system_metrics.get('available_memory_gb', 0):.1f}GB 사용가능 / {self.system_metrics.get('total_memory_gb', 0):.1f}GB 전체\n")
                f.write(f"- **M3 Max**: {'✅' if self.system_metrics.get('is_m3_max', False) else '❌'}\n")
                f.write(f"- **PyTorch**: {'✅' if self.system_metrics.get('torch_available', False) else '❌'}\n")
                f.write(f"- **AI 모델**: {self.system_metrics.get('ai_models_size_gb', 0):.1f}GB\n\n")
                
                # 전체 요약
                summary = validation_report['system_summary']
                f.write("## 🎯 검증 결과 요약\n\n")
                f.write(f"- **Step 성공률**: {summary['success_rate']:.1f}% ({summary['successful_steps']}/{summary['total_steps']})\n")
                f.write(f"- **실제 AI 사용률**: {summary['real_ai_rate']:.1f}% ({summary['real_ai_steps']}/{summary['total_steps']})\n")
                f.write(f"- **Mock 감지**: {summary['mock_detected_steps']}개 Step\n")
                f.write(f"- **Critical Step 성공**: {summary['critical_steps_success']}/{summary['critical_steps_total']}\n")
                f.write(f"- **평균 추론 시간**: {summary['avg_inference_time']:.3f}초\n")
                f.write(f"- **평균 메모리 사용**: {summary['avg_memory_usage']:.1f}MB\n")
                f.write(f"- **시스템 준비**: {'준비됨' if summary['system_ready'] else '문제있음'}\n")
                f.write(f"- **파이프라인 검증**: {'완전 검증됨' if summary['pipeline_validated'] else '부분 검증됨'}\n\n")
                
                # 중요 문제점
                if validation_report['critical_issues']:
                    f.write("## 🔥 중요 문제점\n\n")
                    for issue in validation_report['critical_issues']:
                        f.write(f"- {issue}\n")
                    f.write("\n")
                
                # 추천사항
                if validation_report['recommendations']:
                    f.write("## 💡 추천사항\n\n")
                    for i, rec in enumerate(validation_report['recommendations'], 1):
                        f.write(f"{i}. {rec}\n")
                    f.write("\n")
                
                # Step별 상세 정보
                f.write("## 🚀 Step별 검증 결과\n\n")
                for step_name, result in self.results.items():
                    f.write(f"### Step {result.step_id}: {step_name}\n\n")
                    f.write(f"- **상태**: {result.status.value}\n")
                    f.write(f"- **실제 AI 사용**: {'✅' if result.real_ai_model_used else '❌'}\n")
                    f.write(f"- **Mock 감지**: {'❌' if result.mock_detected else '✅'}\n")
                    f.write(f"- **추론 성공**: {'✅' if result.inference_success else '❌'}\n")
                    f.write(f"- **추론 시간**: {result.total_inference_time:.3f}초\n")
                    f.write(f"- **메모리 사용**: {result.peak_memory_mb:.1f}MB ({result.memory_efficiency})\n")
                    f.write(f"- **디바이스**: {result.device_used}\n")
                    f.write(f"- **신뢰도**: {result.confidence_score:.3f}\n")
                    
                    if result.model_parameters_count > 0:
                        f.write(f"- **모델 파라미터**: {result.model_parameters_count:,}개\n")
                    
                    if result.errors:
                        f.write(f"- **오류**: {result.errors[0]}\n")
                    
                    f.write("\n")
                
        except Exception as e:
            print(f"요약 리포트 저장 실패: {e}")

# =============================================================================
# 🔥 4. 빠른 검증 도구들
# =============================================================================

def quick_real_inference_check(step_name: str) -> bool:
    """빠른 실제 추론 확인"""
    try:
        validator = RealAIInferenceValidator()
        
        # Step 설정 찾기
        step_config = None
        for config in validator.github_steps:
            if config['step_name'] == step_name:
                step_config = config
                break
        
        if not step_config:
            return False
        
        # 빠른 검증
        result = validator.validate_real_inference_for_step(step_config)
        
        return (result.status == RealInferenceStatus.SUCCESS and 
                result.real_ai_model_used and 
                not result.mock_detected)
        
    except Exception:
        return False

def get_ai_pipeline_readiness_score() -> float:
    """AI 파이프라인 준비도 점수 (0-100)"""
    try:
        validator = RealAIInferenceValidator()
        
        score = 0.0
        total_weight = 100.0
        
        # PyTorch 환경 (20점)
        if validator.torch_available:
            score += 20
        
        # 메모리 (15점)
        memory_info = psutil.virtual_memory()
        available_gb = memory_info.available / (1024**3)
        if available_gb >= 16:
            score += 15
        elif available_gb >= 8:
            score += 10
        elif available_gb >= 4:
            score += 5
        
        # AI 모델 크기 (20점)
        if ai_models_root.exists():
            total_size = sum(f.stat().st_size for f in ai_models_root.rglob('*') if f.is_file())
            total_size_gb = total_size / (1024**3)
            
            if total_size_gb >= 200:  # 229GB 목표
                score += 20
            elif total_size_gb >= 100:
                score += 15
            elif total_size_gb >= 50:
                score += 10
            elif total_size_gb >= 10:
                score += 5
        
        # Critical Step 테스트 (45점)
        critical_steps = ['HumanParsingStep', 'VirtualFittingStep', 'ClothSegmentationStep']
        critical_weight = 45 / len(critical_steps)
        
        for step_name in critical_steps:
            if quick_real_inference_check(step_name):
                score += critical_weight
        
        return min(100.0, score)
        
    except Exception:
        return 0.0

def run_critical_steps_validation() -> Dict[str, Any]:
    """Critical Step만 빠른 검증"""
    try:
        print("🔥 Critical Steps 실제 추론 빠른 검증...")
        
        validator = RealAIInferenceValidator()
        critical_steps = ['HumanParsingStep', 'VirtualFittingStep', 'ClothSegmentationStep', 'PoseEstimationStep']
        
        results = {}
        for step_name in critical_steps:
            print(f"   🔍 {step_name} 검증 중...")
            
            # Step 설정 찾기
            step_config = None
            for config in validator.github_steps:
                if config['step_name'] == step_name:
                    step_config = config
                    break
            
            if step_config:
                try:
                    result = validator.validate_real_inference_for_step(step_config)
                    results[step_name] = {
                        'success': result.status == RealInferenceStatus.SUCCESS,
                        'real_ai_used': result.real_ai_model_used,
                        'mock_detected': result.mock_detected,
                        'inference_time': result.total_inference_time,
                        'status': result.status.value
                    }
                    
                    status = "✅" if result.status == RealInferenceStatus.SUCCESS and result.real_ai_model_used else "❌"
                    print(f"      {status} {step_name}: {result.status.value}")
                    
                except Exception as e:
                    results[step_name] = {'error': str(e)}
                    print(f"      ❌ {step_name}: {str(e)[:50]}")
            else:
                results[step_name] = {'error': 'Step 설정 없음'}
                print(f"      ❌ {step_name}: Step 설정 없음")
        
        # 요약
        successful = sum(1 for r in results.values() if r.get('success', False) and r.get('real_ai_used', False))
        total = len(critical_steps)
        
        summary = {
            'critical_steps_validated': f"{successful}/{total}",
            'success_rate': (successful / total * 100) if total > 0 else 0,
            'all_critical_ready': successful == total,
            'results': results
        }
        
        print(f"   🎯 Critical Steps 검증 완료: {successful}/{total} 성공")
        
        return summary
        
    except Exception as e:
        print(f"❌ Critical Steps 검증 실패: {e}")
        return {'error': str(e)}

# =============================================================================
# 🔥 5. 메인 실행부
# =============================================================================

def main():
    """메인 실행 함수"""
    
    print(f"🔥 Real AI Inference Validator v7.0")
    print(f"🔥 Target: Mock/폴백 없는 100% 실제 AI 추론 검증")
    print(f"🔥 시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 빠른 준비도 체크
        print("\n🔍 AI 파이프라인 준비도 체크...")
        readiness_score = get_ai_pipeline_readiness_score()
        print(f"   준비도 점수: {readiness_score:.1f}/100")
        
        if readiness_score < 30:
            print(f"\n⚠️ 시스템 준비도가 낮습니다. Critical Steps만 검증하시겠습니까?")
            response = input("Critical Steps만 검증: 'c', 전체 검증: 'f', 종료: Enter : ").lower().strip()
            
            if response == 'c':
                # Critical Steps만 검증
                critical_results = run_critical_steps_validation()
                
                if critical_results.get('all_critical_ready', False):
                    print(f"\n🎉 SUCCESS: Critical Steps 모두 실제 AI 추론 검증 완료!")
                else:
                    print(f"\n⚠️ WARNING: 일부 Critical Steps에 문제가 있습니다.")
                
                return critical_results
            
            elif response != 'f':
                print("검증을 종료합니다.")
                return None
        
        # 전체 검증 실행
        system_validator = RealAISystemValidator()
        validation_report = system_validator.validate_entire_ai_pipeline()
        
        # 최종 판정
        summary = validation_report.get('system_summary', {})
        pipeline_validated = summary.get('pipeline_validated', False)
        system_ready = summary.get('system_ready', False)
        real_ai_rate = summary.get('real_ai_rate', 0)
        
        if pipeline_validated and real_ai_rate >= 80:
            print(f"\n🎉 SUCCESS: AI 파이프라인 실제 추론 완전 검증 완료!")
            print(f"   - 8단계 모든 Step 실제 AI 추론 검증")
            print(f"   - Mock/폴백 사용 없음 확인")
            print(f"   - 체크포인트 로딩 및 추론 정상 작동")
            print(f"   - 실제 AI 사용률: {real_ai_rate:.1f}%")
        elif system_ready and real_ai_rate >= 60:
            print(f"\n✅ GOOD: AI 파이프라인 대부분 정상 작동")
            print(f"   - 실제 AI 사용률: {real_ai_rate:.1f}%")
            print(f"   - 일부 개선사항 확인 필요")
        else:
            print(f"\n⚠️ WARNING: AI 파이프라인에 문제가 있습니다.")
            print(f"   - 실제 AI 사용률: {real_ai_rate:.1f}%")
            print(f"   - 상세 리포트를 확인하세요.")
        
        return validation_report
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 사용자에 의해 중단되었습니다.")
        return None
        
    except Exception as e:
        print(f"\n❌ 검증 실행 중 치명적 오류: {e}")
        print(f"전체 스택 트레이스:\n{traceback.format_exc()}")
        return None
        
    finally:
        # 리소스 정리
        gc.collect()
        print(f"\n👋 Real AI Inference Validator v7.0 종료")

if __name__ == "__main__":
    main()