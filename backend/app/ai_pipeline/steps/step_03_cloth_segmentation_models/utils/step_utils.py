#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Step Utils
=====================================================================

ClothSegmentationStep 관련 유틸리티 함수들

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import os
import platform
import subprocess
import gc
import threading
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

try:
    import numpy as np
    import cv2
    NUMPY_AVAILABLE = True
    CV2_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    CV2_AVAILABLE = False
    np = None
    cv2 = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

def detect_m3_max() -> bool:
    """M3 Max 감지"""
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            return 'M3' in result.stdout
    except:
        pass
    return False

def get_base_step_mixin_class():
    """BaseStepMixin 클래스를 동적으로 가져오기 (순환참조 방지)"""
    try:
        # 여러 경로 시도
        import_paths = [
            'app.ai_pipeline.steps.base_step_mixin',
            '.base_step_mixin',
            'backend.app.ai_pipeline.steps.base_step_mixin'
        ]
        
        for import_path in import_paths:
            try:
                import importlib
                if import_path.startswith('.'):
                    module = importlib.import_module(import_path, package='app.ai_pipeline.steps')
                else:
                    module = importlib.import_module(import_path)
                base_step_mixin = getattr(module, 'BaseStepMixin', None)
                if base_step_mixin:
                    return base_step_mixin
            except ImportError:
                continue
        
        return None
    except Exception as e:
        logging.getLogger(__name__).error(f"❌ BaseStepMixin 동적 import 실패: {e}")
        return None

def get_central_hub_container():
    """Central Hub Container 가져오기"""
    try:
        from app.ai_pipeline.core.di_container import get_central_hub_container as get_container
        return get_container()
    except ImportError:
        try:
            from ..core.di_container import get_central_hub_container as get_container
            return get_container()
        except ImportError:
            logger.warning("⚠️ Central Hub Container를 찾을 수 없습니다")
            return None

def inject_dependencies_safe(step_instance):
    """의존성 안전 주입"""
    try:
        container = get_central_hub_container()
        if container:
            # 의존성 주입 로직
            if hasattr(step_instance, 'set_model_loader'):
                model_loader = container.get_service('model_loader')
                if model_loader:
                    step_instance.set_model_loader(model_loader)
            
            if hasattr(step_instance, 'set_memory_manager'):
                memory_manager = container.get_service('memory_manager')
                if memory_manager:
                    step_instance.set_memory_manager(memory_manager)
            
            logger.info("✅ 의존성 주입 완료")
        else:
            logger.warning("⚠️ Central Hub Container가 없어 의존성 주입을 건너뜁니다")
    except Exception as e:
        logger.error(f"❌ 의존성 주입 실패: {e}")

def get_service_from_central_hub(service_key: str):
    """Central Hub에서 서비스 가져오기"""
    try:
        container = get_central_hub_container()
        if container:
            return container.get_service(service_key)
        return None
    except Exception as e:
        logger.error(f"❌ 서비스 가져오기 실패: {e}")
        return None

def cleanup_memory():
    """메모리 정리"""
    try:
        # 가비지 컬렉션
        gc.collect()
        
        # PyTorch 메모리 정리
        if TORCH_AVAILABLE:
            if hasattr(torch, 'mps') and torch.mps.is_available():
                torch.mps.empty_cache()
            elif hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("✅ 메모리 정리 완료")
        
    except Exception as e:
        logger.error(f"❌ 메모리 정리 실패: {e}")

def safe_torch_operation(operation_func, *args, **kwargs):
    """안전한 PyTorch 연산 실행"""
    try:
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch가 사용 불가능합니다")
            return None
        
        # 메모리 안전성 체크
        try:
            import psutil
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 90:
                logger.warning(f"⚠️ 메모리 사용량이 높습니다: {memory_usage}%")
                cleanup_memory()
        except ImportError:
            pass
        
        # 연산 실행
        result = operation_func(*args, **kwargs)
        
        # 메모리 정리
        cleanup_memory()
        
        return result
        
    except Exception as e:
        logger.error(f"❌ PyTorch 연산 실패: {e}")
        cleanup_memory()
        return None

def create_cloth_segmentation_step(**kwargs) -> 'ClothSegmentationStep':
    """ClothSegmentationStep 인스턴스 생성"""
    try:
        from .step_core import ClothSegmentationStepCore
        step = ClothSegmentationStepCore(**kwargs)
        inject_dependencies_safe(step)
        return step
    except Exception as e:
        logger.error(f"❌ ClothSegmentationStep 생성 실패: {e}")
        return None

def create_m3_max_segmentation_step(**kwargs) -> 'ClothSegmentationStep':
    """M3 Max 최적화된 ClothSegmentationStep 인스턴스 생성"""
    try:
        # M3 Max 특화 설정
        m3_max_config = {
            'device': 'mps' if TORCH_AVAILABLE and hasattr(torch, 'mps') and torch.mps.is_available() else 'cpu',
            'max_workers': 4,
            'memory_limit': 0.8,  # 80% 메모리 사용 제한
            **kwargs
        }
        
        step = create_cloth_segmentation_step(**m3_max_config)
        return step
        
    except Exception as e:
        logger.error(f"❌ M3 Max ClothSegmentationStep 생성 실패: {e}")
        return None

def test_cloth_segmentation_ai():
    """ClothSegmentationStep AI 테스트"""
    try:
        logger.info("🧪 ClothSegmentationStep AI 테스트 시작")
        
        # 테스트 이미지 생성
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Step 생성
        step = create_cloth_segmentation_step()
        if not step:
            logger.error("❌ Step 생성 실패")
            return False
        
        # 초기화
        if not step.initialize():
            logger.error("❌ Step 초기화 실패")
            return False
        
        # 테스트 실행
        result = step.process(image=test_image)
        
        if result and result.get('success', False):
            logger.info("✅ ClothSegmentationStep AI 테스트 성공")
            return True
        else:
            logger.error("❌ ClothSegmentationStep AI 테스트 실패")
            return False
            
    except Exception as e:
        logger.error(f"❌ ClothSegmentationStep AI 테스트 실패: {e}")
        return False

def test_central_hub_compatibility():
    """Central Hub 호환성 테스트"""
    try:
        logger.info("🧪 Central Hub 호환성 테스트 시작")
        
        # Central Hub Container 테스트
        container = get_central_hub_container()
        if not container:
            logger.warning("⚠️ Central Hub Container를 찾을 수 없습니다")
            return False
        
        # 서비스 가져오기 테스트
        services = ['model_loader', 'memory_manager', 'data_converter']
        for service_key in services:
            service = get_service_from_central_hub(service_key)
            if service:
                logger.info(f"✅ {service_key} 서비스 사용 가능")
            else:
                logger.warning(f"⚠️ {service_key} 서비스 사용 불가능")
        
        logger.info("✅ Central Hub 호환성 테스트 완료")
        return True
        
    except Exception as e:
        logger.error(f"❌ Central Hub 호환성 테스트 실패: {e}")
        return False

def validate_step_configuration(config: Dict[str, Any]) -> Dict[str, Any]:
    """Step 설정 검증"""
    try:
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 필수 설정 검증
        required_configs = ['method', 'quality_level', 'input_size']
        for config_key in required_configs:
            if config_key not in config:
                validation_result['valid'] = False
                validation_result['errors'].append(f"필수 설정 누락: {config_key}")
        
        # 설정값 검증
        if 'method' in config:
            valid_methods = ['u2net_cloth', 'sam_huge', 'deeplabv3_plus', 'hybrid_ai']
            if config['method'] not in valid_methods:
                validation_result['valid'] = False
                validation_result['errors'].append(f"잘못된 method: {config['method']}")
        
        if 'quality_level' in config:
            valid_quality_levels = ['fast', 'balanced', 'high', 'ultra']
            if config['quality_level'] not in valid_quality_levels:
                validation_result['valid'] = False
                validation_result['errors'].append(f"잘못된 quality_level: {config['quality_level']}")
        
        if 'input_size' in config:
            input_size = config['input_size']
            if not isinstance(input_size, (list, tuple)) or len(input_size) != 2:
                validation_result['valid'] = False
                validation_result['errors'].append(f"잘못된 input_size: {input_size}")
            else:
                width, height = input_size
                if width <= 0 or height <= 0:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"input_size는 양수여야 합니다: {input_size}")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"❌ 설정 검증 실패: {e}")
        return {
            'valid': False,
            'errors': [f"설정 검증 실패: {e}"],
            'warnings': []
        }

def get_step_requirements() -> Dict[str, Any]:
    """Step 요구사항 반환"""
    return {
        'python_version': '3.8+',
        'required_packages': [
            'torch>=1.9.0',
            'torchvision>=0.10.0',
            'numpy>=1.21.0',
            'opencv-python>=4.5.0',
            'Pillow>=8.0.0'
        ],
        'optional_packages': [
            'psutil',
            'scikit-image',
            'scipy'
        ],
        'hardware_requirements': {
            'min_memory_gb': 8,
            'recommended_memory_gb': 16,
            'gpu_support': 'optional',
            'mps_support': 'optional'
        },
        'model_requirements': {
            'u2net_cloth': '168.1MB',
            'sam_huge': '2445.7MB',
            'deeplabv3_plus': '233.3MB'
        }
    }

def create_step_documentation() -> Dict[str, Any]:
    """Step 문서화 생성"""
    return {
        'name': 'ClothSegmentationStep',
        'description': '의류 세그멘테이션을 위한 AI Step',
        'version': '1.0',
        'author': 'MyCloset AI Team',
        'methods': {
            'u2net_cloth': {
                'description': 'U2Net 기반 의류 세그멘테이션',
                'size': '168.1MB',
                'speed': 'fast',
                'accuracy': 'high'
            },
            'sam_huge': {
                'description': 'SAM ViT-Huge 기반 세그멘테이션',
                'size': '2445.7MB',
                'speed': 'slow',
                'accuracy': 'very_high'
            },
            'deeplabv3_plus': {
                'description': 'DeepLabV3+ 기반 세그멘테이션',
                'size': '233.3MB',
                'speed': 'medium',
                'accuracy': 'high'
            },
            'hybrid_ai': {
                'description': '여러 모델을 조합한 앙상블',
                'size': 'variable',
                'speed': 'slow',
                'accuracy': 'very_high'
            }
        },
        'configuration': {
            'method': '세그멘테이션 방법 선택',
            'quality_level': '품질 레벨 (fast/balanced/high/ultra)',
            'input_size': '입력 이미지 크기',
            'confidence_threshold': '신뢰도 임계값'
        },
        'output_format': {
            'masks': '세그멘테이션 마스크들',
            'confidence': '신뢰도 점수',
            'processing_time': '처리 시간',
            'method_used': '사용된 방법'
        }
    }
