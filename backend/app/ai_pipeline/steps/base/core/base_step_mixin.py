#!/usr/bin/env python3
"""
🔥 BaseStepMixin - 모든 AI Pipeline Step의 기본 기능을 제공하는 통합 Mixin
================================================================================

✅ 의존성 주입, 성능 추적, 데이터 변환, AI 모델 통합 등 모든 핵심 기능 통합
✅ GitHub 호환성 및 속성 보장
✅ 자동 디바이스 선택 및 최적화
✅ 에러 처리 및 로깅 시스템

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0.0
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union

# 핵심 기능 Mixin들 - 상대 import로 변경
try:
    from ..features.dependency_injection import DependencyInjectionMixin
    from ..features.performance_tracking import PerformanceTrackingMixin
    from ..features.data_conversion import DataConversionMixin
    from ..features.central_hub import CentralHubMixin
    from ..features.ai_model_integration import AIModelIntegrationMixin
    from ..features.data_processing import DataProcessingMixin
    from ..features.advanced_data_management import AdvancedDataManagementMixin
    from ..features.dependency_management import DependencyManagementMixin
    from ..features.github_compatibility import GitHubCompatibilityMixin

    # 유틸리티 Mixin들
    from ..utils.validation import ValidationMixin
    from ..utils.error_handling import ErrorHandlingMixin
    
    ALL_MIXINS_AVAILABLE = True
    print("✅ 모든 Mixin import 성공")
    
except ImportError as e:
    print(f"⚠️ Mixin import 실패: {e}")
    # Mock Mixin 클래스들 생성 (각각 고유한 클래스로 생성)
    class MockDependencyInjectionMixin:
        def __init__(self, **kwargs):
            pass
    
    class MockPerformanceTrackingMixin:
        def __init__(self, **kwargs):
            pass
    
    class MockDataConversionMixin:
        def __init__(self, **kwargs):
            pass
    
    class MockCentralHubMixin:
        def __init__(self, **kwargs):
            pass
    
    class MockAIModelIntegrationMixin:
        def __init__(self, **kwargs):
            pass
    
    class MockDataProcessingMixin:
        def __init__(self, **kwargs):
            pass
    
    class MockAdvancedDataManagementMixin:
        def __init__(self, **kwargs):
            pass
    
    class MockDependencyManagementMixin:
        def __init__(self, **kwargs):
            pass
    
    class MockGitHubCompatibilityMixin:
        def __init__(self, **kwargs):
            pass
    
    class MockValidationMixin:
        def __init__(self, **kwargs):
            pass
    
    class MockErrorHandlingMixin:
        def __init__(self, **kwargs):
            pass
    
    # 각각 고유한 Mock 클래스 할당
    DependencyInjectionMixin = MockDependencyInjectionMixin
    PerformanceTrackingMixin = MockPerformanceTrackingMixin
    DataConversionMixin = MockDataConversionMixin
    CentralHubMixin = MockCentralHubMixin
    AIModelIntegrationMixin = MockAIModelIntegrationMixin
    DataProcessingMixin = MockDataProcessingMixin
    AdvancedDataManagementMixin = MockAdvancedDataManagementMixin
    DependencyManagementMixin = MockDependencyManagementMixin
    GitHubCompatibilityMixin = MockGitHubCompatibilityMixin
    ValidationMixin = MockValidationMixin
    ErrorHandlingMixin = MockErrorHandlingMixin
    
    ALL_MIXINS_AVAILABLE = False
    print("⚠️ Mock Mixin 클래스들 사용")

class BaseStepMixin(
    DependencyInjectionMixin,
    PerformanceTrackingMixin,
    DataConversionMixin,
    CentralHubMixin,
    AIModelIntegrationMixin,
    DataProcessingMixin,
    AdvancedDataManagementMixin,
    DependencyManagementMixin,
    GitHubCompatibilityMixin,
    ValidationMixin,
    ErrorHandlingMixin
):
    """
    🔥 BaseStepMixin - 모든 기능을 통합한 메인 Mixin
    
    상속받는 기능들:
    - DependencyInjectionMixin: 의존성 주입 관리
    - PerformanceTrackingMixin: 성능 추적 및 메트릭
    - DataConversionMixin: 데이터 변환 (API ↔ Step)
    - CentralHubMixin: Central Hub DI Container 연동
    - AIModelIntegrationMixin: AI 모델 통합 및 추론
    - DataProcessingMixin: 데이터 전처리 및 후처리
    - AdvancedDataManagementMixin: 고급 데이터 관리 (DetailedDataSpec, 메모리 최적화 등)
    - DependencyManagementMixin: 의존성 관리 및 검증
    - GitHubCompatibilityMixin: GitHub 호환성 및 속성 보장
    - ValidationMixin: 입력 검증 및 환경 검사
    - ErrorHandlingMixin: 에러 처리 및 로깅
    """
    
    def __init__(self, device: str = "auto", strict_mode: bool = False, **kwargs):
        """BaseStepMixin 초기화"""
        # 기본 속성들 초기화
        self.step_name = kwargs.get('step_name', self.__class__.__name__)
        self.step_id = kwargs.get('step_id', getattr(self, 'STEP_ID', 0))
        self.device = device if device != "auto" else self._get_optimal_device()
        self.strict_mode = strict_mode
        
        # Logger 설정
        self.logger = logging.getLogger(f"steps.{self.step_name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # 기본 상태 초기화
        self.is_initialized = False
        self.is_ready = False
        self.has_model = False
        self.model_loaded = False
        self.warmup_completed = False
        
        # 🔥 dependencies_injected 속성 초기화 추가
        self.dependencies_injected = {
            'model_loader': False,
            'memory_manager': False,
            'data_converter': False,
            'central_hub_container': False
        }
        
        # 설정 초기화
        self.config = self._create_default_config(**kwargs)
        
        # 성능 통계 초기화
        self._initialize_performance_stats()

        # GitHub 호환성 설정
        self._setup_github_compatibility()

        # 의존성 주입 시도
        self._try_dependency_injection()
        
        # 초기화 완료
        self.is_initialized = True
        self.logger.info(f"✅ {self.step_name} 초기화 완료")
    
    def _get_optimal_device(self) -> str:
        """최적 디바이스 선택"""
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def _create_default_config(self, **kwargs) -> Dict[str, Any]:
        """기본 설정 생성"""
        return {
            'device': self.device,
            'strict_mode': self.strict_mode,
            'enable_logging': True,
            'enable_performance_tracking': True,
            'max_retry_attempts': 3,
            'timeout_seconds': 300,
            **kwargs
        }
    
    def _initialize_performance_stats(self):
        """성능 통계 초기화"""
        self.performance_stats = {
            'start_time': time.time(),
            'processing_times': [],
            'memory_usage': [],
            'error_count': 0,
            'success_count': 0
        }
    
    def _setup_github_compatibility(self):
        """GitHub 호환성 설정"""
        # GitHub Actions에서 실행될 때 필요한 속성들 설정
        self.github_compatible = True
        self.attributes_preserved = True
    
    def _try_dependency_injection(self):
        """의존성 주입 시도"""
        try:
            if hasattr(self, '_inject_dependencies'):
                self._inject_dependencies()
        except Exception as e:
            self.logger.warning(f"⚠️ 의존성 주입 실패: {e}")
    
    def _get_central_hub_container(self):
        """표준화된 DI Container 접근"""
        try:
            from app.ai_pipeline.utils.di_container_access import get_di_container
            return get_di_container()
        except ImportError:
            raise ImportError("표준화된 DI Container 접근 유틸리티를 import할 수 없습니다.")

    def _get_service_from_central_hub(self, service_key: str):
        """표준화된 서비스 조회"""
        try:
            from app.ai_pipeline.utils.di_container_access import get_service
            return get_service(service_key)
        except ImportError:
            raise ImportError("표준화된 DI Container 접근 유틸리티를 import할 수 없습니다.")
    
    # 기본 메서드들
    def get_step_info(self) -> Dict[str, Any]:
        """Step 정보 반환"""
        return {
            'step_name': self.step_name,
            'step_id': self.step_id,
            'device': self.device,
            'strict_mode': self.strict_mode,
            'is_initialized': self.is_initialized,
            'is_ready': self.is_ready,
            'has_model': self.has_model
        }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """입력 데이터 검증"""
        try:
            if not isinstance(input_data, dict):
                self.logger.error("❌ 입력 데이터가 딕셔너리가 아님")
                return False
            
            # 기본 검증 로직
            required_keys = ['session_id']
            for key in required_keys:
                if key not in input_data:
                    self.logger.error(f"❌ 필수 키 누락: {key}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 입력 데이터 검증 실패: {e}")
            return False
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """기본 처리 메서드 (하위 클래스에서 오버라이드)"""
        try:
            self.logger.info(f"🚀 {self.step_name} 처리 시작")
            
            # 입력 검증
            if not self.validate_input(input_data):
                return self._create_error_response("입력 데이터 검증 실패")
            
            # 처리 로직 (하위 클래스에서 구현)
            result = self._process_impl(input_data, **kwargs)
            
            # 성능 통계 업데이트
            self._update_performance_stats()
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 처리 중 오류: {e}")
            return self._create_error_response(f"처리 중 오류 발생: {str(e)}")
    
    def _process_impl(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """실제 처리 로직 (하위 클래스에서 구현)"""
        raise NotImplementedError("하위 클래스에서 _process_impl을 구현해야 합니다")
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """에러 응답 생성"""
        return {
            'success': False,
            'error': error_message,
            'step_name': self.step_name,
            'step_id': self.step_id
        }
    
    def _update_performance_stats(self):
        """성능 통계 업데이트"""
        try:
            current_time = time.time()
            if hasattr(self, 'performance_stats'):
                self.performance_stats['processing_times'].append(current_time)
        except Exception:
            pass
    
    def cleanup(self):
        """리소스 정리"""
        try:
            self.logger.info(f"🧹 {self.step_name} 리소스 정리 시작")
            
            # 모델 정리
            if hasattr(self, 'model') and self.model:
                del self.model
                self.model = None
            
            # 캐시 정리
            if hasattr(self, 'cache'):
                self.cache.clear()
            
            self.logger.info(f"✅ {self.step_name} 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 리소스 정리 실패: {e}")
    
    def __del__(self):
        """소멸자"""
        try:
            self.cleanup()
        except Exception:
            pass
