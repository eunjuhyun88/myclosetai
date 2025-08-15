#!/usr/bin/env python3
"""
🔥 MyCloset AI - Central Hub Mixin
==================================

Central Hub DI Container 연동을 담당하는 Mixin 클래스
- Central Hub 서비스 통합
- 의존성 주입 관리
- 환경 최적화

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# GitHub 프로젝트 특별 열거형들
class ProcessMethodSignature(Enum):
    """프로세스 메서드 시그니처"""
    STANDARD = "standard"
    ASYNC = "async"
    BATCH = "batch"
    STREAMING = "streaming"

class DependencyValidationFormat(Enum):
    """의존성 검증 형식"""
    AUTO_DETECT = "auto_detect"
    STRICT = "strict"
    LENIENT = "lenient"
    GITHUB_COMPATIBLE = "github_compatible"

class DataConversionMethod(Enum):
    """데이터 변환 방법"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    HYBRID = "hybrid"
    STEP_BASED = "step_based"

class StepPropertyGuarantee(Enum):
    """Step 속성 보장"""
    NONE = "none"
    BASIC = "basic"
    FULL = "full"
    GITHUB_COMPATIBLE = "github_compatible"

@dataclass
class DetailedDataSpecConfig:
    """DetailedDataSpec 설정 관리"""
    # 입력 사양
    input_data_types: List[str] = field(default_factory=list)
    input_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    input_value_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    preprocessing_required: List[str] = field(default_factory=list)
    
    # 출력 사양  
    output_data_types: List[str] = field(default_factory=list)
    output_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    output_value_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    postprocessing_required: List[str] = field(default_factory=list)
    
    # API 호환성
    api_input_mapping: Dict[str, str] = field(default_factory=dict)
    api_output_mapping: Dict[str, str] = field(default_factory=dict)
    
    # Step 간 연동
    step_input_schema: Dict[str, Any] = field(default_factory=dict)
    step_output_schema: Dict[str, Any] = field(default_factory=dict)
    
    # 전처리/후처리 요구사항
    normalization_mean: Tuple[float, ...] = field(default_factory=lambda: (0.485, 0.456, 0.406))
    normalization_std: Tuple[float, ...] = field(default_factory=lambda: (0.229, 0.224, 0.225))
    preprocessing_steps: List[str] = field(default_factory=list)
    postprocessing_steps: List[str] = field(default_factory=list)
    
    # Step 간 데이터 전달 스키마
    accepts_from_previous_step: Dict[str, Dict[str, str]] = field(default_factory=dict)
    provides_to_next_step: Dict[str, Dict[str, str]] = field(default_factory=dict)

@dataclass
class CentralHubStepConfig:
    """Central Hub 기반 Step 설정 (v20.0)"""
    step_name: str = "BaseStep"
    step_id: int = 0
    device: str = "auto"
    use_fp16: bool = True
    batch_size: int = 1
    confidence_threshold: float = 0.8
    auto_memory_cleanup: bool = True
    auto_warmup: bool = True
    optimization_enabled: bool = True
    quality_level: str = "balanced"
    strict_mode: bool = False
    
    # Central Hub DI Container 설정
    auto_inject_dependencies: bool = True
    require_model_loader: bool = True
    require_memory_manager: bool = False
    require_data_converter: bool = False
    dependency_timeout: float = 30.0
    dependency_retry_count: int = 3
    central_hub_integration: bool = True
    
    # GitHub 프로젝트 특별 설정
    process_method_signature: ProcessMethodSignature = ProcessMethodSignature.STANDARD
    dependency_validation_format: DependencyValidationFormat = DependencyValidationFormat.AUTO_DETECT
    github_compatibility_mode: bool = True
    real_ai_pipeline_support: bool = True
    
    # DetailedDataSpec 설정 (v20.0)
    enable_detailed_data_spec: bool = True
    data_conversion_method: DataConversionMethod = DataConversionMethod.AUTOMATIC
    strict_data_validation: bool = True
    auto_preprocessing: bool = True
    auto_postprocessing: bool = True
    
    # 환경 최적화
    conda_optimized: bool = False
    conda_env: str = "none"
    m3_max_optimized: bool = False
    memory_gb: float = 16.0
    use_unified_memory: bool = False

@dataclass
class CentralHubDependencyStatus:
    """Central Hub 기반 의존성 상태 (v20.0)"""
    model_loader: bool = False
    step_interface: bool = False
    memory_manager: bool = False
    data_converter: bool = False
    central_hub_container: bool = False
    base_initialized: bool = False
    custom_initialized: bool = False
    dependencies_validated: bool = False
    
    # GitHub 특별 상태
    github_compatible: bool = False
    process_method_validated: bool = False
    real_ai_models_loaded: bool = False
    
    # DetailedDataSpec 상태
    detailed_data_spec_loaded: bool = False
    data_conversion_ready: bool = False
    preprocessing_configured: bool = False
    postprocessing_configured: bool = False
    api_mapping_configured: bool = False

@dataclass
class CentralHubPerformanceMetrics:
    """Central Hub 성능 메트릭"""
    memory_optimizations: int = 0
    peak_memory_usage_mb: float = 0.0
    average_memory_usage_mb: float = 0.0
    data_conversions: int = 0
    step_data_transfers: int = 0
    validation_failures: int = 0
    api_conversions: int = 0
    central_hub_requests: int = 0

class CentralHubMixin:
    """Central Hub DI Container 연동을 담당하는 Mixin"""

    def _setup_central_hub_integration(self):
        """Central Hub 통합 설정"""
        try:
            self.logger.info(f"🔄 {self.step_name} Central Hub 통합 설정 시작")
            
            # Central Hub 설정 생성
            self.central_hub_config = self._create_central_hub_config()
            
            # Central Hub 환경 최적화 적용
            self._apply_central_hub_environment_optimization()
            
            # Central Hub 서비스 연결 시도
            self._try_central_hub_service_connection()
            
            self.logger.info(f"✅ {self.step_name} Central Hub 통합 설정 완료")
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} Central Hub 통합 설정 실패: {e}")
            self._central_hub_emergency_setup(e)

    def _create_central_hub_config(self, **kwargs) -> CentralHubStepConfig:
        """Central Hub 설정 생성"""
        try:
            config = CentralHubStepConfig(
                step_name=self.step_name,
                step_id=self.step_id,
                device=self.device,
                strict_mode=self.strict_mode,
                **kwargs
            )
            
            # GitHub 호환성 모드 설정
            if hasattr(self, 'github_compatibility_mode'):
                config.github_compatibility_mode = self.github_compatibility_mode
            
            # 환경별 최적화 설정
            if hasattr(self, 'conda_env') and self.conda_env:
                config.conda_optimized = True
                config.conda_env = self.conda_env
            
            self.logger.debug(f"✅ {self.step_name} Central Hub 설정 생성 완료")
            return config
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} Central Hub 설정 생성 실패: {e}")
            # 기본 설정 반환
            return CentralHubStepConfig(
                step_name=self.step_name,
                step_id=self.step_id,
                device=self.device
            )

    def _apply_central_hub_environment_optimization(self):
        """Central Hub 환경 최적화 적용"""
        try:
            if not hasattr(self, 'central_hub_config'):
                return
            
            config = self.central_hub_config
            
            # M3 Max 최적화
            if config.m3_max_optimized:
                self._apply_m3_max_optimizations()
            
            # Conda 최적화
            if config.conda_optimized:
                self._apply_conda_optimizations()
            
            # 메모리 최적화
            if config.optimization_enabled:
                self._apply_memory_optimizations()
            
            self.logger.debug(f"✅ {self.step_name} Central Hub 환경 최적화 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ {self.step_name} Central Hub 환경 최적화 실패: {e}")

    def _try_central_hub_service_connection(self):
        """Central Hub 서비스 연결 시도"""
        try:
            # Central Hub Container 가져오기 시도
            container = self._get_central_hub_container()
            if container:
                self.central_hub_container = container
                self.logger.info(f"✅ {self.step_name} Central Hub Container 연결 성공")
                
                # 의존성 주입 시도
                if hasattr(container, 'inject_to_step'):
                    injections_made = container.inject_to_step(self)
                    self.logger.info(f"✅ {self.step_name} Central Hub 의존성 주입 완료: {injections_made}개")
                else:
                    self.logger.warning(f"⚠️ {self.step_name} Central Hub inject_to_step 메서드 없음")
            else:
                self.logger.warning(f"⚠️ {self.step_name} Central Hub Container 연결 실패")
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} Central Hub 서비스 연결 실패: {e}")

# ==============================================
# 🔥 표준화된 DI Container 접근 (폴백 제거)
# ==============================================

def _get_central_hub_container(self):
    """표준화된 DI Container 접근"""
    try:
        from app.ai_pipeline.utils.di_container_access import get_di_container
        return get_di_container()
    except ImportError:
        raise ImportError("표준화된 DI Container 접근 유틸리티를 import할 수 없습니다.")

def get_service(self, service_name: str):
    """표준화된 서비스 조회"""
    try:
        from app.ai_pipeline.utils.di_container_access import get_service
        return get_service(service_name)
    except ImportError:
        raise ImportError("표준화된 DI Container 접근 유틸리티를 import할 수 없습니다.")
