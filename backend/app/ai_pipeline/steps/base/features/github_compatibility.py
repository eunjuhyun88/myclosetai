#!/usr/bin/env python3
"""
🔥 MyCloset AI - GitHub Compatibility Mixin
==========================================

GitHub 프로젝트 특별 기능을 담당하는 Mixin 클래스
- Step 속성 보장
- 프로세스 메서드 검증
- 의존성 검증 형식 관리
- GitHub 호환성 모드

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0
"""

import logging
import inspect
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

class GitHubCompatibilityMixin:
    """GitHub 프로젝트 특별 기능을 제공하는 Mixin"""
    
    def _setup_github_compatibility(self):
        """GitHub 호환성 설정"""
        try:
            # GitHub 호환성 모드 활성화
            self.github_compatibility_mode = getattr(self, 'github_compatibility_mode', True)
            
            if self.github_compatibility_mode:
                self.logger.info(f"🚀 {self.step_name} GitHub 호환성 모드 활성화")
                
                # Step 속성 보장 설정
                self._setup_step_property_guarantee()
                
                # 프로세스 메서드 검증
                self._validate_process_method_signature()
                
                # 의존성 검증 형식 설정
                self._setup_dependency_validation_format()
                
                self.logger.info(f"✅ {self.step_name} GitHub 호환성 설정 완료")
            else:
                self.logger.info(f"⚠️ {self.step_name} GitHub 호환성 모드 비활성화")
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} GitHub 호환성 설정 실패: {e}")
    
    def _setup_step_property_guarantee(self):
        """Step 속성 보장 설정"""
        try:
            from .central_hub import StepPropertyGuarantee
            
            # 기본값은 BASIC 보장
            self.step_property_guarantee = getattr(self, 'step_property_guarantee', StepPropertyGuarantee.BASIC)
            
            if self.step_property_guarantee == StepPropertyGuarantee.FULL:
                self.logger.info(f"🛡️ {self.step_name} 완전 속성 보장 모드")
                self._guarantee_all_properties()
            elif self.step_property_guarantee == StepPropertyGuarantee.GITHUB_COMPATIBLE:
                self.logger.info(f"🔧 {self.step_name} GitHub 호환 속성 보장 모드")
                self._guarantee_github_compatible_properties()
            elif self.step_property_guarantee == StepPropertyGuarantee.BASIC:
                self.logger.info(f"⚡ {self.step_name} 기본 속성 보장 모드")
                self._guarantee_basic_properties()
            else:
                self.logger.info(f"⚠️ {self.step_name} 속성 보장 없음")
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} Step 속성 보장 설정 실패: {e}")
    
    def _guarantee_basic_properties(self):
        """기본 속성 보장"""
        try:
            # 필수 기본 속성들
            basic_properties = [
                'step_name', 'step_id', 'device', 'logger', 'is_initialized',
                'is_ready', 'has_model', 'model_loaded'
            ]
            
            for prop in basic_properties:
                if not hasattr(self, prop):
                    self._set_default_property(prop)
            
            self.logger.debug(f"✅ {self.step_name} 기본 속성 보장 완료")
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 기본 속성 보장 실패: {e}")
    
    def _guarantee_github_compatible_properties(self):
        """GitHub 호환 속성 보장"""
        try:
            # GitHub 호환성을 위한 추가 속성들
            github_properties = [
                'github_compatibility_mode', 'real_ai_pipeline_support',
                'process_method_signature', 'dependency_validation_format'
            ]
            
            for prop in github_properties:
                if not hasattr(self, prop):
                    self._set_default_github_property(prop)
            
            # 기본 속성도 보장
            self._guarantee_basic_properties()
            
            self.logger.debug(f"✅ {self.step_name} GitHub 호환 속성 보장 완료")
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} GitHub 호환 속성 보장 실패: {e}")
    
    def _guarantee_all_properties(self):
        """모든 속성 보장"""
        try:
            # 모든 속성 보장
            self._guarantee_github_compatible_properties()
            
            # 추가 고급 속성들
            advanced_properties = [
                'detailed_data_spec', 'performance_metrics', 'dependency_manager',
                'central_hub_config', 'central_hub_container'
            ]
            
            for prop in advanced_properties:
                if not hasattr(self, prop):
                    self._set_default_advanced_property(prop)
            
            self.logger.debug(f"✅ {self.step_name} 모든 속성 보장 완료")
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 모든 속성 보장 실패: {e}")
    
    def _set_default_property(self, prop_name: str):
        """기본 속성 설정"""
        try:
            if prop_name == 'step_name':
                setattr(self, prop_name, self.__class__.__name__)
            elif prop_name == 'step_id':
                setattr(self, prop_name, getattr(self, 'STEP_ID', 0))
            elif prop_name == 'device':
                setattr(self, prop_name, 'auto')
            elif prop_name == 'logger':
                import logging
                setattr(self, prop_name, logging.getLogger(self.__class__.__name__))
            elif prop_name in ['is_initialized', 'is_ready', 'has_model', 'model_loaded']:
                setattr(self, prop_name, False)
                
        except Exception as e:
            self.logger.debug(f"기본 속성 설정 실패 ({prop_name}): {e}")
    
    def _set_default_github_property(self, prop_name: str):
        """GitHub 속성 설정"""
        try:
            if prop_name == 'github_compatibility_mode':
                setattr(self, prop_name, True)
            elif prop_name == 'real_ai_pipeline_support':
                setattr(self, prop_name, True)
            elif prop_name == 'process_method_signature':
                from .central_hub import ProcessMethodSignature
                setattr(self, prop_name, ProcessMethodSignature.STANDARD)
            elif prop_name == 'dependency_validation_format':
                from .central_hub import DependencyValidationFormat
                setattr(self, prop_name, DependencyValidationFormat.GITHUB_COMPATIBLE)
                
        except Exception as e:
            self.logger.debug(f"GitHub 속성 설정 실패 ({prop_name}): {e}")
    
    def _set_default_advanced_property(self, prop_name: str):
        """고급 속성 설정"""
        try:
            if prop_name == 'detailed_data_spec':
                # DetailedDataSpec은 필요할 때 생성
                pass
            elif prop_name == 'performance_metrics':
                from .central_hub import CentralHubPerformanceMetrics
                setattr(self, prop_name, CentralHubPerformanceMetrics())
            elif prop_name == 'dependency_manager':
                # 의존성 관리자는 필요할 때 생성
                pass
            elif prop_name == 'central_hub_config':
                # Central Hub 설정은 필요할 때 생성
                pass
            elif prop_name == 'central_hub_container':
                setattr(self, prop_name, None)
                
        except Exception as e:
            self.logger.debug(f"고급 속성 설정 실패 ({prop_name}): {e}")
    
    def _validate_process_method_signature(self):
        """프로세스 메서드 시그니처 검증"""
        try:
            from .central_hub import ProcessMethodSignature
            
            # process 메서드 존재 확인
            if not hasattr(self, 'process'):
                self.logger.warning(f"⚠️ {self.step_name} process 메서드가 없음")
                return False
            
            # process 메서드 시그니처 분석
            process_method = getattr(self, 'process')
            if not inspect.ismethod(process_method) and not inspect.isfunction(process_method):
                self.logger.warning(f"⚠️ {self.step_name} process가 메서드가 아님")
                return False
            
            # 메서드 시그니처 확인
            sig = inspect.signature(process_method)
            params = list(sig.parameters.keys())
            
            # self 제거 (인스턴스 메서드인 경우)
            if params and params[0] == 'self':
                params = params[1:]
            
            # GitHub 호환성 검증
            if len(params) == 0:
                # 표준 시그니처: process()
                self.process_method_signature = ProcessMethodSignature.STANDARD
                self.logger.debug(f"✅ {self.step_name} 표준 프로세스 메서드 시그니처")
            elif len(params) == 1 and params[0] == 'kwargs':
                # kwargs 시그니처: process(**kwargs)
                self.process_method_signature = ProcessMethodSignature.STANDARD
                self.logger.debug(f"✅ {self.step_name} kwargs 프로세스 메서드 시그니처")
            else:
                # 기타 시그니처
                self.process_method_signature = ProcessMethodSignature.STANDARD
                self.logger.debug(f"✅ {self.step_name} 커스텀 프로세스 메서드 시그니처: {params}")
            
            # 의존성 상태 업데이트
            if hasattr(self, 'dependency_status'):
                self.dependency_status.process_method_validated = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 프로세스 메서드 시그니처 검증 실패: {e}")
            return False
    
    def _setup_dependency_validation_format(self):
        """의존성 검증 형식 설정"""
        try:
            from .central_hub import DependencyValidationFormat
            
            # 기본값은 AUTO_DETECT
            self.dependency_validation_format = getattr(
                self, 'dependency_validation_format', DependencyValidationFormat.AUTO_DETECT
            )
            
            if self.dependency_validation_format == DependencyValidationFormat.GITHUB_COMPATIBLE:
                self.logger.info(f"🔧 {self.step_name} GitHub 호환 의존성 검증 형식")
                self._setup_github_compatible_validation()
            elif self.dependency_validation_format == DependencyValidationFormat.STRICT:
                self.logger.info(f"🛡️ {self.step_name} 엄격한 의존성 검증 형식")
                self._setup_strict_validation()
            elif self.dependency_validation_format == DependencyValidationFormat.LENIENT:
                self.logger.info(f"😌 {self.step_name} 관대한 의존성 검증 형식")
                self._setup_lenient_validation()
            else:
                self.logger.info(f"🔍 {self.step_name} 자동 감지 의존성 검증 형식")
                self._setup_auto_detection_validation()
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 의존성 검증 형식 설정 실패: {e}")
    
    def _setup_github_compatible_validation(self):
        """GitHub 호환 의존성 검증 설정"""
        try:
            # GitHub 호환성을 위한 검증 설정
            self.validation_strictness = 'github_compatible'
            self.allow_missing_optional_deps = True
            self.auto_fallback_on_failure = True
            self.log_validation_warnings = True
            
            self.logger.debug(f"✅ {self.step_name} GitHub 호환 의존성 검증 설정 완료")
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} GitHub 호환 의존성 검증 설정 실패: {e}")
    
    def _setup_strict_validation(self):
        """엄격한 의존성 검증 설정"""
        try:
            # 엄격한 검증 설정
            self.validation_strictness = 'strict'
            self.allow_missing_optional_deps = False
            self.auto_fallback_on_failure = False
            self.log_validation_warnings = True
            
            self.logger.debug(f"✅ {self.step_name} 엄격한 의존성 검증 설정 완료")
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 엄격한 의존성 검증 설정 실패: {e}")
    
    def _setup_lenient_validation(self):
        """관대한 의존성 검증 설정"""
        try:
            # 관대한 검증 설정
            self.validation_strictness = 'lenient'
            self.allow_missing_optional_deps = True
            self.auto_fallback_on_failure = True
            self.log_validation_warnings = False
            
            self.logger.debug(f"✅ {self.step_name} 관대한 의존성 검증 설정 완료")
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 관대한 의존성 검증 설정 실패: {e}")
    
    def _setup_auto_detection_validation(self):
        """자동 감지 의존성 검증 설정"""
        try:
            # 자동 감지 검증 설정
            self.validation_strictness = 'auto'
            self.allow_missing_optional_deps = True
            self.auto_fallback_on_failure = True
            self.log_validation_warnings = True
            
            # 환경에 따른 자동 조정
            if hasattr(self, 'strict_mode') and self.strict_mode:
                self.validation_strictness = 'strict'
                self.allow_missing_optional_deps = False
                self.auto_fallback_on_failure = False
            
            self.logger.debug(f"✅ {self.step_name} 자동 감지 의존성 검증 설정 완료")
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 자동 감지 의존성 검증 설정 실패: {e}")
    
    def get_github_compatibility_status(self) -> Dict[str, Any]:
        """GitHub 호환성 상태 반환"""
        try:
            return {
                'step_name': getattr(self, 'step_name', 'Unknown'),
                'github_compatibility_mode': getattr(self, 'github_compatibility_mode', False),
                'real_ai_pipeline_support': getattr(self, 'real_ai_pipeline_support', False),
                'process_method_signature': getattr(self, 'process_method_signature', 'unknown'),
                'dependency_validation_format': getattr(self, 'dependency_validation_format', 'unknown'),
                'step_property_guarantee': getattr(self, 'step_property_guarantee', 'none'),
                'validation_strictness': getattr(self, 'validation_strictness', 'unknown'),
                'allow_missing_optional_deps': getattr(self, 'allow_missing_optional_deps', True),
                'auto_fallback_on_failure': getattr(self, 'auto_fallback_on_failure', True)
            }
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} GitHub 호환성 상태 반환 실패: {e}")
            return {}
