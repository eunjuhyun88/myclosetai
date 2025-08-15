#!/usr/bin/env python3
"""
🔥 MyCloset AI - Dependency Management Mixin
===========================================

의존성 관리 기능을 담당하는 Mixin 클래스
- Central Hub 의존성 관리자
- 의존성 주입 및 검증
- 서비스 등록 및 관리

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Union

class CentralHubDependencyManager:
    """🔥 Central Hub DI Container 완전 통합 의존성 관리자 v20.0"""
    
    def __init__(self, step_name: str, **kwargs):
        """Central Hub DI Container 완전 통합 초기화"""
        self.step_name = step_name
        self.logger = logging.getLogger(f"CentralHubDependencyManager.{step_name}")
        
        # 🔥 핵심 속성들
        self.step_instance = None
        self.injected_dependencies = {}
        self.injection_attempts = {}
        self.injection_errors = {}
        
        # 🔥 Central Hub DI Container 참조 (지연 초기화)
        self._central_hub_container = None
        self._container_initialized = False
        
        # 🔥 dependency_status 속성 (Central Hub 기반)
        from .central_hub import CentralHubDependencyStatus
        self.dependency_status = CentralHubDependencyStatus()
        
        # 시간 추적
        self.last_injection_time = time.time()
        
        # 성능 메트릭
        self.dependencies_injected = 0
        self.injection_failures = 0
        self.validation_attempts = 0
        self.central_hub_requests = 0
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
        self.logger.debug(f"✅ Central Hub 완전 통합 의존성 관리자 초기화: {step_name}")
    
    def _get_central_hub_container(self):
        """Central Hub DI Container 지연 초기화 (순환참조 방지)"""
        if not self._container_initialized:
            try:
                # Central Hub Container 가져오기 시도
                # 실제 구현에서는 Central Hub에서 Container를 가져오는 로직
                self._central_hub_container = None  # 임시로 None 반환
                self._container_initialized = True
                if self._central_hub_container:
                    self.dependency_status.central_hub_container = True
                    self.logger.debug(f"✅ {self.step_name} Central Hub Container 연결 성공")
                else:
                    self.logger.warning(f"⚠️ {self.step_name} Central Hub Container 연결 실패")
            except Exception as e:
                self.logger.error(f"❌ {self.step_name} Central Hub Container 초기화 실패: {e}")
                self._central_hub_container = None
                self._container_initialized = True
        
        return self._central_hub_container
    
    def set_step_instance(self, step_instance):
        """Step 인스턴스 설정"""
        try:
            with self._lock:
                self.step_instance = step_instance
                self.logger.debug(f"✅ {self.step_name} Step 인스턴스 설정 완료")
                return True
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} Step 인스턴스 설정 실패: {e}")
            return False
    
    def auto_inject_dependencies(self) -> bool:
        """🔥 Central Hub DI Container 완전 통합 자동 의존성 주입"""
        try:
            with self._lock:
                self.logger.info(f"🔄 {self.step_name} Central Hub 완전 통합 자동 의존성 주입 시작...")
                self.central_hub_requests += 1
                
                if not self.step_instance:
                    self.logger.warning(f"⚠️ {self.step_name} Step 인스턴스가 설정되지 않음")
                    return False
                
                container = self._get_central_hub_container()
                if not container:
                    self.logger.error(f"❌ {self.step_name} Central Hub Container 사용 불가")
                    return False
                
                # 🔥 Central Hub의 inject_to_step 메서드 사용 (핵심 기능)
                injections_made = 0
                try:
                    if hasattr(container, 'inject_to_step'):
                        injections_made = container.inject_to_step(self.step_instance)
                        self.logger.info(f"✅ {self.step_name} Central Hub inject_to_step 완료: {injections_made}개")
                    else:
                        # 수동 주입 (폴백)
                        injections_made = self._manual_injection_fallback(container)
                        self.logger.info(f"✅ {self.step_name} Central Hub 수동 주입 완료: {injections_made}개")
                        
                except Exception as e:
                    self.logger.error(f"❌ {self.step_name} Central Hub inject_to_step 실패: {e}")
                    injections_made = self._manual_injection_fallback(container)
                
                # 주입 상태 업데이트
                if injections_made > 0:
                    self.dependencies_injected += injections_made
                    self.dependency_status.base_initialized = True
                    self.logger.info(f"✅ {self.step_name} 의존성 주입 완료: {injections_made}개")
                else:
                    self.logger.warning(f"⚠️ {self.step_name} 의존성 주입 실패")
                
                return injections_made > 0
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 자동 의존성 주입 실패: {e}")
            self.injection_failures += 1
            return False
    
    def _manual_injection_fallback(self, container) -> int:
        """수동 주입 (폴백)"""
        try:
            injections_made = 0
            
            # 기본 서비스들 주입
            if hasattr(container, 'get'):
                # ModelLoader 주입
                if hasattr(container, 'get') and container.get('model_loader'):
                    self.step_instance.model_loader = container.get('model_loader')
                    injections_made += 1
                
                # MemoryManager 주입
                if hasattr(container, 'get') and container.get('memory_manager'):
                    self.step_instance.memory_manager = container.get('memory_manager')
                    injections_made += 1
                
                # DataConverter 주입
                if hasattr(container, 'get') and container.get('data_converter'):
                    self.step_instance.data_converter = container.get('data_converter')
                    injections_made += 1
            
            return injections_made
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 수동 주입 실패: {e}")
            return 0
    
    def validate_dependencies(self) -> bool:
        """의존성 검증"""
        try:
            with self._lock:
                self.validation_attempts += 1
                self.logger.info(f"🔄 {self.step_name} 의존성 검증 시작")
                
                validation_results = []
                
                # ModelLoader 검증
                if hasattr(self.step_instance, 'model_loader') and self.step_instance.model_loader:
                    validation_results.append(('model_loader', True))
                    self.dependency_status.model_loader = True
                else:
                    validation_results.append(('model_loader', False))
                
                # MemoryManager 검증
                if hasattr(self.step_instance, 'memory_manager') and self.step_instance.memory_manager:
                    validation_results.append(('memory_manager', True))
                    self.dependency_status.memory_manager = True
                else:
                    validation_results.append(('memory_manager', False))
                
                # DataConverter 검증
                if hasattr(self.step_instance, 'data_converter') and self.step_instance.data_converter:
                    validation_results.append(('data_converter', True))
                    self.dependency_status.data_converter = True
                else:
                    validation_results.append(('data_converter', False))
                
                # Central Hub Container 검증
                if hasattr(self.step_instance, 'central_hub_container') and self.step_instance.central_hub_container:
                    validation_results.append(('central_hub_container', True))
                    self.dependency_status.central_hub_container = True
                else:
                    validation_results.append(('central_hub_container', False))
                
                # 검증 결과 요약
                successful_validations = sum(1 for _, success in validation_results if success)
                total_validations = len(validation_results)
                
                if successful_validations == total_validations:
                    self.dependency_status.dependencies_validated = True
                    self.logger.info(f"✅ {self.step_name} 모든 의존성 검증 성공")
                else:
                    self.logger.warning(f"⚠️ {self.step_name} 의존성 검증 부분 실패: {successful_validations}/{total_validations}")
                
                return successful_validations == total_validations
                
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 의존성 검증 실패: {e}")
            return False
    
    def get_dependency_status(self) -> Dict[str, Any]:
        """의존성 상태 반환"""
        try:
            return {
                'step_name': self.step_name,
                'dependencies_injected': self.dependencies_injected,
                'injection_failures': self.injection_failures,
                'validation_attempts': self.validation_attempts,
                'central_hub_requests': self.central_hub_requests,
                'last_injection_time': self.last_injection_time,
                'dependency_status': {
                    'model_loader': self.dependency_status.model_loader,
                    'memory_manager': self.dependency_status.memory_manager,
                    'data_converter': self.dependency_status.data_converter,
                    'central_hub_container': self.dependency_status.central_hub_container,
                    'base_initialized': self.dependency_status.base_initialized,
                    'dependencies_validated': self.dependency_status.dependencies_validated
                }
            }
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 의존성 상태 반환 실패: {e}")
            return {}

class DependencyManagementMixin:
    """의존성 관리 기능을 제공하는 Mixin"""
    
    def _setup_dependency_manager(self):
        """의존성 관리자 설정"""
        try:
            self.dependency_manager = CentralHubDependencyManager(
                step_name=self.step_name
            )
            self.dependency_manager.set_step_instance(self)
            
            # 자동 의존성 주입 시도
            if hasattr(self, 'central_hub_config') and self.central_hub_config.auto_inject_dependencies:
                self.dependency_manager.auto_inject_dependencies()
            
            self.logger.info(f"✅ {self.step_name} 의존성 관리자 설정 완료")
            
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 의존성 관리자 설정 실패: {e}")
    
    def get_service(self, service_name: str):
        """서비스 가져오기"""
        try:
            # 먼저 로컬 속성에서 확인
            if hasattr(self, service_name):
                return getattr(self, service_name)
            
            # Central Hub Container에서 확인
            if hasattr(self, 'central_hub_container') and self.central_hub_container:
                if hasattr(self.central_hub_container, 'get'):
                    return self.central_hub_container.get(service_name)
            
            # 의존성 관리자를 통해 확인
            if hasattr(self, 'dependency_manager') and self.dependency_manager:
                return self.dependency_manager._get_central_hub_container()
            
            return None
            
        except Exception as e:
            self.logger.debug(f"서비스 {service_name} 가져오기 실패: {e}")
            return None
    
    def register_service(self, service_name: str, service_instance: Any, singleton: bool = True):
        """서비스 등록"""
        try:
            # Central Hub Container에 등록
            if hasattr(self, 'central_hub_container') and self.central_hub_container:
                if hasattr(self.central_hub_container, 'register'):
                    self.central_hub_container.register(service_name, service_instance, singleton)
                    self.logger.debug(f"✅ 서비스 {service_name} Central Hub 등록 완료")
                    return True
            
            # 로컬 속성으로 설정
            setattr(self, service_name, service_instance)
            self.logger.debug(f"✅ 서비스 {service_name} 로컬 등록 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"서비스 {service_name} 등록 실패: {e}")
            return False
    
    def validate_dependencies(self) -> bool:
        """의존성 검증"""
        try:
            if hasattr(self, 'dependency_manager') and self.dependency_manager:
                return self.dependency_manager.validate_dependencies()
            else:
                self.logger.warning(f"⚠️ {self.step_name} 의존성 관리자가 없음")
                return False
        except Exception as e:
            self.logger.error(f"❌ {self.step_name} 의존성 검증 실패: {e}")
            return False
