#!/usr/bin/env python3
"""
🔥 MyCloset AI - Dependency Injection Mixin
===========================================

의존성 주입 관련 기능을 담당하는 Mixin 클래스
Central Hub DI Container와의 연동을 담당

Author: MyCloset AI Team
Date: 2025-08-14
Version: 2.0
"""

import logging
from typing import Dict, Any, Optional

class DependencyInjectionMixin:
    """의존성 주입 관련 기능을 제공하는 Mixin"""
    
    def set_model_loader(self, model_loader):
        """ModelLoader 의존성 주입 (Central Hub 호환) - 동적 타입 검사"""
        try:
            # 동적 타입 검사
            if not hasattr(model_loader, 'load_model_for_step'):
                self.logger.error("❌ ModelLoader 인터페이스가 올바르지 않습니다")
                return False
            
            self.model_loader = model_loader
            
            # 🔥 Step별 모델 인터페이스 생성
            if hasattr(model_loader, 'create_step_interface'):
                self.model_interface = model_loader.create_step_interface(self.step_name)
                self.logger.debug("✅ Step 모델 인터페이스 생성 완료")
            
            # 🔥 체크포인트 로딩 테스트
            if hasattr(model_loader, 'validate_di_container_integration'):
                validation_result = model_loader.validate_di_container_integration()
                if validation_result.get('di_container_available', False):
                    self.logger.debug("✅ ModelLoader Central Hub 연동 확인됨")
            
            # 의존성 상태 업데이트
            self.dependencies_injected['model_loader'] = True
            if hasattr(self, 'dependency_manager') and self.dependency_manager:
                self.dependency_manager.dependency_status.model_loader = True
                self.dependency_manager.dependency_status.base_initialized = True
            
            self.has_model = True
            self.model_loaded = True
            self.real_ai_pipeline_ready = True
            
            self.logger.info("✅ ModelLoader 의존성 주입 완료 (Central Hub 호환)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 의존성 주입 실패: {e}")
            return False

    def set_memory_manager(self, memory_manager):
        """MemoryManager 의존성 주입 (Central Hub 호환) - 동적 타입 검사"""
        try:
            # 동적 타입 검사
            if not hasattr(memory_manager, 'get_memory_usage'):
                self.logger.error("❌ MemoryManager 인터페이스가 올바르지 않습니다")
                return False
            
            self.memory_manager = memory_manager
            
            # 의존성 상태 업데이트
            self.dependencies_injected['memory_manager'] = True
            if hasattr(self, 'dependency_manager') and self.dependency_manager:
                self.dependency_manager.dependency_status.memory_manager = True
            
            self.logger.debug("✅ MemoryManager 의존성 주입 완료 (Central Hub 호환)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ MemoryManager 의존성 주입 실패: {e}")
            return False

    def set_data_converter(self, data_converter):
        """DataConverter 의존성 주입 (Central Hub 호환) - 동적 타입 검사"""
        try:
            # 동적 타입 검사
            if not hasattr(data_converter, 'convert_image'):
                self.logger.error("❌ DataConverter 인터페이스가 올바르지 않습니다")
                return False
            
            self.data_converter = data_converter
            
            # 의존성 상태 업데이트
            self.dependencies_injected['data_converter'] = True
            if hasattr(self, 'dependency_manager') and self.dependency_manager:
                self.dependency_manager.dependency_status.data_converter = True
            
            self.logger.debug("✅ DataConverter 의존성 주입 완료 (Central Hub 호환)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ DataConverter 의존성 주입 실패: {e}")
            return False

    def set_central_hub_container(self, central_hub_container):
        """Central Hub Container 설정"""
        try:
            # dependency_manager를 통한 주입
            if hasattr(self, 'dependency_manager') and self.dependency_manager:
                self.dependency_manager._central_hub_container = central_hub_container
                self.dependency_manager._container_initialized = True
                self.dependency_manager.dependency_status.central_hub_connected = True
                self.dependency_manager.dependency_status.single_source_of_truth = True
            
            self.central_hub_container = central_hub_container
            self.di_container = central_hub_container  # 기존 호환성
            self.dependencies_injected['central_hub_container'] = True
            
            # 성능 메트릭 업데이트
            if hasattr(self, 'performance_metrics'):
                self.performance_metrics.dependencies_injected += 1
            
            self.logger.debug(f"✅ {self.step_name} Central Hub Container 설정 완료")
            
            # Central Hub Container를 통한 추가 의존성 자동 주입 시도
            self._try_additional_central_hub_injections()
            
            return True
                
        except Exception as e:
            if hasattr(self, 'performance_metrics'):
                self.performance_metrics.injection_failures += 1
            self.logger.error(f"❌ {self.step_name} Central Hub Container 설정 오류: {e}")
            return False

    def set_di_container(self, di_container):
        """DI Container 설정 (기존 API 호환성)"""
        return self.set_central_hub_container(di_container)

    def _try_additional_central_hub_injections(self):
        """Central Hub Container 설정 후 추가 의존성 자동 주입 시도"""
        try:
            if not self.central_hub_container:
                return
            
            # 누락된 의존성들 자동 주입 시도
            if not self.model_loader:
                model_loader = self.central_hub_container.get('model_loader')
                if model_loader:
                    self.set_model_loader(model_loader)
                    self.logger.debug(f"✅ {self.step_name} ModelLoader Central Hub 추가 주입")
            
            if not self.memory_manager:
                memory_manager = self.central_hub_container.get('memory_manager')
                if memory_manager:
                    self.set_memory_manager(memory_manager)
                    self.logger.debug(f"✅ {self.step_name} MemoryManager Central Hub 추가 주입")
            
            if not self.data_converter:
                data_converter = self.central_hub_container.get('data_converter')
                if data_converter:
                    self.set_data_converter(data_converter)
                    self.logger.debug(f"✅ {self.step_name} DataConverter Central Hub 추가 주입")
                    
        except Exception as e:
            self.logger.debug(f"Central Hub 추가 주입 실패: {e}")

    def get_model_loader(self):
        """ModelLoader 반환"""
        return getattr(self, 'model_loader', None)

    def get_memory_manager(self):
        """MemoryManager 반환"""
        return getattr(self, 'memory_manager', None)

    def get_data_converter(self):
        """DataConverter 반환"""
        return getattr(self, 'data_converter', None)

    def get_step_interface(self):
        """Step Interface 반환"""
        return getattr(self, 'model_interface', None)

    def is_model_loaded(self) -> bool:
        """모델이 로드되었는지 확인"""
        return getattr(self, 'model_loaded', False)

    def is_step_ready(self) -> bool:
        """Step이 준비되었는지 확인"""
        return getattr(self, 'is_ready', False)
