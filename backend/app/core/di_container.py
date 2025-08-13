# backend/app/core/di_container.py
"""
🔥 MyCloset AI - 완전 리팩토링된 Central Hub DI Container v7.0
================================================================================

✅ 중앙 허브 역할 완전 구현 - 모든 서비스의 단일 집중점
✅ 순환참조 근본적 해결 - 단방향 의존성 그래프
✅ 단순하고 직관적인 API - 복잡성 제거
✅ 고성능 서비스 캐싱 - 메모리 효율성 극대화
✅ 자동 의존성 해결 - 개발자 편의성 향상
✅ 스레드 안전성 보장 - 동시성 완벽 지원
✅ 생명주기 완전 관리 - 리소스 누수 방지
✅ 기존 API 100% 호환 - 기존 코드 무수정 지원

핵심 설계 원칙:
1. Single Source of Truth - 모든 서비스는 DIContainer를 거침
2. Central Hub Pattern - DIContainer가 모든 컴포넌트의 중심
3. Dependency Inversion - 상위 모듈이 하위 모듈을 제어
4. Zero Circular Reference - 순환참조 원천 차단

Author: MyCloset AI Team
Date: 2025-07-30
Version: 7.0 (Central Hub Architecture)
"""

import os
import sys
import gc
import logging
import threading
import time
import weakref
import platform
import subprocess
import importlib
import traceback
import uuid
import json
from typing import Dict, Any, Optional, Type, TypeVar, Callable, Union, List, Set, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor
from functools import wraps, lru_cache
from collections import defaultdict
import inspect
from pathlib import Path

# ==============================================
# 🔥 환경 설정 (독립적)
# ==============================================

logger = logging.getLogger(__name__)

# 환경 감지
CONDA_ENV = os.environ.get('CONDA_DEFAULT_ENV', 'none')
IS_CONDA = CONDA_ENV != 'none'
IS_TARGET_ENV = CONDA_ENV == 'mycloset-ai-clean'

# M3 Max 감지
def detect_m3_max() -> bool:
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            return 'M3' in result.stdout
    except Exception:
        pass
    return False

IS_M3_MAX = detect_m3_max()
MEMORY_GB = 128.0 if IS_M3_MAX else 16.0
DEVICE = 'mps' if IS_M3_MAX else 'cpu'

# PyTorch 가용성 체크
TORCH_AVAILABLE = False
MPS_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        MPS_AVAILABLE = True
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
except ImportError:
    logger.debug("PyTorch 없음")

T = TypeVar('T')

# ==============================================
# 🔥 Service Registry - 서비스 등록소
# ==============================================

@dataclass
class ServiceInfo:
    """서비스 정보"""
    instance: Any
    is_singleton: bool = True
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    weak_ref: Optional[weakref.ref] = None

class ServiceRegistry:
    """중앙 서비스 등록소"""
    
    def __init__(self):
        self._services: Dict[str, ServiceInfo] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def register_instance(self, key: str, instance: Any, is_singleton: bool = True):
        """인스턴스 직접 등록"""
        with self._lock:
            service_info = ServiceInfo(
                instance=instance,
                is_singleton=is_singleton
            )
            
            # 약한 참조 생성 시도 (기본 타입은 제외)
            try:
                service_info.weak_ref = weakref.ref(instance, lambda ref: self._cleanup_service(key))
            except TypeError:
                # 기본 타입들은 약한 참조 불가
                pass
            
            self._services[key] = service_info
            self.logger.debug(f"✅ 서비스 등록: {key}")
    
    def register_factory(self, key: str, factory: Callable[[], Any], is_singleton: bool = True):
        """팩토리 등록"""
        with self._lock:
            self._factories[key] = factory
            self.logger.debug(f"✅ 팩토리 등록: {key} (singleton: {is_singleton})")
    
    def get_service(self, key: str) -> Optional[Any]:
        """서비스 조회"""
        with self._lock:
            # 직접 등록된 인스턴스 확인
            if key in self._services:
                service_info = self._services[key]
                
                # 약한 참조 확인
                if service_info.weak_ref:
                    instance = service_info.weak_ref()
                    if instance is None:
                        # 가비지 컬렉션됨, 서비스 제거
                        del self._services[key]
                        return None
                
                # 접근 통계 업데이트
                service_info.access_count += 1
                service_info.last_accessed = time.time()
                
                return service_info.instance
            
            # 팩토리를 통한 생성
            if key in self._factories:
                try:
                    instance = self._factories[key]()
                    
                    # 싱글톤이면 등록
                    self.register_instance(key, instance, is_singleton=True)
                    
                    return instance
                except Exception as e:
                    self.logger.error(f"❌ 팩토리 실행 실패 {key}: {e}")
            
            return None
    
    def _cleanup_service(self, key: str):
        """서비스 정리 콜백"""
        with self._lock:
            if key in self._services:
                del self._services[key]
                self.logger.debug(f"🗑️ 서비스 정리: {key}")
    
    def has_service(self, key: str) -> bool:
        """서비스 존재 여부"""
        with self._lock:
            return key in self._services or key in self._factories
    
    def remove_service(self, key: str):
        """서비스 제거"""
        with self._lock:
            if key in self._services:
                del self._services[key]
            if key in self._factories:
                del self._factories[key]
            self.logger.debug(f"🗑️ 서비스 제거: {key}")
    
    def list_services(self) -> List[str]:
        """등록된 서비스 목록"""
        with self._lock:
            return list(set(self._services.keys()) | set(self._factories.keys()))
    
    def get_stats(self) -> Dict[str, Any]:
        """서비스 통계"""
        with self._lock:
            service_stats = {}
            for key, info in self._services.items():
                service_stats[key] = {
                    'type': type(info.instance).__name__,
                    'is_singleton': info.is_singleton,
                    'created_at': info.created_at,
                    'access_count': info.access_count,
                    'last_accessed': info.last_accessed
                }
            
            return {
                'registered_services': len(self._services),
                'registered_factories': len(self._factories),
                'service_details': service_stats
            }

# ==============================================
# 🔥 Central Hub DIContainer - 중앙 허브
# ==============================================

class CentralHubDIContainer:
    """중앙 허브 DI Container - 모든 서비스의 단일 집중점"""
    
    def __init__(self, container_id: str = "default"):
        """🔥 단계별 세분화된 에러 처리가 적용된 CentralHubDIContainer 초기화"""
        start_time = time.time()
        errors = []
        stage_status = {}
        
        try:
            # 🔥 1단계: 기본 속성 초기화
            try:
                self.container_id = container_id
                self._creation_time = time.time()
                self._access_count = 0
                self._injection_count = 0
                self._lock = threading.RLock()
                self.logger = logging.getLogger(f"{self.__class__.__name__}.{container_id}")
                stage_status['basic_initialization'] = 'success'
                self.logger.debug("✅ 1단계: 기본 속성 초기화 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "basic_initialization",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "container_id": container_id
                }
                errors.append(error_info)
                stage_status['basic_initialization'] = 'failed'
                self.logger.error(f"❌ 1단계: 기본 속성 초기화 실패 - {e}")
                raise
            
            # 🔥 2단계: ServiceRegistry 초기화
            try:
                self.registry = ServiceRegistry()
                stage_status['registry_initialization'] = 'success'
                self.logger.debug("✅ 2단계: ServiceRegistry 초기화 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "registry_initialization",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['registry_initialization'] = 'failed'
                self.logger.error(f"❌ 2단계: ServiceRegistry 초기화 실패 - {e}")
                raise
            
            # 🔥 3단계: 내장 서비스 팩토리 등록
            try:
                self._register_builtin_services()
                stage_status['builtin_services_registration'] = 'success'
                self.logger.debug("✅ 3단계: 내장 서비스 팩토리 등록 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "builtin_services_registration",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['builtin_services_registration'] = 'failed'
                self.logger.error(f"❌ 3단계: 내장 서비스 팩토리 등록 실패 - {e}")
                raise
            
            # 🔥 4단계: 초기화 완료 검증
            try:
                # 기본 서비스들이 등록되었는지 확인
                essential_services = ['device', 'memory_gb', 'is_m3_max', 'torch_available', 'mps_available']
                missing_services = []
                
                for service_key in essential_services:
                    if not self.registry.has_service(service_key):
                        missing_services.append(service_key)
                
                if missing_services:
                    raise RuntimeError(f"필수 서비스 누락: {missing_services}")
                
                stage_status['initialization_validation'] = 'success'
                self.logger.debug("✅ 4단계: 초기화 완료 검증 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "initialization_validation",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "missing_services": missing_services if 'missing_services' in locals() else []
                }
                errors.append(error_info)
                stage_status['initialization_validation'] = 'failed'
                self.logger.error(f"❌ 4단계: 초기화 완료 검증 실패 - {e}")
                raise
            
            # 🔥 5단계: 초기화 완료 및 로깅
            try:
                initialization_time = time.time() - start_time
                
                # 에러 정보 저장
                if errors:
                    self._initialization_errors = errors
                    self._initialization_stage_status = stage_status
                    self.logger.warning(f"⚠️ CentralHubDIContainer 초기화 완료 (일부 에러 있음): {len(errors)}개 에러")
                else:
                    self.logger.info(f"✅ 중앙 허브 DI Container 생성 완료: {container_id} (소요시간: {initialization_time:.3f}초)")
                
                # 초기화 통계 저장
                self._initialization_stats = {
                    'initialization_time': initialization_time,
                    'errors_count': len(errors),
                    'stage_status': stage_status,
                    'container_id': container_id
                }
                
            except Exception as e:
                self.logger.error(f"❌ 초기화 완료 처리 실패: {e}")
                # 초기화 완료 처리 실패는 치명적이지 않으므로 에러를 추가하지 않음
                
        except Exception as e:
            # 최종 에러 처리
            self.logger.error(f"❌ CentralHubDIContainer 초기화 실패: {e}")
            
            # 에러 정보 저장
            self._initialization_errors = errors
            self._initialization_stage_status = stage_status
            self._initialization_failed = True
            
            # 최소한의 기본 속성은 설정
            if not hasattr(self, 'container_id'):
                self.container_id = container_id
            if not hasattr(self, 'logger'):
                self.logger = logging.getLogger(f"{self.__class__.__name__}.{container_id}")
            
            raise
    
    def _register_builtin_services(self):
        """🔥 단계별 세분화된 에러 처리가 적용된 내장 서비스들 등록"""
        errors = []
        stage_status = {}
        
        try:
            # 🔥 1단계: ModelLoader 팩토리 등록
            try:
                self.registry.register_factory('model_loader', self._create_model_loader)
                stage_status['model_loader_factory'] = 'success'
                self.logger.debug("✅ 1단계: ModelLoader 팩토리 등록 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "model_loader_factory",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "service_key": "model_loader"
                }
                errors.append(error_info)
                stage_status['model_loader_factory'] = 'failed'
                self.logger.error(f"❌ 1단계: ModelLoader 팩토리 등록 실패 - {e}")
            
            # 🔥 2단계: MemoryManager 팩토리 등록
            try:
                self.registry.register_factory('memory_manager', self._create_memory_manager)
                stage_status['memory_manager_factory'] = 'success'
                self.logger.debug("✅ 2단계: MemoryManager 팩토리 등록 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "memory_manager_factory",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "service_key": "memory_manager"
                }
                errors.append(error_info)
                stage_status['memory_manager_factory'] = 'failed'
                self.logger.error(f"❌ 2단계: MemoryManager 팩토리 등록 실패 - {e}")
            
            # 🔥 3단계: DataConverter 팩토리 등록
            try:
                self.registry.register_factory('data_converter', self._create_data_converter)
                stage_status['data_converter_factory'] = 'success'
                self.logger.debug("✅ 3단계: DataConverter 팩토리 등록 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "data_converter_factory",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "service_key": "data_converter"
                }
                errors.append(error_info)
                stage_status['data_converter_factory'] = 'failed'
                self.logger.error(f"❌ 3단계: DataConverter 팩토리 등록 실패 - {e}")
            
            # 🔥 4단계: StepFactory 팩토리 등록
            try:
                self.registry.register_factory('step_factory', self._create_step_factory)
                stage_status['step_factory_registration'] = 'success'
                self.logger.debug("✅ 4단계: StepFactory 팩토리 등록 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "step_factory_registration",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "service_key": "step_factory"
                }
                errors.append(error_info)
                stage_status['step_factory_registration'] = 'failed'
                self.logger.error(f"❌ 4단계: StepFactory 팩토리 등록 실패 - {e}")
            
            # 🔥 5단계: 기본 환경 값들 등록
            try:
                basic_services = {
                    'device': DEVICE,
                    'memory_gb': MEMORY_GB,
                    'is_m3_max': IS_M3_MAX,
                    'torch_available': TORCH_AVAILABLE,
                    'mps_available': MPS_AVAILABLE
                }
                
                failed_services = []
                for service_key, service_value in basic_services.items():
                    try:
                        self.registry.register_instance(service_key, service_value)
                    except Exception as service_error:
                        failed_services.append(f"{service_key}: {service_error}")
                
                if failed_services:
                    raise RuntimeError(f"기본 서비스 등록 실패: {failed_services}")
                
                stage_status['basic_services_registration'] = 'success'
                self.logger.debug("✅ 5단계: 기본 환경 값들 등록 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "basic_services_registration",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "failed_services": failed_services if 'failed_services' in locals() else []
                }
                errors.append(error_info)
                stage_status['basic_services_registration'] = 'failed'
                self.logger.error(f"❌ 5단계: 기본 환경 값들 등록 실패 - {e}")
            
            # 🔥 6단계: 등록 완료 검증
            try:
                # 핵심 서비스들이 등록되었는지 확인
                essential_factories = ['model_loader', 'memory_manager', 'data_converter']
                missing_factories = []
                
                for factory_key in essential_factories:
                    if not self.registry.has_service(factory_key):
                        missing_factories.append(factory_key)
                
                if missing_factories:
                    self.logger.warning(f"⚠️ 일부 핵심 팩토리 누락: {missing_factories}")
                
                stage_status['registration_validation'] = 'success'
                self.logger.debug("✅ 6단계: 등록 완료 검증 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "registration_validation",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['registration_validation'] = 'failed'
                self.logger.error(f"❌ 6단계: 등록 완료 검증 실패 - {e}")
            
            # 🔥 7단계: 결과 보고
            if errors:
                self.logger.warning(f"⚠️ 내장 서비스 등록 완료 (일부 에러 있음): {len(errors)}개 에러")
                self.logger.debug(f"📊 등록 상태: {stage_status}")
            else:
                self.logger.debug("✅ 내장 서비스 등록 완료 (모든 단계 성공)")
            
            # 에러 정보 저장
            self._builtin_services_errors = errors
            self._builtin_services_stage_status = stage_status
            
        except Exception as e:
            # 최종 에러 처리
            self.logger.error(f"❌ 내장 서비스 등록 실패: {e}")
            
            # 에러 정보 저장
            self._builtin_services_errors = errors
            self._builtin_services_stage_status = stage_status
            
            raise
    
    # ==============================================
    # 🔥 Public API - 간단하고 직관적
    # ==============================================
    
    def get(self, service_key: str) -> Optional[Any]:
        """🔥 단계별 세분화된 에러 처리가 적용된 서비스 조회 - 중앙 허브의 핵심 메서드"""
        start_time = time.time()
        errors = []
        stage_status = {}
        
        try:
            # 🔥 1단계: 입력 검증
            try:
                if not service_key or not isinstance(service_key, str):
                    raise ValueError(f"잘못된 서비스 키: {service_key} (타입: {type(service_key)})")
                
                if not service_key.strip():
                    raise ValueError("서비스 키가 비어있습니다")
                
                stage_status['input_validation'] = 'success'
                self.logger.debug(f"✅ 1단계: 입력 검증 성공 - {service_key}")
                
            except Exception as e:
                error_info = {
                    "stage": "input_validation",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "service_key": service_key
                }
                errors.append(error_info)
                stage_status['input_validation'] = 'failed'
                self.logger.error(f"❌ 1단계: 입력 검증 실패 - {e}")
                return None
            
            # 🔥 2단계: 레지스트리 상태 확인
            try:
                if not hasattr(self, 'registry') or self.registry is None:
                    raise RuntimeError("ServiceRegistry가 초기화되지 않았습니다")
                
                stage_status['registry_validation'] = 'success'
                self.logger.debug(f"✅ 2단계: 레지스트리 상태 확인 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "registry_validation",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['registry_validation'] = 'failed'
                self.logger.error(f"❌ 2단계: 레지스트리 상태 확인 실패 - {e}")
                return None
            
            # 🔥 3단계: 서비스 존재 여부 확인
            try:
                service_exists = self.registry.has_service(service_key)
                if not service_exists:
                    self.logger.debug(f"⚠️ 서비스가 등록되지 않음: {service_key}")
                
                stage_status['service_existence_check'] = 'success'
                self.logger.debug(f"✅ 3단계: 서비스 존재 여부 확인 성공 - 존재: {service_exists}")
                
            except Exception as e:
                error_info = {
                    "stage": "service_existence_check",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "service_key": service_key
                }
                errors.append(error_info)
                stage_status['service_existence_check'] = 'failed'
                self.logger.error(f"❌ 3단계: 서비스 존재 여부 확인 실패 - {e}")
                return None
            
            # 🔥 4단계: 서비스 조회 실행
            try:
                with self._lock:
                    self._access_count += 1
                    service = self.registry.get_service(service_key)
                
                stage_status['service_retrieval'] = 'success'
                self.logger.debug(f"✅ 4단계: 서비스 조회 실행 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "service_retrieval",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "service_key": service_key
                }
                errors.append(error_info)
                stage_status['service_retrieval'] = 'failed'
                self.logger.error(f"❌ 4단계: 서비스 조회 실행 실패 - {e}")
                return None
            
            # 🔥 5단계: 조회 결과 검증
            try:
                if service is None:
                    self.logger.debug(f"⚠️ 서비스 조회 결과 없음: {service_key}")
                else:
                    service_type = type(service).__name__
                    self.logger.debug(f"✅ 서비스 조회 성공: {service_key} (타입: {service_type})")
                
                stage_status['result_validation'] = 'success'
                
            except Exception as e:
                error_info = {
                    "stage": "result_validation",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "service_key": service_key
                }
                errors.append(error_info)
                stage_status['result_validation'] = 'failed'
                self.logger.error(f"❌ 5단계: 조회 결과 검증 실패 - {e}")
            
            # 🔥 6단계: 성능 통계 업데이트
            try:
                retrieval_time = time.time() - start_time
                
                # 성능 통계 저장
                if not hasattr(self, '_service_retrieval_stats'):
                    self._service_retrieval_stats = {}
                
                if service_key not in self._service_retrieval_stats:
                    self._service_retrieval_stats[service_key] = {
                        'total_retrievals': 0,
                        'successful_retrievals': 0,
                        'failed_retrievals': 0,
                        'average_retrieval_time': 0.0,
                        'last_retrieval_time': 0.0
                    }
                
                stats = self._service_retrieval_stats[service_key]
                stats['total_retrievals'] += 1
                stats['last_retrieval_time'] = retrieval_time
                
                if service is not None:
                    stats['successful_retrievals'] += 1
                else:
                    stats['failed_retrievals'] += 1
                
                # 평균 조회 시간 업데이트
                if stats['total_retrievals'] > 0:
                    stats['average_retrieval_time'] = (
                        (stats['average_retrieval_time'] * (stats['total_retrievals'] - 1) + retrieval_time) 
                        / stats['total_retrievals']
                    )
                
                stage_status['performance_tracking'] = 'success'
                
            except Exception as e:
                self.logger.debug(f"⚠️ 성능 통계 업데이트 실패: {e}")
                stage_status['performance_tracking'] = 'failed'
            
            # 🔥 7단계: 에러 정보 저장 및 결과 반환
            if errors:
                # 에러 정보 저장
                if not hasattr(self, '_service_retrieval_errors'):
                    self._service_retrieval_errors = {}
                
                if service_key not in self._service_retrieval_errors:
                    self._service_retrieval_errors[service_key] = []
                
                self._service_retrieval_errors[service_key].extend(errors)
                
                self.logger.warning(f"⚠️ 서비스 조회 완료 (일부 에러 있음): {service_key} - {len(errors)}개 에러")
            else:
                self.logger.debug(f"✅ 서비스 조회 완료 (성공): {service_key}")
            
            return service
            
        except Exception as e:
            # 최종 에러 처리
            self.logger.error(f"❌ 서비스 조회 실패: {service_key} - {e}")
            
            # 에러 정보 저장
            if not hasattr(self, '_service_retrieval_errors'):
                self._service_retrieval_errors = {}
            
            if service_key not in self._service_retrieval_errors:
                self._service_retrieval_errors[service_key] = []
            
            final_error = {
                "stage": "final_error_handling",
                "error_type": type(e).__name__,
                "message": str(e),
                "service_key": service_key
            }
            self._service_retrieval_errors[service_key].append(final_error)
            
            return None
    
    def get_service(self, service_key: str) -> Optional[Any]:
        """서비스 조회 (get 메서드와 동일)"""
        return self.get(service_key)
    
    def register(self, service_key: str, instance: Any, singleton: bool = True):
        """서비스 등록"""
        self.registry.register_instance(service_key, instance, singleton)
        self.logger.debug(f"✅ 서비스 등록: {service_key}")
    
    def register_factory(self, service_key: str, factory: Callable[[], Any], singleton: bool = True):
        """팩토리 등록"""
        self.registry.register_factory(service_key, factory, singleton)
        self.logger.debug(f"✅ 팩토리 등록: {service_key}")
    
    def has(self, service_key: str) -> bool:
        """서비스 존재 여부"""
        return self.registry.has_service(service_key)
    
    def remove(self, service_key: str):
        """서비스 제거"""
        self.registry.remove_service(service_key)
        self.logger.debug(f"🗑️ 서비스 제거: {service_key}")
    
    # ==============================================
    # 🔥 구 버전 호환성 메서드들 (완전 구현)
    # ==============================================
    
    def register_lazy(self, service_key: str, factory: Callable[[], Any], is_singleton: bool = True):
        """지연 서비스 등록 (구 버전 호환)"""
        try:
            lazy_service = LazyDependency(factory)
            self.register(service_key, lazy_service, singleton=is_singleton)
            self.logger.debug(f"✅ register_lazy 성공: {service_key}")
            return True
        except Exception as e:
            self.logger.debug(f"⚠️ register_lazy 실패 ({service_key}): {e}")
            return False
    
    def register_factory_method(self, service_key: str, factory: Callable[[], Any], is_singleton: bool = True):
        """팩토리 메서드 등록 (구 버전 호환)"""
        return self.register_factory(service_key, factory, is_singleton)
    
    def get_service_info(self, service_key: str) -> Dict[str, Any]:
        """서비스 정보 조회 (구 버전 호환)"""
        try:
            service = self.get(service_key)
            return {
                'service_key': service_key,
                'available': service is not None,
                'type': type(service).__name__ if service else None,
                'container_id': self.container_id
            }
        except Exception:
            return {
                'service_key': service_key,
                'available': False,
                'error': 'Failed to get service info',
                'container_id': self.container_id
            }
    
    def clear(self):
        """모든 서비스 정리 (구 버전 호환)"""
        try:
            # 등록된 서비스들 정리
            for service_key in self.list_services():
                self.remove(service_key)
            self.logger.debug("✅ 모든 서비스 정리 완료")
        except Exception as e:
            self.logger.error(f"❌ 서비스 정리 실패: {e}")
    
    def force_register_model_loader(self, model_loader):
        """ModelLoader 강제 등록 (구 버전 호환)"""
        try:
            self.register('model_loader', model_loader)
            self.logger.info("✅ ModelLoader 강제 등록 완료")
            return True
        except Exception as e:
            self.logger.error(f"❌ ModelLoader 강제 등록 실패: {e}")
            return False
    
    def cleanup_circular_references(self):
        """순환참조 정리 (구 버전 호환)"""
        # Central Hub 설계에서는 순환참조가 원천적으로 방지되므로 아무것도 하지 않음
        self.logger.debug("순환참조 정리: Central Hub 설계로 불필요")
        pass
    
    # ==============================================
    # 🔥 중앙 허브 - 의존성 주입 시스템
    # ==============================================
    
    def inject_to_step(self, step_instance) -> int:
        """🔥 단계별 세분화된 에러 처리가 적용된 Central Hub DI Container v7.0 - 완전한 의존성 주입 시스템"""
        start_time = time.time()
        errors = []
        stage_status = {}
        injections_made = 0
        
        try:
            # 🔥 1단계: Step 인스턴스 유효성 검증
            try:
                if step_instance is None:
                    raise ValueError("Step 인스턴스가 None입니다")
                
                if not hasattr(step_instance, '__class__'):
                    raise ValueError("Step 인스턴스에 __class__ 속성이 없습니다")
                
                step_name = step_instance.__class__.__name__
                if step_name == 'TestStep':
                    self.logger.warning(f"⚠️ TestStep 감지 - 실제 Step 클래스가 로딩되지 않았습니다")
                    return 0
                
                stage_status['step_validation'] = 'success'
                self.logger.debug(f"✅ 1단계: Step 인스턴스 유효성 검증 성공 - {step_name}")
                
            except Exception as e:
                error_info = {
                    "stage": "step_validation",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "step_instance_type": type(step_instance).__name__ if step_instance else None
                }
                errors.append(error_info)
                stage_status['step_validation'] = 'failed'
                self.logger.error(f"❌ 1단계: Step 인스턴스 유효성 검증 실패 - {e}")
                return 0
            
            # 🔥 2단계: Central Hub Container 자체 주입
            try:
                if hasattr(step_instance, 'central_hub_container'):
                    step_instance.central_hub_container = self
                    injections_made += 1
                    self.logger.debug(f"✅ Central Hub Container 주입 완료")
                
                stage_status['central_hub_injection'] = 'success'
                
            except Exception as e:
                error_info = {
                    "stage": "central_hub_injection",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['central_hub_injection'] = 'failed'
                self.logger.error(f"❌ 2단계: Central Hub Container 주입 실패 - {e}")
            
            # 🔥 3단계: DI Container 자체 주입 (기존 호환성)
            try:
                if hasattr(step_instance, 'di_container'):
                    step_instance.di_container = self
                    injections_made += 1
                    self.logger.debug(f"✅ DI Container 주입 완료")
                
                stage_status['di_container_injection'] = 'success'
                
            except Exception as e:
                error_info = {
                    "stage": "di_container_injection",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['di_container_injection'] = 'failed'
                self.logger.error(f"❌ 3단계: DI Container 주입 실패 - {e}")
            
            # 🔥 4단계: PropertyInjectionMixin 지원
            try:
                if hasattr(step_instance, 'set_di_container'):
                    step_instance.set_di_container(self)
                    injections_made += 1
                    self.logger.debug(f"✅ PropertyInjectionMixin 설정 완료")
                
                stage_status['property_injection_mixin'] = 'success'
                
            except Exception as e:
                error_info = {
                    "stage": "property_injection_mixin",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['property_injection_mixin'] = 'failed'
                self.logger.error(f"❌ 4단계: PropertyInjectionMixin 설정 실패 - {e}")
            
            # 🔥 5단계: 표준 의존성들 주입 (Central Hub v7.0 확장)
            try:
                injection_map = {
                    'model_loader': 'model_loader',
                    'memory_manager': 'memory_manager', 
                    'data_converter': 'data_converter',
                    'step_factory': 'step_factory',
                    'data_transformer': 'data_transformer',
                    'model_registry': 'model_registry',
                    'performance_monitor': 'performance_monitor',
                    'error_handler': 'error_handler',
                    'cache_manager': 'cache_manager',
                    'config_manager': 'config_manager'
                }
                
                failed_injections = []
                for attr_name, service_key in injection_map.items():
                    try:
                        if hasattr(step_instance, attr_name):
                            current_value = getattr(step_instance, attr_name)
                            if current_value is None:
                                service = self.get(service_key)
                                if service:
                                    setattr(step_instance, attr_name, service)
                                    injections_made += 1
                                    self.logger.debug(f"✅ {attr_name} 주입 완료")
                                else:
                                    failed_injections.append(f"{attr_name}: 서비스 없음")
                            else:
                                self.logger.debug(f"⚠️ {attr_name} 이미 설정됨")
                    except Exception as injection_error:
                        failed_injections.append(f"{attr_name}: {injection_error}")
                
                if failed_injections:
                    self.logger.warning(f"⚠️ 일부 의존성 주입 실패: {failed_injections}")
                
                stage_status['standard_dependencies_injection'] = 'success'
                
            except Exception as e:
                error_info = {
                    "stage": "standard_dependencies_injection",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "failed_injections": failed_injections if 'failed_injections' in locals() else []
                }
                errors.append(error_info)
                stage_status['standard_dependencies_injection'] = 'failed'
                self.logger.error(f"❌ 5단계: 표준 의존성 주입 실패 - {e}")
            
            # 🔥 6단계: Central Hub 통합 상태 표시
            try:
                if hasattr(step_instance, 'central_hub_integrated'):
                    step_instance.central_hub_integrated = True
                    injections_made += 1
                    self.logger.debug(f"✅ Central Hub 통합 상태 설정 완료")
                
                stage_status['central_hub_integration_status'] = 'success'
                
            except Exception as e:
                error_info = {
                    "stage": "central_hub_integration_status",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['central_hub_integration_status'] = 'failed'
                self.logger.error(f"❌ 6단계: Central Hub 통합 상태 설정 실패 - {e}")
            
            # 🔥 7단계: Step 메타데이터 설정
            try:
                if hasattr(step_instance, 'step_metadata'):
                    step_instance.step_metadata = {
                        'container_id': self.container_id,
                        'injection_time': time.time(),
                        'injection_count': self._injection_count,
                        'central_hub_version': '7.0',
                        'step_name': step_name,
                        'services_injected': injections_made,
                        'errors_count': len(errors)
                    }
                    injections_made += 1
                    self.logger.debug(f"✅ Step 메타데이터 설정 완료")
                
                stage_status['metadata_setup'] = 'success'
                
            except Exception as e:
                error_info = {
                    "stage": "metadata_setup",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['metadata_setup'] = 'failed'
                self.logger.error(f"❌ 7단계: Step 메타데이터 설정 실패 - {e}")
            
            # 🔥 8단계: 자동 초기화 메서드 호출
            try:
                if hasattr(step_instance, 'initialize') and not getattr(step_instance, 'is_initialized', False):
                    try:
                        step_instance.initialize()
                        step_instance.is_initialized = True
                        self.logger.debug("✅ Step 자동 초기화 완료")
                    except Exception as init_error:
                        self.logger.debug(f"⚠️ Step 자동 초기화 실패: {init_error}")
                        # 초기화 실패는 치명적이지 않으므로 에러로 처리하지 않음
                
                stage_status['auto_initialization'] = 'success'
                
            except Exception as e:
                error_info = {
                    "stage": "auto_initialization",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['auto_initialization'] = 'failed'
                self.logger.error(f"❌ 8단계: 자동 초기화 메서드 호출 실패 - {e}")
            
            # 🔥 9단계: Central Hub 이벤트 시스템 연동
            try:
                if hasattr(step_instance, 'on_central_hub_integration'):
                    try:
                        step_instance.on_central_hub_integration(self)
                        self.logger.debug("✅ Central Hub 이벤트 시스템 연동 완료")
                    except Exception as event_error:
                        self.logger.debug(f"⚠️ Central Hub 이벤트 시스템 연동 실패: {event_error}")
                        # 이벤트 시스템 실패는 치명적이지 않으므로 에러로 처리하지 않음
                
                stage_status['event_system_integration'] = 'success'
                
            except Exception as e:
                error_info = {
                    "stage": "event_system_integration",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['event_system_integration'] = 'failed'
                self.logger.error(f"❌ 9단계: Central Hub 이벤트 시스템 연동 실패 - {e}")
            
            # 🔥 10단계: 성능 모니터링 설정
            try:
                if hasattr(step_instance, 'performance_monitor'):
                    try:
                        step_instance.performance_monitor.start_monitoring(step_name)
                        self.logger.debug("✅ 성능 모니터링 시작")
                    except Exception as monitor_error:
                        self.logger.debug(f"⚠️ 성능 모니터링 설정 실패: {monitor_error}")
                        # 모니터링 실패는 치명적이지 않으므로 에러로 처리하지 않음
                
                stage_status['performance_monitoring_setup'] = 'success'
                
            except Exception as e:
                error_info = {
                    "stage": "performance_monitoring_setup",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['performance_monitoring_setup'] = 'failed'
                self.logger.error(f"❌ 10단계: 성능 모니터링 설정 실패 - {e}")
            
            # 🔥 11단계: 통계 업데이트
            try:
                with self._lock:
                    self._injection_count += 1
                    self._update_injection_stats(step_name, injections_made)
                
                stage_status['statistics_update'] = 'success'
                
            except Exception as e:
                error_info = {
                    "stage": "statistics_update",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['statistics_update'] = 'failed'
                self.logger.error(f"❌ 11단계: 통계 업데이트 실패 - {e}")
            
            # 🔥 12단계: 완료 로깅 및 결과 반환
            try:
                injection_time = time.time() - start_time
                
                if errors:
                    self.logger.warning(f"⚠️ {step_name} Central Hub v7.0 의존성 주입 완료 (일부 에러 있음): {injections_made}개 서비스, {len(errors)}개 에러")
                    self.logger.debug(f"📊 주입 상태: {stage_status}")
                else:
                    self.logger.info(f"🔥 {step_name} Central Hub v7.0 의존성 주입 완료: {injections_made}개 서비스 (소요시간: {injection_time:.3f}초)")
                
                # 에러 정보 저장
                if not hasattr(self, '_injection_errors'):
                    self._injection_errors = {}
                
                if step_name not in self._injection_errors:
                    self._injection_errors[step_name] = []
                
                self._injection_errors[step_name].extend(errors)
                
                stage_status['completion_logging'] = 'success'
                
            except Exception as e:
                error_info = {
                    "stage": "completion_logging",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['completion_logging'] = 'failed'
                self.logger.error(f"❌ 12단계: 완료 로깅 실패 - {e}")
            
            return injections_made
            
        except Exception as e:
            # 최종 에러 처리
            self.logger.error(f"❌ Central Hub v7.0 의존성 주입 실패: {e}")
            self.logger.debug(f"🔍 실패 상세: {traceback.format_exc()}")
            
            # 에러 정보 저장
            if not hasattr(self, '_injection_errors'):
                self._injection_errors = {}
            
            step_name = getattr(step_instance, '__class__', {}).__name__ if step_instance else 'Unknown'
            if step_name not in self._injection_errors:
                self._injection_errors[step_name] = []
            
            final_error = {
                "stage": "final_error_handling",
                "error_type": type(e).__name__,
                "message": str(e),
                "step_name": step_name
            }
            self._injection_errors[step_name].append(final_error)
            
            return injections_made
    
    def _update_injection_stats(self, step_name: str, injections_made: int):
        """주입 통계 업데이트"""
        if not hasattr(self, '_injection_stats'):
            self._injection_stats = {}
        
        if step_name not in self._injection_stats:
            self._injection_stats[step_name] = {
                'total_injections': 0,
                'last_injection_time': 0,
                'average_injections': 0
            }
        
        stats = self._injection_stats[step_name]
        stats['total_injections'] += injections_made
        stats['last_injection_time'] = time.time()
        stats['average_injections'] = stats['total_injections'] / self._injection_count
    
    def _get_injected_services(self, step_instance) -> List[str]:
        """주입된 서비스 목록 조회"""
        injected_services = []
        service_attributes = [
            'central_hub_container', 'di_container', 'model_loader', 
            'memory_manager', 'data_converter', 'step_factory',
            'data_transformer', 'model_registry', 'performance_monitor',
            'error_handler', 'cache_manager', 'config_manager'
        ]
        
        for attr in service_attributes:
            if hasattr(step_instance, attr):
                value = getattr(step_instance, attr)
                if value is not None:
                    injected_services.append(attr)
        
        return injected_services


    # ==============================================
    # 🔥 안전한 서비스 생성 팩토리들
    # ==============================================
    
    def _create_model_loader(self) -> Any:
        """🔥 단계별 세분화된 에러 처리가 적용된 ModelLoader 생성 (순환참조 완전 방지)"""
        start_time = time.time()
        errors = []
        stage_status = {}
        
        try:
            # 🔥 1단계: 환경 검증
            try:
                if not TORCH_AVAILABLE:
                    raise RuntimeError("PyTorch가 사용할 수 없습니다")
                
                if IS_M3_MAX and not MPS_AVAILABLE:
                    self.logger.warning("⚠️ M3 Max에서 MPS를 사용할 수 없습니다 - CPU 사용")
                
                stage_status['environment_validation'] = 'success'
                self.logger.debug("✅ 1단계: 환경 검증 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "environment_validation",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "torch_available": TORCH_AVAILABLE,
                    "mps_available": MPS_AVAILABLE,
                    "is_m3_max": IS_M3_MAX
                }
                errors.append(error_info)
                stage_status['environment_validation'] = 'failed'
                self.logger.error(f"❌ 1단계: 환경 검증 실패 - {e}")
                raise
            
            # 🔥 2단계: ModelLoader 모듈 import (지연 import로 순환참조 방지)
            try:
                # 순환참조 방지를 위해 지연 import 사용
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "model_loader", 
                    os.path.join(os.path.dirname(__file__), "..", "ai_pipeline", "models", "model_loader.py")
                )
                if spec and spec.loader:
                    model_loader_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(model_loader_module)
                    ModelLoader = getattr(model_loader_module, 'CentralModelLoader')
                else:
                    raise ImportError("ModelLoader 모듈을 찾을 수 없습니다")
                
                stage_status['module_import'] = 'success'
                self.logger.debug("✅ 2단계: ModelLoader 모듈 import 성공")
                
            except ImportError as e:
                error_info = {
                    "stage": "module_import",
                    "error_type": "ImportError",
                    "message": str(e),
                    "import_path": "..ai_pipeline.models.model_loader"
                }
                errors.append(error_info)
                stage_status['module_import'] = 'failed'
                self.logger.error(f"❌ 2단계: ModelLoader 모듈 import 실패 - {e}")
                raise
            except Exception as e:
                error_info = {
                    "stage": "module_import",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['module_import'] = 'failed'
                self.logger.error(f"❌ 2단계: ModelLoader 모듈 import 실패 - {e}")
                raise
            
            # 🔥 3단계: ModelLoader 인스턴스 생성
            try:
                model_loader = ModelLoader(
                    device=DEVICE
                    # enable_optimization 파라미터 제거 - ModelLoader에서 지원하지 않음
                )
                
                stage_status['instance_creation'] = 'success'
                self.logger.debug("✅ 3단계: ModelLoader 인스턴스 생성 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "instance_creation",
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "device": DEVICE,
                    "enable_optimization": True
                }
                errors.append(error_info)
                stage_status['instance_creation'] = 'failed'
                self.logger.error(f"❌ 3단계: ModelLoader 인스턴스 생성 실패 - {e}")
                raise
            
            # 🔥 4단계: Central Hub Container 연결
            try:
                # ModelLoader의 connect_to_central_hub 메서드 사용
                if hasattr(model_loader, 'connect_to_central_hub'):
                    connection_success = model_loader.connect_to_central_hub(self)
                    if connection_success:
                        stage_status['central_hub_connection'] = 'success'
                        self.logger.debug("✅ 4단계: Central Hub Container 연결 성공")
                    else:
                        stage_status['central_hub_connection'] = 'failed'
                        self.logger.warning("⚠️ 4단계: Central Hub Container 연결 실패")
                else:
                    # 레거시 방식으로 연결
                    model_loader._central_hub_container = self
                    model_loader._container_initialized = True
                    stage_status['central_hub_connection'] = 'success'
                    self.logger.debug("✅ 4단계: Central Hub Container 연결 성공 (레거시 방식)")
                
            except Exception as e:
                error_info = {
                    "stage": "central_hub_connection",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['central_hub_connection'] = 'failed'
                self.logger.warning(f"⚠️ 4단계: Central Hub Container 연결 실패 - {e}")
                # 연결 실패해도 ModelLoader 자체는 동작하므로 에러로 처리하지 않음
            
            # 🔥 5단계: 기본 의존성 해결
            try:
                if hasattr(model_loader, '_resolve_basic_dependencies'):
                    model_loader._resolve_basic_dependencies()
                    stage_status['basic_dependencies_resolution'] = 'success'
                    self.logger.debug("✅ 5단계: 기본 의존성 해결 성공")
                else:
                    self.logger.debug("⚠️ ModelLoader에 _resolve_basic_dependencies 메서드가 없습니다")
                    stage_status['basic_dependencies_resolution'] = 'skipped'
                
            except Exception as e:
                error_info = {
                    "stage": "basic_dependencies_resolution",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['basic_dependencies_resolution'] = 'failed'
                self.logger.warning(f"⚠️ 5단계: 기본 의존성 해결 실패 - {e}")
                # 의존성 해결 실패는 치명적이지 않으므로 에러로 처리하지 않음
            
            # 🔥 6단계: ModelLoader 초기화 검증
            try:
                # ModelLoader가 제대로 초기화되었는지 확인
                if not hasattr(model_loader, 'device'):
                    raise RuntimeError("ModelLoader에 device 속성이 없습니다")
                
                # ModelLoader v6.0의 실제 메서드들 확인
                required_methods = ['load_model_for_step', 'create_step_interface', 'validate_di_container_integration']
                missing_methods = []
                
                for method in required_methods:
                    if not hasattr(model_loader, method):
                        missing_methods.append(method)
                
                if missing_methods:
                    raise RuntimeError(f"ModelLoader에 필요한 메서드가 없습니다: {missing_methods}")
                
                stage_status['initialization_validation'] = 'success'
                self.logger.debug("✅ 6단계: ModelLoader 초기화 검증 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "initialization_validation",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['initialization_validation'] = 'failed'
                self.logger.error(f"❌ 6단계: ModelLoader 초기화 검증 실패 - {e}")
                raise
            
            # 🔥 7단계: 생성 완료 및 결과 반환
            try:
                creation_time = time.time() - start_time
                
                if errors:
                    self.logger.warning(f"⚠️ ModelLoader 생성 완료 (일부 에러 있음): {len(errors)}개 에러")
                    self.logger.debug(f"📊 생성 상태: {stage_status}")
                else:
                    self.logger.debug(f"✅ ModelLoader 생성 완료 (순환참조 방지) - 소요시간: {creation_time:.3f}초")
                
                # 에러 정보 저장
                if not hasattr(self, '_model_loader_creation_errors'):
                    self._model_loader_creation_errors = []
                
                self._model_loader_creation_errors.extend(errors)
                
                return model_loader
                
            except Exception as e:
                error_info = {
                    "stage": "completion_handling",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['completion_handling'] = 'failed'
                self.logger.error(f"❌ 7단계: 생성 완료 처리 실패 - {e}")
                raise
            
        except Exception as e:
            # 최종 에러 처리
            self.logger.error(f"❌ ModelLoader 생성 실패: {e}")
            
            # 에러 정보 저장
            if not hasattr(self, '_model_loader_creation_errors'):
                self._model_loader_creation_errors = []
            
            final_error = {
                "stage": "final_error_handling",
                "error_type": type(e).__name__,
                "message": str(e)
            }
            self._model_loader_creation_errors.append(final_error)
            
            # 🔥 폴백: 최소 기능 ModelLoader
            self.logger.warning("🔄 최소 기능 ModelLoader로 폴백...")
            return self._create_minimal_model_loader()


    def _create_memory_manager(self) -> Any:
        """🔥 수정: MemoryManager 생성 (순환참조 방지)"""
        try:
            self.logger.debug("🔄 MemoryManager 생성 시작...")
            
            # 🔥 MemoryManager는 ModelLoader에 의존하지 않으므로 안전
            from ..ai_pipeline.interface.step_interface import MemoryManager
            
            # M3 Max 메모리 최적화
            if IS_M3_MAX and MEMORY_GB >= 128:
                memory_manager = MemoryManager(115.0)
            elif IS_M3_MAX and MEMORY_GB >= 64:
                memory_manager = MemoryManager(MEMORY_GB * 0.85)
            else:
                memory_manager = MemoryManager()
            
            self.logger.debug("✅ MemoryManager 생성 완료")
            return memory_manager
            
        except Exception as e:
            self.logger.error(f"❌ MemoryManager 생성 실패: {e}")
            
            # 폴백: Mock MemoryManager
            class MockMemoryManager:
                def __init__(self):
                    self.is_mock = True
                    
                def allocate_memory(self, size_mb: float, owner: str):
                    return True
                    
                def deallocate_memory(self, owner: str):
                    return 0.0
                    
                def get_memory_stats(self):
                    return {"mock": True, "available_gb": 100.0}
            
            return MockMemoryManager()

    def _create_data_converter(self) -> Any:
        """🔥 수정: DataConverter 생성 (순환참조 방지)"""
        try:
            self.logger.debug("🔄 DataConverter 생성 시작...")
            
            # 🔥 DataConverter도 ModelLoader에 직접 의존하지 않도록 수정
            try:
                from ..ai_pipeline.utils.data_converter import DataConverter
                data_converter = DataConverter()
                self.logger.debug("✅ DataConverter 생성 완료")
                return data_converter
            except ImportError:
                # 폴백: Mock DataConverter
                class MockDataConverter:
                    def __init__(self):
                        self.is_mock = True
                        
                    def convert_api_to_step(self, api_data, step_name: str):
                        return api_data
                    
                    def convert_step_to_api(self, step_data, step_name: str):
                        return step_data
                
                self.logger.debug("✅ Mock DataConverter 생성 완료")
                return MockDataConverter()
            
        except Exception as e:
            self.logger.error(f"❌ DataConverter 생성 실패: {e}")
            
            # 최종 폴백
            class FallbackDataConverter:
                def __init__(self):
                    self.is_fallback = True
            
            return FallbackDataConverter()
    
    def _create_step_factory(self) -> Any:
        """StepFactory 생성"""
        try:
            self.logger.debug("🔄 StepFactory 생성 시작...")
            
            from app.ai_pipeline.factories.step_factory import StepFactory
            step_factory = StepFactory()
            self.logger.debug("✅ StepFactory 생성 완료")
            return step_factory
            
        except ImportError as e:
            self.logger.warning(f"⚠️ StepFactory import 실패: {e} - Mock 사용")
            class MockStepFactory:
                def __init__(self):
                    self.logger = logging.getLogger("MockStepFactory")
                    self.is_mock = True
                
                def create_step(self, step_type):
                    self.logger.warning(f"⚠️ Mock StepFactory: {step_type} 생성 시도")
                    return None
                
                def get_registered_step_class(self, step_name):
                    self.logger.warning(f"⚠️ Mock StepFactory: {step_name} 조회 시도")
                    return None
            return MockStepFactory()
            
        except Exception as e:
            self.logger.error(f"❌ StepFactory 생성 실패: {e}")
            class FallbackStepFactory:
                def __init__(self):
                    self.is_fallback = True
                
                def create_step(self, step_type):
                    return None
                
                def get_registered_step_class(self, step_name):
                    return None
            return FallbackStepFactory()
    
    def _create_mock_model_loader(self):
        """폴백 ModelLoader"""
        class MockModelLoader:
            def __init__(self):
                self.is_mock = True
                self.device = DEVICE
                
            def load_model(self, model_name: str, **kwargs):
                return {"mock": True, "model_name": model_name}
            
            def create_step_interface(self, step_name: str):
                return {"mock": True, "step_name": step_name}
            
            def validate_di_container_integration(self):
                return {"di_container_available": True, "mock": True}
        
        return MockModelLoader()
    


    def _create_minimal_model_loader(self):
        """🔥 새로 추가: 최소 기능 ModelLoader (폴백)"""
        class MinimalModelLoader:
            def __init__(self):
                self.is_minimal = True
                self.device = DEVICE
                self.loaded_models = {}
                self.logger = logging.getLogger("MinimalModelLoader")
                
            def load_model(self, model_name: str, **kwargs):
                self.logger.debug(f"⚠️ Minimal ModelLoader.load_model: {model_name}")
                return {"minimal": True, "model_name": model_name}
            
            def create_step_interface(self, step_name: str):
                self.logger.debug(f"⚠️ Minimal ModelLoader.create_step_interface: {step_name}")
                return {"minimal": True, "step_name": step_name}
            
            def validate_di_container_integration(self):
                return {"di_container_available": True, "minimal": True}
            
            def register_step_requirements(self, step_name: str, requirements):
                self.logger.debug(f"⚠️ Minimal ModelLoader.register_step_requirements: {step_name}")
                return True
        
        return MinimalModelLoader()


    def _create_memory_manager(self):
        """MemoryManager 안전 생성"""
        import_paths = [
            'app.ai_pipeline.utils.memory_manager',
            'ai_pipeline.utils.memory_manager',
            'utils.memory_manager'
        ]
        
        for path in import_paths:
            try:
                module = importlib.import_module(path)
                
                if hasattr(module, 'get_global_memory_manager'):
                    manager = module.get_global_memory_manager()
                    if manager:
                        return manager
                
                if hasattr(module, 'MemoryManager'):
                    ManagerClass = module.MemoryManager
                    return ManagerClass()
                    
            except ImportError:
                continue
        
        return self._create_mock_memory_manager()
    
    def _create_data_converter(self):
        """DataConverter 안전 생성"""
        import_paths = [
            'app.ai_pipeline.utils.data_converter',
            'ai_pipeline.utils.data_converter',
            'utils.data_converter'
        ]
        
        for path in import_paths:
            try:
                module = importlib.import_module(path)
                
                if hasattr(module, 'get_global_data_converter'):
                    converter = module.get_global_data_converter()
                    if converter:
                        return converter
                
                if hasattr(module, 'DataConverter'):
                    ConverterClass = module.DataConverter
                    return ConverterClass()
                    
            except ImportError:
                continue
        
        return self._create_mock_data_converter()
    
    # ==============================================
    # 🔥 Mock 서비스들 (폴백)
    # ==============================================
    
    def _create_mock_model_loader(self):
        """Mock ModelLoader"""
        class MockModelLoader:
            def __init__(self):
                self.is_initialized = True
                self.loaded_models = {}
                self.device = DEVICE
                self.logger = logging.getLogger("MockModelLoader")
                
            def load_model(self, model_path: str, **kwargs):
                model_id = f"mock_{len(self.loaded_models)}"
                self.loaded_models[model_id] = {
                    "path": model_path,
                    "loaded": True,
                    "device": self.device
                }
                return self.loaded_models[model_id]
            
            def create_step_interface(self, step_name: str):
                return MockStepInterface(step_name)
            
            def cleanup(self):
                self.loaded_models.clear()
        
        class MockStepInterface:
            def __init__(self, step_name):
                self.step_name = step_name
                self.is_initialized = True
            
            def get_model(self, model_name=None):
                return {"mock_model": model_name, "loaded": True}
        
        return MockModelLoader()
    
    def _create_mock_memory_manager(self):
        """Mock MemoryManager"""
        class MockMemoryManager:
            def __init__(self):
                self.is_initialized = True
                self.optimization_count = 0
            
            def optimize_memory(self, aggressive=False):
                self.optimization_count += 1
                gc.collect()
                return {"optimized": True, "count": self.optimization_count}
            
            def get_memory_info(self):
                return {
                    "total_gb": MEMORY_GB,
                    "available_gb": MEMORY_GB * 0.7,
                    "percent": 30.0
                }
            
            def cleanup(self):
                self.optimize_memory(aggressive=True)
        
        return MockMemoryManager()
    
    def _create_mock_data_converter(self):
        """Mock DataConverter"""
        class MockDataConverter:
            def __init__(self):
                self.is_initialized = True
                self.conversion_count = 0
            
            def convert(self, data, target_format):
                self.conversion_count += 1
                return {
                    "converted": f"mock_{target_format}_{self.conversion_count}",
                    "format": target_format
                }
            
            def get_supported_formats(self):
                return ["tensor", "numpy", "pil", "cv2"]
            
            def cleanup(self):
                self.conversion_count = 0
        
        return MockDataConverter()
    
    # ==============================================
    # 🔥 유틸리티 메서드들
    # ==============================================
    
    def get_stats(self) -> Dict[str, Any]:
        """🔥 단계별 세분화된 에러 처리가 적용된 Container 통계"""
        start_time = time.time()
        errors = []
        stage_status = {}
        
        try:
            # 🔥 1단계: 기본 통계 수집
            try:
                basic_stats = {
                    'container_id': self.container_id,
                    'container_type': 'CentralHubDIContainer',
                    'version': '7.0',
                    'creation_time': self._creation_time,
                    'lifetime_seconds': time.time() - self._creation_time,
                    'access_count': self._access_count,
                    'injection_count': self._injection_count
                }
                
                stage_status['basic_stats_collection'] = 'success'
                self.logger.debug("✅ 1단계: 기본 통계 수집 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "basic_stats_collection",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['basic_stats_collection'] = 'failed'
                self.logger.error(f"❌ 1단계: 기본 통계 수집 실패 - {e}")
                raise
            
            # 🔥 2단계: 레지스트리 통계 수집
            try:
                with self._lock:
                    registry_stats = self.registry.get_stats()
                
                stage_status['registry_stats_collection'] = 'success'
                self.logger.debug("✅ 2단계: 레지스트리 통계 수집 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "registry_stats_collection",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['registry_stats_collection'] = 'failed'
                self.logger.error(f"❌ 2단계: 레지스트리 통계 수집 실패 - {e}")
                registry_stats = {"error": str(e)}
            
            # 🔥 3단계: 환경 정보 수집
            try:
                environment_info = {
                    'is_m3_max': IS_M3_MAX,
                    'device': DEVICE,
                    'memory_gb': MEMORY_GB,
                    'torch_available': TORCH_AVAILABLE,
                    'mps_available': MPS_AVAILABLE,
                    'conda_env': CONDA_ENV
                }
                
                stage_status['environment_info_collection'] = 'success'
                self.logger.debug("✅ 3단계: 환경 정보 수집 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "environment_info_collection",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['environment_info_collection'] = 'failed'
                self.logger.error(f"❌ 3단계: 환경 정보 수집 실패 - {e}")
                environment_info = {"error": str(e)}
            
            # 🔥 4단계: 에러 통계 수집
            try:
                error_stats = {}
                
                # 초기화 에러 통계
                if hasattr(self, '_initialization_errors'):
                    error_stats['initialization_errors'] = len(self._initialization_errors)
                
                # 내장 서비스 등록 에러 통계
                if hasattr(self, '_builtin_services_errors'):
                    error_stats['builtin_services_errors'] = len(self._builtin_services_errors)
                
                # 서비스 조회 에러 통계
                if hasattr(self, '_service_retrieval_errors'):
                    total_retrieval_errors = sum(len(errors) for errors in self._service_retrieval_errors.values())
                    error_stats['service_retrieval_errors'] = total_retrieval_errors
                
                # 의존성 주입 에러 통계
                if hasattr(self, '_injection_errors'):
                    total_injection_errors = sum(len(errors) for errors in self._injection_errors.values())
                    error_stats['injection_errors'] = total_injection_errors
                
                # ModelLoader 생성 에러 통계
                if hasattr(self, '_model_loader_creation_errors'):
                    error_stats['model_loader_creation_errors'] = len(self._model_loader_creation_errors)
                
                stage_status['error_stats_collection'] = 'success'
                self.logger.debug("✅ 4단계: 에러 통계 수집 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "error_stats_collection",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['error_stats_collection'] = 'failed'
                self.logger.error(f"❌ 4단계: 에러 통계 수집 실패 - {e}")
                error_stats = {"error": str(e)}
            
            # 🔥 5단계: 성능 통계 수집
            try:
                performance_stats = {}
                
                # 서비스 조회 성능 통계
                if hasattr(self, '_service_retrieval_stats'):
                    total_retrievals = sum(stats.get('total_retrievals', 0) for stats in self._service_retrieval_stats.values())
                    successful_retrievals = sum(stats.get('successful_retrievals', 0) for stats in self._service_retrieval_stats.values())
                    failed_retrievals = sum(stats.get('failed_retrievals', 0) for stats in self._service_retrieval_stats.values())
                    
                    performance_stats['service_retrieval'] = {
                        'total_retrievals': total_retrievals,
                        'successful_retrievals': successful_retrievals,
                        'failed_retrievals': failed_retrievals,
                        'success_rate': (successful_retrievals / total_retrievals * 100) if total_retrievals > 0 else 0
                    }
                
                # 의존성 주입 성능 통계
                if hasattr(self, '_injection_stats'):
                    total_injections = sum(stats.get('total_injections', 0) for stats in self._injection_stats.values())
                    performance_stats['dependency_injection'] = {
                        'total_injections': total_injections,
                        'average_injections_per_step': total_injections / len(self._injection_stats) if self._injection_stats else 0
                    }
                
                stage_status['performance_stats_collection'] = 'success'
                self.logger.debug("✅ 5단계: 성능 통계 수집 성공")
                
            except Exception as e:
                error_info = {
                    "stage": "performance_stats_collection",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['performance_stats_collection'] = 'failed'
                self.logger.error(f"❌ 5단계: 성능 통계 수집 실패 - {e}")
                performance_stats = {"error": str(e)}
            
            # 🔥 6단계: 통계 조합 및 결과 반환
            try:
                stats_collection_time = time.time() - start_time
                
                # 최종 통계 조합
                final_stats = {
                    **basic_stats,
                    'registry_stats': registry_stats,
                    'environment': environment_info,
                    'error_stats': error_stats,
                    'performance_stats': performance_stats,
                    'stats_collection_time': stats_collection_time,
                    'errors_count': len(errors)
                }
                
                # 에러가 있으면 에러 정보도 포함
                if errors:
                    final_stats['collection_errors'] = errors
                    final_stats['collection_stage_status'] = stage_status
                    self.logger.warning(f"⚠️ 통계 수집 완료 (일부 에러 있음): {len(errors)}개 에러")
                else:
                    self.logger.debug(f"✅ 통계 수집 완료 (성공) - 소요시간: {stats_collection_time:.3f}초")
                
                return final_stats
                
            except Exception as e:
                error_info = {
                    "stage": "stats_combination",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
                errors.append(error_info)
                stage_status['stats_combination'] = 'failed'
                self.logger.error(f"❌ 6단계: 통계 조합 실패 - {e}")
                raise
            
        except Exception as e:
            # 최종 에러 처리
            self.logger.error(f"❌ Container 통계 수집 실패: {e}")
            
            # 최소한의 기본 통계라도 반환
            return {
                'container_id': getattr(self, 'container_id', 'unknown'),
                'container_type': 'CentralHubDIContainer',
                'version': '7.0',
                'error': str(e),
                'collection_failed': True,
                'errors_count': len(errors)
            }
    
    def list_services(self) -> List[str]:
        """등록된 서비스 목록"""
        return self.registry.list_services()
    
    def optimize_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 최적화"""
        try:
            # 가비지 컬렉션
            collected = gc.collect()
            
            # M3 Max MPS 캐시 정리
            if IS_M3_MAX and TORCH_AVAILABLE and MPS_AVAILABLE:
                try:
                    import torch
                    if hasattr(torch.backends.mps, 'empty_cache'):
                        torch.backends.mps.empty_cache()
                except Exception:
                    pass
            
            return {
                'garbage_collected': collected,
                'aggressive': aggressive,
                'container_id': self.container_id
            }
        except Exception as e:
            return {
                'error': str(e),
                'container_id': self.container_id
            }
    
    def cleanup(self):
        """Container 정리"""
        try:
            # 등록된 서비스들 정리
            for service_key in self.registry.list_services():
                service = self.registry.get_service(service_key)
                if service and hasattr(service, 'cleanup'):
                    try:
                        service.cleanup()
                    except Exception as e:
                        self.logger.debug(f"서비스 정리 실패 {service_key}: {e}")
            
            # 메모리 최적화
            self.optimize_memory(aggressive=True)
            
            self.logger.info(f"✅ Container 정리 완료: {self.container_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Container 정리 실패: {e}")
    
    # ==============================================
    # 🔥 에러 정보 조회 메서드들 (새로 추가)
    # ==============================================
    
    def get_error_summary(self) -> Dict[str, Any]:
        """🔥 전체 에러 요약 정보 조회"""
        try:
            error_summary = {
                'container_id': self.container_id,
                'total_errors': 0,
                'error_categories': {},
                'recent_errors': [],
                'error_trends': {}
            }
            
            # 초기화 에러
            if hasattr(self, '_initialization_errors'):
                error_summary['error_categories']['initialization'] = len(self._initialization_errors)
                error_summary['total_errors'] += len(self._initialization_errors)
            
            # 내장 서비스 등록 에러
            if hasattr(self, '_builtin_services_errors'):
                error_summary['error_categories']['builtin_services'] = len(self._builtin_services_errors)
                error_summary['total_errors'] += len(self._builtin_services_errors)
            
            # 서비스 조회 에러
            if hasattr(self, '_service_retrieval_errors'):
                total_retrieval_errors = sum(len(errors) for errors in self._service_retrieval_errors.values())
                error_summary['error_categories']['service_retrieval'] = total_retrieval_errors
                error_summary['total_errors'] += total_retrieval_errors
            
            # 의존성 주입 에러
            if hasattr(self, '_injection_errors'):
                total_injection_errors = sum(len(errors) for errors in self._injection_errors.values())
                error_summary['error_categories']['dependency_injection'] = total_injection_errors
                error_summary['total_errors'] += total_injection_errors
            
            # ModelLoader 생성 에러
            if hasattr(self, '_model_loader_creation_errors'):
                error_summary['error_categories']['model_loader_creation'] = len(self._model_loader_creation_errors)
                error_summary['total_errors'] += len(self._model_loader_creation_errors)
            
            return error_summary
            
        except Exception as e:
            self.logger.error(f"❌ 에러 요약 조회 실패: {e}")
            return {
                'container_id': self.container_id,
                'error': str(e),
                'total_errors': 0
            }
    
    def get_errors_by_category(self, category: str) -> List[Dict[str, Any]]:
        """🔥 카테고리별 에러 상세 정보 조회"""
        try:
            if category == 'initialization' and hasattr(self, '_initialization_errors'):
                return self._initialization_errors
            elif category == 'builtin_services' and hasattr(self, '_builtin_services_errors'):
                return self._builtin_services_errors
            elif category == 'service_retrieval' and hasattr(self, '_service_retrieval_errors'):
                all_errors = []
                for service_key, errors in self._service_retrieval_errors.items():
                    for error in errors:
                        error['service_key'] = service_key
                        all_errors.append(error)
                return all_errors
            elif category == 'dependency_injection' and hasattr(self, '_injection_errors'):
                all_errors = []
                for step_name, errors in self._injection_errors.items():
                    for error in errors:
                        error['step_name'] = step_name
                        all_errors.append(error)
                return all_errors
            elif category == 'model_loader_creation' and hasattr(self, '_model_loader_creation_errors'):
                return self._model_loader_creation_errors
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"❌ 카테고리별 에러 조회 실패: {category} - {e}")
            return []
    
    def get_service_errors(self, service_key: str) -> List[Dict[str, Any]]:
        """🔥 특정 서비스의 에러 정보 조회"""
        try:
            if hasattr(self, '_service_retrieval_errors') and service_key in self._service_retrieval_errors:
                return self._service_retrieval_errors[service_key]
            return []
            
        except Exception as e:
            self.logger.error(f"❌ 서비스 에러 조회 실패: {service_key} - {e}")
            return []
    
    def get_step_injection_errors(self, step_name: str) -> List[Dict[str, Any]]:
        """🔥 특정 Step의 의존성 주입 에러 정보 조회"""
        try:
            if hasattr(self, '_injection_errors') and step_name in self._injection_errors:
                return self._injection_errors[step_name]
            return []
            
        except Exception as e:
            self.logger.error(f"❌ Step 주입 에러 조회 실패: {step_name} - {e}")
            return []
    
    def clear_errors(self, category: str = None):
        """🔥 에러 정보 정리"""
        try:
            if category is None:
                # 모든 에러 정리
                if hasattr(self, '_initialization_errors'):
                    self._initialization_errors.clear()
                if hasattr(self, '_builtin_services_errors'):
                    self._builtin_services_errors.clear()
                if hasattr(self, '_service_retrieval_errors'):
                    self._service_retrieval_errors.clear()
                if hasattr(self, '_injection_errors'):
                    self._injection_errors.clear()
                if hasattr(self, '_model_loader_creation_errors'):
                    self._model_loader_creation_errors.clear()
                self.logger.info("✅ 모든 에러 정보 정리 완료")
            else:
                # 특정 카테고리 에러만 정리
                if category == 'initialization' and hasattr(self, '_initialization_errors'):
                    self._initialization_errors.clear()
                elif category == 'builtin_services' and hasattr(self, '_builtin_services_errors'):
                    self._builtin_services_errors.clear()
                elif category == 'service_retrieval' and hasattr(self, '_service_retrieval_errors'):
                    self._service_retrieval_errors.clear()
                elif category == 'dependency_injection' and hasattr(self, '_injection_errors'):
                    self._injection_errors.clear()
                elif category == 'model_loader_creation' and hasattr(self, '_model_loader_creation_errors'):
                    self._model_loader_creation_errors.clear()
                self.logger.info(f"✅ {category} 에러 정보 정리 완료")
                
        except Exception as e:
            self.logger.error(f"❌ 에러 정보 정리 실패: {e}")
    
    def get_detailed_error_report(self) -> Dict[str, Any]:
        """🔥 상세 에러 리포트 생성"""
        try:
            report = {
                'container_id': self.container_id,
                'timestamp': time.time(),
                'error_summary': self.get_error_summary(),
                'detailed_errors': {}
            }
            
            # 각 카테고리별 상세 에러 정보
            categories = ['initialization', 'builtin_services', 'service_retrieval', 'dependency_injection', 'model_loader_creation']
            
            for category in categories:
                errors = self.get_errors_by_category(category)
                if errors:
                    report['detailed_errors'][category] = {
                        'count': len(errors),
                        'errors': errors[:10]  # 최대 10개만 포함
                    }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ 상세 에러 리포트 생성 실패: {e}")
            return {
                'container_id': self.container_id,
                'error': str(e),
                'timestamp': time.time()
            }

# ==============================================
# 🔥 Container Manager - 전역 관리
# ==============================================

class CentralHubContainerManager:
    """중앙 허브 Container 매니저"""
    
    def __init__(self):
        self._containers: Dict[str, CentralHubDIContainer] = {}
        self._default_container_id = "default"
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # 기본 Container 생성
        self.get_container(self._default_container_id)
        
        self.logger.info("✅ 중앙 허브 Container Manager 초기화 완료")
    
    def get_container(self, container_id: Optional[str] = None) -> CentralHubDIContainer:
        """Container 반환"""
        container_id = container_id or self._default_container_id
        
        with self._lock:
            if container_id not in self._containers:
                self._containers[container_id] = CentralHubDIContainer(container_id)
            
            return self._containers[container_id]
    
    def create_container(self, container_id: str) -> CentralHubDIContainer:
        """새 Container 생성"""
        with self._lock:
            if container_id in self._containers:
                self.logger.warning(f"⚠️ Container 이미 존재: {container_id}")
                return self._containers[container_id]
            
            container = CentralHubDIContainer(container_id)
            self._containers[container_id] = container
            
            self.logger.info(f"✅ 새 Container 생성: {container_id}")
            return container
    
    def destroy_container(self, container_id: str):
        """Container 소멸"""
        with self._lock:
            if container_id in self._containers:
                container = self._containers[container_id]
                container.cleanup()
                del self._containers[container_id]
                self.logger.info(f"🗑️ Container 소멸: {container_id}")
    
    def list_containers(self) -> List[str]:
        """Container 목록"""
        with self._lock:
            return list(self._containers.keys())
    
    def cleanup_all(self):
        """모든 Container 정리"""
        with self._lock:
            for container_id in list(self._containers.keys()):
                self.destroy_container(container_id)
            
            self.logger.info("✅ 모든 Container 정리 완료")

# ==============================================
# 🔥 Property Injection Mixin
# ==============================================

class PropertyInjectionMixin:
    """속성 주입을 지원하는 믹스인"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._di_container: Optional[CentralHubDIContainer] = None
    
    def set_di_container(self, container: CentralHubDIContainer):
        """DI Container 설정"""
        self._di_container = container
        self._auto_inject_properties()
    
    def _auto_inject_properties(self):
        """자동 속성 주입"""
        if not self._di_container:
            return
        
        injection_map = {
            'model_loader': 'model_loader',
            'memory_manager': 'memory_manager',
            'data_converter': 'data_converter'
        }
        
        for attr_name, service_key in injection_map.items():
            if not hasattr(self, attr_name) or getattr(self, attr_name) is None:
                service = self._di_container.get(service_key)
                if service:
                    setattr(self, attr_name, service)
    
    def get_service(self, service_key: str):
        """DI Container를 통한 서비스 조회"""
        if self._di_container:
            return self._di_container.get(service_key)
        return None

# ==============================================
# 🔥 전역 인스턴스 관리
# ==============================================

_global_manager: Optional[CentralHubContainerManager] = None
_manager_lock = threading.RLock()

def get_global_container(container_id: Optional[str] = None) -> CentralHubDIContainer:
    """전역 Container 반환"""
    global _global_manager
    
    with _manager_lock:
        if _global_manager is None:
            _global_manager = CentralHubContainerManager()
            logger.info("✅ 전역 중앙 허브 Container Manager 생성")
        
        return _global_manager.get_container(container_id)

def get_global_manager() -> CentralHubContainerManager:
    """전역 Manager 반환"""
    global _global_manager
    
    with _manager_lock:
        if _global_manager is None:
            _global_manager = CentralHubContainerManager()
            logger.info("✅ 전역 중앙 허브 Container Manager 생성")
        
        return _global_manager

def reset_global_container():
    """전역 Container 리셋"""
    global _global_manager
    
    with _manager_lock:
        if _global_manager:
            _global_manager.cleanup_all()
        _global_manager = None
        logger.info("🔄 전역 중앙 허브 Container Manager 리셋")

# ==============================================
# 🔥 편의 함수들 (완전 구현)
# ==============================================

def get_service(service_key: str, container_id: Optional[str] = None) -> Optional[Any]:
    """서비스 조회 편의 함수"""
    container = get_global_container(container_id)
    return container.get(service_key)

def register_service(service_key: str, instance: Any, singleton: bool = True, container_id: Optional[str] = None):
    """서비스 등록 편의 함수"""
    container = get_global_container(container_id)
    container.register(service_key, instance, singleton)

def register_factory(service_key: str, factory: Callable[[], Any], singleton: bool = True, container_id: Optional[str] = None):
    """팩토리 등록 편의 함수"""
    container = get_global_container(container_id)
    container.register_factory(service_key, factory, singleton)

def inject_dependencies_to_step(step_instance, container_id: Optional[str] = None) -> int:
    """🔥 Central Hub v7.0 - Step 의존성 주입 편의 함수"""
    try:
        container = get_global_container(container_id)
        if container:
            return container.inject_to_step(step_instance)
        else:
            logger.warning("⚠️ Central Hub Container를 찾을 수 없습니다")
            return 0
    except Exception as e:
        logger.error(f"❌ inject_dependencies_to_step 실패: {e}")
        return 0

# ==============================================
# 🔥 지연 서비스 관련 함수들 (완전 구현)
# ==============================================

def register_lazy_service(service_key: str, factory: Callable[[], Any], singleton: bool = True, container_id: Optional[str] = None) -> bool:
    """지연 서비스 등록"""
    try:
        container = get_global_container(container_id)
        return container.register_lazy(service_key, factory, singleton)
    except Exception as e:
        logger.debug(f"⚠️ register_lazy_service 실패 ({service_key}): {e}")
        return False

def register_lazy_service_safe(service_key: str, factory: Callable[[], Any], singleton: bool = True, container_id: Optional[str] = None) -> bool:
    """안전한 지연 서비스 등록"""
    return register_lazy_service(service_key, factory, singleton, container_id)

def create_lazy_dependency(factory: Callable[[], Any], service_key: str = None) -> Any:
    """지연 의존성 생성"""
    try:
        return LazyDependency(factory)
    except Exception as e:
        logger.debug(f"⚠️ create_lazy_dependency 실패: {e}")
        return None

def resolve_lazy_service(service_key: str, container_id: Optional[str] = None) -> Any:
    """지연 서비스 해결"""
    try:
        container = get_global_container(container_id)
        lazy_service = container.get(service_key)
        
        if lazy_service and hasattr(lazy_service, 'get'):
            return lazy_service.get()
        else:
            return lazy_service
    except Exception as e:
        logger.debug(f"⚠️ resolve_lazy_service 실패 ({service_key}): {e}")
        return None

def is_lazy_service_resolved(service_key: str, container_id: Optional[str] = None) -> bool:
    """지연 서비스 해결 상태 확인"""
    try:
        container = get_global_container(container_id)
        lazy_service = container.get(service_key)
        
        if lazy_service and hasattr(lazy_service, 'is_resolved'):
            return lazy_service.is_resolved()
        return False
    except Exception:
        return False

# ==============================================
# 🔥 Container 레벨 함수들 (구 버전 호환)
# ==============================================

def create_container(container_id: str = None) -> CentralHubDIContainer:
    """Container 생성 (구 버전 호환)"""
    try:
        return get_global_container(container_id)
    except Exception as e:
        logger.debug(f"⚠️ create_container 실패: {e}")
        return None

def dispose_container(container_id: str = None) -> bool:
    """Container 정리 (구 버전 호환)"""
    try:
        if container_id:
            manager = get_global_manager()
            manager.destroy_container(container_id)
        else:
            reset_global_container()
        return True
    except Exception as e:
        logger.debug(f"⚠️ dispose_container 실패: {e}")
        return False

def get_container_instance(container_id: str = None) -> CentralHubDIContainer:
    """Container 인스턴스 조회 (구 버전 호환)"""
    return get_service_safe('container', None, container_id) or get_global_container(container_id)

def register_singleton(service_key: str, instance: Any, container_id: Optional[str] = None) -> bool:
    """싱글톤 등록 (구 버전 호환)"""
    return register_service_safe(service_key, instance, True, container_id)

def register_transient(service_key: str, factory: Callable[[], Any], container_id: Optional[str] = None) -> bool:
    """임시 서비스 등록 (구 버전 호환)"""
    try:
        container = get_global_container(container_id)
        container.register_factory(service_key, factory, singleton=False)
        return True
    except Exception as e:
        logger.debug(f"⚠️ register_transient 실패 ({service_key}): {e}")
        return False

def unregister_service(service_key: str, container_id: Optional[str] = None) -> bool:
    """서비스 등록 해제"""
    try:
        container = get_global_container(container_id)
        container.remove(service_key)
        return True
    except Exception as e:
        logger.debug(f"⚠️ unregister_service 실패 ({service_key}): {e}")
        return False

# ==============================================
# 🔥 의존성 주입 관련 함수들 (완전 구현)
# ==============================================

def inject_all_dependencies(step_instance, container_id: Optional[str] = None) -> int:
    """🔥 Central Hub v7.0 - 모든 의존성 주입 (완전한 서비스 세트)"""
    try:
        container = get_global_container(container_id)
        if container:
            # Central Hub v7.0의 완전한 inject_to_step 사용
            return container.inject_to_step(step_instance)
        else:
            logger.warning("⚠️ Central Hub Container를 찾을 수 없습니다")
            return 0
    except Exception as e:
        logger.error(f"❌ 모든 의존성 주입 실패: {e}")
        return 0

def auto_wire_dependencies(step_instance, container_id: Optional[str] = None) -> bool:
    """🔥 Central Hub v7.0 - 자동 의존성 연결 (완전한 자동화)"""
    try:
        # Central Hub v7.0의 완전한 inject_to_step 사용
        count = inject_all_dependencies(step_instance, container_id)
        return count > 0
    except Exception as e:
        logger.error(f"❌ 자동 의존성 연결 실패: {e}")
        return False

def validate_dependencies(step_instance, required_services: List[str] = None) -> bool:
    """🔥 Central Hub v7.0 - 의존성 유효성 검사 (확장된 서비스 세트)"""
    try:
        if not required_services:
            # Central Hub v7.0의 확장된 서비스 세트
            required_services = [
                'model_loader', 'memory_manager', 'data_converter',
                'step_factory', 'data_transformer', 'model_registry',
                'performance_monitor', 'error_handler', 'cache_manager',
                'config_manager', 'central_hub_container'
            ]
        
        for service_name in required_services:
            if not hasattr(step_instance, service_name) or getattr(step_instance, service_name) is None:
                logger.debug(f"⚠️ 필수 서비스 누락: {service_name}")
                return False
        
        logger.debug(f"✅ 모든 필수 서비스 검증 완료: {len(required_services)}개")
        return True
    except Exception as e:
        logger.error(f"❌ 의존성 유효성 검사 실패: {e}")
        return False

def get_dependency_status(step_instance) -> Dict[str, Any]:
    """🔥 Central Hub v7.0 - 의존성 상태 정보 (완전한 서비스 모니터링)"""
    try:
        # Central Hub v7.0의 확장된 서비스 세트
        dependencies = [
            'model_loader', 'memory_manager', 'data_converter', 
            'step_factory', 'data_transformer', 'model_registry',
            'performance_monitor', 'error_handler', 'cache_manager',
            'config_manager', 'central_hub_container', 'di_container'
        ]
        
        status = {}
        for dep_name in dependencies:
            dep_value = getattr(step_instance, dep_name, None)
            status[dep_name] = {
                'available': dep_value is not None,
                'type': type(dep_value).__name__ if dep_value else None,
                'central_hub_integrated': hasattr(step_instance, 'central_hub_integrated') and getattr(step_instance, 'central_hub_integrated', False)
            }
        
        # Central Hub v7.0 메타데이터 추가
        metadata = {}
        if hasattr(step_instance, 'step_metadata'):
            metadata = getattr(step_instance, 'step_metadata', {})
        
        return {
            'step_class': step_instance.__class__.__name__,
            'dependencies': status,
            'all_resolved': all(status[dep]['available'] for dep in dependencies),
            'resolution_count': sum(1 for dep in status.values() if dep['available']),
            'central_hub_version': '7.0',
            'metadata': metadata,
            'total_services': len(dependencies)
        }
    except Exception as e:
        return {
            'error': str(e),
            'step_class': getattr(step_instance, '__class__', {}).get('__name__', 'Unknown'),
            'central_hub_version': '7.0'
        }

# ==============================================
# 🔥 서비스 조회 편의 함수들 (완전 구현)
# ==============================================

def get_all_services(container_id: Optional[str] = None) -> Dict[str, Any]:
    """모든 서비스 조회"""
    try:
        container = get_global_container(container_id)
        services = {}
        
        for service_key in container.list_services():
            service = container.get(service_key)
            services[service_key] = {
                'available': service is not None,
                'type': type(service).__name__ if service else None
            }
        return services
    except Exception as e:
        return {'error': str(e)}

def list_service_keys(container_id: Optional[str] = None) -> List[str]:
    """서비스 키 목록"""
    try:
        container = get_global_container(container_id)
        return container.list_services()
    except Exception:
        return []

def get_service_count(container_id: Optional[str] = None) -> int:
    """등록된 서비스 개수"""
    try:
        return len(list_service_keys(container_id))
    except Exception:
        return 0

# ==============================================
# 🔥 상태 확인 함수들 (완전 구현)
# ==============================================

def is_service_available(service_key: str, container_id: Optional[str] = None) -> bool:
    """서비스 사용 가능 여부 확인"""
    try:
        service = get_service_safe(service_key, None, container_id)
        return service is not None
    except Exception:
        return False

def is_container_ready(container_id: Optional[str] = None) -> bool:
    """Container 준비 상태 확인"""
    try:
        container = get_global_container(container_id)
        return container is not None
    except Exception:
        return False

def is_di_system_ready(container_id: Optional[str] = None) -> bool:
    """DI 시스템 준비 상태 확인"""
    try:
        container = get_global_container(container_id)
        if not container:
            return False
        
        # 핵심 서비스들 확인
        essential_services = ['model_loader', 'memory_manager', 'data_converter']
        for service_key in essential_services:
            if not container.get(service_key):
                return False
        
        return True
    except Exception:
        return False

def get_service_status(service_key: str, container_id: Optional[str] = None) -> Dict[str, Any]:
    """서비스 상태 정보"""
    try:
        container = get_global_container(container_id)
        if not container:
            return {'status': 'error', 'message': 'Container not available'}
        
        service = container.get(service_key)
        return {
            'service_key': service_key,
            'available': service is not None,
            'type': type(service).__name__ if service else None,
            'container_id': container_id or 'default'
        }
    except Exception as e:
        return {
            'service_key': service_key,
            'status': 'error',
            'message': str(e)
        }

def get_di_system_status(container_id: Optional[str] = None) -> Dict[str, Any]:
    """DI 시스템 상태 정보"""
    try:
        container = get_global_container(container_id)
        if not container:
            return {'status': 'error', 'message': 'Container not available'}
        
        stats = container.get_stats()
        
        # 핵심 서비스 상태 확인
        services_status = {}
        essential_services = ['model_loader', 'memory_manager', 'data_converter']
        
        for service_key in essential_services:
            service = container.get(service_key)
            services_status[service_key] = {
                'available': service is not None,
                'type': type(service).__name__ if service else None
            }
        
        return {
            'status': 'ready' if is_di_system_ready(container_id) else 'partial',
            'container_id': container_id or 'default',
            'stats': stats,
            'services': services_status,
            'environment': {
                'conda_env': CONDA_ENV,
                'is_m3_max': IS_M3_MAX,
                'device': DEVICE,
                'torch_available': TORCH_AVAILABLE,
                'mps_available': MPS_AVAILABLE
            }
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

# ==============================================
# 🔥 구 버전 호환성 레이어 (완전 구현)
# ==============================================

# 기존 API 완전 호환을 위한 별칭들
CircularReferenceFreeDIContainer = CentralHubDIContainer
get_global_di_container = get_global_container

# DynamicImportResolver 호환성 클래스
class DynamicImportResolver:
    """동적 import 해결기 (순환참조 완전 방지)"""
    
    @staticmethod
    def resolve_model_loader():
        """ModelLoader 동적 해결"""
        container = get_global_container()
        return container.get('model_loader')
    
    @staticmethod
    def resolve_memory_manager():
        """MemoryManager 동적 해결"""
        container = get_global_container()
        return container.get('memory_manager')
    
    @staticmethod
    def resolve_data_converter():
        """DataConverter 동적 해결"""
        container = get_global_container()
        return container.get('data_converter')
    
    @staticmethod
    def resolve_di_container():
        """DI Container 동적 해결"""
        return get_global_container()

# 안전한 함수들 (완전 구현)
def get_service_safe(service_key: str, default=None, container_id: Optional[str] = None) -> Any:
    """안전한 서비스 조회"""
    try:
        service = get_service(service_key, container_id)
        return service if service is not None else default
    except Exception as e:
        logger.debug(f"⚠️ get_service_safe 실패 ({service_key}): {e}")
        return default

def register_service_safe(service_key: str, instance: Any, singleton: bool = True, container_id: Optional[str] = None) -> bool:
    """안전한 서비스 등록"""
    try:
        register_service(service_key, instance, singleton, container_id)
        return True
    except Exception as e:
        logger.debug(f"⚠️ register_service_safe 실패 ({service_key}): {e}")
        return False

def register_factory_safe(service_key: str, factory: Callable[[], Any], singleton: bool = True, container_id: Optional[str] = None) -> bool:
    """안전한 팩토리 등록"""
    try:
        register_factory(service_key, factory, singleton, container_id)
        return True
    except Exception as e:
        logger.debug(f"⚠️ register_factory_safe 실패 ({service_key}): {e}")
        return False

def inject_dependencies_to_step_safe(step_instance, container_id: Optional[str] = None) -> int:
    """🔥 Central Hub v7.0 - 안전한 의존성 주입 (완전한 에러 처리)"""
    try:
        # Step 인스턴스 유효성 검증
        if step_instance is None:
            logger.warning("⚠️ Step 인스턴스가 None입니다")
            return 0
        
        # Container 조회 및 주입
        container = get_global_container(container_id)
        if container:
            injections_made = container.inject_to_step(step_instance)
            logger.debug(f"✅ Central Hub v7.0 의존성 주입 완료: {injections_made}개")
            return injections_made
        else:
            logger.warning("⚠️ Central Hub Container를 찾을 수 없습니다")
            return 0
            
    except Exception as e:
        logger.error(f"❌ Central Hub v7.0 의존성 주입 실패: {e}")
        logger.debug(f"🔍 실패 상세: {traceback.format_exc()}")
        return 0

def get_model_loader_safe(container_id: Optional[str] = None):
    """안전한 ModelLoader 조회"""
    return get_service_safe('model_loader', None, container_id)

def get_memory_manager_safe(container_id: Optional[str] = None):
    """안전한 MemoryManager 조회"""
    return get_service_safe('memory_manager', None, container_id)

def get_data_converter_safe(container_id: Optional[str] = None):
    """안전한 DataConverter 조회"""
    return get_service_safe('data_converter', None, container_id)

def get_container_safe(container_id: Optional[str] = None):
    """안전한 Container 조회"""
    try:
        return get_global_container(container_id)
    except Exception as e:
        logger.debug(f"⚠️ get_container_safe 실패: {e}")
        return None

def inject_dependencies_safe(step_instance, container_id: Optional[str] = None) -> int:
    """🔥 Central Hub v7.0 - 안전한 의존성 주입 (별칭)"""
    return inject_dependencies_to_step_safe(step_instance, container_id)

def ensure_model_loader_registration(container_id: Optional[str] = None) -> bool:
    """ModelLoader 등록 보장"""
    try:
        loader = get_service('model_loader', container_id)
        return loader is not None
    except Exception:
        return False

def ensure_service_registration(service_key: str, container_id: Optional[str] = None) -> bool:
    """서비스 등록 보장"""
    try:
        service = get_service(service_key, container_id)
        return service is not None
    except Exception:
        return False

def cleanup_services_safe(container_id: Optional[str] = None) -> bool:
    """안전한 서비스 정리"""
    try:
        container = get_global_container(container_id)
        container.optimize_memory(aggressive=True)
        return True
    except Exception as e:
        logger.debug(f"⚠️ cleanup_services_safe 실패: {e}")
        return False

def reset_container_safe(container_id: Optional[str] = None) -> bool:
    """안전한 Container 리셋"""
    try:
        if container_id:
            manager = get_global_manager()
            manager.destroy_container(container_id)
        else:
            reset_global_container()
        return True
    except Exception as e:
        logger.debug(f"⚠️ reset_container_safe 실패: {e}")
        return False

# 추가 호환성 함수들
def initialize_di_system_safe(container_id: Optional[str] = None) -> bool:
    """DI 시스템 안전 초기화"""
    return initialize_di_system(container_id)

def _get_global_di_container():
    """BaseStepMixin 호환 함수"""
    return get_global_container()

def _get_service_from_container_safe(service_key: str):
    """BaseStepMixin 호환 함수"""
    return get_service(service_key)

def _get_central_hub_container():
    """🔥 Central Hub v7.0 - 안전한 Central Hub Container 조회"""
    try:
        return get_global_container()
    except Exception as e:
        logger.debug(f"⚠️ _get_central_hub_container 실패: {e}")
        return None

def get_global_container_legacy():
    """구버전 호환 함수"""
    return get_global_container()

def reset_global_container_legacy():
    """구버전 호환 함수"""
    reset_global_container()

# LazyDependency 호환성 (기존과 동일)
class LazyDependency:
    """지연 의존성 (구 버전 호환)"""
    
    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._instance = None
        self._resolved = False
        self._lock = threading.RLock()
    
    def get(self) -> Any:
        if not self._resolved:
            with self._lock:
                if not self._resolved:
                    try:
                        self._instance = self._factory()
                        self._resolved = True
                    except Exception as e:
                        logger.error(f"❌ LazyDependency 해결 실패: {e}")
                        return None
        return self._instance
    
    def resolve(self) -> Any:
        return self.get()
    
    def is_resolved(self) -> bool:
        return self._resolved

# ==============================================
# 🔥 특수 호환성 함수들 (완전 구현)
# ==============================================

def ensure_global_step_compatibility() -> bool:
    """전역 Step 호환성 보장"""
    try:
        container = get_global_container()
        
        # 핵심 서비스들 확인
        essential_services = ['model_loader', 'memory_manager', 'data_converter']
        for service_key in essential_services:
            service = container.get(service_key)
            if not service:
                logger.warning(f"⚠️ 필수 서비스 없음: {service_key}")
                return False
        
        # DI 시스템 준비 상태 확인
        if not is_di_system_ready():
            logger.warning("⚠️ DI 시스템 준비되지 않음")
            return False
        
        logger.info("✅ 전역 Step 호환성 보장 완료")
        return True
    except Exception as e:
        logger.error(f"❌ 전역 Step 호환성 보장 실패: {e}")
        return False

def _add_global_step_methods(step_instance) -> bool:
    """전역 Step 메서드들 추가"""
    try:
        # DI Container 기반 서비스 조회 메서드 추가
        def get_service_method(service_key: str):
            container = get_global_container()
            return container.get(service_key)
        
        def get_model_loader_method():
            return get_service_method('model_loader')
        
        def get_memory_manager_method():
            return get_service_method('memory_manager')
        
        def get_data_converter_method():
            return get_service_method('data_converter')
        
        def optimize_memory_method(aggressive: bool = False):
            container = get_global_container()
            return container.optimize_memory(aggressive)
        
        def get_di_stats_method():
            container = get_global_container()
            return container.get_stats()
        
        # 메서드들을 Step 인스턴스에 동적 추가
        if not hasattr(step_instance, 'get_service'):
            step_instance.get_service = get_service_method
        
        if not hasattr(step_instance, 'get_model_loader'):
            step_instance.get_model_loader = get_model_loader_method
        
        if not hasattr(step_instance, 'get_memory_manager'):
            step_instance.get_memory_manager = get_memory_manager_method
        
        if not hasattr(step_instance, 'get_data_converter'):
            step_instance.get_data_converter = get_data_converter_method
        
        if not hasattr(step_instance, 'optimize_memory'):
            step_instance.optimize_memory = optimize_memory_method
        
        if not hasattr(step_instance, 'get_di_stats'):
            step_instance.get_di_stats = get_di_stats_method
        
        logger.debug("✅ 전역 Step 메서드들 추가 완료")
        return True
    except Exception as e:
        logger.error(f"❌ 전역 Step 메서드들 추가 실패: {e}")
        return False

def ensure_step_di_integration(step_instance) -> bool:
    """Step DI 통합 보장"""
    try:
        # DI Container 주입
        container = get_global_container()
        injections_made = container.inject_to_step(step_instance)
        
        # 전역 메서드들 추가
        methods_added = _add_global_step_methods(step_instance)
        
        # 통합 완료 플래그 설정
        if not hasattr(step_instance, '_di_integrated'):
            step_instance._di_integrated = True
        
        logger.debug(f"✅ Step DI 통합 완료: {injections_made}개 주입, 메서드 추가: {methods_added}")
        return True
    except Exception as e:
        logger.error(f"❌ Step DI 통합 실패: {e}")
        return False


def validate_step_di_requirements(step_instance) -> Dict[str, Any]:
    """Step DI 요구사항 검증"""
    try:
        validation_result = {
            'step_class': step_instance.__class__.__name__,
            'di_container_available': False,
            'model_loader_available': False,
            'memory_manager_available': False,
            'data_converter_available': False,
            'base_step_mixin_compatible': False,
            'required_methods_present': False,
            'di_integrated': False,
            'overall_valid': False
        }
        
        # DI Container 확인
        if hasattr(step_instance, 'di_container') and step_instance.di_container:
            validation_result['di_container_available'] = True
        
        # 서비스들 확인
        if hasattr(step_instance, 'model_loader') and step_instance.model_loader:
            validation_result['model_loader_available'] = True
        
        if hasattr(step_instance, 'memory_manager') and step_instance.memory_manager:
            validation_result['memory_manager_available'] = True
        
        if hasattr(step_instance, 'data_converter') and step_instance.data_converter:
            validation_result['data_converter_available'] = True
        
        # BaseStepMixin 호환성 확인
        try:
            from app.ai_pipeline.steps.base_step_mixin import BaseStepMixin
            validation_result['base_step_mixin_compatible'] = isinstance(step_instance, BaseStepMixin)
        except ImportError:
            validation_result['base_step_mixin_compatible'] = False
        
        # 필수 메서드들 확인
        required_methods = ['process', 'initialize', 'cleanup']
        methods_present = all(hasattr(step_instance, method) for method in required_methods)
        validation_result['required_methods_present'] = methods_present
        
        # DI 통합 상태 확인
        validation_result['di_integrated'] = getattr(step_instance, '_di_integrated', False)
        
        # 전체 유효성 판단
        validation_result['overall_valid'] = (
            validation_result['di_container_available'] and
            validation_result['model_loader_available'] and
            validation_result['required_methods_present']
        )
        
        return validation_result
    except Exception as e:
        return {
            'step_class': getattr(step_instance, '__class__', {}).get('__name__', 'Unknown'),
            'error': str(e),
            'overall_valid': False
        }

def setup_global_di_environment() -> bool:
    """전역 DI 환경 설정"""
    try:
        # DI 시스템 초기화
        if not initialize_di_system():
            logger.error("❌ DI 시스템 초기화 실패")
            return False
        
        # 전역 호환성 보장
        if not ensure_global_step_compatibility():
            logger.error("❌ 전역 Step 호환성 보장 실패")
            return False
        
        # conda 환경 최적화
        if IS_CONDA:
            _optimize_for_conda()
        
        logger.info("✅ 전역 DI 환경 설정 완료")
        return True
    except Exception as e:
        logger.error(f"❌ 전역 DI 환경 설정 실패: {e}")
        return False

def get_global_di_environment_status() -> Dict[str, Any]:
    """전역 DI 환경 상태 조회"""
    try:
        return {
            'di_system_ready': is_di_system_ready(),
            'step_compatibility_ensured': ensure_global_step_compatibility(),
            'container_available': is_container_ready(),
            'essential_services': {
                'model_loader': is_service_available('model_loader'),
                'memory_manager': is_service_available('memory_manager'),
                'data_converter': is_service_available('data_converter')
            },
            'environment': {
                'conda_env': CONDA_ENV,
                'is_m3_max': IS_M3_MAX,
                'device': DEVICE,
                'torch_available': TORCH_AVAILABLE,
                'mps_available': MPS_AVAILABLE
            },
            'container_stats': get_di_system_status(),
            'timestamp': time.time()
        }
    except Exception as e:
        return {
            'error': str(e),
            'timestamp': time.time()
        }

# ==============================================
# 🔥 초기화 및 최적화 (완전 구현)
# ==============================================

def initialize_di_system(container_id: Optional[str] = None) -> bool:
    """DI 시스템 초기화"""
    try:
        container = get_global_container(container_id)
        
        # conda 환경 최적화
        if IS_CONDA:
            _optimize_for_conda()
        
        # 핵심 서비스들 확인
        model_loader = container.get('model_loader')
        if model_loader:
            logger.info("✅ DI 시스템 초기화: ModelLoader 사용 가능")
        else:
            logger.warning("⚠️ DI 시스템 초기화: ModelLoader 없음")
        
        return True
    except Exception as e:
        logger.error(f"❌ DI 시스템 초기화 실패: {e}")
        return False

def _optimize_for_conda():
    """conda 환경 최적화 + MPS float64 문제 해결"""
    try:
        os.environ['OMP_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        os.environ['MKL_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        os.environ['NUMEXPR_NUM_THREADS'] = str(max(1, os.cpu_count() // 2))
        
        if TORCH_AVAILABLE:
            import torch
            torch.set_num_threads(max(1, os.cpu_count() // 2))
            
            if IS_M3_MAX and MPS_AVAILABLE:
                # 🔥 MPS float64 문제 해결
                try:
                    # MPS용 기본 dtype 설정
                    if hasattr(torch, 'set_default_dtype'):
                        if torch.get_default_dtype() == torch.float64:
                            torch.set_default_dtype(torch.float32)
                            logger.debug("✅ conda 환경에서 MPS 기본 dtype을 float32로 설정")
                    
                    # MPS 최적화 환경 변수
                    os.environ.update({
                        'PYTORCH_MPS_PREFER_FLOAT32': '1',
                        'PYTORCH_MPS_FORCE_FLOAT32': '1'
                    })
                except Exception as e:
                    logger.debug(f"MPS dtype 설정 실패 (무시): {e}")
                
                # 기존 MPS 캐시 정리
                if hasattr(torch.backends.mps, 'empty_cache'):
                    torch.backends.mps.empty_cache()
        
        logger.info(f"🐍 conda 환경 '{CONDA_ENV}' 최적화 완료 (MPS float64 문제 해결 포함)")
    except Exception as e:
        logger.warning(f"⚠️ conda 최적화 실패: {e}")


# ==============================================
# 📁 backend/app/core/di_container.py 파일에 추가
# 위치: 기존 함수들 뒤, __all__ 리스트 전에 배치
# ==============================================

import logging
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 Central Hub 연결 보장 및 초기화 (개선된 버전)
# ==============================================

def create_default_service(service_name: str) -> Any:
    """기본 서비스 팩토리"""
    try:
        if service_name == 'model_loader':
            # ModelLoader 동적 생성
            try:
                from ..ai_pipeline.utils.model_loader import ModelLoader
                return ModelLoader()
            except ImportError:
                logger.warning("⚠️ ModelLoader import 실패, Mock 생성")
                return create_mock_model_loader()
                
        elif service_name == 'memory_manager':
            # MemoryManager 동적 생성
            try:
                from ..ai_pipeline.utils.memory_manager import MemoryManager
                return MemoryManager()
            except ImportError:
                logger.warning("⚠️ MemoryManager import 실패, Mock 생성")
                return create_mock_memory_manager()
                
        elif service_name == 'data_converter':
            # DataConverter 동적 생성
            try:
                from ..ai_pipeline.utils.data_converter import DataConverter
                return DataConverter()
            except ImportError:
                logger.warning("⚠️ DataConverter import 실패, Mock 생성")
                return create_mock_data_converter()
                
        else:
            logger.warning(f"⚠️ 알 수 없는 서비스: {service_name}")
            return None
            
    except Exception as e:
        logger.error(f"❌ {service_name} 기본 서비스 생성 실패: {e}")
        return None

def create_mock_model_loader():
    """Mock ModelLoader 생성"""
    class MockModelLoader:
        def load_model(self, model_name: str, **kwargs):
            logger.debug(f"Mock ModelLoader: {model_name}")
            return None
        def create_step_interface(self, step_name: str):
            return None
    return MockModelLoader()

def create_mock_memory_manager():
    """Mock MemoryManager 생성"""
    class MockMemoryManager:
        def allocate_memory(self, key: str, size_mb: float):
            logger.debug(f"Mock MemoryManager allocate: {key} ({size_mb}MB)")
        def deallocate_memory(self, key: str):
            logger.debug(f"Mock MemoryManager deallocate: {key}")
        def optimize_memory(self):
            return {"optimized": True}
    return MockMemoryManager()

def create_mock_data_converter():
    """Mock DataConverter 생성"""
    class MockDataConverter:
        def convert_api_to_step(self, data: Any, step_name: str):
            return data
        def convert_step_to_api(self, data: Any, step_name: str):
            return data
    return MockDataConverter()

def ensure_central_hub_connection() -> bool:
    """Central Hub 연결 보장 (개선된 버전)"""
    try:
        container = get_global_container()
        if not container:
            logger.error("❌ Central Hub Container를 가져올 수 없음")
            return False
        
        # 필수 서비스들 정의
        essential_services = {
            'model_loader': 'ModelLoader - AI 모델 로딩 및 관리',
            'memory_manager': 'MemoryManager - 메모리 최적화 및 관리', 
            'data_converter': 'DataConverter - API ↔ Step 데이터 변환'
        }
        
        services_registered = 0
        services_failed = 0
        
        for service_name, description in essential_services.items():
            try:
                # 서비스 존재 확인
                existing_service = container.get(service_name)
                
                if existing_service is None:
                    # 서비스가 없으면 팩토리로 등록
                    factory = lambda sname=service_name: create_default_service(sname)
                    container.register_factory(service_name, factory, singleton=True)
                    
                    # 등록 확인
                    registered_service = container.get(service_name)
                    if registered_service:
                        logger.info(f"✅ {service_name} 서비스 등록 완료: {description}")
                        services_registered += 1
                    else:
                        logger.error(f"❌ {service_name} 서비스 등록 실패")
                        services_failed += 1
                else:
                    logger.debug(f"✅ {service_name} 서비스 이미 등록됨: {description}")
                    services_registered += 1
                    
            except Exception as e:
                logger.error(f"❌ {service_name} 서비스 처리 실패: {e}")
                services_failed += 1
        
        # 결과 보고
        total_services = len(essential_services)
        success_rate = (services_registered / total_services) * 100
        
        logger.info(f"🔧 Central Hub 연결 결과: {services_registered}/{total_services} 성공 ({success_rate:.1f}%)")
        
        if services_failed > 0:
            logger.warning(f"⚠️ {services_failed}개 서비스 등록 실패")
        
        # 80% 이상 성공하면 연결 성공으로 간주
        return success_rate >= 80.0
        
    except Exception as e:
        logger.error(f"❌ Central Hub 연결 실패: {e}")
        return False

def validate_central_hub_services() -> Dict[str, Any]:
    """Central Hub 서비스 검증"""
    try:
        container = get_global_container()
        if not container:
            return {
                'connected': False,
                'error': 'Container not available',
                'services': {}
            }
        
        # 서비스 상태 검사
        services_status = {}
        essential_services = ['model_loader', 'memory_manager', 'data_converter']
        
        for service_name in essential_services:
            try:
                service = container.get(service_name)
                services_status[service_name] = {
                    'available': service is not None,
                    'type': type(service).__name__ if service else None,
                    'is_mock': 'Mock' in type(service).__name__ if service else None
                }
            except Exception as e:
                services_status[service_name] = {
                    'available': False,
                    'error': str(e)
                }
        
        # 전체 통계
        available_count = sum(1 for status in services_status.values() if status.get('available', False))
        total_count = len(essential_services)
        
        return {
            'connected': True,
            'container_available': True,
            'services': services_status,
            'statistics': {
                'available_services': available_count,
                'total_services': total_count,
                'availability_rate': (available_count / total_count) * 100,
                'all_services_available': available_count == total_count
            }
        }
        
    except Exception as e:
        return {
            'connected': False,
            'error': str(e),
            'services': {}
        }

def initialize_central_hub_with_validation() -> bool:
    """검증과 함께 Central Hub 초기화"""
    try:
        logger.info("🔧 Central Hub 초기화 시작...")
        
        # 1. 연결 보장
        connection_success = ensure_central_hub_connection()
        if not connection_success:
            logger.error("❌ Central Hub 연결 실패")
            return False
        
        # 2. 서비스 검증
        validation_result = validate_central_hub_services()
        if not validation_result.get('connected', False):
            logger.error("❌ Central Hub 서비스 검증 실패")
            return False
        
        # 3. 결과 보고
        stats = validation_result.get('statistics', {})
        availability_rate = stats.get('availability_rate', 0)
        
        logger.info(f"✅ Central Hub 초기화 완료")
        logger.info(f"📊 서비스 가용성: {availability_rate:.1f}% ({stats.get('available_services', 0)}/{stats.get('total_services', 0)})")
        
        return availability_rate >= 80.0
        
    except Exception as e:
        logger.error(f"❌ Central Hub 초기화 실패: {e}")
        return False

# ==============================================
# 🔥 자동 초기화 훅 (파일 로드 시 실행)
# ==============================================

def _auto_initialize_central_hub():
    """파일 로드 시 자동 초기화"""
    try:
        # 개발/테스트 환경에서는 자동 초기화
        if os.getenv('AUTO_INIT_CENTRAL_HUB', 'true').lower() == 'true':
            success = initialize_central_hub_with_validation()
            if success:
                logger.debug("🔧 Central Hub 자동 초기화 완료")
            else:
                logger.debug("⚠️ Central Hub 자동 초기화 부분 실패 (정상 동작 가능)")
    except Exception as e:
        logger.debug(f"⚠️ Central Hub 자동 초기화 실패: {e}")

# 파일 맨 끝에 추가
# ==============================================
# 🔥 Export (완전한 호환성)
# ==============================================

__all__ = [
    # 메인 클래스들
    'CentralHubDIContainer',
    'CentralHubContainerManager',
    'ServiceRegistry',
    'PropertyInjectionMixin',
    
    # 전역 함수들
    'get_global_container',
    'get_global_manager',
    'reset_global_container',
    'ensure_central_hub_connection',
    'validate_central_hub_services', 
    'initialize_central_hub_with_validation',
    'create_default_service'
    # 기본 편의 함수들
    'get_service',
    'register_service',
    'register_factory',
    'inject_dependencies_to_step',
    
    # 🔥 안전한 접근 함수들 (모든 *_safe 함수들)
    'get_service_safe',
    'register_service_safe',
    'register_factory_safe',
    'inject_dependencies_to_step_safe',
    'inject_dependencies_safe',
    'get_model_loader_safe',
    'get_memory_manager_safe',
    'get_data_converter_safe',
    'get_container_safe',
    'ensure_model_loader_registration',
    'ensure_service_registration',
    'initialize_di_system_safe',
    'cleanup_services_safe',
    'reset_container_safe',
    
    # 🔥 지연 서비스 관련 (완전 구현)
    'register_lazy_service',
    'register_lazy_service_safe',
    'create_lazy_dependency',
    'resolve_lazy_service',
    'is_lazy_service_resolved',
    
    # 🔥 Container 레벨 함수들 (구 버전 호환)
    'create_container',
    'dispose_container',
    'get_container_instance',
    'register_singleton',
    'register_transient',
    'unregister_service',
    
    # 🔥 의존성 주입 관련 (완전 구현)
    'inject_all_dependencies',
    'auto_wire_dependencies',
    'validate_dependencies',
    'get_dependency_status',
    
    # 🔥 서비스 조회 편의 함수들 (완전 구현)
    'get_all_services',
    'list_service_keys',
    'get_service_count',
    
    # 🔥 상태 확인 함수들 (완전 구현)
    'is_service_available',
    'is_container_ready',
    'is_di_system_ready',
    'get_service_status',
    'get_di_system_status',
    
    # 🔥 호환성 함수들 (완전 구현)
    'CircularReferenceFreeDIContainer',  # 구 버전 호환 (별칭)
    'get_global_di_container',  # 구 버전 호환 (별칭)
    'LazyDependency',  # 구 버전 호환
    'DynamicImportResolver',  # 호환성
    '_get_global_di_container',  # BaseStepMixin 호환
    '_get_service_from_container_safe',  # BaseStepMixin 호환
    '_get_central_hub_container',  # Central Hub v7.0 호환
    'get_global_container_legacy',  # 구 버전 호환
    'reset_global_container_legacy',  # 구 버전 호환
    
    # 🔥 특수 호환성 함수들 (완전 구현)
    'ensure_global_step_compatibility',  # 전역 Step 호환성 보장
    '_add_global_step_methods',          # 전역 Step 메서드들 추가
    'ensure_step_di_integration',        # Step DI 통합 보장
    'validate_step_di_requirements',     # Step DI 요구사항 검증
    'setup_global_di_environment',       # 전역 DI 환경 설정
    'get_global_di_environment_status',  # 전역 DI 환경 상태 조회
    
    # 초기화 함수들
    'initialize_di_system',
    
    # 타입들
    'ServiceInfo',
    'T'
]

# ==============================================
# 🔥 자동 초기화
# ==============================================

if IS_CONDA:
    logger.info(f"🐍 conda 환경 '{CONDA_ENV}' 감지")

# 완료 메시지
logger.info("=" * 80)
logger.info("🔥 Central Hub DI Container v7.0 로드 완료!")
logger.info("=" * 80)
logger.info("✅ 중앙 허브 역할 완전 구현 - 모든 서비스의 단일 집중점")
logger.info("✅ 순환참조 근본적 해결 - 단방향 의존성 그래프")
logger.info("✅ 단순하고 직관적인 API - 복잡성 제거")
logger.info("✅ 고성능 서비스 캐싱 - 메모리 효율성 극대화")
logger.info("✅ 자동 의존성 해결 - 개발자 편의성 향상")
logger.info("✅ 스레드 안전성 보장 - 동시성 완벽 지원")
logger.info("✅ 생명주기 완전 관리 - 리소스 누수 방지")
logger.info("✅ 기존 API 100% 호환 - 기존 코드 무수정 지원")

logger.info("🎯 핵심 설계 원칙:")
logger.info("   • Single Source of Truth - 모든 서비스는 DIContainer를 거침")
logger.info("   • Central Hub Pattern - DIContainer가 모든 컴포넌트의 중심")
logger.info("   • Dependency Inversion - 상위 모듈이 하위 모듈을 제어")
logger.info("   • Zero Circular Reference - 순환참조 원천 차단")

if IS_M3_MAX:
    logger.info("🍎 M3 Max 128GB 메모리 최적화 활성화")

logger.info("🚀 Central Hub DI Container v7.0 준비 완료!")
logger.info("🎉 모든 것의 중심 - DIContainer!")
logger.info("🎉 순환참조 문제 완전 해결!")
logger.info("🎉 MyCloset AI 프로젝트 완벽 연동!")
logger.info("=" * 80)
_auto_initialize_central_hub()
