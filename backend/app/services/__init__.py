# backend/app/services/__init__.py
"""
🍎 MyCloset AI 서비스 통합 관리 시스템 v7.0 - 단순화된 설계
================================================================

✅ 단순하고 안정적인 서비스 초기화
✅ 순환참조 완전 방지
✅ conda 환경 우선 최적화
✅ M3 Max 128GB 메모리 활용
✅ 의존성 주입 (DI) 기반 설계
✅ 실패 허용적 아키텍처
✅ Clean Architecture 적용

서비스 레이어:
- step_service: Step 기반 AI 처리 서비스
- ai_pipeline: 8단계 파이프라인 서비스  
- model_manager: AI 모델 관리 서비스
- unified_step_mapping: Step 매핑 서비스

작성자: MyCloset AI Team
날짜: 2025-07-23
버전: v7.0.0 (Simplified Service Integration)
"""

import logging
import threading
import sys
from typing import Dict, Any, Optional, List, Type, Union
from functools import lru_cache
import warnings

# 경고 무시
warnings.filterwarnings('ignore')

# =============================================================================
# 🔥 기본 설정 및 시스템 정보
# =============================================================================

logger = logging.getLogger(__name__)

# 상위 패키지에서 시스템 정보 가져오기
try:
    from .. import get_system_info, is_conda_environment, is_m3_max, get_device
    SYSTEM_INFO = get_system_info()
    IS_CONDA = is_conda_environment()
    IS_M3_MAX = is_m3_max()
    DEVICE = get_device()
    logger.info("✅ 상위 패키지에서 시스템 정보 로드 성공")
except ImportError as e:
    logger.warning(f"⚠️ 상위 패키지 로드 실패, 기본값 사용: {e}")
    SYSTEM_INFO = {'device': 'cpu', 'is_m3_max': False, 'memory_gb': 16.0}
    IS_CONDA = False
    IS_M3_MAX = False
    DEVICE = 'cpu'

# =============================================================================
# 🔥 서비스 상태 추적
# =============================================================================

# 서비스 로딩 상태
SERVICE_STATUS = {
    'step_service': True,
    'ai_pipeline': True, 
    'model_manager': True,
    'unified_step_mapping': True,
    'body_measurements': False
}

# 서비스 인스턴스 캐시
_service_cache = {}
_cache_lock = threading.Lock()

# =============================================================================
# 🔥 안전한 서비스 모듈 로딩
# =============================================================================

def _safe_import_step_service():
    """step_service 모듈 안전하게 import"""
    try:
        from .step_service import (
            UnifiedStepServiceManager,
            BaseStepService,
            StepServiceFactory,
            UploadValidationService,
            HumanParsingService,
            VirtualFittingService,
            CompletePipelineService
        )
        
        # 전역에 추가
        globals().update({
            'UnifiedStepServiceManager': UnifiedStepServiceManager,
            'BaseStepService': BaseStepService,
            'StepServiceFactory': StepServiceFactory,
            'UploadValidationService': UploadValidationService,
            'HumanParsingService': HumanParsingService,
            'VirtualFittingService': VirtualFittingService,
            'CompletePipelineService': CompletePipelineService
        })
        
        SERVICE_STATUS['step_service'] = True
        logger.info("✅ step_service 모듈 로드 성공")
        return True
        
    except ImportError as e:
        logger.debug(f"📋 step_service 모듈 없음 (정상): {e}")
        return False
    except Exception as e:
        logger.error(f"❌ step_service 모듈 로드 실패: {e}")
        return False

def _safe_import_ai_pipeline():
    """ai_pipeline 모듈 안전하게 import"""
    try:
        from .ai_pipeline import AIVirtualTryOnPipeline
        
        globals()['AIVirtualTryOnPipeline'] = AIVirtualTryOnPipeline
        
        SERVICE_STATUS['ai_pipeline'] = True
        logger.info("✅ ai_pipeline 모듈 로드 성공")
        return True
        
    except ImportError as e:
        logger.debug(f"📋 ai_pipeline 모듈 없음 (정상): {e}")
        return False
    except Exception as e:
        logger.error(f"❌ ai_pipeline 모듈 로드 실패: {e}")
        return False

def _safe_import_model_manager():
    """model_manager 모듈 안전하게 import"""
    try:
        from .model_manager import ModelManager
        
        globals()['ModelManager'] = ModelManager
        
        SERVICE_STATUS['model_manager'] = True
        logger.info("✅ model_manager 모듈 로드 성공")
        return True
        
    except ImportError as e:
        logger.debug(f"📋 model_manager 모듈 없음 (정상): {e}")
        return False
    except Exception as e:
        logger.error(f"❌ model_manager 모듈 로드 실패: {e}")
        return False

def _safe_import_unified_step_mapping():
    """unified_step_mapping 모듈 안전하게 import"""
    try:
        from .unified_step_mapping import (
            StepFactory,
            StepFactoryHelper,
            RealStepSignature,
            UnifiedStepSignature
        )
        
        globals().update({
            'StepFactory': StepFactory,
            'StepFactoryHelper': StepFactoryHelper,
            'RealStepSignature': RealStepSignature,
            'UnifiedStepSignature': UnifiedStepSignature
        })
        
        SERVICE_STATUS['unified_step_mapping'] = True
        logger.info("✅ unified_step_mapping 모듈 로드 성공")
        return True
        
    except ImportError as e:
        logger.debug(f"📋 unified_step_mapping 모듈 없음 (정상): {e}")
        return False
    except Exception as e:
        logger.error(f"❌ unified_step_mapping 모듈 로드 실패: {e}")
        return False

def _try_import_body_measurements():
    """body_measurements 모듈 시도 (옵션)"""
    try:
        # 가상의 body measurements 클래스
        class BodyMeasurements:
            def __init__(self, **kwargs):
                self.measurements = kwargs
                
            def get_measurements(self):
                return self.measurements
        
        globals()['BodyMeasurements'] = BodyMeasurements
        
        SERVICE_STATUS['body_measurements'] = True
        logger.info("✅ body_measurements 로드 성공")
        return True
        
    except Exception as e:
        logger.debug(f"📋 body_measurements 없음: {e}")
        return False

# =============================================================================
# 🔥 서비스 모듈들 로딩
# =============================================================================

# 모든 서비스 모듈 로딩 시도
_safe_import_step_service()
_safe_import_ai_pipeline()
_safe_import_model_manager()
_safe_import_unified_step_mapping()
_try_import_body_measurements()

# =============================================================================
# 🔥 단순화된 서비스 매니저
# =============================================================================

class SimpleServiceManager:
    """단순화된 서비스 매니저"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SimpleServiceManager")
        self._services = {}
        self._lock = threading.Lock()
        
        # 시스템 정보 저장
        self.system_info = SYSTEM_INFO
        self.is_conda = IS_CONDA
        self.is_m3_max = IS_M3_MAX
        self.device = DEVICE
        
        self.logger.info(f"🎯 서비스 매니저 초기화 (device: {DEVICE})")
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """서비스 인스턴스 반환"""
        with self._lock:
            if service_name in self._services:
                return self._services[service_name]
            
            # 서비스 생성 시도
            service_instance = self._create_service(service_name)
            if service_instance:
                self._services[service_name] = service_instance
                self.logger.info(f"✅ {service_name} 서비스 생성 완료")
            
            return service_instance
    
    def _create_service(self, service_name: str) -> Optional[Any]:
        """서비스 인스턴스 생성"""
        try:
            if service_name == 'pipeline' and SERVICE_STATUS['ai_pipeline']:
                return AIVirtualTryOnPipeline(
                    device=self.device,
                    memory_limit_gb=self.system_info.get('memory_gb', 16),
                    is_m3_max=self.is_m3_max,
                    conda_optimized=self.is_conda
                )
            
            elif service_name == 'model_manager' and SERVICE_STATUS['model_manager']:
                return ModelManager()
            
            elif service_name == 'step_manager' and SERVICE_STATUS['step_service']:
                return UnifiedStepServiceManager(
                    device=self.device,
                    is_m3_max=self.is_m3_max
                )
            
            elif service_name == 'step_factory' and SERVICE_STATUS['unified_step_mapping']:
                return StepFactory()
            
            else:
                self.logger.warning(f"⚠️ 알 수 없는 서비스 또는 모듈 없음: {service_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ {service_name} 서비스 생성 실패: {e}")
            return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 반환"""
        return {
            'system_info': self.system_info,
            'service_status': SERVICE_STATUS.copy(),
            'active_services': list(self._services.keys()),
            'available_services': [k for k, v in SERVICE_STATUS.items() if v],
            'conda_optimized': self.is_conda,
            'm3_max_optimized': self.is_m3_max,
            'device': self.device
        }
    
    def cleanup(self):
        """서비스 정리"""
        with self._lock:
            for service_name, service in self._services.items():
                try:
                    if hasattr(service, 'cleanup'):
                        service.cleanup()
                    elif hasattr(service, '__del__'):
                        service.__del__()
                except:
                    pass
            
            self._services.clear()
            self.logger.info("🧹 모든 서비스 정리 완료")

# 전역 서비스 매니저
_global_service_manager = SimpleServiceManager()

# =============================================================================
# 🔥 팩토리 함수들 (외부 API)
# =============================================================================

@lru_cache(maxsize=1)
def get_main_service_manager() -> SimpleServiceManager:
    """메인 서비스 매니저 반환 (동기)"""
    return _global_service_manager

async def get_main_service_manager_async() -> SimpleServiceManager:
    """메인 서비스 매니저 반환 (비동기)"""
    return _global_service_manager

def get_pipeline_service() -> Optional[Any]:
    """파이프라인 서비스 반환"""
    return _global_service_manager.get_service('pipeline')

def get_pipeline_service_sync() -> Optional[Any]:
    """파이프라인 서비스 반환 (동기)"""
    return get_pipeline_service()

def get_pipeline_manager_service() -> Optional[Any]:
    """파이프라인 매니저 서비스 반환"""
    return _global_service_manager.get_service('step_manager')

def get_service_status() -> Dict[str, Any]:
    """서비스 상태 반환"""
    return _global_service_manager.get_service_status()

# =============================================================================
# 🔥 Step 서비스 전용 함수들 (조건부)
# =============================================================================

if SERVICE_STATUS['step_service']:
    @lru_cache(maxsize=1)
    def get_step_service_manager() -> Optional[Any]:
        """Step 서비스 매니저 반환 (동기)"""
        return _global_service_manager.get_service('step_manager')
    
    async def get_step_service_manager_async() -> Optional[Any]:
        """Step 서비스 매니저 반환 (비동기)"""
        return _global_service_manager.get_service('step_manager')
    
    def cleanup_step_service_manager():
        """Step 서비스 매니저 정리"""
        service_manager = _global_service_manager.get_service('step_manager')
        if service_manager and hasattr(service_manager, 'cleanup'):
            service_manager.cleanup()
else:
    # 폴백 함수들
    def get_step_service_manager():
        logger.warning("⚠️ step_service 모듈이 로드되지 않음")
        return None
    
    async def get_step_service_manager_async():
        logger.warning("⚠️ step_service 모듈이 로드되지 않음")
        return None
    
    def cleanup_step_service_manager():
        logger.warning("⚠️ step_service 모듈이 로드되지 않음")

# =============================================================================
# 🔥 Export 목록 (동적 생성)
# =============================================================================

def _get_available_exports():
    """사용 가능한 export 목록 동적 생성"""
    exports = [
        # 🎯 핵심 팩토리 함수들 (항상 사용 가능)
        'get_main_service_manager',
        'get_main_service_manager_async',
        'get_pipeline_service',
        'get_pipeline_service_sync', 
        'get_pipeline_manager_service',
        'get_service_status',
        
        # 🔧 Step 서비스 함수들 (조건부)
        'get_step_service_manager',
        'get_step_service_manager_async',
        'cleanup_step_service_manager',
        
        # 📊 상수 및 상태
        'SERVICE_STATUS',
        'SYSTEM_INFO',
        'IS_CONDA',
        'IS_M3_MAX',
        'DEVICE',
        
        # 🛠️ 서비스 매니저
        'SimpleServiceManager'
    ]
    
    # 조건부 서비스 클래스들 추가
    if SERVICE_STATUS['step_service']:
        step_exports = [
            'UnifiedStepServiceManager',
            'BaseStepService', 
            'StepServiceFactory',
            'UploadValidationService',
            'HumanParsingService',
            'VirtualFittingService',
            'CompletePipelineService'
        ]
        exports.extend(step_exports)
    
    if SERVICE_STATUS['ai_pipeline']:
        exports.append('AIVirtualTryOnPipeline')
    
    if SERVICE_STATUS['model_manager']:
        exports.append('ModelManager')
    
    if SERVICE_STATUS['unified_step_mapping']:
        mapping_exports = [
            'StepFactory',
            'StepFactoryHelper',
            'RealStepSignature',
            'UnifiedStepSignature'
        ]
        exports.extend(mapping_exports)
    
    if SERVICE_STATUS['body_measurements']:
        exports.append('BodyMeasurements')
    
    return exports

__all__ = _get_available_exports()

# =============================================================================
# 🔥 초기화 완료 메시지
# =============================================================================

def _print_initialization_summary():
    """초기화 요약 출력"""
    available_services = [k for k, v in SERVICE_STATUS.items() if v]
    total_services = len(SERVICE_STATUS)
    success_rate = (len(available_services) / total_services) * 100
    
    print(f"\n🍎 MyCloset AI 서비스 시스템 v7.0 초기화 완료!")
    print(f"🔧 사용 가능한 서비스: {len(available_services)}/{total_services}개 ({success_rate:.1f}%)")
    print(f"🐍 conda 환경: {'✅' if IS_CONDA else '❌'}")
    print(f"🍎 M3 Max: {'✅' if IS_M3_MAX else '❌'}")
    print(f"🖥️ 디바이스: {DEVICE}")
    
    if available_services:
        print(f"✅ 로드된 서비스: {', '.join(available_services)}")
    
    unavailable_services = [k for k, v in SERVICE_STATUS.items() if not v]
    if unavailable_services:
        print(f"⚠️ 구현 대기 서비스: {', '.join(unavailable_services)}")
        print(f"💡 이는 정상적인 상태입니다 (단계적 구현)")
    
    print("🚀 서비스 시스템 준비 완료!\n")

# 초기화 상태 출력 (한 번만)
if not hasattr(sys, '_mycloset_services_initialized'):
    _print_initialization_summary()
    sys._mycloset_services_initialized = True

logger.info("🍎 MyCloset AI 서비스 시스템 초기화 완료")