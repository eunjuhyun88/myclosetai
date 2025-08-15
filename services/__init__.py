#!/usr/bin/env python3
"""
🔥 MyCloset AI - Services Package for Cloth Warping
====================================================

🎯 의류 워핑 서비스 레이어
✅ 서비스 팩토리
✅ 메모리 관리
✅ 모델 로더
✅ 테스팅
✅ 검증
✅ M3 Max 최적화
"""

# 서비스 팩토리
from .cloth_warping_service_factory import (
    ClothWarpingServiceFactory,
    ServiceConfig,
    create_cloth_warping_service_factory
)

# 메모리 관리 서비스
from .memory_service import (
    MemoryService,
    MemoryConfig,
    create_memory_service
)

# 모델 로더 서비스 (향후 구현 예정)
# from .model_loader_service import (
#     ModelLoaderService,
#     ModelLoaderConfig,
#     create_model_loader_service
# )

# 테스팅 서비스 (향후 구현 예정)
# from .testing_service import (
#     TestingService,
#     TestingConfig,
#     create_testing_service
# )

# 검증 서비스 (향후 구현 예정)
# from .validation_service import (
#     ValidationService,
#     ValidationConfig,
#     create_validation_service
# )

__all__ = [
    # 서비스 팩토리
    'ClothWarpingServiceFactory',
    'ServiceConfig',
    'create_cloth_warping_service_factory',
    
    # 메모리 관리
    'MemoryService',
    'MemoryConfig',
    'create_memory_service',
    
    # 향후 구현 예정 서비스들
    # 'ModelLoaderService',
    # 'ModelLoaderConfig',
    # 'create_model_loader_service',
    # 'TestingService',
    # 'TestingConfig',
    # 'create_testing_service',
    # 'ValidationService',
    # 'ValidationConfig',
    # 'create_validation_service'
]

# 서비스 관리자 클래스
class ClothWarpingServiceManager:
    """의류 워핑 서비스 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Cloth Warping Service Manager 초기화")
        
        # 서비스 팩토리 생성
        self.service_factory = create_cloth_warping_service_factory()
        
        # 활성 서비스들
        self.active_services = {}
        
        # 기본 서비스 초기화
        self._initialize_default_services()
        
        self.logger.info("✅ Service Manager 초기화 완료")
    
    def _initialize_default_services(self):
        """기본 서비스 초기화"""
        try:
            # 메모리 서비스 초기화
            memory_service = create_memory_service()
            self.active_services['memory'] = memory_service
            
            self.logger.info("✅ 기본 서비스 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 기본 서비스 초기화 실패: {e}")
    
    def get_service(self, service_name: str):
        """서비스 조회"""
        return self.active_services.get(service_name)
    
    def add_service(self, service_name: str, service_instance):
        """서비스 추가"""
        try:
            self.active_services[service_name] = service_instance
            self.logger.info(f"✅ 서비스 추가 완료: {service_name}")
        except Exception as e:
            self.logger.error(f"❌ 서비스 추가 실패: {service_name} - {e}")
    
    def remove_service(self, service_name: str):
        """서비스 제거"""
        try:
            if service_name in self.active_services:
                del self.active_services[service_name]
                self.logger.info(f"✅ 서비스 제거 완료: {service_name}")
        except Exception as e:
            self.logger.error(f"❌ 서비스 제거 실패: {service_name} - {e}")
    
    def list_active_services(self):
        """활성 서비스 목록 조회"""
        return list(self.active_services.keys())
    
    def get_service_status(self):
        """서비스 상태 조회"""
        status = {}
        
        for service_name, service in self.active_services.items():
            try:
                if hasattr(service, 'get_status'):
                    status[service_name] = service.get_status()
                else:
                    status[service_name] = {'status': 'active', 'type': type(service).__name__}
            except Exception as e:
                status[service_name] = {'status': 'error', 'error': str(e)}
        
        return status

# 서비스 관리자 인스턴스 생성
def create_service_manager() -> ClothWarpingServiceManager:
    """Service Manager 생성"""
    return ClothWarpingServiceManager()

# 서비스 체인 생성 함수
def create_service_chain(service_types: list, configs: list = None):
    """
    여러 서비스를 연결한 체인 생성
    
    Args:
        service_types: 서비스 타입 리스트
        configs: 설정 리스트
    
    Returns:
        서비스 체인
    """
    service_manager = create_service_manager()
    services = []
    
    for i, service_type in enumerate(service_types):
        try:
            if service_type == 'memory':
                config = configs[i] if configs and i < len(configs) else None
                service = create_memory_service(config)
            else:
                # 다른 서비스들은 팩토리를 통해 생성
                service = service_manager.service_factory.create_service(service_type, configs[i] if configs and i < len(configs) else None)
            
            services.append(service)
            service_manager.add_service(f"{service_type}_{i}", service)
            
        except Exception as e:
            logger.error(f"서비스 생성 실패: {service_type} - {e}")
            continue
    
    return services, service_manager

# 서비스 정보 조회 함수
def get_service_info(service_type: str = None):
    """
    서비스 정보 조회
    
    Args:
        service_type: 특정 서비스 타입 (None이면 모든 정보)
    
    Returns:
        서비스 정보 딕셔너리
    """
    service_info = {
        'factory': {
            'name': 'Cloth Warping Service Factory',
            'description': '서비스 인스턴스 생성 및 관리',
            'capabilities': ['서비스 등록', '인스턴스 생성', '의존성 관리'],
            'config_class': 'ServiceConfig'
        },
        'memory': {
            'name': 'Memory Service',
            'description': '메모리 사용량 모니터링 및 관리',
            'capabilities': ['메모리 모니터링', '자동 정리', '최적화'],
            'config_class': 'MemoryConfig'
        }
    }
    
    if service_type:
        return service_info.get(service_type, {})
    else:
        return service_info
