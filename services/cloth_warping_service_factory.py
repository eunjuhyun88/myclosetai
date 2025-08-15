#!/usr/bin/env python3
"""
🔥 MyCloset AI - Cloth Warping Service Factory
===============================================

🎯 의류 워핑 서비스 팩토리
✅ 서비스 인스턴스 생성
✅ 설정 기반 서비스 구성
✅ 의존성 주입 관리
✅ M3 Max 최적화
"""

import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """서비스 설정"""
    service_type: str
    config: Dict[str, Any]
    dependencies: List[str] = None
    priority: int = 0

class ClothWarpingServiceFactory:
    """의류 워핑 서비스 팩토리"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Cloth Warping Service Factory 초기화")
        
        # 등록된 서비스들
        self.registered_services = {}
        
        # 서비스 인스턴스 캐시
        self.service_instances = {}
        
        # 기본 서비스 등록
        self._register_default_services()
        
        self.logger.info("✅ Service Factory 초기화 완료")
    
    def _register_default_services(self):
        """기본 서비스 등록"""
        default_services = {
            'memory': {
                'class': 'MemoryService',
                'module': 'services.memory_service',
                'config': {'max_memory_gb': 8, 'cleanup_threshold': 0.8}
            },
            'model_loader': {
                'class': 'ModelLoaderService',
                'module': 'services.model_loader_service',
                'config': {'checkpoint_dir': './checkpoints', 'device': 'mps'}
            },
            'testing': {
                'class': 'TestingService',
                'module': 'services.testing_service',
                'config': {'enable_automated_tests': True, 'test_timeout': 300}
            },
            'validation': {
                'class': 'ValidationService',
                'module': 'services.validation_service',
                'config': {'validation_threshold': 0.8, 'enable_metrics': True}
            }
        }
        
        for service_name, service_info in default_services.items():
            self.register_service(service_name, service_info)
    
    def register_service(self, service_name: str, service_info: Dict[str, Any]):
        """서비스 등록"""
        try:
            self.registered_services[service_name] = service_info
            self.logger.info(f"✅ 서비스 등록 완료: {service_name}")
        except Exception as e:
            self.logger.error(f"❌ 서비스 등록 실패: {service_name} - {e}")
    
    def create_service(self, service_name: str, config: Dict[str, Any] = None) -> Any:
        """서비스 인스턴스 생성"""
        try:
            if service_name not in self.registered_services:
                raise ValueError(f"등록되지 않은 서비스: {service_name}")
            
            # 캐시된 인스턴스 확인
            cache_key = f"{service_name}_{hash(str(config)) if config else 'default'}"
            if cache_key in self.service_instances:
                self.logger.debug(f"캐시된 서비스 인스턴스 사용: {service_name}")
                return self.service_instances[cache_key]
            
            # 서비스 정보 가져오기
            service_info = self.registered_services[service_name]
            
            # 모듈 동적 import
            module_name = service_info['module']
            class_name = service_info['class']
            
            module = __import__(module_name, fromlist=[class_name])
            service_class = getattr(module, class_name)
            
            # 설정 병합
            merged_config = service_info.get('config', {}).copy()
            if config:
                merged_config.update(config)
            
            # 서비스 인스턴스 생성
            service_instance = service_class(**merged_config)
            
            # 캐시에 저장
            self.service_instances[cache_key] = service_instance
            
            self.logger.info(f"✅ 서비스 생성 완료: {service_name}")
            return service_instance
            
        except Exception as e:
            self.logger.error(f"❌ 서비스 생성 실패: {service_name} - {e}")
            raise RuntimeError(f"서비스 생성 중 오류 발생: {e}")
    
    def create_service_chain(self, service_names: List[str], 
                            configs: List[Dict[str, Any]] = None) -> List[Any]:
        """서비스 체인 생성"""
        services = []
        
        try:
            for i, service_name in enumerate(service_names):
                config = configs[i] if configs and i < len(configs) else None
                service = self.create_service(service_name, config)
                services.append(service)
                
                self.logger.debug(f"서비스 체인 구성: {service_name} ({i+1}/{len(service_names)})")
            
            self.logger.info(f"✅ 서비스 체인 생성 완료: {len(services)}개 서비스")
            return services
            
        except Exception as e:
            self.logger.error(f"❌ 서비스 체인 생성 실패: {e}")
            raise RuntimeError(f"서비스 체인 생성 중 오류 발생: {e}")
    
    def get_service_info(self, service_name: str = None) -> Dict[str, Any]:
        """서비스 정보 조회"""
        if service_name:
            return self.registered_services.get(service_name, {})
        else:
            return self.registered_services.copy()
    
    def list_available_services(self) -> List[str]:
        """사용 가능한 서비스 목록 조회"""
        return list(self.registered_services.keys())
    
    def validate_service_dependencies(self, service_name: str) -> bool:
        """서비스 의존성 검증"""
        try:
            if service_name not in self.registered_services:
                return False
            
            service_info = self.registered_services[service_name]
            dependencies = service_info.get('dependencies', [])
            
            for dependency in dependencies:
                if dependency not in self.registered_services:
                    self.logger.warning(f"의존성 누락: {service_name} -> {dependency}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"의존성 검증 실패: {service_name} - {e}")
            return False
    
    def cleanup_service_cache(self, service_name: str = None):
        """서비스 캐시 정리"""
        try:
            if service_name:
                # 특정 서비스 캐시만 정리
                keys_to_remove = [key for key in self.service_instances.keys() 
                                 if key.startswith(service_name)]
                for key in keys_to_remove:
                    del self.service_instances[key]
                self.logger.info(f"✅ 서비스 캐시 정리 완료: {service_name}")
            else:
                # 전체 캐시 정리
                self.service_instances.clear()
                self.logger.info("✅ 전체 서비스 캐시 정리 완료")
                
        except Exception as e:
            self.logger.error(f"❌ 서비스 캐시 정리 실패: {e}")
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """서비스 통계 조회"""
        try:
            stats = {
                'total_registered_services': len(self.registered_services),
                'total_cached_instances': len(self.service_instances),
                'registered_services': list(self.registered_services.keys()),
                'cached_services': list(set([key.split('_')[0] for key in self.service_instances.keys()]))
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ 서비스 통계 조회 실패: {e}")
            return {}

# 서비스 팩토리 인스턴스 생성
def create_cloth_warping_service_factory() -> ClothWarpingServiceFactory:
    """Cloth Warping Service Factory 생성"""
    return ClothWarpingServiceFactory()

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 서비스 팩토리 생성
    factory = create_cloth_warping_service_factory()
    
    # 등록된 서비스 정보 출력
    services = factory.get_service_info()
    print(f"등록된 서비스: {list(services.keys())}")
    
    # 서비스 체인 생성
    service_chain = factory.create_service_chain(['memory', 'model_loader'])
    print(f"생성된 서비스 체인: {len(service_chain)}개")
    
    # 서비스 통계 출력
    stats = factory.get_service_statistics()
    print(f"서비스 통계: {stats}")
