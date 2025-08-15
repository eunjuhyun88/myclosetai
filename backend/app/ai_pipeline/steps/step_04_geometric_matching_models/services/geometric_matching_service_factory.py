#!/usr/bin/env python3
"""
🔥 MyCloset AI - Geometric Matching Service Factory
===================================================

🎯 기하학적 매칭 서비스 팩토리
✅ 다양한 서비스 인스턴스 생성
✅ 설정 기반 서비스 관리
✅ 의존성 주입 및 라이프사이클 관리
✅ M3 Max 최적화
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import yaml

logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """서비스 설정"""
    service_type: str
    service_name: str
    config_path: Optional[str] = None
    enable_caching: bool = True
    enable_monitoring: bool = True
    max_instances: int = 10
    timeout_seconds: int = 300

class BaseService(ABC):
    """기본 서비스 클래스"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.is_initialized = False
        self.creation_time = None
    
    @abstractmethod
    def initialize(self) -> bool:
        """서비스 초기화"""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """서비스 실행"""
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """서비스 정리"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """서비스 상태 반환"""
        return {
            "service_type": self.config.service_type,
            "service_name": self.config.service_name,
            "is_initialized": self.is_initialized,
            "creation_time": self.creation_time,
            "enable_caching": self.config.enable_caching,
            "enable_monitoring": self.config.enable_monitoring
        }

class GeometricMatchingServiceFactory:
    """기하학적 매칭 서비스 팩토리"""
    
    def __init__(self, base_config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Geometric Matching 서비스 팩토리 초기화")
        
        # 기본 설정 경로
        self.base_config_path = Path(base_config_path) if base_config_path else Path("./configs")
        
        # 서비스 레지스트리
        self.service_registry: Dict[str, Type[BaseService]] = {}
        
        # 활성 서비스 인스턴스
        self.active_services: Dict[str, BaseService] = {}
        
        # 서비스 설정 캐시
        self.service_configs: Dict[str, ServiceConfig] = {}
        
        # 기본 서비스 등록
        self._register_default_services()
        
        # 설정 파일 로드
        self._load_service_configs()
        
        self.logger.info("✅ Geometric Matching 서비스 팩토리 초기화 완료")
    
    def _register_default_services(self):
        """기본 서비스 등록"""
        # 여기에 실제 서비스 클래스들을 등록
        # 예: self.register_service("preprocessing", GeometricMatchingPreprocessorService)
        pass
    
    def _load_service_configs(self):
        """서비스 설정 로드"""
        try:
            config_file = self.base_config_path / "geometric_matching_services.yaml"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    configs = yaml.safe_load(f)
                
                for service_name, config_data in configs.items():
                    service_config = ServiceConfig(
                        service_type=config_data.get('type', 'unknown'),
                        service_name=service_name,
                        config_path=config_data.get('config_path'),
                        enable_caching=config_data.get('enable_caching', True),
                        enable_monitoring=config_data.get('enable_monitoring', True),
                        max_instances=config_data.get('max_instances', 10),
                        timeout_seconds=config_data.get('timeout_seconds', 300)
                    )
                    self.service_configs[service_name] = service_config
                
                self.logger.info(f"✅ 서비스 설정 로드 완료: {len(self.service_configs)}개")
            else:
                self.logger.warning(f"서비스 설정 파일을 찾을 수 없습니다: {config_file}")
                
        except Exception as e:
            self.logger.error(f"서비스 설정 로드 실패: {e}")
    
    def register_service(self, service_type: str, service_class: Type[BaseService]):
        """서비스 등록"""
        if service_type in self.service_registry:
            self.logger.warning(f"서비스 타입이 이미 등록되어 있습니다: {service_type}")
            return False
        
        self.service_registry[service_type] = service_class
        self.logger.info(f"✅ 서비스 등록 완료: {service_type}")
        return True
    
    def unregister_service(self, service_type: str):
        """서비스 등록 해제"""
        if service_type not in self.service_registry:
            self.logger.warning(f"등록되지 않은 서비스 타입입니다: {service_type}")
            return False
        
        del self.service_registry[service_type]
        self.logger.info(f"✅ 서비스 등록 해제 완료: {service_type}")
        return True
    
    def create_service(self, service_type: str, service_name: str, 
                      config: Optional[ServiceConfig] = None) -> Optional[BaseService]:
        """서비스 생성"""
        try:
            # 1. 서비스 타입 확인
            if service_type not in self.service_registry:
                self.logger.error(f"등록되지 않은 서비스 타입입니다: {service_type}")
                return None
            
            # 2. 설정 준비
            if config is None:
                config = self.service_configs.get(service_name, ServiceConfig(
                    service_type=service_type,
                    service_name=service_name
                ))
            
            # 3. 인스턴스 수 제한 확인
            if len(self.active_services) >= config.max_instances:
                self.logger.warning(f"최대 서비스 인스턴스 수에 도달했습니다: {config.max_instances}")
                return None
            
            # 4. 서비스 인스턴스 생성
            service_class = self.service_registry[service_type]
            service_instance = service_class(config)
            
            # 5. 서비스 초기화
            if service_instance.initialize():
                # 6. 활성 서비스에 추가
                service_key = f"{service_type}_{service_name}"
                self.active_services[service_key] = service_instance
                
                self.logger.info(f"✅ 서비스 생성 완료: {service_key}")
                return service_instance
            else:
                self.logger.error(f"서비스 초기화 실패: {service_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"서비스 생성 실패: {e}")
            return None
    
    def get_service(self, service_type: str, service_name: str) -> Optional[BaseService]:
        """서비스 조회"""
        service_key = f"{service_type}_{service_name}"
        
        if service_key in self.active_services:
            return self.active_services[service_key]
        
        # 서비스가 없으면 생성 시도
        return self.create_service(service_type, service_name)
    
    def destroy_service(self, service_type: str, service_name: str) -> bool:
        """서비스 제거"""
        service_key = f"{service_type}_{service_name}"
        
        if service_key not in self.active_services:
            self.logger.warning(f"존재하지 않는 서비스입니다: {service_key}")
            return False
        
        try:
            service = self.active_services[service_key]
            
            # 서비스 정리
            if service.cleanup():
                # 활성 서비스에서 제거
                del self.active_services[service_key]
                self.logger.info(f"✅ 서비스 제거 완료: {service_key}")
                return True
            else:
                self.logger.error(f"서비스 정리 실패: {service_key}")
                return False
                
        except Exception as e:
            self.logger.error(f"서비스 제거 실패: {e}")
            return False
    
    def destroy_all_services(self) -> bool:
        """모든 서비스 제거"""
        try:
            service_keys = list(self.active_services.keys())
            success_count = 0
            
            for service_key in service_keys:
                service_type, service_name = service_key.split('_', 1)
                if self.destroy_service(service_type, service_name):
                    success_count += 1
            
            self.logger.info(f"✅ 모든 서비스 제거 완료: {success_count}/{len(service_keys)}")
            return success_count == len(service_keys)
            
        except Exception as e:
            self.logger.error(f"모든 서비스 제거 실패: {e}")
            return False
    
    def get_service_status(self, service_type: str, service_name: str) -> Optional[Dict[str, Any]]:
        """서비스 상태 조회"""
        service = self.get_service(service_type, service_name)
        if service:
            return service.get_status()
        return None
    
    def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """모든 서비스 상태 조회"""
        status_dict = {}
        
        for service_key, service in self.active_services.items():
            status_dict[service_key] = service.get_status()
        
        return status_dict
    
    def list_available_services(self) -> List[str]:
        """사용 가능한 서비스 타입 목록"""
        return list(self.service_registry.keys())
    
    def list_active_services(self) -> List[str]:
        """활성 서비스 목록"""
        return list(self.active_services.keys())
    
    def get_service_config(self, service_name: str) -> Optional[ServiceConfig]:
        """서비스 설정 조회"""
        return self.service_configs.get(service_name)
    
    def update_service_config(self, service_name: str, config: ServiceConfig) -> bool:
        """서비스 설정 업데이트"""
        try:
            self.service_configs[service_name] = config
            
            # 설정 파일에 저장
            config_file = self.base_config_path / "geometric_matching_services.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            configs_dict = {}
            for name, cfg in self.service_configs.items():
                configs_dict[name] = {
                    'type': cfg.service_type,
                    'config_path': cfg.config_path,
                    'enable_caching': cfg.enable_caching,
                    'enable_monitoring': cfg.enable_monitoring,
                    'max_instances': cfg.max_instances,
                    'timeout_seconds': cfg.timeout_seconds
                }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(configs_dict, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"✅ 서비스 설정 업데이트 완료: {service_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"서비스 설정 업데이트 실패: {e}")
            return False
    
    def get_factory_info(self) -> Dict[str, Any]:
        """팩토리 정보 반환"""
        return {
            "base_config_path": str(self.base_config_path),
            "registered_services": len(self.service_registry),
            "active_services": len(self.active_services),
            "service_configs": len(self.service_configs),
            "available_service_types": self.list_available_services(),
            "active_service_names": self.list_active_services()
        }

# 서비스 팩토리 인스턴스 생성
def create_geometric_matching_service_factory(base_config_path: Optional[str] = None) -> GeometricMatchingServiceFactory:
    """Geometric Matching 서비스 팩토리 생성"""
    return GeometricMatchingServiceFactory(base_config_path)

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 서비스 팩토리 생성
    factory = create_geometric_matching_service_factory()
    
    # 팩토리 정보 출력
    factory_info = factory.get_factory_info()
    print("팩토리 정보:")
    for key, value in factory_info.items():
        print(f"  {key}: {value}")
    print()
    
    # 사용 가능한 서비스 타입 출력
    available_services = factory.list_available_services()
    print(f"사용 가능한 서비스 타입: {available_services}")
    
    # 활성 서비스 목록 출력
    active_services = factory.list_active_services()
    print(f"활성 서비스: {active_services}")
    
    # 서비스 설정 목록 출력
    service_configs = factory.service_configs
    print(f"서비스 설정: {len(service_configs)}개")
    for name, config in service_configs.items():
        print(f"  {name}: {config.service_type}")
