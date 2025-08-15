#!/usr/bin/env python3
"""
🔥 MyCloset AI - Cloth Warping Service Factory
==============================================

🎯 의류 워핑 서비스 팩토리
✅ 서비스 인스턴스 생성
✅ 설정 기반 서비스 선택
✅ 의존성 주입
✅ M3 Max 최적화
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Type
from dataclasses import dataclass
import torch

logger = logging.getLogger(__name__)

@dataclass
class ServiceFactoryConfig:
    """서비스 팩토리 설정"""
    enable_advanced_post_processing: bool = True
    enable_high_resolution_processing: bool = True
    enable_preprocessing: bool = True
    enable_quality_enhancement: bool = True
    enable_special_case_processing: bool = True
    use_mps: bool = True
    memory_efficient: bool = True

class ClothWarpingServiceFactory:
    """의류 워핑 서비스 팩토리"""
    
    def __init__(self, config: ServiceFactoryConfig = None):
        self.config = config or ServiceFactoryConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Cloth Warping 서비스 팩토리 초기화")
        
        # 서비스 레지스트리
        self.service_registry = {}
        self._register_services()
        
        self.logger.info("✅ Cloth Warping 서비스 팩토리 초기화 완료")
    
    def _register_services(self):
        """서비스들을 레지스트리에 등록합니다."""
        try:
            # 프로세서 서비스들
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'processors'))
            from advanced_post_processor import ClothWarpingAdvancedPostProcessor
            from high_resolution_processor import ClothWarpingHighResolutionProcessor
            from preprocessing import ClothWarpingPreprocessor
            from quality_enhancer import ClothWarpingQualityEnhancer
            from special_case_processor import ClothWarpingSpecialCaseProcessor
            
            self.service_registry.update({
                'advanced_post_processor': ClothWarpingAdvancedPostProcessor,
                'high_resolution_processor': ClothWarpingHighResolutionProcessor,
                'preprocessor': ClothWarpingPreprocessor,
                'quality_enhancer': ClothWarpingQualityEnhancer,
                'special_case_processor': ClothWarpingSpecialCaseProcessor
            })
            
            self.logger.info("프로세서 서비스 등록 완료")
            
        except ImportError as e:
            self.logger.error(f"프로세서 서비스 등록 실패: {e}")
    
    def create_advanced_post_processor(self, **kwargs) -> Any:
        """고급 후처리기를 생성합니다."""
        if not self.config.enable_advanced_post_processing:
            raise ValueError("고급 후처리가 비활성화되어 있습니다.")
        
        try:
            processor_class = self.service_registry.get('advanced_post_processor')
            if processor_class:
                return processor_class(**kwargs)
            else:
                raise ValueError("고급 후처리기 클래스를 찾을 수 없습니다.")
        except Exception as e:
            self.logger.error(f"고급 후처리기 생성 실패: {e}")
            raise
    
    def create_high_resolution_processor(self, **kwargs) -> Any:
        """고해상도 처리기를 생성합니다."""
        if not self.config.enable_high_resolution_processing:
            raise ValueError("고해상도 처리가 비활성화되어 있습니다.")
        
        try:
            processor_class = self.service_registry.get('high_resolution_processor')
            if processor_class:
                return processor_class(**kwargs)
            else:
                raise ValueError("고해상도 처리기 클래스를 찾을 수 없습니다.")
        except Exception as e:
            self.logger.error(f"고해상도 처리기 생성 실패: {e}")
            raise
    
    def create_preprocessor(self, **kwargs) -> Any:
        """전처리기를 생성합니다."""
        if not self.config.enable_preprocessing:
            raise ValueError("전처리가 비활성화되어 있습니다.")
        
        try:
            processor_class = self.service_registry.get('preprocessor')
            if processor_class:
                return processor_class(**kwargs)
            else:
                raise ValueError("전처리기 클래스를 찾을 수 없습니다.")
        except Exception as e:
            self.logger.error(f"전처리기 생성 실패: {e}")
            raise
    
    def create_quality_enhancer(self, **kwargs) -> Any:
        """품질 향상기를 생성합니다."""
        if not self.config.enable_quality_enhancement:
            raise ValueError("품질 향상이 비활성화되어 있습니다.")
        
        try:
            processor_class = self.service_registry.get('quality_enhancer')
            if processor_class:
                return processor_class(**kwargs)
            else:
                raise ValueError("품질 향상기 클래스를 찾을 수 없습니다.")
        except Exception as e:
            self.logger.error(f"품질 향상기 생성 실패: {e}")
            raise
    
    def create_special_case_processor(self, **kwargs) -> Any:
        """특수 케이스 처리기를 생성합니다."""
        if not self.config.enable_special_case_processing:
            raise ValueError("특수 케이스 처리가 비활성화되어 있습니다.")
        
        try:
            processor_class = self.service_registry.get('special_case_processor')
            if processor_class:
                return processor_class(**kwargs)
            else:
                raise ValueError("특수 케이스 처리기 클래스를 찾을 수 없습니다.")
        except Exception as e:
            self.logger.error(f"특수 케이스 처리기 생성 실패: {e}")
            raise
    
    def create_all_processors(self, **kwargs) -> Dict[str, Any]:
        """모든 프로세서를 생성합니다."""
        processors = {}
        
        try:
            if self.config.enable_advanced_post_processing:
                processors['advanced_post_processor'] = self.create_advanced_post_processor(**kwargs)
            
            if self.config.enable_high_resolution_processing:
                processors['high_resolution_processor'] = self.create_high_resolution_processor(**kwargs)
            
            if self.config.enable_preprocessing:
                processors['preprocessor'] = self.create_preprocessor(**kwargs)
            
            if self.config.enable_quality_enhancement:
                processors['quality_enhancer'] = self.create_quality_enhancer(**kwargs)
            
            if self.config.enable_special_case_processing:
                processors['special_case_processor'] = self.create_special_case_processor(**kwargs)
            
            self.logger.info(f"총 {len(processors)}개의 프로세서 생성 완료")
            return processors
            
        except Exception as e:
            self.logger.error(f"프로세서 일괄 생성 실패: {e}")
            raise
    
    def get_available_services(self) -> List[str]:
        """사용 가능한 서비스 목록을 반환합니다."""
        return list(self.service_registry.keys())
    
    def get_service_config(self) -> Dict[str, Any]:
        """서비스 팩토리 설정을 반환합니다."""
        return {
            'enable_advanced_post_processing': self.config.enable_advanced_post_processing,
            'enable_high_resolution_processing': self.config.enable_high_resolution_processing,
            'enable_preprocessing': self.config.enable_preprocessing,
            'enable_quality_enhancement': self.config.enable_quality_enhancement,
            'enable_special_case_processing': self.config.enable_special_case_processing,
            'use_mps': self.config.use_mps,
            'memory_efficient': self.config.memory_efficient,
            'available_services': self.get_available_services()
        }
    
    def validate_service_creation(self) -> Dict[str, bool]:
        """서비스 생성 가능성을 검증합니다."""
        validation_results = {}
        
        for service_name in self.service_registry.keys():
            try:
                if service_name == 'advanced_post_processor' and self.config.enable_advanced_post_processing:
                    self.create_advanced_post_processor()
                    validation_results[service_name] = True
                elif service_name == 'high_resolution_processor' and self.config.enable_high_resolution_processing:
                    self.create_high_resolution_processor()
                    validation_results[service_name] = True
                elif service_name == 'preprocessor' and self.config.enable_preprocessing:
                    self.create_preprocessor()
                    validation_results[service_name] = True
                elif service_name == 'quality_enhancer' and self.config.enable_quality_enhancement:
                    self.create_quality_enhancer()
                    validation_results[service_name] = True
                elif service_name == 'special_case_processor' and self.config.enable_special_case_processing:
                    self.create_special_case_processor()
                    validation_results[service_name] = True
                else:
                    validation_results[service_name] = False
            except Exception as e:
                self.logger.warning(f"{service_name} 검증 실패: {e}")
                validation_results[service_name] = False
        
        return validation_results

# 사용 예시
if __name__ == "__main__":
    # 설정
    config = ServiceFactoryConfig(
        enable_advanced_post_processing=True,
        enable_high_resolution_processing=True,
        enable_preprocessing=True,
        enable_quality_enhancement=True,
        enable_special_case_processing=True,
        use_mps=True,
        memory_efficient=True
    )
    
    # 서비스 팩토리 초기화
    factory = ClothWarpingServiceFactory(config)
    
    # 사용 가능한 서비스 확인
    available_services = factory.get_available_services()
    print(f"사용 가능한 서비스: {available_services}")
    
    # 서비스 생성 검증
    validation_results = factory.validate_service_creation()
    print(f"서비스 생성 검증 결과: {validation_results}")
    
    # 모든 프로세서 생성
    try:
        all_processors = factory.create_all_processors()
        print(f"생성된 프로세서 수: {len(all_processors)}")
        
        # 설정 정보 출력
        service_config = factory.get_service_config()
        print(f"서비스 설정: {service_config}")
        
    except Exception as e:
        print(f"프로세서 생성 실패: {e}")
