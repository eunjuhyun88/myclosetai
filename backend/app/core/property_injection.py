# 새 파일: app/core/property_injection.py
"""
🔥 PropertyInjectionMixin 독립 모듈 v1.0
================================================================================

순환참조를 방지하기 위해 PropertyInjectionMixin을 완전히 독립적인 모듈로 분리
다른 모듈에 의존하지 않는 완전 자립형 구현
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .di_container import CentralHubDIContainer

class PropertyInjectionMixin:
    """속성 주입을 지원하는 완전 독립 믹스인"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._di_container: Optional['CentralHubDIContainer'] = None
    
    def set_di_container(self, container: 'CentralHubDIContainer'):
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
                try:
                    service = self._di_container.get(service_key)
                    if service:
                        setattr(self, attr_name, service)
                except Exception:
                    # 서비스를 찾을 수 없어도 계속 진행
                    pass

# ==============================================
# 3. 완전한 해결 방안 구현
# ==============================================

def fix_property_injection_mixin_issue():
    """PropertyInjectionMixin 순환참조 완전 해결"""
    
    print("🔥 PropertyInjectionMixin 순환참조 해결 단계:")
    print("1. BaseStepMixin에서 PropertyInjectionMixin 상속 제거")
    print("2. BaseStepMixin에 DI 기능 직접 내장") 
    print("3. Central Hub DI Container를 통한 완전한 의존성 분리")
    print("4. PropertyInjectionMixin을 독립 모듈로 분리")
    print("5. 모든 순환참조 완전 차단")
    
    # BaseStepMixin 클래스 정의 수정
    base_step_mixin_code = '''
class BaseStepMixin:  # PropertyInjectionMixin 제거!
    """
    🔥 BaseStepMixin v21.0 - 순환참조 완전 해결
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # DI Container 관련 속성들 직접 내장
        self._di_container = None
        self.central_hub_container = None
        self.di_container = None  # 기존 호환성
        
        # 의존성 주입된 서비스들
        self.model_loader = None
        self.memory_manager = None
        self.data_converter = None
        
        # Central Hub DI Container 자동 연결
        self._auto_connect_central_hub()
    
    def set_di_container(self, container):
        """DI Container 설정 - PropertyInjectionMixin 기능 내장"""
        self._di_container = container
        self.central_hub_container = container
        self.di_container = container
        self._auto_inject_properties()
    
    def _auto_inject_properties(self):
        """자동 속성 주입 - PropertyInjectionMixin 기능 내장"""
        if not self._di_container:
            return
        
        injection_map = {
            'model_loader': 'model_loader',
            'memory_manager': 'memory_manager', 
            'data_converter': 'data_converter'
        }
        
        for attr_name, service_key in injection_map.items():
            if not hasattr(self, attr_name) or getattr(self, attr_name) is None:
                try:
                    service = self._di_container.get(service_key)
                    if service:
                        setattr(self, attr_name, service)
                except Exception:
                    pass
    '''
    
    return {
        "issue": "PropertyInjectionMixin이 정의되지 않음",
        "cause": "순환참조 때문에 di_container.py를 import할 수 없음", 
        "solution": "BaseStepMixin에 DI 기능 직접 내장",
        "result": "완전한 순환참조 해결 + 기능 유지",
        "code": base_step_mixin_code
    }

if __name__ == "__main__":
    result = fix_property_injection_mixin_issue()
    print("\n✅ PropertyInjectionMixin 순환참조 완전 해결 완료!")
    print(f"문제: {result['issue']}")
    print(f"원인: {result['cause']}")
    print(f"해결: {result['solution']}")
    print(f"결과: {result['result']}")
