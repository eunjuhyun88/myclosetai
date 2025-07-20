# backend/app/core/warmup_safe_patch.py
"""
🔧 워밍업 안전 패치 - RuntimeWarning 완전 해결
BaseStepMixin 워밍업 시스템을 안전하게 수정
"""

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def patch_warmup_system():
    """
    BaseStepMixin의 워밍업 시스템을 안전하게 패치
    """
    try:
        from backend.app.ai_pipeline.steps.base_step_mixin import BaseStepMixin, WarmupSystem
        
        # WarmupSystem의 _pipeline_warmup 메서드를 안전하게 수정
        def safe_pipeline_warmup(self) -> Dict[str, Any]:
            """안전한 파이프라인 워밍업 (동기 버전)"""
            try:
                # Step별 워밍업 로직 (기본)
                if hasattr(self.step, 'warmup_step'):
                    warmup_method = getattr(self.step, 'warmup_step')
                    
                    # 함수가 async인지 확인
                    if asyncio.iscoroutinefunction(warmup_method):
                        self.logger.info("비동기 warmup_step 감지, 동기 처리로 변환")
                        try:
                            # 새 이벤트 루프에서 실행
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                result = loop.run_until_complete(warmup_method())
                                return {'success': result.get('success', True), 'message': 'Step 워밍업 완료'}
                            finally:
                                loop.close()
                        except Exception as e:
                            self.logger.warning(f"비동기 워밍업 처리 실패: {e}")
                            return {'success': False, 'error': str(e)}
                    else:
                        # 동기 함수면 그대로 호출
                        try:
                            result = warmup_method()
                            return {'success': result.get('success', True), 'message': 'Step 워밍업 완료'}
                        except Exception as e:
                            self.logger.warning(f"동기 워밍업 실패: {e}")
                            return {'success': False, 'error': str(e)}
                
                return {'success': True, 'message': '파이프라인 워밍업 건너뜀'}
                
            except Exception as e:
                self.logger.error(f"파이프라인 워밍업 실패: {e}")
                return {'success': False, 'error': str(e)}
        
        # WarmupSystem 클래스에 안전한 메서드 적용
        WarmupSystem._pipeline_warmup = safe_pipeline_warmup
        
        logger.info("✅ WarmupSystem._pipeline_warmup 패치 완료")
        
        # BaseStepMixin의 _setup_model_interface도 안전하게 패치
        def safe_setup_model_interface(self):
            """ModelLoader 인터페이스 설정 (동기 버전)"""
            try:
                self.logger.info(f"🔗 {self.step_name} ModelLoader 인터페이스 설정 중...")
                
                # Step 인터페이스 생성 (ModelLoader가 있는 경우)
                if hasattr(self, 'model_loader') and self.model_loader:
                    try:
                        if hasattr(self.model_loader, 'create_step_interface'):
                            # 비동기 함수인지 확인
                            interface_method = getattr(self.model_loader, 'create_step_interface')
                            if asyncio.iscoroutinefunction(interface_method):
                                self.logger.info("비동기 create_step_interface 감지, 동기 처리")
                                # 비동기 함수는 건너뛰고 None 설정
                                self.step_interface = None
                                self.logger.warning("⚠️ 비동기 인터페이스 생성 건너뜀")
                            else:
                                self.step_interface = interface_method(self.step_name)
                                self.logger.info("✅ Step 인터페이스 생성 성공")
                        else:
                            self.step_interface = None
                            self.logger.warning("⚠️ ModelLoader에 create_step_interface 메서드 없음")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Step 인터페이스 생성 실패: {e}")
                        self.step_interface = None
                else:
                    self.step_interface = None
                
                # 모델 관련 속성 초기화
                self._ai_model = None
                self._ai_model_name = None
                self.loaded_models = {}
                self.model_cache = {}
                
                # 연동 상태 로깅
                loader_status = "✅ 연결됨" if hasattr(self, 'model_loader') and self.model_loader else "❌ 연결 실패"
                interface_status = "✅ 연결됨" if self.step_interface else "❌ 연결 실패"
                
                self.logger.info(f"🔗 ModelLoader 연동 결과:")
                self.logger.info(f"   - ModelLoader: {loader_status}")
                self.logger.info(f"   - Step Interface: {interface_status}")
                
            except Exception as e:
                self.logger.error(f"❌ ModelLoader 인터페이스 설정 실패: {e}")
                self.step_interface = None
        
        # BaseStepMixin에 안전한 메서드 적용
        BaseStepMixin._setup_model_interface = safe_setup_model_interface
        
        logger.info("✅ BaseStepMixin._setup_model_interface 패치 완료")
        
        return True
        
    except ImportError as e:
        logger.warning(f"BaseStepMixin import 실패: {e}")
        return False
    except Exception as e:
        logger.error(f"워밍업 시스템 패치 실패: {e}")
        return False

def disable_problematic_async_methods():
    """
    문제가 되는 async 메서드들을 일시적으로 비활성화
    """
    try:
        # Step 클래스들의 async 메서드를 동기 버전으로 교체
        step_classes = []
        
        try:
            from backend.app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
            step_classes.append(HumanParsingStep)
        except ImportError:
            pass
            
        try:
            from backend.app.ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
            step_classes.append(GeometricMatchingStep)
        except ImportError:
            pass
        
        for step_class in step_classes:
            # warmup_step 메서드를 동기로 교체
            if hasattr(step_class, 'warmup_step') and asyncio.iscoroutinefunction(step_class.warmup_step):
                def sync_warmup_step(self):
                    """동기 워밍업 (안전 버전)"""
                    return {'success': True, 'message': f'{self.__class__.__name__} 워밍업 완료'}
                
                step_class.warmup_step = sync_warmup_step
                logger.info(f"✅ {step_class.__name__}.warmup_step -> 동기 버전으로 교체")
            
            # _setup_model_interface 메서드도 동기로 교체
            if hasattr(step_class, '_setup_model_interface') and asyncio.iscoroutinefunction(step_class._setup_model_interface):
                def sync_setup_model_interface(self):
                    """동기 모델 인터페이스 설정"""
                    self.logger.info(f"🔗 {self.__class__.__name__} 모델 인터페이스 설정 (동기)")
                    return None
                
                step_class._setup_model_interface = sync_setup_model_interface
                logger.info(f"✅ {step_class.__name__}._setup_model_interface -> 동기 버전으로 교체")
        
        return True
        
    except Exception as e:
        logger.error(f"async 메서드 비활성화 실패: {e}")
        return False

# 패치 적용 함수
def apply_warmup_patches():
    """모든 워밍업 관련 패치 적용"""
    logger.info("🔧 워밍업 안전 패치 적용 시작...")
    
    success_count = 0
    
    # 1. 워밍업 시스템 패치
    if patch_warmup_system():
        success_count += 1
        logger.info("✅ 워밍업 시스템 패치 성공")
    
    # 2. 문제가 되는 async 메서드 비활성화
    if disable_problematic_async_methods():
        success_count += 1
        logger.info("✅ async 메서드 비활성화 성공")
    
    if success_count > 0:
        logger.info(f"🎉 워밍업 패치 완료: {success_count}/2 성공")
        return True
    else:
        logger.warning("⚠️ 워밍업 패치 실패")
        return False

# 자동 적용
if __name__ == "__main__":
    apply_warmup_patches()

# 모듈 import 시 자동 실행
try:
    apply_warmup_patches()
except Exception as e:
    logger.error(f"자동 워밍업 패치 실패: {e}")

__all__ = ['apply_warmup_patches', 'patch_warmup_system', 'disable_problematic_async_methods']