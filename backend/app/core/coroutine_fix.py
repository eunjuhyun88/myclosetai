# backend/app/core/coroutine_fix.py
"""
🔧 Coroutine 오류 즉시 해결 패치
coroutine 'was never awaited' 및 'object is not callable' 완전 해결
"""

import asyncio
import inspect
import logging
from typing import Any, Callable, Coroutine, Union
from functools import wraps

logger = logging.getLogger(__name__)

class CoroutineFixer:
    """Coroutine 관련 오류 완전 해결 클래스"""
    
    @staticmethod
    def fix_coroutine_call(func_or_method: Any) -> Any:
        """
        Coroutine 함수를 안전하게 동기 함수로 변환
        """
        if not asyncio.iscoroutinefunction(func_or_method):
            return func_or_method
        
        @wraps(func_or_method)
        def sync_wrapper(*args, **kwargs):
            try:
                # 현재 이벤트 루프 확인
                try:
                    loop = asyncio.get_running_loop()
                    # 이미 실행 중인 루프가 있으면 태스크로 실행
                    task = asyncio.create_task(func_or_method(*args, **kwargs))
                    return task
                except RuntimeError:
                    # 실행 중인 루프가 없으면 새 루프 생성
                    return asyncio.run(func_or_method(*args, **kwargs))
            except Exception as e:
                logger.warning(f"Coroutine 변환 실패: {e}")
                return None
        
        return sync_wrapper
    
    @staticmethod
    def fix_step_class(step_class: Any) -> Any:
        """
        Step 클래스의 모든 async 메서드를 동기 메서드로 변환
        """
        if not hasattr(step_class, '__dict__'):
            return step_class
        
        for attr_name in dir(step_class):
            if attr_name.startswith('_'):
                continue
                
            attr = getattr(step_class, attr_name, None)
            if attr and asyncio.iscoroutinefunction(attr):
                logger.info(f"🔧 {step_class.__name__}.{attr_name} -> 동기 버전으로 변환")
                sync_version = CoroutineFixer.fix_coroutine_call(attr)
                setattr(step_class, attr_name, sync_version)
        
        return step_class
    
    @staticmethod
    def patch_base_step_mixin():
        """
        BaseStepMixin의 워밍업 관련 메서드들을 안전하게 패치
        """
        try:
            from backend.app.ai_pipeline.steps.base_step_mixin import BaseStepMixin
            
            # _pipeline_warmup 메서드를 안전하게 수정
            def safe_pipeline_warmup(self):
                """안전한 파이프라인 워밍업 (동기)"""
                try:
                    # Step별 워밍업 로직 (기본)
                    if hasattr(self, 'warmup_step'):
                        warmup_method = getattr(self, 'warmup_step')
                        
                        # async 함수면 동기로 변환하여 호출
                        if asyncio.iscoroutinefunction(warmup_method):
                            try:
                                result = asyncio.run(warmup_method())
                                return {'success': result.get('success', True), 'message': 'Step 워밍업 완료'}
                            except Exception as e:
                                logger.warning(f"비동기 워밍업 실패: {e}")
                                return {'success': False, 'error': str(e)}
                        else:
                            result = warmup_method()
                            return {'success': result.get('success', True), 'message': 'Step 워밍업 완료'}
                    
                    return {'success': True, 'message': '파이프라인 워밍업 건너뜀'}
                    
                except Exception as e:
                    return {'success': False, 'error': str(e)}
            
            # BaseStepMixin에 안전한 메서드 적용
            BaseStepMixin._pipeline_warmup = safe_pipeline_warmup
            
            logger.info("✅ BaseStepMixin 워밍업 메서드 패치 완료")
            return True
            
        except ImportError as e:
            logger.warning(f"BaseStepMixin import 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"BaseStepMixin 패치 실패: {e}")
            return False

def apply_coroutine_fixes():
    """
    전체 시스템에 Coroutine 수정 적용
    """
    logger.info("🔧 Coroutine 오류 수정 적용 시작...")
    
    # 1. BaseStepMixin 패치
    if CoroutineFixer.patch_base_step_mixin():
        logger.info("✅ BaseStepMixin 패치 완료")
    
    # 2. Step 클래스들 패치
    step_modules = [
        'backend.app.ai_pipeline.steps.step_01_human_parsing',
        'backend.app.ai_pipeline.steps.step_02_pose_estimation', 
        'backend.app.ai_pipeline.steps.step_03_cloth_segmentation',
        'backend.app.ai_pipeline.steps.step_04_geometric_matching',
        'backend.app.ai_pipeline.steps.step_05_cloth_warping',
        'backend.app.ai_pipeline.steps.step_06_virtual_fitting',
        'backend.app.ai_pipeline.steps.step_07_post_processing',
        'backend.app.ai_pipeline.steps.step_08_quality_assessment'
    ]
    
    fixed_count = 0
    for module_name in step_modules:
        try:
            module = __import__(module_name, fromlist=[''])
            
            # 모듈의 모든 클래스 확인
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (inspect.isclass(attr) and 
                    attr_name.endswith('Step') and 
                    hasattr(attr, '__init__')):
                    
                    CoroutineFixer.fix_step_class(attr)
                    fixed_count += 1
                    logger.info(f"✅ {attr_name} 클래스 패치 완료")
                    
        except ImportError:
            logger.debug(f"모듈 import 실패: {module_name}")
        except Exception as e:
            logger.warning(f"모듈 패치 실패 {module_name}: {e}")
    
    logger.info(f"🎉 Coroutine 수정 완료: {fixed_count}개 클래스 패치됨")
    return fixed_count > 0

# 자동 적용
if __name__ == "__main__":
    apply_coroutine_fixes()

# 모듈 import 시 자동 실행
try:
    apply_coroutine_fixes()
except Exception as e:
    logger.error(f"자동 Coroutine 수정 실패: {e}")

__all__ = ['CoroutineFixer', 'apply_coroutine_fixes']