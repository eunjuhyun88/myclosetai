
"""
🔧 워밍업 오류 패치 - dict object is not callable 해결
이 파일을 import하면 자동으로 워밍업 오류가 패치됩니다.
"""

import logging
from app.utils.safe_caller import safe_call, safe_warmup

logger = logging.getLogger(__name__)

def patch_warmup_methods():
    """워밍업 메서드들을 안전한 버전으로 패치"""
    
    # 공통적으로 문제가 되는 모듈들
    modules_to_patch = [
        'app.ai_pipeline.steps',
        'app.ai_pipeline.pipeline_manager',
        'app.services.ai_models'
    ]
    
    for module_name in modules_to_patch:
        try:
            import importlib
            module = importlib.import_module(module_name)
            
            # 모듈 내의 클래스들에서 warmup 메서드 패치
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, 'warmup') and callable(attr):
                    original_warmup = attr.warmup
                    
                    def safe_warmup_wrapper(*args, **kwargs):
                        return safe_warmup(original_warmup, *args, **kwargs)
                    
                    attr.warmup = safe_warmup_wrapper
                    logger.debug(f"✅ {module_name}.{attr_name}.warmup 패치 완료")
                    
        except Exception as e:
            logger.warning(f"⚠️ 모듈 패치 실패 {module_name}: {e}")

# 자동 패치 실행
try:
    patch_warmup_methods()
    logger.info("✅ 워밍업 패치 적용 완료")
except Exception as e:
    logger.error(f"❌ 워밍업 패치 실패: {e}")

__all__ = ['patch_warmup_methods']
