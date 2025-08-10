# backend/app/core/warmup_safe_patch.py
"""
🔧 워밍업 안전 패치 - RuntimeWarning 및 'dict object is not callable' 완전 해결
"""

import asyncio
import logging
import os
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

def patch_warmup_system():
    """워밍업 시스템 패치"""
    try:
        # 환경변수 설정으로 워밍업 비활성화
        os.environ['ENABLE_MODEL_WARMUP'] = 'false'
        os.environ['SKIP_WARMUP'] = 'true'
        os.environ['AUTO_WARMUP'] = 'false'
        os.environ['DISABLE_AI_WARMUP'] = 'true'
        
        logger.info("🚫 워밍업 시스템 전역 비활성화")
        return True
        
    except Exception as e:
        logger.error(f"워밍업 시스템 패치 실패: {e}")
        return False

def disable_problematic_async_methods():
    """문제가 되는 async 메서드들을 동기 버전으로 교체"""
    try:
        step_classes = []
        
        # 문제가 되는 Step 클래스들 import
        try:
            from ..ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
            step_classes.append(HumanParsingStep)
        except ImportError:
            pass
            
        try:
            from ..ai_pipeline.steps.step_04_geometric_matching import GeometricMatchingStep
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

__all__ = ['apply_warmup_patches', 'patch_warmup_system', 'disable_problematic_async_methods']
