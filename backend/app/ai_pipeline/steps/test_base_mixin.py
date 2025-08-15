#!/usr/bin/env python3
"""
🔥 BaseStepMixin 직접 테스트
================================

BaseStepMixin이 제대로 작동하는지 테스트
"""

import sys
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_base_mixin():
    """BaseStepMixin 직접 테스트"""
    try:
        logger.info("🔍 BaseStepMixin import 테스트 시작")
        
        # BaseStepMixin import 시도
        from base.core.base_step_mixin import BaseStepMixin
        logger.info("✅ BaseStepMixin import 성공")
        
        # BaseStepMixin 인스턴스 생성 테스트
        logger.info("🔍 BaseStepMixin 인스턴스 생성 테스트")
        
        # 간단한 인스턴스 생성
        step = BaseStepMixin(step_name="test_step", step_id=1)
        logger.info("✅ BaseStepMixin 인스턴스 생성 성공")
        
        # 기본 속성 확인
        logger.info(f"   step_name: {step.step_name}")
        logger.info(f"   step_id: {step.step_id}")
        logger.info(f"   device: {step.device}")
        logger.info(f"   is_initialized: {step.is_initialized}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ BaseStepMixin 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_base_mixin()
    if success:
        print("✅ BaseStepMixin 테스트 성공")
    else:
        print("❌ BaseStepMixin 테스트 실패")
