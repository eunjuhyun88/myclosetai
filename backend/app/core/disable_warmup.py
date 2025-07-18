# backend/app/core/disable_warmup.py
"""
🔧 워밍업 비활성화 패치 - dict object is not callable 오류 임시 해결
"""

import os
import logging

logger = logging.getLogger(__name__)

def disable_warmup_globally():
    """전역적으로 워밍업 비활성화"""
    
    # 환경변수 설정
    os.environ['ENABLE_MODEL_WARMUP'] = 'false'
    os.environ['SKIP_WARMUP'] = 'true'
    os.environ['AUTO_WARMUP'] = 'false'
    
    logger.info("🚫 워밍업 전역적으로 비활성화됨")
    
    return True

# 모듈 import 시 자동 실행
try:
    disable_warmup_globally()
    logger.info("✅ 워밍업 비활성화 패치 적용 완료")
except Exception as e:
    logger.error(f"❌ 워밍업 비활성화 실패: {e}")

__all__ = ['disable_warmup_globally']