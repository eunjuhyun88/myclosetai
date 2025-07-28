# backend/app/ai_pipeline/interfaces/__init__.py
"""
🔧 Interface 경로 호환성 패치 (Clean Version)
============================================

이 파일은 기존 코드에서 잘못된 경로로 import하는 문제를 해결합니다:
- 기존: app.ai_pipeline.interfaces (틀림)
- 올바른 경로: app.ai_pipeline.interface (맞음)

이 패치를 통해 기존 코드 수정 없이 호환성을 제공합니다.
Logger 메시지는 최소화하여 로그 과다 출력을 방지합니다.
"""

import logging
import warnings

logger = logging.getLogger(__name__)

# 조용한 호환성 경고 (개발 환경에서만)
warnings.filterwarnings('ignore', category=DeprecationWarning, module='app.ai_pipeline.interfaces')

# 올바른 경로에서 모든 클래스와 함수를 import
try:
    from ..interface.step_interface import *
    # 성공 로그는 DEBUG 레벨로 변경 (과다 출력 방지)
    logger.debug("StepInterface 호환성 패치 적용 성공")
except ImportError as e:
    logger.error(f"❌ StepInterface 호환성 패치 실패: {e}")
    
    # 폴백 구현
    class StepInterface:
        """폴백 StepInterface"""
        def __init__(self, step_name: str, **kwargs):
            self.step_name = step_name
            self.logger = logging.getLogger(f"FallbackStepInterface.{step_name}")
            self.logger.warning("⚠️ 폴백 StepInterface 사용 중")
        
        def register_model_requirement(self, *args, **kwargs):
            self.logger.debug("폴백 모드 - register_model_requirement 무시됨")
            return True
        
        def list_available_models(self, *args, **kwargs):
            self.logger.debug("폴백 모드 - 빈 모델 목록 반환")
            return []
        
        def get_model(self, *args, **kwargs):
            self.logger.debug("폴백 모드 - None 반환")
            return None
        
        def load_model(self, *args, **kwargs):
            self.logger.debug("폴백 모드 - None 반환")
            return None

# backward compatibility를 위한 alias 설정
try:
    from ..interface.step_interface import (
        GitHubStepModelInterface as StepModelInterface,
        GitHubStepConfig as StepConfig,
        GitHubStepType as StepType,
        GitHubStepPriority as StepPriority,
        create_github_step_interface_with_diagnostics as create_step_interface,
        create_optimized_github_interface as create_optimized_interface,
        get_github_environment_info as get_environment_info,
        optimize_github_environment as optimize_environment
    )
    
    # DEBUG 레벨로 변경 (과다 로그 방지)
    logger.debug("StepInterface 별칭 설정 완료")
    
except ImportError:
    logger.error("❌ StepInterface 별칭 설정 실패 - 폴백 모드")

__all__ = [
    'StepInterface',
    'StepModelInterface', 
    'StepConfig',
    'StepType',
    'StepPriority',
    'create_step_interface',
    'create_optimized_interface',
    'get_environment_info',
    'optimize_environment'
]

# 초기화 로그도 DEBUG 레벨로 변경 (과다 출력 방지)
logger.debug("Interface 호환성 패치 로드 완료")