#!/usr/bin/env python3
"""
🔥 MyCloset AI - 표준화된 Import 경로 가이드
============================================

모든 Step에서 사용해야 하는 표준화된 import 경로들을 정의
폴백 시스템 없이 명확하고 일관된 import 패턴 제공

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0 (표준화된 Import 가이드)
"""

# ==============================================
# 🔥 1. BaseStepMixin Import (모든 Step에서 동일)
# ==============================================

# ✅ 올바른 방법 (표준화된 경로)
from ...base import BaseStepMixin

# ❌ 잘못된 방법들 (폴백 시스템)
# from app.ai_pipeline.steps.base import BaseStepMixin  # 절대 경로
# from ..base.core.base_step_mixin import BaseStepMixin  # 직접 core 접근
# import sys; sys.path.insert(0, path); from __init__ import BaseStepMixin  # 경로 조작

# ==============================================
# 🔥 2. AI 모델 Import (각 Step별로 표준화)
# ==============================================

# ✅ 올바른 방법 (상대 경로)
from .models.model_name import ModelClass

# ❌ 잘못된 방법들
# from app.ai_pipeline.steps.step_XX.models.model_name import ModelClass  # 절대 경로
# import sys; sys.path.insert(0, models_dir); from model_name import ModelClass  # 경로 조작

# ==============================================
# 🔥 3. DI Container 접근 (표준화된 유틸리티 사용)
# ==============================================

# ✅ 올바른 방법 (표준화된 유틸리티)
from app.ai_pipeline.utils.di_container_access import get_service, register_service

# ❌ 잘못된 방법들
# from app.core.di_container import get_global_container  # 직접 접근
# from app.api.central_hub import get_service  # 중간 계층 통과

# ==============================================
# 🔥 4. 유틸리티 Import (표준화된 경로)
# ==============================================

# ✅ 올바른 방법
from .utils.utility_name import UtilityClass
from ..utils.common_utility import CommonUtility

# ❌ 잘못된 방법들
# import os; current_dir = os.path.dirname(os.path.abspath(__file__))  # 경로 계산
# import sys; sys.path.insert(0, utility_path)  # 경로 조작

# ==============================================
# 🔥 5. 실제 사용 예시
# ==============================================

class ExampleStep(BaseStepMixin):
    """표준화된 import를 사용하는 Step 예시"""
    
    def __init__(self, **kwargs):
        # BaseStepMixin 초기화
        super().__init__(**kwargs)
        
        # AI 모델 로드
        self.model = self._load_model()
        
        # DI Container에서 서비스 가져오기
        self.session_manager = get_service('session_manager')
    
    def _load_model(self):
        """AI 모델 로드 (표준화된 방식)"""
        try:
            from .models.example_model import ExampleModel
            return ExampleModel()
        except ImportError as e:
            raise ImportError(f"AI 모델을 로드할 수 없습니다: {e}")
    
    def process(self, input_data):
        """데이터 처리 (표준화된 서비스 사용)"""
        # DI Container에서 서비스 조회
        data_converter = get_service('data_converter')
        if data_converter:
            processed_data = data_converter.convert(input_data)
            return processed_data
        else:
            raise RuntimeError("필수 서비스를 찾을 수 없습니다: data_converter")

# ==============================================
# 🔥 6. Import 검증 함수
# ==============================================

def validate_imports():
    """필수 import들이 올바르게 작동하는지 검증"""
    try:
        # BaseStepMixin 검증
        from ...base import BaseStepMixin
        print("✅ BaseStepMixin import 성공")
        
        # DI Container 접근 유틸리티 검증
        from app.ai_pipeline.utils.di_container_access import get_service
        print("✅ DI Container 접근 유틸리티 import 성공")
        
        return True
    except ImportError as e:
        print(f"❌ Import 검증 실패: {e}")
        return False

# ==============================================
# 🔥 7. 공개 API
# ==============================================

__all__ = [
    'ExampleStep',
    'validate_imports'
]

# ==============================================
# 🔥 8. 사용법 안내
# ==============================================

"""
📋 표준화된 Import 사용법:

1. BaseStepMixin: 항상 `from ...base import BaseStepMixin` 사용
2. AI 모델: 항상 `from .models.model_name import ModelClass` 사용
3. DI Container: 항상 `from app.ai_pipeline.utils.di_container_access import get_service` 사용
4. 유틸리티: 상대 경로를 우선으로 사용

🚫 금지사항:
- 절대 경로 import (app.ai_pipeline.steps...)
- 경로 조작 (sys.path.insert, os.path 조작)
- 폴백 시스템 (try-except로 여러 경로 시도)
- 직접 core 모듈 접근

✅ 권장사항:
- 상대 경로 우선 사용
- 표준화된 유틸리티 사용
- 명확한 에러 메시지 제공
- 일관된 import 패턴 유지
"""
