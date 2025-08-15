#!/usr/bin/env python3
"""
🔥 MyCloset AI - Quality Assessment Model Loader
================================================

✅ 통일된 모델 로딩 시스템
✅ 체크포인트 자동 감지
✅ 최적 모델 선택

Author: MyCloset AI Team
Date: 2025-08-14
Version: 1.0 (통일된 구조)
"""

import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class QualityAssessmentModelLoader:
    """Quality Assessment 모델 로더 - 통일된 구조"""
    
    def __init__(self):
        self.supported_models = [
            'quality_net', 'assessment_network', 'evaluation_model', 'quality_estimator'
        ]
        self.model_paths = {}
        self._discover_models()
    
    def _discover_models(self):
        """사용 가능한 모델들을 자동으로 발견"""
        try:
            # AI 모델 디렉토리에서 체크포인트 탐색
            ai_models_root = Path("/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models")
            if ai_models_root.exists():
                for checkpoint_file in ai_models_root.rglob("*.pth"):
                    if any(keyword in checkpoint_file.name.lower() for keyword in 
                          ['quality', 'assessment', 'evaluation', 'estimator']):
                        self.model_paths[checkpoint_file.stem] = str(checkpoint_file)
                        logger.info(f"✅ Quality Assessment 모델 발견: {checkpoint_file.name}")
        except Exception as e:
            logger.warning(f"⚠️ 모델 탐색 중 오류: {e}")
    
    def load_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """지정된 모델 로드"""
        try:
            if model_name in self.model_paths:
                return {
                    'model_name': model_name,
                    'checkpoint_path': self.model_paths[model_name],
                    'status': 'available'
                }
            else:
                logger.warning(f"⚠️ 모델을 찾을 수 없음: {model_name}")
                return None
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            return None
    
    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        return list(self.model_paths.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """모델 정보 반환"""
        if model_name in self.model_paths:
            checkpoint_path = Path(self.model_paths[model_name])
            return {
                'name': model_name,
                'path': str(checkpoint_path),
                'size_mb': checkpoint_path.stat().st_size / (1024 * 1024),
                'modified': checkpoint_path.stat().st_mtime
            }
        return None
