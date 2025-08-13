"""
모델 경로 매핑 유틸리티
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class EnhancedModelPathMapper:
    """향상된 모델 경로 매퍼"""
    
    def __init__(self, ai_models_root: str = "ai_models"):
        self.ai_models_root = ai_models_root
        self.ai_models_path = self._auto_detect_ai_models_path()
        self.model_cache = {}
        
        logger.info(f"🔍 EnhancedModelPathMapper 초기화: {self.ai_models_path}")
    
    def _auto_detect_ai_models_path(self) -> Path:
        """AI 모델 경로 자동 감지"""
        possible_paths = [
            Path(self.ai_models_root),
            Path("ai_models"),
            Path("../ai_models"),
            Path("../../ai_models"),
            Path("backend/ai_models"),
            Path("models"),
            Path("backend/models"),
            Path("backend/app/ai_models"),
            Path("backend/app/models"),
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                logger.info(f"✅ AI 모델 경로 발견: {path}")
                return path
        
        # 기본 경로 반환
        default_path = Path(self.ai_models_root)
        logger.warning(f"⚠️ AI 모델 경로를 찾을 수 없음. 기본 경로 사용: {default_path}")
        return default_path
    
    def find_model_file(self, filename: str) -> Optional[Path]:
        """모델 파일 찾기"""
        if filename in self.model_cache:
            return self.model_cache[filename]
        
        # 직접 경로 확인
        direct_path = self.ai_models_path / filename
        if direct_path.exists():
            self.model_cache[filename] = direct_path
            return direct_path
        
        # 하위 디렉토리에서 검색
        for root, dirs, files in os.walk(self.ai_models_path):
            if filename in files:
                found_path = Path(root) / filename
                self.model_cache[filename] = found_path
                logger.info(f"✅ 모델 파일 발견: {found_path}")
                return found_path
        
        logger.warning(f"⚠️ 모델 파일을 찾을 수 없음: {filename}")
        return None
    
    def get_geometric_matching_models(self) -> Dict[str, Path]:
        """기하학적 매칭 모델들 반환"""
        model_files = {
            'gmm_model': 'gmm_model.pth',
            'tps_model': 'tps_model.pth',
            'optical_flow_model': 'optical_flow_model.pth',
            'keypoint_matcher': 'keypoint_matcher.pth',
            'advanced_geometric_matcher': 'advanced_geometric_matcher.pth',
            'deeplab_backbone': 'deeplab_backbone.pth',
            'aspp_module': 'aspp_module.pth',
            'self_attention_matcher': 'self_attention_matcher.pth',
            'edge_aware_transform': 'edge_aware_transform.pth',
            'progressive_refinement': 'progressive_refinement.pth'
        }
        
        found_models = {}
        for model_name, filename in model_files.items():
            model_path = self.find_model_file(filename)
            if model_path:
                found_models[model_name] = model_path
                logger.info(f"✅ {model_name} 모델 발견: {model_path}")
            else:
                logger.warning(f"⚠️ {model_name} 모델을 찾을 수 없음: {filename}")
        
        return found_models
    
    def get_model_info(self, model_path: Path) -> Dict[str, any]:
        """모델 정보 반환"""
        if not model_path.exists():
            return {}
        
        try:
            stat = model_path.stat()
            return {
                'path': str(model_path),
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': stat.st_mtime,
                'exists': True
            }
        except Exception as e:
            logger.error(f"모델 정보 조회 실패: {e}")
            return {'exists': False, 'error': str(e)}
    
    def list_available_models(self) -> List[str]:
        """사용 가능한 모델 목록 반환"""
        available_models = []
        
        if not self.ai_models_path.exists():
            return available_models
        
        for root, dirs, files in os.walk(self.ai_models_path):
            for file in files:
                if file.endswith(('.pth', '.pt', '.ckpt', '.h5')):
                    relative_path = Path(root).relative_to(self.ai_models_path)
                    model_path = relative_path / file
                    available_models.append(str(model_path))
        
        return sorted(available_models)
    
    def validate_model_paths(self) -> Dict[str, bool]:
        """모델 경로 검증"""
        geometric_models = self.get_geometric_matching_models()
        validation_results = {}
        
        for model_name, model_path in geometric_models.items():
            validation_results[model_name] = model_path.exists() if model_path else False
        
        return validation_results
