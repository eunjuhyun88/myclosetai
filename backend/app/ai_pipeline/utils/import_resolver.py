# backend/app/ai_pipeline/utils/import_resolver.py (새 파일 생성)
"""
🔥 Import 경로 문제 해결을 위한 동적 Import Resolver
================================================================
✅ attempted relative import beyond top-level package 오류 해결
✅ 동적 경로 탐지 및 절대 경로 변환
✅ 프로젝트 구조 변화에 유연하게 대응
✅ 폴백 메커니즘 포함
"""

import os
import sys
import importlib
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List, Union

logger = logging.getLogger(__name__)

class ImportResolver:
    """Import 경로 문제 해결을 위한 동적 리졸버"""
    
    def __init__(self):
        self.project_root = self._find_project_root()
        self.backend_root = self._find_backend_root()
        self._add_paths_to_sys()
        self._setup_module_aliases()
    
    def _find_project_root(self) -> Path:
        """프로젝트 루트 디렉토리 찾기"""
        current = Path(__file__).parent
        while current.parent != current:
            if (current / "backend").exists() or (current / ".git").exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def _find_backend_root(self) -> Path:
        """백엔드 루트 디렉토리 찾기"""
        backend_path = self.project_root / "backend"
        if backend_path.exists():
            return backend_path
        
        # 현재 파일이 backend 내부에 있는 경우
        current = Path(__file__).parent
        while current.parent != current:
            if current.name == "backend":
                return current
            current = current.parent
        
        return self.project_root
    
    def _add_paths_to_sys(self):
        """sys.path에 필요한 경로들 추가"""
        paths_to_add = [
            str(self.project_root),
            str(self.backend_root),
            str(self.backend_root / "app"),
            str(self.backend_root / "app" / "ai_pipeline"),
            str(self.backend_root / "app" / "ai_pipeline" / "utils"),
            str(self.backend_root / "app" / "ai_pipeline" / "factories"),
            str(self.backend_root / "app" / "ai_pipeline" / "interface"),
            str(self.backend_root / "app" / "ai_pipeline" / "steps"),
            str(self.backend_root / "app" / "services"),
        ]
        
        for path in paths_to_add:
            if os.path.exists(path) and path not in sys.path:
                sys.path.insert(0, path)
                logger.debug(f"✅ sys.path에 추가: {path}")
    
    def _setup_module_aliases(self):
        """모듈 별칭 설정으로 import 호환성 향상"""
        try:
            # step_model_requests.py 별칭 설정
            aliases = [
                ('step_model_requests', 'step_model_requirements'),
                ('step_model_requirements', 'step_model_requests'),
            ]
            
            for alias_from, alias_to in aliases:
                try:
                    if alias_to in sys.modules and alias_from not in sys.modules:
                        sys.modules[alias_from] = sys.modules[alias_to]
                        logger.debug(f"✅ 모듈 별칭 생성: {alias_from} -> {alias_to}")
                except Exception as e:
                    logger.debug(f"⚠️ 모듈 별칭 생성 실패: {e}")
        except Exception as e:
            logger.debug(f"⚠️ 모듈 별칭 설정 실패: {e}")
    
    def safe_import(self, module_path: str, fallback_paths: Optional[List[str]] = None) -> Optional[Any]:
        """안전한 모듈 import (폴백 경로 포함)"""
        all_paths = [module_path]
        if fallback_paths:
            all_paths.extend(fallback_paths)
        
        for path in all_paths:
            try:
                # relative import를 절대 import로 변환
                if path.startswith('.'):
                    # 현재 모듈의 패키지 기준으로 절대 경로 계산
                    current_package = self._get_current_package()
                    if current_package:
                        abs_path = self._resolve_relative_path(path, current_package)
                        if abs_path:
                            path = abs_path
                
                module = importlib.import_module(path)
                logger.debug(f"✅ 모듈 import 성공: {path}")
                return module
                
            except ImportError as e:
                logger.debug(f"⚠️ {path} import 실패: {e}")
                continue
            except Exception as e:
                logger.debug(f"❌ {path} import 오류: {e}")
                continue
        
        logger.warning(f"❌ 모든 경로에서 import 실패: {all_paths}")
        return None
    
    def _get_current_package(self) -> Optional[str]:
        """현재 모듈의 패키지명 반환"""
        try:
            frame = sys._getframe(2)  # 호출한 모듈의 프레임
            module_name = frame.f_globals.get('__name__', '')
            if '.' in module_name:
                return '.'.join(module_name.split('.')[:-1])
            return None
        except:
            return None
    
    def _resolve_relative_path(self, relative_path: str, current_package: str) -> Optional[str]:
        """상대 경로를 절대 경로로 변환"""
        try:
            if relative_path.startswith('..'):
                # 부모 패키지로 이동
                dots = len(relative_path) - len(relative_path.lstrip('.'))
                package_parts = current_package.split('.')
                
                if dots > len(package_parts):
                    return None
                
                target_package = '.'.join(package_parts[:-dots+1])
                module_part = relative_path[dots:]
                
                if module_part:
                    return f"{target_package}{module_part}"
                else:
                    return target_package
            
            elif relative_path.startswith('.'):
                # 현재 패키지 내 모듈
                module_part = relative_path[1:]
                if module_part:
                    return f"{current_package}{module_part}"
                else:
                    return current_package
        except Exception as e:
            logger.debug(f"⚠️ 상대 경로 변환 실패: {e}")
            return None
    
    def import_class_safe(self, module_path: str, class_name: str, fallback_paths: Optional[List[str]] = None) -> Optional[Any]:
        """클래스 안전 import"""
        module = self.safe_import(module_path, fallback_paths)
        if module and hasattr(module, class_name):
            return getattr(module, class_name)
        
        logger.warning(f"❌ 클래스를 찾을 수 없음: {class_name} in {module_path}")
        return None
    
    def import_function_safe(self, module_path: str, function_name: str, fallback_paths: Optional[List[str]] = None) -> Optional[Any]:
        """함수 안전 import"""
        module = self.safe_import(module_path, fallback_paths)
        if module and hasattr(module, function_name):
            return getattr(module, function_name)
        
        logger.warning(f"❌ 함수를 찾을 수 없음: {function_name} in {module_path}")
        return None

# 전역 리졸버 인스턴스
_global_resolver = None

def get_import_resolver() -> ImportResolver:
    """전역 Import Resolver 반환"""
    global _global_resolver
    if _global_resolver is None:
        _global_resolver = ImportResolver()
    return _global_resolver

def safe_import(module_path: str, fallback_paths: Optional[List[str]] = None) -> Optional[Any]:
    """편의 함수: 안전한 모듈 import"""
    resolver = get_import_resolver()
    return resolver.safe_import(module_path, fallback_paths)

def import_class(module_path: str, class_name: str, fallback_paths: Optional[List[str]] = None) -> Optional[Any]:
    """편의 함수: 클래스 안전 import"""
    resolver = get_import_resolver()
    return resolver.import_class_safe(module_path, class_name, fallback_paths)

def import_function(module_path: str, function_name: str, fallback_paths: Optional[List[str]] = None) -> Optional[Any]:
    """편의 함수: 함수 안전 import"""
    resolver = get_import_resolver()
    return resolver.import_function_safe(module_path, function_name, fallback_paths)

# 🔥 step_model_requirements 전용 import 함수
def import_step_model_requirements():
    """step_model_requirements 전용 import 함수"""
    resolver = get_import_resolver()
    
    # 다양한 경로로 시도
    paths = [
        'backend.app.ai_pipeline.utils.step_model_requests',
        'app.ai_pipeline.utils.step_model_requests', 
        'ai_pipeline.utils.step_model_requests',
        'backend.app.ai_pipeline.utils.step_model_requirements',
        'app.ai_pipeline.utils.step_model_requirements',
        'ai_pipeline.utils.step_model_requirements',
        'step_model_requests',
        'step_model_requirements'
    ]
    
    for path in paths:
        module = resolver.safe_import(path)
        if module and hasattr(module, 'get_enhanced_step_request'):
            logger.info(f"✅ step_model_requirements import 성공: {path}")
            return {
                'get_enhanced_step_request': module.get_enhanced_step_request,
                'REAL_STEP_MODEL_REQUESTS': getattr(module, 'REAL_STEP_MODEL_REQUESTS', {})
            }
    
    logger.warning("❌ step_model_requirements import 모든 경로 실패")
    return None

# 모듈 로드 시 자동으로 경로 설정
resolver = get_import_resolver()
logger.info("✅ Import Resolver 초기화 완료")
logger.info(f"📁 프로젝트 루트: {resolver.project_root}")
logger.info(f"📁 백엔드 루트: {resolver.backend_root}")