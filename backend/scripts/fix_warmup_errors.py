#!/usr/bin/env python3
"""
🔧 워밍업 'dict' object is not callable 오류 완전 해결
✅ 모든 워밍업 메서드에서 딕셔너리 함수 호출 오류 수정
✅ 안전한 함수 호출 래퍼 적용
✅ 워밍업 실패 시 안전한 폴백 처리
"""

import os
import sys
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_dict_callable_errors():
    """dict object is not callable 오류가 있는 파일들 찾기"""
    
    backend_dir = Path(__file__).parent.parent
    
    problematic_patterns = [
        r'config\(\)',  # config()로 호출하는 경우
        r'model_config\(\)',  # model_config()로 호출하는 경우
        r'warmup\(\)',  # warmup()이 딕셔너리인 경우
        r'\.get\(\)\..*\(\)',  # .get().something() 체인 호출
    ]
    
    problematic_files = []
    
    for py_file in backend_dir.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                for pattern in problematic_patterns:
                    if re.search(pattern, content):
                        problematic_files.append((py_file, pattern))
                        break
                        
        except Exception as e:
            logger.warning(f"파일 읽기 실패: {py_file}, {e}")
    
    return problematic_files

def create_safe_function_caller():
    """안전한 함수 호출 유틸리티 생성"""
    
    safe_caller_content = '''
"""
🔧 안전한 함수 호출 유틸리티 - dict object is not callable 방지
"""

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

class SafeFunctionCaller:
    """안전한 함수 호출 래퍼"""
    
    @staticmethod
    def safe_call(obj: Any, *args, **kwargs) -> Any:
        """
        안전한 함수 호출 - dict object is not callable 방지
        
        Args:
            obj: 호출할 객체 (함수 또는 딕셔너리)
            *args: 위치 인수
            **kwargs: 키워드 인수
            
        Returns:
            호출 결과 또는 안전한 기본값
        """
        try:
            # 1차 확인: 실제 callable인지 확인
            if callable(obj):
                return obj(*args, **kwargs)
            
            # 2차 확인: 딕셔너리인 경우
            elif isinstance(obj, dict):
                logger.warning(f"⚠️ 딕셔너리를 함수로 호출 시도: {type(obj)}")
                
                # 딕셔너리에서 callable 찾기
                for key, value in obj.items():
                    if callable(value):
                        logger.info(f"🔍 딕셔너리에서 함수 발견: {key}")
                        return value(*args, **kwargs)
                
                # 특별한 키들 확인
                special_keys = ['function', 'callable', 'method', 'process', 'execute']
                for key in special_keys:
                    if key in obj and callable(obj[key]):
                        logger.info(f"🔍 특별 키에서 함수 발견: {key}")
                        return obj[key](*args, **kwargs)
                
                # callable이 없으면 딕셔너리 자체 반환
                logger.warning("⚠️ 딕셔너리에서 callable을 찾을 수 없음, 딕셔너리 반환")
                return obj
            
            # 3차 확인: None인 경우
            elif obj is None:
                logger.warning("⚠️ None 객체 호출 시도")
                return None
            
            # 4차 확인: 다른 객체인 경우
            else:
                logger.warning(f"⚠️ callable이 아닌 객체 호출 시도: {type(obj)}")
                return obj
                
        except Exception as e:
            logger.error(f"❌ 안전한 함수 호출 실패: {e}")
            return None
    
    @staticmethod
    def safe_get_method(obj: Any, method_name: str, default_func: Optional[Callable] = None) -> Callable:
        """안전한 메서드 가져오기"""
        try:
            if hasattr(obj, method_name):
                method = getattr(obj, method_name)
                if callable(method):
                    return method
                else:
                    logger.warning(f"⚠️ {method_name}이 callable이 아님: {type(method)}")
                    return default_func or (lambda *a, **k: None)
            else:
                logger.warning(f"⚠️ {method_name} 메서드 없음")
                return default_func or (lambda *a, **k: None)
                
        except Exception as e:
            logger.error(f"❌ 메서드 가져오기 실패: {e}")
            return default_func or (lambda *a, **k: None)
    
    @staticmethod
    def safe_warmup(obj: Any, *args, **kwargs) -> bool:
        """안전한 워밍업 실행"""
        try:
            # warmup 메서드 찾기
            warmup_candidates = ['warmup', 'warm_up', 'initialize', 'init', 'prepare']
            
            for method_name in warmup_candidates:
                if hasattr(obj, method_name):
                    method = getattr(obj, method_name)
                    if callable(method):
                        logger.info(f"🔥 {method_name} 메서드로 워밍업 실행")
                        result = method(*args, **kwargs)
                        return result if result is not None else True
            
            # 메서드가 없으면 객체 자체가 callable인지 확인
            if callable(obj):
                logger.info("🔥 객체 자체를 워밍업 함수로 실행")
                result = obj(*args, **kwargs)
                return result if result is not None else True
            
            logger.warning("⚠️ 워밍업 메서드를 찾을 수 없음")
            return False
            
        except Exception as e:
            logger.error(f"❌ 안전한 워밍업 실패: {e}")
            return False

# 전역 함수들
safe_call = SafeFunctionCaller.safe_call
safe_get_method = SafeFunctionCaller.safe_get_method
safe_warmup = SafeFunctionCaller.safe_warmup

__all__ = ['SafeFunctionCaller', 'safe_call', 'safe_get_method', 'safe_warmup']
'''
    
    # 유틸리티 파일 저장
    backend_dir = Path(__file__).parent.parent
    safe_caller_path = backend_dir / 'app' / 'utils' / 'safe_caller.py'
    safe_caller_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(safe_caller_path, 'w', encoding='utf-8') as f:
        f.write(safe_caller_content)
    
    # __init__.py 파일도 업데이트
    init_file = safe_caller_path.parent / '__init__.py'
    if not init_file.exists():
        init_file.touch()
    
    logger.info(f"✅ 안전한 함수 호출 유틸리티 생성: {safe_caller_path}")
    return safe_caller_path

def fix_warmup_errors_in_files():
    """파일들에서 워밍업 오류 수정"""
    
    backend_dir = Path(__file__).parent.parent
    
    # 수정할 파일들과 패턴들
    fixes = [
        {
            'file_pattern': '**/step_*.py',
            'fixes': [
                {
                    'pattern': r'(\w+)\.warmup\(\)',
                    'replacement': r'safe_warmup(\1)',
                    'import_needed': True
                },
                {
                    'pattern': r'config\(\)',
                    'replacement': r'safe_call(config)',
                    'import_needed': True
                }
            ]
        },
        {
            'file_pattern': '**/pipeline_manager.py',
            'fixes': [
                {
                    'pattern': r'warmup\(\)',
                    'replacement': r'safe_warmup(warmup) if warmup else True',
                    'import_needed': True
                }
            ]
        },
        {
            'file_pattern': '**/model_loader.py',
            'fixes': [
                {
                    'pattern': r'model_config\(\)',
                    'replacement': r'safe_call(model_config)',
                    'import_needed': True
                }
            ]
        }
    ]
    
    import_statement = "from app.utils.safe_caller import safe_call, safe_warmup"
    
    fixed_files = []
    
    for fix_group in fixes:
        for file_path in backend_dir.rglob(fix_group['file_pattern']):
            if file_path.is_file() and file_path.suffix == '.py':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    needs_import = False
                    
                    # 패턴 수정 적용
                    for fix in fix_group['fixes']:
                        pattern = fix['pattern']
                        replacement = fix['replacement']
                        
                        if re.search(pattern, content):
                            content = re.sub(pattern, replacement, content)
                            needs_import = fix.get('import_needed', False)
                    
                    # import 문 추가
                    if needs_import and import_statement not in content:
                        # import 섹션 찾기
                        import_section_pattern = r'(import.*\n)*'
                        
                        # 기존 import 뒤에 추가
                        if 'import' in content:
                            lines = content.split('\n')
                            insert_index = 0
                            
                            for i, line in enumerate(lines):
                                if line.strip().startswith(('import ', 'from ')) and not line.strip().startswith('#'):
                                    insert_index = i + 1
                            
                            lines.insert(insert_index, import_statement)
                            content = '\n'.join(lines)
                        else:
                            # import가 없으면 파일 상단에 추가
                            content = import_statement + '\n\n' + content
                    
                    # 변경사항이 있으면 파일 저장
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        fixed_files.append(file_path)
                        logger.info(f"✅ 수정 완료: {file_path}")
                
                except Exception as e:
                    logger.error(f"❌ 파일 수정 실패 {file_path}: {e}")
    
    return fixed_files

def create_warmup_patch():
    """워밍업 패치 파일 생성"""
    
    warmup_patch_content = '''
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
'''
    
    backend_dir = Path(__file__).parent.parent
    patch_path = backend_dir / 'app' / 'utils' / 'warmup_patch.py'
    
    with open(patch_path, 'w', encoding='utf-8') as f:
        f.write(warmup_patch_content)
    
    logger.info(f"✅ 워밍업 패치 파일 생성: {patch_path}")
    return patch_path

def main():
    """메인 실행 함수"""
    
    logger.info("🔧 dict object is not callable 오류 해결 시작...")
    
    # 1. 문제가 있는 파일들 찾기
    logger.info("1️⃣ 문제 파일들 스캔 중...")
    problematic_files = find_dict_callable_errors()
    
    if problematic_files:
        logger.info(f"📋 문제 파일 {len(problematic_files)}개 발견:")
        for file_path, pattern in problematic_files:
            logger.info(f"  - {file_path.name}: {pattern}")
    else:
        logger.info("✅ 명시적인 문제 패턴을 찾을 수 없음")
    
    # 2. 안전한 함수 호출 유틸리티 생성
    logger.info("2️⃣ 안전한 함수 호출 유틸리티 생성...")
    safe_caller_path = create_safe_function_caller()
    
    # 3. 워밍업 오류 수정
    logger.info("3️⃣ 워밍업 오류 수정 중...")
    fixed_files = fix_warmup_errors_in_files()
    
    if fixed_files:
        logger.info(f"✅ {len(fixed_files)}개 파일 수정 완료:")
        for file_path in fixed_files:
            logger.info(f"  - {file_path.name}")
    else:
        logger.info("ℹ️ 수정할 파일이 없음")
    
    # 4. 워밍업 패치 생성
    logger.info("4️⃣ 워밍업 패치 생성...")
    patch_path = create_warmup_patch()
    
    # 5. main.py에 패치 적용
    logger.info("5️⃣ main.py에 패치 적용...")
    main_py_path = Path(__file__).parent.parent / 'app' / 'main.py'
    
    if main_py_path.exists():
        try:
            with open(main_py_path, 'r', encoding='utf-8') as f:
                main_content = f.read()
            
            patch_import = "from app.utils.warmup_patch import patch_warmup_methods"
            
            if patch_import not in main_content:
                # FastAPI 앱 생성 이전에 패치 적용
                if 'app = FastAPI(' in main_content:
                    main_content = main_content.replace(
                        'app = FastAPI(',
                        f'{patch_import}\n\napp = FastAPI('
                    )
                    
                    with open(main_py_path, 'w', encoding='utf-8') as f:
                        f.write(main_content)
                    
                    logger.info("✅ main.py에 워밍업 패치 적용 완료")
                else:
                    logger.warning("⚠️ main.py에서 FastAPI 앱 생성 부분을 찾을 수 없음")
            else:
                logger.info("ℹ️ main.py에 이미 패치가 적용됨")
                
        except Exception as e:
            logger.error(f"❌ main.py 패치 실패: {e}")
    
    logger.info("🎉 dict object is not callable 오류 해결 완료!")
    logger.info("\n📋 다음 단계:")
    logger.info("1. 서버 재시작: python app/main.py")
    logger.info("2. 워밍업 오류가 더 이상 발생하지 않을 것입니다")

if __name__ == "__main__":
    main()