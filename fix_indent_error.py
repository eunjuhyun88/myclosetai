#!/usr/bin/env python3
"""
🔧 MyCloset AI - 인덴테이션 오류 수정 스크립트
ootdiffusion_path_fix.py 및 기타 Python 파일들의 인덴테이션 문제 해결

현재 디렉토리에서 실행하세요: python fix_indent_error.py
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple

def fix_indentation_errors(file_path: Path) -> bool:
    """Python 파일의 인덴테이션 오류 수정"""
    try:
        print(f"🔧 인덴테이션 수정 중: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        fixed_lines = []
        in_function = False
        in_class = False
        expected_indent = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 빈 줄이나 주석은 그대로 유지
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue
            
            # 현재 줄의 들여쓰기 수준 계산
            current_indent = len(line) - len(line.lstrip())
            
            # 함수나 클래스 정의 감지
            if stripped.startswith(('def ', 'class ', 'async def ')):
                if stripped.startswith('class '):
                    in_class = True
                    in_function = False
                    expected_indent = current_indent + 4
                elif stripped.startswith(('def ', 'async def ')):
                    in_function = True
                    if in_class:
                        expected_indent = current_indent + 4
                    else:
                        expected_indent = current_indent + 4
                fixed_lines.append(line)
                continue
            
            # 제어문 감지 (if, for, while, try, except, etc.)
            control_keywords = [
                'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 
                'except:', 'except ', 'finally:', 'with ', 'match ', 'case '
            ]
            
            if any(stripped.startswith(keyword) for keyword in control_keywords):
                # 제어문 다음 줄은 들여쓰기 증가
                if i + 1 < len(lines) and lines[i + 1].strip():
                    next_line = lines[i + 1]
                    next_indent = len(next_line) - len(next_line.lstrip())
                    if next_indent <= current_indent:
                        # 다음 줄의 들여쓰기가 부족한 경우
                        expected_indent = current_indent + 4
                
                fixed_lines.append(line)
                continue
            
            # 일반적인 들여쓰기 오류 수정
            if in_function or in_class:
                # 예상되는 들여쓰기와 현재 들여쓰기 비교
                if current_indent > 0 and current_indent % 4 != 0:
                    # 4의 배수가 아닌 들여쓰기를 4의 배수로 조정
                    corrected_indent = (current_indent // 4 + 1) * 4
                    fixed_line = ' ' * corrected_indent + stripped + '\n'
                    fixed_lines.append(fixed_line)
                    print(f"   수정됨 {i+1}행: {current_indent} → {corrected_indent} 스페이스")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        # 수정된 내용을 파일에 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        
        print(f"✅ {file_path.name} 인덴테이션 수정 완료")
        return True
        
    except Exception as e:
        print(f"❌ {file_path.name} 수정 실패: {e}")
        return False

def find_python_files_with_errors(directory: Path) -> List[Path]:
    """인덴테이션 오류가 있는 Python 파일들 찾기"""
    problematic_files = []
    
    for py_file in directory.rglob("*.py"):
        try:
            # Python 구문 검사
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            compile(content, str(py_file), 'exec')
            
        except IndentationError:
            problematic_files.append(py_file)
        except SyntaxError as e:
            if 'indent' in str(e).lower():
                problematic_files.append(py_file)
        except Exception:
            # 다른 오류는 무시
            pass
    
    return problematic_files

def fix_specific_files():
    """특정 파일들의 인덴테이션 수정"""
    project_root = Path.cwd()
    
    # 수정할 파일들 목록
    files_to_fix = [
        "ootdiffusion_path_fix.py",
        "integrate_virtual_fitting_v2.py",
        "verify_models.py"
    ]
    
    print("🔧 특정 파일들의 인덴테이션 수정")
    print("=" * 40)
    
    for filename in files_to_fix:
        file_path = project_root / filename
        if file_path.exists():
            fix_indentation_errors(file_path)
        else:
            print(f"⚠️ 파일 없음: {filename}")
    
    # 백엔드 디렉토리에서 문제가 있는 파일들 찾기
    backend_dir = project_root / "backend"
    if backend_dir.exists():
        print(f"\n🔍 {backend_dir} 에서 인덴테이션 오류 검사 중...")
        problematic_files = find_python_files_with_errors(backend_dir)
        
        if problematic_files:
            print(f"❌ 인덴테이션 오류 파일 {len(problematic_files)}개 발견:")
            for file_path in problematic_files:
                print(f"   📄 {file_path.relative_to(project_root)}")
                fix_indentation_errors(file_path)
        else:
            print("✅ 백엔드에서 인덴테이션 오류 없음")

def create_quick_syntax_checker():
    """빠른 구문 검사기 생성"""
    checker_script = '''#!/usr/bin/env python3
"""빠른 Python 구문 검사기"""

import sys
from pathlib import Path

def check_syntax(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        compile(content, str(file_path), 'exec')
        print(f"✅ {file_path}")
        return True
    except SyntaxError as e:
        print(f"❌ {file_path}:{e.lineno} - {e.msg}")
        return False
    except Exception as e:
        print(f"⚠️ {file_path} - {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        check_syntax(file_path)
    else:
        print("사용법: python syntax_check.py <파일경로>")
'''
    
    checker_path = Path.cwd() / "syntax_check.py"
    with open(checker_path, 'w', encoding='utf-8') as f:
        f.write(checker_script)
    
    os.chmod(checker_path, 0o755)
    print(f"📝 구문 검사기 생성: {checker_path}")

def main():
    """메인 실행 함수"""
    print("🔧 MyCloset AI 인덴테이션 오류 수정 스크립트")
    print("=" * 50)
    
    # 1. 특정 파일들 수정
    fix_specific_files()
    
    # 2. 빠른 구문 검사기 생성
    print(f"\n📝 도구 생성 중...")
    create_quick_syntax_checker()
    
    print(f"\n✅ 인덴테이션 수정 완료!")
    print(f"\n🚀 다음 단계:")
    print(f"1. python verify_models.py")
    print(f"2. ./fix_conda_env.sh")
    print(f"3. cd backend && python app/main.py")

if __name__ == "__main__":
    main()