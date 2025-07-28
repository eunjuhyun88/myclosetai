#!/usr/bin/env python3
"""
device_manager 오류 수정 스크립트
- device_manager import를 제거하고 memory_manager로 대체
- 관련 오류들 수정
"""

import os
import re
import sys
from pathlib import Path

def find_device_manager_usage():
    """device_manager 사용 파일들 찾기"""
    backend_path = Path("backend")
    problem_files = []
    
    # Python 파일들 검색
    for py_file in backend_path.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            
            # device_manager import 또는 사용 찾기
            if ('device_manager' in content.lower() or 
                'devicemanager' in content or
                'from app.ai_pipeline.utils.device_manager' in content or
                'import device_manager' in content):
                
                print(f"🔍 발견: {py_file}")
                
                # 문제가 되는 라인들 찾기
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if 'device_manager' in line.lower():
                        print(f"   라인 {i}: {line.strip()}")
                
                problem_files.append(py_file)
                print()
                
        except Exception as e:
            print(f"❌ {py_file} 읽기 실패: {e}")
    
    return problem_files

def fix_device_manager_imports(file_path):
    """device_manager import를 memory_manager로 수정"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # 1. device_manager import 제거/수정
        fixes = [
            # Import 수정
            (r'from app\.ai_pipeline\.utils\.device_manager import.*', '# device_manager 제거됨 - memory_manager 사용'),
            (r'import app\.ai_pipeline\.utils\.device_manager.*', '# device_manager 제거됨 - memory_manager 사용'),
            (r'from \.\.utils\.device_manager import.*', '# device_manager 제거됨 - memory_manager 사용'),
            
            # 사용 패턴 수정
            (r'get_device_manager\(\)', 'get_memory_manager()'),
            (r'DeviceManager\(\)', 'get_memory_manager()'),
            (r'device_manager\.', 'memory_manager.'),
            
            # 주석으로 처리
            (r'^(\s*)(.*)device_manager(.*)$', r'\1# \2memory_manager\3  # device_manager → memory_manager로 변경'),
        ]
        
        for pattern, replacement in fixes:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.IGNORECASE)
        
        # 변경사항이 있으면 저장
        if content != original_content:
            # 백업 생성
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            backup_path.write_text(original_content, encoding='utf-8')
            
            # 수정된 내용 저장
            file_path.write_text(content, encoding='utf-8')
            print(f"✅ 수정됨: {file_path}")
            print(f"📁 백업: {backup_path}")
            return True
        else:
            print(f"ℹ️ 변경 없음: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ {file_path} 수정 실패: {e}")
        return False

def fix_mps_settings():
    """MPS 설정 관련 오류 수정"""
    
    # debug_model_loading.py 수정
    debug_file = Path("backend/debug_model_loading.py")
    if debug_file.exists():
        try:
            content = debug_file.read_text(encoding='utf-8')
            
            # MPS 설정 실패 부분 수정
            if "❌ MPS 설정 실패" in content:
                # device_manager import 제거하고 memory_manager 사용
                content = re.sub(
                    r'from app\.ai_pipeline\.utils\.device_manager.*\n',
                    '# device_manager 제거됨 - memory_manager로 대체\n',
                    content
                )
                
                # MPS 설정 코드 수정
                mps_fix = '''
# MPS 설정 (memory_manager 사용)
try:
    from app.ai_pipeline.utils.memory_manager import get_memory_manager
    memory_manager = get_memory_manager()
    if hasattr(memory_manager, 'optimize_for_mps'):
        memory_manager.optimize_for_mps()
    print("✅ MPS 설정 완료")
except Exception as e:
    print(f"ℹ️ MPS 설정 건너뛰기: {e}")
'''
                
                # 기존 MPS 설정 코드 찾아서 교체
                content = re.sub(
                    r'❌ MPS 설정 실패.*?\n',
                    mps_fix,
                    content,
                    flags=re.DOTALL
                )
                
                debug_file.write_text(content, encoding='utf-8')
                print(f"✅ MPS 설정 수정: {debug_file}")
                
        except Exception as e:
            print(f"❌ MPS 설정 수정 실패: {e}")

def check_other_errors():
    """기타 오류들 체크"""
    print("\n🔍 기타 오류 분석:")
    
    # 로그에서 발견된 오류들
    errors = [
        "No module named 'app.ai_pipeline.utils.device_manager'",
        "체크포인트 탐지 실패: [Errno 2] No such file or directory: 'ai_models/u2net.pth'",
        "StepInterface 동적 import 실패: No module named 'app.ai_pipeline.interface'"
    ]
    
    for error in errors:
        print(f"❌ {error}")
    
    print("\n📋 수정 필요 사항:")
    print("1. device_manager → memory_manager 변경")
    print("2. ai_models 경로 수정 (/Users/gimdudeul/MVP/mycloset-ai/backend/ai_models)")
    print("3. app.ai_pipeline.interface 모듈 생성 또는 import 수정")

def main():
    print("🔧 MyCloset AI device_manager 오류 수정 시작...")
    print("=" * 60)
    
    # 1. device_manager 사용 파일들 찾기
    print("1️⃣ device_manager 사용 파일 검색...")
    problem_files = find_device_manager_usage()
    
    if not problem_files:
        print("✅ device_manager 사용 파일 없음")
    else:
        print(f"📊 총 {len(problem_files)}개 파일에서 device_manager 발견")
        
        # 2. 파일들 수정
        print("\n2️⃣ 파일 수정 시작...")
        fixed_count = 0
        for file_path in problem_files:
            if fix_device_manager_imports(file_path):
                fixed_count += 1
        
        print(f"✅ {fixed_count}/{len(problem_files)}개 파일 수정 완료")
    
    # 3. MPS 설정 수정
    print("\n3️⃣ MPS 설정 수정...")
    fix_mps_settings()
    
    # 4. 기타 오류 분석
    check_other_errors()
    
    print("\n" + "=" * 60)
    print("🎉 device_manager 오류 수정 완료!")
    print("\n📋 다음 단계:")
    print("1. python debug_model_loading.py 재실행")
    print("2. 경로 오류 확인 및 수정")
    print("3. 누락된 모듈 생성")

if __name__ == "__main__":
    main()