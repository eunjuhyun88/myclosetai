#!/usr/bin/env python3
"""
PIL.Image.VERSION을 PIL.__version__으로 자동 수정하는 스크립트
"""

import os
import re
import shutil
from pathlib import Path

def fix_pil_version_in_file(file_path):
    """파일에서 PIL.Image.VERSION을 수정"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # PIL.Image.VERSION -> PIL.__version__ 으로 변경
        patterns_to_fix = [
            (r'PIL\.Image\.VERSION', 'PIL.__version__'),
            (r'Image\.VERSION', 'PIL.__version__'),
            (r'from PIL import.*Image.*\n.*Image\.VERSION', 'import PIL\nPIL.__version__'),
        ]
        
        changes_made = []
        for pattern, replacement in patterns_to_fix:
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                changes_made.append(f"{pattern} -> {replacement}")
        
        if content != original_content:
            # 백업 생성
            backup_path = f"{file_path}.backup_pil_fix"
            shutil.copy2(file_path, backup_path)
            
            # 수정된 내용 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ 수정됨: {file_path}")
            print(f"📝 백업: {backup_path}")
            for change in changes_made:
                print(f"   - {change}")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ 오류: {file_path} - {e}")
        return False

def main():
    """메인 함수"""
    print("🔧 PIL.Image.VERSION 자동 수정 시작...")
    
    fixed_files = []
    
    # Python 파일들 검색 및 수정
    for py_file in Path('.').rglob('*.py'):
        if fix_pil_version_in_file(py_file):
            fixed_files.append(str(py_file))
    
    print(f"\n🎉 수정 완료: {len(fixed_files)}개 파일")
    for file in fixed_files:
        print(f"  - {file}")
    
    if fixed_files:
        print("\n⚠️  백업 파일들이 생성되었습니다.")
        print("수정 후 테스트가 완료되면 백업 파일들을 삭제하세요:")
        print("find . -name '*.backup_pil_fix' -delete")

if __name__ == "__main__":
    main()
