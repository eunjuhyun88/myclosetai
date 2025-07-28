#!/usr/bin/env python3
"""
경로 디버깅 스크립트 - graphonomy.pth 파일 찾기
"""

import os
from pathlib import Path

def find_graphonomy_file():
    """graphonomy.pth 파일 위치 찾기"""
    print("🔍 graphonomy.pth 파일 찾기")
    print("=" * 50)
    
    # 현재 작업 디렉토리
    current_dir = Path.cwd()
    print(f"📁 현재 작업 디렉토리: {current_dir}")
    
    # 가능한 경로들
    possible_paths = [
        current_dir / "ai_models" / "step_01_human_parsing" / "graphonomy.pth",
        current_dir / "../ai_models" / "step_01_human_parsing" / "graphonomy.pth",
        current_dir / "../../ai_models" / "step_01_human_parsing" / "graphonomy.pth",
        current_dir.parent / "ai_models" / "step_01_human_parsing" / "graphonomy.pth",
        current_dir.parent.parent / "ai_models" / "step_01_human_parsing" / "graphonomy.pth",
    ]
    
    # ai_models 디렉토리 찾기
    ai_models_dirs = []
    for path in possible_paths:
        ai_models_dir = path.parent.parent
        if ai_models_dir.exists() and ai_models_dir.name == "ai_models":
            ai_models_dirs.append(ai_models_dir)
    
    print(f"📂 발견된 ai_models 디렉토리들:")
    for i, ai_dir in enumerate(set(ai_models_dirs)):
        print(f"   {i+1}. {ai_dir.resolve()}")
    
    # graphonomy.pth 파일 직접 찾기
    found_files = []
    search_dirs = [current_dir, current_dir.parent, current_dir.parent.parent]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            for file_path in search_dir.rglob("graphonomy.pth"):
                found_files.append(file_path.resolve())
    
    print(f"\n🎯 발견된 graphonomy.pth 파일들:")
    for i, file_path in enumerate(found_files):
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024**2)
            print(f"   {i+1}. {file_path} ({size_mb:.1f}MB)")
    
    # 최적 경로 추천
    if found_files:
        best_file = found_files[0]
        relative_path = best_file.relative_to(current_dir) if current_dir in best_file.parents else best_file
        print(f"\n✅ 권장 경로: {relative_path}")
        
        # 코드 생성
        print(f"\n📝 수정할 코드:")
        print(f'self.ai_models_root = Path("{relative_path.parent.parent}")')
        
        return str(relative_path.parent.parent)
    else:
        print("❌ graphonomy.pth 파일을 찾을 수 없습니다!")
        return None

if __name__ == "__main__":
    find_graphonomy_file()