#!/usr/bin/env python3
"""
대안 graphonomy 파일 찾기 및 테스트
"""

import os
from pathlib import Path
import torch

def find_alternative_graphonomy():
    """다른 graphonomy 관련 파일들 찾기"""
    print("🔍 대안 graphonomy 파일 찾기")
    print("=" * 50)
    
    current_dir = Path.cwd()
    ai_models_dir = current_dir / "ai_models"
    
    # graphonomy 관련 모든 파일 찾기
    graphonomy_files = []
    
    if ai_models_dir.exists():
        # 다양한 이름으로 검색
        search_patterns = [
            "*graphonomy*",
            "*parsing*",
            "*schp*",
            "*human*parsing*",
            "*lip*",
            "*atr*"
        ]
        
        for pattern in search_patterns:
            for file_path in ai_models_dir.rglob(pattern):
                if file_path.is_file() and file_path.suffix in ['.pth', '.bin', '.safetensors']:
                    graphonomy_files.append(file_path)
    
    # 중복 제거 및 크기 정보 추가
    unique_files = {}
    for file_path in graphonomy_files:
        key = str(file_path.resolve())
        if key not in unique_files:
            size_mb = file_path.stat().st_size / (1024**2)
            unique_files[key] = {
                'path': file_path,
                'size_mb': size_mb
            }
    
    # 크기별로 정렬
    sorted_files = sorted(unique_files.values(), key=lambda x: x['size_mb'], reverse=True)
    
    print(f"📊 발견된 파일들 ({len(sorted_files)}개):")
    for i, file_info in enumerate(sorted_files[:10]):  # 상위 10개만
        file_path = file_info['path']
        size_mb = file_info['size_mb']
        
        # 파일 상태 확인
        status = "🔥 대형" if size_mb > 100 else "📦 중형" if size_mb > 10 else "📄 소형"
        
        print(f"   {i+1}. {file_path.name}")
        print(f"      경로: {file_path}")
        print(f"      크기: {size_mb:.1f}MB {status}")
        
        # 로딩 테스트
        if size_mb > 10:  # 10MB 이상만 테스트
            test_result = test_file_loading(file_path)
            print(f"      로딩: {test_result}")
        
        print()
    
    # 권장사항 제시
    print("💡 권장사항:")
    
    # 가장 큰 파일이 현재 문제 파일인지 확인
    if sorted_files:
        largest_file = sorted_files[0]
        if "graphonomy.pth" in str(largest_file['path']):
            print("   1. 현재 파일이 손상된 것 같습니다")
            
            # 두 번째로 큰 파일 찾기
            for file_info in sorted_files[1:]:
                if file_info['size_mb'] > 100:
                    alternative_path = file_info['path']
                    print(f"   2. 대안 파일 사용 권장: {alternative_path.name}")
                    print(f"      → 이 파일을 graphonomy.pth로 복사하세요")
                    
                    # 복사 명령어 제시
                    source = alternative_path
                    target = current_dir / "ai_models" / "step_01_human_parsing" / "graphonomy_backup.pth"
                    print(f"   3. 복사 명령어:")
                    print(f"      cp '{source}' '{target}'")
                    break
            
        print("   4. 또는 온라인에서 새로운 graphonomy.pth 다운로드")
        print("      - Hugging Face: https://huggingface.co/models?search=graphonomy")
        print("      - GitHub: https://github.com/Engineering-Course/LIP_SSL")
    
    return sorted_files

def test_file_loading(file_path: Path) -> str:
    """파일 로딩 테스트"""
    try:
        # 빠른 헤더 체크
        with open(file_path, 'rb') as f:
            header = f.read(100)
        
        # ZIP 형식 확인
        if header.startswith(b'PK'):
            return "✅ ZIP 형식 (PyTorch 표준)"
        elif b'torch' in header[:50]:
            return "✅ Torch 형식"
        elif header.startswith(b'\x80'):
            return "✅ Pickle 형식"
        else:
            return "❓ 알 수 없는 형식"
            
    except Exception as e:
        return f"❌ 테스트 실패: {e}"

if __name__ == "__main__":
    find_alternative_graphonomy()