#!/usr/bin/env python3
"""
graphonomy.pth 파일 완전성 검사
"""

import os
from pathlib import Path
import zipfile

def check_graphonomy_integrity():
    """graphonomy.pth 파일 완전성 검사"""
    print("🔍 graphonomy.pth 파일 완전성 검사")
    print("=" * 50)
    
    file_path = Path("ai_models/step_01_human_parsing/graphonomy.pth")
    
    if not file_path.exists():
        print("❌ 파일이 존재하지 않습니다")
        return False
    
    file_size = file_path.stat().st_size
    print(f"📊 파일 크기: {file_size / (1024**2):.1f}MB")
    
    # 파일 헤더 검사
    try:
        with open(file_path, 'rb') as f:
            header = f.read(100)
            print(f"📋 파일 헤더: {header[:20].hex()}")
            
            # ZIP 파일인지 확인
            if header.startswith(b'PK'):
                print("✅ ZIP 형식 감지")
                
                # ZIP 파일로 열어보기
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        file_list = zip_ref.namelist()
                        print(f"📦 ZIP 내용: {len(file_list)}개 파일")
                        for i, name in enumerate(file_list[:10]):  # 처음 10개만
                            print(f"   {i+1}. {name}")
                        
                        # version 파일 확인
                        if 'version' in file_list:
                            print("✅ version 레코드 발견")
                            version_data = zip_ref.read('version')
                            print(f"📋 버전: {version_data}")
                        else:
                            print("❌ version 레코드 없음 - 파일 손상")
                            return False
                            
                except zipfile.BadZipFile:
                    print("❌ 손상된 ZIP 파일")
                    return False
                    
            else:
                print("⚠️ ZIP 형식이 아님")
                return False
                
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")
        return False
    
    print("✅ 파일 완전성 검사 완료")
    return True

def suggest_alternatives():
    """대안 제시"""
    print("\n💡 해결 방안:")
    print("1. 파일이 손상된 것 같습니다")
    print("2. 다음 중 하나를 시도하세요:")
    print()
    print("🔄 방법 1: 대안 파일 사용")
    print("   → graphonomy_alternative.pth (104MB) 사용")
    print("   → 이미 시스템에서 정상 작동 중")
    print()
    print("🔄 방법 2: 새 파일 다운로드")
    print("   → 원본 소스에서 다시 다운로드")
    print()
    print("🔄 방법 3: 현재 상태 유지 (권장)")
    print("   → 5개 모델이 이미 정상 작동 중")
    print("   → 추가 조치 불필요")

if __name__ == "__main__":
    integrity_ok = check_graphonomy_integrity()
    if not integrity_ok:
        suggest_alternatives()