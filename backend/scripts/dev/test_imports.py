#!/usr/bin/env python3
"""Import 체인 테스트"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

print("🔍 Import 체인 테스트 시작...")

def test_import(module_name, description):
    try:
        exec(f"import {module_name}")
        print(f"✅ {description}: 성공")
        return True
    except Exception as e:
        print(f"❌ {description}: 실패 - {e}")
        return False

# 기본 라이브러리 테스트
test_import("torch", "PyTorch")
test_import("fastapi", "FastAPI")
test_import("PIL", "PIL/Pillow")
test_import("cv2", "OpenCV")

# 프로젝트 모듈 테스트
test_import("app.core.config", "Core Config")
test_import("app.models.schemas", "Data Schemas")

print("\n🎯 Import 테스트 완료!")
