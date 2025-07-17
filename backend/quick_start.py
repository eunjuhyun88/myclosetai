#!/usr/bin/env python3
"""
MyCloset AI 빠른 실행 스크립트
"""

import sys
import os
from pathlib import Path

# 경로 설정
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print("🚀 MyCloset AI 서버 시작")
print(f"📁 작업 디렉토리: {current_dir}")

# M3 Max 환경변수 설정
os.environ.update({
    'PYTORCH_ENABLE_MPS_FALLBACK': '1',
    'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
    'OMP_NUM_THREADS': '16'
})

try:
    print("📋 Import 테스트 중...")
    
    # 기본 import 테스트
    try:
        from app.api.health import router
        print("✅ Health API 로드됨")
    except Exception as e:
        print(f"⚠️ Health API 로드 실패: {e}")
    
    try:
        import app.main
        print("✅ Main 모듈 로드 성공")
    except Exception as e:
        print(f"❌ Main 모듈 로드 실패: {e}")
        sys.exit(1)
    
    # 서버 실행
    print("🌐 서버 시작 중...")
    exec(open('app/main.py').read())
    
except KeyboardInterrupt:
    print("\n🛑 서버가 종료되었습니다")
except Exception as e:
    print(f"❌ 서버 실행 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
