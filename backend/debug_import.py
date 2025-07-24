# backend/debug_import.py
"""
Import 오류 정확히 진단하는 스크립트
"""

import sys
import traceback
from pathlib import Path

# 프로젝트 루트 경로 추가
backend_root = Path(__file__).parent
sys.path.insert(0, str(backend_root))

print("🔍 Import 오류 디버그 시작...")
print(f"📁 작업 디렉토리: {Path.cwd()}")
print(f"📁 backend 루트: {backend_root}")
print(f"🐍 Python 경로: {sys.path[:3]}...")

# 1. 기본 모듈들 테스트
print("\n1️⃣ 기본 모듈 테스트:")
try:
    import torch
    print(f"✅ torch {torch.__version__}")
except ImportError as e:
    print(f"❌ torch: {e}")

try:
    import numpy as np
    print(f"✅ numpy {np.__version__}")
except ImportError as e:
    print(f"❌ numpy: {e}")

try:
    from PIL import Image
    print(f"✅ PIL")
except ImportError as e:
    print(f"❌ PIL: {e}")

# 2. app 패키지 테스트
print("\n2️⃣ app 패키지 테스트:")
try:
    import app
    print(f"✅ app 패키지: {app}")
except ImportError as e:
    print(f"❌ app 패키지: {e}")
    print(f"📋 상세 오류:\n{traceback.format_exc()}")

# 3. ai_pipeline 패키지 테스트
print("\n3️⃣ ai_pipeline 패키지 테스트:")
try:
    import app.ai_pipeline
    print(f"✅ app.ai_pipeline: {app.ai_pipeline}")
except ImportError as e:
    print(f"❌ app.ai_pipeline: {e}")
    print(f"📋 상세 오류:\n{traceback.format_exc()}")

# 4. pipeline_manager 파일 존재 확인
print("\n4️⃣ pipeline_manager 파일 확인:")
pipeline_manager_path = backend_root / "app" / "ai_pipeline" / "pipeline_manager.py"
print(f"📁 경로: {pipeline_manager_path}")
print(f"📄 존재: {pipeline_manager_path.exists()}")

if pipeline_manager_path.exists():
    try:
        file_size = pipeline_manager_path.stat().st_size
        print(f"📊 파일 크기: {file_size:,} bytes")
        
        # 파일 내용 확인 (첫 10줄)
        with open(pipeline_manager_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]
            print(f"📝 첫 10줄:")
            for i, line in enumerate(lines, 1):
                print(f"   {i:2d}: {line.strip()}")
    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")

# 5. PipelineManager import 테스트
print("\n5️⃣ PipelineManager import 테스트:")
try:
    from app.ai_pipeline.pipeline_manager import PipelineManager
    print(f"✅ PipelineManager 클래스: {PipelineManager}")
    print(f"✅ PipelineManager 타입: {type(PipelineManager)}")
except ImportError as e:
    print(f"❌ PipelineManager import 실패: {e}")
    print(f"📋 상세 오류:\n{traceback.format_exc()}")
except Exception as e:
    print(f"❌ PipelineManager 기타 오류: {e}")
    print(f"📋 상세 오류:\n{traceback.format_exc()}")

# 6. 개별 클래스들 import 테스트
print("\n6️⃣ 개별 클래스 import 테스트:")
classes_to_test = [
    'PipelineManager',
    'DIBasedPipelineManager', 
    'PipelineConfig',
    'ProcessingResult',
    'QualityLevel',
    'PipelineMode',
    'create_pipeline',
    'create_m3_max_pipeline'
]

for class_name in classes_to_test:
    try:
        exec(f"from app.ai_pipeline.pipeline_manager import {class_name}")
        print(f"✅ {class_name}")
    except ImportError as e:
        print(f"❌ {class_name}: {e}")
    except Exception as e:
        print(f"⚠️ {class_name}: {e}")

print("\n🏁 Import 디버그 완료!")