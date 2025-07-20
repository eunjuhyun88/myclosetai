
import sys
import os
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"현재 디렉토리: {current_dir}")
print(f"Python 경로: {sys.path[:3]}")  # 처음 3개만 출력

# Step 05 import 테스트
try:
    from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep
    print("✅ ClothWarpingStep import 성공")
    
    step = ClothWarpingStep()
    print(f"✅ Step 생성 성공: {step.step_name}")
    print(f"✅ 디바이스: {step.device}")
    print(f"✅ M3 Max: {step.is_m3_max}")
    
except ImportError as e:
    print(f"❌ Import 실패: {e}")
    
    # 파일 존재 확인
    step_file = Path("app/ai_pipeline/steps/step_05_cloth_warping.py")
    if step_file.exists():
        print(f"✅ 파일 존재: {step_file}")
    else:
        print(f"❌ 파일 없음: {step_file}")
    
    # __init__.py 파일들 확인
    init_files = [
        "app/__init__.py",
        "app/ai_pipeline/__init__.py", 
        "app/ai_pipeline/steps/__init__.py"
    ]
    
    for init_file in init_files:
        if Path(init_file).exists():
            print(f"✅ {init_file} 존재")
        else:
            print(f"❌ {init_file} 없음 - 생성 필요")
EOF

# 2️⃣ 필요한 __init__.py 파일들 생성
echo "# MyCloset AI App" > app/__init__.py
echo "# AI Pipeline" > app/ai_pipeline/__init__.py  
echo "# AI Pipeline Steps" > app/ai_pipeline/steps/__init__.py

# 3️⃣ 다시 import 테스트
python3 << 'EOF'
import sys
import os

# 경로 추가
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep
    print("✅ ClothWarpingStep import 성공")
    
    step = ClothWarpingStep()
    print(f"✅ Logger: {hasattr(step, 'logger')}")
    print(f"✅ 초기화: {step.is_initialized}")
    print(f"✅ Step 이름: {step.step_name}")
    print(f"✅ 디바이스: {step.device}")
    
except Exception as e:
    print(f"❌ 에러: {e}")
    import traceback
    traceback.print_exc()
EOF

# 4️⃣ 완전한 처리 테스트 (경로 수정됨)
python3 << 'EOF'
import sys
import os
import asyncio
import numpy as np
import time

# 경로 추가
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.ai_pipeline.steps.step_05_cloth_warping import ClothWarpingStep

async def complete_test():
    print("🧪 완전한 처리 테스트...")
    
    # Step 생성
    step = ClothWarpingStep(config={
        'ai_model_enabled': True,
        'physics_enabled': True,
        'enable_visualization': True,
        'cache_enabled': True
    })
    
    print(f"📋 Step 정보:")
    print(f"  - 이름: {step.step_name}")
    print(f"  - 디바이스: {step.device}")  
    print(f"  - M3 Max: {step.is_m3_max}")
    print(f"  - 메모리: {step.memory_gb}GB")
    print(f"  - 초기화: {step.is_initialized}")
    
    # 테스트 이미지 (고해상도)
    cloth_img = np.random.randint(0, 255, (512, 384, 3), dtype=np.uint8)
    person_img = np.random.randint(0, 255, (512, 384, 3), dtype=np.uint8)
    
    # 패턴 추가 (더 현실적)
    cloth_img[100:400, 100:284] = [255, 100, 100]  # 빨간 영역
    person_img[50:450, 142:242] = [100, 255, 100]  # 초록 영역
    
    print(f"🎨 이미지 생성: cloth={cloth_img.shape}, person={person_img.shape}")
    
    # 처리 실행
    start_time = time.time()
    try:
        result = await step.process(
            cloth_image=cloth_img,
            person_image=person_img,
            fabric_type="cotton",
            clothing_type="shirt"
        )
        
        processing_time = time.time() - start_time
        print(f"⏱️ 처리 시간: {processing_time:.2f}초")
        
        if result['success']:
            print("🎉 처리 성공!")
            print(f"🎯 신뢰도: {result.get('confidence', 0):.3f}")
            print(f"⭐ 품질 점수: {result.get('quality_score', 0):.3f}")
            print(f"📝 품질 등급: {result.get('quality_grade', 'N/A')}")
            print(f"👗 피팅 적합성: {'✅ 적합' if result.get('suitable_for_fitting') else '❌ 부적합'}")
            
            # 상세 분석
            analysis = result.get('warping_analysis', {})
            if analysis:
                print("📊 워핑 분석 상세:")
                print(f"  - 변형 품질: {analysis.get('deformation_quality', 0):.3f}")
                print(f"  - 물리 품질: {analysis.get('physics_quality', 0):.3f}")
                print(f"  - 텍스처 품질: {analysis.get('texture_quality', 0):.3f}")
                print(f"  - 전체 점수: {analysis.get('overall_score', 0):.3f}")
            
            # 시각화 확인
            vis_size = len(result.get('visualization', '')) if result.get('visualization') else 0
            prog_size = len(result.get('progress_visualization', '')) if result.get('progress_visualization') else 0
            print(f"🎨 시각화: {vis_size} bytes")
            print(f"📈 진행 시각화: {prog_size} bytes")
            
            # 시스템 정보
            device_info = result.get('device_info', {})
            if device_info:
                print("💻 디바이스 정보:")
                print(f"  - 디바이스: {device_info.get('device')}")
                print(f"  - 타입: {device_info.get('device_type')}")
                print(f"  - 최적화: {device_info.get('optimization_level')}")
            
        else:
            print("❌ 처리 실패")
            print(f"에러: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"💥 예외 발생: {e}")
        import traceback
        traceback.print_exc()

# 테스트 실행
asyncio.run(complete_test())
EOF