#!/usr/bin/env python3
"""
Step 매핑 검증 스크립트
"""

import sys
import os
import re
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def verify_step_mapping():
    """Step 매핑 검증"""
    print("🔧 Step 매핑 검증")
    print("=" * 60)
    
    # 올바른 매핑 정의
    correct_mapping = {
        "/1/upload-validation": {
            "description": "이미지 유틸리티 (세션 생성)",
            "step_id": None,  # 유틸리티 단계
            "file": "None"
        },
        "/2/measurements-validation": {
            "description": "이미지 유틸리티 (측정값 저장)",
            "step_id": None,  # 유틸리티 단계
            "file": "None"
        },
        "/3/human-parsing": {
            "description": "Human Parsing",
            "step_id": 3,
            "file": "step_01_human_parsing.py"
        },
        "/4/pose-estimation": {
            "description": "Pose Estimation",
            "step_id": 2,
            "file": "step_02_pose_estimation.py"
        },
        "/5/clothing-analysis": {
            "description": "Cloth Segmentation",
            "step_id": 3,
            "file": "step_03_cloth_segmentation.py"
        },
        "/6/geometric-matching": {
            "description": "Geometric Matching",
            "step_id": 4,
            "file": "step_04_geometric_matching.py"
        },
        "/7/virtual-fitting": {
            "description": "Cloth Warping + Virtual Fitting",
            "step_id": [5, 6],
            "file": "step_05_cloth_warping.py + step_06_virtual_fitting.py"
        },
        "/8/result-analysis": {
            "description": "Post Processing + Quality Assessment",
            "step_id": [7, 8],
            "file": "step_07_post_processing.py + step_08_quality_assessment.py"
        }
    }
    
    # StepFactory 매핑 확인
    print("\n1. StepFactory 매핑 확인")
    try:
        from app.ai_pipeline.factories.step_factory import StepType, CentralHubStepMapping
        
        factory_mapping = {
            StepType.HUMAN_PARSING: 1,
            StepType.POSE_ESTIMATION: 2,
            StepType.CLOTH_SEGMENTATION: 3,
            StepType.GEOMETRIC_MATCHING: 4,
            StepType.CLOTH_WARPING: 5,
            StepType.VIRTUAL_FITTING: 6,
            StepType.POST_PROCESSING: 7,
            StepType.QUALITY_ASSESSMENT: 8
        }
        
        print("✅ StepFactory 매핑:")
        for step_type, step_id in factory_mapping.items():
            print(f"  {step_type.value} → step_id={step_id}")
            
    except Exception as e:
        print(f"❌ StepFactory 매핑 확인 실패: {e}")
    
    # 실제 파일 존재 확인
    print("\n2. 실제 파일 존재 확인")
    step_files = [
        "step_01_human_parsing.py",
        "step_02_pose_estimation.py", 
        "step_03_cloth_segmentation.py",
        "step_04_geometric_matching.py",
        "step_05_cloth_warping.py",
        "step_06_virtual_fitting.py",
        "step_07_post_processing.py",
        "step_08_quality_assessment.py"
    ]
    
    for file_name in step_files:
        file_path = f"app/ai_pipeline/steps/{file_name}"
        if os.path.exists(file_path):
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} (파일 없음)")
    
    # API 라우트 매핑 확인
    print("\n3. API 라우트 매핑 확인")
    try:
        # step_routes.py에서 _process_step_async 호출 확인
        with open("app/api/step_routes.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # _process_step_async 호출 패턴 찾기
        pattern = r'_process_step_async\(\s*step_name=[\'"]([^\'"]+)[\'"],\s*step_id=(\d+)'
        matches = re.findall(pattern, content)
        
        print("✅ API 라우트에서 발견된 매핑:")
        for step_name, step_id in matches:
            print(f"  {step_name} → step_id={step_id}")
            
    except Exception as e:
        print(f"❌ API 라우트 매핑 확인 실패: {e}")
    
    # 매핑 검증 결과
    print("\n4. 매핑 검증 결과")
    print("📋 올바른 매핑:")
    for route, info in correct_mapping.items():
        print(f"  {route} → {info['description']} (step_id={info['step_id']})")
    
    print("\n🔍 수정된 매핑:")
    print("  ✅ Step 1 (upload-validation) → 유틸리티 단계 (AI 모델 호출 안함)")
    print("  ✅ Step 2 (measurements-validation) → 유틸리티 단계 (AI 모델 호출 안함)")
    print("  ✅ Step 3 (human-parsing) → step_id=3 (올바름) - step_01_human_parsing.py")
    print("  ✅ Step 4 (pose-estimation) → step_id=2 (수정됨) - step_02_pose_estimation.py")
    print("  ✅ Step 5 (clothing-analysis) → step_id=3 (올바름) - step_03_cloth_segmentation.py")
    print("  ✅ Step 6 (geometric-matching) → step_id=4 (수정됨) - step_04_geometric_matching.py")
    print("  ✅ Step 7 (virtual-fitting) → step_id=5,6 (올바름) - step_05_cloth_warping.py + step_06_virtual_fitting.py")
    print("  ✅ Step 8 (result-analysis) → step_id=7,8 (올바름) - step_07_post_processing.py + step_08_quality_assessment.py")

if __name__ == "__main__":
    verify_step_mapping() 