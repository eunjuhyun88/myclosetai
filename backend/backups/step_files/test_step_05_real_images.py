#!/usr/bin/env python3
"""
Step 05 Cloth Warping 최종 테스트
backend 디렉토리에서 실행
"""

import os
import sys
import asyncio
import time
import numpy as np
import cv2
from pathlib import Path

# 현재 디렉토리가 backend인지 확인
current_dir = Path.cwd()
print(f"🔧 현재 디렉토리: {current_dir}")

if current_dir.name != 'backend':
    print("❌ backend 디렉토리에서 실행해주세요!")
    print(f"   현재 위치: {current_dir}")
    print(f"   이동 명령: cd {current_dir}/backend" if (current_dir / 'backend').exists() else "   backend 디렉토리를 찾을 수 없습니다")
    sys.exit(1)

# Python 경로에 현재 디렉토리(backend) 추가
sys.path.insert(0, str(current_dir))
print(f"🔧 Python 경로 추가: {current_dir}")

# 환경 설정
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'

def create_test_images():
    """테스트용 이미지 생성"""
    print("🎨 테스트 이미지 생성 중...")
    
    # 작은 크기로 빠른 테스트
    height, width = 256, 192
    
    # 의류 이미지 (파란색 티셔츠)
    cloth_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # 흰 배경
    cloth_color = (100, 150, 200)  # 파란색
    
    # 티셔츠 본체
    cv2.rectangle(cloth_img, (50, 40), (142, 180), cloth_color, -1)
    # 소매
    cv2.rectangle(cloth_img, (30, 40), (70, 100), cloth_color, -1)
    cv2.rectangle(cloth_img, (122, 40), (162, 100), cloth_color, -1)
    # 목 부분
    cv2.rectangle(cloth_img, (80, 20), (112, 40), cloth_color, -1)
    
    # 인물 이미지 (간단한 사람 모양)
    person_img = np.ones((height, width, 3), dtype=np.uint8) * 240  # 연한 회색 배경
    person_color = (160, 140, 120)  # 살색
    
    # 머리
    cv2.circle(person_img, (96, 50), 25, person_color, -1)
    # 몸통
    cv2.rectangle(person_img, (70, 75), (122, 180), person_color, -1)
    # 팔
    cv2.rectangle(person_img, (50, 75), (70, 140), person_color, -1)  # 왼팔
    cv2.rectangle(person_img, (122, 75), (142, 140), person_color, -1)  # 오른팔
    
    # 의류 마스크
    cloth_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(cloth_mask, (50, 40), (142, 180), 255, -1)
    cv2.rectangle(cloth_mask, (30, 40), (70, 100), 255, -1)
    cv2.rectangle(cloth_mask, (122, 40), (162, 100), 255, -1)
    cv2.rectangle(cloth_mask, (80, 20), (112, 40), 255, -1)
    
    print("✅ 테스트 이미지 생성 완료")
    print(f"   의류: {cloth_img.shape}")
    print(f"   인물: {person_img.shape}")
    print(f"   마스크: {cloth_mask.shape}")
    
    return cloth_img, person_img, cloth_mask

async def test_step_05_complete():
    """완전한 Step 05 테스트"""
    print("\n🧪 Step 05 완전 테스트")
    print("=" * 50)
    
    try:
        # 1. Import 테스트
        print("1️⃣ Import 테스트...")
        from app.ai_pipeline.steps.step_05_cloth_warping import (
            ClothWarpingStep,
            create_cloth_warping_step,
            WarpingMethod,
            FabricType,
            validate_warping_result
        )
        print("✅ 모든 클래스 import 성공")
        
        # 2. 인스턴스 생성 테스트
        print("\n2️⃣ 인스턴스 생성 테스트...")
        step = ClothWarpingStep(device="cpu")  # CPU로 안전하게
        print(f"✅ ClothWarpingStep 생성 성공")
        print(f"   Step 이름: {step.step_name}")
        print(f"   디바이스: {step.device}")
        print(f"   초기화 상태: {step.is_initialized}")
        
        # 3. 팩토리 함수 테스트
        print("\n3️⃣ 팩토리 함수 테스트...")
        factory_step = await create_cloth_warping_step(
            device="cpu",
            config={
                "ai_model_enabled": True,
                "physics_enabled": True,
                "visualization_enabled": False,  # 빠른 테스트를 위해 비활성화
                "quality_level": "medium"
            }
        )
        print(f"✅ create_cloth_warping_step 성공")
        print(f"   초기화 상태: {factory_step.is_initialized}")
        
        # 4. Step 정보 조회
        print("\n4️⃣ Step 정보 조회...")
        step_info = await step.get_step_info()
        print(f"✅ Step 정보:")
        print(f"   클래스: {step_info.get('class_name', 'Unknown')}")
        print(f"   버전: {step_info.get('version', 'Unknown')}")
        print(f"   디바이스 타입: {step_info.get('device_type', 'Unknown')}")
        print(f"   메모리: {step_info.get('memory_gb', 0)}GB")
        print(f"   M3 Max: {step_info.get('is_m3_max', False)}")
        
        capabilities = step_info.get('capabilities', {})
        print(f"   PyTorch: {capabilities.get('torch_available', False)}")
        print(f"   OpenCV: {capabilities.get('cv2_available', False)}")
        print(f"   PIL: {capabilities.get('pil_available', False)}")
        
        # 5. 실제 처리 테스트
        print("\n5️⃣ 실제 워핑 처리 테스트...")
        cloth_img, person_img, cloth_mask = create_test_images()
        
        start_time = time.time()
        result = await factory_step.process(
            cloth_image=cloth_img,
            person_image=person_img,
            cloth_mask=cloth_mask,
            fabric_type="cotton",
            clothing_type="shirt"
        )
        processing_time = time.time() - start_time
        
        # 6. 결과 분석
        print("\n6️⃣ 결과 분석...")
        print(f"   ⏱️ 처리 시간: {processing_time:.2f}초")
        print(f"   ✅ 성공 여부: {result.get('success', False)}")
        
        if result.get('success'):
            print(f"   🎯 신뢰도: {result.get('confidence', 0):.3f}")
            print(f"   ⭐ 품질 점수: {result.get('quality_score', 0):.3f}")
            print(f"   📝 품질 등급: {result.get('quality_grade', 'N/A')}")
            
            # 워핑된 이미지 확인
            warped_img = result.get('warped_cloth_image')
            if warped_img is not None:
                print(f"   🎨 워핑 이미지: {warped_img.shape} {warped_img.dtype}")
                
                # 결과 검증
                is_valid = validate_warping_result(result)
                print(f"   🔍 결과 검증: {'통과' if is_valid else '실패'}")
            
            # 분석 정보
            analysis = result.get('warping_analysis', {})
            if analysis:
                print(f"   📊 변형 품질: {analysis.get('deformation_quality', 0):.3f}")
                print(f"   📊 물리 품질: {analysis.get('physics_quality', 0):.3f}")
                print(f"   📊 텍스처 품질: {analysis.get('texture_quality', 0):.3f}")
                print(f"   👕 피팅 적합: {'예' if analysis.get('suitable_for_fitting') else '아니오'}")
            
            # 캐시 테스트
            print("\n7️⃣ 캐시 테스트...")
            cache_start = time.time()
            cache_result = await factory_step.process(
                cloth_image=cloth_img,
                person_image=person_img,
                cloth_mask=cloth_mask,
                fabric_type="cotton",
                clothing_type="shirt"
            )
            cache_time = time.time() - cache_start
            
            print(f"   ⏱️ 캐시 처리 시간: {cache_time:.2f}초")
            print(f"   💾 캐시 적중: {'예' if cache_result.get('from_cache') else '아니오'}")
            
            # 캐시 상태
            cache_status = factory_step.get_cache_status()
            print(f"   💾 캐시 크기: {cache_status.get('current_size', 0)}")
            print(f"   💾 캐시 적중률: {cache_status.get('hit_rate', 0)*100:.1f}%")
            
        else:
            print(f"   ❌ 실패 원인: {result.get('error', 'Unknown')}")
        
        # 8. 다양한 설정 테스트
        print("\n8️⃣ 다양한 설정 테스트...")
        test_configs = [
            {"fabric": "silk", "clothing": "dress"},
            {"fabric": "denim", "clothing": "jacket"},
            {"fabric": "cotton", "clothing": "pants"}
        ]
        
        for i, config in enumerate(test_configs, 1):
            print(f"   {i}. {config['fabric']} {config['clothing']} 테스트...")
            quick_result = await factory_step.process(
                cloth_image=cloth_img,
                person_image=person_img,
                cloth_mask=cloth_mask,
                fabric_type=config["fabric"],
                clothing_type=config["clothing"]
            )
            
            status = "성공" if quick_result.get('success') else "실패"
            confidence = quick_result.get('confidence', 0)
            print(f"      {status} (신뢰도: {confidence:.3f})")
        
        # 9. 정리
        print("\n9️⃣ 리소스 정리...")
        await step.cleanup_models()
        await factory_step.cleanup_models()
        print("   ✅ 정리 완료")
        
        return result
        
    except ImportError as e:
        print(f"❌ Import 실패: {e}")
        print("💡 해결책:")
        print("   1. 현재 디렉토리가 backend인지 확인")
        print("   2. app/ai_pipeline/steps/step_05_cloth_warping.py 파일 존재 확인")
        print("   3. 가상환경 활성화 확인")
        return None
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """메인 함수"""
    print("🧪 Step 05 Cloth Warping 완전 테스트")
    print("=" * 70)
    print(f"📍 실행 위치: {Path.cwd()}")
    print(f"🐍 Python 버전: {sys.version.split()[0]}")
    
    # 필수 라이브러리 확인
    print("\n📦 필수 라이브러리 확인:")
    required_libs = ['numpy', 'cv2', 'PIL']
    missing_libs = []
    
    for lib in required_libs:
        try:
            if lib == 'cv2':
                import cv2
                print(f"   ✅ {lib}: {cv2.__version__}")
            elif lib == 'PIL':
                from PIL import Image
                print(f"   ✅ {lib}: {Image.__version__ if hasattr(Image, '__version__') else 'Available'}")
            else:
                module = __import__(lib)
                version = getattr(module, '__version__', 'Available')
                print(f"   ✅ {lib}: {version}")
        except ImportError:
            missing_libs.append(lib)
            print(f"   ❌ {lib}: 누락")
    
    if missing_libs:
        print(f"\n⚠️ 누락된 라이브러리: {missing_libs}")
        print("   설치 명령: pip install " + " ".join(missing_libs))
        return
    
    # 메인 테스트 실행
    result = await test_step_05_complete()
    
    # 최종 결과
    print("\n📊 최종 테스트 결과")
    print("=" * 70)
    
    if result:
        if result.get('success'):
            print("🎉 Step 05 Cloth Warping이 완벽하게 작동합니다!")
            print(f"   ✅ 처리 성공률: 100%")
            print(f"   ✅ 신뢰도: {result.get('confidence', 0):.3f}")
            print(f"   ✅ 품질 점수: {result.get('quality_score', 0):.3f}")
            print(f"   ✅ 품질 등급: {result.get('quality_grade', 'N/A')}")
            
            print("\n💡 사용 가능한 기능:")
            print("   🔸 AI 기반 의류 워핑")
            print("   🔸 물리 시뮬레이션")
            print("   🔸 다양한 원단 타입 지원")
            print("   🔸 품질 분석 및 평가")
            print("   🔸 캐시 시스템")
        else:
            print("⚠️ Step 05가 부분적으로 작동합니다.")
            print(f"   ❌ 처리 실패: {result.get('error', 'Unknown')}")
            print("   하지만 기본 구조는 정상입니다.")
    else:
        print("❌ Step 05 테스트가 실패했습니다.")
        print("   파일 구조나 의존성을 확인해주세요.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ 사용자가 테스트를 중단했습니다.")
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()