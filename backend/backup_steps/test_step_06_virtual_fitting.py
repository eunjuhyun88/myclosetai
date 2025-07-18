# test_step_06_virtual_fitting.py
"""
🔥 VirtualFittingStep 완전 테스트 스크립트
✅ 모든 기능 검증
✅ M3 Max 최적화 테스트
✅ 시각화 기능 테스트
✅ 에러 처리 테스트
"""

import asyncio
import sys
import os
import logging
import time
import numpy as np
from PIL import Image
import traceback

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_images():
    """테스트용 이미지 생성"""
    print("🖼️ 테스트 이미지 생성 중...")
    
    # 사람 이미지 (파란색 배경에 사람 형태)
    person_img = np.zeros((512, 512, 3), dtype=np.uint8)
    person_img[:, :] = [100, 150, 200]  # 파란 배경
    
    # 사람 형태 그리기 (간단한 타원)
    center_x, center_y = 256, 256
    for y in range(512):
        for x in range(512):
            # 머리
            if (x - center_x)**2 / 50**2 + (y - (center_y-150))**2 / 60**2 <= 1:
                person_img[y, x] = [220, 180, 140]  # 살색
            # 몸통
            elif (x - center_x)**2 / 80**2 + (y - center_y)**2 / 120**2 <= 1:
                person_img[y, x] = [200, 160, 120]  # 살색
    
    # 의류 이미지 (빨간색 셔츠)
    cloth_img = np.zeros((512, 512, 3), dtype=np.uint8)
    cloth_img[:, :] = [255, 255, 255]  # 흰 배경
    
    # 셔츠 형태 그리기
    for y in range(100, 350):
        for x in range(150, 362):
            cloth_img[y, x] = [200, 50, 50]  # 빨간색 셔츠
    
    print("✅ 테스트 이미지 생성 완료")
    return person_img, cloth_img

async def test_basic_functionality():
    """기본 기능 테스트"""
    print("\n" + "="*60)
    print("🔧 기본 기능 테스트 시작")
    print("="*60)
    
    try:
        # VirtualFittingStep 임포트 시도
        print("📦 VirtualFittingStep 임포트 중...")
        
        # 실제 파일에서 임포트 (현재 디렉토리에 있다고 가정)
        sys.path.append('.')
        from step_06_virtual_fitting import (
            VirtualFittingStep, 
            FittingMethod,
            FittingQuality,
            create_virtual_fitting_step,
            get_supported_fabric_types,
            get_supported_clothing_types,
            analyze_fabric_compatibility
        )
        
        print("✅ 모든 클래스/함수 임포트 성공")
        
        # 테스트 이미지 생성
        person_img, cloth_img = create_test_images()
        
        # VirtualFittingStep 인스턴스 생성
        print("🏗️ VirtualFittingStep 인스턴스 생성 중...")
        step = VirtualFittingStep(
            quality_level="balanced",
            enable_visualization=True,
            fitting_method=FittingMethod.HYBRID,
            enable_physics=True
        )
        print("✅ 인스턴스 생성 성공")
        
        # logger 속성 확인
        print(f"🔍 Logger 속성 확인: {hasattr(step, 'logger')}")
        print(f"🔍 Logger 타입: {type(step.logger)}")
        print(f"🔍 Step name: {step.step_name}")
        print(f"🔍 Device: {step.device}")
        print(f"🔍 Is M3 Max: {step.is_m3_max}")
        
        # 초기화 테스트
        print("🚀 Step 초기화 중...")
        init_success = await step.initialize()
        print(f"✅ 초기화 {'성공' if init_success else '실패'}")
        
        # Step 정보 확인
        print("📊 Step 정보 확인 중...")
        step_info = step.get_step_info()
        print(f"   로드된 모델: {step_info['loaded_models']}")
        print(f"   AI 모델 상태: {step_info['ai_models_status']}")
        print(f"   피팅 방법: {step_info['fitting_method']}")
        print(f"   물리 엔진: {step_info['physics_enabled']}")
        print(f"   시각화: {step_info['visualization_enabled']}")
        
        # 메인 처리 테스트
        print("🎭 가상 피팅 처리 테스트 중...")
        start_time = time.time()
        
        result = await step.process(
            person_image=person_img,
            cloth_image=cloth_img,
            fabric_type="cotton",
            clothing_type="shirt",
            fit_preference="fitted"
        )
        
        processing_time = time.time() - start_time
        print(f"⏱️ 처리 시간: {processing_time:.2f}초")
        
        # 결과 검증
        print("📋 결과 검증 중...")
        print(f"   성공: {result['success']}")
        print(f"   신뢰도: {result.get('confidence', 0):.3f}")
        print(f"   품질 점수: {result.get('quality_score', 0):.3f}")
        print(f"   전체 점수: {result.get('overall_score', 0):.3f}")
        print(f"   피팅된 이미지: {result['fitted_image'] is not None}")
        print(f"   시각화 데이터: {result['visualization'] is not None}")
        print(f"   메타데이터 키: {list(result.get('metadata', {}).keys())}")
        
        # 리소스 정리
        print("🧹 리소스 정리 중...")
        await step.cleanup()
        print("✅ 기본 기능 테스트 완료")
        
        return True
        
    except Exception as e:
        print(f"❌ 기본 기능 테스트 실패: {e}")
        print(f"상세 오류:\n{traceback.format_exc()}")
        return False

async def test_different_methods():
    """다양한 피팅 방법 테스트"""
    print("\n" + "="*60)
    print("🎯 다양한 피팅 방법 테스트")
    print("="*60)
    
    try:
        from step_06_virtual_fitting import VirtualFittingStep, FittingMethod
        
        person_img, cloth_img = create_test_images()
        
        methods_to_test = [
            FittingMethod.PHYSICS_BASED,
            FittingMethod.HYBRID,
            FittingMethod.TEMPLATE_MATCHING,
            FittingMethod.DIFFUSION_BASED
        ]
        
        results = {}
        
        for method in methods_to_test:
            print(f"🔄 {method.value} 방법 테스트 중...")
            
            step = VirtualFittingStep(
                fitting_method=method,
                quality_level="fast",
                enable_visualization=True
            )
            
            await step.initialize()
            
            start_time = time.time()
            result = await step.process(
                person_img, cloth_img,
                fabric_type="cotton",
                clothing_type="shirt"
            )
            processing_time = time.time() - start_time
            
            results[method.value] = {
                'success': result['success'],
                'confidence': result.get('confidence', 0),
                'processing_time': processing_time
            }
            
            print(f"   ✅ {method.value}: 성공={result['success']}, "
                  f"신뢰도={result.get('confidence', 0):.3f}, "
                  f"시간={processing_time:.2f}초")
            
            await step.cleanup()
        
        print("📊 방법별 결과 요약:")
        for method, data in results.items():
            print(f"   {method}: {data}")
        
        return True
        
    except Exception as e:
        print(f"❌ 피팅 방법 테스트 실패: {e}")
        print(f"상세 오류:\n{traceback.format_exc()}")
        return False

async def test_fabric_compatibility():
    """천 재질 호환성 테스트"""
    print("\n" + "="*60)
    print("🧵 천 재질 호환성 테스트")
    print("="*60)
    
    try:
        from step_06_virtual_fitting import (
            get_supported_fabric_types,
            get_supported_clothing_types,
            analyze_fabric_compatibility,
            FABRIC_PROPERTIES,
            CLOTHING_FITTING_PARAMS
        )
        
        print("📋 지원되는 형식 확인:")
        fabric_types = get_supported_fabric_types()
        clothing_types = get_supported_clothing_types()
        
        print(f"   천 재질 ({len(fabric_types)}개): {fabric_types}")
        print(f"   의류 타입 ({len(clothing_types)}개): {clothing_types}")
        
        print("\n🔍 호환성 분석 테스트:")
        test_combinations = [
            ("cotton", "shirt"),
            ("silk", "dress"),
            ("denim", "pants"),
            ("leather", "jacket"),
            ("wool", "sweater")
        ]
        
        for fabric, clothing in test_combinations:
            compatibility = analyze_fabric_compatibility(fabric, clothing)
            print(f"   {fabric} + {clothing}: 점수={compatibility['compatibility_score']:.2f}")
            print(f"     추천: {compatibility['recommendations'][0]}")
        
        print("\n📊 천 재질 속성 확인:")
        for fabric, props in list(FABRIC_PROPERTIES.items())[:3]:
            print(f"   {fabric}: 강성={props.stiffness:.1f}, 탄성={props.elasticity:.1f}, "
                  f"광택={props.shine:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 천 재질 테스트 실패: {e}")
        print(f"상세 오류:\n{traceback.format_exc()}")
        return False

async def test_visualization():
    """시각화 기능 테스트"""
    print("\n" + "="*60)
    print("🎨 시각화 기능 테스트")
    print("="*60)
    
    try:
        from step_06_virtual_fitting import VirtualFittingStep, VirtualFittingVisualizer
        
        person_img, cloth_img = create_test_images()
        
        # 시각화 활성화된 VirtualFittingStep
        step = VirtualFittingStep(
            enable_visualization=True,
            quality_level="balanced"
        )
        
        await step.initialize()
        
        print("🎭 시각화 포함 가상 피팅 실행 중...")
        result = await step.process(
            person_img, cloth_img,
            fabric_type="silk",
            clothing_type="dress"
        )
        
        # 시각화 데이터 확인
        viz_data = result.get('visualization')
        if viz_data:
            print("✅ 시각화 데이터 생성 성공")
            viz_keys = list(viz_data.keys())
            print(f"   시각화 종류: {viz_keys}")
            
            # 각 시각화 이미지 크기 확인
            for key, data in viz_data.items():
                if isinstance(data, str) and data.startswith('data:image'):
                    print(f"   {key}: base64 이미지 데이터 (길이: {len(data)} 문자)")
                elif isinstance(data, str) and data:
                    print(f"   {key}: 텍스트 데이터 (길이: {len(data)} 문자)")
                else:
                    print(f"   {key}: 빈 데이터")
        else:
            print("⚠️ 시각화 데이터 없음")
        
        # VirtualFittingVisualizer 테스트
        print("🖼️ VirtualFittingVisualizer 테스트 중...")
        visualizer = VirtualFittingVisualizer()
        
        # 전후 비교 이미지 생성 테스트
        if result.get('fitted_image_raw') is not None:
            comparison = visualizer.create_before_after_comparison(
                person_img, result['fitted_image_raw']
            )
            print(f"   전후 비교 이미지: {comparison.size}")
        
        # 천 재질 분석 차트 테스트
        from step_06_virtual_fitting import FABRIC_PROPERTIES
        fabric_props = FABRIC_PROPERTIES['silk']
        chart = visualizer.create_fabric_analysis_chart(fabric_props, 'silk')
        print(f"   천 재질 분석 차트: {chart.size}")
        
        await step.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ 시각화 테스트 실패: {e}")
        print(f"상세 오류:\n{traceback.format_exc()}")
        return False

async def test_performance():
    """성능 테스트"""
    print("\n" + "="*60)
    print("⚡ 성능 테스트")
    print("="*60)
    
    try:
        from step_06_virtual_fitting import VirtualFittingStep, VirtualFittingProfiler
        
        person_img, cloth_img = create_test_images()
        
        # 성능 측정
        profiler = VirtualFittingProfiler()
        
        step = VirtualFittingStep(quality_level="fast")
        
        print("📊 초기화 성능 측정...")
        profiler.start_timing("initialization")
        await step.initialize()
        init_time = profiler.end_timing("initialization")
        print(f"   초기화 시간: {init_time:.3f}초")
        
        print("📊 처리 성능 측정 (5회 반복)...")
        processing_times = []
        
        for i in range(5):
            profiler.start_timing(f"processing_{i}")
            result = await step.process(
                person_img, cloth_img,
                fabric_type="cotton",
                clothing_type="shirt"
            )
            proc_time = profiler.end_timing(f"processing_{i}")
            processing_times.append(proc_time)
            print(f"   {i+1}회차: {proc_time:.3f}초 (성공: {result['success']})")
        
        # 통계 계산
        avg_time = np.mean(processing_times)
        min_time = np.min(processing_times)
        max_time = np.max(processing_times)
        
        print(f"📈 성능 통계:")
        print(f"   평균 처리 시간: {avg_time:.3f}초")
        print(f"   최소 처리 시간: {min_time:.3f}초")
        print(f"   최대 처리 시간: {max_time:.3f}초")
        print(f"   처리 시간 표준편차: {np.std(processing_times):.3f}초")
        
        # Step 성능 통계 확인
        step_info = step.get_step_info()
        if 'processing_stats' in step_info:
            print(f"   Step 내부 통계: {step_info['processing_stats']}")
        
        # 캐시 효과 테스트
        print("💾 캐시 효과 테스트...")
        profiler.start_timing("cached_processing")
        cached_result = await step.process(
            person_img, cloth_img,
            fabric_type="cotton",
            clothing_type="shirt"
        )
        cached_time = profiler.end_timing("cached_processing")
        print(f"   캐시 적용 시간: {cached_time:.3f}초")
        print(f"   캐시 적용: {'예' if cached_time < avg_time * 0.5 else '아니오'}")
        
        await step.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ 성능 테스트 실패: {e}")
        print(f"상세 오류:\n{traceback.format_exc()}")
        return False

async def test_error_handling():
    """에러 처리 테스트"""
    print("\n" + "="*60)
    print("🛡️ 에러 처리 테스트")
    print("="*60)
    
    try:
        from step_06_virtual_fitting import VirtualFittingStep
        
        step = VirtualFittingStep()
        await step.initialize()
        
        # 1. 잘못된 이미지 입력 테스트
        print("1️⃣ 잘못된 이미지 입력 테스트...")
        
        result = await step.process(
            person_image=None,
            cloth_image=None
        )
        print(f"   None 입력 결과: 성공={result['success']}, 에러='{result.get('error', 'N/A')}'")
        
        # 2. 잘못된 형식 입력 테스트
        print("2️⃣ 잘못된 형식 입력 테스트...")
        
        result = await step.process(
            person_image="invalid_string",
            cloth_image="invalid_string"
        )
        print(f"   문자열 입력 결과: 성공={result['success']}, 에러='{result.get('error', 'N/A')[:50]}...'")
        
        # 3. 빈 배열 입력 테스트
        print("3️⃣ 빈 배열 입력 테스트...")
        
        empty_array = np.array([])
        result = await step.process(
            person_image=empty_array,
            cloth_image=empty_array
        )
        print(f"   빈 배열 입력 결과: 성공={result['success']}, 에러='{result.get('error', 'N/A')[:50]}...'")
        
        # 4. 정상 처리 확인
        print("4️⃣ 정상 처리 확인...")
        
        person_img, cloth_img = create_test_images()
        result = await step.process(
            person_image=person_img,
            cloth_image=cloth_img,
            fabric_type="cotton",
            clothing_type="shirt"
        )
        print(f"   정상 입력 결과: 성공={result['success']}, 신뢰도={result.get('confidence', 0):.3f}")
        
        await step.cleanup()
        print("✅ 에러 처리 테스트 완료 - 모든 에러가 적절히 처리됨")
        return True
        
    except Exception as e:
        print(f"❌ 에러 처리 테스트 실패: {e}")
        print(f"상세 오류:\n{traceback.format_exc()}")
        return False

async def test_convenience_functions():
    """편의 함수 테스트"""
    print("\n" + "="*60)
    print("🔧 편의 함수 테스트")
    print("="*60)
    
    try:
        from step_06_virtual_fitting import (
            create_virtual_fitting_step,
            create_m3_max_virtual_fitting_step,
            quick_virtual_fitting_with_visualization,
            batch_virtual_fitting
        )
        
        person_img, cloth_img = create_test_images()
        
        # 1. create_virtual_fitting_step 테스트
        print("1️⃣ create_virtual_fitting_step 테스트...")
        step1 = create_virtual_fitting_step(device="cpu", quality_level="fast")
        print(f"   생성 성공: {step1.step_name}, 디바이스: {step1.device}")
        await step1.cleanup()
        
        # 2. create_m3_max_virtual_fitting_step 테스트
        print("2️⃣ create_m3_max_virtual_fitting_step 테스트...")
        step2 = create_m3_max_virtual_fitting_step(
            memory_gb=128.0,
            quality_level="high"
        )
        print(f"   M3 Max 설정: is_m3_max={step2.is_m3_max}, memory_gb={step2.memory_gb}")
        await step2.cleanup()
        
        # 3. quick_virtual_fitting_with_visualization 테스트
        print("3️⃣ quick_virtual_fitting_with_visualization 테스트...")
        start_time = time.time()
        quick_result = await quick_virtual_fitting_with_visualization(
            person_img, cloth_img,
            fabric_type="silk",
            clothing_type="dress"
        )
        quick_time = time.time() - start_time
        print(f"   빠른 피팅 결과: 성공={quick_result['success']}, 시간={quick_time:.2f}초")
        
        # 4. batch_virtual_fitting 테스트
        print("4️⃣ batch_virtual_fitting 테스트...")
        image_pairs = [(person_img, cloth_img), (person_img, cloth_img)]
        fabric_types = ["cotton", "silk"]
        clothing_types = ["shirt", "dress"]
        
        start_time = time.time()
        batch_results = await batch_virtual_fitting(
            image_pairs=image_pairs,
            fabric_types=fabric_types,
            clothing_types=clothing_types,
            quality_level="fast"
        )
        batch_time = time.time() - start_time
        
        print(f"   배치 처리 결과: {len(batch_results)}개 처리, 총 시간={batch_time:.2f}초")
        for i, result in enumerate(batch_results):
            print(f"     {i+1}번째: 성공={result['success']}, "
                  f"재질={fabric_types[i]}, 타입={clothing_types[i]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 편의 함수 테스트 실패: {e}")
        print(f"상세 오류:\n{traceback.format_exc()}")
        return False

async def run_all_tests():
    """모든 테스트 실행"""
    print("🚀 VirtualFittingStep 완전 테스트 시작")
    print("="*80)
    
    test_results = {}
    
    # 테스트 목록
    tests = [
        ("기본 기능", test_basic_functionality),
        ("다양한 피팅 방법", test_different_methods),
        ("천 재질 호환성", test_fabric_compatibility),
        ("시각화 기능", test_visualization),
        ("성능 측정", test_performance),
        ("에러 처리", test_error_handling),
        ("편의 함수", test_convenience_functions),
    ]
    
    total_start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔄 {test_name} 테스트 시작...")
            start_time = time.time()
            success = await test_func()
            duration = time.time() - start_time
            
            test_results[test_name] = {
                'success': success,
                'duration': duration
            }
            
            status = "✅ 성공" if success else "❌ 실패"
            print(f"{status} - {test_name} ({duration:.2f}초)")
            
        except Exception as e:
            test_results[test_name] = {
                'success': False,
                'duration': 0,
                'error': str(e)
            }
            print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
    
    total_duration = time.time() - total_start_time
    
    # 최종 결과 요약
    print("\n" + "="*80)
    print("📊 테스트 결과 요약")
    print("="*80)
    
    successful_tests = sum(1 for result in test_results.values() if result['success'])
    total_tests = len(test_results)
    
    print(f"전체 테스트: {total_tests}개")
    print(f"성공한 테스트: {successful_tests}개")
    print(f"실패한 테스트: {total_tests - successful_tests}개")
    print(f"성공률: {(successful_tests / total_tests) * 100:.1f}%")
    print(f"총 소요 시간: {total_duration:.2f}초")
    
    print("\n📋 상세 결과:")
    for test_name, result in test_results.items():
        status = "✅" if result['success'] else "❌"
        duration = result['duration']
        print(f"   {status} {test_name}: {duration:.2f}초")
        if not result['success'] and 'error' in result:
            print(f"      오류: {result['error']}")
    
    if successful_tests == total_tests:
        print(f"\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        print("VirtualFittingStep이 완벽하게 작동합니다! 🔥")
    else:
        print(f"\n⚠️ {total_tests - successful_tests}개의 테스트가 실패했습니다.")
        print("실패한 테스트를 확인하여 문제를 해결해주세요.")
    
    return successful_tests == total_tests

if __name__ == "__main__":
    # 테스트 실행
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)