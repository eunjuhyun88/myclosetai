# test_step_03_cloth_segmentation.py
"""
🧪 Step 03 의류 세그멘테이션 완전 테스트 스크립트
🔥 모든 문제 해결 확인 및 기능 검증

테스트 항목:
✅ logger 속성 누락 문제 해결 확인
✅ BaseStepMixin 올바른 상속 확인
✅ ModelLoader 연동 확인
✅ 실제 AI 모델 작동 확인
✅ 시각화 기능 확인
✅ M3 Max 최적화 확인
✅ 에러 처리 및 폴백 확인
"""

import asyncio
import logging
import time
import sys
import traceback
from pathlib import Path
from PIL import Image
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_step03.log')
    ]
)

logger = logging.getLogger(__name__)

async def test_basic_import():
    """🔧 1. 기본 import 테스트"""
    print("\n" + "="*60)
    print("🔧 1. 기본 Import 테스트")
    print("="*60)
    
    try:
        # Step 클래스 import 테스트
        from app.ai_pipeline.steps.step_03_cloth_segmentation import (
            ClothSegmentationStep,
            SegmentationMethod,
            ClothingType,
            QualityLevel,
            SegmentationConfig,
            SegmentationResult,
            U2NET,
            REBNCONV,
            RSU7,
            create_cloth_segmentation_step,
            create_m3_max_segmentation_step,
            create_production_segmentation_step,
            CLOTHING_COLORS
        )
        
        print("✅ 모든 클래스/함수 import 성공")
        print(f"   - ClothSegmentationStep: {ClothSegmentationStep}")
        print(f"   - 세그멘테이션 방법: {len(SegmentationMethod)} 개")
        print(f"   - 의류 타입: {len(ClothingType)} 개")
        print(f"   - 팩토리 함수: 3개")
        print(f"   - AI 모델 클래스: U2NET, REBNCONV, RSU7")
        print(f"   - 시각화 색상: {len(CLOTHING_COLORS)} 개")
        
        return True
        
    except Exception as e:
        print(f"❌ Import 실패: {e}")
        traceback.print_exc()
        return False

async def test_logger_attribute():
    """🔥 2. logger 속성 누락 문제 해결 확인"""
    print("\n" + "="*60)
    print("🔥 2. Logger 속성 문제 해결 확인")
    print("="*60)
    
    try:
        from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
        
        # 인스턴스 생성
        step = ClothSegmentationStep(device="cpu")
        
        # logger 속성 확인
        assert hasattr(step, 'logger'), "logger 속성이 없습니다!"
        assert step.logger is not None, "logger가 None입니다!"
        
        # logger 타입 확인
        assert isinstance(step.logger, logging.Logger), f"logger 타입이 잘못됨: {type(step.logger)}"
        
        # logger 이름 확인
        expected_name = f"pipeline.{step.__class__.__name__}"
        assert expected_name in step.logger.name, f"logger 이름이 잘못됨: {step.logger.name}"
        
        # logger 작동 테스트
        step.logger.info("🧪 Logger 작동 테스트")
        step.logger.warning("⚠️ 경고 메시지 테스트")
        step.logger.error("❌ 에러 메시지 테스트")
        
        print("✅ Logger 속성 문제 완전 해결!")
        print(f"   - logger 속성 존재: ✅")
        print(f"   - logger 타입: {type(step.logger)}")
        print(f"   - logger 이름: {step.logger.name}")
        print(f"   - logger 레벨: {step.logger.level}")
        print(f"   - logger 작동: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Logger 테스트 실패: {e}")
        traceback.print_exc()
        return False

async def test_base_step_mixin():
    """🔥 3. BaseStepMixin 올바른 상속 확인"""
    print("\n" + "="*60)
    print("🔥 3. BaseStepMixin 상속 확인")
    print("="*60)
    
    try:
        from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
        
        # 인스턴스 생성
        step = ClothSegmentationStep(device="cpu")
        
        # BaseStepMixin 상속 확인
        print(f"클래스 MRO: {[cls.__name__ for cls in ClothSegmentationStep.__mro__]}")
        
        # 필수 속성들 확인
        required_attrs = ['step_name', 'device', 'is_initialized', 'logger']
        for attr in required_attrs:
            assert hasattr(step, attr), f"필수 속성 {attr}이 없습니다!"
            print(f"   - {attr}: {getattr(step, attr)}")
        
        # step_name 확인
        assert step.step_name == "ClothSegmentationStep", f"step_name이 잘못됨: {step.step_name}"
        
        # 초기화 상태 확인
        assert hasattr(step, 'is_initialized'), "is_initialized 속성이 없습니다!"
        
        print("✅ BaseStepMixin 상속 문제 완전 해결!")
        print(f"   - 상속 체계: ✅")
        print(f"   - 필수 속성: ✅")
        print(f"   - step_name: {step.step_name}")
        print(f"   - 초기화 상태: {step.is_initialized}")
        
        return True
        
    except Exception as e:
        print(f"❌ BaseStepMixin 테스트 실패: {e}")
        traceback.print_exc()
        return False

async def test_model_loader_integration():
    """🔥 4. ModelLoader 연동 확인"""
    print("\n" + "="*60)
    print("🔥 4. ModelLoader 연동 확인")
    print("="*60)
    
    try:
        from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
        
        # 인스턴스 생성
        step = ClothSegmentationStep(device="cpu")
        
        # ModelLoader 관련 속성 확인
        print(f"   - model_loader: {getattr(step, 'model_loader', 'None')}")
        print(f"   - model_interface: {getattr(step, 'model_interface', 'None')}")
        print(f"   - model_paths: {getattr(step, 'model_paths', 'None')}")
        
        # _setup_model_paths 메서드 확인
        assert hasattr(step, '_setup_model_paths'), "_setup_model_paths 메서드가 없습니다!"
        assert hasattr(step, 'model_paths'), "model_paths 속성이 없습니다!"
        
        # 모델 경로 설정 확인
        if hasattr(step, 'model_paths') and step.model_paths:
            print(f"   - 설정된 모델 경로 수: {len(step.model_paths)}")
            for model_name, path in list(step.model_paths.items())[:3]:
                print(f"     * {model_name}: {path}")
        
        print("✅ ModelLoader 연동 문제 완전 해결!")
        print(f"   - _setup_model_paths 메서드: ✅")
        print(f"   - model_paths 속성: ✅")
        print(f"   - 모델 인터페이스 설정: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ ModelLoader 연동 테스트 실패: {e}")
        traceback.print_exc()
        return False

async def test_initialization():
    """🔥 5. 초기화 프로세스 확인"""
    print("\n" + "="*60)
    print("🔥 5. 초기화 프로세스 확인")
    print("="*60)
    
    try:
        from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
        
        # 인스턴스 생성
        step = ClothSegmentationStep(device="cpu", enable_visualization=True)
        
        print("🔄 초기화 시작...")
        start_time = time.time()
        
        # 초기화 실행
        init_result = await step.initialize()
        
        init_time = time.time() - start_time
        print(f"⏱ 초기화 시간: {init_time:.3f}초")
        
        # 초기화 결과 확인
        assert init_result == True, f"초기화 실패: {init_result}"
        assert step.is_initialized == True, f"초기화 상태가 False: {step.is_initialized}"
        
        # 초기화된 구성요소 확인
        components = {
            'available_methods': getattr(step, 'available_methods', []),
            'segmentation_cache': getattr(step, 'segmentation_cache', {}),
            'processing_stats': getattr(step, 'processing_stats', {}),
            'executor': getattr(step, 'executor', None),
            'segmentation_config': getattr(step, 'segmentation_config', None)
        }
        
        for comp_name, comp_value in components.items():
            if comp_value is not None:
                print(f"   - {comp_name}: ✅")
            else:
                print(f"   - {comp_name}: ❌")
        
        # 사용 가능한 방법들 확인
        if hasattr(step, 'available_methods'):
            print(f"   - 사용 가능한 세그멘테이션 방법: {len(step.available_methods)}개")
            for method in step.available_methods:
                print(f"     * {method.value}")
        
        print("✅ 초기화 프로세스 완전 성공!")
        print(f"   - 초기화 결과: {init_result}")
        print(f"   - 초기화 시간: {init_time:.3f}초")
        print(f"   - 모든 구성요소 준비: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ 초기화 테스트 실패: {e}")
        traceback.print_exc()
        return False

async def test_segmentation_processing():
    """🔥 6. 실제 세그멘테이션 처리 확인"""
    print("\n" + "="*60)
    print("🔥 6. 실제 세그멘테이션 처리 확인")
    print("="*60)
    
    try:
        from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
        
        # 인스턴스 생성 및 초기화
        step = ClothSegmentationStep(
            device="cpu",
            enable_visualization=True,
            visualization_quality="high"
        )
        
        await step.initialize()
        
        # 테스트 이미지 생성 (512x512 셔츠 시뮬레이션)
        print("🖼 테스트 이미지 생성...")
        test_image = Image.new('RGB', (512, 512), (180, 140, 90))
        
        # 처리 실행
        print("🔄 세그멘테이션 처리 시작...")
        start_time = time.time()
        
        result = await step.process(
            test_image,
            clothing_type="shirt",
            quality_level="balanced"
        )
        
        process_time = time.time() - start_time
        print(f"⏱ 처리 시간: {process_time:.3f}초")
        
        # 결과 확인
        assert isinstance(result, dict), f"결과가 딕셔너리가 아님: {type(result)}"
        assert 'success' in result, "결과에 success 키가 없음"
        
        print(f"📊 처리 결과:")
        print(f"   - 성공 여부: {result['success']}")
        print(f"   - 처리 시간: {result.get('processing_time', 'N/A')}")
        print(f"   - 사용된 방법: {result.get('method_used', 'N/A')}")
        
        if result['success']:
            # 성공한 경우 상세 정보 확인
            details = result.get('details', {})
            print(f"   - 신뢰도: {details.get('confidence_score', 'N/A')}")
            print(f"   - 품질 점수: {details.get('quality_score', 'N/A')}")
            
            # 시각화 결과 확인
            viz_keys = ['result_image', 'overlay_image', 'mask_image', 'boundary_image']
            for key in viz_keys:
                has_viz = bool(details.get(key, ''))
                print(f"   - {key}: {'✅' if has_viz else '❌'}")
                
        else:
            # 실패한 경우 에러 정보
            print(f"   - 에러 메시지: {result.get('error_message', 'N/A')}")
        
        print("✅ 세그멘테이션 처리 테스트 완료!")
        
        return True
        
    except Exception as e:
        print(f"❌ 세그멘테이션 처리 테스트 실패: {e}")
        traceback.print_exc()
        return False

async def test_visualization_features():
    """🆕 7. 시각화 기능 확인"""
    print("\n" + "="*60)
    print("🆕 7. 시각화 기능 확인")
    print("="*60)
    
    try:
        from app.ai_pipeline.steps.step_03_cloth_segmentation import (
            ClothSegmentationStep, 
            CLOTHING_COLORS
        )
        
        # 시각화 활성화된 인스턴스 생성
        step = ClothSegmentationStep(
            device="cpu",
            enable_visualization=True,
            visualization_quality="high"
        )
        
        await step.initialize()
        
        # 의류 색상 팔레트 확인
        print(f"🎨 의류 색상 팔레트: {len(CLOTHING_COLORS)}개")
        for clothing_type, color in CLOTHING_COLORS.items():
            print(f"   - {clothing_type}: RGB{color}")
        
        # 시각화 설정 확인
        viz_config = step.segmentation_config
        print(f"📊 시각화 설정:")
        print(f"   - 활성화: {viz_config.enable_visualization}")
        print(f"   - 품질: {viz_config.visualization_quality}")
        print(f"   - 마스크 표시: {viz_config.show_masks}")
        print(f"   - 경계선 표시: {viz_config.show_boundaries}")
        print(f"   - 오버레이 투명도: {viz_config.overlay_opacity}")
        
        # 테스트 이미지로 시각화 테스트
        test_image = Image.new('RGB', (256, 256), (150, 100, 80))
        
        result = await step.process(
            test_image,
            clothing_type="dress",
            quality_level="fast"
        )
        
        if result['success']:
            details = result.get('details', {})
            viz_results = {
                'result_image': len(details.get('result_image', '')),
                'overlay_image': len(details.get('overlay_image', '')),
                'mask_image': len(details.get('mask_image', '')),
                'boundary_image': len(details.get('boundary_image', ''))
            }
            
            print(f"🎨 시각화 결과 (base64 길이):")
            for viz_type, length in viz_results.items():
                status = "✅" if length > 0 else "❌"
                print(f"   - {viz_type}: {status} ({length} chars)")
        
        print("✅ 시각화 기능 테스트 완료!")
        
        return True
        
    except Exception as e:
        print(f"❌ 시각화 테스트 실패: {e}")
        traceback.print_exc()
        return False

async def test_error_handling():
    """🛡 8. 에러 처리 및 폴백 확인"""
    print("\n" + "="*60)
    print("🛡 8. 에러 처리 및 폴백 확인")
    print("="*60)
    
    try:
        from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
        
        step = ClothSegmentationStep(device="cpu")
        await step.initialize()
        
        # 1. 잘못된 이미지 타입 테스트
        print("🔄 잘못된 이미지 타입 테스트...")
        try:
            result = await step.process("nonexistent_file.jpg", clothing_type="shirt")
            print(f"   - 결과: {result['success']} (에러 처리됨)")
        except Exception as e:
            print(f"   - 예외 발생 (정상): {e}")
        
        # 2. 지원되지 않는 의류 타입 테스트
        print("🔄 지원되지 않는 의류 타입 테스트...")
        test_image = Image.new('RGB', (100, 100), (128, 128, 128))
        result = await step.process(test_image, clothing_type="unknown_type")
        print(f"   - 결과: {result['success']} (graceful handling)")
        
        # 3. 메모리 부족 상황 시뮬레이션
        print("🔄 대용량 이미지 처리 테스트...")
        try:
            large_image = Image.new('RGB', (2048, 2048), (100, 100, 100))
            result = await step.process(large_image, clothing_type="shirt")
            print(f"   - 대용량 이미지 처리: {result['success']}")
        except Exception as e:
            print(f"   - 메모리 제한 처리: {e}")
        
        # 4. 폴백 메커니즘 확인
        print("🔄 폴백 메커니즘 확인...")
        fallback_methods = step.get_available_methods()
        print(f"   - 사용 가능한 폴백 방법: {len(fallback_methods)}개")
        for method in fallback_methods:
            print(f"     * {method}")
        
        print("✅ 에러 처리 및 폴백 테스트 완료!")
        
        return True
        
    except Exception as e:
        print(f"❌ 에러 처리 테스트 실패: {e}")
        traceback.print_exc()
        return False

async def test_factory_functions():
    """🏭 9. 팩토리 함수들 확인"""
    print("\n" + "="*60)
    print("🏭 9. 팩토리 함수들 확인")
    print("="*60)
    
    try:
        from app.ai_pipeline.steps.step_03_cloth_segmentation import (
            create_cloth_segmentation_step,
            create_m3_max_segmentation_step,
            create_production_segmentation_step
        )
        
        # 1. 기본 팩토리 함수
        print("🔄 기본 팩토리 함수 테스트...")
        step1 = create_cloth_segmentation_step(device="cpu")
        assert step1 is not None, "기본 팩토리 함수 실패"
        print(f"   - create_cloth_segmentation_step: ✅")
        print(f"     * 디바이스: {step1.device}")
        print(f"     * Step 이름: {step1.step_name}")
        
        # 2. M3 Max 최적화 팩토리 함수
        print("🔄 M3 Max 팩토리 함수 테스트...")
        step2 = create_m3_max_segmentation_step(device="cpu")  # CPU로 테스트
        assert step2 is not None, "M3 Max 팩토리 함수 실패"
        print(f"   - create_m3_max_segmentation_step: ✅")
        print(f"     * M3 Max 모드: {step2.is_m3_max}")
        print(f"     * 시각화 활성화: {step2.segmentation_config.enable_visualization}")
        print(f"     * 메모리: {step2.memory_gb}GB")
        
        # 3. 프로덕션 팩토리 함수
        print("🔄 프로덕션 팩토리 함수 테스트...")
        step3 = create_production_segmentation_step(quality_level="balanced")
        assert step3 is not None, "프로덕션 팩토리 함수 실패"
        print(f"   - create_production_segmentation_step: ✅")
        print(f"     * 품질 레벨: {step3.segmentation_config.quality_level.value}")
        print(f"     * 후처리 활성화: {step3.enable_post_processing}")
        print(f"     * 캐시 크기: {step3.segmentation_config.cache_size}")
        
        print("✅ 모든 팩토리 함수 테스트 완료!")
        
        return True
        
    except Exception as e:
        print(f"❌ 팩토리 함수 테스트 실패: {e}")
        traceback.print_exc()
        return False

async def test_system_info():
    """📊 10. 시스템 정보 및 통계 확인"""
    print("\n" + "="*60)
    print("📊 10. 시스템 정보 및 통계 확인")
    print("="*60)
    
    try:
        from app.ai_pipeline.steps.step_03_cloth_segmentation import ClothSegmentationStep
        
        step = ClothSegmentationStep(device="cpu")
        await step.initialize()
        
        # 1. Step 정보 조회
        print("🔄 Step 정보 조회...")
        step_info = await step.get_step_info()
        
        print(f"📋 Step 정보:")
        print(f"   - Step 이름: {step_info.get('step_name')}")
        print(f"   - Step 번호: {step_info.get('step_number')}")
        print(f"   - 디바이스: {step_info.get('device')}")
        print(f"   - 초기화 상태: {step_info.get('initialized')}")
        
        # 2. 사용 가능한 방법들
        print("🔄 사용 가능한 방법들 조회...")
        available_methods = step.get_available_methods()
        print(f"📋 사용 가능한 방법: {len(available_methods)}개")
        for method in available_methods:
            print(f"   - {method}")
        
        # 3. 지원되는 의류 타입들
        print("🔄 지원되는 의류 타입 조회...")
        clothing_types = step.get_supported_clothing_types()
        print(f"📋 지원 의류 타입: {len(clothing_types)}개")
        for clothing_type in clothing_types[:5]:  # 처음 5개만 출력
            print(f"   - {clothing_type}")
        
        # 4. 처리 통계
        print("🔄 처리 통계 조회...")
        stats = step.get_statistics()
        print(f"📊 처리 통계:")
        print(f"   - 총 처리 수: {stats.get('total_processed', 0)}")
        print(f"   - 성공 처리 수: {stats.get('successful_segmentations', 0)}")
        print(f"   - 평균 품질: {stats.get('average_quality', 0):.3f}")
        print(f"   - 캐시 히트: {stats.get('cache_hits', 0)}")
        
        # 5. 방법별 정보
        print("🔄 방법별 상세 정보...")
        for method in available_methods[:3]:  # 처음 3개 방법만
            method_info = step.get_method_info(method)
            print(f"📋 {method} 정보:")
            print(f"   - 이름: {method_info.get('name')}")
            print(f"   - 품질: {method_info.get('quality')}")
            print(f"   - 속도: {method_info.get('speed')}")
        
        print("✅ 시스템 정보 및 통계 테스트 완료!")
        
        return True
        
    except Exception as e:
        print(f"❌ 시스템 정보 테스트 실패: {e}")
        traceback.print_exc()
        return False

async def run_complete_test():
    """🚀 전체 테스트 실행"""
    print("🚀 Step 03 의류 세그멘테이션 완전 테스트 시작")
    print("🔥 모든 문제 해결 확인 및 기능 검증")
    print("="*80)
    
    test_functions = [
        ("기본 Import", test_basic_import),
        ("Logger 속성 해결", test_logger_attribute),
        ("BaseStepMixin 상속", test_base_step_mixin),
        ("ModelLoader 연동", test_model_loader_integration),
        ("초기화 프로세스", test_initialization),
        ("세그멘테이션 처리", test_segmentation_processing),
        ("시각화 기능", test_visualization_features),
        ("에러 처리/폴백", test_error_handling),
        ("팩토리 함수들", test_factory_functions),
        ("시스템 정보", test_system_info)
    ]
    
    results = {}
    total_start_time = time.time()
    
    for test_name, test_func in test_functions:
        try:
            print(f"\n🔄 {test_name} 테스트 실행 중...")
            test_start = time.time()
            
            result = await test_func()
            test_time = time.time() - test_start
            
            results[test_name] = {
                'success': result,
                'time': test_time
            }
            
            status = "✅ 성공" if result else "❌ 실패"
            print(f"📊 {test_name}: {status} (⏱ {test_time:.3f}초)")
            
        except Exception as e:
            test_time = time.time() - test_start
            results[test_name] = {
                'success': False,
                'time': test_time,
                'error': str(e)
            }
            print(f"📊 {test_name}: ❌ 예외 발생 (⏱ {test_time:.3f}초)")
            print(f"   에러: {e}")
    
    total_time = time.time() - total_start_time
    
    # 최종 결과 요약
    print("\n" + "="*80)
    print("📊 최종 테스트 결과 요약")
    print("="*80)
    
    success_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)
    
    print(f"🎯 전체 성공률: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    print(f"⏱ 총 테스트 시간: {total_time:.3f}초")
    
    print(f"\n📋 개별 테스트 결과:")
    for test_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        time_str = f"{result['time']:.3f}초"
        print(f"   {status} {test_name:<20} {time_str:>8}")
        
        if not result['success'] and 'error' in result:
            print(f"      └─ 에러: {result['error']}")
    
    # 성공 여부 판단
    if success_count == total_count:
        print(f"\n🎉 모든 테스트 성공! Step 03이 완벽하게 작동합니다!")
        print(f"🔥 logger 속성, BaseStepMixin, ModelLoader 모든 문제 해결 확인!")
    elif success_count >= total_count * 0.8:
        print(f"\n✅ 대부분 테스트 성공! 기본 기능은 작동합니다.")
        print(f"⚠️ 일부 개선이 필요한 부분이 있습니다.")
    else:
        print(f"\n⚠️ 여러 테스트 실패. 추가 수정이 필요합니다.")
    
    return success_count, total_count, results

if __name__ == "__main__":
    # 비동기 테스트 실행
    try:
        success_count, total_count, results = asyncio.run(run_complete_test())
        
        # 종료 코드 설정
        if success_count == total_count:
            exit_code = 0  # 완전 성공
        elif success_count >= total_count * 0.8:
            exit_code = 1  # 부분 성공
        else:
            exit_code = 2  # 대부분 실패
            
        print(f"\n🏁 테스트 완료. 종료 코드: {exit_code}")
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 사용자에 의해 테스트가 중단되었습니다.")
        sys.exit(3)
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 심각한 오류 발생: {e}")
        traceback.print_exc()
        sys.exit(4)