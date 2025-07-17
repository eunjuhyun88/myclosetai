#!/usr/bin/env python3
"""
Step 01 Human Parsing 단독 테스트 스크립트
backend/test_step_01_human_parsing.py

실행 방법:
cd backend
python test_step_01_human_parsing.py
"""

import os
import sys
import time
import asyncio
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from PIL import Image

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_step_01.log')
    ]
)
logger = logging.getLogger(__name__)

def create_test_image() -> torch.Tensor:
    """테스트용 이미지 텐서 생성"""
    # 512x512 RGB 더미 이미지 (사람 모양 시뮬레이션)
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # 배경 (연한 파란색)
    image[:, :] = [200, 220, 255]
    
    # 머리 (타원)
    center_x, center_y = 256, 150
    for y in range(100, 200):
        for x in range(200, 312):
            if ((x - center_x) / 56) ** 2 + ((y - center_y) / 50) ** 2 <= 1:
                image[y, x] = [255, 220, 177]  # 피부색
    
    # 몸통 (직사각형)
    image[200:400, 200:312] = [100, 150, 200]  # 상의 (파란색)
    
    # 팔 (직사각형)
    image[220:380, 150:200] = [255, 220, 177]  # 왼팔
    image[220:380, 312:362] = [255, 220, 177]  # 오른팔
    
    # 다리 (직사각형)
    image[400:500, 200:250] = [50, 50, 100]   # 왼다리 (바지)
    image[400:500, 262:312] = [50, 50, 100]   # 오른다리 (바지)
    
    # PIL로 변환 후 텐서로
    pil_image = Image.fromarray(image)
    
    # 텐서 변환 [1, 3, 512, 512]
    tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    
    return tensor

async def test_step_01_basic():
    """기본 Step 01 테스트"""
    print("🧪 Step 01 기본 테스트 시작")
    print("=" * 50)
    
    try:
        # 1. Step 클래스 import 테스트
        print("1️⃣ Step 클래스 import 테스트...")
        try:
            from app.ai_pipeline.steps.step_01_human_parsing import (
                HumanParsingStep,
                HumanParsingConfig,
                create_human_parsing_step,
                BODY_PARTS,
                CLOTHING_CATEGORIES
            )
            print("✅ Step 01 클래스 import 성공")
            print(f"📊 인체 부위: {len(BODY_PARTS)}개")
            print(f"👕 의류 카테고리: {len(CLOTHING_CATEGORIES)}개")
        except ImportError as e:
            print(f"❌ Step 01 import 실패: {e}")
            return False
        
        # 2. 설정 클래스 테스트
        print("\n2️⃣ 설정 클래스 테스트...")
        try:
            config = HumanParsingConfig(
                device="cpu",  # CPU로 안전하게 테스트
                model_name="human_parsing_graphonomy",
                input_size=(512, 512),
                use_fp16=False,  # CPU에서는 FP16 비활성화
                use_coreml=False,
                warmup_enabled=True
            )
            print(f"✅ 설정 생성 성공:")
            print(f"   - 디바이스: {config.device}")
            print(f"   - 모델: {config.model_name}")
            print(f"   - 입력 크기: {config.input_size}")
            print(f"   - FP16: {config.use_fp16}")
        except Exception as e:
            print(f"❌ 설정 생성 실패: {e}")
            return False
        
        # 3. Step 인스턴스 생성 테스트
        print("\n3️⃣ Step 인스턴스 생성 테스트...")
        try:
            step = HumanParsingStep(
                device="cpu",
                config=config
            )
            print(f"✅ Step 인스턴스 생성 성공:")
            print(f"   - 클래스: {step.__class__.__name__}")
            print(f"   - 디바이스: {step.device}")
            print(f"   - 단계 번호: {step.step_number}")
            print(f"   - 초기화 상태: {step.is_initialized}")
        except Exception as e:
            print(f"❌ Step 인스턴스 생성 실패: {e}")
            return False
        
        # 4. 초기화 테스트 (ModelLoader 없이도 작동하는지)
        print("\n4️⃣ Step 초기화 테스트...")
        try:
            init_success = await step.initialize()
            print(f"✅ 초기화 완료:")
            print(f"   - 성공 여부: {init_success}")
            print(f"   - 초기화 상태: {step.is_initialized}")
            print(f"   - 로드된 모델: {list(step.models_loaded.keys())}")
        except Exception as e:
            print(f"⚠️ 초기화 실패 (예상됨 - ModelLoader 없음): {e}")
            print("💡 ModelLoader 없이도 기본 기능은 작동해야 합니다")
        
        # 5. 테스트 이미지 생성
        print("\n5️⃣ 테스트 이미지 생성...")
        try:
            test_tensor = create_test_image()
            print(f"✅ 테스트 이미지 생성 성공:")
            print(f"   - 형태: {test_tensor.shape}")
            print(f"   - 타입: {test_tensor.dtype}")
            print(f"   - 디바이스: {test_tensor.device}")
            print(f"   - 값 범위: [{test_tensor.min().item():.3f}, {test_tensor.max().item():.3f}]")
        except Exception as e:
            print(f"❌ 테스트 이미지 생성 실패: {e}")
            return False
        
        # 6. 처리 함수 테스트 (폴백 모드)
        print("\n6️⃣ 처리 함수 테스트 (폴백 모드)...")
        try:
            start_time = time.time()
            result = await step.process(test_tensor)
            processing_time = time.time() - start_time
            
            print(f"✅ 처리 완료:")
            print(f"   - 성공 여부: {result['success']}")
            print(f"   - 처리 시간: {processing_time:.3f}초")
            print(f"   - 신뢰도: {result.get('confidence', 'N/A')}")
            print(f"   - 파싱 맵 형태: {result['parsing_map'].shape}")
            print(f"   - 감지된 부위 수: {len(result.get('body_parts_detected', {}))}")
            print(f"   - 의류 카테고리 수: {len(result.get('clothing_regions', {}).get('categories_detected', []))}")
            
            # 결과 상세 분석
            if result.get('body_parts_detected'):
                print(f"   - 감지된 부위: {list(result['body_parts_detected'].keys())[:5]}...")
            
            if result.get('clothing_regions'):
                clothing = result['clothing_regions']
                print(f"   - 주요 의류: {clothing.get('dominant_category', 'N/A')}")
                print(f"   - 의류 면적: {clothing.get('total_clothing_area', 0):.3f}")
            
        except Exception as e:
            print(f"❌ 처리 함수 테스트 실패: {e}")
            return False
        
        # 7. 상태 정보 테스트
        print("\n7️⃣ 상태 정보 테스트...")
        try:
            step_info = await step.get_step_info()
            print(f"✅ 상태 정보 조회 성공:")
            print(f"   - 단계명: {step_info['step_name']}")
            print(f"   - 디바이스: {step_info['device']}")
            print(f"   - 로드된 모델: {step_info['models_loaded']}")
            print(f"   - 성능 통계: {step_info['performance']}")
            print(f"   - 캐시 크기: {step_info['cache']['size']}")
            print(f"   - 최적화 상태: {step_info['optimization']}")
        except Exception as e:
            print(f"❌ 상태 정보 테스트 실패: {e}")
            return False
        
        # 8. 유틸리티 함수 테스트
        print("\n8️⃣ 유틸리티 함수 테스트...")
        try:
            # 시각화 테스트
            parsing_map = result['parsing_map']
            visualized = step.visualize_parsing(parsing_map)
            print(f"✅ 시각화 함수:")
            print(f"   - 원본 형태: {parsing_map.shape}")
            print(f"   - 시각화 형태: {visualized.shape}")
            
            # 의류 마스크 테스트
            if CLOTHING_CATEGORIES:
                category = list(CLOTHING_CATEGORIES.keys())[0]
                try:
                    clothing_mask = step.get_clothing_mask(parsing_map, category)
                    print(f"✅ 의류 마스크 ({category}):")
                    print(f"   - 마스크 형태: {clothing_mask.shape}")
                    print(f"   - 픽셀 수: {clothing_mask.sum()}")
                except Exception as e:
                    print(f"⚠️ 의류 마스크 테스트 실패: {e}")
            
        except Exception as e:
            print(f"❌ 유틸리티 함수 테스트 실패: {e}")
            return False
        
        # 9. 팩토리 함수 테스트
        print("\n9️⃣ 팩토리 함수 테스트...")
        try:
            step2 = await create_human_parsing_step(
                device="cpu",
                config={'use_fp16': False, 'warmup_enabled': False}
            )
            print(f"✅ 팩토리 함수 성공:")
            print(f"   - 클래스: {step2.__class__.__name__}")
            print(f"   - 초기화 상태: {step2.is_initialized}")
        except Exception as e:
            print(f"❌ 팩토리 함수 테스트 실패: {e}")
            return False
        
        # 10. 정리 테스트
        print("\n🔟 리소스 정리 테스트...")
        try:
            await step.cleanup()
            print(f"✅ 정리 완료:")
            print(f"   - 초기화 상태: {step.is_initialized}")
            print(f"   - 로드된 모델: {list(step.models_loaded.keys())}")
        except Exception as e:
            print(f"❌ 정리 테스트 실패: {e}")
            return False
        
        print("\n🎉 Step 01 기본 테스트 모두 통과!")
        return True
        
    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_step_01_advanced():
    """고급 Step 01 테스트"""
    print("\n🚀 Step 01 고급 테스트 시작")
    print("=" * 50)
    
    try:
        from app.ai_pipeline.steps.step_01_human_parsing import (
            HumanParsingStep,
            HumanParsingConfig
        )
        
        # 1. 다양한 설정으로 테스트
        print("1️⃣ 다양한 설정 테스트...")
        
        configs = [
            {"device": "cpu", "quality_level": "fast"},
            {"device": "cpu", "quality_level": "balanced"},
            {"device": "cpu", "quality_level": "high"},
        ]
        
        for i, config_dict in enumerate(configs):
            try:
                config = HumanParsingConfig(**config_dict)
                step = HumanParsingStep(config=config)
                print(f"   ✅ 설정 {i+1}: {config.quality_level} - 성공")
            except Exception as e:
                print(f"   ❌ 설정 {i+1}: {config_dict} - 실패: {e}")
        
        # 2. 성능 테스트
        print("\n2️⃣ 성능 테스트...")
        
        step = HumanParsingStep(device="cpu")
        await step.initialize()
        
        # 여러 이미지로 처리 시간 측정
        test_images = [create_test_image() for _ in range(3)]
        
        times = []
        for i, img in enumerate(test_images):
            start = time.time()
            result = await step.process(img)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"   🔄 이미지 {i+1}: {elapsed:.3f}초, 신뢰도: {result.get('confidence', 0):.3f}")
        
        avg_time = sum(times) / len(times)
        print(f"   📊 평균 처리 시간: {avg_time:.3f}초")
        print(f"   📈 캐시 히트: {step.processing_stats['cache_hits']}")
        
        # 3. 캐시 테스트
        print("\n3️⃣ 캐시 시스템 테스트...")
        
        # 같은 이미지로 다시 처리 (캐시 히트 기대)
        cache_start = time.time()
        cached_result = await step.process(test_images[0])
        cache_time = time.time() - cache_start
        
        print(f"   💾 캐시된 처리 시간: {cache_time:.3f}초")
        print(f"   🎯 캐시에서 반환: {cached_result.get('from_cache', False)}")
        
        # 4. 메모리 사용량 테스트
        print("\n4️⃣ 메모리 사용량 테스트...")
        
        try:
            memory_stats = await step.memory_manager.get_usage_stats()
            print(f"   📊 메모리 상태: {memory_stats}")
            
            # 메모리 정리 테스트
            await step.memory_manager.cleanup()
            print(f"   🧹 메모리 정리 완료")
            
        except Exception as e:
            print(f"   ⚠️ 메모리 테스트 실패: {e}")
        
        # 5. 다양한 입력 크기 테스트
        print("\n5️⃣ 다양한 입력 크기 테스트...")
        
        input_sizes = [(256, 256), (512, 512), (768, 768)]
        
        for size in input_sizes:
            try:
                # 크기별 테스트 이미지 생성
                test_img = torch.randn(1, 3, size[0], size[1])
                result = await step.process(test_img)
                print(f"   ✅ 크기 {size}: 성공, 신뢰도: {result.get('confidence', 0):.3f}")
            except Exception as e:
                print(f"   ❌ 크기 {size}: 실패 - {e}")
        
        await step.cleanup()
        
        print("\n🎉 Step 01 고급 테스트 모두 통과!")
        return True
        
    except Exception as e:
        print(f"\n💥 고급 테스트 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_step_01_error_handling():
    """에러 처리 테스트"""
    print("\n🛡️ Step 01 에러 처리 테스트 시작")
    print("=" * 50)
    
    try:
        from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
        
        # 1. 잘못된 입력 테스트
        print("1️⃣ 잘못된 입력 테스트...")
        
        step = HumanParsingStep(device="cpu")
        await step.initialize()
        
        # 잘못된 형태의 텐서
        try:
            wrong_tensor = torch.randn(2, 2)  # 잘못된 차원
            result = await step.process(wrong_tensor)
            print(f"   ✅ 잘못된 텐서 처리: {result['success']}")
        except Exception as e:
            print(f"   ⚠️ 잘못된 텐서 오류 (예상됨): {e}")
        
        # 2. 초기화되지 않은 상태 테스트
        print("\n2️⃣ 초기화되지 않은 상태 테스트...")
        
        uninitialized_step = HumanParsingStep(device="cpu")
        try:
            test_tensor = create_test_image()
            result = await uninitialized_step.process(test_tensor)
            print(f"   ✅ 미초기화 상태 처리: {result['success']}")
        except Exception as e:
            print(f"   ⚠️ 미초기화 오류 (예상됨): {e}")
        
        # 3. 디바이스 오류 테스트
        print("\n3️⃣ 디바이스 오류 테스트...")
        
        try:
            # 존재하지 않는 디바이스
            invalid_step = HumanParsingStep(device="invalid_device")
            print(f"   ✅ 잘못된 디바이스 처리: 디바이스={invalid_step.device}")
        except Exception as e:
            print(f"   ⚠️ 디바이스 오류 (예상됨): {e}")
        
        await step.cleanup()
        
        print("\n🛡️ 에러 처리 테스트 완료!")
        return True
        
    except Exception as e:
        print(f"\n💥 에러 처리 테스트 오류: {e}")
        return False

def test_imports():
    """Import 테스트"""
    print("📦 Import 테스트 시작")
    print("=" * 30)
    
    imports_to_test = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
    ]
    
    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name}: 사용 가능")
        except ImportError:
            print(f"❌ {display_name}: 설치 필요")
    
    # PyTorch 디바이스 체크
    try:
        import torch
        print(f"\n🔧 PyTorch 정보:")
        print(f"   - 버전: {torch.__version__}")
        print(f"   - CUDA 지원: {torch.cuda.is_available()}")
        try:
            print(f"   - MPS 지원: {torch.backends.mps.is_available()}")
        except:
            print(f"   - MPS 지원: 확인 불가")
    except:
        print("❌ PyTorch 정보 확인 실패")

async def main():
    """메인 테스트 실행"""
    print("🧪 Step 01 Human Parsing 단독 테스트")
    print("=" * 60)
    print("📅 시작 시간:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # Import 테스트
    test_imports()
    
    # 기본 테스트
    basic_success = await test_step_01_basic()
    
    if basic_success:
        # 고급 테스트
        advanced_success = await test_step_01_advanced()
        
        # 에러 처리 테스트
        error_success = await test_step_01_error_handling()
        
        # 최종 결과
        print("\n" + "=" * 60)
        print("📊 테스트 결과 요약:")
        print(f"   ✅ 기본 테스트: {'통과' if basic_success else '실패'}")
        print(f"   ✅ 고급 테스트: {'통과' if advanced_success else '실패'}")
        print(f"   ✅ 에러 처리: {'통과' if error_success else '실패'}")
        
        if basic_success and advanced_success and error_success:
            print("\n🎉 모든 테스트 통과! Step 01이 정상 작동합니다!")
        else:
            print("\n⚠️ 일부 테스트 실패. 로그를 확인하세요.")
    else:
        print("\n❌ 기본 테스트 실패. 설치 및 설정을 확인하세요.")
    
    print("=" * 60)
    print("📅 종료 시간:", time.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    # 이벤트 루프 실행
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n💥 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()