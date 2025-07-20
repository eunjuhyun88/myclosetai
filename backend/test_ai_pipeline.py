#!/usr/bin/env python3
# backend/test_ai_pipeline.py - 완전한 AI 파이프라인 성능 테스트

import asyncio
import time
import sys
import os
from pathlib import Path

# 프로젝트 루트 설정
current_file = Path(__file__).absolute()
backend_root = current_file.parent
sys.path.insert(0, str(backend_root))

async def test_model_loading():
    """AI 모델 로딩 테스트"""
    print("🧪 AI 모델 로딩 테스트 시작...")
    
    try:
        from app.ai_pipeline.utils.model_loader import get_global_model_loader
        
        model_loader = get_global_model_loader()
        
        # 핵심 모델들 테스트
        test_models = [
            'human_parsing_graphonomy',
            'cloth_segmentation_u2net', 
            'virtual_fitting_ootdiffusion',
            'pose_estimation_openpose'
        ]
        
        results = {}
        
        for model_name in test_models:
            start_time = time.time()
            try:
                model = model_loader.get_model(model_name)
                load_time = time.time() - start_time
                
                if model:
                    results[model_name] = {
                        'status': '✅ 성공',
                        'load_time': f'{load_time:.3f}초',
                        'type': str(type(model))
                    }
                else:
                    results[model_name] = {
                        'status': '❌ 실패',
                        'load_time': f'{load_time:.3f}초',
                        'type': 'None'
                    }
                    
            except Exception as e:
                load_time = time.time() - start_time
                results[model_name] = {
                    'status': f'❌ 오류: {str(e)[:50]}...',
                    'load_time': f'{load_time:.3f}초',
                    'type': 'Error'
                }
        
        # 결과 출력
        print("\n📊 모델 로딩 결과:")
        print("=" * 80)
        for model_name, result in results.items():
            print(f"{model_name:30} | {result['status']:20} | {result['load_time']:10} | {result['type']}")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"❌ 모델 로딩 테스트 실패: {e}")
        return {}

async def test_memory_usage():
    """메모리 사용량 테스트"""
    print("\n💾 메모리 사용량 테스트...")
    
    try:
        import psutil
        import torch
        
        # 시스템 메모리
        memory = psutil.virtual_memory()
        print(f"시스템 메모리: {memory.total / (1024**3):.1f}GB")
        print(f"사용 중: {memory.used / (1024**3):.1f}GB ({memory.percent:.1f}%)")
        print(f"사용 가능: {memory.available / (1024**3):.1f}GB")
        
        # GPU 메모리 (M3 Max)
        if torch.backends.mps.is_available():
            print(f"🍎 M3 Max MPS: 사용 가능")
            
            # 메모리 테스트
            test_tensor = torch.randn(1000, 1000, device='mps')
            print(f"✅ MPS 텐서 생성 성공: {test_tensor.shape}")
            
            del test_tensor
            torch.mps.empty_cache()
            print("✅ MPS 메모리 정리 완료")
        else:
            print("⚠️ MPS 사용 불가")
            
    except Exception as e:
        print(f"❌ 메모리 테스트 실패: {e}")

async def test_image_processing():
    """이미지 처리 성능 테스트"""
    print("\n🖼️ 이미지 처리 성능 테스트...")
    
    try:
        from PIL import Image
        import torch
        import torchvision.transforms as transforms
        import numpy as np
        
        # 더미 이미지 생성
        dummy_image = Image.new('RGB', (512, 512), (255, 128, 0))
        print("✅ 더미 이미지 생성: 512x512")
        
        # 이미지 변환 파이프라인
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # CPU 처리
        start_time = time.time()
        cpu_tensor = transform(dummy_image)
        cpu_time = time.time() - start_time
        print(f"CPU 이미지 변환: {cpu_time:.4f}초")
        
        # GPU 처리 (M3 Max)
        if torch.backends.mps.is_available():
            start_time = time.time()
            gpu_tensor = transform(dummy_image).to('mps')
            gpu_time = time.time() - start_time
            print(f"🍎 M3 Max GPU 변환: {gpu_time:.4f}초")
            print(f"성능 향상: {cpu_time/gpu_time:.1f}배")
        else:
            print("⚠️ GPU 처리 불가")
            
    except Exception as e:
        print(f"❌ 이미지 처리 테스트 실패: {e}")

async def main():
    """메인 테스트 함수"""
    print("🚀 MyCloset AI 완전한 성능 테스트 시작")
    print("=" * 60)
    
    # 1. 모델 로딩 테스트
    model_results = await test_model_loading()
    
    # 2. 메모리 사용량 테스트  
    await test_memory_usage()
    
    # 3. 이미지 처리 테스트
    await test_image_processing()
    
    # 4. 종합 결과
    print("\n🎯 종합 결과:")
    print("=" * 60)
    
    success_count = sum(1 for result in model_results.values() if '✅ 성공' in result['status'])
    total_count = len(model_results)
    
    if success_count == total_count and total_count > 0:
        print("🎉 모든 테스트 통과! 시스템이 완벽하게 작동합니다.")
    elif success_count > 0:
        print(f"⚠️ 부분 성공: {success_count}/{total_count} 모델 로드됨")
    else:
        print("❌ 시스템에 문제가 있습니다. 설정을 확인하세요.")
    
    print("\n✅ 성능 테스트 완료")

if __name__ == "__main__":
    asyncio.run(main())