#!/usr/bin/env python3
"""
ModelLoader 캐시 무효화 및 실제 대형 모델 로딩 테스트
실행: python model_loader_cache_fix.py
"""

import sys
import os
sys.path.append('.')

import time
import torch
from pathlib import Path

def test_model_loader_cache_fix():
    """ModelLoader 캐시 문제 해결 및 테스트"""
    
    print("🎯 ModelLoader 캐시 무효화 및 직접 로딩 테스트")
    print("=" * 60)
    
    try:
        from app.ai_pipeline.utils.model_loader import get_global_model_loader
        
        # ModelLoader 인스턴스 가져오기
        loader = get_global_model_loader()
        print("✅ ModelLoader 인스턴스 획득")
        
        # 캐시 강제 무효화
        if hasattr(loader, '_loaded_models'):
            loader._loaded_models.clear()
            print("🧹 기존 캐시 제거")
        
        if hasattr(loader, 'loaded_models'):
            loader.loaded_models.clear()
            print("🧹 loaded_models 캐시 제거")
        
        # 실제 대형 모델 파일 경로
        large_models = {
            'sam_vit_h_direct': 'ai_models/sam_vit_h_4b8939.pth',  # 2.4GB
            'human_parsing_direct': 'ai_models/step_06_virtual_fitting/ootdiffusion/checkpoints/humanparsing/exp-schp-201908261155-lip.pth',  # 255MB
            'open_clip_direct': 'ai_models/ultra_models/clip_vit_g14/open_clip_pytorch_model.bin'  # 5.2GB
        }
        
        for model_name, model_path in large_models.items():
            file_path = Path(model_path)
            
            if not file_path.exists():
                print(f"❌ 파일 없음: {model_path}")
                continue
                
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"\n🔄 {model_name} 테스트 ({file_size_mb:.1f}MB)")
            
            # 1. 직접 PyTorch 로딩 (기준선)
            try:
                start = time.time()
                direct_model = torch.load(file_path, map_location='cpu', weights_only=False)
                direct_time = time.time() - start
                
                # 실제 파라미터 수 계산
                total_params = 0
                state_dict = direct_model
                if isinstance(direct_model, dict):
                    if 'state_dict' in direct_model:
                        state_dict = direct_model['state_dict']
                    elif 'model' in direct_model:
                        state_dict = direct_model['model']
                
                for key, tensor in state_dict.items():
                    if hasattr(tensor, 'numel'):
                        total_params += tensor.numel()
                
                print(f"  📊 직접 로딩: {direct_time:.2f}s, {total_params:,} 파라미터")
                
                # 2. ModelLoader available_models에 강제 추가
                loader.available_models[model_name] = {
                    'path': str(file_path),
                    'size_mb': file_size_mb,
                    'model_type': 'large_model',
                    'device': 'mps',
                    'verified': True,
                    'params_count': total_params
                }
                
                # 3. ModelLoader로 로딩 (캐시 없이)
                start = time.time()
                loader_model = loader.load_model(model_name)
                loader_time = time.time() - start
                
                print(f"  🔧 ModelLoader: {loader_time:.2f}s, {type(loader_model)}")
                
                # 4. 결과 비교
                if loader_model is not None:
                    if isinstance(loader_model, dict) and len(loader_model) > 100:
                        print(f"  ✅ 성공! 실제 대형 모델 로딩됨 ({len(loader_model)} 키)")
                    else:
                        print(f"  ⚠️ 의심스러운 결과: {len(loader_model) if isinstance(loader_model, dict) else 'N/A'} 키")
                else:
                    print("  ❌ ModelLoader 실패")
                    
            except Exception as e:
                print(f"  ❌ 테스트 실패: {e}")
        
        # 5. ModelLoader 내부 상태 확인
        print(f"\n📊 ModelLoader 상태:")
        print(f"  - 사용 가능한 모델: {len(loader.available_models)}")
        print(f"  - 로드된 모델: {len(getattr(loader, '_loaded_models', {}))}")
        print(f"  - 캐시 크기: {len(getattr(loader, 'loaded_models', {}))}")
        
        # 6. 추천 해결책
        print(f"\n🎯 문제 해결 방안:")
        print(f"  1. 캐시 무효화 메서드 추가")
        print(f"  2. 파일 크기 기반 우선순위 수정")
        print(f"  3. 직접 경로 지정 옵션 추가")
        print(f"  4. 검증 강화 (파라미터 수 기준)")
        
        return True
        
    except Exception as e:
        print(f"❌ 전체 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loader_cache_fix()