#!/usr/bin/env python3
"""
🔍 모델 로딩 디버깅 스크립트
왜 가중치 로딩이 부분적으로만 되는지 분석
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_model_loading():
    """모델 로딩 디버깅"""
    print("🔍 모델 로딩 디버깅 시작...")
    
    try:
        from app.ai_pipeline.utils.model_loader import (
            ModelLoader, 
            load_model_for_step,
            CheckpointAnalyzer,
            DynamicModelCreator
        )
        
        # 1. 실제 모델 파일 경로 확인
        print("\n1️⃣ 실제 모델 파일 경로 확인...")
        
        model_paths = {
            'human_parsing': 'ai_models/step_01_human_parsing/graphonomy.pth',
            'pose_estimation': 'ai_models/step_02_pose_estimation/hrnet_w48_coco_384x288.pth',
            'cloth_segmentation': 'ai_models/step_03/sam.pth'
        }
        
        for step, path in model_paths.items():
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024 * 1024)  # MB
                print(f"   ✅ {step}: {path} ({size:.1f} MB)")
            else:
                print(f"   ❌ {step}: {path} (파일 없음)")
        
        # 2. 체크포인트 분석 테스트
        print("\n2️⃣ 체크포인트 분석 테스트...")
        
        analyzer = CheckpointAnalyzer()
        
        for step, path in model_paths.items():
            if os.path.exists(path):
                print(f"\n   🔍 {step} 체크포인트 분석...")
                try:
                    analysis = analyzer.analyze_checkpoint(path)
                    print(f"      ✅ 분석 성공: {analysis.get('architecture_type', 'unknown')}")
                    print(f"      📊 파라미터 수: {analysis.get('total_params', 0):,}")
                    print(f"      🏗️ 아키텍처: {analysis.get('architecture_type', 'unknown')}")
                    print(f"      📥 입력 채널: {analysis.get('input_channels', 0)}")
                    print(f"      🎯 클래스 수: {analysis.get('num_classes', 0)}")
                except Exception as e:
                    print(f"      ❌ 분석 실패: {e}")
        
        # 3. 동적 모델 생성 테스트
        print("\n3️⃣ 동적 모델 생성 테스트...")
        
        creator = DynamicModelCreator()
        
        for step, path in model_paths.items():
            if os.path.exists(path):
                print(f"\n   🔧 {step} 동적 모델 생성...")
                try:
                    model = creator.create_model_from_checkpoint(path, step)
                    if model:
                        param_count = sum(p.numel() for p in model.parameters())
                        print(f"      ✅ 모델 생성 성공: {param_count:,} 파라미터")
                    else:
                        print(f"      ❌ 모델 생성 실패")
                except Exception as e:
                    print(f"      ❌ 모델 생성 오류: {e}")
        
        # 4. 키 매핑 상세 분석
        print("\n4️⃣ 키 매핑 상세 분석...")
        
        import torch
        
        for step, path in model_paths.items():
            if os.path.exists(path):
                print(f"\n   🔑 {step} 키 매핑 분석...")
                try:
                    # 체크포인트 로드
                    checkpoint = torch.load(path, map_location='cpu')
                    
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                    
                    print(f"      📊 체크포인트 키 수: {len(state_dict)}")
                    
                    # 키 패턴 분석
                    key_patterns = {}
                    for key in state_dict.keys():
                        parts = key.split('.')
                        if len(parts) > 0:
                            prefix = parts[0]
                            key_patterns[prefix] = key_patterns.get(prefix, 0) + 1
                    
                    print(f"      🏷️ 주요 키 패턴:")
                    for pattern, count in sorted(key_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"         - {pattern}: {count}개")
                    
                    # 텐서 크기 분석
                    tensor_sizes = {}
                    for key, tensor in state_dict.items():
                        if hasattr(tensor, 'shape'):
                            size_str = 'x'.join(map(str, tensor.shape))
                            tensor_sizes[size_str] = tensor_sizes.get(size_str, 0) + 1
                    
                    print(f"      📏 주요 텐서 크기:")
                    for size, count in sorted(tensor_sizes.items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"         - {size}: {count}개")
                        
                except Exception as e:
                    print(f"      ❌ 키 매핑 분석 오류: {e}")
        
        # 5. ModelLoader 직접 테스트
        print("\n5️⃣ ModelLoader 직접 테스트...")
        
        loader = ModelLoader()
        
        for step in ['human_parsing', 'pose_estimation', 'cloth_segmentation']:
            print(f"\n   🚀 {step} ModelLoader 테스트...")
            try:
                model = loader.load_model_for_step(step)
                if model:
                    param_count = sum(p.numel() for p in model.parameters())
                    print(f"      ✅ 로딩 성공: {param_count:,} 파라미터")
                else:
                    print(f"      ❌ 로딩 실패")
            except Exception as e:
                print(f"      ❌ 로딩 오류: {e}")
        
        print("\n🎉 모델 로딩 디버깅 완료!")
        
    except Exception as e:
        print(f"❌ 디버깅 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_loading()