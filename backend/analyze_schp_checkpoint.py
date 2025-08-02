#!/usr/bin/env python3
"""
SCHP 체크포인트 상세 구조 분석
"""
import torch
import os
from collections import defaultdict

def analyze_schp_checkpoint():
    checkpoint_path = 'ai_models/Self-Correction-Human-Parsing/exp-schp-201908261155-atr.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없음: {checkpoint_path}")
        return
    
    try:
        print(f"🔍 SCHP 체크포인트 상세 분석 시작: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"📊 체크포인트 최상위 키: {list(checkpoint.keys())}")
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"📊 state_dict 총 키 수: {len(state_dict)}")
            
            # context_encoding 관련 키들 찾기
            print(f"\n🔍 CONTEXT_ENCODING 관련 키들:")
            context_keys = []
            for key in state_dict.keys():
                if 'context_encoding' in key:
                    context_keys.append(key)
                    tensor = state_dict[key]
                    print(f"  {key}: {tensor.shape}")
            
            # fushion 관련 키들 찾기
            print(f"\n🔍 FUSHION 관련 키들:")
            fushion_keys = []
            for key in state_dict.keys():
                if 'fushion' in key:
                    fushion_keys.append(key)
                    tensor = state_dict[key]
                    print(f"  {key}: {tensor.shape}")
            
            # edge 관련 키들 찾기
            print(f"\n🔍 EDGE 관련 키들:")
            edge_keys = []
            for key in state_dict.keys():
                if 'edge' in key:
                    edge_keys.append(key)
                    tensor = state_dict[key]
                    print(f"  {key}: {tensor.shape}")
            
            # decoder 관련 키들 찾기
            print(f"\n🔍 DECODER 관련 키들:")
            decoder_keys = []
            for key in state_dict.keys():
                if 'decoder' in key:
                    decoder_keys.append(key)
                    tensor = state_dict[key]
                    print(f"  {key}: {tensor.shape}")
            
            # layer4 관련 키들 (마지막 백본 레이어)
            print(f"\n🔍 LAYER4 관련 키들:")
            layer4_keys = []
            for key in state_dict.keys():
                if 'layer4' in key and 'conv' in key:
                    layer4_keys.append(key)
                    tensor = state_dict[key]
                    print(f"  {key}: {tensor.shape}")
            
            # 모델 구조 추정
            print(f"\n🔍 모델 구조 추정:")
            
            # 입력 채널 수
            conv1_weight = state_dict['module.conv1.weight']
            input_channels = conv1_weight.shape[1]
            print(f"  입력 채널 수: {input_channels}")
            
            # 출력 클래스 수
            num_classes = None
            for key in ['module.fushion.3.weight', 'module.classifier.weight']:
                if key in state_dict:
                    num_classes = state_dict[key].shape[0]
                    print(f"  출력 클래스 수: {num_classes} (from {key})")
                    break
            
            # layer4 출력 채널 수
            layer4_output = None
            for key in state_dict.keys():
                if 'layer4' in key and 'conv' in key and 'weight' in key and 'downsample' not in key:
                    # layer4의 마지막 conv 레이어 찾기
                    if 'conv3' in key or 'conv2' in key:
                        layer4_output = state_dict[key].shape[0]
                        print(f"  layer4 출력 채널: {layer4_output} (from {key})")
                        break
            
            print(f"\n✅ 체크포인트 분석 완료!")
            
        else:
            print("❌ state_dict를 찾을 수 없음")
            
    except Exception as e:
        print(f"❌ 체크포인트 분석 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_schp_checkpoint() 