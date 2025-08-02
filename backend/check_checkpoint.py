#!/usr/bin/env python3
"""
체크포인트의 conv1.weight 형태 확인
"""
import torch
import sys
import os

def check_checkpoint():
    checkpoint_path = 'ai_models/Self-Correction-Human-Parsing/exp-schp-201908261155-atr.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없음: {checkpoint_path}")
        return
    
    try:
        print(f"🔍 체크포인트 로딩 중: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"📊 체크포인트 키들: {list(checkpoint.keys())}")
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"📊 state_dict 키 수: {len(state_dict)}")
            print(f"📊 state_dict 키들 (처음 10개): {list(state_dict.keys())[:10]}")
            
            # conv1.weight 찾기
            conv1_key = None
            for key in state_dict.keys():
                if 'conv1' in key and 'weight' in key:
                    conv1_key = key
                    break
            
            if conv1_key:
                conv1_weight = state_dict[conv1_key]
                print(f"🎯 {conv1_key} 형태: {conv1_weight.shape}")
                print(f"🎯 kernel_size 추정: {conv1_weight.shape[2]}x{conv1_weight.shape[3]}")
            else:
                print("❌ conv1.weight를 찾을 수 없음")
                
            # 다른 중요한 레이어들도 확인
            important_keys = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'aspp', 'classifier']
            for key in important_keys:
                for state_key in state_dict.keys():
                    if key in state_key and 'weight' in state_key:
                        weight = state_dict[state_key]
                        print(f"🔍 {state_key}: {weight.shape}")
                        break
        else:
            print("❌ state_dict를 찾을 수 없음")
            
    except Exception as e:
        print(f"❌ 체크포인트 로딩 실패: {e}")

if __name__ == "__main__":
    check_checkpoint() 