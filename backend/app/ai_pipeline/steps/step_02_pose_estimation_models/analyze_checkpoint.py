#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 02: Pose Estimation 체크포인트 분석
========================================================

다운로드된 체크포인트의 구조를 분석합니다.
"""

import torch
import os

def analyze_checkpoint(checkpoint_path):
    """체크포인트 구조 분석"""
    print(f"🔍 체크포인트 분석: {checkpoint_path}")
    print("=" * 60)
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"📊 총 키 개수: {len(checkpoint)}")
    print("\n🔑 키 구조 분석:")
    
    # 키를 카테고리별로 분류
    conv_keys = []
    bn_keys = []
    linear_keys = []
    other_keys = []
    
    for key, tensor in checkpoint.items():
        if 'conv' in key:
            conv_keys.append((key, tensor.shape))
        elif 'bn' in key or 'norm' in key:
            bn_keys.append((key, tensor.shape))
        elif 'linear' in key or 'fc' in key:
            linear_keys.append((key, tensor.shape))
        else:
            other_keys.append((key, tensor.shape))
    
    print(f"\n📐 Convolution 레이어 ({len(conv_keys)}개):")
    for key, shape in conv_keys[:10]:  # 처음 10개만 표시
        print(f"  {key}: {shape}")
    if len(conv_keys) > 10:
        print(f"  ... 및 {len(conv_keys) - 10}개 더")
    
    print(f"\n⚖️ BatchNorm 레이어 ({len(bn_keys)}개):")
    for key, shape in bn_keys[:10]:
        print(f"  {key}: {shape}")
    if len(bn_keys) > 10:
        print(f"  ... 및 {len(bn_keys) - 10}개 더")
    
    print(f"\n🔗 Linear 레이어 ({len(linear_keys)}개):")
    for key, shape in linear_keys[:10]:
        print(f"  {key}: {shape}")
    if len(linear_keys) > 10:
        print(f"  ... 및 {len(linear_keys) - 10}개 더")
    
    print(f"\n🔧 기타 레이어 ({len(other_keys)}개):")
    for key, shape in other_keys[:10]:
        print(f"  {key}: {shape}")
    if len(other_keys) > 10:
        print(f"  ... 및 {len(other_keys) - 10}개 더")
    
    # 모델 아키텍처 추정
    print("\n🏗️ 모델 아키텍처 추정:")
    
    # 입력 채널 수 추정
    input_channels = None
    for key, shape in conv_keys:
        if 'conv1' in key and len(shape) == 4:
            input_channels = shape[1]
            break
    
    if input_channels:
        print(f"  입력 채널 수: {input_channels}")
    
    # 출력 채널 수 추정
    output_channels = None
    for key, shape in conv_keys:
        if 'conv' in key and len(shape) == 4:
            output_channels = shape[0]
            break
    
    if output_channels:
        print(f"  출력 채널 수: {output_channels}")
    
    # 레이어 깊이 추정
    layer_depths = {}
    for key in checkpoint.keys():
        if '.' in key:
            layer_name = key.split('.')[0]
            if layer_name not in layer_depths:
                layer_depths[layer_name] = 0
            layer_depths[layer_name] += 1
    
    print(f"  주요 레이어 구조:")
    for layer, count in sorted(layer_depths.items()):
        print(f"    {layer}: {count}개 서브레이어")
    
    return checkpoint

if __name__ == "__main__":
    checkpoint_path = "checkpoints/checkpoints/hrnet_pose_2025_advanced.pth"
    
    if os.path.exists(checkpoint_path):
        checkpoint = analyze_checkpoint(checkpoint_path)
    else:
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
