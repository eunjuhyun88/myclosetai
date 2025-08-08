#!/usr/bin/env python3
"""
빠른 체크포인트 분석 스크립트
"""

import torch
import os

def analyze_checkpoint(checkpoint_path):
    """체크포인트를 분석합니다."""
    try:
        print(f"\n=== 분석: {os.path.basename(checkpoint_path)} ===")
        
        # 파일 크기
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print(f"파일 크기: {file_size:.2f} MB")
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"체크포인트 타입: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"체크포인트 키들: {list(checkpoint.keys())}")
            
            # state_dict 분석
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"State dict 키 개수: {len(state_dict)}")
                print(f"State dict 키들 (처음 10개): {list(state_dict.keys())[:10]}")
                
                # 파라미터 수 계산
                total_params = sum(tensor.numel() for tensor in state_dict.values())
                print(f"총 파라미터 수: {total_params:,}")
                
                # 주요 레이어 차원
                print("\n주요 레이어 차원:")
                for i, (key, value) in enumerate(state_dict.items()):
                    print(f"  {key}: {value.shape}")
                    if i >= 9:  # 처음 10개만
                        break
                        
            # 직접 state_dict인 경우
            elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                print(f"직접 state_dict, 키 개수: {len(checkpoint)}")
                total_params = sum(tensor.numel() for tensor in checkpoint.values())
                print(f"총 파라미터 수: {total_params:,}")
                
                print("\n주요 레이어 차원:")
                for i, (key, value) in enumerate(checkpoint.items()):
                    print(f"  {key}: {value.shape}")
                    if i >= 9:
                        break
                        
    except Exception as e:
        print(f"에러: {e}")

def main():
    """메인 함수"""
    # 주요 체크포인트들 분석
    checkpoints = [
        "ai_models/step_01_human_parsing/graphonomy.pth",
        "ai_models/step_02_pose_estimation/hrnet_w48_coco_256x192.pth",
        "ai_models/step_03_cloth_segmentation/u2net.pth",
        "ai_models/step_04_geometric_matching/gmm_final.pth",
        "ai_models/step_05_cloth_warping/tom_final.pth",
        "ai_models/step_06_virtual_fitting/hrviton_final.pth"
    ]
    
    for checkpoint_path in checkpoints:
        if os.path.exists(checkpoint_path):
            analyze_checkpoint(checkpoint_path)
        else:
            print(f"\n파일 없음: {checkpoint_path}")

if __name__ == "__main__":
    main()
