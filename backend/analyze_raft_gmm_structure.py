#!/usr/bin/env python3
"""
🔍 RAFT & GMM 체크포인트 구조 분석
================================================================================
✅ RAFT와 GMM 체크포인트의 실제 키 구조 분석
✅ 정확한 모델 아키텍처 생성 가이드
================================================================================
"""

import os
import sys
import torch
from pathlib import Path

# 프로젝트 경로 설정
current_file = Path(__file__).resolve()
backend_root = current_file.parent
sys.path.insert(0, str(backend_root))
sys.path.insert(0, str(backend_root / "app"))

def analyze_raft_gmm_structure():
    """RAFT와 GMM 체크포인트 구조 분석"""
    print("🔍 RAFT & GMM 체크포인트 구조 분석")
    print("=" * 80)
    
    # 체크포인트 파일들
    checkpoint_files = [
        backend_root / "ai_models" / "step_04" / "raft.pth",
        backend_root / "ai_models" / "step_04" / "gmm.pth"
    ]
    
    for checkpoint_path in checkpoint_files:
        if not checkpoint_path.exists():
            print(f"❌ 체크포인트 파일이 존재하지 않음: {checkpoint_path}")
            continue
            
        print(f"\n📁 체크포인트: {checkpoint_path.name}")
        print(f"📊 크기: {checkpoint_path.stat().st_size / (1024*1024):.1f}MB")
        
        try:
            # 체크포인트 로딩
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # state_dict 추출
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            print(f"📊 키 수: {len(state_dict)}")
            
            # 키 구조 분석
            print("🔍 키 구조:")
            for i, (key, value) in enumerate(list(state_dict.items())[:10]):  # 처음 10개만
                print(f"   {key}: {value.shape}")
            
            if len(state_dict) > 10:
                print(f"   ... (총 {len(state_dict)}개 키)")
            
            # 키 패턴 분석
            key_patterns = {}
            for key in state_dict.keys():
                parts = key.split('.')
                if len(parts) >= 2:
                    pattern = f"{parts[0]}.{parts[1]}"
                    key_patterns[pattern] = key_patterns.get(pattern, 0) + 1
            
            print(f"\n📊 키 패턴 분석:")
            for pattern, count in sorted(key_patterns.items())[:10]:
                print(f"   {pattern}.*: {count}개")
            
        except Exception as e:
            print(f"❌ 분석 실패: {e}")

if __name__ == "__main__":
    analyze_raft_gmm_structure()
