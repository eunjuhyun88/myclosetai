#!/usr/bin/env python3
"""
🔍 체크포인트 키 구조 디버깅
================================================================================
✅ 실제 체크포인트 키 구조 확인
✅ 생성된 모델 키 구조 확인
✅ 키 매칭 분석
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

def debug_checkpoint_keys():
    """체크포인트 키 구조 디버깅"""
    print("🔍 체크포인트 키 구조 디버깅")
    print("=" * 80)
    
    try:
        # 1. 실제 체크포인트 로딩
        checkpoint_path = backend_root / "ai_models" / "openpose.pth"
        
        if not checkpoint_path.exists():
            print(f"❌ 체크포인트 파일이 존재하지 않음: {checkpoint_path}")
            return
        
        print(f"📁 체크포인트: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # state_dict 추출
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        print(f"📊 실제 체크포인트 키 수: {len(state_dict)}")
        print("🔍 실제 체크포인트 키들:")
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            if hasattr(tensor, 'shape'):
                print(f"   {key}: {tensor.shape}")
        
        # 2. 생성된 모델 로딩
        print(f"\n🏗️ 생성된 모델 키 구조:")
        from app.ai_pipeline.utils.model_architectures import OpenPoseModel
        
        model = OpenPoseModel()
        model_state_dict = model.state_dict()
        
        print(f"📊 생성된 모델 키 수: {len(model_state_dict)}")
        print("🔍 생성된 모델 키들:")
        for key in sorted(model_state_dict.keys()):
            tensor = model_state_dict[key]
            if hasattr(tensor, 'shape'):
                print(f"   {key}: {tensor.shape}")
        
        # 3. 키 매칭 분석
        print(f"\n🔍 키 매칭 분석:")
        matched_keys = []
        unmatched_checkpoint_keys = []
        unmatched_model_keys = []
        
        for key in state_dict.keys():
            if key in model_state_dict:
                matched_keys.append(key)
            else:
                unmatched_checkpoint_keys.append(key)
        
        for key in model_state_dict.keys():
            if key not in state_dict:
                unmatched_model_keys.append(key)
        
        print(f"✅ 매칭된 키: {len(matched_keys)}개")
        for key in matched_keys:
            print(f"   ✅ {key}")
        
        print(f"❌ 체크포인트에만 있는 키: {len(unmatched_checkpoint_keys)}개")
        for key in unmatched_checkpoint_keys[:5]:  # 처음 5개만
            print(f"   ❌ {key}")
        
        print(f"❌ 모델에만 있는 키: {len(unmatched_model_keys)}개")
        for key in unmatched_model_keys[:5]:  # 처음 5개만
            print(f"   ❌ {key}")
        
        # 4. 매칭률 계산
        match_rate = len(matched_keys) / len(model_state_dict) if model_state_dict else 0
        print(f"\n📊 매칭률: {match_rate:.1%} ({len(matched_keys)}/{len(model_state_dict)})")
        
        return {
            'checkpoint_keys': list(state_dict.keys()),
            'model_keys': list(model_state_dict.keys()),
            'matched_keys': matched_keys,
            'match_rate': match_rate
        }
        
    except Exception as e:
        print(f"❌ 디버깅 실패: {e}")
        return None

if __name__ == "__main__":
    debug_checkpoint_keys()
