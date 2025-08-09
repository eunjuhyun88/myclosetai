#!/usr/bin/env python3
"""
GMM 모델의 매칭되지 않는 키들을 분석하는 스크립트
"""

import torch
import os
from collections import defaultdict

def analyze_gmm_missing_keys():
    """GMM 체크포인트에서 매칭되지 않는 키들을 분석"""
    
    checkpoint_path = "ai_models/step_04/gmm.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
    
    print(f"🔍 GMM 매칭되지 않는 키 분석: {checkpoint_path}")
    
    try:
        # 체크포인트 로드
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
        
        print(f"📊 총 키 수: {len(state_dict)}")
        
        # 현재 모델 아키텍처에서 생성되는 키들
        from app.ai_pipeline.utils.model_architectures import GMMModel
        
        model = GMMModel()
        model_state_dict = model.state_dict()
        
        print(f"🏗️ 모델에서 생성되는 키 수: {len(model_state_dict)}")
        
        # 매칭되지 않는 키들 찾기
        checkpoint_keys = set(state_dict.keys())
        model_keys = set(model_state_dict.keys())
        
        missing_in_model = checkpoint_keys - model_keys
        extra_in_model = model_keys - checkpoint_keys
        
        print(f"\n❌ 체크포인트에 있지만 모델에 없는 키들 ({len(missing_in_model)}개):")
        for key in sorted(missing_in_model):
            print(f"   - {key}")
        
        print(f"\n➕ 모델에 있지만 체크포인트에 없는 키들 ({len(extra_in_model)}개):")
        for key in sorted(extra_in_model):
            print(f"   - {key}")
        
        # 매칭률 계산
        matched_keys = checkpoint_keys & model_keys
        total_keys = len(checkpoint_keys)
        match_rate = len(matched_keys) / total_keys * 100
        
        print(f"\n📊 매칭 통계:")
        print(f"   - 총 키 수: {total_keys}")
        print(f"   - 매칭된 키 수: {len(matched_keys)}")
        print(f"   - 매칭되지 않은 키 수: {len(missing_in_model)}")
        print(f"   - 매칭률: {match_rate:.1f}%")
        
        # 매칭되지 않는 키들의 패턴 분석
        if missing_in_model:
            print(f"\n🔍 매칭되지 않는 키 패턴 분석:")
            pattern_count = defaultdict(int)
            for key in missing_in_model:
                parts = key.split('.')
                if len(parts) >= 2:
                    pattern = f"{parts[0]}.{parts[1]}"
                    pattern_count[pattern] += 1
            
            for pattern, count in sorted(pattern_count.items(), key=lambda x: x[1], reverse=True):
                print(f"   - {pattern}.* : {count}개")
        
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")

if __name__ == "__main__":
    analyze_gmm_missing_keys()
