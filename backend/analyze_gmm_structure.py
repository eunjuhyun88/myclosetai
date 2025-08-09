#!/usr/bin/env python3
"""
GMM 체크포인트 구조 분석 스크립트
"""

import torch
import os
import json
from collections import defaultdict

def analyze_gmm_checkpoint():
    """GMM 체크포인트의 실제 구조를 분석"""
    
    checkpoint_path = "ai_models/step_04/gmm.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        return
    
    print(f"🔍 GMM 체크포인트 분석: {checkpoint_path}")
    
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
        
        print(f"✅ 체크포인트 로드 성공")
        print(f"📊 총 키 수: {len(state_dict)}")
        
        # 키 구조 분석
        key_structure = defaultdict(list)
        key_patterns = defaultdict(int)
        
        for key, tensor in state_dict.items():
            # 키 패턴 분석
            parts = key.split('.')
            if len(parts) >= 2:
                pattern = f"{parts[0]}.{parts[1]}"
                key_patterns[pattern] += 1
            
            # 텐서 정보
            shape = list(tensor.shape)
            dtype = str(tensor.dtype)
            
            key_structure[key] = {
                'shape': shape,
                'dtype': dtype,
                'num_params': tensor.numel()
            }
        
        print(f"\n🔍 키 패턴 분석:")
        for pattern, count in sorted(key_patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"   {pattern}.* : {count}개")
        
        print(f"\n📋 상세 키 구조:")
        for key, info in key_structure.items():
            print(f"   {key}: {info['shape']} ({info['dtype']}) - {info['num_params']}개 파라미터")
        
        # 주요 모듈별 분석
        modules = defaultdict(list)
        for key in state_dict.keys():
            if '.' in key:
                module = key.split('.')[0]
                modules[module].append(key)
        
        print(f"\n🏗️ 모듈별 구조:")
        for module, keys in modules.items():
            print(f"\n   📦 {module} 모듈 ({len(keys)}개 키):")
            for key in sorted(keys):
                info = key_structure[key]
                print(f"      {key}: {info['shape']}")
        
        # 결과 저장
        analysis_result = {
            'total_keys': len(state_dict),
            'key_patterns': dict(key_patterns),
            'key_structure': dict(key_structure),
            'modules': dict(modules)
        }
        
        with open('gmm_checkpoint_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 분석 결과가 'gmm_checkpoint_analysis.json'에 저장되었습니다.")
        
        return analysis_result
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return None

if __name__ == "__main__":
    analyze_gmm_checkpoint()
