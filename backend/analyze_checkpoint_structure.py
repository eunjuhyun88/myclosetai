#!/usr/bin/env python3
"""
🔍 체크포인트 구조 분석 스크립트
================================================================================
✅ 실제 체크포인트 파일의 키 구조 분석
✅ 모델 아키텍처와의 매칭 분석
✅ 정확한 모델 구조 생성 가이드
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

def analyze_checkpoint_structure(checkpoint_path):
    """체크포인트 파일의 구조를 분석"""
    print(f"🔍 체크포인트 구조 분석: {checkpoint_path.name}")
    print(f"📁 경로: {checkpoint_path}")
    print(f"📊 크기: {checkpoint_path.stat().st_size / (1024*1024):.1f}MB")
    
    try:
        # 체크포인트 로딩
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"📦 체크포인트 타입: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"📋 최상위 키들: {list(checkpoint.keys())}")
            
            # state_dict 찾기
            state_dict = None
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("✅ 'state_dict' 키 발견")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("✅ 'model' 키 발견")
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("✅ 'model_state_dict' 키 발견")
            else:
                # 최상위가 state_dict인 경우
                state_dict = checkpoint
                print("✅ 최상위가 state_dict")
            
            if state_dict:
                print(f"📊 state_dict 키 수: {len(state_dict)}")
                
                # 키 패턴 분석
                key_patterns = {}
                for key in list(state_dict.keys())[:20]:  # 처음 20개만 분석
                    parts = key.split('.')
                    if len(parts) >= 2:
                        layer_type = parts[1] if len(parts) > 1 else parts[0]
                        if layer_type not in key_patterns:
                            key_patterns[layer_type] = []
                        key_patterns[layer_type].append(key)
                
                print("🔍 키 패턴 분석:")
                for layer_type, keys in key_patterns.items():
                    print(f"   {layer_type}: {len(keys)}개 키")
                    for key in keys[:3]:  # 처음 3개만 표시
                        print(f"     - {key}")
                    if len(keys) > 3:
                        print(f"     ... (총 {len(keys)}개)")
                
                # 텐서 크기 분석
                print("📏 텐서 크기 분석:")
                total_params = 0
                for key, tensor in list(state_dict.items())[:10]:  # 처음 10개만
                    if hasattr(tensor, 'shape'):
                        params = tensor.numel()
                        total_params += params
                        print(f"   {key}: {tensor.shape} ({params:,} 파라미터)")
                
                print(f"📊 총 파라미터 수 (샘플): {total_params:,}")
                
                return {
                    'success': True,
                    'state_dict': state_dict,
                    'total_keys': len(state_dict),
                    'key_patterns': key_patterns
                }
        
        print("❌ 유효한 state_dict를 찾을 수 없음")
        return {'success': False, 'error': '유효한 state_dict를 찾을 수 없습니다'}
        
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """메인 함수"""
    print("🔍 체크포인트 구조 분석")
    print("=" * 80)
    
    # AI 모델 디렉토리에서 체크포인트 파일들 찾기
    ai_models_dir = backend_root / "ai_models"
    
    if not ai_models_dir.exists():
        print(f"❌ AI 모델 디렉토리가 존재하지 않음: {ai_models_dir}")
        return
    
    # 체크포인트 파일들 찾기
    checkpoint_files = []
    for ext in ['*.pth', '*.pt']:
        checkpoint_files.extend(ai_models_dir.rglob(ext))
    
    if not checkpoint_files:
        print("❌ 체크포인트 파일을 찾을 수 없음")
        return
    
    print(f"✅ {len(checkpoint_files)}개의 체크포인트 파일 발견")
    
    # 처음 3개 파일만 분석
    for i, checkpoint_path in enumerate(checkpoint_files[:3], 1):
        print(f"\n{'='*60}")
        result = analyze_checkpoint_structure(checkpoint_path)
        
        if result['success']:
            print(f"✅ 분석 완료: {checkpoint_path.name}")
        else:
            print(f"❌ 분석 실패: {result['error']}")
        
        print(f"{'='*60}")

if __name__ == "__main__":
    main() 