#!/usr/bin/env python3
"""
Step 5와 Step 6 체크포인트 분석 도구
"""

import os
import sys
import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

def analyze_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """체크포인트 파일을 분석합니다."""
    try:
        if not os.path.exists(checkpoint_path):
            return {"error": f"파일이 존재하지 않음: {checkpoint_path}"}
        
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        
        # PyTorch 체크포인트 로드 시도
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                # state_dict 형태
                keys = list(checkpoint.keys())
                total_params = sum(p.numel() for p in checkpoint.values() if hasattr(p, 'numel'))
                
                return {
                    "type": "state_dict",
                    "file_size_mb": round(file_size, 2),
                    "keys": keys,
                    "num_keys": len(keys),
                    "total_params": total_params,
                    "sample_keys": keys[:10] if len(keys) > 10 else keys
                }
            elif hasattr(checkpoint, 'state_dict'):
                # 모델 객체
                state_dict = checkpoint.state_dict()
                keys = list(state_dict.keys())
                total_params = sum(p.numel() for p in state_dict.values())
                
                return {
                    "type": "model_object",
                    "file_size_mb": round(file_size, 2),
                    "keys": keys,
                    "num_keys": len(keys),
                    "total_params": total_params,
                    "sample_keys": keys[:10] if len(keys) > 10 else keys
                }
            else:
                return {
                    "type": "unknown",
                    "file_size_mb": round(file_size, 2),
                    "checkpoint_type": str(type(checkpoint))
                }
                
        except Exception as e:
            return {
                "error": f"PyTorch 로드 실패: {str(e)}",
                "file_size_mb": round(file_size, 2)
            }
            
    except Exception as e:
        return {"error": f"분석 실패: {str(e)}"}

def find_checkpoint_files(directory: str) -> List[str]:
    """디렉토리에서 체크포인트 파일들을 찾습니다."""
    checkpoint_files = []
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.pth', '.ckpt', '.safetensors')):
                    checkpoint_files.append(os.path.join(root, file))
    return checkpoint_files

def analyze_step5_checkpoints():
    """Step 5 체크포인트들을 분석합니다."""
    print("🔍 Step 5 체크포인트 분석 시작...")
    
    step5_path = "ai_models/step_05"
    checkpoint_files = find_checkpoint_files(step5_path)
    
    if not checkpoint_files:
        print("⚠️ Step 5 디렉토리에서 체크포인트 파일을 찾을 수 없습니다.")
        return {}
    
    results = {}
    
    for checkpoint_path in checkpoint_files:
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"\n📁 분석 중: {checkpoint_name}")
        
        result = analyze_checkpoint(checkpoint_path)
        results[checkpoint_name] = result
        
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(f"✅ 타입: {result['type']}")
            print(f"📊 크기: {result['file_size_mb']}MB")
            print(f"🔑 키 개수: {result['num_keys']}")
            if 'sample_keys' in result:
                print(f"🔑 샘플 키: {result['sample_keys'][:3]}")
    
    return results

def analyze_step6_checkpoints():
    """Step 6 체크포인트들을 분석합니다."""
    print("\n🔍 Step 6 체크포인트 분석 시작...")
    
    step6_path = "ai_models/step_06_virtual_fitting"
    checkpoint_files = find_checkpoint_files(step6_path)
    
    if not checkpoint_files:
        print("⚠️ Step 6 디렉토리에서 체크포인트 파일을 찾을 수 없습니다.")
        return {}
    
    results = {}
    
    for checkpoint_path in checkpoint_files[:10]:  # 처음 10개만
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"\n📁 분석 중: {checkpoint_name}")
        
        result = analyze_checkpoint(checkpoint_path)
        results[checkpoint_name] = result
        
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(f"✅ 타입: {result['type']}")
            print(f"📊 크기: {result['file_size_mb']}MB")
            print(f"🔑 키 개수: {result['num_keys']}")
            if 'sample_keys' in result:
                print(f"🔑 샘플 키: {result['sample_keys'][:3]}")
    
    return results

def analyze_safetensors():
    """SafeTensors 파일들을 분석합니다."""
    print("\n🔍 SafeTensors 파일 분석 시작...")
    
    step6_path = "ai_models/step_06_virtual_fitting/ootdiffusion"
    
    try:
        from safetensors import safe_open
        
        safetensor_files = []
        for root, dirs, files in os.walk(step6_path):
            for file in files:
                if file.endswith('.safetensors'):
                    safetensor_files.append(os.path.join(root, file))
        
        if not safetensor_files:
            print("⚠️ SafeTensors 파일을 찾을 수 없습니다.")
            return {}
        
        results = {}
        
        for file_path in safetensor_files[:5]:  # 처음 5개만
            print(f"\n📁 분석 중: {os.path.basename(file_path)}")
            
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata()
                    tensor_names = f.keys()
                    
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    
                    result = {
                        "type": "safetensors",
                        "file_size_mb": round(file_size, 2),
                        "num_tensors": len(tensor_names),
                        "tensor_names": list(tensor_names)[:10],
                        "metadata": metadata
                    }
                    
                    results[os.path.basename(file_path)] = result
                    
                    print(f"✅ 타입: safetensors")
                    print(f"📊 크기: {result['file_size_mb']}MB")
                    print(f"🔑 텐서 개수: {result['num_tensors']}")
                    print(f"🔑 샘플 텐서: {result['tensor_names'][:3]}")
                    
            except Exception as e:
                print(f"❌ SafeTensors 로드 실패: {str(e)}")
                results[os.path.basename(file_path)] = {"error": str(e)}
                
    except ImportError:
        print("⚠️ SafeTensors 라이브러리가 설치되지 않음")
        return {}
    
    return results

def main():
    """메인 분석 함수"""
    print("🚀 Step 5 & Step 6 체크포인트 분석 도구")
    print("=" * 50)
    
    # Step 5 분석
    step5_results = analyze_step5_checkpoints()
    
    # Step 6 분석  
    step6_results = analyze_step6_checkpoints()
    
    # SafeTensors 분석
    safetensor_results = analyze_safetensors()
    
    # 결과 저장
    all_results = {
        "step5": step5_results,
        "step6": step6_results,
        "safetensors": safetensor_results
    }
    
    with open("step5_step6_checkpoint_analysis.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 분석 완료! 결과가 step5_step6_checkpoint_analysis.json에 저장되었습니다.")
    
    # 요약 출력
    print("\n📋 분석 요약:")
    print(f"Step 5 체크포인트: {len(step5_results)}개")
    print(f"Step 6 체크포인트: {len(step6_results)}개") 
    print(f"SafeTensors 파일: {len(safetensor_results)}개")

if __name__ == "__main__":
    main()
