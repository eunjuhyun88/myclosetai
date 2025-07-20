#!/usr/bin/env python3
"""
🔧 개선된 MyCloset AI 모델 검증 스크립트 v2.0
✅ unet_vton 폴더 검증 로직 완전 개선
✅ 더 정확한 모델 탐지
✅ 상세한 디버깅 정보 제공
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def check_unet_vton_detailed(step_06_path: Path) -> Dict:
    """unet_vton 상세 검증"""
    result = {
        "found": False,
        "type": None,
        "size_mb": 0.0,
        "files": [],
        "issues": [],
        "path": None
    }
    
    # 가능한 unet_vton 경로들
    possible_paths = [
        step_06_path / "unet_vton",
        step_06_path / "unet_vton.pth", 
        step_06_path / "unet_vton.safetensors",
        step_06_path / "unet_vton.bin"
    ]
    
    for path in possible_paths:
        if path.exists():
            result["found"] = True
            result["path"] = path
            
            if path.is_dir():
                result["type"] = "folder"
                # 폴더 내 파일들 확인
                files = list(path.rglob("*"))
                result["files"] = [f.name for f in files if f.is_file()]
                
                # 총 크기 계산
                total_size = 0
                for file_path in files:
                    if file_path.is_file():
                        try:
                            total_size += file_path.stat().st_size
                        except OSError:
                            pass
                result["size_mb"] = total_size / (1024 * 1024)
                
                # 필수 파일 확인
                model_files = [f for f in result["files"] if any(ext in f.lower() for ext in ['.pth', '.safetensors', '.bin', '.ckpt'])]
                if not model_files:
                    result["issues"].append("모델 파일이 없음")
                elif len(result["files"]) < 2:
                    result["issues"].append("파일 수가 너무 적음")
                    
            elif path.is_file():
                result["type"] = "file"
                result["files"] = [path.name]
                result["size_mb"] = path.stat().st_size / (1024 * 1024)
                
            # 크기 검증
            if result["size_mb"] < 10:
                result["issues"].append(f"크기가 너무 작음: {result['size_mb']:.1f}MB")
            elif result["size_mb"] > 10000:
                result["issues"].append(f"크기가 너무 큼: {result['size_mb']:.1f}MB")
                
            break
    
    return result

def verify_step_06_enhanced(checkpoints_path: Path) -> Dict:
    """Step 06 Virtual Fitting 향상된 검증"""
    step_06_path = checkpoints_path / "step_06_virtual_fitting"
    
    if not step_06_path.exists():
        return {
            "status": "error",
            "message": "step_06_virtual_fitting 폴더가 존재하지 않음"
        }
    
    # 전체 모델 수 및 크기
    all_files = list(step_06_path.rglob("*"))
    model_files = [f for f in all_files if f.is_file()]
    total_size = sum(f.stat().st_size for f in model_files if f.is_file()) / (1024 * 1024)
    
    # unet_vton 상세 검증
    unet_result = check_unet_vton_detailed(step_06_path)
    
    # 기타 필수 모델 확인
    required_models = {
        "diffusion_pytorch_model.safetensors": False,
        "pytorch_model.bin": False,
        "config.json": False
    }
    
    for model_name in required_models:
        for file_path in model_files:
            if model_name in file_path.name:
                required_models[model_name] = True
                break
    
    # 결과 정리
    result = {
        "step_name": "Virtual Fitting",
        "model_count": len(model_files),
        "total_size_mb": total_size,
        "unet_vton": unet_result,
        "required_models": required_models,
        "missing_required": [name for name, found in required_models.items() if not found],
        "status": "success" if unet_result["found"] and not unet_result["issues"] else "warning"
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description="개선된 모델 검증 스크립트")
    parser.add_argument("--step", type=int, help="특정 Step만 검증")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    
    args = parser.parse_args()
    
    # 프로젝트 루트 찾기
    current_dir = Path.cwd()
    while current_dir != current_dir.parent:
        if (current_dir / "backend").exists():
            project_root = current_dir
            break
        current_dir = current_dir.parent
    else:
        print("❌ 프로젝트 루트를 찾을 수 없습니다.")
        return
    
    checkpoints_path = project_root / "backend" / "app" / "ai_pipeline" / "models" / "checkpoints"
    
    if not checkpoints_path.exists():
        print(f"❌ 체크포인트 경로가 존재하지 않습니다: {checkpoints_path}")
        return
    
    print("🔍 개선된 MyCloset AI 모델 검증 v2.0")
    print("=" * 50)
    print(f"📁 체크포인트 경로: {checkpoints_path}")
    
    if args.step == 6:
        print("\n🎯 Step 06 Virtual Fitting 상세 검증")
        print("-" * 30)
        
        result = verify_step_06_enhanced(checkpoints_path)
        
        print(f"📝 {result['step_name']}")
        print(f"🔢 모델 수: {result['model_count']}개")
        print(f"💾 총 크기: {result['total_size_mb']:.1f}MB")
        
        # unet_vton 상세 정보
        unet = result['unet_vton']
        if unet['found']:
            status_emoji = "✅" if not unet['issues'] else "⚠️"
            print(f"\n{status_emoji} unet_vton 발견!")
            print(f"   📍 경로: {unet['path']}")
            print(f"   📊 타입: {unet['type']}")
            print(f"   💾 크기: {unet['size_mb']:.1f}MB")
            if unet['type'] == 'folder':
                print(f"   📁 파일 수: {len(unet['files'])}개")
                if args.debug:
                    print(f"   📄 파일들: {', '.join(unet['files'][:5])}")
            
            if unet['issues']:
                print(f"   🚨 이슈: {', '.join(unet['issues'])}")
        else:
            print("\n❌ unet_vton을 찾을 수 없습니다!")
        
        # 기타 필수 모델들
        print(f"\n📋 기타 필수 모델:")
        for model, found in result['required_models'].items():
            status = "✅" if found else "❌"
            print(f"   {status} {model}")
        
        # 전체 상태
        if result['status'] == 'success':
            print(f"\n🎉 Step 06 검증 완료! 모든 필수 모델이 준비되었습니다.")
        else:
            print(f"\n⚠️ Step 06에 일부 개선이 필요합니다.")
            if result['missing_required']:
                print(f"   누락: {', '.join(result['missing_required'])}")

if __name__ == "__main__":
    main()