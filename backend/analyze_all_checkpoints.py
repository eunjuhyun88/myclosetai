#!/usr/bin/env python3
"""
MyCloset-AI 전체 체크포인트 분석 스크립트
모든 스텝의 AI 모델 체크포인트를 분석하고 구조를 파악합니다.
"""

import os
import torch
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings

# PyTorch 경고 무시
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class CheckpointAnalyzer:
    def __init__(self, ai_models_dir: str = "ai_models"):
        self.ai_models_dir = Path(ai_models_dir)
        self.analysis_results = {}
        
    def analyze_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """개별 체크포인트 파일을 분석합니다."""
        try:
            print(f"\n=== 분석 중: {checkpoint_path.name} ===")
            
            # 파일 크기 확인
            file_size = checkpoint_path.stat().st_size / (1024 * 1024)  # MB
            
            # 체크포인트 로드
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            
            analysis = {
                "file_path": str(checkpoint_path),
                "file_size_mb": round(file_size, 2),
                "checkpoint_type": str(type(checkpoint)),
                "is_dict": isinstance(checkpoint, dict),
                "keys": [],
                "key_count": 0,
                "state_dict_keys": [],
                "state_dict_key_count": 0,
                "sample_shapes": {},
                "error": None
            }
            
            if isinstance(checkpoint, dict):
                analysis["keys"] = list(checkpoint.keys())
                analysis["key_count"] = len(checkpoint)
                
                # state_dict가 있는지 확인
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    analysis["state_dict_keys"] = list(state_dict.keys())[:20]  # 처음 20개만
                    analysis["state_dict_key_count"] = len(state_dict)
                    
                    # 샘플 텐서 차원 확인
                    for i, (key, value) in enumerate(state_dict.items()):
                        if hasattr(value, 'shape'):
                            analysis["sample_shapes"][key] = list(value.shape)
                        if i >= 9:  # 처음 10개만
                            break
                            
            return analysis
            
        except Exception as e:
            return {
                "file_path": str(checkpoint_path),
                "file_size_mb": round(file_size, 2),
                "error": str(e),
                "checkpoint_type": "Error",
                "is_dict": False,
                "keys": [],
                "key_count": 0,
                "state_dict_keys": [],
                "state_dict_key_count": 0,
                "sample_shapes": {}
            }
    
    def find_checkpoint_files(self, step_dir: Path) -> List[Path]:
        """스텝 디렉토리에서 체크포인트 파일들을 찾습니다."""
        checkpoint_extensions = ['.pth', '.pt', '.bin', '.safetensors']
        checkpoint_files = []
        
        for ext in checkpoint_extensions:
            checkpoint_files.extend(step_dir.glob(f"*{ext}"))
            
        return checkpoint_files
    
    def analyze_step(self, step_name: str) -> Dict[str, Any]:
        """특정 스텝의 모든 체크포인트를 분석합니다."""
        step_dir = self.ai_models_dir / step_name
        if not step_dir.exists():
            return {"error": f"Step directory not found: {step_name}"}
            
        print(f"\n{'='*60}")
        print(f"스텝 분석: {step_name}")
        print(f"{'='*60}")
        
        checkpoint_files = self.find_checkpoint_files(step_dir)
        
        if not checkpoint_files:
            return {"error": f"No checkpoint files found in {step_name}"}
            
        step_analysis = {
            "step_name": step_name,
            "checkpoint_count": len(checkpoint_files),
            "checkpoints": {}
        }
        
        for checkpoint_file in checkpoint_files:
            analysis = self.analyze_checkpoint(checkpoint_file)
            step_analysis["checkpoints"][checkpoint_file.name] = analysis
            
        return step_analysis
    
    def analyze_all_steps(self) -> Dict[str, Any]:
        """모든 스텝을 분석합니다."""
        steps = [
            "step_01_human_parsing",
            "step_02_pose_estimation", 
            "step_03_cloth_segmentation",
            "step_04_geometric_matching",
            "step_05_cloth_warping",
            "step_06_virtual_fitting",
            "step_07_post_processing",
            "step_08_quality_assessment"
        ]
        
        all_analysis = {
            "total_steps": len(steps),
            "steps": {}
        }
        
        for step in steps:
            step_analysis = self.analyze_step(step)
            all_analysis["steps"][step] = step_analysis
            
        return all_analysis
    
    def save_analysis(self, analysis: Dict[str, Any], output_file: str = "checkpoint_analysis_report.json"):
        """분석 결과를 JSON 파일로 저장합니다."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n분석 결과가 {output_file}에 저장되었습니다.")
    
    def print_summary(self, analysis: Dict[str, Any]):
        """분석 결과 요약을 출력합니다."""
        print(f"\n{'='*80}")
        print("체크포인트 분석 요약")
        print(f"{'='*80}")
        
        total_checkpoints = 0
        total_size_mb = 0
        
        for step_name, step_data in analysis["steps"].items():
            if "error" in step_data:
                print(f"\n{step_name}: {step_data['error']}")
                continue
                
            print(f"\n{step_name}:")
            print(f"  체크포인트 개수: {step_data['checkpoint_count']}")
            
            step_size = 0
            for checkpoint_name, checkpoint_data in step_data["checkpoints"].items():
                if "error" not in checkpoint_data:
                    step_size += checkpoint_data.get("file_size_mb", 0)
                    total_checkpoints += 1
                    
            print(f"  총 크기: {step_size:.2f} MB")
            total_size_mb += step_size
            
        print(f"\n전체 통계:")
        print(f"  총 체크포인트 개수: {total_checkpoints}")
        print(f"  총 크기: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")

def main():
    """메인 실행 함수"""
    print("MyCloset-AI 체크포인트 분석 시작...")
    
    analyzer = CheckpointAnalyzer()
    
    # 전체 분석 실행
    analysis = analyzer.analyze_all_steps()
    
    # 결과 저장
    analyzer.save_analysis(analysis)
    
    # 요약 출력
    analyzer.print_summary(analysis)
    
    print("\n분석 완료!")

if __name__ == "__main__":
    main()
