#!/usr/bin/env python3
"""
MyCloset-AI 모델 아키텍처 및 체크포인트 구조 분석기
각 모델의 논문 구조와 실제 체크포인트 구조를 분석합니다.
"""

import os
import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
from collections import defaultdict

# PyTorch 경고 무시
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class ModelArchitectureAnalyzer:
    def __init__(self, ai_models_dir: str = "ai_models"):
        self.ai_models_dir = Path(ai_models_dir)
        self.analysis_results = {}
        
        # 모델별 논문 정보
        self.model_papers = {
            # Human Parsing Models
            "graphonomy": {
                "paper": "Graphonomy: Universal Human Parsing via Graph Transfer Learning",
                "architecture": "Graph Neural Network + CNN",
                "input_channels": 3,
                "output_classes": 20,
                "backbone": "ResNet-101"
            },
            "schp": {
                "paper": "Self-Correction for Human Parsing",
                "architecture": "Self-Correction Network",
                "input_channels": 3,
                "output_classes": 20,
                "backbone": "ResNet-101"
            },
            "deeplabv3plus": {
                "paper": "Encoder-Decoder with Atrous Separable Convolution",
                "architecture": "Encoder-Decoder with ASPP",
                "input_channels": 3,
                "output_classes": 21,
                "backbone": "ResNet-101"
            },
            
            # Pose Estimation Models
            "hrnet": {
                "paper": "Deep High-Resolution Representation Learning",
                "architecture": "High-Resolution Network",
                "input_channels": 3,
                "output_keypoints": 17,
                "backbone": "HRNet-W48"
            },
            "openpose": {
                "paper": "Realtime Multi-Person 2D Pose Estimation",
                "architecture": "Multi-Stage CNN",
                "input_channels": 3,
                "output_keypoints": 18,
                "backbone": "VGG-19"
            },
            "yolov8": {
                "paper": "YOLOv8: A State-of-the-Art Real-Time Object Detection",
                "architecture": "CSPDarknet + PANet",
                "input_channels": 3,
                "output_keypoints": 17,
                "backbone": "CSPDarknet"
            },
            
            # Segmentation Models
            "sam": {
                "paper": "Segment Anything Model",
                "architecture": "Vision Transformer + Prompt Encoder",
                "input_channels": 3,
                "output_embeddings": 256,
                "backbone": "ViT-H/14"
            },
            "u2net": {
                "paper": "U²-Net: Going Deeper with Nested U-Structure",
                "architecture": "Nested U-Structure",
                "input_channels": 3,
                "output_channels": 1,
                "backbone": "ResNet-34"
            },
            "mobile_sam": {
                "paper": "MobileSAM: Fast Segment Anything Model",
                "architecture": "Lightweight Vision Transformer",
                "input_channels": 3,
                "output_embeddings": 256,
                "backbone": "TinyViT"
            },
            
            # Geometric Matching Models
            "gmm": {
                "paper": "VITON: An Image-based Virtual Try-on Network",
                "architecture": "Geometric Matching Module",
                "input_channels": 6,
                "output_channels": 2,
                "backbone": "ResNet-101"
            },
            "tps": {
                "paper": "Thin-Plate Spline Transformation",
                "architecture": "TPS Transformation Network",
                "input_channels": 6,
                "output_channels": 2,
                "backbone": "ResNet-101"
            },
            "raft": {
                "paper": "RAFT: Recurrent All-Pairs Field Transforms",
                "architecture": "Recurrent All-Pairs Field Transforms",
                "input_channels": 6,
                "output_channels": 2,
                "backbone": "ResNet-50"
            },
            
            # Virtual Try-on Models
            "ootd": {
                "paper": "OOTDiffusion: Outfitting Fusion based Latent Diffusion",
                "architecture": "Latent Diffusion Model",
                "input_channels": 4,
                "output_channels": 3,
                "backbone": "UNet"
            },
            "viton": {
                "paper": "VITON: An Image-based Virtual Try-on Network",
                "architecture": "Two-Stage Pipeline",
                "input_channels": 6,
                "output_channels": 3,
                "backbone": "ResNet-101"
            },
            "hrviton": {
                "paper": "HR-VITON: High-Resolution Virtual Try-On",
                "architecture": "High-Resolution Pipeline",
                "input_channels": 6,
                "output_channels": 3,
                "backbone": "HRNet"
            },
            
            # Enhancement Models
            "realesrgan": {
                "paper": "Real-ESRGAN: Training Real-World Blind Super-Resolution",
                "architecture": "Enhanced SRGAN",
                "input_channels": 3,
                "output_channels": 3,
                "backbone": "RRDB"
            },
            "gfpgan": {
                "paper": "GFPGAN: Towards Real-World Blind Face Restoration",
                "architecture": "Generative Facial Prior GAN",
                "input_channels": 3,
                "output_channels": 3,
                "backbone": "StyleGAN2"
            },
            "swinir": {
                "paper": "SwinIR: Image Restoration Using Swin Transformer",
                "architecture": "Swin Transformer",
                "input_channels": 3,
                "output_channels": 3,
                "backbone": "Swin Transformer"
            },
            
            # Quality Assessment Models
            "clip": {
                "paper": "Learning Transferable Visual Representations",
                "architecture": "Vision-Language Model",
                "input_channels": 3,
                "output_embeddings": 512,
                "backbone": "ViT-B/32"
            },
            "lpips": {
                "paper": "The Unreasonable Effectiveness of Deep Features",
                "architecture": "Learned Perceptual Similarity",
                "input_channels": 3,
                "output_similarity": 1,
                "backbone": "AlexNet"
            }
        }
    
    def analyze_checkpoint_structure(self, checkpoint_path: Path) -> Dict[str, Any]:
        """체크포인트의 상세 구조를 분석합니다."""
        try:
            print(f"\n=== 체크포인트 구조 분석: {checkpoint_path.name} ===")
            
            # 파일 크기 확인
            file_size = checkpoint_path.stat().st_size / (1024 * 1024)  # MB
            
            # 체크포인트 로드
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            
            analysis = {
                "file_path": str(checkpoint_path),
                "file_size_mb": round(file_size, 2),
                "checkpoint_type": str(type(checkpoint)),
                "is_dict": isinstance(checkpoint, dict),
                "structure": {},
                "layer_analysis": {},
                "channel_analysis": {},
                "parameter_count": 0,
                "model_architecture": self._detect_model_architecture(checkpoint_path.name),
                "error": None
            }
            
            if isinstance(checkpoint, dict):
                analysis["structure"]["keys"] = list(checkpoint.keys())
                analysis["structure"]["key_count"] = len(checkpoint)
                
                # state_dict 분석
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    analysis["layer_analysis"] = self._analyze_layers(state_dict)
                    analysis["channel_analysis"] = self._analyze_channels(state_dict)
                    analysis["parameter_count"] = self._count_parameters(state_dict)
                    
                # 직접 state_dict인 경우
                elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    analysis["layer_analysis"] = self._analyze_layers(checkpoint)
                    analysis["channel_analysis"] = self._analyze_channels(checkpoint)
                    analysis["parameter_count"] = self._count_parameters(checkpoint)
                    
            return analysis
            
        except Exception as e:
            return {
                "file_path": str(checkpoint_path),
                "file_size_mb": round(file_size, 2),
                "error": str(e),
                "checkpoint_type": "Error",
                "is_dict": False,
                "structure": {},
                "layer_analysis": {},
                "channel_analysis": {},
                "parameter_count": 0,
                "model_architecture": "Unknown"
            }
    
    def _detect_model_architecture(self, filename: str) -> str:
        """파일명으로부터 모델 아키텍처를 추정합니다."""
        filename_lower = filename.lower()
        
        for model_name, info in self.model_papers.items():
            if model_name in filename_lower:
                return model_name
                
        # 특수한 경우들
        if "diffusion" in filename_lower:
            return "diffusion"
        elif "pytorch_model" in filename_lower:
            return "pytorch_converted"
        elif "safetensors" in filename_lower:
            return "safetensors"
            
        return "unknown"
    
    def _analyze_layers(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """레이어 구조를 분석합니다."""
        layer_analysis = {
            "total_layers": len(state_dict),
            "layer_types": defaultdict(int),
            "layer_depths": [],
            "activation_layers": [],
            "normalization_layers": [],
            "convolution_layers": [],
            "linear_layers": []
        }
        
        for key, tensor in state_dict.items():
            # 레이어 타입 분류
            if "conv" in key:
                layer_analysis["layer_types"]["convolution"] += 1
                layer_analysis["convolution_layers"].append({
                    "name": key,
                    "shape": list(tensor.shape),
                    "channels": tensor.shape[0] if len(tensor.shape) >= 4 else None
                })
            elif "linear" in key or "fc" in key:
                layer_analysis["layer_types"]["linear"] += 1
                layer_analysis["linear_layers"].append({
                    "name": key,
                    "shape": list(tensor.shape)
                })
            elif "bn" in key or "norm" in key:
                layer_analysis["layer_types"]["normalization"] += 1
                layer_analysis["normalization_layers"].append({
                    "name": key,
                    "shape": list(tensor.shape)
                })
            elif "relu" in key or "sigmoid" in key or "tanh" in key:
                layer_analysis["layer_types"]["activation"] += 1
                layer_analysis["activation_layers"].append({
                    "name": key,
                    "shape": list(tensor.shape)
                })
            
            # 레이어 깊이 추정
            if "." in key:
                depth = len(key.split("."))
                layer_analysis["layer_depths"].append(depth)
        
        return layer_analysis
    
    def _analyze_channels(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """채널 구조를 분석합니다."""
        channel_analysis = {
            "input_channels": None,
            "output_channels": None,
            "hidden_channels": [],
            "channel_progression": [],
            "max_channels": 0,
            "min_channels": float('inf')
        }
        
        conv_layers = []
        for key, tensor in state_dict.items():
            if "conv" in key and len(tensor.shape) >= 4:
                conv_layers.append({
                    "name": key,
                    "in_channels": tensor.shape[1],
                    "out_channels": tensor.shape[0],
                    "kernel_size": tensor.shape[2:4] if len(tensor.shape) >= 4 else None
                })
                
                channel_analysis["hidden_channels"].append(tensor.shape[0])
                channel_analysis["max_channels"] = max(channel_analysis["max_channels"], tensor.shape[0])
                channel_analysis["min_channels"] = min(channel_analysis["min_channels"], tensor.shape[0])
        
        if conv_layers:
            # 입력 채널 추정 (첫 번째 conv 레이어)
            first_conv = conv_layers[0]
            channel_analysis["input_channels"] = first_conv["in_channels"]
            
            # 출력 채널 추정 (마지막 conv 레이어)
            last_conv = conv_layers[-1]
            channel_analysis["output_channels"] = last_conv["out_channels"]
            
            # 채널 진행 추적
            channel_analysis["channel_progression"] = [layer["out_channels"] for layer in conv_layers]
        
        return channel_analysis
    
    def _count_parameters(self, state_dict: Dict[str, torch.Tensor]) -> int:
        """모델 파라미터 수를 계산합니다."""
        total_params = 0
        for tensor in state_dict.values():
            if hasattr(tensor, 'numel'):
                total_params += tensor.numel()
        return total_params
    
    def analyze_step_models(self, step_name: str) -> Dict[str, Any]:
        """특정 스텝의 모든 모델을 분석합니다."""
        step_dir = self.ai_models_dir / step_name
        if not step_dir.exists():
            return {"error": f"Step directory not found: {step_name}"}
            
        print(f"\n{'='*80}")
        print(f"스텝 분석: {step_name}")
        print(f"{'='*80}")
        
        # 체크포인트 파일 찾기
        checkpoint_extensions = ['.pth', '.pt', '.bin', '.safetensors']
        checkpoint_files = []
        for ext in checkpoint_extensions:
            checkpoint_files.extend(step_dir.glob(f"*{ext}"))
        
        if not checkpoint_files:
            return {"error": f"No checkpoint files found in {step_name}"}
            
        step_analysis = {
            "step_name": step_name,
            "checkpoint_count": len(checkpoint_files),
            "models": {},
            "total_parameters": 0,
            "total_size_mb": 0,
            "architecture_summary": defaultdict(int)
        }
        
        for checkpoint_file in checkpoint_files:
            analysis = self.analyze_checkpoint_structure(checkpoint_file)
            step_analysis["models"][checkpoint_file.name] = analysis
            
            if "error" not in analysis:
                step_analysis["total_parameters"] += analysis.get("parameter_count", 0)
                step_analysis["total_size_mb"] += analysis.get("file_size_mb", 0)
                arch = analysis.get("model_architecture", "unknown")
                step_analysis["architecture_summary"][arch] += 1
        
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
            "steps": {},
            "overall_summary": {
                "total_models": 0,
                "total_parameters": 0,
                "total_size_gb": 0,
                "architecture_distribution": defaultdict(int)
            }
        }
        
        for step in steps:
            step_analysis = self.analyze_step_models(step)
            all_analysis["steps"][step] = step_analysis
            
            if "error" not in step_analysis:
                all_analysis["overall_summary"]["total_models"] += step_analysis["checkpoint_count"]
                all_analysis["overall_summary"]["total_parameters"] += step_analysis["total_parameters"]
                all_analysis["overall_summary"]["total_size_gb"] += step_analysis["total_size_mb"] / 1024
                
                for arch, count in step_analysis["architecture_summary"].items():
                    all_analysis["overall_summary"]["architecture_distribution"][arch] += count
        
        return all_analysis
    
    def save_analysis(self, analysis: Dict[str, Any], output_file: str = "model_architecture_analysis.json"):
        """분석 결과를 JSON 파일로 저장합니다."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n분석 결과가 {output_file}에 저장되었습니다.")
    
    def print_summary(self, analysis: Dict[str, Any]):
        """분석 결과 요약을 출력합니다."""
        print(f"\n{'='*100}")
        print("모델 아키텍처 분석 요약")
        print(f"{'='*100}")
        
        summary = analysis["overall_summary"]
        print(f"\n전체 통계:")
        print(f"  총 모델 수: {summary['total_models']:,}")
        print(f"  총 파라미터 수: {summary['total_parameters']:,}")
        print(f"  총 크기: {summary['total_size_gb']:.2f} GB")
        
        print(f"\n아키텍처 분포:")
        for arch, count in summary["architecture_distribution"].items():
            print(f"  {arch}: {count}개")
        
        print(f"\n스텝별 상세:")
        for step_name, step_data in analysis["steps"].items():
            if "error" in step_data:
                print(f"\n{step_name}: {step_data['error']}")
                continue
                
            print(f"\n{step_name}:")
            print(f"  모델 수: {step_data['checkpoint_count']}")
            print(f"  파라미터 수: {step_data['total_parameters']:,}")
            print(f"  크기: {step_data['total_size_mb']:.2f} MB")
            print(f"  아키텍처: {dict(step_data['architecture_summary'])}")

def main():
    """메인 실행 함수"""
    print("MyCloset-AI 모델 아키텍처 분석 시작...")
    
    analyzer = ModelArchitectureAnalyzer()
    
    # 전체 분석 실행
    analysis = analyzer.analyze_all_steps()
    
    # 결과 저장
    analyzer.save_analysis(analysis)
    
    # 요약 출력
    analyzer.print_summary(analysis)
    
    print("\n분석 완료!")

if __name__ == "__main__":
    main()
