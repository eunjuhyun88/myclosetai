#!/usr/bin/env python3
"""
MyCloset-AI 전체 체크포인트 종합 분석기
모든 체크포인트의 구조, 채널 수, 레이어 수, 신경망 구조를 완전히 분석합니다.
"""

import os
import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
from collections import defaultdict
import time

# PyTorch 경고 무시
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class ComprehensiveCheckpointAnalyzer:
    def __init__(self, ai_models_dir: str = "ai_models"):
        self.ai_models_dir = Path(ai_models_dir)
        self.analysis_results = {}
        
        # 모델별 논문 정보 및 아키텍처
        self.model_architectures = {
            # Human Parsing Models
            "graphonomy": {
                "paper": "Graphonomy: Universal Human Parsing via Graph Transfer Learning",
                "architecture": "Graph Neural Network + CNN",
                "input_channels": 3,
                "output_classes": 20,
                "backbone": "ResNet-101",
                "expected_layers": ["backbone", "graphonomy", "classifier"]
            },
            "schp": {
                "paper": "Self-Correction for Human Parsing",
                "architecture": "Self-Correction Network",
                "input_channels": 3,
                "output_classes": 20,
                "backbone": "ResNet-101",
                "expected_layers": ["backbone", "schp", "classifier"]
            },
            "deeplabv3plus": {
                "paper": "Encoder-Decoder with Atrous Separable Convolution",
                "architecture": "Encoder-Decoder with ASPP",
                "input_channels": 3,
                "output_classes": 21,
                "backbone": "ResNet-101",
                "expected_layers": ["backbone", "aspp", "decoder"]
            },
            
            # Pose Estimation Models
            "hrnet": {
                "paper": "Deep High-Resolution Representation Learning",
                "architecture": "High-Resolution Network",
                "input_channels": 3,
                "output_keypoints": 17,
                "backbone": "HRNet-W48",
                "expected_layers": ["backbone", "hrnet", "keypoint_head"]
            },
            "openpose": {
                "paper": "Realtime Multi-Person 2D Pose Estimation",
                "architecture": "Multi-Stage CNN",
                "input_channels": 3,
                "output_keypoints": 18,
                "backbone": "VGG-19",
                "expected_layers": ["vgg", "stages", "paf", "heatmap"]
            },
            "yolov8": {
                "paper": "YOLOv8: A State-of-the-Art Real-Time Object Detection",
                "architecture": "CSPDarknet + PANet",
                "input_channels": 3,
                "output_keypoints": 17,
                "backbone": "CSPDarknet",
                "expected_layers": ["backbone", "neck", "head"]
            },
            
            # Segmentation Models
            "sam": {
                "paper": "Segment Anything Model",
                "architecture": "Vision Transformer + Prompt Encoder",
                "input_channels": 3,
                "output_embeddings": 256,
                "backbone": "ViT-H/14",
                "expected_layers": ["image_encoder", "prompt_encoder", "mask_decoder"]
            },
            "u2net": {
                "paper": "U²-Net: Going Deeper with Nested U-Structure",
                "architecture": "Nested U-Structure",
                "input_channels": 3,
                "output_channels": 1,
                "backbone": "ResNet-34",
                "expected_layers": ["features", "classifier"]
            },
            "mobile_sam": {
                "paper": "MobileSAM: Fast Segment Anything Model",
                "architecture": "Lightweight Vision Transformer",
                "input_channels": 3,
                "output_embeddings": 256,
                "backbone": "TinyViT",
                "expected_layers": ["image_encoder", "prompt_encoder", "mask_decoder"]
            },
            
            # Geometric Matching Models
            "gmm": {
                "paper": "VITON: An Image-based Virtual Try-on Network",
                "architecture": "Geometric Matching Module",
                "input_channels": 6,
                "output_channels": 2,
                "backbone": "ResNet-101",
                "expected_layers": ["gmm_backbone", "gmm_head"]
            },
            "tps": {
                "paper": "Thin-Plate Spline Transformation",
                "architecture": "TPS Transformation Network",
                "input_channels": 6,
                "output_channels": 2,
                "backbone": "ResNet-101",
                "expected_layers": ["backbone", "tps_head"]
            },
            "raft": {
                "paper": "RAFT: Recurrent All-Pairs Field Transforms",
                "architecture": "Recurrent All-Pairs Field Transforms",
                "input_channels": 6,
                "output_channels": 2,
                "backbone": "ResNet-50",
                "expected_layers": ["feature_encoder", "context_encoder", "flow_head"]
            },
            
            # Virtual Try-on Models
            "ootd": {
                "paper": "OOTDiffusion: Outfitting Fusion based Latent Diffusion",
                "architecture": "Latent Diffusion Model",
                "input_channels": 4,
                "output_channels": 3,
                "backbone": "UNet",
                "expected_layers": ["unet", "text_encoder", "vae"]
            },
            "viton": {
                "paper": "VITON: An Image-based Virtual Try-on Network",
                "architecture": "Two-Stage Pipeline",
                "input_channels": 6,
                "output_channels": 3,
                "backbone": "ResNet-101",
                "expected_layers": ["backbone", "tryon_head"]
            },
            "hrviton": {
                "paper": "HR-VITON: High-Resolution Virtual Try-On",
                "architecture": "High-Resolution Pipeline",
                "input_channels": 6,
                "output_channels": 3,
                "backbone": "HRNet",
                "expected_layers": ["backbone", "tryon_head"]
            },
            
            # Enhancement Models
            "realesrgan": {
                "paper": "Real-ESRGAN: Training Real-World Blind Super-Resolution",
                "architecture": "Enhanced SRGAN",
                "input_channels": 3,
                "output_channels": 3,
                "backbone": "RRDB",
                "expected_layers": ["rrdb", "upsampling", "conv"]
            },
            "gfpgan": {
                "paper": "GFPGAN: Towards Real-World Blind Face Restoration",
                "architecture": "Generative Facial Prior GAN",
                "input_channels": 3,
                "output_channels": 3,
                "backbone": "StyleGAN2",
                "expected_layers": ["generator", "discriminator"]
            },
            "swinir": {
                "paper": "SwinIR: Image Restoration Using Swin Transformer",
                "architecture": "Swin Transformer",
                "input_channels": 3,
                "output_channels": 3,
                "backbone": "Swin Transformer",
                "expected_layers": ["patch_embed", "swin_blocks", "upsampling"]
            },
            
            # Quality Assessment Models
            "clip": {
                "paper": "Learning Transferable Visual Representations",
                "architecture": "Vision-Language Model",
                "input_channels": 3,
                "output_embeddings": 512,
                "backbone": "ViT-B/32",
                "expected_layers": ["visual", "text_encoder"]
            },
            "lpips": {
                "paper": "The Unreasonable Effectiveness of Deep Features",
                "architecture": "Learned Perceptual Similarity",
                "input_channels": 3,
                "output_similarity": 1,
                "backbone": "AlexNet",
                "expected_layers": ["features", "classifier"]
            }
        }
    
    def analyze_checkpoint_comprehensive(self, checkpoint_path: Path) -> Dict[str, Any]:
        """체크포인트를 종합적으로 분석합니다."""
        file_size = 0
        try:
            print(f"\n=== 종합 분석: {checkpoint_path.name} ===")
            
            # 파일 존재 확인
            if not checkpoint_path.exists():
                return {
                    "file_path": str(checkpoint_path),
                    "file_size_mb": 0,
                    "error": "File not found",
                    "checkpoint_type": "Error",
                    "is_dict": False,
                    "structure_analysis": {},
                    "layer_analysis": {},
                    "channel_analysis": {},
                    "parameter_analysis": {},
                    "architecture_detection": {},
                    "model_compatibility": {}
                }
            
            # 파일 크기 확인
            file_size = checkpoint_path.stat().st_size / (1024 * 1024)  # MB
            
            # 체크포인트 로드
            start_time = time.time()
            checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            load_time = time.time() - start_time
            
            analysis = {
                "file_path": str(checkpoint_path),
                "file_size_mb": round(file_size, 2),
                "load_time_seconds": round(load_time, 3),
                "checkpoint_type": str(type(checkpoint)),
                "is_dict": isinstance(checkpoint, dict),
                "structure_analysis": {},
                "layer_analysis": {},
                "channel_analysis": {},
                "parameter_analysis": {},
                "architecture_detection": {},
                "model_compatibility": {},
                "error": None
            }
            
            if isinstance(checkpoint, dict):
                analysis["structure_analysis"] = self._analyze_structure(checkpoint)
                
                # state_dict 분석
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    analysis["layer_analysis"] = self._analyze_layers_detailed(state_dict)
                    analysis["channel_analysis"] = self._analyze_channels_detailed(state_dict)
                    analysis["parameter_analysis"] = self._analyze_parameters(state_dict)
                    analysis["architecture_detection"] = self._detect_architecture(state_dict, checkpoint_path.name)
                    analysis["model_compatibility"] = self._check_model_compatibility(state_dict, checkpoint_path.name)
                    
                # 직접 state_dict인 경우
                elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    analysis["layer_analysis"] = self._analyze_layers_detailed(checkpoint)
                    analysis["channel_analysis"] = self._analyze_channels_detailed(checkpoint)
                    analysis["parameter_analysis"] = self._analyze_parameters(checkpoint)
                    analysis["architecture_detection"] = self._detect_architecture(checkpoint, checkpoint_path.name)
                    analysis["model_compatibility"] = self._check_model_compatibility(checkpoint, checkpoint_path.name)
                    
            return analysis
            
        except Exception as e:
            return {
                "file_path": str(checkpoint_path),
                "file_size_mb": round(file_size, 2),
                "error": str(e),
                "checkpoint_type": "Error",
                "is_dict": False,
                "structure_analysis": {},
                "layer_analysis": {},
                "channel_analysis": {},
                "parameter_analysis": {},
                "architecture_detection": {},
                "model_compatibility": {}
            }
    
    def _analyze_structure(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """체크포인트 구조를 분석합니다."""
        return {
            "keys": list(checkpoint.keys()),
            "key_count": len(checkpoint),
            "has_state_dict": 'state_dict' in checkpoint,
            "has_meta": 'meta' in checkpoint,
            "has_config": 'config' in checkpoint,
            "has_optimizer": 'optimizer' in checkpoint,
            "has_scheduler": 'scheduler' in checkpoint
        }
    
    def _analyze_layers_detailed(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """레이어를 상세히 분석합니다."""
        layer_analysis = {
            "total_layers": len(state_dict),
            "layer_types": defaultdict(int),
            "layer_depths": [],
            "layer_groups": defaultdict(list),
            "convolution_layers": [],
            "linear_layers": [],
            "normalization_layers": [],
            "activation_layers": [],
            "embedding_layers": [],
            "attention_layers": []
        }
        
        for key, tensor in state_dict.items():
            # 레이어 타입 분류
            if "conv" in key:
                layer_analysis["layer_types"]["convolution"] += 1
                layer_analysis["convolution_layers"].append({
                    "name": key,
                    "shape": list(tensor.shape),
                    "in_channels": tensor.shape[1] if len(tensor.shape) >= 4 else None,
                    "out_channels": tensor.shape[0] if len(tensor.shape) >= 4 else None,
                    "kernel_size": tensor.shape[2:4] if len(tensor.shape) >= 4 else None
                })
            elif "linear" in key or "fc" in key:
                layer_analysis["layer_types"]["linear"] += 1
                layer_analysis["linear_layers"].append({
                    "name": key,
                    "shape": list(tensor.shape),
                    "in_features": tensor.shape[1] if len(tensor.shape) >= 2 else None,
                    "out_features": tensor.shape[0] if len(tensor.shape) >= 2 else None
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
            elif "embed" in key:
                layer_analysis["layer_types"]["embedding"] += 1
                layer_analysis["embedding_layers"].append({
                    "name": key,
                    "shape": list(tensor.shape)
                })
            elif "attn" in key or "attention" in key:
                layer_analysis["layer_types"]["attention"] += 1
                layer_analysis["attention_layers"].append({
                    "name": key,
                    "shape": list(tensor.shape)
                })
            
            # 레이어 그룹 분류
            if "." in key:
                group = key.split(".")[0]
                layer_analysis["layer_groups"][group].append(key)
                depth = len(key.split("."))
                layer_analysis["layer_depths"].append(depth)
        
        return layer_analysis
    
    def _analyze_channels_detailed(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """채널 구조를 상세히 분석합니다."""
        channel_analysis = {
            "input_channels": None,
            "output_channels": None,
            "hidden_channels": [],
            "channel_progression": [],
            "max_channels": 0,
            "min_channels": float('inf'),
            "channel_distribution": defaultdict(int),
            "conv_channel_analysis": []
        }
        
        conv_layers = []
        for key, tensor in state_dict.items():
            if "conv" in key and len(tensor.shape) >= 4:
                conv_info = {
                    "name": key,
                    "in_channels": tensor.shape[1],
                    "out_channels": tensor.shape[0],
                    "kernel_size": tensor.shape[2:4] if len(tensor.shape) >= 4 else None
                }
                conv_layers.append(conv_info)
                channel_analysis["conv_channel_analysis"].append(conv_info)
                
                channel_analysis["hidden_channels"].append(tensor.shape[0])
                channel_analysis["max_channels"] = max(channel_analysis["max_channels"], tensor.shape[0])
                channel_analysis["min_channels"] = min(channel_analysis["min_channels"], tensor.shape[0])
                channel_analysis["channel_distribution"][tensor.shape[0]] += 1
        
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
    
    def _analyze_parameters(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """파라미터를 분석합니다."""
        total_params = 0
        trainable_params = 0
        param_distribution = defaultdict(int)
        
        for key, tensor in state_dict.items():
            if hasattr(tensor, 'numel'):
                num_params = tensor.numel()
                total_params += num_params
                
                # 학습 가능한 파라미터 (bias, weight는 학습 가능, running_mean/var는 아님)
                if not any(x in key for x in ['running_mean', 'running_var', 'num_batches_tracked']):
                    trainable_params += num_params
                
                # 파라미터 크기 분포
                if num_params < 1000:
                    param_distribution["small"] += 1
                elif num_params < 100000:
                    param_distribution["medium"] += 1
                elif num_params < 1000000:
                    param_distribution["large"] += 1
                else:
                    param_distribution["huge"] += 1
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "parameter_distribution": dict(param_distribution),
            "average_parameters_per_layer": total_params / len(state_dict) if state_dict else 0
        }
    
    def _detect_architecture(self, state_dict: Dict[str, torch.Tensor], filename: str) -> Dict[str, Any]:
        """아키텍처를 감지합니다."""
        filename_lower = filename.lower()
        
        # 파일명 기반 감지
        detected_arch = "unknown"
        for model_name in self.model_architectures.keys():
            if model_name in filename_lower:
                detected_arch = model_name
                break
        
        # 레이어 구조 기반 감지
        layer_based_detection = self._detect_by_layer_structure(state_dict)
        
        return {
            "filename_based": detected_arch,
            "layer_based": layer_based_detection,
            "confidence": self._calculate_detection_confidence(state_dict, detected_arch),
            "expected_architecture": self.model_architectures.get(detected_arch, {})
        }
    
    def _detect_by_layer_structure(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """레이어 구조로 아키텍처를 감지합니다."""
        keys = list(state_dict.keys())
        
        # Vision Transformer 감지
        if any("attn" in key for key in keys) and any("patch_embed" in key for key in keys):
            return "vision_transformer"
        
        # ResNet 감지
        if any("layer" in key and "conv" in key for key in keys):
            return "resnet"
        
        # HRNet 감지
        if any("backbone" in key for key in keys) and any("hrnet" in key for key in keys):
            return "hrnet"
        
        # U-Net 감지
        if any("down" in key for key in keys) and any("up" in key for key in keys):
            return "unet"
        
        # CNN 감지
        if any("conv" in key for key in keys):
            return "cnn"
        
        return "unknown"
    
    def _calculate_detection_confidence(self, state_dict: Dict[str, torch.Tensor], detected_arch: str) -> float:
        """감지 신뢰도를 계산합니다."""
        if detected_arch == "unknown":
            return 0.0
        
        expected_arch = self.model_architectures.get(detected_arch, {})
        expected_layers = expected_arch.get("expected_layers", [])
        
        if not expected_layers:
            return 0.5
        
        keys = list(state_dict.keys())
        matched_layers = sum(1 for layer in expected_layers if any(layer in key for key in keys))
        
        return matched_layers / len(expected_layers)
    
    def _check_model_compatibility(self, state_dict: Dict[str, torch.Tensor], filename: str) -> Dict[str, Any]:
        """모델 호환성을 확인합니다."""
        filename_lower = filename.lower()
        
        # 파일명 기반 모델 타입 감지
        model_type = "unknown"
        for model_name in self.model_architectures.keys():
            if model_name in filename_lower:
                model_type = model_name
                break
        
        if model_type == "unknown":
            return {"compatible": False, "reason": "Unknown model type"}
        
        expected_arch = self.model_architectures.get(model_type, {})
        expected_input_channels = expected_arch.get("input_channels", None)
        
        # 입력 채널 확인
        actual_input_channels = None
        for key, tensor in state_dict.items():
            if "conv" in key and len(tensor.shape) >= 4:
                actual_input_channels = tensor.shape[1]
                break
        
        input_compatible = True
        if expected_input_channels and actual_input_channels:
            input_compatible = expected_input_channels == actual_input_channels
        
        return {
            "compatible": input_compatible,
            "model_type": model_type,
            "expected_input_channels": expected_input_channels,
            "actual_input_channels": actual_input_channels,
            "input_compatible": input_compatible
        }
    
    def find_all_checkpoints(self) -> List[Path]:
        """모든 체크포인트 파일을 찾습니다."""
        checkpoint_extensions = ['.pth', '.pt', '.bin', '.safetensors']
        checkpoint_files = []
        
        for ext in checkpoint_extensions:
            checkpoint_files.extend(self.ai_models_dir.rglob(f"*{ext}"))
            
        return checkpoint_files
    
    def analyze_all_checkpoints(self) -> Dict[str, Any]:
        """모든 체크포인트를 분석합니다."""
        print("모든 체크포인트 파일을 찾는 중...")
        checkpoint_files = self.find_all_checkpoints()
        print(f"총 {len(checkpoint_files)}개의 체크포인트 파일을 찾았습니다.")
        
        all_analysis = {
            "total_checkpoints": len(checkpoint_files),
            "checkpoints": {},
            "summary": {
                "total_parameters": 0,
                "total_size_gb": 0,
                "architecture_distribution": defaultdict(int),
                "model_type_distribution": defaultdict(int),
                "compatible_models": 0,
                "incompatible_models": 0
            }
        }
        
        for i, checkpoint_file in enumerate(checkpoint_files):
            print(f"\n진행률: {i+1}/{len(checkpoint_files)} ({((i+1)/len(checkpoint_files)*100):.1f}%)")
            
            analysis = self.analyze_checkpoint_comprehensive(checkpoint_file)
            all_analysis["checkpoints"][str(checkpoint_file)] = analysis
            
            # 요약 통계 업데이트
            if "error" not in analysis:
                all_analysis["summary"]["total_parameters"] += analysis.get("parameter_analysis", {}).get("total_parameters", 0)
                all_analysis["summary"]["total_size_gb"] += analysis.get("file_size_mb", 0) / 1024
                
                arch_detection = analysis.get("architecture_detection", {})
                filename_arch = arch_detection.get("filename_based", "unknown")
                all_analysis["summary"]["architecture_distribution"][filename_arch] += 1
                
                compatibility = analysis.get("model_compatibility", {})
                if compatibility.get("compatible", False):
                    all_analysis["summary"]["compatible_models"] += 1
                else:
                    all_analysis["summary"]["incompatible_models"] += 1
                
                model_type = compatibility.get("model_type", "unknown")
                all_analysis["summary"]["model_type_distribution"][model_type] += 1
        
        return all_analysis
    
    def save_analysis(self, analysis: Dict[str, Any], output_file: str = "comprehensive_checkpoint_analysis.json"):
        """분석 결과를 JSON 파일로 저장합니다."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n분석 결과가 {output_file}에 저장되었습니다.")
    
    def print_summary(self, analysis: Dict[str, Any]):
        """분석 결과 요약을 출력합니다."""
        print(f"\n{'='*120}")
        print("체크포인트 종합 분석 요약")
        print(f"{'='*120}")
        
        summary = analysis["summary"]
        print(f"\n전체 통계:")
        print(f"  총 체크포인트 수: {analysis['total_checkpoints']:,}")
        print(f"  총 파라미터 수: {summary['total_parameters']:,}")
        print(f"  총 크기: {summary['total_size_gb']:.2f} GB")
        print(f"  호환 가능한 모델: {summary['compatible_models']}")
        print(f"  호환 불가능한 모델: {summary['incompatible_models']}")
        
        print(f"\n아키텍처 분포:")
        for arch, count in summary["architecture_distribution"].items():
            print(f"  {arch}: {count}개")
        
        print(f"\n모델 타입 분포:")
        for model_type, count in summary["model_type_distribution"].items():
            print(f"  {model_type}: {count}개")

def main():
    """메인 실행 함수"""
    print("MyCloset-AI 체크포인트 종합 분석 시작...")
    
    analyzer = ComprehensiveCheckpointAnalyzer()
    
    # 전체 분석 실행
    analysis = analyzer.analyze_all_checkpoints()
    
    # 결과 저장
    analyzer.save_analysis(analysis)
    
    # 요약 출력
    analyzer.print_summary(analysis)
    
    print("\n분석 완료!")

if __name__ == "__main__":
    main()
