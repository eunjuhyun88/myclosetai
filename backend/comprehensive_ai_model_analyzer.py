#!/usr/bin/env python3
"""
🔥 종합 AI 모델 분석 및 업데이트 도구
====================================

모든 AI 모델 파일들을 분석하고 호환성 문제를 수정하는 종합 도구

Author: MyCloset AI Team
Date: 2025-08-08
Version: 1.0
"""

import os
import sys
import time
import logging
import traceback
import json
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import subprocess

# PyTorch 관련
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# NumPy 관련
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# SafeTensors 관련
try:
    import safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class AIModelInfo:
    """AI 모델 정보"""
    path: str
    size_mb: float = 0.0
    exists: bool = False
    valid: bool = False
    structure_type: str = "unknown"
    architecture_hints: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    fixed: bool = False
    step_category: str = "unknown"
    model_type: str = "unknown"

class ComprehensiveAIModelAnalyzer:
    """종합 AI 모델 분석 및 수정 도구"""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.models = {}
        self.analysis_results = {}
        
        # Step별 모델 카테고리 정의
        self.step_categories = {
            'step_01': {
                'name': 'Human Parsing',
                'models': ['graphonomy', 'u2net', 'deeplabv3plus', 'hrnet'],
                'paths': [
                    'ai_models/Self-Correction-Human-Parsing/exp-schp-201908261155-atr.pth',
                    'ai_models/step_01_human_parsing/deeplabv3plus.pth',
                    'ai_models/step_03_cloth_segmentation/u2net.pth'
                ]
            },
            'step_02': {
                'name': 'Pose Estimation',
                'models': ['hrnet', 'openpose', 'yolo', 'mediapipe'],
                'paths': [
                    'ai_models/step_02_pose_estimation/hrnet_w48_coco_384x288.pth',
                    'ai_models/step_02_pose_estimation/body_pose_model.pth',
                    'ai_models/step_02_pose_estimation/yolov8n-pose.pt',
                    'ai_models/openpose.pth'
                ]
            },
            'step_03': {
                'name': 'Cloth Segmentation',
                'models': ['sam', 'u2net', 'deeplabv3', 'mobile_sam'],
                'paths': [
                    'ai_models/step_03_cloth_segmentation/sam_vit_h_4b8939.pth',
                    'ai_models/step_03_cloth_segmentation/u2net.pth',
                    'ai_models/step_03_cloth_segmentation/deeplabv3_resnet101_ultra.pth',
                    'ai_models/step_03_cloth_segmentation/mobile_sam.pt'
                ]
            },
            'step_04': {
                'name': 'Geometric Matching',
                'models': ['gmm', 'tps', 'raft', 'optical_flow'],
                'paths': [
                    'ai_models/step_04_geometric_matching/gmm_final.pth',
                    'ai_models/step_04_geometric_matching/tps_network.pth',
                    'ai_models/step_04_geometric_matching/raft-things.pth',
                    'ai_models/step_04_geometric_matching/sam_vit_h_4b8939.pth'
                ]
            },
            'step_05': {
                'name': 'Cloth Warping',
                'models': ['tom', 'viton_hd', 'tps', 'dpt', 'vgg19'],
                'paths': [
                    'ai_models/step_05_cloth_warping/tom_final.pth',
                    'ai_models/step_05_cloth_warping/viton_hd_warping.pth',
                    'ai_models/step_05_cloth_warping/tps_transformation.pth',
                    'ai_models/step_05_cloth_warping/dpt_hybrid_midas.pth',
                    'ai_models/step_05_cloth_warping/vgg19_warping.pth'
                ]
            },
            'step_06': {
                'name': 'Virtual Fitting',
                'models': ['stable_diffusion', 'ootd', 'viton_hd', 'hrviton'],
                'paths': [
                    'ai_models/step_06_virtual_fitting/stable_diffusion_4.8gb.pth',
                    'ai_models/step_06_virtual_fitting/ootd_3.2gb.pth',
                    'ai_models/step_06_virtual_fitting/viton_hd_2.1gb.pth',
                    'ai_models/step_06_virtual_fitting/hrviton_final.pth',
                    'ai_models/step_06_virtual_fitting/validated_checkpoints/ootd_checkpoint.pth'
                ]
            },
            'step_07': {
                'name': 'Post Processing',
                'models': ['real_esrgan', 'swinir', 'gfpgan', 'densenet'],
                'paths': [
                    'ai_models/step_07_post_processing/RealESRGAN_x4plus.pth',
                    'ai_models/step_07_post_processing/swinir_real_sr_x4_large.pth',
                    'ai_models/step_07_post_processing/GFPGAN.pth',
                    'ai_models/step_07_post_processing/densenet161_enhance.pth'
                ]
            },
            'step_08': {
                'name': 'Quality Assessment',
                'models': ['clip', 'lpips', 'alex', 'vit'],
                'paths': [
                    'ai_models/step_08_quality_assessment/clip_vit_b32.pth',
                    'ai_models/step_08_quality_assessment/ViT-L-14.pt',
                    'ai_models/step_08_quality_assessment/lpips_alex.pth',
                    'ai_models/step_08_quality_assessment/alex.pth'
                ]
            }
        }
        
    def find_all_ai_models(self) -> List[str]:
        """모든 AI 모델 파일 찾기"""
        model_files = []
        
        # 다양한 확장자 검색
        extensions = ["*.pth", "*.pt", "*.safetensors", "*.ckpt", "*.bin"]
        
        for ext in extensions:
            files = list(self.root_dir.rglob(ext))
            model_files.extend([str(f) for f in files])
        
        # 중복 제거 및 정렬
        model_files = sorted(list(set(model_files)))
        
        # 불필요한 파일 필터링
        filtered_files = []
        exclude_patterns = [
            'distutils-precedence.pth',
            '__pycache__',
            '.git',
            'node_modules',
            'venv',
            'env',
            '.conda'
        ]
        
        for file_path in model_files:
            if not any(pattern in file_path for pattern in exclude_patterns):
                filtered_files.append(file_path)
        
        logger.info(f"🔍 발견된 AI 모델 파일: {len(filtered_files)}개")
        return filtered_files
    
    def categorize_model(self, model_path: str) -> Tuple[str, str]:
        """모델을 Step별로 분류"""
        model_path_lower = model_path.lower()
        
        for step_key, step_info in self.step_categories.items():
            for model_name in step_info['models']:
                if model_name.lower() in model_path_lower:
                    return step_key, model_name
        
        # 경로 기반 분류
        for step_key, step_info in self.step_categories.items():
            for expected_path in step_info['paths']:
                if expected_path.lower() in model_path_lower:
                    return step_key, "unknown"
        
        return "unknown", "unknown"
    
    def analyze_model(self, model_path: str) -> AIModelInfo:
        """개별 AI 모델 분석"""
        info = AIModelInfo(path=model_path)
        
        try:
            if not Path(model_path).exists():
                info.issues.append("파일이 존재하지 않음")
                return info
            
            info.exists = True
            info.size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            
            # Step 분류
            step_category, model_type = self.categorize_model(model_path)
            info.step_category = step_category
            info.model_type = model_type
            
            # 🔥 다양한 로딩 방법 시도
            model_data = None
            loading_method = None
            
            # 방법 1: weights_only=True (안전한 방법)
            try:
                model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                loading_method = 'weights_only_true'
                info.recommendations.append("안전한 weights_only=True로 로딩됨")
            except Exception as e1:
                # 방법 2: weights_only=False (전통적인 방법)
                try:
                    model_data = torch.load(model_path, map_location='cpu', weights_only=False)
                    loading_method = 'weights_only_false'
                    info.recommendations.append("weights_only=False로 로딩됨 (보안 주의)")
                except Exception as e2:
                    # 방법 3: TorchScript 모델
                    try:
                        model_data = torch.jit.load(model_path, map_location='cpu')
                        loading_method = 'torchscript'
                        info.recommendations.append("TorchScript 모델로 로딩됨")
                    except Exception as e3:
                        # 방법 4: SafeTensors
                        if SAFETENSORS_AVAILABLE:
                            try:
                                with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
                                    keys = list(f.keys())
                                    model_data = {key: f.get_tensor(key) for key in keys}
                                loading_method = 'safetensors'
                                info.recommendations.append("SafeTensors로 로딩됨")
                            except Exception as e4:
                                info.issues.append(f"모든 로딩 방법 실패: {e4}")
                                return info
                        else:
                            info.issues.append(f"모든 로딩 방법 실패: {e3}")
                            return info
            
            # 🔥 구조 타입 분류 및 검증
            if isinstance(model_data, dict):
                info.structure_type = 'dict'
                
                # 다양한 구조 타입 처리
                if 'state_dict' in model_data:
                    # 표준 PyTorch 모델
                    info.structure_type = 'state_dict'
                    state_dict = model_data['state_dict']
                    
                    # 아키텍처 감지
                    info.architecture_hints = self._detect_architecture_from_keys(state_dict.keys())
                    info.valid = True
                    info.recommendations.append("표준 state_dict 구조")
                    
                elif 'model' in model_data:
                    # 모델 래퍼 구조
                    info.structure_type = 'model_wrapper'
                    info.valid = True
                    info.recommendations.append("모델 래퍼 구조")
                    
                elif 'weights' in model_data:
                    # 가중치만 있는 구조
                    info.structure_type = 'weights_only'
                    info.valid = True
                    info.recommendations.append("가중치 전용 구조")
                    
                elif 'parameters' in model_data:
                    # 파라미터만 있는 구조
                    info.structure_type = 'parameters_only'
                    info.valid = True
                    info.recommendations.append("파라미터 전용 구조")
                    
                else:
                    # 커스텀 딕셔너리 구조
                    info.structure_type = 'custom_dict'
                    
                    # 커스텀 구조에서도 파라미터 찾기 시도
                    total_params = 0
                    param_keys = []
                    
                    # 🔥 중첩된 구조 처리 (RealESRGAN 등)
                    def extract_tensors(obj, prefix=""):
                        nonlocal total_params, param_keys
                        if isinstance(obj, torch.Tensor):
                            total_params += obj.numel()
                            param_keys.append(prefix)
                        elif isinstance(obj, dict):
                            for key, value in obj.items():
                                new_prefix = f"{prefix}.{key}" if prefix else key
                                extract_tensors(value, new_prefix)
                    
                    extract_tensors(model_data)
                    
                    if total_params > 0:
                        info.architecture_hints = self._detect_architecture_from_keys(param_keys)
                        info.valid = True
                        info.recommendations.append("커스텀 구조에서 파라미터 발견")
                    else:
                        info.issues.append("파라미터를 찾을 수 없음")
                        info.recommendations.append("커스텀 구조 검증 필요")
                        
            elif isinstance(model_data, torch.Tensor):
                # 직접 텐서 형태
                info.structure_type = 'tensor'
                info.valid = True
                info.recommendations.append("직접 텐서 형태")
                
            elif hasattr(model_data, 'state_dict'):
                # TorchScript 모델
                info.structure_type = 'torchscript'
                try:
                    state_dict = model_data.state_dict()
                    info.architecture_hints = self._detect_architecture_from_keys(state_dict.keys())
                    info.valid = True
                    info.recommendations.append("TorchScript 모델")
                except Exception as e:
                    info.issues.append(f"TorchScript state_dict 접근 실패: {e}")
                    
            else:
                info.structure_type = str(type(model_data))
                info.issues.append(f"지원하지 않는 타입: {type(model_data)}")
                
        except Exception as e:
            info.issues.append(f"모델 분석 중 오류: {e}")
        
        return info
    
    def _detect_architecture_from_keys(self, keys: List[str]) -> List[str]:
        """키 목록에서 아키텍처 감지"""
        hints = []
        
        # 확장된 아키텍처 키워드
        architecture_keywords = {
            'graphonomy': ['backbone', 'decoder', 'classifier', 'schp', 'hrnet'],
            'u2net': ['stage1', 'stage2', 'stage3', 'stage4', 'side', 'u2net'],
            'deeplabv3plus': ['backbone', 'decoder', 'classifier', 'aspp', 'deeplab'],
            'gmm': ['feature_extraction', 'regression', 'gmm', 'geometric', 'pretrained.model'],
            'tps': ['localization_net', 'grid_generator', 'tps', 'transformation'],
            'raft': ['feature_encoder', 'context_encoder', 'flow_head', 'raft'],
            'sam': ['image_encoder', 'prompt_encoder', 'mask_decoder', 'sam'],
            'stable_diffusion': ['unet', 'vae', 'text_encoder', 'diffusion', 'model'],
            'ootd': ['unet_vton', 'unet_garm', 'vae', 'ootd'],
            'real_esrgan': ['body', 'upsampling', 'esrgan', 'real_esrgan'],
            'swinir': ['layers', 'patch_embed', 'norm', 'swin', 'swinir'],
            'clip': ['visual', 'transformer', 'text_projection', 'clip'],
            'vit': ['cls_token', 'pos_embed', 'patch_embed', 'blocks', 'attn', 'mlp'],
            'hrnet': ['hrnet', 'stage', 'transition', 'hrnet_w'],
            'openpose': ['pose', 'body', 'hand', 'face', 'openpose'],
            'yolo': ['yolo', 'detect', 'anchor', 'yolov'],
            'mediapipe': ['mediapipe', 'landmark', 'pose'],
            'viton': ['viton', 'vton', 'warping', 'tom'],
            'dpt': ['dpt', 'depth', 'midas'],
            'efficientnet': ['efficientnet', 'efficient'],
            'resnet': ['resnet', 'residual'],
            'mobilenet': ['mobilenet', 'mobile'],
            'densenet': ['densenet', 'dense'],
            'unet': ['down_blocks', 'up_blocks', 'conv_in', 'conv_out', 'time_embedding'],
            'diffusion': ['down_blocks', 'up_blocks', 'time_embedding', 'conv_in', 'conv_out']
        }
        
        for arch_name, keywords in architecture_keywords.items():
            matches = sum(1 for keyword in keywords if any(keyword.lower() in key.lower() for key in keys))
            if matches > 0:
                hints.append(f"{arch_name} (매칭: {matches}개)")
        
        return hints
    
    def analyze_all_models(self) -> Dict[str, Any]:
        """모든 AI 모델 분석"""
        print("🔍 모든 AI 모델 파일 검색 중...")
        model_files = self.find_all_ai_models()
        
        print(f"📊 총 {len(model_files)}개의 AI 모델 파일 발견")
        
        results = {
            'analysis_time': datetime.now().isoformat(),
            'total_models': len(model_files),
            'valid_models': 0,
            'invalid_models': 0,
            'models': {},
            'step_summary': {},
            'architecture_summary': {},
            'issues_summary': {},
            'recommendations': []
        }
        
        for i, model_path in enumerate(model_files, 1):
            print(f"🔍 분석 중... ({i}/{len(model_files)}): {Path(model_path).name}")
            
            info = self.analyze_model(model_path)
            self.models[model_path] = info
            
            # 결과 저장
            results['models'][model_path] = {
                'size_mb': info.size_mb,
                'exists': info.exists,
                'valid': info.valid,
                'structure_type': info.structure_type,
                'architecture_hints': info.architecture_hints,
                'issues': info.issues,
                'recommendations': info.recommendations,
                'step_category': info.step_category,
                'model_type': info.model_type
            }
            
            # 통계 업데이트
            if info.valid:
                results['valid_models'] += 1
            else:
                results['invalid_models'] += 1
            
            # Step별 요약
            if info.step_category not in results['step_summary']:
                results['step_summary'][info.step_category] = {
                    'total': 0,
                    'valid': 0,
                    'invalid': 0,
                    'models': []
                }
            
            results['step_summary'][info.step_category]['total'] += 1
            if info.valid:
                results['step_summary'][info.step_category]['valid'] += 1
            else:
                results['step_summary'][info.step_category]['invalid'] += 1
            
            results['step_summary'][info.step_category]['models'].append({
                'path': model_path,
                'name': Path(model_path).name,
                'valid': info.valid,
                'size_mb': info.size_mb
            })
            
            # 아키텍처 요약
            for hint in info.architecture_hints:
                arch_name = hint.split(' (')[0]
                if arch_name not in results['architecture_summary']:
                    results['architecture_summary'][arch_name] = 0
                results['architecture_summary'][arch_name] += 1
            
            # 문제점 요약
            for issue in info.issues:
                if issue not in results['issues_summary']:
                    results['issues_summary'][issue] = 0
                results['issues_summary'][issue] += 1
        
        return results
    
    def fix_model_compatibility(self, model_path: str) -> bool:
        """모델 호환성 수정"""
        if model_path not in self.models:
            print(f"❌ 모델 정보가 없음: {model_path}")
            return False
        
        info = self.models[model_path]
        
        if not info.exists:
            print(f"❌ 파일이 존재하지 않음: {model_path}")
            return False
        
        if info.valid:
            print(f"✅ 이미 유효한 모델: {model_path}")
            return True
        
        print(f"🔧 호환성 수정 시도: {model_path}")
        
        try:
            # 백업 생성
            backup_path = f"{model_path}.backup"
            if not Path(backup_path).exists():
                shutil.copy2(model_path, backup_path)
                print(f"📦 백업 생성: {backup_path}")
            
            # 모델 로딩
            model_data = None
            
            # 다양한 로딩 방법 시도
            for method in ['weights_only_true', 'weights_only_false', 'torchscript', 'safetensors']:
                try:
                    if method == 'weights_only_true':
                        model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                    elif method == 'weights_only_false':
                        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
                    elif method == 'torchscript':
                        model_data = torch.jit.load(model_path, map_location='cpu')
                    elif method == 'safetensors' and SAFETENSORS_AVAILABLE:
                        with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
                            keys = list(f.keys())
                            model_data = {key: f.get_tensor(key) for key in keys}
                    
                    print(f"✅ {method}로 로딩 성공")
                    break
                except Exception as e:
                    print(f"❌ {method} 로딩 실패: {e}")
                    continue
            
            if model_data is None:
                print(f"❌ 모든 로딩 방법 실패")
                return False
            
            # 호환성 수정
            fixed_model = self._fix_model_structure(model_data, info)
            
            if fixed_model is not None:
                # 수정된 모델 저장
                torch.save(fixed_model, model_path)
                print(f"✅ 호환성 수정 완료: {model_path}")
                
                # 정보 업데이트
                info.fixed = True
                info.valid = True
                info.recommendations.append("호환성 수정 완료")
                
                return True
            else:
                print(f"❌ 호환성 수정 실패")
                return False
                
        except Exception as e:
            print(f"❌ 호환성 수정 중 오류: {e}")
            return False
    
    def _fix_model_structure(self, model_data: Any, info: AIModelInfo) -> Optional[Any]:
        """모델 구조 수정"""
        try:
            if isinstance(model_data, dict):
                # 1. state_dict 키 매핑 수정
                if 'state_dict' in model_data:
                    state_dict = model_data['state_dict']
                    fixed_state_dict = self._fix_state_dict_keys(state_dict)
                    model_data['state_dict'] = fixed_state_dict
                    return model_data
                
                # 2. 직접 딕셔너리인 경우
                else:
                    fixed_dict = self._fix_state_dict_keys(model_data)
                    return {'state_dict': fixed_dict}
            
            elif hasattr(model_data, 'state_dict'):
                # TorchScript 모델
                state_dict = model_data.state_dict()
                fixed_state_dict = self._fix_state_dict_keys(state_dict)
                return {'state_dict': fixed_state_dict}
            
            else:
                return model_data
                
        except Exception as e:
            print(f"❌ 구조 수정 중 오류: {e}")
            return None
    
    def _fix_state_dict_keys(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """state_dict 키 수정"""
        fixed_state_dict = {}
        
        # 키 매핑 규칙
        key_mappings = {
            # TPS 관련 키 매핑
            'control_points': 'tps_control_points',
            'weights': 'tps_weights', 
            'affine_params': 'tps_affine_params',
            
            # GMM 관련 키 매핑
            'pretrained.model': 'backbone',
            'feature_extraction': 'encoder',
            'regression': 'decoder',
            
            # 일반적인 키 매핑
            'module.': '',  # DataParallel 제거
            'model.': '',   # 모델 래퍼 제거
        }
        
        for key, tensor in state_dict.items():
            new_key = key
            
            # 키 매핑 적용
            for old_pattern, new_pattern in key_mappings.items():
                if old_pattern in new_key:
                    new_key = new_key.replace(old_pattern, new_pattern)
                    break
            
            fixed_state_dict[new_key] = tensor
        
        return fixed_state_dict
    
    def generate_comprehensive_report(self) -> str:
        """종합 리포트 생성"""
        report = []
        report.append("🔥 종합 AI 모델 분석 리포트")
        report.append("=" * 80)
        report.append(f"📅 분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 전체 통계
        total = len(self.models)
        valid = sum(1 for info in self.models.values() if info.valid)
        invalid = total - valid
        
        report.append(f"📊 전체 AI 모델: {total}개")
        report.append(f"✅ 유효한 모델: {valid}개")
        report.append(f"❌ 무효한 모델: {invalid}개")
        report.append("")
        
        # Step별 요약
        report.append("🎯 Step별 모델 현황:")
        for step_key, step_info in self.step_categories.items():
            step_models = [info for info in self.models.values() if info.step_category == step_key]
            if step_models:
                step_valid = sum(1 for info in step_models if info.valid)
                step_invalid = len(step_models) - step_valid
                report.append(f"   {step_info['name']} ({step_key}): {len(step_models)}개 (✅{step_valid}개, ❌{step_invalid}개)")
        report.append("")
        
        # 아키텍처별 요약
        architecture_counts = {}
        for info in self.models.values():
            for hint in info.architecture_hints:
                arch_name = hint.split(' (')[0]
                architecture_counts[arch_name] = architecture_counts.get(arch_name, 0) + 1
        
        if architecture_counts:
            report.append("🏗️ 아키텍처별 분포:")
            for arch, count in sorted(architecture_counts.items(), key=lambda x: x[1], reverse=True):
                report.append(f"   {arch}: {count}개")
            report.append("")
        
        # 문제점별 요약
        issue_counts = {}
        for info in self.models.values():
            for issue in info.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        if issue_counts:
            report.append("🚨 주요 문제점:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"   {issue}: {count}개")
            report.append("")
        
        # 상세 분석 결과
        report.append("📋 상세 분석 결과:")
        for model_path, info in self.models.items():
            status = "✅" if info.valid else "❌"
            step_name = self.step_categories.get(info.step_category, {}).get('name', info.step_category)
            report.append(f"{status} {Path(model_path).name}")
            report.append(f"   📁 크기: {info.size_mb:.1f}MB")
            report.append(f"   🎯 Step: {step_name}")
            report.append(f"   🏗️ 구조: {info.structure_type}")
            
            if info.architecture_hints:
                report.append(f"   🏛️ 아키텍처: {', '.join(info.architecture_hints)}")
            
            if info.issues:
                report.append(f"   ⚠️ 문제점: {', '.join(info.issues)}")
            
            if info.recommendations:
                report.append(f"   💡 권장사항: {', '.join(info.recommendations)}")
            
            report.append("")
        
        return "\n".join(report)
    
    def save_analysis_results(self, output_path: str = "comprehensive_ai_model_analysis.json"):
        """분석 결과 저장"""
        results = {
            'analysis_time': datetime.now().isoformat(),
            'total_models': len(self.models),
            'valid_models': sum(1 for info in self.models.values() if info.valid),
            'invalid_models': sum(1 for info in self.models.values() if not info.valid),
            'models': {}
        }
        
        for model_path, info in self.models.items():
            results['models'][model_path] = {
                'size_mb': info.size_mb,
                'exists': info.exists,
                'valid': info.valid,
                'structure_type': info.structure_type,
                'architecture_hints': info.architecture_hints,
                'issues': info.issues,
                'recommendations': info.recommendations,
                'step_category': info.step_category,
                'model_type': info.model_type,
                'fixed': info.fixed
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 분석 결과 저장: {output_path}")

def main():
    """메인 함수"""
    print("🔥 종합 AI 모델 분석 및 업데이트 도구")
    print("=" * 80)
    
    # 분석기 초기화
    analyzer = ComprehensiveAIModelAnalyzer()
    
    # 1. 모든 AI 모델 분석
    print("\n📋 1단계: 모든 AI 모델 분석")
    results = analyzer.analyze_all_models()
    
    # 2. 분석 결과 출력
    print("\n📊 분석 결과:")
    print(f"   총 AI 모델: {results['total_models']}개")
    print(f"   유효한 모델: {results['valid_models']}개")
    print(f"   무효한 모델: {results['invalid_models']}개")
    
    # 3. Step별 현황 출력
    print("\n🎯 Step별 현황:")
    for step_key, step_info in analyzer.step_categories.items():
        step_models = [info for info in analyzer.models.values() if info.step_category == step_key]
        if step_models:
            step_valid = sum(1 for info in step_models if info.valid)
            step_invalid = len(step_models) - step_valid
            print(f"   {step_info['name']}: {len(step_models)}개 (✅{step_valid}개, ❌{step_invalid}개)")
    
    # 4. 호환성 수정 시도
    print("\n🔧 2단계: 호환성 수정 시도")
    fixed_count = 0
    
    for model_path, info in analyzer.models.items():
        if not info.valid and info.exists:
            if analyzer.fix_model_compatibility(model_path):
                fixed_count += 1
    
    print(f"\n✅ 수정 완료: {fixed_count}개")
    
    # 5. 최종 리포트 생성
    print("\n📋 3단계: 최종 리포트 생성")
    report = analyzer.generate_comprehensive_report()
    print(report)
    
    # 6. 결과 저장
    analyzer.save_analysis_results()
    
    print("\n🎉 종합 AI 모델 분석 및 수정 완료!")

if __name__ == "__main__":
    main()
