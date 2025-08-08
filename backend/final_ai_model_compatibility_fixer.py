#!/usr/bin/env python3
"""
🔥 최종 AI 모델 호환성 개선 도구
================================

모든 AI 모델들의 호환성을 개선하고 체크포인트 키 매핑을 수정하는 도구

Author: MyCloset AI Team
Date: 2025-08-08
Version: 3.0
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# PyTorch 관련
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# SafeTensors 관련
try:
    import safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logger = logging.getLogger(__name__)

class FinalAIModelCompatibilityFixer:
    """최종 AI 모델 호환성 개선 도구"""
    
    def __init__(self):
        self.fixed_models = []
        self.failed_models = []
        self.compatibility_issues = []
        
        # Step별 호환성 매핑 규칙
        self.compatibility_mappings = {
            'step_01': {
                'graphonomy': {
                    'old_keys': ['backbone', 'decoder', 'classifier'],
                    'new_keys': ['hrnet_backbone', 'hrnet_decoder', 'hrnet_classifier'],
                    'architecture': 'hrnet'
                },
                'u2net': {
                    'old_keys': ['stage1', 'stage2', 'stage3', 'stage4'],
                    'new_keys': ['u2net_stage1', 'u2net_stage2', 'u2net_stage3', 'u2net_stage4'],
                    'architecture': 'u2net'
                },
                'deeplabv3plus': {
                    'old_keys': ['backbone', 'decoder', 'classifier'],
                    'new_keys': ['deeplab_backbone', 'deeplab_decoder', 'deeplab_classifier'],
                    'architecture': 'deeplabv3plus'
                }
            },
            'step_02': {
                'hrnet': {
                    'old_keys': ['hrnet', 'stage', 'transition'],
                    'new_keys': ['pose_hrnet', 'pose_stage', 'pose_transition'],
                    'architecture': 'hrnet'
                },
                'openpose': {
                    'old_keys': ['pose', 'body', 'hand'],
                    'new_keys': ['openpose_pose', 'openpose_body', 'openpose_hand'],
                    'architecture': 'openpose'
                }
            },
            'step_03': {
                'sam': {
                    'old_keys': ['image_encoder', 'prompt_encoder', 'mask_decoder'],
                    'new_keys': ['sam_image_encoder', 'sam_prompt_encoder', 'sam_mask_decoder'],
                    'architecture': 'sam'
                },
                'u2net': {
                    'old_keys': ['stage1', 'stage2', 'stage3', 'stage4'],
                    'new_keys': ['seg_u2net_stage1', 'seg_u2net_stage2', 'seg_u2net_stage3', 'seg_u2net_stage4'],
                    'architecture': 'u2net'
                }
            },
            'step_04': {
                'gmm': {
                    'old_keys': ['feature_extraction', 'regression', 'pretrained.model'],
                    'new_keys': ['gmm_feature_extraction', 'gmm_regression', 'gmm_backbone'],
                    'architecture': 'gmm'
                },
                'tps': {
                    'old_keys': ['localization_net', 'grid_generator', 'control_points'],
                    'new_keys': ['tps_localization_net', 'tps_grid_generator', 'tps_control_points'],
                    'architecture': 'tps'
                },
                'raft': {
                    'old_keys': ['feature_encoder', 'context_encoder', 'flow_head'],
                    'new_keys': ['raft_feature_encoder', 'raft_context_encoder', 'raft_flow_head'],
                    'architecture': 'raft'
                }
            },
            'step_05': {
                'tom': {
                    'old_keys': ['feature_extraction', 'regression'],
                    'new_keys': ['tom_feature_extraction', 'tom_regression'],
                    'architecture': 'tom'
                },
                'viton_hd': {
                    'old_keys': ['warping', 'generator'],
                    'new_keys': ['viton_warping', 'viton_generator'],
                    'architecture': 'viton_hd'
                },
                'dpt': {
                    'old_keys': ['dpt', 'depth'],
                    'new_keys': ['dpt_depth', 'dpt_backbone'],
                    'architecture': 'dpt'
                }
            },
            'step_06': {
                'stable_diffusion': {
                    'old_keys': ['unet', 'vae', 'text_encoder'],
                    'new_keys': ['sd_unet', 'sd_vae', 'sd_text_encoder'],
                    'architecture': 'stable_diffusion'
                },
                'ootd': {
                    'old_keys': ['unet_vton', 'unet_garm', 'vae'],
                    'new_keys': ['ootd_unet_vton', 'ootd_unet_garm', 'ootd_vae'],
                    'architecture': 'ootd'
                }
            },
            'step_07': {
                'real_esrgan': {
                    'old_keys': ['body', 'upsampling'],
                    'new_keys': ['esrgan_body', 'esrgan_upsampling'],
                    'architecture': 'real_esrgan'
                },
                'swinir': {
                    'old_keys': ['layers', 'patch_embed', 'norm'],
                    'new_keys': ['swinir_layers', 'swinir_patch_embed', 'swinir_norm'],
                    'architecture': 'swinir'
                }
            },
            'step_08': {
                'clip': {
                    'old_keys': ['visual', 'transformer', 'text_projection'],
                    'new_keys': ['clip_visual', 'clip_transformer', 'clip_text_projection'],
                    'architecture': 'clip'
                },
                'lpips': {
                    'old_keys': ['alex', 'net'],
                    'new_keys': ['lpips_alex', 'lpips_net'],
                    'architecture': 'lpips'
                }
            }
        }
    
    def find_all_ai_models(self) -> List[str]:
        """모든 AI 모델 파일 찾기"""
        model_files = []
        
        # 다양한 확장자 검색
        extensions = ["*.pth", "*.pt", "*.safetensors", "*.ckpt", "*.bin"]
        
        for ext in extensions:
            files = list(Path(".").rglob(ext))
            model_files.extend([str(f) for f in files])
        
        # 중복 제거 및 정렬
        model_files = sorted(list(set(model_files)))
        
        # 불필요한 파일 필터링
        exclude_patterns = [
            'distutils-precedence.pth',
            '__pycache__',
            '.git',
            'node_modules',
            'venv',
            'env',
            '.conda',
            '.backup',
            '_temp',
            '_old'
        ]
        
        filtered_files = []
        for file_path in model_files:
            if not any(pattern in file_path for pattern in exclude_patterns):
                filtered_files.append(file_path)
        
        return filtered_files
    
    def categorize_model(self, model_path: str) -> Tuple[str, str]:
        """모델을 Step별로 분류"""
        model_path_lower = model_path.lower()
        
        # Step별 키워드 매핑
        step_keywords = {
            'step_01': ['human_parsing', 'graphonomy', 'u2net', 'deeplab'],
            'step_02': ['pose_estimation', 'hrnet', 'openpose', 'yolo'],
            'step_03': ['cloth_segmentation', 'sam', 'segmentation'],
            'step_04': ['geometric_matching', 'gmm', 'tps', 'raft'],
            'step_05': ['cloth_warping', 'tom', 'viton', 'dpt'],
            'step_06': ['virtual_fitting', 'stable_diffusion', 'ootd', 'diffusion'],
            'step_07': ['post_processing', 'real_esrgan', 'swinir', 'gfpgan'],
            'step_08': ['quality_assessment', 'clip', 'lpips', 'alex']
        }
        
        for step_key, keywords in step_keywords.items():
            for keyword in keywords:
                if keyword in model_path_lower:
                    return step_key, keyword
        
        return "unknown", "unknown"
    
    def detect_model_architecture(self, state_dict_keys: List[str]) -> str:
        """모델 아키텍처 감지"""
        architecture_keywords = {
            'graphonomy': ['backbone', 'decoder', 'classifier', 'hrnet'],
            'u2net': ['stage1', 'stage2', 'stage3', 'stage4', 'u2net'],
            'deeplabv3plus': ['backbone', 'decoder', 'classifier', 'aspp', 'deeplab'],
            'gmm': ['feature_extraction', 'regression', 'gmm', 'geometric'],
            'tps': ['localization_net', 'grid_generator', 'tps', 'transformation'],
            'raft': ['feature_encoder', 'context_encoder', 'flow_head', 'raft'],
            'sam': ['image_encoder', 'prompt_encoder', 'mask_decoder', 'sam'],
            'stable_diffusion': ['unet', 'vae', 'text_encoder', 'diffusion'],
            'ootd': ['unet_vton', 'unet_garm', 'vae', 'ootd'],
            'real_esrgan': ['body', 'upsampling', 'esrgan'],
            'swinir': ['layers', 'patch_embed', 'norm', 'swin'],
            'clip': ['visual', 'transformer', 'text_projection', 'clip'],
            'hrnet': ['hrnet', 'stage', 'transition'],
            'openpose': ['pose', 'body', 'hand', 'face'],
            'tom': ['feature_extraction', 'regression', 'tom'],
            'viton_hd': ['viton', 'vton', 'warping'],
            'dpt': ['dpt', 'depth', 'midas']
        }
        
        for arch_name, keywords in architecture_keywords.items():
            matches = sum(1 for keyword in keywords if any(keyword.lower() in key.lower() for key in state_dict_keys))
            if matches > 0:
                return arch_name
        
        return "unknown"
    
    def fix_model_compatibility(self, model_path: str) -> bool:
        """모델 호환성 수정"""
        try:
            print(f"\n🔧 호환성 수정 시도: {Path(model_path).name}")
            
            # 백업 생성
            backup_path = f"{model_path}.backup"
            if not Path(backup_path).exists():
                shutil.copy2(model_path, backup_path)
                print(f"   📦 백업 생성: {backup_path}")
            
            # 모델 로딩
            model_data = None
            
            # 다양한 로딩 방법 시도
            for method in ['weights_only_true', 'weights_only_false', 'safetensors']:
                try:
                    if method == 'weights_only_true':
                        model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                    elif method == 'weights_only_false':
                        model_data = torch.load(model_path, map_location='cpu', weights_only=False)
                    elif method == 'safetensors' and SAFETENSORS_AVAILABLE:
                        with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
                            keys = list(f.keys())
                            model_data = {key: f.get_tensor(key) for key in keys}
                    
                    if model_data is not None:
                        print(f"   ✅ {method}로 로딩 성공")
                        break
                except Exception as e:
                    print(f"   ❌ {method} 로딩 실패: {e}")
                    continue
            
            if model_data is None:
                print(f"   ❌ 모든 로딩 방법 실패")
                return False
            
            # Step 분류
            step_category, model_type = self.categorize_model(model_path)
            print(f"   🎯 Step 분류: {step_category} ({model_type})")
            
            # state_dict 추출
            state_dict = None
            if isinstance(model_data, dict):
                if 'state_dict' in model_data:
                    state_dict = model_data['state_dict']
                else:
                    state_dict = model_data
            else:
                print(f"   ❌ state_dict를 찾을 수 없음")
                return False
            
            # 아키텍처 감지
            architecture = self.detect_model_architecture(list(state_dict.keys()))
            print(f"   🏗️ 아키텍처: {architecture}")
            
            # 호환성 매핑 적용
            fixed_state_dict = self._apply_compatibility_mapping(
                state_dict, step_category, architecture
            )
            
            # 수정된 모델 저장
            if isinstance(model_data, dict) and 'state_dict' in model_data:
                model_data['state_dict'] = fixed_state_dict
            else:
                model_data = {'state_dict': fixed_state_dict}
            
            torch.save(model_data, model_path)
            print(f"   ✅ 호환성 수정 완료")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 호환성 수정 실패: {e}")
            return False
    
    def _apply_compatibility_mapping(self, state_dict: Dict[str, torch.Tensor], 
                                   step_category: str, architecture: str) -> Dict[str, torch.Tensor]:
        """호환성 매핑 적용"""
        fixed_state_dict = {}
        
        # Step별 매핑 규칙 적용
        if step_category in self.compatibility_mappings:
            step_mappings = self.compatibility_mappings[step_category]
            
            for model_name, mapping_info in step_mappings.items():
                if model_name.lower() in architecture.lower():
                    old_keys = mapping_info['old_keys']
                    new_keys = mapping_info['new_keys']
                    
                    # 키 매핑 적용
                    for old_key, new_key in zip(old_keys, new_keys):
                        for state_key in state_dict.keys():
                            if old_key.lower() in state_key.lower():
                                new_state_key = state_key.replace(old_key, new_key)
                                fixed_state_dict[new_state_key] = state_dict[state_key]
                                print(f"   🔄 키 매핑: {state_key} → {new_state_key}")
                                break
        
        # 매핑되지 않은 키들은 그대로 유지
        for key, tensor in state_dict.items():
            if key not in fixed_state_dict:
                fixed_state_dict[key] = tensor
        
        return fixed_state_dict
    
    def fix_tps_compatibility_issues(self):
        """TPS 호환성 문제 특별 수정"""
        print("\n🔧 TPS 호환성 문제 특별 수정")
        
        # TPS 관련 모델들 찾기
        tps_models = [
            "backend/ai_models/step_04_geometric_matching/tps_network.pth",
            "backend/ai_models/step_05_cloth_warping/tps_transformation.pth"
        ]
        
        for model_path in tps_models:
            if Path(model_path).exists():
                print(f"\n🔧 TPS 모델 수정: {Path(model_path).name}")
                
                try:
                    # 백업 생성
                    backup_path = f"{model_path}.backup"
                    if not Path(backup_path).exists():
                        shutil.copy2(model_path, backup_path)
                    
                    # 모델 로딩
                    model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                    
                    if isinstance(model_data, dict):
                        state_dict = model_data.get('state_dict', model_data)
                    else:
                        state_dict = model_data
                    
                    # TPS 키 매핑 수정
                    fixed_state_dict = {}
                    for key, tensor in state_dict.items():
                        new_key = key
                        
                        # TPS 관련 키 매핑
                        if 'control_points' in key:
                            new_key = key.replace('control_points', 'tps_control_points')
                        elif 'weights' in key:
                            new_key = key.replace('weights', 'tps_weights')
                        elif 'affine_params' in key:
                            new_key = key.replace('affine_params', 'tps_affine_params')
                        
                        fixed_state_dict[new_key] = tensor
                    
                    # 수정된 모델 저장
                    if isinstance(model_data, dict) and 'state_dict' in model_data:
                        model_data['state_dict'] = fixed_state_dict
                    else:
                        model_data = {'state_dict': fixed_state_dict}
                    
                    torch.save(model_data, model_path)
                    print(f"   ✅ TPS 호환성 수정 완료")
                    self.fixed_models.append(model_path)
                    
                except Exception as e:
                    print(f"   ❌ TPS 수정 실패: {e}")
                    self.failed_models.append(model_path)
    
    def fix_gmm_compatibility_issues(self):
        """GMM 호환성 문제 특별 수정"""
        print("\n🔧 GMM 호환성 문제 특별 수정")
        
        # GMM 관련 모델들 찾기
        gmm_models = [
            "backend/ai_models/step_04_geometric_matching/gmm_final.pth"
        ]
        
        for model_path in gmm_models:
            if Path(model_path).exists():
                print(f"\n🔧 GMM 모델 수정: {Path(model_path).name}")
                
                try:
                    # 백업 생성
                    backup_path = f"{model_path}.backup"
                    if not Path(backup_path).exists():
                        shutil.copy2(model_path, backup_path)
                    
                    # 모델 로딩
                    model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                    
                    if isinstance(model_data, dict):
                        state_dict = model_data.get('state_dict', model_data)
                    else:
                        state_dict = model_data
                    
                    # GMM 키 매핑 수정
                    fixed_state_dict = {}
                    for key, tensor in state_dict.items():
                        new_key = key
                        
                        # GMM 관련 키 매핑
                        if 'pretrained.model' in key:
                            new_key = key.replace('pretrained.model', 'gmm_backbone')
                        elif 'feature_extraction' in key:
                            new_key = key.replace('feature_extraction', 'gmm_feature_extraction')
                        elif 'regression' in key:
                            new_key = key.replace('regression', 'gmm_regression')
                        
                        fixed_state_dict[new_key] = tensor
                    
                    # 수정된 모델 저장
                    if isinstance(model_data, dict) and 'state_dict' in model_data:
                        model_data['state_dict'] = fixed_state_dict
                    else:
                        model_data = {'state_dict': fixed_state_dict}
                    
                    torch.save(model_data, model_path)
                    print(f"   ✅ GMM 호환성 수정 완료")
                    self.fixed_models.append(model_path)
                    
                except Exception as e:
                    print(f"   ❌ GMM 수정 실패: {e}")
                    self.failed_models.append(model_path)
    
    def verify_compatibility(self):
        """호환성 검증"""
        print(f"\n🔍 호환성 검증:")
        
        verified_count = 0
        for model_path in self.fixed_models:
            print(f"\n🔍 검증 중: {Path(model_path).name}")
            
            try:
                # 모델 로딩 테스트
                model_data = torch.load(model_path, map_location='cpu', weights_only=True)
                
                if isinstance(model_data, dict):
                    state_dict = model_data.get('state_dict', model_data)
                else:
                    state_dict = model_data
                
                # 키 개수 확인
                key_count = len(state_dict.keys())
                print(f"   ✅ 로딩 성공 (키 수: {key_count})")
                
                # TPS 키 확인
                tps_keys = [key for key in state_dict.keys() if 'tps_' in key]
                if tps_keys:
                    print(f"   🔍 TPS 키 발견: {len(tps_keys)}개")
                
                # GMM 키 확인
                gmm_keys = [key for key in state_dict.keys() if 'gmm_' in key]
                if gmm_keys:
                    print(f"   🔍 GMM 키 발견: {len(gmm_keys)}개")
                
                verified_count += 1
                
            except Exception as e:
                print(f"   ❌ 검증 실패: {e}")
        
        print(f"\n📊 검증 결과: {verified_count}/{len(self.fixed_models)}개 성공")
    
    def generate_compatibility_report(self):
        """호환성 리포트 생성"""
        report = []
        report.append("🔥 AI 모델 호환성 개선 리포트")
        report.append("=" * 80)
        report.append(f"📅 개선 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append(f"📊 호환성 개선 결과:")
        report.append(f"   ✅ 성공: {len(self.fixed_models)}개")
        report.append(f"   ❌ 실패: {len(self.failed_models)}개")
        report.append("")
        
        if self.fixed_models:
            report.append("✅ 호환성 개선 완료된 모델들:")
            for model_path in self.fixed_models:
                report.append(f"   - {Path(model_path).name}")
            report.append("")
        
        if self.failed_models:
            report.append("❌ 호환성 개선 실패한 모델들:")
            for model_path in self.failed_models:
                report.append(f"   - {Path(model_path).name}")
            report.append("")
        
        if self.compatibility_issues:
            report.append("⚠️ 발견된 호환성 문제들:")
            for issue in self.compatibility_issues:
                report.append(f"   - {issue}")
            report.append("")
        
        return "\n".join(report)

def main():
    """메인 함수"""
    print("🔥 최종 AI 모델 호환성 개선 도구")
    print("=" * 80)
    
    # 호환성 개선기 초기화
    fixer = FinalAIModelCompatibilityFixer()
    
    # 1. 모든 AI 모델 파일 찾기
    print("\n📋 1단계: 모든 AI 모델 파일 검색")
    model_files = fixer.find_all_ai_models()
    print(f"   발견된 AI 모델: {len(model_files)}개")
    
    # 2. TPS 호환성 문제 특별 수정
    print("\n🔧 2단계: TPS 호환성 문제 특별 수정")
    fixer.fix_tps_compatibility_issues()
    
    # 3. GMM 호환성 문제 특별 수정
    print("\n🔧 3단계: GMM 호환성 문제 특별 수정")
    fixer.fix_gmm_compatibility_issues()
    
    # 4. 일반적인 호환성 수정
    print("\n🔧 4단계: 일반적인 호환성 수정")
    for i, model_path in enumerate(model_files[:10], 1):  # 처음 10개만 처리
        print(f"\n🔧 처리 중... ({i}/{min(10, len(model_files))})")
        if fixer.fix_model_compatibility(model_path):
            fixer.fixed_models.append(model_path)
        else:
            fixer.failed_models.append(model_path)
    
    # 5. 호환성 검증
    print("\n🔍 5단계: 호환성 검증")
    fixer.verify_compatibility()
    
    # 6. 호환성 리포트 생성
    print("\n📋 6단계: 호환성 리포트 생성")
    report = fixer.generate_compatibility_report()
    print(report)
    
    # 7. 리포트 저장
    with open("ai_model_compatibility_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n💾 호환성 리포트 저장: ai_model_compatibility_report.txt")
    print("\n🎉 AI 모델 호환성 개선 완료!")

if __name__ == "__main__":
    main()
