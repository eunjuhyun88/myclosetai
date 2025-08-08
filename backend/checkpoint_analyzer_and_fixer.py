#!/usr/bin/env python3
"""
🔥 체크포인트 분석 및 호환성 수정 도구
====================================

모든 체크포인트 파일들을 분석하고 호환성 문제를 수정하는 도구

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
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import shutil

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

logger = logging.getLogger(__name__)

@dataclass
class CheckpointInfo:
    """체크포인트 정보"""
    path: str
    size_mb: float = 0.0
    exists: bool = False
    valid: bool = False
    structure_type: str = "unknown"
    architecture_hints: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    fixed: bool = False

class CheckpointAnalyzerAndFixer:
    """체크포인트 분석 및 수정 도구"""
    
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.checkpoints = {}
        self.analysis_results = {}
        
    def find_all_checkpoints(self) -> List[str]:
        """모든 체크포인트 파일 찾기"""
        checkpoint_files = []
        
        # 다양한 확장자 검색
        extensions = ["*.pth", "*.pt", "*.safetensors", "*.ckpt", "*.bin"]
        
        for ext in extensions:
            files = list(self.root_dir.rglob(ext))
            checkpoint_files.extend([str(f) for f in files])
        
        # 중복 제거 및 정렬
        checkpoint_files = sorted(list(set(checkpoint_files)))
        
        logger.info(f"🔍 발견된 체크포인트 파일: {len(checkpoint_files)}개")
        return checkpoint_files
    
    def analyze_checkpoint(self, checkpoint_path: str) -> CheckpointInfo:
        """개별 체크포인트 분석"""
        info = CheckpointInfo(path=checkpoint_path)
        
        try:
            if not Path(checkpoint_path).exists():
                info.issues.append("파일이 존재하지 않음")
                return info
            
            info.exists = True
            info.size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)
            
            # 🔥 다양한 로딩 방법 시도
            checkpoint = None
            loading_method = None
            
            # 방법 1: weights_only=True (안전한 방법)
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                loading_method = 'weights_only_true'
                info.recommendations.append("안전한 weights_only=True로 로딩됨")
            except Exception as e1:
                # 방법 2: weights_only=False (전통적인 방법)
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    loading_method = 'weights_only_false'
                    info.recommendations.append("weights_only=False로 로딩됨 (보안 주의)")
                except Exception as e2:
                    # 방법 3: TorchScript 모델
                    try:
                        checkpoint = torch.jit.load(checkpoint_path, map_location='cpu')
                        loading_method = 'torchscript'
                        info.recommendations.append("TorchScript 모델로 로딩됨")
                    except Exception as e3:
                        # 방법 4: SafeTensors (별도 라이브러리 필요)
                        try:
                            from safetensors import safe_open
                            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                                checkpoint = {key: f.get_tensor(key) for key in f.keys()}
                            loading_method = 'safetensors'
                            info.recommendations.append("SafeTensors로 로딩됨")
                        except Exception as e4:
                            # 방법 5: SafeTensors (keys() 메서드 사용)
                            try:
                                from safetensors import safe_open
                                with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                                    keys = list(f.keys())
                                    checkpoint = {key: f.get_tensor(key) for key in keys}
                                loading_method = 'safetensors_keys'
                                info.recommendations.append("SafeTensors (keys)로 로딩됨")
                            except Exception as e5:
                                info.issues.append(f"모든 로딩 방법 실패: {e5}")
                                return info
            
            # 🔥 구조 타입 분류 및 검증
            if isinstance(checkpoint, dict):
                info.structure_type = 'dict'
                
                # 다양한 구조 타입 처리
                if 'state_dict' in checkpoint:
                    # 표준 PyTorch 모델
                    info.structure_type = 'state_dict'
                    state_dict = checkpoint['state_dict']
                    
                    # 아키텍처 감지
                    info.architecture_hints = self._detect_architecture_from_keys(state_dict.keys())
                    info.valid = True
                    info.recommendations.append("표준 state_dict 구조")
                    
                elif 'model' in checkpoint:
                    # 모델 래퍼 구조
                    info.structure_type = 'model_wrapper'
                    info.valid = True
                    info.recommendations.append("모델 래퍼 구조")
                    
                elif 'weights' in checkpoint:
                    # 가중치만 있는 구조
                    info.structure_type = 'weights_only'
                    info.valid = True
                    info.recommendations.append("가중치 전용 구조")
                    
                elif 'parameters' in checkpoint:
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
                    
                    extract_tensors(checkpoint)
                    
                    if total_params > 0:
                        info.architecture_hints = self._detect_architecture_from_keys(param_keys)
                        info.valid = True
                        info.recommendations.append("커스텀 구조에서 파라미터 발견")
                    else:
                        info.issues.append("파라미터를 찾을 수 없음")
                        info.recommendations.append("커스텀 구조 검증 필요")
                        
            elif isinstance(checkpoint, torch.Tensor):
                # 직접 텐서 형태
                info.structure_type = 'tensor'
                info.valid = True
                info.recommendations.append("직접 텐서 형태")
                
            elif hasattr(checkpoint, 'state_dict'):
                # TorchScript 모델
                info.structure_type = 'torchscript'
                try:
                    state_dict = checkpoint.state_dict()
                    info.architecture_hints = self._detect_architecture_from_keys(state_dict.keys())
                    info.valid = True
                    info.recommendations.append("TorchScript 모델")
                except Exception as e:
                    info.issues.append(f"TorchScript state_dict 접근 실패: {e}")
                    
            else:
                info.structure_type = str(type(checkpoint))
                info.issues.append(f"지원하지 않는 타입: {type(checkpoint)}")
                
        except Exception as e:
            info.issues.append(f"체크포인트 분석 중 오류: {e}")
        
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
    
    def analyze_all_checkpoints(self) -> Dict[str, Any]:
        """모든 체크포인트 분석"""
        print("🔍 모든 체크포인트 파일 검색 중...")
        checkpoint_files = self.find_all_checkpoints()
        
        print(f"📊 총 {len(checkpoint_files)}개의 체크포인트 파일 발견")
        
        results = {
            'analysis_time': datetime.now().isoformat(),
            'total_checkpoints': len(checkpoint_files),
            'valid_checkpoints': 0,
            'invalid_checkpoints': 0,
            'checkpoints': {},
            'architecture_summary': {},
            'issues_summary': {},
            'recommendations': []
        }
        
        for i, checkpoint_path in enumerate(checkpoint_files, 1):
            print(f"🔍 분석 중... ({i}/{len(checkpoint_files)}): {Path(checkpoint_path).name}")
            
            info = self.analyze_checkpoint(checkpoint_path)
            self.checkpoints[checkpoint_path] = info
            
            # 결과 저장
            results['checkpoints'][checkpoint_path] = {
                'size_mb': info.size_mb,
                'exists': info.exists,
                'valid': info.valid,
                'structure_type': info.structure_type,
                'architecture_hints': info.architecture_hints,
                'issues': info.issues,
                'recommendations': info.recommendations
            }
            
            # 통계 업데이트
            if info.valid:
                results['valid_checkpoints'] += 1
            else:
                results['invalid_checkpoints'] += 1
            
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
    
    def fix_checkpoint_compatibility(self, checkpoint_path: str) -> bool:
        """체크포인트 호환성 수정"""
        if checkpoint_path not in self.checkpoints:
            print(f"❌ 체크포인트 정보가 없음: {checkpoint_path}")
            return False
        
        info = self.checkpoints[checkpoint_path]
        
        if not info.exists:
            print(f"❌ 파일이 존재하지 않음: {checkpoint_path}")
            return False
        
        if info.valid:
            print(f"✅ 이미 유효한 체크포인트: {checkpoint_path}")
            return True
        
        print(f"🔧 호환성 수정 시도: {checkpoint_path}")
        
        try:
            # 백업 생성
            backup_path = f"{checkpoint_path}.backup"
            if not Path(backup_path).exists():
                shutil.copy2(checkpoint_path, backup_path)
                print(f"📦 백업 생성: {backup_path}")
            
            # 체크포인트 로딩
            checkpoint = None
            
            # 다양한 로딩 방법 시도
            for method in ['weights_only_true', 'weights_only_false', 'torchscript', 'safetensors']:
                try:
                    if method == 'weights_only_true':
                        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                    elif method == 'weights_only_false':
                        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    elif method == 'torchscript':
                        checkpoint = torch.jit.load(checkpoint_path, map_location='cpu')
                    elif method == 'safetensors':
                        from safetensors import safe_open
                        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                            checkpoint = {key: f.get_tensor(key) for key in f.keys()}
                    
                    print(f"✅ {method}로 로딩 성공")
                    break
                except Exception as e:
                    print(f"❌ {method} 로딩 실패: {e}")
                    continue
            
            if checkpoint is None:
                print(f"❌ 모든 로딩 방법 실패")
                return False
            
            # 호환성 수정
            fixed_checkpoint = self._fix_checkpoint_structure(checkpoint, info)
            
            if fixed_checkpoint is not None:
                # 수정된 체크포인트 저장
                torch.save(fixed_checkpoint, checkpoint_path)
                print(f"✅ 호환성 수정 완료: {checkpoint_path}")
                
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
    
    def _fix_checkpoint_structure(self, checkpoint: Any, info: CheckpointInfo) -> Optional[Any]:
        """체크포인트 구조 수정"""
        try:
            if isinstance(checkpoint, dict):
                # 1. state_dict 키 매핑 수정
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    fixed_state_dict = self._fix_state_dict_keys(state_dict)
                    checkpoint['state_dict'] = fixed_state_dict
                    return checkpoint
                
                # 2. 직접 딕셔너리인 경우
                else:
                    fixed_dict = self._fix_state_dict_keys(checkpoint)
                    return {'state_dict': fixed_dict}
            
            elif hasattr(checkpoint, 'state_dict'):
                # TorchScript 모델
                state_dict = checkpoint.state_dict()
                fixed_state_dict = self._fix_state_dict_keys(state_dict)
                return {'state_dict': fixed_state_dict}
            
            else:
                return checkpoint
                
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
    
    def generate_compatibility_report(self) -> str:
        """호환성 리포트 생성"""
        report = []
        report.append("🔥 체크포인트 호환성 분석 리포트")
        report.append("=" * 60)
        report.append(f"📅 분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 전체 통계
        total = len(self.checkpoints)
        valid = sum(1 for info in self.checkpoints.values() if info.valid)
        invalid = total - valid
        
        report.append(f"📊 전체 체크포인트: {total}개")
        report.append(f"✅ 유효한 체크포인트: {valid}개")
        report.append(f"❌ 무효한 체크포인트: {invalid}개")
        report.append("")
        
        # 아키텍처별 요약
        architecture_counts = {}
        for info in self.checkpoints.values():
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
        for info in self.checkpoints.values():
            for issue in info.issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        if issue_counts:
            report.append("🚨 주요 문제점:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"   {issue}: {count}개")
            report.append("")
        
        # 상세 분석 결과
        report.append("📋 상세 분석 결과:")
        for checkpoint_path, info in self.checkpoints.items():
            status = "✅" if info.valid else "❌"
            report.append(f"{status} {Path(checkpoint_path).name}")
            report.append(f"   📁 크기: {info.size_mb:.1f}MB")
            report.append(f"   🏗️ 구조: {info.structure_type}")
            
            if info.architecture_hints:
                report.append(f"   🏛️ 아키텍처: {', '.join(info.architecture_hints)}")
            
            if info.issues:
                report.append(f"   ⚠️ 문제점: {', '.join(info.issues)}")
            
            if info.recommendations:
                report.append(f"   💡 권장사항: {', '.join(info.recommendations)}")
            
            report.append("")
        
        return "\n".join(report)
    
    def save_analysis_results(self, output_path: str = "checkpoint_analysis_results.json"):
        """분석 결과 저장"""
        results = {
            'analysis_time': datetime.now().isoformat(),
            'total_checkpoints': len(self.checkpoints),
            'valid_checkpoints': sum(1 for info in self.checkpoints.values() if info.valid),
            'invalid_checkpoints': sum(1 for info in self.checkpoints.values() if not info.valid),
            'checkpoints': {}
        }
        
        for checkpoint_path, info in self.checkpoints.items():
            results['checkpoints'][checkpoint_path] = {
                'size_mb': info.size_mb,
                'exists': info.exists,
                'valid': info.valid,
                'structure_type': info.structure_type,
                'architecture_hints': info.architecture_hints,
                'issues': info.issues,
                'recommendations': info.recommendations,
                'fixed': info.fixed
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 분석 결과 저장: {output_path}")

def main():
    """메인 함수"""
    print("🔥 체크포인트 분석 및 호환성 수정 도구")
    print("=" * 60)
    
    # 분석기 초기화
    analyzer = CheckpointAnalyzerAndFixer()
    
    # 1. 모든 체크포인트 분석
    print("\n📋 1단계: 모든 체크포인트 분석")
    results = analyzer.analyze_all_checkpoints()
    
    # 2. 분석 결과 출력
    print("\n📊 분석 결과:")
    print(f"   총 체크포인트: {results['total_checkpoints']}개")
    print(f"   유효한 체크포인트: {results['valid_checkpoints']}개")
    print(f"   무효한 체크포인트: {results['invalid_checkpoints']}개")
    
    # 3. 호환성 수정 시도
    print("\n🔧 2단계: 호환성 수정 시도")
    fixed_count = 0
    
    for checkpoint_path, info in analyzer.checkpoints.items():
        if not info.valid and info.exists:
            if analyzer.fix_checkpoint_compatibility(checkpoint_path):
                fixed_count += 1
    
    print(f"\n✅ 수정 완료: {fixed_count}개")
    
    # 4. 최종 리포트 생성
    print("\n📋 3단계: 최종 리포트 생성")
    report = analyzer.generate_compatibility_report()
    print(report)
    
    # 5. 결과 저장
    analyzer.save_analysis_results()
    
    print("\n🎉 체크포인트 분석 및 수정 완료!")

if __name__ == "__main__":
    main()
