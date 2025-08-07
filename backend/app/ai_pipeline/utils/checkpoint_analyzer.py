#!/usr/bin/env python3
"""
🔥 체크포인트 구조 분석기
======================

다양한 형태의 체크포인트 파일들의 구조를 분석하고 분류하는 시스템

Author: MyCloset AI Team
Date: 2025-08-07
Version: 1.0
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

# PyTorch 관련
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CheckpointStructure:
    """체크포인트 구조 정보"""
    file_path: str
    file_size_mb: float
    structure_type: str  # 'state_dict', 'direct_tensor', 'custom_dict', 'unknown'
    top_level_keys: List[str] = field(default_factory=list)
    state_dict_keys: List[str] = field(default_factory=list)
    total_parameters: int = 0
    layer_groups: Dict[str, int] = field(default_factory=dict)
    architecture_hints: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class CheckpointAnalyzer:
    """체크포인트 구조 분석기"""
    
    def __init__(self):
        self.analyzed_checkpoints = {}
        self.structure_patterns = {}
        
    def analyze_checkpoint_structure(self, checkpoint_path: str) -> CheckpointStructure:
        """체크포인트 구조 분석"""
        structure = CheckpointStructure(
            file_path=checkpoint_path,
            file_size_mb=0.0,
            structure_type='unknown'
        )
        
        try:
            if not Path(checkpoint_path).exists():
                structure.issues.append("파일이 존재하지 않음")
                return structure
            
            structure.file_size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)
            
            # 체크포인트 로드 (안전한 방법)
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            except Exception as e:
                structure.issues.append(f"체크포인트 로딩 실패: {e}")
                return structure
            
            # 구조 분석
            if isinstance(checkpoint, dict):
                structure.top_level_keys = list(checkpoint.keys())
                
                # 구조 타입 분류
                if 'state_dict' in checkpoint:
                    structure.structure_type = 'state_dict'
                    state_dict = checkpoint['state_dict']
                    structure.state_dict_keys = list(state_dict.keys())
                    
                    # 파라미터 수 계산
                    total_params = 0
                    for key, tensor in state_dict.items():
                        if hasattr(tensor, 'numel'):
                            total_params += tensor.numel()
                    structure.total_parameters = total_params
                    
                    # 레이어 그룹 분석
                    layer_groups = {}
                    for key in state_dict.keys():
                        if '.' in key:
                            layer_group = key.split('.')[0]
                            layer_groups[layer_group] = layer_groups.get(layer_group, 0) + 1
                    structure.layer_groups = layer_groups
                    
                    # 아키텍처 힌트 분석
                    structure.architecture_hints = self._detect_architecture_hints(state_dict.keys())
                    
                elif 'model' in checkpoint:
                    structure.structure_type = 'model_wrapper'
                    structure.recommendations.append("model 키가 있는 래퍼 구조")
                    
                elif 'weights' in checkpoint:
                    structure.structure_type = 'weights_only'
                    structure.recommendations.append("weights 키만 있는 구조")
                    
                elif 'parameters' in checkpoint:
                    structure.structure_type = 'parameters_only'
                    structure.recommendations.append("parameters 키만 있는 구조")
                    
                else:
                    structure.structure_type = 'custom_dict'
                    structure.recommendations.append("커스텀 딕셔너리 구조")
                    
            elif isinstance(checkpoint, torch.Tensor):
                structure.structure_type = 'direct_tensor'
                structure.recommendations.append("직접 텐서 형태")
                
            else:
                structure.structure_type = 'unknown'
                structure.issues.append(f"예상하지 못한 타입: {type(checkpoint)}")
                
        except Exception as e:
            structure.issues.append(f"분석 중 오류: {e}")
        
        return structure
    
    def _detect_architecture_hints(self, keys: List[str]) -> List[str]:
        """키 목록에서 아키텍처 힌트 감지"""
        hints = []
        
        # 아키텍처별 키워드 매칭
        architecture_keywords = {
            'graphonomy': ['backbone', 'decoder', 'classifier', 'schp'],
            'u2net': ['stage1', 'stage2', 'stage3', 'stage4', 'side'],
            'deeplabv3plus': ['backbone', 'decoder', 'classifier', 'aspp'],
            'gmm': ['feature_extraction', 'regression', 'gmm'],
            'tps': ['localization_net', 'grid_generator', 'tps'],
            'raft': ['feature_encoder', 'context_encoder', 'flow_head', 'raft'],
            'sam': ['image_encoder', 'prompt_encoder', 'mask_decoder', 'sam'],
            'stable_diffusion': ['unet', 'vae', 'text_encoder', 'diffusion'],
            'ootd': ['unet_vton', 'unet_garm', 'vae', 'ootd'],
            'real_esrgan': ['body', 'upsampling', 'esrgan'],
            'swinir': ['layers', 'patch_embed', 'norm', 'swin'],
            'clip': ['visual', 'transformer', 'text_projection', 'clip'],
            'hrnet': ['hrnet', 'stage', 'transition'],
            'openpose': ['pose', 'body', 'hand', 'face'],
            'yolo': ['yolo', 'detect', 'anchor'],
            'mediapipe': ['mediapipe', 'landmark', 'pose']
        }
        
        for arch_name, keywords in architecture_keywords.items():
            matches = sum(1 for keyword in keywords if any(keyword.lower() in key.lower() for key in keys))
            if matches > 0:
                hints.append(f"{arch_name} (매칭: {matches}개)")
        
        return hints
    
    def analyze_all_checkpoints(self, base_path: str = "ai_models") -> Dict[str, CheckpointStructure]:
        """모든 체크포인트 파일 분석"""
        print("🔍 체크포인트 구조 분석 시작...")
        print("=" * 60)
        
        checkpoint_files = []
        
        # 모든 .pth, .pt, .safetensors 파일 찾기
        for ext in ['*.pth', '*.pt', '*.safetensors']:
            checkpoint_files.extend(Path(base_path).rglob(ext))
        
        print(f"📁 발견된 체크포인트 파일: {len(checkpoint_files)}개")
        
        analyzed_results = {}
        
        for i, checkpoint_path in enumerate(checkpoint_files, 1):
            print(f"\n📊 [{i}/{len(checkpoint_files)}] 분석 중: {checkpoint_path.name}")
            
            structure = self.analyze_checkpoint_structure(str(checkpoint_path))
            analyzed_results[str(checkpoint_path)] = structure
            
            # 결과 출력
            status = "✅" if not structure.issues else "❌"
            print(f"{status} {checkpoint_path.name} ({structure.file_size_mb:.1f}MB)")
            print(f"   🏗️ 구조 타입: {structure.structure_type}")
            print(f"   📊 파라미터: {structure.total_parameters:,}")
            print(f"   🏛️ 아키텍처 힌트: {', '.join(structure.architecture_hints) if structure.architecture_hints else 'None'}")
            
            if structure.issues:
                print(f"   ⚠️ 문제점: {structure.issues}")
        
        # 구조 타입별 통계
        structure_types = {}
        for structure in analyzed_results.values():
            structure_type = structure.structure_type
            structure_types[structure_type] = structure_types.get(structure_type, 0) + 1
        
        print(f"\n📋 구조 타입별 통계:")
        for structure_type, count in structure_types.items():
            print(f"   {structure_type}: {count}개")
        
        return analyzed_results
    
    def generate_structure_patterns(self, analyzed_results: Dict[str, CheckpointStructure]) -> Dict[str, Any]:
        """구조 패턴 생성"""
        patterns = {
            'structure_types': {},
            'architecture_patterns': {},
            'common_issues': {},
            'recommendations': {}
        }
        
        # 구조 타입별 패턴
        for file_path, structure in analyzed_results.items():
            structure_type = structure.structure_type
            if structure_type not in patterns['structure_types']:
                patterns['structure_types'][structure_type] = {
                    'count': 0,
                    'examples': [],
                    'total_parameters': 0,
                    'avg_file_size': 0
                }
            
            pattern = patterns['structure_types'][structure_type]
            pattern['count'] += 1
            pattern['examples'].append(file_path)
            pattern['total_parameters'] += structure.total_parameters
            pattern['avg_file_size'] += structure.file_size_mb
        
        # 평균 계산
        for structure_type, pattern in patterns['structure_types'].items():
            if pattern['count'] > 0:
                pattern['avg_parameters'] = pattern['total_parameters'] // pattern['count']
                pattern['avg_file_size'] = pattern['avg_file_size'] / pattern['count']
        
        # 아키텍처 패턴
        for file_path, structure in analyzed_results.items():
            for hint in structure.architecture_hints:
                arch_name = hint.split(' ')[0]
                if arch_name not in patterns['architecture_patterns']:
                    patterns['architecture_patterns'][arch_name] = {
                        'count': 0,
                        'files': [],
                        'structure_types': set()
                    }
                
                patterns['architecture_patterns'][arch_name]['count'] += 1
                patterns['architecture_patterns'][arch_name]['files'].append(file_path)
                patterns['architecture_patterns'][arch_name]['structure_types'].add(structure.structure_type)
        
        # set을 list로 변환 (JSON 직렬화를 위해)
        for arch_pattern in patterns['architecture_patterns'].values():
            arch_pattern['structure_types'] = list(arch_pattern['structure_types'])
        
        return patterns
    
    def save_analysis_report(self, analyzed_results: Dict[str, CheckpointStructure], 
                           patterns: Dict[str, Any], output_path: str = "checkpoint_analysis_report.json"):
        """분석 리포트 저장"""
        report = {
            'analysis_time': datetime.now().isoformat(),
            'total_checkpoints': len(analyzed_results),
            'patterns': patterns,
            'detailed_results': {}
        }
        
        # 상세 결과 (파일 크기 제한을 위해 주요 정보만)
        for file_path, structure in analyzed_results.items():
            report['detailed_results'][file_path] = {
                'file_size_mb': structure.file_size_mb,
                'structure_type': structure.structure_type,
                'total_parameters': structure.total_parameters,
                'architecture_hints': structure.architecture_hints,
                'issues': structure.issues,
                'recommendations': structure.recommendations
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 분석 리포트 저장됨: {output_path}")

def main():
    """메인 실행 함수"""
    analyzer = CheckpointAnalyzer()
    
    # 모든 체크포인트 분석
    analyzed_results = analyzer.analyze_all_checkpoints()
    
    # 패턴 생성
    patterns = analyzer.generate_structure_patterns(analyzed_results)
    
    # 리포트 저장
    analyzer.save_analysis_report(analyzed_results, patterns)
    
    print("\n🎉 체크포인트 구조 분석 완료!")

if __name__ == "__main__":
    main()
