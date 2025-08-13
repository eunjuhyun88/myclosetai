"""
🔥 Cloth Segmentation 체크포인트 분석기
====================================

체크포인트 구조 분석 및 키 매핑 시스템

Author: MyCloset AI Team
Date: 2025-08-07
Version: 1.0
"""

import os
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class CheckpointAnalyzer:
    """체크포인트 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CheckpointAnalyzer")
        self.analyzed_checkpoints = {}
    
    def analyze_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """체크포인트 분석"""
        try:
            self.logger.info(f"🔍 체크포인트 분석 시작: {checkpoint_path}")
            
            if not os.path.exists(checkpoint_path):
                self.logger.warning(f"⚠️ 체크포인트 파일이 존재하지 않음: {checkpoint_path}")
                return {}
            
            # 체크포인트 로딩
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # state_dict 추출
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 분석 결과
            analysis = {
                'file_path': checkpoint_path,
                'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
                'total_keys': len(state_dict),
                'key_patterns': self._analyze_key_patterns(state_dict),
                'layer_types': self._analyze_layer_types(state_dict),
                'parameter_count': sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel')),
                'architecture_hints': self._infer_architecture(state_dict),
                'sample_keys': list(state_dict.keys())[:10]
            }
            
            self.analyzed_checkpoints[checkpoint_path] = analysis
            self.logger.info(f"✅ 체크포인트 분석 완료: {analysis['total_keys']}개 키")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ 체크포인트 분석 실패: {e}")
            return {}
    
    def map_checkpoint_keys(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """체크포인트 키 매핑"""
        try:
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # 키 매핑 규칙 적용
            mapped_dict = {}
            for key, value in state_dict.items():
                # module. 접두사 제거
                if key.startswith('module.'):
                    mapped_key = key[7:]  # 'module.' 제거
                else:
                    mapped_key = key
                
                mapped_dict[mapped_key] = value
            
            self.logger.info(f"✅ 체크포인트 키 매핑 완료: {len(mapped_dict)}개 키")
            return mapped_dict
            
        except Exception as e:
            self.logger.error(f"❌ 체크포인트 키 매핑 실패: {e}")
            return checkpoint
    
    def _analyze_key_patterns(self, state_dict: Dict[str, Any]) -> Dict[str, List[str]]:
        """키 패턴 분석"""
        patterns = {}
        for key in state_dict.keys():
            parts = key.split('.')
            if parts:
                prefix = parts[0]
                if prefix not in patterns:
                    patterns[prefix] = []
                patterns[prefix].append(key)
        return patterns
    
    def _analyze_layer_types(self, state_dict: Dict[str, Any]) -> Dict[str, int]:
        """레이어 타입 분석"""
        layer_types = {}
        for key in state_dict.keys():
            if 'conv' in key.lower():
                layer_types['conv'] = layer_types.get('conv', 0) + 1
            elif 'bn' in key.lower() or 'batch' in key.lower():
                layer_types['batch_norm'] = layer_types.get('batch_norm', 0) + 1
            elif 'linear' in key.lower() or 'fc' in key.lower():
                layer_types['linear'] = layer_types.get('linear', 0) + 1
            elif 'attention' in key.lower():
                layer_types['attention'] = layer_types.get('attention', 0) + 1
            else:
                layer_types['other'] = layer_types.get('other', 0) + 1
        return layer_types
    
    def _infer_architecture(self, state_dict: Dict[str, Any]) -> List[str]:
        """아키텍처 추론"""
        hints = []
        keys = list(state_dict.keys())
        
        # U2Net 패턴
        if any('u2net' in key.lower() for key in keys) or any('rsu' in key.lower() for key in keys):
            hints.append('U2Net')
        
        # SAM 패턴
        if any('sam' in key.lower() for key in keys) or any('vit' in key.lower() for key in keys):
            hints.append('SAM')
        
        # DeepLabV3+ 패턴
        if any('deeplab' in key.lower() for key in keys) or any('aspp' in key.lower() for key in keys):
            hints.append('DeepLabV3+')
        
        return hints
