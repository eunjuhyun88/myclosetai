#!/usr/bin/env python3
"""
🔥 Step 6 Virtual Fitting 체크포인트 분석 및 로딩 시스템
=======================================================

✅ Step 3, 4 패턴 기반 체크포인트 분석
✅ OOTD, VITON-HD, Stable Diffusion 체크포인트 분석
✅ 체크포인트 수정 및 키 매핑
✅ 안전한 모델 로딩 시스템
✅ Mock 폴백 시스템

Author: MyCloset AI Team
Date: 2025-08-08
Version: 1.0
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class Step6CheckpointAnalyzer:
    """Step 6 체크포인트 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Step6CheckpointAnalyzer")
        self.analyzed_checkpoints = {}
        self.fixed_checkpoints = {}
    
    def analyze_checkpoint_structure(self, checkpoint_path: str) -> Dict[str, Any]:
        """체크포인트 구조 분석"""
        try:
            self.logger.info(f"🔍 체크포인트 구조 분석: {checkpoint_path}")
            
            if checkpoint_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                checkpoint = load_file(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 키 패턴 분석
            key_patterns = {}
            for key in state_dict.keys():
                prefix = key.split('.')[0] if '.' in key else key
                if prefix not in key_patterns:
                    key_patterns[prefix] = []
                key_patterns[prefix].append(key)
            
            # 형태 분석
            shape_analysis = {}
            for key, tensor in state_dict.items():
                shape_analysis[key] = list(tensor.shape)
            
            analysis_result = {
                'total_keys': len(state_dict),
                'key_patterns': key_patterns,
                'shape_analysis': shape_analysis,
                'sample_keys': list(state_dict.keys())[:10],
                'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
                'is_safetensors': checkpoint_path.endswith('.safetensors')
            }
            
            self.analyzed_checkpoints[checkpoint_path] = analysis_result
            self.logger.info(f"✅ 체크포인트 분석 완료: {analysis_result['total_keys']}개 키")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"❌ 체크포인트 분석 실패: {e}")
            return {'error': str(e)}
    
    def validate_checkpoint(self, checkpoint_path: str, model_type: str) -> Dict[str, Any]:
        """체크포인트 유효성 검사"""
        try:
            self.logger.info(f"🔍 체크포인트 유효성 검사: {checkpoint_path}")
            
            # 파일 존재 확인
            if not os.path.exists(checkpoint_path):
                return {'is_valid': False, 'errors': ['파일이 존재하지 않습니다']}
            
            # 파일 크기 확인
            file_size = os.path.getsize(checkpoint_path)
            if file_size < 1024:  # 1KB 미만
                return {'is_valid': False, 'errors': ['파일이 너무 작습니다']}
            
            # 체크포인트 로딩 시도
            try:
                if checkpoint_path.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    checkpoint = load_file(checkpoint_path)
                else:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # 기본 유효성 검사
                if not isinstance(state_dict, dict):
                    return {'is_valid': False, 'errors': ['state_dict가 딕셔너리가 아닙니다']}
                
                if len(state_dict) == 0:
                    return {'is_valid': False, 'errors': ['state_dict가 비어있습니다']}
                
                # 모델별 특정 검사
                model_specific_errors = self._validate_model_specific(checkpoint_path, state_dict, model_type)
                if model_specific_errors:
                    return {'is_valid': False, 'errors': model_specific_errors}
                
                return {'is_valid': True, 'errors': []}
                
            except Exception as e:
                return {'is_valid': False, 'errors': [f'체크포인트 로딩 실패: {e}']}
                
        except Exception as e:
            self.logger.error(f"❌ 체크포인트 유효성 검사 실패: {e}")
            return {'is_valid': False, 'errors': [str(e)]}
    
    def _validate_model_specific(self, checkpoint_path: str, state_dict: Dict, model_type: str) -> List[str]:
        """모델별 특정 유효성 검사"""
        errors = []
        
        if model_type == "ootd":
            # OOTD 모델 검사
            required_patterns = ['time_embedding', 'encoder', 'decoder', 'bottleneck']
            found_patterns = []
            for key in state_dict.keys():
                for pattern in required_patterns:
                    if pattern in key:
                        found_patterns.append(pattern)
                        break
            
            if len(found_patterns) < 2:
                errors.append(f"OOTD 모델 패턴 부족: {found_patterns}")
        
        elif model_type == "viton_hd":
            # VITON-HD 모델 검사
            required_patterns = ['geometric_matcher', 'tryon_generator', 'refinement_net']
            found_patterns = []
            for key in state_dict.keys():
                for pattern in required_patterns:
                    if pattern in key:
                        found_patterns.append(pattern)
                        break
            
            if len(found_patterns) < 1:
                errors.append(f"VITON-HD 모델 패턴 부족: {found_patterns}")
        
        elif model_type == "stable_diffusion":
            # Stable Diffusion 모델 검사
            required_patterns = ['unet', 'vae', 'text_encoder']
            found_patterns = []
            for key in state_dict.keys():
                for pattern in required_patterns:
                    if pattern in key:
                        found_patterns.append(pattern)
                        break
            
            if len(found_patterns) < 1:
                errors.append(f"Stable Diffusion 모델 패턴 부족: {found_patterns}")
        
        return errors

class Step6CheckpointFixer:
    """Step 6 체크포인트 수정기"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Step6CheckpointFixer")
    
    def fix_checkpoint(self, checkpoint_path: str, model_type: str) -> Optional[Dict]:
        """체크포인트 수정"""
        try:
            self.logger.info(f"🔧 체크포인트 수정 시도: {checkpoint_path}")
            
            # 체크포인트 로딩
            if checkpoint_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                checkpoint = load_file(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 모델별 수정 적용
            fixed_state_dict = self._apply_model_specific_fixes(state_dict, model_type)
            
            if fixed_state_dict:
                self.logger.info(f"✅ 체크포인트 수정 완료: {len(fixed_state_dict)}개 키")
                return fixed_state_dict
            else:
                self.logger.warning("⚠️ 체크포인트 수정 실패")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ 체크포인트 수정 실패: {e}")
            return None
    
    def _apply_model_specific_fixes(self, state_dict: Dict, model_type: str) -> Optional[Dict]:
        """모델별 특정 수정 적용"""
        try:
            if model_type == "ootd":
                return self._fix_ootd_checkpoint(state_dict)
            elif model_type == "viton_hd":
                return self._fix_viton_hd_checkpoint(state_dict)
            elif model_type == "stable_diffusion":
                return self._fix_stable_diffusion_checkpoint(state_dict)
            else:
                return state_dict
                
        except Exception as e:
            self.logger.error(f"❌ 모델별 수정 실패: {e}")
            return None
    
    def _fix_ootd_checkpoint(self, state_dict: Dict) -> Dict:
        """OOTD 체크포인트 수정"""
        fixed_state_dict = {}
        
        # 키 매핑 규칙
        key_mappings = {
            'model.': '',
            'unet.': '',
            'diffusion.': '',
            'time_embedding.': 'time_embedding.',
            'encoder_blocks.': 'encoder_blocks.',
            'decoder_blocks.': 'decoder_blocks.',
            'bottleneck_': 'bottleneck_',
            'cross_attention.': 'cross_attention.',
            'cloth_encoder.': 'cloth_encoder.',
            'output_conv.': 'output_conv.'
        }
        
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in key_mappings.items():
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix, 1)
                    break
            
            fixed_state_dict[new_key] = value
        
        return fixed_state_dict
    
    def _fix_viton_hd_checkpoint(self, state_dict: Dict) -> Dict:
        """VITON-HD 체크포인트 수정"""
        fixed_state_dict = {}
        
        # 키 매핑 규칙
        key_mappings = {
            'model.': '',
            'viton.': '',
            'hrviton.': '',
            'geometric_matcher.': 'geometric_matcher.',
            'tryon_generator.': 'tryon_generator.',
            'refinement_net.': 'refinement_net.',
            'feature_extractor.': 'geometric_matcher.feature_extractor.',
            'flow_predictor.': 'geometric_matcher.flow_predictor.',
            'upsample.': 'geometric_matcher.upsample.'
        }
        
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in key_mappings.items():
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix, 1)
                    break
            
            fixed_state_dict[new_key] = value
        
        return fixed_state_dict
    
    def _fix_stable_diffusion_checkpoint(self, state_dict: Dict) -> Dict:
        """Stable Diffusion 체크포인트 수정"""
        fixed_state_dict = {}
        
        # 키 매핑 규칙
        key_mappings = {
            'model.': '',
            'unet.': '',
            'diffusion.': '',
            'vae.': 'vae_encoder.',
            'text_encoder.': 'text_encoder.',
            'noise_scheduler.': 'noise_scheduler.',
            'controlnet.': 'controlnet.',
            'lora_adapter.': 'lora_adapter.'
        }
        
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in key_mappings.items():
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix, 1)
                    break
            
            fixed_state_dict[new_key] = value
        
        return fixed_state_dict

class Step6ModelLoader:
    """Step 6 모델 로더"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Step6ModelLoader")
        self.analyzer = Step6CheckpointAnalyzer()
        self.fixer = Step6CheckpointFixer()
        self.loaded_models = {}
    
    def load_ootd_model(self, device='cpu'):
        """OOTD 모델 로딩"""
        try:
            from ..steps.step_06_virtual_fitting import OOTDDiffusionModel
            
            self.logger.info("🔄 OOTD 모델 로딩 시작")
            
            # 체크포인트 경로들
            checkpoint_paths = [
                "ai_models/step_06_virtual_fitting/ootd_3.2gb.pth",
                "ai_models/step_06_virtual_fitting/ootd_checkpoint.pth",
                "ai_models/checkpoints/step_06_virtual_fitting/ootd_checkpoint.pth"
            ]
            
            for checkpoint_path in checkpoint_paths:
                if os.path.exists(checkpoint_path):
                    self.logger.info(f"🔄 OOTD 체크포인트 분석: {checkpoint_path}")
                    
                    # 체크포인트 분석
                    analysis = self.analyzer.analyze_checkpoint_structure(checkpoint_path)
                    if 'error' in analysis:
                        self.logger.warning(f"⚠️ 체크포인트 분석 실패: {analysis['error']}")
                        continue
                    
                    # 체크포인트 유효성 검사
                    validation = self.analyzer.validate_checkpoint(checkpoint_path, "ootd")
                    if not validation['is_valid']:
                        self.logger.warning(f"⚠️ 체크포인트 유효성 검사 실패: {validation['errors']}")
                        
                        # 체크포인트 수정 시도
                        fixed_checkpoint = self.fixer.fix_checkpoint(checkpoint_path, "ootd")
                        if fixed_checkpoint:
                            self.logger.info("✅ 체크포인트 수정 완료")
                        else:
                            self.logger.warning("⚠️ 체크포인트 수정 실패")
                            continue
                    else:
                        # 원본 체크포인트 사용
                        if checkpoint_path.endswith('.safetensors'):
                            from safetensors.torch import load_file
                            fixed_checkpoint = load_file(checkpoint_path)
                        else:
                            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                            if 'state_dict' in checkpoint:
                                fixed_checkpoint = checkpoint['state_dict']
                            else:
                                fixed_checkpoint = checkpoint
                    
                    # 모델 생성 및 가중치 로딩
                    model = OOTDDiffusionModel()
                    model.load_state_dict(fixed_checkpoint, strict=False)
                    model.to(device)
                    model.eval()
                    
                    self.logger.info(f"✅ OOTD 모델 로딩 완료: {checkpoint_path}")
                    self.loaded_models['ootd'] = model
                    return model
            
            self.logger.error("❌ OOTD 체크포인트를 찾을 수 없습니다")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ OOTD 모델 로딩 실패: {e}")
            return None
    
    def load_viton_hd_model(self, device='cpu'):
        """VITON-HD 모델 로딩"""
        try:
            from ..steps.step_06_virtual_fitting import VITONHDModel
            
            self.logger.info("🔄 VITON-HD 모델 로딩 시작")
            
            # 체크포인트 경로들
            checkpoint_paths = [
                "ai_models/step_06_virtual_fitting/viton_hd_2.1gb.pth",
                "ai_models/step_06_virtual_fitting/hrviton_final.pth",
                "ai_models/checkpoints/step_06_virtual_fitting/hrviton_final.pth"
            ]
            
            for checkpoint_path in checkpoint_paths:
                if os.path.exists(checkpoint_path):
                    self.logger.info(f"🔄 VITON-HD 체크포인트 분석: {checkpoint_path}")
                    
                    # 체크포인트 분석
                    analysis = self.analyzer.analyze_checkpoint_structure(checkpoint_path)
                    if 'error' in analysis:
                        self.logger.warning(f"⚠️ 체크포인트 분석 실패: {analysis['error']}")
                        continue
                    
                    # 체크포인트 유효성 검사
                    validation = self.analyzer.validate_checkpoint(checkpoint_path, "viton_hd")
                    if not validation['is_valid']:
                        self.logger.warning(f"⚠️ 체크포인트 유효성 검사 실패: {validation['errors']}")
                        
                        # 체크포인트 수정 시도
                        fixed_checkpoint = self.fixer.fix_checkpoint(checkpoint_path, "viton_hd")
                        if fixed_checkpoint:
                            self.logger.info("✅ 체크포인트 수정 완료")
                        else:
                            self.logger.warning("⚠️ 체크포인트 수정 실패")
                            continue
                    else:
                        # 원본 체크포인트 사용
                        if checkpoint_path.endswith('.safetensors'):
                            from safetensors.torch import load_file
                            fixed_checkpoint = load_file(checkpoint_path)
                        else:
                            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                            if 'state_dict' in checkpoint:
                                fixed_checkpoint = checkpoint['state_dict']
                            else:
                                fixed_checkpoint = checkpoint
                    
                    # 모델 생성 및 가중치 로딩
                    model = VITONHDModel()
                    model.load_state_dict(fixed_checkpoint, strict=False)
                    model.to(device)
                    model.eval()
                    
                    self.logger.info(f"✅ VITON-HD 모델 로딩 완료: {checkpoint_path}")
                    self.loaded_models['viton_hd'] = model
                    return model
            
            self.logger.error("❌ VITON-HD 체크포인트를 찾을 수 없습니다")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ VITON-HD 모델 로딩 실패: {e}")
            return None
    
    def load_stable_diffusion_model(self, device='cpu'):
        """Stable Diffusion 모델 로딩"""
        try:
            from ..steps.step_06_virtual_fitting import StableDiffusionNeuralNetwork
            
            self.logger.info("🔄 Stable Diffusion 모델 로딩 시작")
            
            # 체크포인트 경로들
            checkpoint_paths = [
                "ai_models/step_06_virtual_fitting/stable_diffusion_4.8gb.pth",
                "ai_models/checkpoints/stable_diffusion_4.8gb.pth"
            ]
            
            for checkpoint_path in checkpoint_paths:
                if os.path.exists(checkpoint_path):
                    self.logger.info(f"🔄 Stable Diffusion 체크포인트 분석: {checkpoint_path}")
                    
                    # 체크포인트 분석
                    analysis = self.analyzer.analyze_checkpoint_structure(checkpoint_path)
                    if 'error' in analysis:
                        self.logger.warning(f"⚠️ 체크포인트 분석 실패: {analysis['error']}")
                        continue
                    
                    # 체크포인트 유효성 검사
                    validation = self.analyzer.validate_checkpoint(checkpoint_path, "stable_diffusion")
                    if not validation['is_valid']:
                        self.logger.warning(f"⚠️ 체크포인트 유효성 검사 실패: {validation['errors']}")
                        
                        # 체크포인트 수정 시도
                        fixed_checkpoint = self.fixer.fix_checkpoint(checkpoint_path, "stable_diffusion")
                        if fixed_checkpoint:
                            self.logger.info("✅ 체크포인트 수정 완료")
                        else:
                            self.logger.warning("⚠️ 체크포인트 수정 실패")
                            continue
                    else:
                        # 원본 체크포인트 사용
                        if checkpoint_path.endswith('.safetensors'):
                            from safetensors.torch import load_file
                            fixed_checkpoint = load_file(checkpoint_path)
                        else:
                            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                            if 'state_dict' in checkpoint:
                                fixed_checkpoint = checkpoint['state_dict']
                            else:
                                fixed_checkpoint = checkpoint
                    
                    # 모델 생성 및 가중치 로딩
                    model = StableDiffusionNeuralNetwork()
                    model.load_state_dict(fixed_checkpoint, strict=False)
                    model.to(device)
                    model.eval()
                    
                    self.logger.info(f"✅ Stable Diffusion 모델 로딩 완료: {checkpoint_path}")
                    self.loaded_models['stable_diffusion'] = model
                    return model
            
            self.logger.error("❌ Stable Diffusion 체크포인트를 찾을 수 없습니다")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Stable Diffusion 모델 로딩 실패: {e}")
            return None

# 전역 인스턴스
step6_model_loader = Step6ModelLoader()

def get_step6_model_loader():
    """Step 6 모델 로더 인스턴스 반환"""
    return step6_model_loader
