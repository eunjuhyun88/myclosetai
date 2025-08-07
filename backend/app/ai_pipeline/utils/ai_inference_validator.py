#!/usr/bin/env python3
"""
🔥 AI 추론 검증 시스템
=====================

실제 AI 추론에 필요한 모든 요소들을 검증하고 상태를 확인하는 시스템

Author: MyCloset AI Team
Date: 2025-08-07
Version: 1.0
"""

import os
import sys
import time
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

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

# OpenCV 관련
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class AIInferenceRequirements:
    """AI 추론에 필요한 요구사항들"""
    
    # 🔥 1. 모델 파일 요구사항
    model_files: Dict[str, bool] = field(default_factory=dict)
    model_sizes: Dict[str, float] = field(default_factory=dict)  # MB 단위
    model_formats: Dict[str, str] = field(default_factory=dict)  # .pth, .safetensors 등
    
    # 🔥 2. 메모리 요구사항
    gpu_memory_required: float = 0.0  # GB
    system_memory_required: float = 0.0  # GB
    model_memory_usage: Dict[str, float] = field(default_factory=dict)  # MB 단위
    
    # 🔥 3. 디바이스 요구사항
    device_available: bool = False
    device_type: str = "unknown"  # cpu, cuda, mps
    device_memory: float = 0.0  # GB
    
    # 🔥 4. 라이브러리 요구사항
    required_libraries: Dict[str, bool] = field(default_factory=dict)
    library_versions: Dict[str, str] = field(default_factory=dict)
    
    # 🔥 5. 체크포인트 요구사항
    checkpoint_loaded: bool = False
    checkpoint_keys: List[str] = field(default_factory=list)
    checkpoint_size: float = 0.0  # MB
    
    # 🔥 6. 입력 데이터 요구사항
    input_shape: Tuple[int, ...] = field(default_factory=tuple)
    input_dtype: str = "unknown"
    input_normalization: bool = False
    
    # 🔥 7. 출력 데이터 요구사항
    output_shape: Tuple[int, ...] = field(default_factory=tuple)
    output_dtype: str = "unknown"
    output_postprocessing: bool = False

class AIInferenceValidator:
    """AI 추론 검증 시스템"""
    
    def __init__(self):
        self.requirements = AIInferenceRequirements()
        self.validation_results = {}
        
    def validate_step_01_human_parsing(self) -> Dict[str, Any]:
        """Step 1 (Human Parsing) 추론 요구사항 검증"""
        result = {
            'step_name': 'Human Parsing',
            'validation_time': datetime.now().isoformat(),
            'requirements_met': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 🔥 1. 모델 파일 검증 (실제 경로)
            model_files = {
                'graphonomy': 'ai_models/Self-Correction-Human-Parsing/exp-schp-201908261155-atr.pth',
                'u2net': 'ai_models/step_03_cloth_segmentation/u2net.pth',
                'deeplabv3plus': 'ai_models/step_01_human_parsing/deeplabv3plus.pth'
            }
            
            for model_name, file_path in model_files.items():
                if Path(file_path).exists():
                    size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                    self.requirements.model_files[model_name] = True
                    self.requirements.model_sizes[model_name] = size_mb
                    result[f'{model_name}_loaded'] = True
                    result[f'{model_name}_size_mb'] = size_mb
                else:
                    self.requirements.model_files[model_name] = False
                    result[f'{model_name}_loaded'] = False
                    result['issues'].append(f"{model_name} 모델 파일 없음: {file_path}")
            
            # 🔥 2. 메모리 요구사항 계산
            total_model_size = sum(self.requirements.model_sizes.values())
            self.requirements.gpu_memory_required = total_model_size / 1024 * 2  # 2배 버퍼
            self.requirements.system_memory_required = total_model_size / 1024 * 3  # 3배 버퍼
            
            result['total_model_size_mb'] = total_model_size
            result['gpu_memory_required_gb'] = self.requirements.gpu_memory_required
            result['system_memory_required_gb'] = self.requirements.system_memory_required
            
            # 🔥 3. 디바이스 검증
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    self.requirements.device_available = True
                    self.requirements.device_type = "cuda"
                    self.requirements.device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.requirements.device_available = True
                    self.requirements.device_type = "mps"
                    # MPS 메모리는 시스템 메모리와 공유
                    self.requirements.device_memory = 0.0
                else:
                    self.requirements.device_available = True
                    self.requirements.device_type = "cpu"
                    self.requirements.device_memory = 0.0
                
                result['device_available'] = self.requirements.device_available
                result['device_type'] = self.requirements.device_type
                result['device_memory_gb'] = self.requirements.device_memory
            
            # 🔥 4. 라이브러리 검증
            required_libs = {
                'torch': TORCH_AVAILABLE,
                'numpy': NUMPY_AVAILABLE,
                'opencv': CV2_AVAILABLE
            }
            
            for lib_name, available in required_libs.items():
                self.requirements.required_libraries[lib_name] = available
                if available:
                    result[f'{lib_name}_available'] = True
                else:
                    result[f'{lib_name}_available'] = False
                    result['issues'].append(f"{lib_name} 라이브러리 없음")
            
            # 🔥 5. 체크포인트 검증
            if self.requirements.model_files.get('graphonomy', False):
                try:
                    checkpoint_path = model_files['graphonomy']
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    
                    if isinstance(checkpoint, dict):
                        self.requirements.checkpoint_loaded = True
                        self.requirements.checkpoint_keys = list(checkpoint.keys())
                        self.requirements.checkpoint_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)
                        
                        result['checkpoint_loaded'] = True
                        result['checkpoint_keys_count'] = len(self.requirements.checkpoint_keys)
                        result['checkpoint_size_mb'] = self.requirements.checkpoint_size
                        
                        # state_dict 확인
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                            result['state_dict_keys_count'] = len(state_dict)
                            result['state_dict_sample_keys'] = list(state_dict.keys())[:5]
                    else:
                        result['issues'].append("체크포인트가 딕셔너리 형태가 아님")
                        
                except Exception as e:
                    result['issues'].append(f"체크포인트 로딩 실패: {e}")
            
            # 🔥 6. 입력 데이터 요구사항
            self.requirements.input_shape = (3, 512, 512)  # RGB, 512x512
            self.requirements.input_dtype = "float32"
            self.requirements.input_normalization = True
            
            result['input_shape'] = self.requirements.input_shape
            result['input_dtype'] = self.requirements.input_dtype
            result['input_normalization'] = self.requirements.input_normalization
            
            # 🔥 7. 출력 데이터 요구사항
            self.requirements.output_shape = (20, 512, 512)  # 20개 클래스, 512x512
            self.requirements.output_dtype = "float32"
            self.requirements.output_postprocessing = True
            
            result['output_shape'] = self.requirements.output_shape
            result['output_dtype'] = self.requirements.output_dtype
            result['output_postprocessing'] = self.requirements.output_postprocessing
            
            # 🔥 8. 최종 검증 결과
            all_requirements_met = (
                all(self.requirements.model_files.values()) and
                self.requirements.device_available and
                all(self.requirements.required_libraries.values()) and
                self.requirements.checkpoint_loaded
            )
            
            result['requirements_met'] = all_requirements_met
            
            # 🔥 9. 권장사항 생성
            if not all_requirements_met:
                if not all(self.requirements.model_files.values()):
                    result['recommendations'].append("모델 파일들을 다운로드하세요")
                
                if not self.requirements.device_available:
                    result['recommendations'].append("GPU 또는 MPS 디바이스를 확인하세요")
                
                if not all(self.requirements.required_libraries.values()):
                    result['recommendations'].append("필요한 라이브러리들을 설치하세요")
                
                if not self.requirements.checkpoint_loaded:
                    result['recommendations'].append("체크포인트 파일을 확인하세요")
            
            # 메모리 권장사항
            if self.requirements.gpu_memory_required > self.requirements.device_memory:
                result['recommendations'].append(f"GPU 메모리 부족: {self.requirements.gpu_memory_required:.1f}GB 필요, {self.requirements.device_memory:.1f}GB 사용 가능")
            
        except Exception as e:
            result['issues'].append(f"검증 중 오류 발생: {e}")
            result['traceback'] = traceback.format_exc()
        
        self.validation_results['step_01'] = result
        return result
    
    def validate_step_02_pose_estimation(self) -> Dict[str, Any]:
        """Step 2 (Pose Estimation) 추론 요구사항 검증"""
        result = {
            'step_name': 'Pose Estimation',
            'validation_time': datetime.now().isoformat(),
            'requirements_met': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 🔥 1. 모델 파일 검증 (실제 경로)
            model_files = {
                'body_pose_model': 'ai_models/step_06_virtual_fitting/ootdiffusion/checkpoints/openpose/ckpts/body_pose_model.pth',
                'yolov8m_pose': 'ai_models/step_02_pose_estimation/yolov8m-pose.pt',
                'openpose': 'ai_models/openpose.pth',
                'hrnet_w48': 'ai_models/step_02_pose_estimation/hrnet_w48_coco_384x288.pth'
            }
            
            for model_name, file_path in model_files.items():
                if Path(file_path).exists():
                    size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                    self.requirements.model_files[model_name] = True
                    self.requirements.model_sizes[model_name] = size_mb
                    result[f'{model_name}_loaded'] = True
                    result[f'{model_name}_size_mb'] = size_mb
                else:
                    self.requirements.model_files[model_name] = False
                    result[f'{model_name}_loaded'] = False
                    result['issues'].append(f"{model_name} 모델 파일 없음: {file_path}")
            
            # 🔥 2. 메모리 요구사항 계산
            total_model_size = sum(self.requirements.model_sizes.values())
            self.requirements.gpu_memory_required = total_model_size / 1024 * 1.5  # 1.5배 버퍼
            self.requirements.system_memory_required = total_model_size / 1024 * 2  # 2배 버퍼
            
            result['total_model_size_mb'] = total_model_size
            result['gpu_memory_required_gb'] = self.requirements.gpu_memory_required
            result['system_memory_required_gb'] = self.requirements.system_memory_required
            
            # 🔥 3. 디바이스 검증
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    self.requirements.device_available = True
                    self.requirements.device_type = "cuda"
                    self.requirements.device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.requirements.device_available = True
                    self.requirements.device_type = "mps"
                    self.requirements.device_memory = 0.0
                else:
                    self.requirements.device_available = True
                    self.requirements.device_type = "cpu"
                    self.requirements.device_memory = 0.0
                
                result['device_available'] = self.requirements.device_available
                result['device_type'] = self.requirements.device_type
                result['device_memory_gb'] = self.requirements.device_memory
            
            # 🔥 4. 라이브러리 검증
            required_libs = {
                'torch': TORCH_AVAILABLE,
                'numpy': NUMPY_AVAILABLE,
                'opencv': CV2_AVAILABLE,
                'mediapipe': self._check_library_availability('mediapipe')
            }
            
            for lib_name, available in required_libs.items():
                self.requirements.required_libraries[lib_name] = available
                if available:
                    result[f'{lib_name}_available'] = True
                else:
                    result[f'{lib_name}_available'] = False
                    result['issues'].append(f"{lib_name} 라이브러리 없음")
            
            # 🔥 5. 종합 판정
            all_models_exist = all(self.requirements.model_files.values())
            all_libs_available = all(self.requirements.required_libraries.values())
            
            result['requirements_met'] = all([all_models_exist, all_libs_available, self.requirements.device_available])
            
            if result['requirements_met']:
                result['recommendations'].append("🎉 Step 2 모든 요구사항 충족!")
            else:
                result['issues'].append("❌ Step 2 요구사항 미충족")
                
        except Exception as e:
            result['issues'].append(f"❌ 검증 중 오류: {e}")
        
        return result
    
    def validate_step_03_cloth_segmentation(self) -> Dict[str, Any]:
        """Step 3 (Cloth Segmentation) 추론 요구사항 검증"""
        result = {
            'step_name': 'Cloth Segmentation',
            'validation_time': datetime.now().isoformat(),
            'requirements_met': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 🔥 1. 모델 파일 검증 (실제 경로)
            model_files = {
                'sam_vit_h': 'ai_models/step_03_cloth_segmentation/sam_vit_h_4b8939.pth',
                'u2net': 'ai_models/step_03_cloth_segmentation/u2net.pth',
                'deeplabv3_resnet101': 'ai_models/step_03_cloth_segmentation/deeplabv3_resnet101_ultra.pth',
                'mobile_sam': 'ai_models/step_03_cloth_segmentation/mobile_sam_alternative.pt'
            }
            
            for model_name, file_path in model_files.items():
                if Path(file_path).exists():
                    size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                    self.requirements.model_files[model_name] = True
                    self.requirements.model_sizes[model_name] = size_mb
                    result[f'{model_name}_loaded'] = True
                    result[f'{model_name}_size_mb'] = size_mb
                else:
                    self.requirements.model_files[model_name] = False
                    result[f'{model_name}_loaded'] = False
                    result['issues'].append(f"{model_name} 모델 파일 없음: {file_path}")
            
            # 🔥 2. 메모리 요구사항 계산 (SAM 모델이 큼)
            total_model_size = sum(self.requirements.model_sizes.values())
            self.requirements.gpu_memory_required = total_model_size / 1024 * 3  # 3배 버퍼
            self.requirements.system_memory_required = total_model_size / 1024 * 4  # 4배 버퍼
            
            result['total_model_size_mb'] = total_model_size
            result['gpu_memory_required_gb'] = self.requirements.gpu_memory_required
            result['system_memory_required_gb'] = self.requirements.system_memory_required
            
            # 🔥 3. 디바이스 검증
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    self.requirements.device_available = True
                    self.requirements.device_type = "cuda"
                    self.requirements.device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.requirements.device_available = True
                    self.requirements.device_type = "mps"
                    self.requirements.device_memory = 0.0
                else:
                    self.requirements.device_available = True
                    self.requirements.device_type = "cpu"
                    self.requirements.device_memory = 0.0
                
                result['device_available'] = self.requirements.device_available
                result['device_type'] = self.requirements.device_type
                result['device_memory_gb'] = self.requirements.device_memory
            
            # 🔥 4. 라이브러리 검증
            required_libs = {
                'torch': TORCH_AVAILABLE,
                'numpy': NUMPY_AVAILABLE,
                'opencv': CV2_AVAILABLE,
                'PIL': PIL_AVAILABLE,
                'segment_anything': self._check_library_availability('segment_anything')
            }
            
            for lib_name, available in required_libs.items():
                self.requirements.required_libraries[lib_name] = available
                if available:
                    result[f'{lib_name}_available'] = True
                else:
                    result[f'{lib_name}_available'] = False
                    result['issues'].append(f"{lib_name} 라이브러리 없음")
            
            # 🔥 5. 종합 판정
            all_models_exist = all(self.requirements.model_files.values())
            all_libs_available = all(self.requirements.required_libraries.values())
            
            result['requirements_met'] = all([all_models_exist, all_libs_available, self.requirements.device_available])
            
            if result['requirements_met']:
                result['recommendations'].append("🎉 Step 3 모든 요구사항 충족!")
            else:
                result['issues'].append("❌ Step 3 요구사항 미충족")
                
        except Exception as e:
            result['issues'].append(f"❌ 검증 중 오류: {e}")
        
        return result
    
    def validate_step_04_geometric_matching(self) -> Dict[str, Any]:
        """Step 4 (Geometric Matching) 추론 요구사항 검증"""
        result = {
            'step_name': 'Geometric Matching',
            'validation_time': datetime.now().isoformat(),
            'requirements_met': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 🔥 1. 모델 파일 검증 (실제 경로)
            model_files = {
                'gmm_final': 'ai_models/step_04_geometric_matching/gmm_final.pth',
                'tps_network': 'ai_models/step_04_geometric_matching/tps_network.pth',
                'raft_things': 'ai_models/step_04_geometric_matching/raft-things.pth',
                'sam_vit_h': 'ai_models/step_04_geometric_matching/sam_vit_h_4b8939.pth'
            }
            
            for model_name, file_path in model_files.items():
                if Path(file_path).exists():
                    size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                    self.requirements.model_files[model_name] = True
                    self.requirements.model_sizes[model_name] = size_mb
                    result[f'{model_name}_loaded'] = True
                    result[f'{model_name}_size_mb'] = size_mb
                else:
                    self.requirements.model_files[model_name] = False
                    result[f'{model_name}_loaded'] = False
                    result['issues'].append(f"{model_name} 모델 파일 없음: {file_path}")
            
            # 🔥 2. MPS 타입 호환성 검증
            if TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                result['mps_available'] = True
                result['mps_type_compatibility'] = True
                result['recommendations'].append("MPS 디바이스에서 torch.float32 타입 통일 필요")
            else:
                result['mps_available'] = False
            
            # 🔥 3. 메모리 요구사항
            total_model_size = sum(self.requirements.model_sizes.values())
            result['total_model_size_mb'] = total_model_size
            result['gpu_memory_required_gb'] = total_model_size / 1024 * 2
            
            # 🔥 4. 최종 검증 결과
            all_requirements_met = all(self.requirements.model_files.values())
            result['requirements_met'] = all_requirements_met
            
            # 🔥 5. 권장사항 생성
            if not all_requirements_met:
                if not all(self.requirements.model_files.values()):
                    result['recommendations'].append("모델 파일들을 다운로드하세요")
            else:
                result['recommendations'].append("모든 모델 파일이 준비되었습니다!")
            
        except Exception as e:
            result['issues'].append(f"검증 중 오류 발생: {e}")
            result['traceback'] = traceback.format_exc()
        
        self.validation_results['step_04'] = result
        return result
    
    def validate_step_05_cloth_warping(self) -> Dict[str, Any]:
        """Step 5 (Cloth Warping) 추론 요구사항 검증"""
        result = {
            'step_name': 'Cloth Warping',
            'validation_time': datetime.now().isoformat(),
            'requirements_met': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 🔥 1. 모델 파일 검증 (실제 경로)
            model_files = {
                'tom_final': 'ai_models/step_05_cloth_warping/tom_final.pth',
                'viton_hd_warping': 'ai_models/step_05_cloth_warping/viton_hd_warping.pth',
                'tps_transformation': 'ai_models/step_05_cloth_warping/tps_transformation.pth',
                'dpt_hybrid_midas': 'ai_models/step_05_cloth_warping/dpt_hybrid_midas.pth',
                'vgg19_warping': 'ai_models/step_05_cloth_warping/vgg19_warping.pth',
                'u2net_warping': 'ai_models/step_05_cloth_warping/u2net_warping.pth'
            }
            
            for model_name, file_path in model_files.items():
                if Path(file_path).exists():
                    size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                    self.requirements.model_files[model_name] = True
                    self.requirements.model_sizes[model_name] = size_mb
                    result[f'{model_name}_loaded'] = True
                    result[f'{model_name}_size_mb'] = size_mb
                else:
                    self.requirements.model_files[model_name] = False
                    result[f'{model_name}_loaded'] = False
                    result['issues'].append(f"{model_name} 모델 파일 없음: {file_path}")
            
            # 🔥 2. 메모리 요구사항
            total_model_size = sum(self.requirements.model_sizes.values())
            result['total_model_size_mb'] = total_model_size
            result['gpu_memory_required_gb'] = total_model_size / 1024 * 2
            
            # 🔥 3. 최종 검증 결과
            all_requirements_met = all(self.requirements.model_files.values())
            result['requirements_met'] = all_requirements_met
            
            # 🔥 4. 권장사항 생성
            if not all_requirements_met:
                if not all(self.requirements.model_files.values()):
                    result['recommendations'].append("모델 파일들을 다운로드하세요")
            else:
                result['recommendations'].append("모든 모델 파일이 준비되었습니다!")
                
        except Exception as e:
            result['issues'].append(f"검증 중 오류 발생: {e}")
            result['traceback'] = traceback.format_exc()
        
        self.validation_results['step_05'] = result
        return result
    
    def validate_step_06_virtual_fitting(self) -> Dict[str, Any]:
        """Step 6 (Virtual Fitting) 추론 요구사항 검증"""
        result = {
            'step_name': 'Virtual Fitting',
            'validation_time': datetime.now().isoformat(),
            'requirements_met': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 🔥 1. 모델 파일 검증 (실제 경로)
            model_files = {
                'stable_diffusion': 'ai_models/step_06_virtual_fitting/stable_diffusion_4.8gb.pth',
                'viton_hd': 'ai_models/step_06_virtual_fitting/viton_hd_2.1gb.pth',
                'ootd': 'ai_models/step_06_virtual_fitting/ootd_3.2gb.pth',
                'hrviton': 'ai_models/step_06_virtual_fitting/hrviton_final.pth',
                'ootd_checkpoint': 'ai_models/step_06_virtual_fitting/validated_checkpoints/ootd_checkpoint.pth'
            }
            
            for model_name, file_path in model_files.items():
                if Path(file_path).exists():
                    size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                    self.requirements.model_files[model_name] = True
                    self.requirements.model_sizes[model_name] = size_mb
                    result[f'{model_name}_loaded'] = True
                    result[f'{model_name}_size_mb'] = size_mb
                else:
                    self.requirements.model_files[model_name] = False
                    result[f'{model_name}_loaded'] = False
                    result['issues'].append(f"{model_name} 모델 파일 없음: {file_path}")
            
            # 🔥 2. 메모리 요구사항 (가장 큰 단계)
            total_model_size = sum(self.requirements.model_sizes.values())
            result['total_model_size_mb'] = total_model_size
            result['gpu_memory_required_gb'] = total_model_size / 1024 * 3  # 3배 버퍼 (가장 큰 단계)
            
            # 🔥 3. 최종 검증 결과
            all_requirements_met = all(self.requirements.model_files.values())
            result['requirements_met'] = all_requirements_met
            
            # 🔥 4. 권장사항 생성
            if not all_requirements_met:
                if not all(self.requirements.model_files.values()):
                    result['recommendations'].append("모델 파일들을 다운로드하세요")
            else:
                result['recommendations'].append("모든 모델 파일이 준비되었습니다!")
                result['recommendations'].append("가장 큰 메모리 사용량 단계 - 메모리 모니터링 필요")
                
        except Exception as e:
            result['issues'].append(f"검증 중 오류 발생: {e}")
            result['traceback'] = traceback.format_exc()
        
        self.validation_results['step_06'] = result
        return result
    
    def validate_step_07_post_processing(self) -> Dict[str, Any]:
        """Step 7 (Post Processing) 추론 요구사항 검증"""
        result = {
            'step_name': 'Post Processing',
            'validation_time': datetime.now().isoformat(),
            'requirements_met': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 🔥 1. 모델 파일 검증 (실제 경로)
            model_files = {
                'densenet161_enhance': 'ai_models/step_07_post_processing/densenet161_enhance.pth',
                'mobilenet_v3_ultra': 'ai_models/step_07_post_processing/mobilenet_v3_ultra.pth',
                'GFPGAN': 'ai_models/step_07_post_processing/GFPGAN.pth',
                'resnet101_enhance_ultra': 'ai_models/step_07_post_processing/resnet101_enhance_ultra.pth',
                'RealESRGAN_x2plus': 'ai_models/step_07_post_processing/RealESRGAN_x2plus.pth',
                'ESRGAN_x8': 'ai_models/step_07_post_processing/ESRGAN_x8.pth',
                'swinir_real_sr': 'ai_models/step_07_post_processing/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth',
                'swinir_large': 'ai_models/step_07_post_processing/swinir_real_sr_x4_large.pth',
                'RealESRGAN_x4plus': 'ai_models/step_07_post_processing/RealESRGAN_x4plus.pth'
            }
            
            for model_name, file_path in model_files.items():
                if Path(file_path).exists():
                    size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                    self.requirements.model_files[model_name] = True
                    self.requirements.model_sizes[model_name] = size_mb
                    result[f'{model_name}_loaded'] = True
                    result[f'{model_name}_size_mb'] = size_mb
                else:
                    self.requirements.model_files[model_name] = False
                    result[f'{model_name}_loaded'] = False
                    result['issues'].append(f"{model_name} 모델 파일 없음: {file_path}")
            
            # 🔥 2. 메모리 요구사항
            total_model_size = sum(self.requirements.model_sizes.values())
            result['total_model_size_mb'] = total_model_size
            result['gpu_memory_required_gb'] = total_model_size / 1024 * 2
            
            # 🔥 3. 최종 검증 결과
            all_requirements_met = all(self.requirements.model_files.values())
            result['requirements_met'] = all_requirements_met
            
            # 🔥 4. 권장사항 생성
            if not all_requirements_met:
                if not all(self.requirements.model_files.values()):
                    result['recommendations'].append("모델 파일들을 다운로드하세요")
            else:
                result['recommendations'].append("모든 모델 파일이 준비되었습니다!")
                
        except Exception as e:
            result['issues'].append(f"검증 중 오류 발생: {e}")
            result['traceback'] = traceback.format_exc()
        
        self.validation_results['step_07'] = result
        return result
    
    def validate_step_08_quality_assessment(self) -> Dict[str, Any]:
        """Step 8 (Quality Assessment) 추론 요구사항 검증"""
        result = {
            'step_name': 'Quality Assessment',
            'validation_time': datetime.now().isoformat(),
            'requirements_met': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 🔥 1. 모델 파일 검증 (실제 경로)
            model_files = {
                'clip_vit_b32': 'ai_models/step_08_quality_assessment/clip_vit_b32.pth',
                'alex': 'ai_models/step_08_quality_assessment/alex.pth',
                'ViT_B_32': 'ai_models/step_08_quality_assessment/ViT-B-32.pt',
                'ViT_L_14': 'ai_models/step_08_quality_assessment/ViT-L-14.pt',
                'lpips_alex': 'ai_models/step_08_quality_assessment/lpips_alex.pth'
            }
            
            for model_name, file_path in model_files.items():
                if Path(file_path).exists():
                    size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                    self.requirements.model_files[model_name] = True
                    self.requirements.model_sizes[model_name] = size_mb
                    result[f'{model_name}_loaded'] = True
                    result[f'{model_name}_size_mb'] = size_mb
                else:
                    self.requirements.model_files[model_name] = False
                    result[f'{model_name}_loaded'] = False
                    result['issues'].append(f"{model_name} 모델 파일 없음: {file_path}")
            
            # 🔥 2. 메모리 요구사항
            total_model_size = sum(self.requirements.model_sizes.values())
            result['total_model_size_mb'] = total_model_size
            result['gpu_memory_required_gb'] = total_model_size / 1024 * 2
            
            # 🔥 3. 최종 검증 결과
            all_requirements_met = all(self.requirements.model_files.values())
            result['requirements_met'] = all_requirements_met
            
            # 🔥 4. 권장사항 생성
            if not all_requirements_met:
                if not all(self.requirements.model_files.values()):
                    result['recommendations'].append("모델 파일들을 다운로드하세요")
            else:
                result['recommendations'].append("모든 모델 파일이 준비되었습니다!")
                
        except Exception as e:
            result['issues'].append(f"검증 중 오류 발생: {e}")
            result['traceback'] = traceback.format_exc()
        
        self.validation_results['step_08'] = result
        return result
    
    def validate_checkpoint_content(self, checkpoint_path: str) -> Dict[str, Any]:
        """체크포인트 내용 검증 (다양한 구조 타입 지원)"""
        result = {
            'checkpoint_path': checkpoint_path,
            'exists': False,
            'valid': False,
            'size_mb': 0.0,
            'structure': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            if not Path(checkpoint_path).exists():
                result['issues'].append("체크포인트 파일이 존재하지 않음")
                return result
            
            result['exists'] = True
            result['size_mb'] = Path(checkpoint_path).stat().st_size / (1024 * 1024)
            
            # 🔥 다양한 로딩 방법 시도
            checkpoint = None
            loading_method = None
            
            # 방법 1: weights_only=True (안전한 방법)
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                loading_method = 'weights_only_true'
                result['recommendations'].append("안전한 weights_only=True로 로딩됨")
            except Exception as e1:
                # 방법 2: weights_only=False (전통적인 방법)
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    loading_method = 'weights_only_false'
                    result['recommendations'].append("weights_only=False로 로딩됨 (보안 주의)")
                except Exception as e2:
                    # 방법 3: TorchScript 모델
                    try:
                        checkpoint = torch.jit.load(checkpoint_path, map_location='cpu')
                        loading_method = 'torchscript'
                        result['recommendations'].append("TorchScript 모델로 로딩됨")
                    except Exception as e3:
                        # 방법 4: SafeTensors (별도 라이브러리 필요)
                        try:
                            from safetensors import safe_open
                            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                                checkpoint = {key: f.get_tensor(key) for key in f.keys()}
                            loading_method = 'safetensors'
                            result['recommendations'].append("SafeTensors로 로딩됨")
                        except Exception as e4:
                            result['issues'].append(f"모든 로딩 방법 실패: {e4}")
                            return result
            
            result['structure']['loading_method'] = loading_method
            
            # 🔥 구조 타입 분류 및 검증
            if isinstance(checkpoint, dict):
                result['structure']['type'] = 'dict'
                result['structure']['keys'] = list(checkpoint.keys())
                result['structure']['key_count'] = len(checkpoint.keys())
                
                # 다양한 구조 타입 처리
                if 'state_dict' in checkpoint:
                    # 표준 PyTorch 모델
                    result['structure']['subtype'] = 'state_dict'
                    state_dict = checkpoint['state_dict']
                    result['structure']['state_dict_keys'] = list(state_dict.keys())[:10]
                    result['structure']['state_dict_count'] = len(state_dict.keys())
                    
                    # 파라미터 수 계산
                    total_params = 0
                    for key, tensor in state_dict.items():
                        if hasattr(tensor, 'numel'):
                            total_params += tensor.numel()
                    result['structure']['total_parameters'] = total_params
                    
                    # 아키텍처 감지
                    architecture_hints = self._detect_architecture_from_keys(state_dict.keys())
                    result['structure']['architecture_hints'] = architecture_hints
                    
                    result['valid'] = True
                    result['recommendations'].append("표준 state_dict 구조")
                    
                elif 'model' in checkpoint:
                    # 모델 래퍼 구조
                    result['structure']['subtype'] = 'model_wrapper'
                    result['valid'] = True
                    result['recommendations'].append("모델 래퍼 구조")
                    
                elif 'weights' in checkpoint:
                    # 가중치만 있는 구조
                    result['structure']['subtype'] = 'weights_only'
                    result['valid'] = True
                    result['recommendations'].append("가중치 전용 구조")
                    
                elif 'parameters' in checkpoint:
                    # 파라미터만 있는 구조
                    result['structure']['subtype'] = 'parameters_only'
                    result['valid'] = True
                    result['recommendations'].append("파라미터 전용 구조")
                    
                else:
                    # 커스텀 딕셔너리 구조
                    result['structure']['subtype'] = 'custom_dict'
                    
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
                        result['structure']['total_parameters'] = total_params
                        result['structure']['param_keys'] = param_keys[:10]  # 처음 10개만
                        result['valid'] = True
                        result['recommendations'].append("커스텀 구조에서 파라미터 발견")
                        
                        # 아키텍처 힌트 감지
                        architecture_hints = self._detect_architecture_from_keys(param_keys)
                        result['structure']['architecture_hints'] = architecture_hints
                    else:
                        result['issues'].append("파라미터를 찾을 수 없음")
                        result['recommendations'].append("커스텀 구조 검증 필요")
                        
            elif isinstance(checkpoint, torch.Tensor):
                # 직접 텐서 형태
                result['structure']['type'] = 'tensor'
                result['structure']['shape'] = list(checkpoint.shape)
                result['structure']['dtype'] = str(checkpoint.dtype)
                result['structure']['total_parameters'] = checkpoint.numel()
                result['valid'] = True
                result['recommendations'].append("직접 텐서 형태")
                
            elif hasattr(checkpoint, 'state_dict'):
                # TorchScript 모델
                result['structure']['type'] = 'torchscript'
                try:
                    state_dict = checkpoint.state_dict()
                    result['structure']['state_dict_keys'] = list(state_dict.keys())[:10]
                    result['structure']['state_dict_count'] = len(state_dict.keys())
                    
                    total_params = 0
                    for key, tensor in state_dict.items():
                        if hasattr(tensor, 'numel'):
                            total_params += tensor.numel()
                    result['structure']['total_parameters'] = total_params
                    
                    result['valid'] = True
                    result['recommendations'].append("TorchScript 모델")
                except Exception as e:
                    result['issues'].append(f"TorchScript state_dict 접근 실패: {e}")
                    
            else:
                result['structure']['type'] = str(type(checkpoint))
                result['issues'].append(f"지원하지 않는 타입: {type(checkpoint)}")
                
        except Exception as e:
            result['issues'].append(f"체크포인트 검증 중 오류: {e}")
        
        return result
    
    def _detect_architecture_from_keys(self, keys: List[str]) -> List[str]:
        """키 목록에서 아키텍처 감지"""
        hints = []
        
        # 확장된 아키텍처 키워드
        architecture_keywords = {
            'graphonomy': ['backbone', 'decoder', 'classifier', 'schp', 'hrnet'],
            'u2net': ['stage1', 'stage2', 'stage3', 'stage4', 'side', 'u2net'],
            'deeplabv3plus': ['backbone', 'decoder', 'classifier', 'aspp', 'deeplab'],
            'gmm': ['feature_extraction', 'regression', 'gmm', 'geometric'],
            'tps': ['localization_net', 'grid_generator', 'tps', 'transformation'],
            'raft': ['feature_encoder', 'context_encoder', 'flow_head', 'raft'],
            'sam': ['image_encoder', 'prompt_encoder', 'mask_decoder', 'sam'],
            'stable_diffusion': ['unet', 'vae', 'text_encoder', 'diffusion', 'model'],
            'ootd': ['unet_vton', 'unet_garm', 'vae', 'ootd'],
            'real_esrgan': ['body', 'upsampling', 'esrgan', 'real_esrgan'],
            'swinir': ['layers', 'patch_embed', 'norm', 'swin', 'swinir'],
            'clip': ['visual', 'transformer', 'text_projection', 'clip'],
            'hrnet': ['hrnet', 'stage', 'transition', 'hrnet_w'],
            'openpose': ['pose', 'body', 'hand', 'face', 'openpose'],
            'yolo': ['yolo', 'detect', 'anchor', 'yolov'],
            'mediapipe': ['mediapipe', 'landmark', 'pose'],
            'viton': ['viton', 'vton', 'warping', 'tom'],
            'dpt': ['dpt', 'depth', 'midas'],
            'efficientnet': ['efficientnet', 'efficient'],
            'resnet': ['resnet', 'residual'],
            'mobilenet': ['mobilenet', 'mobile'],
            'densenet': ['densenet', 'dense']
        }
        
        for arch_name, keywords in architecture_keywords.items():
            matches = sum(1 for keyword in keywords if any(keyword.lower() in key.lower() for key in keys))
            if matches > 0:
                hints.append(f"{arch_name} (매칭: {matches}개)")
        
        return hints
    
    def validate_model_architecture(self, checkpoint_path: str, expected_architecture: str = None) -> Dict[str, Any]:
        """모델 아키텍처 검증"""
        result = {
            'checkpoint_path': checkpoint_path,
            'architecture_valid': False,
            'expected_architecture': expected_architecture,
            'detected_architecture': None,
            'layer_analysis': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                
                # 아키텍처 감지
                architecture_indicators = {
                    'graphonomy': ['backbone', 'decoder', 'classifier'],
                    'u2net': ['stage1', 'stage2', 'stage3', 'stage4'],
                    'deeplabv3plus': ['backbone', 'decoder', 'classifier'],
                    'gmm': ['feature_extraction', 'regression'],
                    'tps': ['localization_net', 'grid_generator'],
                    'raft': ['feature_encoder', 'context_encoder', 'flow_head'],
                    'sam': ['image_encoder', 'prompt_encoder', 'mask_decoder'],
                    'stable_diffusion': ['unet', 'vae', 'text_encoder'],
                    'ootd': ['unet_vton', 'unet_garm', 'vae'],
                    'real_esrgan': ['body', 'upsampling'],
                    'swinir': ['layers', 'patch_embed', 'norm'],
                    'clip': ['visual', 'transformer', 'text_projection']
                }
                
                detected_arch = None
                max_matches = 0
                
                for arch_name, indicators in architecture_indicators.items():
                    matches = sum(1 for indicator in indicators if any(indicator in key for key in state_dict.keys()))
                    if matches > max_matches:
                        max_matches = matches
                        detected_arch = arch_name
                
                result['detected_architecture'] = detected_arch
                result['architecture_valid'] = detected_arch is not None
                
                if expected_architecture and detected_arch != expected_architecture:
                    result['issues'].append(f"예상 아키텍처: {expected_architecture}, 감지된 아키텍처: {detected_arch}")
                elif detected_arch:
                    result['recommendations'].append(f"감지된 아키텍처: {detected_arch}")
                
                # 레이어 분석
                layer_groups = {}
                for key in state_dict.keys():
                    if '.' in key:
                        layer_group = key.split('.')[0]
                        layer_groups[layer_group] = layer_groups.get(layer_group, 0) + 1
                
                result['layer_analysis'] = layer_groups
                
        except Exception as e:
            result['issues'].append(f"아키텍처 분석 실패: {e}")
        
        return result
    
    def comprehensive_model_validation(self) -> Dict[str, Any]:
        """종합 모델 검증 (파일 + 체크포인트 + 아키텍처)"""
        comprehensive_result = {
            'validation_time': datetime.now().isoformat(),
            'total_models': 0,
            'valid_models': 0,
            'invalid_models': 0,
            'detailed_results': {},
            'summary': {
                'critical_issues': [],
                'warnings': [],
                'recommendations': []
            }
        }
        
        # 각 단계별 모델 검증
        step_validations = [
            ('step_01', self.validate_step_01_human_parsing),
            ('step_02', self.validate_step_02_pose_estimation),
            ('step_03', self.validate_step_03_cloth_segmentation),
            ('step_04', self.validate_step_04_geometric_matching),
            ('step_05', self.validate_step_05_cloth_warping),
            ('step_06', self.validate_step_06_virtual_fitting),
            ('step_07', self.validate_step_07_post_processing),
            ('step_08', self.validate_step_08_quality_assessment)
        ]
        
        for step_name, validation_func in step_validations:
            step_result = validation_func()
            comprehensive_result['detailed_results'][step_name] = step_result
            
            # 체크포인트 상세 검증
            if step_result.get('requirements_met', False):
                model_files = {
                    'step_01': {
                        'graphonomy': 'ai_models/Self-Correction-Human-Parsing/exp-schp-201908261155-atr.pth',
                        'u2net': 'ai_models/step_03_cloth_segmentation/u2net.pth',
                        'deeplabv3plus': 'ai_models/step_01_human_parsing/deeplabv3plus.pth'
                    },
                    'step_02': {
                        'body_pose_model': 'ai_models/step_02_pose_estimation/body_pose_model.pth',
                        'yolov8n_pose': 'ai_models/step_02_pose_estimation/yolov8n-pose.pt',
                        'openpose': 'ai_models/step_02_pose_estimation/openpose.pth',
                        'hrnet_w48': 'ai_models/step_02_pose_estimation/hrnet_w48_coco_384x288.pth'
                    },
                    'step_03': {
                        'sam_vit_h': 'ai_models/step_03_cloth_segmentation/sam_vit_h_4b8939.pth',
                        'u2net': 'ai_models/step_03_cloth_segmentation/u2net.pth',
                        'deeplabv3_resnet101': 'ai_models/step_03_cloth_segmentation/deeplabv3_resnet101_ultra.pth',
                        'mobile_sam': 'ai_models/step_03_cloth_segmentation/mobile_sam.pt'
                    },
                    'step_04': {
                        'gmm_final': 'ai_models/step_04_geometric_matching/gmm_final.pth',
                        'tps_network': 'ai_models/step_04_geometric_matching/tps_network.pth',
                        'raft_things': 'ai_models/step_04_geometric_matching/raft-things.pth',
                        'sam_vit_h': 'ai_models/step_04_geometric_matching/sam_vit_h_4b8939.pth'
                    },
                    'step_05': {
                        'tom_final': 'ai_models/step_05_cloth_warping/tom_final.pth',
                        'viton_hd_warping': 'ai_models/step_05_cloth_warping/viton_hd_warping.pth',
                        'tps_transformation': 'ai_models/step_05_cloth_warping/tps_transformation.pth'
                    },
                    'step_06': {
                        'stable_diffusion': 'ai_models/step_06_virtual_fitting/stable_diffusion_4.8gb.pth',
                        'ootd': 'ai_models/step_06_virtual_fitting/ootd_3.2gb.pth',
                        'ootd_checkpoint': 'ai_models/step_06_virtual_fitting/validated_checkpoints/ootd_checkpoint.pth'
                    },
                    'step_07': {
                        'RealESRGAN_x4plus': 'ai_models/step_07_post_processing/RealESRGAN_x4plus.pth',
                        'swinir_large': 'ai_models/step_07_post_processing/swinir_real_sr_x4_large.pth',
                        'GFPGAN': 'ai_models/step_07_post_processing/GFPGAN.pth'
                    },
                    'step_08': {
                        'clip_vit_b32': 'ai_models/step_08_quality_assessment/clip_vit_b32.pth',
                        'ViT_L_14': 'ai_models/step_08_quality_assessment/ViT-L-14.pt',
                        'lpips_alex': 'ai_models/step_08_quality_assessment/lpips_alex.pth'
                    }
                }
                
                if step_name in model_files:
                    step_checkpoint_results = {}
                    for model_name, file_path in model_files[step_name].items():
                        # 체크포인트 내용 검증
                        checkpoint_result = self.validate_checkpoint_content(file_path)
                        step_checkpoint_results[model_name] = checkpoint_result
                        
                        # 아키텍처 검증
                        architecture_result = self.validate_model_architecture(file_path)
                        step_checkpoint_results[f"{model_name}_architecture"] = architecture_result
                        
                        # 통계 업데이트
                        comprehensive_result['total_models'] += 1
                        if checkpoint_result['valid']:
                            comprehensive_result['valid_models'] += 1
                        else:
                            comprehensive_result['invalid_models'] += 1
                            comprehensive_result['summary']['critical_issues'].append(f"{step_name}/{model_name}: 체크포인트 검증 실패")
                    
                    comprehensive_result['detailed_results'][f"{step_name}_checkpoints"] = step_checkpoint_results
        
        # 종합 권장사항
        if comprehensive_result['invalid_models'] > 0:
            comprehensive_result['summary']['recommendations'].append(f"{comprehensive_result['invalid_models']}개 모델의 체크포인트를 확인하세요")
        else:
            comprehensive_result['summary']['recommendations'].append("모든 모델이 유효합니다!")
        
        return comprehensive_result
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """종합 검증 리포트 생성"""
        report = {
            'validation_time': datetime.now().isoformat(),
            'total_steps_validated': len(self.validation_results),
            'overall_status': 'unknown',
            'steps': self.validation_results,
            'summary': {
                'total_issues': 0,
                'total_recommendations': 0,
                'critical_issues': [],
                'memory_requirements': {},
                'device_requirements': {}
            }
        }
        
        # 종합 분석
        all_requirements_met = True
        total_issues = 0
        total_recommendations = 0
        critical_issues = []
        
        for step_name, step_result in self.validation_results.items():
            if not step_result.get('requirements_met', False):
                all_requirements_met = False
            
            total_issues += len(step_result.get('issues', []))
            total_recommendations += len(step_result.get('recommendations', []))
            
            # 치명적 문제들 수집
            for issue in step_result.get('issues', []):
                if any(keyword in issue.lower() for keyword in ['없음', '실패', '오류', '부족']):
                    critical_issues.append(f"{step_name}: {issue}")
        
        report['overall_status'] = 'ready' if all_requirements_met else 'issues_detected'
        report['summary']['total_issues'] = total_issues
        report['summary']['total_recommendations'] = total_recommendations
        report['summary']['critical_issues'] = critical_issues
        
        return report

def get_ai_inference_validator() -> AIInferenceValidator:
    """AI 추론 검증기 싱글톤 인스턴스 반환"""
    if not hasattr(get_ai_inference_validator, '_instance'):
        get_ai_inference_validator._instance = AIInferenceValidator()
    return get_ai_inference_validator._instance

# 🔥 사용 예시
if __name__ == "__main__":
    validator = get_ai_inference_validator()
    
    print("🔍 AI 추론 요구사항 검증 시작...")
    print("=" * 60)
    
    # 기본 검증
    print("📋 1단계: 기본 파일 존재 여부 검증")
    step1_result = validator.validate_step_01_human_parsing()
    print(f"📊 Step 1 (Human Parsing): {'✅' if step1_result['requirements_met'] else '❌'}")
    
    step2_result = validator.validate_step_02_pose_estimation()
    print(f"📊 Step 2 (Pose Estimation): {'✅' if step2_result['requirements_met'] else '❌'}")
    
    step3_result = validator.validate_step_03_cloth_segmentation()
    print(f"📊 Step 3 (Cloth Segmentation): {'✅' if step3_result['requirements_met'] else '❌'}")
    
    step4_result = validator.validate_step_04_geometric_matching()
    print(f"📊 Step 4 (Geometric Matching): {'✅' if step4_result['requirements_met'] else '❌'}")
    
    step5_result = validator.validate_step_05_cloth_warping()
    print(f"📊 Step 5 (Cloth Warping): {'✅' if step5_result['requirements_met'] else '❌'}")

    step6_result = validator.validate_step_06_virtual_fitting()
    print(f"📊 Step 6 (Virtual Fitting): {'✅' if step6_result['requirements_met'] else '❌'}")

    step7_result = validator.validate_step_07_post_processing()
    print(f"📊 Step 7 (Post Processing): {'✅' if step7_result['requirements_met'] else '❌'}")

    step8_result = validator.validate_step_08_quality_assessment()
    print(f"📊 Step 8 (Quality Assessment): {'✅' if step8_result['requirements_met'] else '❌'}")
    
    print("\n📋 2단계: 체크포인트 내용 및 아키텍처 검증")
    print("=" * 60)
    
    # 종합 검증 (체크포인트 + 아키텍처)
    comprehensive_result = validator.comprehensive_model_validation()
    
    print(f"📊 총 모델 수: {comprehensive_result['total_models']}개")
    print(f"✅ 유효한 모델: {comprehensive_result['valid_models']}개")
    print(f"❌ 무효한 모델: {comprehensive_result['invalid_models']}개")
    
    # 주요 모델들의 체크포인트 검증 결과 출력
    key_models = [
        ('Step 1 - Graphonomy', 'ai_models/Self-Correction-Human-Parsing/exp-schp-201908261155-atr.pth'),
        ('Step 2 - HRNet', 'ai_models/step_02_pose_estimation/hrnet_w48_coco_384x288.pth'),
        ('Step 3 - SAM ViT-H', 'ai_models/step_03_cloth_segmentation/sam_vit_h_4b8939.pth'),
        ('Step 4 - GMM', 'ai_models/step_04_geometric_matching/gmm_final.pth'),
        ('Step 6 - Stable Diffusion', 'ai_models/step_06_virtual_fitting/stable_diffusion_4.8gb.pth'),
        ('Step 6 - OOTD', 'ai_models/step_06_virtual_fitting/ootd_3.2gb.pth')
    ]
    
    for model_name, checkpoint_path in key_models:
        checkpoint_result = validator.validate_checkpoint_content(checkpoint_path)
        architecture_result = validator.validate_model_architecture(checkpoint_path)
        
        status = "✅" if checkpoint_result['valid'] else "❌"
        arch_status = "✅" if architecture_result['architecture_valid'] else "❌"
        
        print(f"{status} {model_name}:")
        print(f"   📁 파일 크기: {checkpoint_result['size_mb']:.1f}MB")
        print(f"   🔧 체크포인트: {'유효' if checkpoint_result['valid'] else '무효'}")
        if checkpoint_result['valid'] and 'structure' in checkpoint_result:
            print(f"   📊 파라미터 수: {checkpoint_result['structure'].get('total_parameters', 0):,}")
            print(f"   🏗️ 레이어 수: {checkpoint_result['structure'].get('state_dict_count', 0)}")
        print(f"   🏛️ 아키텍처: {arch_status} {architecture_result.get('detected_architecture', 'Unknown')}")
        
        if checkpoint_result['issues']:
            print(f"   ⚠️ 문제점: {checkpoint_result['issues']}")
    
    # 종합 리포트
    report = validator.get_comprehensive_report()
    print(f"\n📋 종합 상태: {report['overall_status']}")
    print(f"🔧 총 문제점: {report['summary']['total_issues']}개")
    print(f"💡 권장사항: {report['summary']['total_recommendations']}개")
    
    if report['summary']['critical_issues']:
        print("\n🚨 치명적 문제점들:")
        for issue in report['summary']['critical_issues']:
            print(f"   - {issue}")
    
    if comprehensive_result['summary']['critical_issues']:
        print("\n🚨 체크포인트 문제점들:")
        for issue in comprehensive_result['summary']['critical_issues']:
            print(f"   - {issue}")
