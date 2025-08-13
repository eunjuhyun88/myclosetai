#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Validation Service
=====================================================================

데이터 검증을 위한 전용 서비스

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import cv2

logger = logging.getLogger(__name__)

class ValidationService:
    """검증 서비스"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """초기화"""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.ValidationService")
        
    def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """입력 데이터 검증"""
        try:
            # 이미지 검증
            if 'image' not in data:
                return False, "이미지가 없습니다"
            
            image = data['image']
            if not self._validate_image(image):
                return False, "유효하지 않은 이미지입니다"
            
            # 추가 데이터 검증
            if 'person_parsing' in data:
                if not self._validate_person_parsing(data['person_parsing']):
                    return False, "유효하지 않은 person_parsing 데이터입니다"
            
            if 'pose_info' in data:
                if not self._validate_pose_info(data['pose_info']):
                    return False, "유효하지 않은 pose_info 데이터입니다"
            
            return True, "검증 성공"
            
        except Exception as e:
            self.logger.error(f"❌ 입력 검증 실패: {e}")
            return False, f"검증 실패: {str(e)}"
    
    def validate_output(self, result: Dict[str, Any]) -> Tuple[bool, str]:
        """출력 데이터 검증"""
        try:
            # 기본 필수 필드 검증
            required_fields = ['success', 'masks', 'confidence']
            for field in required_fields:
                if field not in result:
                    return False, f"필수 필드가 없습니다: {field}"
            
            # 성공 여부 검증
            if not result['success']:
                return True, "실패한 결과이지만 유효한 형식입니다"
            
            # 마스크 검증
            if not self._validate_masks(result['masks']):
                return False, "유효하지 않은 마스크 데이터입니다"
            
            # 신뢰도 검증
            confidence = result.get('confidence', 0)
            if not (0 <= confidence <= 1):
                return False, "신뢰도가 유효하지 않습니다 (0-1 범위)"
            
            return True, "검증 성공"
            
        except Exception as e:
            self.logger.error(f"❌ 출력 검증 실패: {e}")
            return False, f"검증 실패: {str(e)}"
    
    def _validate_image(self, image: Any) -> bool:
        """이미지 검증"""
        try:
            if image is None:
                return False
            
            # NumPy 배열 검증
            if isinstance(image, np.ndarray):
                if image.size == 0:
                    return False
                if len(image.shape) != 3:
                    return False
                if image.shape[2] != 3:
                    return False
                return True
            
            # PIL Image 검증
            if hasattr(image, 'convert'):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 이미지 검증 실패: {e}")
            return False
    
    def _validate_masks(self, masks: Dict[str, Any]) -> bool:
        """마스크 검증"""
        try:
            if not isinstance(masks, dict):
                return False
            
            for mask_key, mask in masks.items():
                if mask is not None:
                    if isinstance(mask, np.ndarray):
                        if mask.size == 0:
                            return False
                        if len(mask.shape) != 2:
                            return False
                    else:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 마스크 검증 실패: {e}")
            return False
    
    def _validate_person_parsing(self, person_parsing: Dict[str, Any]) -> bool:
        """person_parsing 검증"""
        try:
            if not isinstance(person_parsing, dict):
                return False
            
            # 기본 구조 검증
            if 'regions' in person_parsing:
                if not isinstance(person_parsing['regions'], dict):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ person_parsing 검증 실패: {e}")
            return False
    
    def _validate_pose_info(self, pose_info: Dict[str, Any]) -> bool:
        """pose_info 검증"""
        try:
            if not isinstance(pose_info, dict):
                return False
            
            # 기본 구조 검증
            if 'keypoints' in pose_info:
                if not isinstance(pose_info['keypoints'], dict):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ pose_info 검증 실패: {e}")
            return False
    
    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """설정 검증"""
        try:
            # 필수 설정 검증
            required_configs = ['method', 'quality_level', 'input_size']
            for config_key in required_configs:
                if config_key not in config:
                    return False, f"필수 설정이 없습니다: {config_key}"
            
            # 값 검증
            if not isinstance(config['input_size'], (list, tuple)):
                return False, "input_size는 리스트 또는 튜플이어야 합니다"
            
            if len(config['input_size']) != 2:
                return False, "input_size는 2개의 요소를 가져야 합니다"
            
            return True, "설정 검증 성공"
            
        except Exception as e:
            self.logger.error(f"❌ 설정 검증 실패: {e}")
            return False, f"설정 검증 실패: {str(e)}"
