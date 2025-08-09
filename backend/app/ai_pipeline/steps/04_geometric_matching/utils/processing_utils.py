"""
처리 유틸리티들
"""

import os
import sys
import time
import logging
import subprocess
import platform
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def detect_m3_max() -> bool:
    """M3 Max 감지"""
    try:
        if platform.system() == 'Darwin':
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            return 'M3' in result.stdout
    except Exception as e:
        logger.warning(f"M3 Max 감지 실패: {e}")
    
    return False


def make_resnet_layer(block_class, inplanes, planes, blocks, stride=1, dilation=1, downsample=None):
    """ResNet 레이어 생성"""
    layers = []
    if block_class.__name__ == 'CommonBottleneckBlock':
        layers.append(block_class(inplanes, planes, stride, dilation, downsample))
        inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(block_class(inplanes, planes, 1, dilation))
    else:
        layers.append(block_class(inplanes, planes, stride))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block_class(inplanes, planes, 1))
    return layers


def _get_central_hub_container():
    """Central Hub 컨테이너 가져오기"""
    try:
        from app.core.di_container import CentralHubDIContainer
        return CentralHubDIContainer.get_instance()
    except ImportError:
        logger.warning("CentralHubDIContainer를 가져올 수 없음")
        return None


def _inject_dependencies_safe(step_instance):
    """의존성 안전 주입"""
    try:
        container = _get_central_hub_container()
        if container:
            # 필요한 서비스들 주입
            services_to_inject = [
                'model_loader',
                'memory_manager',
                'data_converter',
                'cache_manager'
            ]
            
            for service_name in services_to_inject:
                if hasattr(step_instance, f'_{service_name}'):
                    try:
                        service = container.get_service(service_name)
                        if service:
                            setattr(step_instance, f'_{service_name}', service)
                            logger.info(f"✅ {service_name} 서비스 주입 성공")
                    except Exception as e:
                        logger.warning(f"⚠️ {service_name} 서비스 주입 실패: {e}")
        
    except Exception as e:
        logger.warning(f"의존성 주입 실패: {e}")


def validate_input_images(person_image, clothing_image) -> Tuple[bool, str]:
    """입력 이미지 검증"""
    try:
        # 이미지 존재 확인
        if person_image is None:
            return False, "사람 이미지가 없습니다."
        
        if clothing_image is None:
            return False, "의류 이미지가 없습니다."
        
        # 이미지 형태 확인
        if hasattr(person_image, 'shape'):
            if len(person_image.shape) != 3:
                return False, "사람 이미지가 3차원이 아닙니다."
        
        if hasattr(clothing_image, 'shape'):
            if len(clothing_image.shape) != 3:
                return False, "의류 이미지가 3차원이 아닙니다."
        
        return True, "입력 이미지 검증 성공"
    
    except Exception as e:
        return False, f"이미지 검증 중 오류 발생: {e}"


def preprocess_images(person_image, clothing_image, target_size=(256, 192)):
    """이미지 전처리"""
    try:
        import torch
        import torch.nn.functional as F
        import numpy as np
        
        # 이미지를 텐서로 변환
        if isinstance(person_image, np.ndarray):
            person_tensor = torch.from_numpy(person_image).float()
        elif isinstance(person_image, torch.Tensor):
            person_tensor = person_image
        else:
            raise ValueError("지원되지 않는 이미지 형식")
        
        if isinstance(clothing_image, np.ndarray):
            clothing_tensor = torch.from_numpy(clothing_image).float()
        elif isinstance(clothing_image, torch.Tensor):
            clothing_tensor = clothing_image
        else:
            raise ValueError("지원되지 않는 이미지 형식")
        
        # 배치 차원 추가
        if person_tensor.dim() == 3:
            person_tensor = person_tensor.unsqueeze(0)
        if clothing_tensor.dim() == 3:
            clothing_tensor = clothing_tensor.unsqueeze(0)
        
        # 크기 조정
        if person_tensor.shape[-2:] != target_size:
            person_tensor = F.interpolate(person_tensor, size=target_size, mode='bilinear', align_corners=False)
        
        if clothing_tensor.shape[-2:] != target_size:
            clothing_tensor = F.interpolate(clothing_tensor, size=target_size, mode='bilinear', align_corners=False)
        
        # 정규화 (0-1 범위)
        if person_tensor.max() > 1:
            person_tensor = person_tensor / 255.0
        if clothing_tensor.max() > 1:
            clothing_tensor = clothing_tensor / 255.0
        
        return person_tensor, clothing_tensor
    
    except Exception as e:
        logger.error(f"이미지 전처리 실패: {e}")
        raise


def postprocess_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """결과 후처리"""
    try:
        processed_results = results.copy()
        
        # 신뢰도 점수 계산
        if 'confidence' not in processed_results:
            processed_results['confidence'] = 0.5  # 기본값
        
        # 품질 점수 계산
        if 'quality_score' not in processed_results:
            processed_results['quality_score'] = 0.5  # 기본값
        
        # 메타데이터 추가
        processed_results['timestamp'] = time.time()
        processed_results['version'] = '1.0.0'
        
        return processed_results
    
    except Exception as e:
        logger.error(f"결과 후처리 실패: {e}")
        return results


def create_identity_grid(batch_size: int, H: int, W: int, device='cpu'):
    """항등 그리드 생성"""
    try:
        import torch
        
        # 정규화된 좌표 그리드 생성
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        # 배치 차원 추가
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        return grid
    
    except Exception as e:
        logger.error(f"항등 그리드 생성 실패: {e}")
        raise
