#!/usr/bin/env python3
"""
🔥 MyCloset AI - Cloth Warping Model Loader Service
==================================================

🎯 의류 워핑 모델 로더 서비스
✅ 모델 로딩 및 관리
✅ 체크포인트 관리
✅ 모델 버전 관리
✅ M3 Max 최적화
"""

import logging
import torch
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class ModelLoaderConfig:
    """모델 로더 설정"""
    models_directory: str = "models"
    checkpoints_directory: str = "checkpoints"
    enable_model_caching: bool = True
    enable_auto_download: bool = False
    use_mps: bool = True

class ClothWarpingModelLoaderService:
    """의류 워핑 모델 로더 서비스"""
    
    def __init__(self, config: ModelLoaderConfig = None):
        self.config = config or ModelLoaderConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.info("🎯 Cloth Warping 모델 로더 서비스 초기화")
        
        # 모델 캐시
        self.model_cache = {}
        self.loaded_models = {}
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() and self.config.use_mps else "cpu")
        
        # 디렉토리 확인 및 생성
        self._ensure_directories()
        
        self.logger.info("✅ Cloth Warping 모델 로더 서비스 초기화 완료")
    
    def _ensure_directories(self):
        """필요한 디렉토리들을 생성합니다."""
        directories = [self.config.models_directory, self.config.checkpoints_directory]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                self.logger.info(f"디렉토리 생성: {directory}")
    
    def load_model(self, model_name: str, model_path: str = None) -> Optional[torch.nn.Module]:
        """모델을 로드합니다."""
        try:
            # 캐시에서 확인
            if self.config.enable_model_caching and model_name in self.model_cache:
                self.logger.info(f"캐시에서 모델 로드: {model_name}")
                return self.model_cache[model_name]
            
            # 모델 경로 결정
            if model_path is None:
                model_path = os.path.join(self.config.models_directory, f"{model_name}.pth")
            
            # 모델 로드
            if os.path.exists(model_path):
                model = torch.load(model_path, map_location=self.device)
                model.to(self.device)
                
                # 캐시에 저장
                if self.config.enable_model_caching:
                    self.model_cache[model_name] = model
                
                self.loaded_models[model_name] = {
                    'path': model_path,
                    'device': str(self.device),
                    'loaded_at': torch.cuda.Event() if self.device.type == 'cuda' else None
                }
                
                self.logger.info(f"모델 로드 완료: {model_name}")
                return model
            else:
                self.logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {model_name}, 오류: {e}")
            return None
    
    def save_model(self, model: torch.nn.Module, model_name: str, save_path: str = None) -> bool:
        """모델을 저장합니다."""
        try:
            # 저장 경로 결정
            if save_path is None:
                save_path = os.path.join(self.config.models_directory, f"{model_name}.pth")
            
            # 모델 저장
            torch.save(model, save_path)
            
            self.logger.info(f"모델 저장 완료: {model_name} -> {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 저장 실패: {model_name}, 오류: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_name: str, checkpoint_path: str = None) -> Optional[Dict[str, Any]]:
        """체크포인트를 로드합니다."""
        try:
            # 체크포인트 경로 결정
            if checkpoint_path is None:
                checkpoint_path = os.path.join(self.config.checkpoints_directory, f"{checkpoint_name}.pth")
            
            # 체크포인트 로드
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.logger.info(f"체크포인트 로드 완료: {checkpoint_name}")
                return checkpoint
            else:
                self.logger.error(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"체크포인트 로드 실패: {checkpoint_name}, 오류: {e}")
            return None
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_name: str, save_path: str = None) -> bool:
        """체크포인트를 저장합니다."""
        try:
            # 저장 경로 결정
            if save_path is None:
                save_path = os.path.join(self.config.checkpoints_directory, f"{checkpoint_name}.pth")
            
            # 체크포인트 저장
            torch.save(checkpoint_data, save_path)
            
            self.logger.info(f"체크포인트 저장 완료: {checkpoint_name} -> {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"체크포인트 저장 실패: {checkpoint_name}, 오류: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """사용 가능한 모델 목록을 반환합니다."""
        try:
            models = []
            if os.path.exists(self.config.models_directory):
                for file in os.listdir(self.config.models_directory):
                    if file.endswith('.pth'):
                        models.append(file[:-4])  # .pth 확장자 제거
            return models
        except Exception as e:
            self.logger.error(f"모델 목록 조회 실패: {e}")
            return []
    
    def get_available_checkpoints(self) -> List[str]:
        """사용 가능한 체크포인트 목록을 반환합니다."""
        try:
            checkpoints = []
            if os.path.exists(self.config.checkpoints_directory):
                for file in os.listdir(self.config.checkpoints_directory):
                    if file.endswith('.pth'):
                        checkpoints.append(file[:-4])  # .pth 확장자 제거
            return checkpoints
        except Exception as e:
            self.logger.error(f"체크포인트 목록 조회 실패: {e}")
            return []
    
    def clear_cache(self) -> bool:
        """모델 캐시를 정리합니다."""
        try:
            cache_size = len(self.model_cache)
            self.model_cache.clear()
            self.logger.info(f"모델 캐시 정리 완료: {cache_size}개 모델 제거")
            return True
        except Exception as e:
            self.logger.error(f"모델 캐시 정리 실패: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """모델 정보를 반환합니다."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        return None
    
    def get_service_stats(self) -> Dict[str, Any]:
        """서비스 통계를 반환합니다."""
        return {
            'loaded_models': len(self.loaded_models),
            'cached_models': len(self.model_cache),
            'available_models': self.get_available_models(),
            'available_checkpoints': self.get_available_checkpoints(),
            'device': str(self.device),
            'config': self.config.__dict__
        }

# 사용 예시
if __name__ == "__main__":
    # 설정
    config = ModelLoaderConfig(
        models_directory="models",
        checkpoints_directory="checkpoints",
        enable_model_caching=True,
        enable_auto_download=False,
        use_mps=True
    )
    
    # 모델 로더 서비스 초기화
    model_loader = ClothWarpingModelLoaderService(config)
    
    # 사용 가능한 모델 및 체크포인트 확인
    available_models = model_loader.get_available_models()
    available_checkpoints = model_loader.get_available_checkpoints()
    
    print(f"사용 가능한 모델: {available_models}")
    print(f"사용 가능한 체크포인트: {available_checkpoints}")
    
    # 서비스 통계
    stats = model_loader.get_service_stats()
    print(f"서비스 통계: {stats}")
    
    # 테스트 모델 생성 및 저장
    test_model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1)
    )
    
    # 모델 저장
    save_success = model_loader.save_model(test_model, "test_model")
    print(f"테스트 모델 저장: {'성공' if save_success else '실패'}")
    
    # 모델 로드
    loaded_model = model_loader.load_model("test_model")
    print(f"테스트 모델 로드: {'성공' if loaded_model is not None else '실패'}")
    
    # 체크포인트 저장
    checkpoint_data = {
        'model_state_dict': test_model.state_dict(),
        'epoch': 100,
        'loss': 0.01
    }
    
    save_checkpoint_success = model_loader.save_checkpoint(checkpoint_data, "test_checkpoint")
    print(f"테스트 체크포인트 저장: {'성공' if save_checkpoint_success else '실패'}")
    
    # 체크포인트 로드
    loaded_checkpoint = model_loader.load_checkpoint("test_checkpoint")
    print(f"테스트 체크포인트 로드: {'성공' if loaded_checkpoint is not None else '실패'}")
