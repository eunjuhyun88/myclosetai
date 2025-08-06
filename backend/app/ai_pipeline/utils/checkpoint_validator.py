"""
체크포인트 검증 및 이동 유틸리티
"""
import os
import shutil
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

class CheckpointValidator:
    """체크포인트 파일 검증 및 이동 클래스"""
    
    def __init__(self, base_path: str = None):
        if base_path is None:
            # 현재 작업 디렉토리 기준으로 절대 경로 설정
            current_dir = Path.cwd()
            self.base_path = current_dir / "backend/ai_models/step_06_virtual_fitting"
        else:
            self.base_path = Path(base_path)
        
        # 경로가 존재하는지 확인하고 수정
        if not self.base_path.exists():
            # backend가 중복된 경우 수정
            backend_pattern = "backend" + "/" + "backend"
            if backend_pattern in str(self.base_path):
                self.base_path = Path(str(self.base_path).replace(backend_pattern, "backend"))
        
        self.validated_path = self.base_path / "validated_checkpoints"
        self.validated_path.mkdir(parents=True, exist_ok=True)
        
    def validate_and_move_checkpoints(self) -> Dict[str, bool]:
        """체크포인트 파일들을 검증하고 올바른 위치로 이동"""
        results = {}
        
        # OOTD 체크포인트 검증 및 이동
        results['ootd'] = self._validate_ootd_checkpoints()
        
        # VITON-HD 체크포인트 검증 및 이동
        results['viton_hd'] = self._validate_viton_hd_checkpoints()
        
        # Stable Diffusion 체크포인트 검증 및 이동
        results['stable_diffusion'] = self._validate_stable_diffusion_checkpoints()
        
        return results
    
    def _validate_ootd_checkpoints(self) -> bool:
        """OOTD 체크포인트 검증 및 이동"""
        logger.info("🔍 OOTD 체크포인트 검증 시작...")
        
        # OOTD 체크포인트 소스 경로들
        ootd_sources = [
            "ootdiffusion/diffusion_pytorch_model.bin",
            "ootdiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton/diffusion_pytorch_model.safetensors",
            "ootdiffusion/checkpoints/ootd/ootd_dc/checkpoint-36000/unet_vton/diffusion_pytorch_model.safetensors",
            "ootdiffusion/unet/ootdiffusion/unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "pytorch_model.bin"
        ]
        
        for source in ootd_sources:
            source_path = self.base_path / source
            if source_path.exists():
                try:
                    # 체크포인트 로딩 및 검증
                    if self._validate_checkpoint_structure(source_path, "ootd"):
                        # 올바른 위치로 복사
                        target_path = self.validated_path / "ootd_checkpoint.pth"
                        shutil.copy2(source_path, target_path)
                        logger.info(f"✅ OOTD 체크포인트 검증 및 이동 완료: {source} -> {target_path}")
                        return True
                except Exception as e:
                    logger.warning(f"⚠️ OOTD 체크포인트 검증 실패 ({source}): {e}")
        
        logger.error("❌ OOTD 체크포인트 검증 실패")
        return False
    
    def _validate_viton_hd_checkpoints(self) -> bool:
        """VITON-HD 체크포인트 검증 및 이동"""
        logger.info("🔍 VITON-HD 체크포인트 검증 시작...")
        
        # VITON-HD 체크포인트 소스 경로들
        viton_sources = [
            "ultra_models/viton_hd_2.1gb.pth",
            "viton_hd_2.1gb.pth"
        ]
        
        for source in viton_sources:
            source_path = self.base_path / source
            if source_path.exists():
                try:
                    # 체크포인트 로딩 및 검증
                    if self._validate_checkpoint_structure(source_path, "viton_hd"):
                        # 올바른 위치로 복사
                        target_path = self.validated_path / "viton_hd_checkpoint.pth"
                        shutil.copy2(source_path, target_path)
                        logger.info(f"✅ VITON-HD 체크포인트 검증 및 이동 완료: {source} -> {target_path}")
                        return True
                except Exception as e:
                    logger.warning(f"⚠️ VITON-HD 체크포인트 검증 실패 ({source}): {e}")
        
        logger.error("❌ VITON-HD 체크포인트 검증 실패")
        return False
    
    def _validate_stable_diffusion_checkpoints(self) -> bool:
        """Stable Diffusion 체크포인트 검증 및 이동"""
        logger.info("🔍 Stable Diffusion 체크포인트 검증 시작...")
        
        # Stable Diffusion 체크포인트 소스 경로들
        diffusion_sources = [
            "ootdiffusion/diffusion_pytorch_model.bin",
            "ootdiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000/unet_vton/diffusion_pytorch_model.safetensors",
            "ootdiffusion/checkpoints/ootd/ootd_dc/checkpoint-36000/unet_vton/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.safetensors",
            "pytorch_model.bin",
            "ultra_models/stable_diffusion_4.8gb.pth",
            "stable_diffusion_4.8gb.pth"
        ]
        
        for source in diffusion_sources:
            source_path = self.base_path / source
            if source_path.exists():
                try:
                    # 체크포인트 로딩 및 검증
                    if self._validate_checkpoint_structure(source_path, "stable_diffusion"):
                        # 올바른 위치로 복사
                        target_path = self.validated_path / "stable_diffusion_checkpoint.pth"
                        shutil.copy2(source_path, target_path)
                        logger.info(f"✅ Stable Diffusion 체크포인트 검증 및 이동 완료: {source} -> {target_path}")
                        return True
                except Exception as e:
                    logger.warning(f"⚠️ Stable Diffusion 체크포인트 검증 실패 ({source}): {e}")
        
        logger.error("❌ Stable Diffusion 체크포인트 검증 실패")
        return False
    
    def _validate_checkpoint_structure(self, checkpoint_path: Path, model_type: str) -> bool:
        """체크포인트 구조 검증"""
        try:
            # 체크포인트 로딩
            if checkpoint_path.suffix == '.safetensors':
                checkpoint = load_file(str(checkpoint_path))
            else:
                checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            
            # state_dict 추출
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 모델 타입별 검증
            if model_type == "ootd":
                return self._validate_ootd_structure(state_dict)
            elif model_type == "viton_hd":
                return self._validate_viton_hd_structure(state_dict)
            elif model_type == "stable_diffusion":
                return self._validate_stable_diffusion_structure(state_dict)
            
            return True
            
        except Exception as e:
            logger.error(f"체크포인트 구조 검증 실패 ({checkpoint_path}): {e}")
            return False
    
    def _validate_ootd_structure(self, state_dict: Dict) -> bool:
        """OOTD 체크포인트 구조 검증"""
        required_keys = [
            'conv_in.weight', 'conv_in.bias',
            'time_embedding.linear_1.weight', 'time_embedding.linear_1.bias',
            'time_embedding.linear_2.weight', 'time_embedding.linear_2.bias'
        ]
        
        found_keys = 0
        for key in required_keys:
            if key in state_dict:
                found_keys += 1
                logger.info(f"✅ OOTD 키 발견: {key} (shape: {state_dict[key].shape})")
        
        logger.info(f"OOTD 검증 결과: {found_keys}/{len(required_keys)} 키 발견")
        return found_keys >= 2  # 최소 2개 키는 있어야 함
    
    def _validate_viton_hd_structure(self, state_dict: Dict) -> bool:
        """VITON-HD 체크포인트 구조 검증"""
        # VITON-HD 관련 키 패턴 확인
        viton_keys = [key for key in state_dict.keys() if any(pattern in key for pattern in ['viton', 'hrviton', 'geometric', 'tryon'])]
        
        if viton_keys:
            logger.info(f"✅ VITON-HD 키 발견: {len(viton_keys)}개")
            for key in viton_keys[:5]:  # 처음 5개만 로그
                logger.info(f"  - {key} (shape: {state_dict[key].shape})")
            return True
        
        logger.warning("⚠️ VITON-HD 관련 키를 찾을 수 없음")
        return False
    
    def _validate_stable_diffusion_structure(self, state_dict: Dict) -> bool:
        """Stable Diffusion 체크포인트 구조 검증"""
        # Stable Diffusion 관련 키 패턴 확인
        diffusion_keys = [key for key in state_dict.keys() if any(pattern in key for pattern in ['unet', 'vae', 'text_encoder', 'diffusion'])]
        
        if diffusion_keys:
            logger.info(f"✅ Stable Diffusion 키 발견: {len(diffusion_keys)}개")
            for key in diffusion_keys[:5]:  # 처음 5개만 로그
                logger.info(f"  - {key} (shape: {state_dict[key].shape})")
            return True
        
        logger.warning("⚠️ Stable Diffusion 관련 키를 찾을 수 없음")
        return False
    
    def get_validated_checkpoint_paths(self) -> Dict[str, str]:
        """검증된 체크포인트 경로들 반환"""
        paths = {}
        
        ootd_path = self.validated_path / "ootd_checkpoint.pth"
        if ootd_path.exists():
            paths['ootd'] = str(ootd_path)
        
        viton_path = self.validated_path / "viton_hd_checkpoint.pth"
        if viton_path.exists():
            paths['viton_hd'] = str(viton_path)
        
        diffusion_path = self.validated_path / "stable_diffusion_checkpoint.pth"
        if diffusion_path.exists():
            paths['stable_diffusion'] = str(diffusion_path)
        
        return paths

def validate_all_checkpoints() -> Dict[str, str]:
    """모든 체크포인트 검증 및 이동"""
    validator = CheckpointValidator()
    results = validator.validate_and_move_checkpoints()
    
    logger.info("📊 체크포인트 검증 결과:")
    for model, success in results.items():
        status = "✅ 성공" if success else "❌ 실패"
        logger.info(f"  - {model}: {status}")
    
    return validator.get_validated_checkpoint_paths()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    validated_paths = validate_all_checkpoints()
    print("검증된 체크포인트 경로:", validated_paths) 