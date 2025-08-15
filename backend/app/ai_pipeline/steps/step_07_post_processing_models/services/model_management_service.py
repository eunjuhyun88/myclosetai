"""
Model Management Service
모델의 생명주기, 버전 관리, 배포를 담당하는 서비스 클래스
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple
import os
import json
import shutil
import hashlib
import time
from datetime import datetime
from dataclasses import dataclass, asdict
import logging
import pickle
from pathlib import Path

# 로깅 설정
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """모델 정보를 저장하는 데이터 클래스"""
    model_name: str
    model_type: str
    version: str
    checkpoint_path: str
    model_size: int  # bytes
    creation_date: str
    last_modified: str
    checksum: str
    device: str
    status: str  # 'active', 'inactive', 'deprecated'
    performance_metrics: Dict[str, float]
    dependencies: List[str]
    description: str

@dataclass
class DeploymentConfig:
    """배포 설정을 저장하는 데이터 클래스"""
    target_device: str
    optimization_level: str  # 'none', 'basic', 'advanced'
    quantization: bool
    pruning: bool
    batch_size: int
    memory_limit: int  # MB
    timeout: int  # seconds
    fallback_model: Optional[str]

class ModelManagementService:
    """
    모델의 생명주기, 버전 관리, 배포를 담당하는 서비스 클래스
    """
    
    def __init__(self, base_path: str = "models", device: Optional[torch.device] = None):
        """
        Args:
            base_path: 모델 저장 기본 경로
            device: 사용할 디바이스
        """
        self.base_path = Path(base_path)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 저장소 경로들
        self.models_dir = self.base_path / "active"
        self.archive_dir = self.base_path / "archive"
        self.backup_dir = self.base_path / "backup"
        self.temp_dir = self.base_path / "temp"
        
        # 디렉토리 생성
        self._create_directories()
        
        # 모델 레지스트리 파일
        self.registry_file = self.base_path / "model_registry.json"
        self.model_registry = self._load_registry()
        
        # 모델 관리 설정
        self.management_config = {
            'max_models_per_type': 5,
            'auto_backup': True,
            'backup_interval': 24 * 60 * 60,  # 24시간
            'cleanup_old_versions': True,
            'max_archive_size': 10 * 1024 * 1024 * 1024,  # 10GB
            'checksum_verification': True
        }
        
        logger.info(f"ModelManagementService initialized at {self.base_path}")
    
    def _create_directories(self):
        """필요한 디렉토리들을 생성"""
        try:
            for directory in [self.models_dir, self.archive_dir, self.backup_dir, self.temp_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            logger.info("모델 관리 디렉토리 생성 완료")
        except Exception as e:
            logger.error(f"디렉토리 생성 중 오류 발생: {e}")
    
    def _load_registry(self) -> Dict[str, ModelInfo]:
        """모델 레지스트리 로드"""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
                    return {k: ModelInfo(**v) for k, v in registry_data.items()}
            return {}
        except Exception as e:
            logger.error(f"모델 레지스트리 로드 중 오류 발생: {e}")
            return {}
    
    def _save_registry(self):
        """모델 레지스트리 저장"""
        try:
            registry_data = {k: asdict(v) for k, v in self.model_registry.items()}
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            logger.debug("모델 레지스트리 저장 완료")
        except Exception as e:
            logger.error(f"모델 레지스트리 저장 중 오류 발생: {e}")
    
    def register_model(self, model: nn.Module, model_name: str, model_type: str, 
                      checkpoint_path: str, version: str = "1.0.0", 
                      description: str = "") -> str:
        """
        새로운 모델을 레지스트리에 등록
        
        Args:
            model: 등록할 모델
            model_name: 모델 이름
            model_type: 모델 타입
            checkpoint_path: 체크포인트 파일 경로
            version: 모델 버전
            description: 모델 설명
            
        Returns:
            등록된 모델의 고유 ID
        """
        try:
            # 모델 ID 생성
            model_id = f"{model_type}_{model_name}_{version}"
            
            # 체크포인트 파일 검증
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
            
            # 모델 크기 계산
            model_size = os.path.getsize(checkpoint_path)
            
            # 체크섬 계산
            checksum = self._calculate_checksum(checkpoint_path)
            
            # 현재 시간
            current_time = datetime.now().isoformat()
            
            # 모델 정보 생성
            model_info = ModelInfo(
                model_name=model_name,
                model_type=model_type,
                version=version,
                checkpoint_path=checkpoint_path,
                model_size=model_size,
                creation_date=current_time,
                last_modified=current_time,
                checksum=checksum,
                device=str(self.device),
                status='active',
                performance_metrics={},
                dependencies=self._get_model_dependencies(model),
                description=description
            )
            
            # 레지스트리에 추가
            self.model_registry[model_id] = model_info
            
            # 레지스트리 저장
            self._save_registry()
            
            logger.info(f"모델 등록 완료: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"모델 등록 중 오류 발생: {e}")
            raise
    
    def _calculate_checksum(self, file_path: str) -> str:
        """파일 체크섬 계산"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"체크섬 계산 중 오류 발생: {e}")
            return ""
    
    def _get_model_dependencies(self, model: nn.Module) -> List[str]:
        """모델의 의존성 정보 추출"""
        try:
            dependencies = []
            
            # PyTorch 버전
            dependencies.append(f"torch=={torch.__version__}")
            
            # CUDA 버전 (사용 가능한 경우)
            if torch.cuda.is_available():
                dependencies.append(f"cuda=={torch.version.cuda}")
            
            # 모델의 특별한 의존성들
            if hasattr(model, 'dependencies'):
                dependencies.extend(model.dependencies)
            
            return dependencies
        except Exception as e:
            logger.error(f"모델 의존성 추출 중 오류 발생: {e}")
            return []
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """모델 정보 조회"""
        try:
            return self.model_registry.get(model_id)
        except Exception as e:
            logger.error(f"모델 정보 조회 중 오류 발생: {e}")
            return None
    
    def list_models(self, model_type: Optional[str] = None, status: Optional[str] = None) -> List[ModelInfo]:
        """모델 목록 조회"""
        try:
            models = list(self.model_registry.values())
            
            if model_type:
                models = [m for m in models if m.model_type == model_type]
            
            if status:
                models = [m for m in models if m.status == status]
            
            return sorted(models, key=lambda x: x.last_modified, reverse=True)
        except Exception as e:
            logger.error(f"모델 목록 조회 중 오류 발생: {e}")
            return []
    
    def update_model_status(self, model_id: str, status: str) -> bool:
        """모델 상태 업데이트"""
        try:
            if model_id not in self.model_registry:
                logger.warning(f"모델을 찾을 수 없습니다: {model_id}")
                return False
            
            self.model_registry[model_id].status = status
            self.model_registry[model_id].last_modified = datetime.now().isoformat()
            
            self._save_registry()
            logger.info(f"모델 상태 업데이트 완료: {model_id} -> {status}")
            return True
            
        except Exception as e:
            logger.error(f"모델 상태 업데이트 중 오류 발생: {e}")
            return False
    
    def backup_model(self, model_id: str) -> bool:
        """모델 백업"""
        try:
            if model_id not in self.model_registry:
                logger.warning(f"모델을 찾을 수 없습니다: {model_id}")
                return False
            
            model_info = self.model_registry[model_id]
            
            # 백업 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{model_id}_{timestamp}.pth"
            backup_path = self.backup_dir / backup_filename
            
            # 체크포인트 파일 복사
            shutil.copy2(model_info.checkpoint_path, backup_path)
            
            # 백업 정보 저장
            backup_info = {
                'model_id': model_id,
                'backup_path': str(backup_path),
                'backup_date': timestamp,
                'original_checksum': model_info.checksum,
                'backup_checksum': self._calculate_checksum(str(backup_path))
            }
            
            backup_info_path = self.backup_dir / f"{model_id}_{timestamp}_info.json"
            with open(backup_info_path, 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"모델 백업 완료: {model_id} -> {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"모델 백업 중 오류 발생: {e}")
            return False
    
    def restore_model(self, model_id: str, backup_timestamp: str) -> bool:
        """모델 복원"""
        try:
            if model_id not in self.model_registry:
                logger.warning(f"모델을 찾을 수 없습니다: {model_id}")
                return False
            
            # 백업 파일 찾기
            backup_pattern = f"{model_id}_{backup_timestamp}"
            backup_files = list(self.backup_dir.glob(f"{backup_pattern}*.pth"))
            
            if not backup_files:
                logger.warning(f"백업 파일을 찾을 수 없습니다: {backup_pattern}")
                return False
            
            backup_path = backup_files[0]
            
            # 백업 정보 파일 찾기
            info_files = list(self.backup_dir.glob(f"{backup_pattern}*_info.json"))
            if info_files:
                with open(info_files[0], 'r', encoding='utf-8') as f:
                    backup_info = json.load(f)
                
                # 체크섬 검증
                if self.management_config['checksum_verification']:
                    current_checksum = self._calculate_checksum(str(backup_path))
                    if current_checksum != backup_info['backup_checksum']:
                        logger.error("백업 파일 체크섬이 일치하지 않습니다")
                        return False
            
            # 현재 체크포인트 백업
            model_info = self.model_registry[model_id]
            current_backup = f"{model_info.checkpoint_path}.backup"
            shutil.copy2(model_info.checkpoint_path, current_backup)
            
            # 백업에서 복원
            shutil.copy2(backup_path, model_info.checkpoint_path)
            
            # 레지스트리 업데이트
            model_info.last_modified = datetime.now().isoformat()
            model_info.checksum = self._calculate_checksum(model_info.checkpoint_path)
            
            self._save_registry()
            
            logger.info(f"모델 복원 완료: {model_id} from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"모델 복원 중 오류 발생: {e}")
            return False
    
    def deploy_model(self, model_id: str, deployment_config: DeploymentConfig) -> bool:
        """모델 배포"""
        try:
            if model_id not in self.model_registry:
                logger.warning(f"모델을 찾을 수 없습니다: {model_id}")
                return False
            
            model_info = self.model_registry[model_id]
            
            # 배포 디렉토리 생성
            deploy_dir = self.temp_dir / f"deploy_{model_id}_{int(time.time())}"
            deploy_dir.mkdir(parents=True, exist_ok=True)
            
            # 모델 로드 및 최적화
            checkpoint = torch.load(model_info.checkpoint_path, map_location='cpu')
            model = checkpoint.get('model', checkpoint)
            
            # 최적화 적용
            if deployment_config.quantization:
                model = self._quantize_model(model)
            
            if deployment_config.pruning:
                model = self._prune_model(model)
            
            # 최적화된 모델 저장
            optimized_path = deploy_dir / "optimized_model.pth"
            torch.save({
                'model': model,
                'deployment_config': asdict(deployment_config),
                'original_model_id': model_id,
                'deployment_date': datetime.now().isoformat()
            }, optimized_path)
            
            # 배포 정보 저장
            deployment_info = {
                'model_id': model_id,
                'deployment_config': asdict(deployment_config),
                'deployment_path': str(optimized_path),
                'deployment_date': datetime.now().isoformat(),
                'status': 'deployed'
            }
            
            deployment_info_path = deploy_dir / "deployment_info.json"
            with open(deployment_info_path, 'w', encoding='utf-8') as f:
                json.dump(deployment_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"모델 배포 완료: {model_id} -> {deploy_dir}")
            return True
            
        except Exception as e:
            logger.error(f"모델 배포 중 오류 발생: {e}")
            return False
    
    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """모델 양자화 (간단한 구현)"""
        try:
            # 동적 양자화 적용
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
            return quantized_model
        except Exception as e:
            logger.warning(f"모델 양자화 중 오류 발생, 원본 모델 반환: {e}")
            return model
    
    def _prune_model(self, model: nn.Module) -> nn.Module:
        """모델 프루닝 (간단한 구현)"""
        try:
            # 간단한 가중치 프루닝
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    # 가중치의 절댓값이 작은 것들을 0으로 설정
                    weight = module.weight.data
                    threshold = torch.quantile(torch.abs(weight), 0.1)  # 하위 10% 제거
                    mask = torch.abs(weight) > threshold
                    module.weight.data = weight * mask.float()
            
            return model
        except Exception as e:
            logger.warning(f"모델 프루닝 중 오류 발생, 원본 모델 반환: {e}")
            return model
    
    def cleanup_old_versions(self, model_type: str, keep_count: int = 3) -> int:
        """오래된 모델 버전 정리"""
        try:
            if not self.management_config['cleanup_old_versions']:
                return 0
            
            # 해당 타입의 모델들 조회
            models = [m for m in self.model_registry.values() if m.model_type == model_type]
            
            if len(models) <= keep_count:
                return 0
            
            # 수정일 기준으로 정렬하고 오래된 것들 제거
            models.sort(key=lambda x: x.last_modified)
            models_to_remove = models[:-keep_count]
            
            removed_count = 0
            for model in models_to_remove:
                model_id = f"{model.model_type}_{model.model_name}_{model.version}"
                
                # 아카이브로 이동
                archive_path = self.archive_dir / f"{model_id}.pth"
                if os.path.exists(model.checkpoint_path):
                    shutil.move(model.checkpoint_path, archive_path)
                
                # 레지스트리에서 제거
                del self.model_registry[model_id]
                removed_count += 1
            
            self._save_registry()
            logger.info(f"오래된 모델 버전 {removed_count}개 정리 완료")
            return removed_count
            
        except Exception as e:
            logger.error(f"오래된 모델 버전 정리 중 오류 발생: {e}")
            return 0
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """모델 통계 정보 반환"""
        try:
            total_models = len(self.model_registry)
            active_models = len([m for m in self.model_registry.values() if m.status == 'active'])
            inactive_models = len([m for m in self.model_registry.values() if m.status == 'inactive'])
            deprecated_models = len([m for m in self.model_registry.values() if m.status == 'deprecated'])
            
            total_size = sum(m.model_size for m in self.model_registry.values())
            
            # 타입별 통계
            type_stats = {}
            for model in self.model_registry.values():
                if model.model_type not in type_stats:
                    type_stats[model.model_type] = 0
                type_stats[model.model_type] += 1
            
            return {
                'total_models': total_models,
                'active_models': active_models,
                'inactive_models': inactive_models,
                'deprecated_models': deprecated_models,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'type_distribution': type_stats
            }
            
        except Exception as e:
            logger.error(f"모델 통계 생성 중 오류 발생: {e}")
            return {}
    
    def validate_model_integrity(self, model_id: str) -> bool:
        """모델 무결성 검증"""
        try:
            if model_id not in self.model_registry:
                return False
            
            model_info = self.model_registry[model_id]
            
            # 파일 존재 여부 확인
            if not os.path.exists(model_info.checkpoint_path):
                logger.error(f"체크포인트 파일이 존재하지 않습니다: {model_info.checkpoint_path}")
                return False
            
            # 체크섬 검증
            if self.management_config['checksum_verification']:
                current_checksum = self._calculate_checksum(model_info.checkpoint_path)
                if current_checksum != model_info.checksum:
                    logger.error(f"체크섬이 일치하지 않습니다: {model_id}")
                    return False
            
            # 모델 로드 테스트
            try:
                checkpoint = torch.load(model_info.checkpoint_path, map_location='cpu')
                logger.info(f"모델 무결성 검증 성공: {model_id}")
                return True
            except Exception as e:
                logger.error(f"모델 로드 테스트 실패: {model_id}, 오류: {e}")
                return False
                
        except Exception as e:
            logger.error(f"모델 무결성 검증 중 오류 발생: {e}")
            return False

class PostProcessingModelManagementService:
    """후처리 모델 관리 서비스"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # MPS 디바이스 확인
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.logger.info(f"🎯 Post Processing 모델 관리 서비스 초기화 (디바이스: {self.device})")
        
        # 기본 모델 관리 서비스 초기화
        base_path = self.config.get('base_path', 'post_processing_models')
        self.model_service = ModelManagementService(base_path=base_path, device=self.device)
        
        # 후처리 모델 설정
        self.post_processing_config = {
            'model_types': ['quality_enhancer', 'artifact_remover', 'resolution_enhancer', 'color_corrector', 'final_optimizer'],
            'auto_update': True,
            'version_control': True,
            'performance_tracking': True
        }
        
        # 설정 병합
        self.post_processing_config.update(self.config)
        
        # 후처리 모델 레지스트리
        self.post_processing_models = {}
        
        self.logger.info("✅ Post Processing 모델 관리 서비스 초기화 완료")
    
    def register_post_processing_model(self, model_name: str, model_type: str, 
                                     checkpoint_path: str, version: str = "1.0.0") -> str:
        """
        후처리 모델을 등록합니다.
        
        Args:
            model_name: 모델 이름
            model_type: 모델 타입
            checkpoint_path: 체크포인트 경로
            version: 모델 버전
            
        Returns:
            등록된 모델 ID
        """
        try:
            if model_type not in self.post_processing_config['model_types']:
                raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
            
            # 모델 정보 생성
            model_info = ModelInfo(
                model_name=model_name,
                model_type=model_type,
                version=version,
                checkpoint_path=checkpoint_path,
                model_size=os.path.getsize(checkpoint_path),
                creation_date=datetime.now().isoformat(),
                last_modified=datetime.now().isoformat(),
                checksum=self.model_service._calculate_checksum(checkpoint_path),
                device=str(self.device),
                status='active',
                performance_metrics={},
                dependencies=[],
                description=f"Post Processing {model_type} model"
            )
            
            # 모델 등록
            model_id = self.model_service.register_model(model_info)
            
            # 후처리 모델 레지스트리에 추가
            self.post_processing_models[model_id] = model_info
            
            self.logger.info(f"후처리 모델 등록 완료: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"후처리 모델 등록 실패: {e}")
            raise
    
    def get_post_processing_model(self, model_type: str, version: str = None) -> Optional[ModelInfo]:
        """
        후처리 모델을 조회합니다.
        
        Args:
            model_type: 모델 타입
            version: 모델 버전 (None이면 최신 버전)
            
        Returns:
            모델 정보 또는 None
        """
        try:
            # 해당 타입의 모델들 조회
            models = [m for m in self.post_processing_models.values() if m.model_type == model_type]
            
            if not models:
                return None
            
            if version is None:
                # 최신 버전 반환
                latest_model = max(models, key=lambda x: x.version)
                return latest_model
            else:
                # 특정 버전 반환
                for model in models:
                    if model.version == version:
                        return model
                return None
                
        except Exception as e:
            self.logger.error(f"후처리 모델 조회 실패: {e}")
            return None
    
    def update_post_processing_model(self, model_id: str, new_checkpoint_path: str, 
                                   new_version: str = None) -> bool:
        """
        후처리 모델을 업데이트합니다.
        
        Args:
            model_id: 모델 ID
            new_checkpoint_path: 새로운 체크포인트 경로
            new_version: 새로운 버전
            
        Returns:
            업데이트 성공 여부
        """
        try:
            if model_id not in self.post_processing_models:
                raise ValueError(f"모델을 찾을 수 없습니다: {model_id}")
            
            # 기존 모델 정보
            old_model = self.post_processing_models[model_id]
            
            # 새 버전 결정
            if new_version is None:
                # 기존 버전에서 패치 버전 증가
                version_parts = old_model.version.split('.')
                if len(version_parts) >= 3:
                    version_parts[2] = str(int(version_parts[2]) + 1)
                    new_version = '.'.join(version_parts)
                else:
                    new_version = old_model.version + ".1"
            
            # 새 모델 정보 생성
            new_model_info = ModelInfo(
                model_name=old_model.model_name,
                model_type=old_model.model_type,
                version=new_version,
                checkpoint_path=new_checkpoint_path,
                model_size=os.path.getsize(new_checkpoint_path),
                creation_date=datetime.now().isoformat(),
                last_modified=datetime.now().isoformat(),
                checksum=self.model_service._calculate_checksum(new_checkpoint_path),
                device=str(self.device),
                status='active',
                performance_metrics=old_model.performance_metrics.copy(),
                dependencies=old_model.dependencies.copy(),
                description=old_model.description
            )
            
            # 기존 모델 비활성화
            old_model.status = 'inactive'
            
            # 새 모델 등록
            new_model_id = self.register_post_processing_model(
                new_model_info.model_name,
                new_model_info.model_type,
                new_model_info.checkpoint_path,
                new_model_info.version
            )
            
            self.logger.info(f"후처리 모델 업데이트 완료: {model_id} -> {new_model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"후처리 모델 업데이트 실패: {e}")
            return False
    
    def get_post_processing_model_stats(self) -> Dict[str, Any]:
        """후처리 모델 통계를 반환합니다."""
        try:
            # 기본 모델 통계
            base_stats = self.model_service.get_model_statistics()
            
            # 후처리 모델 통계
            post_processing_stats = {
                'total_post_processing_models': len(self.post_processing_models),
                'models_by_type': {},
                'active_models': len([m for m in self.post_processing_models.values() if m.status == 'active']),
                'inactive_models': len([m for m in self.post_processing_models.values() if m.status == 'inactive'])
            }
            
            # 타입별 모델 수 계산
            for model in self.post_processing_models.values():
                if model.model_type not in post_processing_stats['models_by_type']:
                    post_processing_stats['models_by_type'][model.model_type] = 0
                post_processing_stats['models_by_type'][model.model_type] += 1
            
            return {
                **base_stats,
                'post_processing': post_processing_stats,
                'device': str(self.device),
                'config': self.post_processing_config
            }
            
        except Exception as e:
            self.logger.error(f"후처리 모델 통계 조회 실패: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """리소스 정리를 수행합니다."""
        try:
            # 모델 서비스 정리
            self.model_service.cleanup()
            
            # 후처리 모델 레지스트리 정리
            self.post_processing_models.clear()
            
            self.logger.info("Post Processing 모델 관리 서비스 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"리소스 정리 실패: {e}")

# 사용 예시
if __name__ == "__main__":
    # 설정
    config = {
        'base_path': 'post_processing_models',
        'auto_update': True,
        'version_control': True,
        'performance_tracking': True
    }
    
    # Post Processing 모델 관리 서비스 초기화
    model_service = PostProcessingModelManagementService(config)
    
    # 모델 등록 예시
    try:
        model_id = model_service.register_post_processing_model(
            model_name="quality_enhancer_v1",
            model_type="quality_enhancer",
            checkpoint_path="/path/to/checkpoint.pth",
            version="1.0.0"
        )
        print(f"모델 등록 성공: {model_id}")
        
        # 모델 조회
        model_info = model_service.get_post_processing_model("quality_enhancer")
        if model_info:
            print(f"모델 정보: {model_info.model_name} v{model_info.version}")
        
        # 모델 통계
        stats = model_service.get_post_processing_model_stats()
        print(f"모델 통계: {stats}")
        
    except Exception as e:
        print(f"모델 등록 실패: {e}")
    
    # 리소스 정리
    model_service.cleanup()
