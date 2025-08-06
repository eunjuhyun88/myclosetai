"""
🔥 Graphonomy 체크포인트 통합 시스템
모든 체크포인트 관련 로직을 Graphonomy 기준으로 통합
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ResNetGraphonomyModel(nn.Module):
    """ResNet 기반 Graphonomy 모델"""
    def __init__(self, num_classes=20):
        super().__init__()
        from .graphonomy_models import ResNet101Backbone
        self.backbone = ResNet101Backbone()
        self.classifier = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.edge_head = nn.Conv2d(2048, 1, kernel_size=1)
    
    def forward(self, x):
        features = self.backbone(x)
        parsing = self.classifier(features['layer4'])
        edge = self.edge_head(features['layer4'])
        return {'parsing': parsing, 'edge': edge}


class SimpleGraphonomyModel(nn.Module):
    """단순한 Graphonomy 모델"""
    def __init__(self, num_classes=20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        features = self.features(x)
        parsing = self.classifier(features)
        return {'parsing': parsing}


class FallbackGraphonomyModel(nn.Module):
    """폴백 모델"""
    def __init__(self, num_classes=20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        features = self.features(x)
        parsing = self.classifier(features)
        return {'parsing': parsing}


class GraphonomyCheckpointAnalyzer:
    """Graphonomy 체크포인트 분석기"""
    
    def __init__(self):
        self.logger = logger
    
    def analyze_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """체크포인트 구조 분석"""
        try:
            checkpoint_keys = list(checkpoint_data.keys())
            
            # 체크포인트 타입 분석
            checkpoint_type = self._detect_checkpoint_type(checkpoint_keys)
            
            # 키 매핑 규칙 생성
            key_mappings = self._generate_key_mappings(checkpoint_type)
            
            return {
                'type': checkpoint_type,
                'keys': checkpoint_keys,
                'key_count': len(checkpoint_keys),
                'mappings': key_mappings,
                'has_resnet': any('resnet' in key.lower() or 'layer' in key.lower() for key in checkpoint_keys),
                'has_aspp': any('aspp' in key.lower() for key in checkpoint_keys),
                'has_classifier': any('classifier' in key.lower() for key in checkpoint_keys),
                'has_self_correction': any('self_correction' in key.lower() for key in checkpoint_keys),
                'has_progressive': any('progressive' in key.lower() for key in checkpoint_keys)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ 체크포인트 분석 실패: {e}")
            return {'type': 'unknown', 'keys': [], 'mappings': {}}
    
    def _detect_checkpoint_type(self, checkpoint_keys: List[str]) -> str:
        """체크포인트 타입 감지"""
        if any('module.' in key for key in checkpoint_keys):
            return 'graphonomy_schp'  # Self-Correction Human Parsing
        elif any('backbone.' in key for key in checkpoint_keys):
            return 'graphonomy_resnet'
        elif any('aspp.' in key for key in checkpoint_keys):
            return 'graphonomy_aspp'
        elif any('self_correction' in key for key in checkpoint_keys):
            return 'graphonomy_advanced'
        else:
            return 'graphonomy_standard'
    
    def _generate_key_mappings(self, checkpoint_type: str) -> Dict[str, str]:
        """키 매핑 규칙 생성"""
        base_mappings = {
            # ResNet 백본 매핑
            'module.conv1.weight': 'conv1.weight',
            'module.bn1.weight': 'bn1.weight',
            'module.bn1.bias': 'bn1.bias',
            'module.bn1.running_mean': 'bn1.running_mean',
            'module.bn1.running_var': 'bn1.running_var',
            
            # Layer 매핑
            'module.layer1.': 'layer1.',
            'module.layer2.': 'layer2.',
            'module.layer3.': 'layer3.',
            'module.layer4.': 'layer4.',
            
            # ASPP 매핑
            'module.aspp.convs.0.weight': 'aspp.conv1x1.0.weight',
            'module.aspp.convs.0.bias': 'aspp.conv1x1.0.bias',
            'module.aspp.convs.1.weight': 'aspp.atrous_convs.0.0.weight',
            'module.aspp.convs.1.bias': 'aspp.atrous_convs.0.0.bias',
            'module.aspp.convs.2.weight': 'aspp.atrous_convs.1.0.weight',
            'module.aspp.convs.2.bias': 'aspp.atrous_convs.1.0.bias',
            'module.aspp.convs.3.weight': 'aspp.atrous_convs.2.0.weight',
            'module.aspp.convs.3.bias': 'aspp.atrous_convs.2.0.bias',
            'module.aspp.convs.4.weight': 'aspp.conv_global.weight',
            'module.aspp.convs.4.bias': 'aspp.conv_global.bias',
            'module.aspp.convs.5.weight': 'aspp.conv_out.weight',
            'module.aspp.convs.5.bias': 'aspp.conv_out.bias',
            
            # 분류기 매핑
            'module.classifier.weight': 'classifier.weight',
            'module.classifier.bias': 'classifier.bias',
            
            # Edge Detection 매핑
            'module.edge_head.weight': 'edge_head.weight',
            'module.edge_head.bias': 'edge_head.bias'
        }
        
        # 체크포인트 타입별 추가 매핑
        if checkpoint_type == 'graphonomy_schp':
            base_mappings.update({
                'module.self_correction.': 'self_correction.',
                'module.progressive_parsing.': 'progressive_parsing.',
                'module.iterative_refinement.': 'iterative_refinement.'
            })
        
        return base_mappings


class GraphonomyModelFactory:
    """Graphonomy 모델 팩토리"""
    
    def __init__(self):
        self.logger = logger
    
    def create_model(self, checkpoint_info: Dict[str, Any]) -> nn.Module:
        """체크포인트 정보에 따라 적절한 모델 생성"""
        try:
            if checkpoint_info['has_resnet'] and checkpoint_info['has_aspp']:
                return self._create_full_graphonomy_model()
            elif checkpoint_info['has_resnet']:
                return self._create_resnet_graphonomy_model()
            else:
                return self._create_simple_graphonomy_model()
                
        except Exception as e:
            self.logger.warning(f"⚠️ 모델 생성 실패: {e}")
            return self._create_fallback_model()
    
    def _create_full_graphonomy_model(self) -> nn.Module:
        """완전한 Graphonomy 모델 (ResNet + ASPP + 모든 모듈)"""
        from .graphonomy_models import AdvancedGraphonomyResNetASPP
        return AdvancedGraphonomyResNetASPP(num_classes=20)
    
    def _create_resnet_graphonomy_model(self) -> nn.Module:
        """ResNet 기반 Graphonomy 모델"""
        return ResNetGraphonomyModel(num_classes=20)
    
    def _create_simple_graphonomy_model(self) -> nn.Module:
        """단순한 Graphonomy 모델"""
        return SimpleGraphonomyModel(num_classes=20)
    
    def _create_fallback_model(self) -> nn.Module:
        """폴백 모델"""
        return FallbackGraphonomyModel(num_classes=20)


class GraphonomyCheckpointLoader:
    """Graphonomy 체크포인트 로더"""
    
    def __init__(self):
        self.logger = logger
        self.analyzer = GraphonomyCheckpointAnalyzer()
        self.model_factory = GraphonomyModelFactory()
    
    def load_checkpoint(self, checkpoint_path: Path, device: str = 'cpu') -> Optional[nn.Module]:
        """체크포인트 로딩"""
        try:
            self.logger.info(f"🔄 Graphonomy 체크포인트 로딩: {checkpoint_path}")
            
            # 1. 체크포인트 파일 로딩
            checkpoint_data = self._load_checkpoint_file(checkpoint_path)
            if checkpoint_data is None:
                return None
            
            # 2. 체크포인트 분석
            checkpoint_info = self.analyzer.analyze_checkpoint(checkpoint_data)
            
            # 3. 모델 생성
            model = self.model_factory.create_model(checkpoint_info)
            
            # 4. 가중치 로딩
            self._load_weights(checkpoint_data, model, checkpoint_info)
            
            # 5. 모델 설정
            model.to(device)
            model.eval()
            
            self.logger.info("✅ Graphonomy 체크포인트 로딩 완료")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ 체크포인트 로딩 실패: {e}")
            return None
    
    def _load_checkpoint_file(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """체크포인트 파일 로딩"""
        try:
            if not checkpoint_path.exists():
                self.logger.error(f"❌ 체크포인트 파일 없음: {checkpoint_path}")
                return None
            
            # PyTorch 체크포인트 로딩
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 체크포인트 구조 정규화
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    return checkpoint['state_dict']
                elif 'model' in checkpoint:
                    return checkpoint['model']
                else:
                    return checkpoint
            else:
                return checkpoint.state_dict()
                
        except Exception as e:
            self.logger.error(f"❌ 체크포인트 파일 로딩 실패: {e}")
            return None
    
    def _load_weights(self, checkpoint_data: Dict[str, Any], model: nn.Module, checkpoint_info: Dict[str, Any]):
        """가중치 로딩"""
        try:
            model_state_dict = model.state_dict()
            mapped_dict = {}
            
            key_mappings = checkpoint_info['mappings']
            
            # 키 매핑 실행
            for checkpoint_key, value in checkpoint_data.items():
                mapped_key = self._map_key(checkpoint_key, model_state_dict, key_mappings)
                if mapped_key:
                    mapped_dict[mapped_key] = value
            
            # 매핑된 가중치 로드
            if mapped_dict:
                missing_keys, unexpected_keys = model.load_state_dict(mapped_dict, strict=False)
                self.logger.info(f"✅ 가중치 로딩 성공: {len(mapped_dict)}개 키")
                
                if missing_keys:
                    self.logger.warning(f"⚠️ 누락된 키: {len(missing_keys)}개")
                if unexpected_keys:
                    self.logger.warning(f"⚠️ 예상치 못한 키: {len(unexpected_keys)}개")
            else:
                self.logger.warning("⚠️ 매핑된 키가 없음 - 랜덤 초기화 사용")
                
        except Exception as e:
            self.logger.error(f"❌ 가중치 로딩 실패: {e}")
    
    def _map_key(self, checkpoint_key: str, model_state_dict: Dict[str, Any], key_mappings: Dict[str, str]) -> Optional[str]:
        """키 매핑"""
        # 1. 직접 매핑
        if checkpoint_key in key_mappings:
            model_key = key_mappings[checkpoint_key]
            if model_key in model_state_dict:
                return model_key
        
        # 2. module. 접두사 제거
        clean_key = checkpoint_key.replace('module.', '')
        if clean_key in model_state_dict:
            return clean_key
        
        # 3. 패턴 매핑
        for pattern, replacement in key_mappings.items():
            if pattern in checkpoint_key:
                mapped_key = checkpoint_key.replace(pattern, replacement)
                if mapped_key in model_state_dict:
                    return mapped_key
        
        # 4. 직접 매칭
        if checkpoint_key in model_state_dict:
            return checkpoint_key
        
        return None


class UnifiedGraphonomyCheckpointSystem:
    """통합된 Graphonomy 체크포인트 시스템"""
    
    def __init__(self):
        self.logger = logger
        self.loader = GraphonomyCheckpointLoader()
    
    def create_model_from_checkpoint(self, checkpoint_data: Dict[str, Any], device: str = 'cpu') -> Optional[nn.Module]:
        """체크포인트에서 모델 생성 (기존 메서드 호환성)"""
        try:
            # 체크포인트 분석
            analyzer = GraphonomyCheckpointAnalyzer()
            checkpoint_info = analyzer.analyze_checkpoint(checkpoint_data)
            
            # 모델 생성
            model_factory = GraphonomyModelFactory()
            model = model_factory.create_model(checkpoint_info)
            
            # 가중치 로딩
            self.loader._load_weights(checkpoint_data, model, checkpoint_info)
            
            # 모델 설정
            model.to(device)
            model.eval()
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ 체크포인트에서 모델 생성 실패: {e}")
            return None
    
    def load_model_from_file(self, checkpoint_path: Path, device: str = 'cpu') -> Optional[nn.Module]:
        """파일에서 모델 로딩"""
        return self.loader.load_checkpoint(checkpoint_path, device)
    
    def get_checkpoint_info(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """체크포인트 정보 반환"""
        analyzer = GraphonomyCheckpointAnalyzer()
        return analyzer.analyze_checkpoint(checkpoint_data) 