# backend/app/ai_pipeline/utils/neural_architecture_manager.py
"""
🧠 Neural Architecture Manager - 신경망 수준 모델 관리 시스템
================================================================================
✅ Neural Architecture Pattern - 모든 모델을 nn.Module로 통합
✅ Memory-Efficient Loading - 레이어별 점진적 로딩
✅ Dynamic Computation Graph - AutoGrad 완전 활용
✅ Hardware-Aware Optimization - M3 Max MPS 최적화
✅ Gradient-Free Inference - 추론 시 메모리 최적화

핵심 신경망 설계 원칙:
1. 모든 모델을 nn.Module로 통합
2. 레이어별 점진적 로딩으로 메모리 효율성 극대화
3. AutoGrad 기반 동적 그래프 완전 활용
4. M3 Max 하드웨어 특화 최적화
5. 추론 시 gradient-free 모드로 메모리 최적화
================================================================================
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import gc

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==============================================
# 🔥 1. 신경망 아키텍처 패턴 정의
# ==============================================

class NeuralArchitectureType(Enum):
    """신경망 아키텍처 타입"""
    RESNET = "resnet"
    VIT = "vision_transformer"
    CONV_NEXT = "convnext"
    EFFICIENT_NET = "efficientnet"
    SWIN_TRANSFORMER = "swin_transformer"
    CUSTOM = "custom"

@dataclass
class LayerConfig:
    """레이어 설정"""
    layer_type: str
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    activation: str = "relu"
    normalization: str = "batch_norm"
    dropout: float = 0.0
    memory_priority: int = 1  # 1-10, 높을수록 우선 로딩

@dataclass
class NeuralArchitectureConfig:
    """신경망 아키텍처 설정"""
    architecture_type: NeuralArchitectureType
    input_size: Tuple[int, int] = (512, 512)
    num_classes: int = 20
    layers: List[LayerConfig] = field(default_factory=list)
    memory_limit_gb: float = 8.0
    enable_gradient_checkpointing: bool = True
    precision: str = "float32"  # float16, float32
    device: str = "auto"

# ==============================================
# 🔥 2. 레이어별 점진적 로딩 시스템
# ==============================================

class ProgressiveLayerLoader:
    """레이어별 점진적 로딩 시스템"""
    
    def __init__(self, config: NeuralArchitectureConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ProgressiveLayerLoader")
        self.loaded_layers = {}
        self.layer_memory_usage = {}
        self.loading_queue = []
        
        # 메모리 모니터링
        self.memory_threshold = config.memory_limit_gb * 0.8
        self.current_memory_usage = 0.0
        
        # 스레드 안전성
        self._lock = threading.RLock()
        
    def add_layer(self, layer_name: str, layer_config: LayerConfig, 
                  layer_module: nn.Module) -> bool:
        """레이어 추가 (메모리 우선순위 기반)"""
        try:
            with self._lock:
                # 메모리 사용량 추정
                estimated_memory = self._estimate_layer_memory(layer_config)
                
                # 우선순위 기반 큐에 추가
                priority_item = {
                    'name': layer_name,
                    'config': layer_config,
                    'module': layer_module,
                    'memory': estimated_memory,
                    'priority': layer_config.memory_priority
                }
                
                self.loading_queue.append(priority_item)
                
                # 우선순위 정렬
                self.loading_queue.sort(key=lambda x: x['priority'], reverse=True)
                
                self.logger.debug(f"✅ 레이어 추가: {layer_name} (메모리: {estimated_memory:.2f}MB)")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ 레이어 추가 실패: {e}")
            return False
    
    def load_layers_progressively(self, target_memory_gb: float = None) -> Dict[str, bool]:
        """점진적 레이어 로딩"""
        if target_memory_gb is None:
            target_memory_gb = self.memory_threshold
            
        try:
            with self._lock:
                results = {}
                current_memory = self.current_memory_usage
                
                for item in self.loading_queue:
                    layer_name = item['name']
                    
                    # 이미 로드된 레이어 스킵
                    if layer_name in self.loaded_layers:
                        results[layer_name] = True
                        continue
                    
                    # 메모리 한계 확인
                    if current_memory + item['memory'] > target_memory_gb * 1024:  # GB to MB
                        self.logger.debug(f"⚠️ 메모리 한계 도달: {layer_name} 로딩 건너뜀")
                        results[layer_name] = False
                        continue
                    
                    # 레이어 로딩
                    try:
                        self._load_single_layer(item)
                        self.loaded_layers[layer_name] = item['module']
                        self.layer_memory_usage[layer_name] = item['memory']
                        current_memory += item['memory']
                        results[layer_name] = True
                        
                        self.logger.info(f"✅ 레이어 로딩 완료: {layer_name} ({item['memory']:.2f}MB)")
                        
                    except Exception as e:
                        self.logger.error(f"❌ 레이어 로딩 실패: {layer_name} - {e}")
                        results[layer_name] = False
                
                self.current_memory_usage = current_memory
                return results
                
        except Exception as e:
            self.logger.error(f"❌ 점진적 로딩 실패: {e}")
            return {}
    
    def _load_single_layer(self, layer_item: Dict[str, Any]):
        """단일 레이어 로딩"""
        layer_name = layer_item['name']
        layer_module = layer_item['module']
        
        # 디바이스 이동
        device = self._get_optimal_device()
        layer_module.to(device)
        
        # 정밀도 설정
        if self.config.precision == "float16":
            layer_module.half()
        
        # 평가 모드 설정
        layer_module.eval()
        
        # Gradient checkpointing (메모리 절약)
        if self.config.enable_gradient_checkpointing:
            layer_module = torch.utils.checkpoint.checkpoint_wrapper(layer_module)
        
        self.logger.debug(f"✅ 단일 레이어 로딩 완료: {layer_name}")
    
    def _estimate_layer_memory(self, layer_config: LayerConfig) -> float:
        """레이어 메모리 사용량 추정 (MB)"""
        try:
            # 기본 메모리 계산
            base_memory = layer_config.in_channels * layer_config.out_channels * layer_config.kernel_size ** 2
            
            # 정밀도별 메모리 계산
            if self.config.precision == "float16":
                memory_bytes = base_memory * 2  # 2 bytes per parameter
            else:
                memory_bytes = base_memory * 4  # 4 bytes per parameter
            
            # 추가 메모리 (activation, gradients 등)
            total_memory = memory_bytes * 3  # 3x for activations and gradients
            
            return total_memory / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            self.logger.warning(f"⚠️ 메모리 추정 실패: {e}")
            return 100.0  # 기본값 100MB
    
    def _get_optimal_device(self) -> str:
        """최적 디바이스 선택"""
        if self.config.device == "auto":
            if TORCH_AVAILABLE:
                if torch.backends.mps.is_available():
                    return "mps"
                elif torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            else:
                return "cpu"
        return self.config.device

# ==============================================
# 🔥 3. 동적 계산 그래프 관리
# ==============================================

class DynamicComputationGraph:
    """AutoGrad 기반 동적 계산 그래프 관리"""
    
    def __init__(self, enable_autograd: bool = True):
        self.enable_autograd = enable_autograd
        self.logger = logging.getLogger(f"{__name__}.DynamicComputationGraph")
        self.graph_nodes = {}
        self.computation_paths = []
        
    def register_node(self, node_name: str, node_module: nn.Module, 
                     dependencies: List[str] = None) -> bool:
        """계산 그래프 노드 등록"""
        try:
            self.graph_nodes[node_name] = {
                'module': node_module,
                'dependencies': dependencies or [],
                'output_cache': None,
                'gradient_mode': self.enable_autograd
            }
            
            self.logger.debug(f"✅ 그래프 노드 등록: {node_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 그래프 노드 등록 실패: {e}")
            return False
    
    def execute_forward_pass(self, input_tensor: torch.Tensor, 
                           target_nodes: List[str] = None) -> Dict[str, torch.Tensor]:
        """순전파 실행 (동적 그래프)"""
        try:
            if target_nodes is None:
                target_nodes = list(self.graph_nodes.keys())
            
            results = {}
            
            for node_name in target_nodes:
                if node_name not in self.graph_nodes:
                    continue
                
                node_info = self.graph_nodes[node_name]
                module = node_info['module']
                
                # 의존성 확인
                if not self._check_dependencies(node_name, results):
                    continue
                
                # 입력 준비
                node_input = self._prepare_node_input(node_name, input_tensor, results)
                
                # 순전파 실행
                with torch.set_grad_enabled(node_info['gradient_mode']):
                    if node_info['gradient_mode']:
                        # Gradient 모드
                        output = module(node_input)
                    else:
                        # Gradient-free 모드 (메모리 절약)
                        with torch.no_grad():
                            output = module(node_input)
                
                # 결과 캐싱
                node_info['output_cache'] = output
                results[node_name] = output
                
                self.logger.debug(f"✅ 노드 실행 완료: {node_name}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ 순전파 실행 실패: {e}")
            return {}
    
    def execute_backward_pass(self, loss: torch.Tensor, 
                            target_nodes: List[str] = None) -> Dict[str, torch.Tensor]:
        """역전파 실행 (AutoGrad)"""
        if not self.enable_autograd:
            self.logger.warning("⚠️ AutoGrad 비활성화 - 역전파 건너뜀")
            return {}
        
        try:
            if target_nodes is None:
                target_nodes = list(self.graph_nodes.keys())
            
            gradients = {}
            
            # 역전파 실행
            loss.backward()
            
            for node_name in target_nodes:
                if node_name not in self.graph_nodes:
                    continue
                
                module = self.graph_nodes[node_name]['module']
                
                # 그래디언트 수집
                node_gradients = []
                for param in module.parameters():
                    if param.grad is not None:
                        node_gradients.append(param.grad.clone())
                
                if node_gradients:
                    gradients[node_name] = node_gradients
                
                self.logger.debug(f"✅ 그래디언트 수집 완료: {node_name}")
            
            return gradients
            
        except Exception as e:
            self.logger.error(f"❌ 역전파 실행 실패: {e}")
            return {}
    
    def _check_dependencies(self, node_name: str, 
                          available_results: Dict[str, torch.Tensor]) -> bool:
        """의존성 확인"""
        node_info = self.graph_nodes[node_name]
        dependencies = node_info['dependencies']
        
        for dep in dependencies:
            if dep not in available_results:
                self.logger.warning(f"⚠️ 의존성 누락: {node_name} -> {dep}")
                return False
        
        return True
    
    def _prepare_node_input(self, node_name: str, base_input: torch.Tensor,
                           available_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """노드 입력 준비"""
        node_info = self.graph_nodes[node_name]
        dependencies = node_info['dependencies']
        
        if not dependencies:
            return base_input
        
        # 의존성 출력들을 결합
        dependency_outputs = [available_results[dep] for dep in dependencies]
        
        if len(dependency_outputs) == 1:
            return dependency_outputs[0]
        else:
            # 여러 의존성이 있는 경우 concatenate
            return torch.cat(dependency_outputs, dim=1)

# ==============================================
# 🔥 4. 하드웨어 인식 최적화
# ==============================================

class HardwareAwareOptimizer:
    """M3 Max 하드웨어 특화 최적화"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._detect_optimal_device(device)
        self.logger = logging.getLogger(f"{__name__}.HardwareAwareOptimizer")
        self.is_m3_max = self._detect_m3_max()
        
        # M3 Max 특화 설정
        if self.is_m3_max:
            self._setup_m3_max_optimizations()
    
    def _detect_optimal_device(self, device: str) -> str:
        """최적 디바이스 감지"""
        if device == "auto":
            if TORCH_AVAILABLE:
                if torch.backends.mps.is_available():
                    return "mps"
                elif torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            else:
                return "cpu"
        return device
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 환경 감지"""
        try:
            import platform
            import psutil
            
            # Apple Silicon 확인
            is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
            
            # 메모리 용량 확인 (M3 Max는 128GB)
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # CPU 코어 수 확인 (M3 Max는 12코어 이상)
            cpu_cores = psutil.cpu_count(logical=False) or 4
            
            return is_apple_silicon and memory_gb >= 120 and cpu_cores >= 12
            
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 감지 실패: {e}")
            return False
    
    def _setup_m3_max_optimizations(self):
        """M3 Max 특화 최적화 설정"""
        try:
            if not self.is_m3_max:
                return
            
            self.logger.info("🍎 M3 Max 특화 최적화 설정")
            
            # 환경 변수 설정
            os.environ.update({
                'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',
                'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.0',
                'METAL_DEVICE_WRAPPER_TYPE': '1',
                'METAL_PERFORMANCE_SHADERS_ENABLED': '1',
                'PYTORCH_MPS_PREFER_METAL': '1',
                'PYTORCH_ENABLE_MPS_FALLBACK': '1'
            })
            
            # PyTorch 스레드 최적화
            if TORCH_AVAILABLE:
                torch.set_num_threads(min(16, os.cpu_count() or 8))
            
            self.logger.info("✅ M3 Max 최적화 설정 완료")
            
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 설정 실패: {e}")
    
    def optimize_model_for_device(self, model: nn.Module) -> nn.Module:
        """디바이스별 모델 최적화"""
        try:
            # 디바이스 이동
            model.to(self.device)
            
            # M3 Max 특화 최적화
            if self.is_m3_max and self.device == "mps":
                model = self._apply_m3_max_optimizations(model)
            
            # 평가 모드 설정
            model.eval()
            
            self.logger.info(f"✅ 모델 최적화 완료 (device: {self.device})")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ 모델 최적화 실패: {e}")
            return model
    
    def _apply_m3_max_optimizations(self, model: nn.Module) -> nn.Module:
        """M3 Max 특화 최적화 적용"""
        try:
            # Float32 강제 사용 (M3 Max에서 안정성)
            model = model.float()
            
            # 메모리 효율적인 연산 설정
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    # M3 Max에서 최적화된 컨볼루션 설정
                    module.padding_mode = 'zeros'
                elif isinstance(module, nn.BatchNorm2d):
                    # 배치 정규화 최적화
                    module.track_running_stats = True
            
            self.logger.debug("✅ M3 Max 특화 최적화 적용 완료")
            return model
            
        except Exception as e:
            self.logger.warning(f"⚠️ M3 Max 최적화 적용 실패: {e}")
            return model

# ==============================================
# 🔥 5. 메인 신경망 아키텍처 매니저
# ==============================================

class NeuralArchitectureManager:
    """신경망 아키텍처 통합 관리자"""
    
    def __init__(self, config: NeuralArchitectureConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.NeuralArchitectureManager")
        
        # 구성 요소 초기화
        self.layer_loader = ProgressiveLayerLoader(config)
        self.computation_graph = DynamicComputationGraph(enable_autograd=True)
        self.hardware_optimizer = HardwareAwareOptimizer(config.device)
        
        # 모델 상태
        self.model = None
        self.is_initialized = False
        self.memory_usage = 0.0
        
        # 초기화
        self._initialize_architecture()
    
    def _initialize_architecture(self):
        """아키텍처 초기화"""
        try:
            self.logger.info("🧠 신경망 아키텍처 초기화 시작")
            
            # 아키텍처 타입별 모델 생성
            self.model = self._create_architecture()
            
            # 하드웨어 최적화 적용
            self.model = self.hardware_optimizer.optimize_model_for_device(self.model)
            
            # 레이어별 점진적 로딩 설정
            self._setup_progressive_loading()
            
            # 계산 그래프 등록
            self._register_computation_graph()
            
            self.is_initialized = True
            self.logger.info("✅ 신경망 아키텍처 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 아키텍처 초기화 실패: {e}")
            self.is_initialized = False
    
    def _create_architecture(self) -> nn.Module:
        """아키텍처 타입별 모델 생성"""
        try:
            if self.config.architecture_type == NeuralArchitectureType.RESNET:
                return self._create_resnet_architecture()
            elif self.config.architecture_type == NeuralArchitectureType.VIT:
                return self._create_vision_transformer_architecture()
            elif self.config.architecture_type == NeuralArchitectureType.CONV_NEXT:
                return self._create_convnext_architecture()
            elif self.config.architecture_type == NeuralArchitectureType.EFFICIENT_NET:
                return self._create_efficientnet_architecture()
            elif self.config.architecture_type == NeuralArchitectureType.SWIN_TRANSFORMER:
                return self._create_swin_transformer_architecture()
            else:
                return self._create_custom_architecture()
                
        except Exception as e:
            self.logger.error(f"❌ 아키텍처 생성 실패: {e}")
            return self._create_fallback_architecture()
    
    def _create_resnet_architecture(self) -> nn.Module:
        """ResNet 아키텍처 생성"""
        class ResNetArchitecture(nn.Module):
            def __init__(self, config: NeuralArchitectureConfig):
                super().__init__()
                self.config = config
                
                # 입력 레이어
                self.input_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                
                # ResNet 블록들
                self.layer1 = self._make_layer(64, 64, 2)
                self.layer2 = self._make_layer(64, 128, 2, stride=2)
                self.layer3 = self._make_layer(128, 256, 2, stride=2)
                self.layer4 = self._make_layer(256, 512, 2, stride=2)
                
                # 분류 헤드
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, config.num_classes)
                
            def _make_layer(self, in_channels, out_channels, blocks, stride=1):
                layers = []
                layers.append(self._make_block(in_channels, out_channels, stride))
                for _ in range(1, blocks):
                    layers.append(self._make_block(out_channels, out_channels))
                return nn.Sequential(*layers)
            
            def _make_block(self, in_channels, out_channels, stride=1):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels)
                )
            
            def forward(self, x):
                x = self.input_conv(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                
                return x
        
        return ResNetArchitecture(self.config)
    
    def _create_vision_transformer_architecture(self) -> nn.Module:
        """Vision Transformer 아키텍처 생성"""
        class VisionTransformerArchitecture(nn.Module):
            def __init__(self, config: NeuralArchitectureConfig):
                super().__init__()
                self.config = config
                
                # 패치 임베딩
                patch_size = 16
                embed_dim = 768
                self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
                
                # 위치 임베딩
                num_patches = (config.input_size[0] // patch_size) ** 2
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
                
                # Transformer 블록들
                self.transformer_blocks = nn.ModuleList([
                    self._make_transformer_block(embed_dim) for _ in range(12)
                ])
                
                # 분류 헤드
                self.norm = nn.LayerNorm(embed_dim)
                self.head = nn.Linear(embed_dim, config.num_classes)
                
            def _make_transformer_block(self, embed_dim):
                return nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.MultiheadAttention(embed_dim, num_heads=12, batch_first=True),
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                )
            
            def forward(self, x):
                # 패치 임베딩
                x = self.patch_embed(x)
                b, c, h, w = x.shape
                x = x.flatten(2).transpose(1, 2)  # (B, N, C)
                
                # 위치 임베딩 추가
                x = x + self.pos_embed
                
                # Transformer 블록들
                for block in self.transformer_blocks:
                    x = x + block(x)
                
                # 분류
                x = self.norm(x)
                x = x.mean(dim=1)  # Global average pooling
                x = self.head(x)
                
                return x
        
        return VisionTransformerArchitecture(self.config)
    
    def _create_convnext_architecture(self) -> nn.Module:
        """ConvNeXt 아키텍처 생성"""
        class ConvNeXtArchitecture(nn.Module):
            def __init__(self, config: NeuralArchitectureConfig):
                super().__init__()
                self.config = config
                
                # Stem
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 96, kernel_size=4, stride=4),
                    nn.LayerNorm([96, config.input_size[0]//4, config.input_size[1]//4])
                )
                
                # ConvNeXt 블록들
                self.stages = nn.ModuleList([
                    self._make_convnext_stage(96, 192, 3),
                    self._make_convnext_stage(192, 384, 3),
                    self._make_convnext_stage(384, 768, 9),
                    self._make_convnext_stage(768, 768, 3)
                ])
                
                # 분류 헤드
                self.norm = nn.LayerNorm(768)
                self.head = nn.Linear(768, config.num_classes)
                
            def _make_convnext_stage(self, in_channels, out_channels, num_blocks):
                layers = []
                if in_channels != out_channels:
                    layers.append(nn.Conv2d(in_channels, out_channels, 2, 2))
                
                for _ in range(num_blocks):
                    layers.append(self._make_convnext_block(out_channels))
                
                return nn.Sequential(*layers)
            
            def _make_convnext_block(self, channels):
                return nn.Sequential(
                    nn.Conv2d(channels, channels, 7, 1, 3, groups=channels),
                    nn.LayerNorm([channels, 1, 1]),
                    nn.Conv2d(channels, channels * 4, 1),
                    nn.GELU(),
                    nn.Conv2d(channels * 4, channels, 1),
                    nn.Dropout(0.1)
                )
            
            def forward(self, x):
                x = self.stem(x)
                
                for stage in self.stages:
                    x = stage(x)
                
                x = self.norm(x)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
                x = self.head(x)
                
                return x
        
        return ConvNeXtArchitecture(self.config)
    
    def _create_efficientnet_architecture(self) -> nn.Module:
        """EfficientNet 아키텍처 생성"""
        class EfficientNetArchitecture(nn.Module):
            def __init__(self, config: NeuralArchitectureConfig):
                super().__init__()
                self.config = config
                
                # Stem
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True)
                )
                
                # MBConv 블록들
                self.blocks = nn.ModuleList([
                    self._make_mbconv_block(32, 16, 1, 1),
                    self._make_mbconv_block(16, 24, 6, 2),
                    self._make_mbconv_block(24, 40, 6, 2),
                    self._make_mbconv_block(40, 80, 6, 2),
                    self._make_mbconv_block(80, 112, 6, 1),
                    self._make_mbconv_block(112, 192, 6, 2),
                    self._make_mbconv_block(192, 320, 6, 1),
                ])
                
                # Head
                self.head = nn.Sequential(
                    nn.Conv2d(320, 1280, kernel_size=1),
                    nn.BatchNorm2d(1280),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(1280, config.num_classes)
                )
                
            def _make_mbconv_block(self, in_channels, out_channels, expansion, stride):
                return nn.Sequential(
                    nn.Conv2d(in_channels, in_channels * expansion, 1),
                    nn.BatchNorm2d(in_channels * expansion),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels * expansion, in_channels * expansion, 3, stride, 1, groups=in_channels * expansion),
                    nn.BatchNorm2d(in_channels * expansion),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels * expansion, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            
            def forward(self, x):
                x = self.stem(x)
                
                for block in self.blocks:
                    x = block(x)
                
                x = self.head(x)
                return x
        
        return EfficientNetArchitecture(self.config)
    
    def _create_swin_transformer_architecture(self) -> nn.Module:
        """Swin Transformer 아키텍처 생성"""
        class SwinTransformerArchitecture(nn.Module):
            def __init__(self, config: NeuralArchitectureConfig):
                super().__init__()
                self.config = config
                
                # Patch embedding
                self.patch_embed = nn.Conv2d(3, 96, kernel_size=4, stride=4)
                
                # Swin Transformer 블록들
                self.stages = nn.ModuleList([
                    self._make_swin_stage(96, 96, 2, 7),
                    self._make_swin_stage(96, 192, 2, 7),
                    self._make_swin_stage(192, 384, 2, 7),
                    self._make_swin_stage(384, 768, 2, 7)
                ])
                
                # Head
                self.norm = nn.LayerNorm(768)
                self.head = nn.Linear(768, config.num_classes)
                
            def _make_swin_stage(self, in_channels, out_channels, num_blocks, window_size):
                layers = []
                if in_channels != out_channels:
                    layers.append(nn.Conv2d(in_channels, out_channels, 2, 2))
                
                for _ in range(num_blocks):
                    layers.append(self._make_swin_block(out_channels, window_size))
                
                return nn.Sequential(*layers)
            
            def _make_swin_block(self, channels, window_size):
                return nn.Sequential(
                    nn.LayerNorm(channels),
                    nn.MultiheadAttention(channels, num_heads=8, batch_first=True),
                    nn.LayerNorm(channels),
                    nn.Linear(channels, channels * 4),
                    nn.GELU(),
                    nn.Linear(channels * 4, channels)
                )
            
            def forward(self, x):
                x = self.patch_embed(x)
                
                for stage in self.stages:
                    x = stage(x)
                
                x = self.norm(x)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
                x = self.head(x)
                
                return x
        
        return SwinTransformerArchitecture(self.config)
    
    def _create_custom_architecture(self) -> nn.Module:
        """커스텀 아키텍처 생성"""
        class CustomArchitecture(nn.Module):
            def __init__(self, config: NeuralArchitectureConfig):
                super().__init__()
                self.config = config
                
                # 사용자 정의 레이어들
                layers = []
                in_channels = 3
                
                for layer_config in config.layers:
                    if layer_config.layer_type == "conv":
                        layer = nn.Conv2d(
                            in_channels, layer_config.out_channels,
                            kernel_size=layer_config.kernel_size,
                            stride=layer_config.stride,
                            padding=layer_config.padding
                        )
                        layers.append(layer)
                        
                        if layer_config.normalization == "batch_norm":
                            layers.append(nn.BatchNorm2d(layer_config.out_channels))
                        
                        if layer_config.activation == "relu":
                            layers.append(nn.ReLU(inplace=True))
                        elif layer_config.activation == "gelu":
                            layers.append(nn.GELU())
                        
                        if layer_config.dropout > 0:
                            layers.append(nn.Dropout(layer_config.dropout))
                        
                        in_channels = layer_config.out_channels
                
                self.features = nn.Sequential(*layers)
                self.classifier = nn.Linear(in_channels, config.num_classes)
                
            def forward(self, x):
                x = self.features(x)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
        
        return CustomArchitecture(self.config)
    
    def _create_fallback_architecture(self) -> nn.Module:
        """폴백 아키텍처 생성"""
        class FallbackArchitecture(nn.Module):
            def __init__(self, config: NeuralArchitectureConfig):
                super().__init__()
                self.config = config
                
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, config.num_classes)
                )
                
            def forward(self, x):
                return self.features(x)
        
        return FallbackArchitecture(self.config)
    
    def _setup_progressive_loading(self):
        """점진적 로딩 설정"""
        try:
            if self.model is None:
                return
            
            # 모델의 모든 레이어를 점진적 로더에 등록
            for name, module in self.model.named_modules():
                if len(list(module.children())) == 0:  # Leaf module
                    layer_config = LayerConfig(
                        layer_type=type(module).__name__,
                        in_channels=getattr(module, 'in_channels', 0),
                        out_channels=getattr(module, 'out_channels', 0),
                        memory_priority=5  # 기본 우선순위
                    )
                    
                    self.layer_loader.add_layer(name, layer_config, module)
            
            self.logger.info("✅ 점진적 로딩 설정 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 점진적 로딩 설정 실패: {e}")
    
    def _register_computation_graph(self):
        """계산 그래프 등록"""
        try:
            if self.model is None:
                return
            
            # 모델의 주요 컴포넌트들을 계산 그래프에 등록
            for name, module in self.model.named_children():
                self.computation_graph.register_node(name, module)
            
            self.logger.info("✅ 계산 그래프 등록 완료")
            
        except Exception as e:
            self.logger.error(f"❌ 계산 그래프 등록 실패: {e}")
    
    def load_layers_progressively(self, target_memory_gb: float = None) -> Dict[str, bool]:
        """점진적 레이어 로딩 실행"""
        return self.layer_loader.load_layers_progressively(target_memory_gb)
    
    def forward(self, input_tensor: torch.Tensor, 
               enable_gradients: bool = False) -> torch.Tensor:
        """순전파 실행"""
        try:
            if not self.is_initialized:
                raise RuntimeError("아키텍처가 초기화되지 않았습니다")
            
            # Gradient 모드 설정
            self.computation_graph.enable_autograd = enable_gradients
            
            # 계산 그래프를 통한 순전파
            results = self.computation_graph.execute_forward_pass(input_tensor)
            
            # 최종 출력 반환
            if results:
                return list(results.values())[-1]  # 마지막 노드의 출력
            else:
                # 폴백: 직접 모델 실행
                with torch.set_grad_enabled(enable_gradients):
                    return self.model(input_tensor)
                
        except Exception as e:
            self.logger.error(f"❌ 순전파 실행 실패: {e}")
            raise
    
    def backward(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
        """역전파 실행"""
        return self.computation_graph.execute_backward_pass(loss)
    
    def get_memory_usage(self) -> float:
        """메모리 사용량 조회 (GB)"""
        return self.layer_loader.current_memory_usage / 1024
    
    def optimize_memory(self) -> Dict[str, Any]:
        """메모리 최적화"""
        try:
            # 가비지 컬렉션
            gc.collect()
            
            # PyTorch 캐시 정리
            if TORCH_AVAILABLE:
                if self.config.device == "mps" and torch.backends.mps.is_available():
                    try:
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                    except:
                        pass
                elif self.config.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 메모리 사용량 업데이트
            self.memory_usage = self.get_memory_usage()
            
            return {
                "success": True,
                "memory_usage_gb": self.memory_usage,
                "optimization_time": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 메모리 최적화 실패: {e}")
            return {"success": False, "error": str(e)}

# ==============================================
# 🔥 6. 팩토리 함수들
# ==============================================

def create_neural_architecture_manager(
    architecture_type: str = "resnet",
    input_size: Tuple[int, int] = (512, 512),
    num_classes: int = 20,
    memory_limit_gb: float = 8.0,
    device: str = "auto"
) -> NeuralArchitectureManager:
    """신경망 아키텍처 매니저 생성"""
    try:
        # 아키텍처 타입 변환
        arch_type = NeuralArchitectureType(architecture_type.lower())
        
        # 설정 생성
        config = NeuralArchitectureConfig(
            architecture_type=arch_type,
            input_size=input_size,
            num_classes=num_classes,
            memory_limit_gb=memory_limit_gb,
            device=device
        )
        
        # 매니저 생성
        manager = NeuralArchitectureManager(config)
        
        logger.info(f"✅ 신경망 아키텍처 매니저 생성 완료: {architecture_type}")
        return manager
        
    except Exception as e:
        logger.error(f"❌ 신경망 아키텍처 매니저 생성 실패: {e}")
        raise

def get_optimal_architecture_for_task(
    task_type: str,
    input_size: Tuple[int, int] = (512, 512),
    num_classes: int = 20,
    memory_limit_gb: float = 8.0
) -> NeuralArchitectureManager:
    """태스크별 최적 아키텍처 선택"""
    try:
        # 태스크별 최적 아키텍처 매핑
        task_architecture_map = {
            "human_parsing": "resnet",
            "pose_estimation": "convnext",
            "cloth_segmentation": "efficientnet",
            "geometric_matching": "vision_transformer",
            "virtual_fitting": "swin_transformer",
            "quality_assessment": "resnet"
        }
        
        architecture_type = task_architecture_map.get(task_type, "resnet")
        
        return create_neural_architecture_manager(
            architecture_type=architecture_type,
            input_size=input_size,
            num_classes=num_classes,
            memory_limit_gb=memory_limit_gb
        )
        
    except Exception as e:
        logger.error(f"❌ 최적 아키텍처 선택 실패: {e}")
        raise
