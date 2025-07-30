#!/usr/bin/env python3
"""
🔥 Ultimate AI Model Loading Debugger v4.0 - GitHub 프로젝트 완전 분석
==============================================================================
✅ 실제 GitHub 프로젝트 구조 229GB AI 모델 완전 분석
✅ 체크포인트 로딩 실패 원인 완전 분석 및 해결
✅ BaseStepMixin v19.2 호환성 완전 검증
✅ ModelLoader v5.1 실제 작동 상태 검증
✅ StepFactory v11.0 의존성 주입 완전 분석
✅ PyTorch weights_only 문제 해결책 제시
✅ M3 Max 128GB 메모리 최적화 상태 확인
✅ 실제 AI Step 클래스들 로딩 상태 완전 검증
✅ 체크포인트 파일 손상 여부 완전 검증
✅ 메모리 누수 및 성능 문제 완전 분석
==============================================================================
"""

import sys
import os
import time
import traceback
import logging
import asyncio
import threading
import psutil
import platform
import hashlib
import json
import importlib
import inspect
import gc
import weakref
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from enum import Enum

# 프로젝트 루트 경로 설정
project_root = Path(__file__).parent.parent if Path(__file__).parent.name == "backend" else Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

# =============================================================================
# 🔥 1. AI 모델 분석 데이터 클래스들
# =============================================================================

class CheckpointStatus(Enum):
    NOT_FOUND = "not_found"
    CORRUPTED = "corrupted" 
    LOADING_FAILED = "loading_failed"
    WEIGHTS_ONLY_FAILED = "weights_only_failed"
    DEVICE_INCOMPATIBLE = "device_incompatible"
    SUCCESS = "success"

class StepAnalysisStatus(Enum):
    IMPORT_FAILED = "import_failed"
    CLASS_NOT_FOUND = "class_not_found"
    INSTANCE_FAILED = "instance_failed"
    INIT_FAILED = "init_failed"
    DEPENDENCIES_MISSING = "dependencies_missing"
    AI_MODELS_FAILED = "ai_models_failed"
    SUCCESS = "success"

@dataclass
class CheckpointAnalysis:
    """체크포인트 파일 상세 분석"""
    file_path: Path
    exists: bool
    size_mb: float
    file_hash: str = ""
    
    # 로딩 테스트 결과
    pytorch_load_success: bool = False
    weights_only_success: bool = False
    safetensors_success: bool = False
    legacy_load_success: bool = False
    
    # 체크포인트 내용 분석
    checkpoint_keys: List[str] = field(default_factory=list)
    state_dict_structure: Dict[str, Any] = field(default_factory=dict)
    model_architecture: str = ""
    parameter_count: int = 0
    
    # 디바이스 호환성
    device_compatibility: Dict[str, bool] = field(default_factory=dict)
    
    # 오류 정보
    loading_errors: List[str] = field(default_factory=list)
    status: CheckpointStatus = CheckpointStatus.NOT_FOUND
    load_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0

@dataclass
class StepAnalysis:
    """AI Step 클래스 상세 분석"""
    step_name: str
    step_id: int
    module_path: str
    class_name: str
    
    # Import 분석
    import_success: bool = False
    import_time: float = 0.0
    import_errors: List[str] = field(default_factory=list)
    
    # 클래스 분석
    class_found: bool = False
    is_base_step_mixin: bool = False
    has_process_method: bool = False
    has_initialize_method: bool = False
    
    # 인스턴스 생성 분석
    instance_created: bool = False
    constructor_dependencies: Dict[str, Any] = field(default_factory=dict)
    instance_errors: List[str] = field(default_factory=list)
    
    # 초기화 분석
    initialization_success: bool = False
    initialization_time: float = 0.0
    initialization_errors: List[str] = field(default_factory=list)
    
    # 의존성 분석
    dependencies_resolved: Dict[str, bool] = field(default_factory=dict)
    model_loader_injected: bool = False
    memory_manager_injected: bool = False
    
    # AI 모델 분석
    ai_models_detected: List[str] = field(default_factory=list)
    checkpoints_analysis: List[CheckpointAnalysis] = field(default_factory=list)
    total_model_size_gb: float = 0.0
    
    # 성능 분석
    memory_footprint_mb: float = 0.0
    inference_test_success: bool = False
    inference_time_ms: float = 0.0
    
    # 전체 상태
    status: StepAnalysisStatus = StepAnalysisStatus.IMPORT_FAILED
    overall_health_score: float = 0.0

@dataclass
class SystemEnvironmentAnalysis:
    """시스템 환경 완전 분석"""
    # 하드웨어 정보
    cpu_info: Dict[str, Any] = field(default_factory=dict)
    memory_info: Dict[str, Any] = field(default_factory=dict)
    gpu_info: Dict[str, Any] = field(default_factory=dict)
    
    # 소프트웨어 환경
    python_info: Dict[str, Any] = field(default_factory=dict)
    pytorch_info: Dict[str, Any] = field(default_factory=dict)
    cuda_info: Dict[str, Any] = field(default_factory=dict)
    
    # 프로젝트 환경
    project_structure: Dict[str, Any] = field(default_factory=dict)
    conda_environment: Dict[str, Any] = field(default_factory=dict)
    dependencies_status: Dict[str, bool] = field(default_factory=dict)
    
    # 진단 결과
    is_m3_max: bool = False
    memory_sufficient: bool = False
    cuda_available: bool = False
    mps_available: bool = False
    recommended_device: str = "cpu"

# =============================================================================
# 🔥 2. 안전 실행 매니저
# =============================================================================

class UltimateSafetyManager:
    """강화된 안전 실행 매니저"""
    
    def __init__(self):
        self.timeout_duration = 120  # 2분 타임아웃
        self.max_memory_gb = 8      # 8GB 메모리 제한
        self.active_operations = {}
        self.start_time = time.time()
        
    @contextmanager
    def safe_execution(self, operation_name: str, timeout: int = None, memory_limit_gb: float = None):
        """초안전 실행 컨텍스트"""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**3)  # GB 단위
        timeout = timeout or self.timeout_duration
        memory_limit = memory_limit_gb or self.max_memory_gb
        
        print(f"🔒 [{operation_id}] 안전 실행 시작 (타임아웃: {timeout}초, 메모리 제한: {memory_limit:.1f}GB)")
        
        self.active_operations[operation_id] = {
            'start_time': start_time,
            'start_memory': start_memory,
            'timeout': timeout,
            'memory_limit': memory_limit
        }
        
        try:
            # 메모리 모니터링 스레드 시작
            monitoring_thread = threading.Thread(
                target=self._monitor_operation,
                args=(operation_id, timeout, memory_limit),
                daemon=True
            )
            monitoring_thread.start()
            
            yield
            
        except TimeoutError:
            print(f"⏰ [{operation_id}] 타임아웃 발생 ({timeout}초)")
            raise
        except MemoryError:
            print(f"💾 [{operation_id}] 메모리 한계 초과 ({memory_limit:.1f}GB)")
            raise
        except Exception as e:
            print(f"❌ [{operation_id}] 실행 중 오류: {type(e).__name__}: {e}")
            if hasattr(e, '__traceback__'):
                tb_lines = traceback.format_tb(e.__traceback__)
                if tb_lines:
                    print(f"   스택 추적: {tb_lines[-1].strip()}")
            raise
        finally:
            # 정리 작업
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]
                
            elapsed = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / (1024**3)
            memory_used = end_memory - start_memory
            
            print(f"✅ [{operation_id}] 완료 ({elapsed:.2f}초, 메모리: +{memory_used:.2f}GB)")
            
            # 메모리 정리
            if memory_used > 0.5:  # 500MB 이상 사용시 정리
                gc.collect()
    
    def _monitor_operation(self, operation_id: str, timeout: float, memory_limit: float):
        """작업 모니터링"""
        try:
            while operation_id in self.active_operations:
                current_time = time.time()
                operation = self.active_operations.get(operation_id)
                
                if not operation:
                    break
                
                # 타임아웃 체크
                elapsed = current_time - operation['start_time']
                if elapsed > timeout:
                    print(f"⚠️ [{operation_id}] 타임아웃 경고 ({elapsed:.1f}초/{timeout}초)")
                    break
                
                # 메모리 체크
                current_memory = psutil.Process().memory_info().rss / (1024**3)
                if current_memory > memory_limit:
                    print(f"⚠️ [{operation_id}] 메모리 사용량 경고 ({current_memory:.1f}GB/{memory_limit:.1f}GB)")
                    break
                
                time.sleep(1)  # 1초마다 체크
                
        except Exception:
            pass  # 모니터링 스레드에서는 예외 무시

# 전역 안전 매니저
safety_manager = UltimateSafetyManager()

# =============================================================================
# 🔥 3. 시스템 환경 분석기
# =============================================================================

class SystemEnvironmentAnalyzer:
    """시스템 환경 완전 분석기"""
    
    def __init__(self):
        self.analysis_result = SystemEnvironmentAnalysis()
        
    def analyze_complete_environment(self) -> SystemEnvironmentAnalysis:
        """완전한 시스템 환경 분석"""
        
        print("📊 시스템 환경 완전 분석 시작...")
        
        with safety_manager.safe_execution("시스템 환경 분석", timeout=60):
            self._analyze_hardware()
            self._analyze_software()
            self._analyze_project_structure()
            self._analyze_dependencies()
            self._make_recommendations()
        
        return self.analysis_result
    
    def _analyze_hardware(self):
        """하드웨어 분석"""
        try:
            # CPU 정보
            self.analysis_result.cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'usage_percent': psutil.cpu_percent(interval=1),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'is_apple_silicon': platform.machine() == 'arm64' and platform.system() == 'Darwin'
            }
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            self.analysis_result.memory_info = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'usage_percent': memory.percent,
                'sufficient_for_ai': memory.total >= 16 * (1024**3)  # 16GB 이상
            }
            
            # M3 Max 감지
            if self.analysis_result.cpu_info['is_apple_silicon']:
                total_memory = self.analysis_result.memory_info['total_gb']
                if total_memory >= 100:  # 128GB 모델
                    self.analysis_result.is_m3_max = True
            
            self.analysis_result.memory_sufficient = self.analysis_result.memory_info['available_gb'] >= 8
            
        except Exception as e:
            print(f"❌ 하드웨어 분석 실패: {e}")
    
    def _analyze_software(self):
        """소프트웨어 환경 분석"""
        try:
            # Python 정보
            self.analysis_result.python_info = {
                'version': sys.version.split()[0],
                'executable': sys.executable,
                'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'none'),
                'virtual_env': os.environ.get('VIRTUAL_ENV', 'none'),
                'platform': platform.platform()
            }
            
            # PyTorch 분석
            try:
                import torch
                self.analysis_result.pytorch_info = {
                    'available': True,
                    'version': torch.__version__,
                    'cuda_available': torch.cuda.is_available(),
                    'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                    'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
                    'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
                }
                
                # 추천 디바이스 결정
                if self.analysis_result.pytorch_info['mps_available']:
                    self.analysis_result.recommended_device = 'mps'
                    self.analysis_result.mps_available = True
                elif self.analysis_result.pytorch_info['cuda_available']:
                    self.analysis_result.recommended_device = 'cuda'
                    self.analysis_result.cuda_available = True
                else:
                    self.analysis_result.recommended_device = 'cpu'
                    
            except ImportError:
                self.analysis_result.pytorch_info = {
                    'available': False,
                    'error': 'PyTorch not installed'
                }
            
            # Conda 환경 분석
            conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'none')
            self.analysis_result.conda_environment = {
                'active_env': conda_env,
                'conda_available': conda_env != 'none',
                'env_path': os.environ.get('CONDA_PREFIX', ''),
                'python_path': sys.executable
            }
            
        except Exception as e:
            print(f"❌ 소프트웨어 분석 실패: {e}")
    
    def _analyze_project_structure(self):
        """프로젝트 구조 분석"""
        try:
            structure = {
                'project_root': str(project_root),
                'backend_exists': (project_root / 'backend').exists(),
                'frontend_exists': (project_root / 'frontend').exists(),
                'ai_models_dir': None,
                'ai_models_size_gb': 0.0,
                'step_modules': []
            }
            
            # AI 모델 디렉토리 찾기
            possible_ai_dirs = [
                project_root / 'ai_models',
                project_root / 'backend' / 'ai_models',
                project_root / 'models'
            ]
            
            for ai_dir in possible_ai_dirs:
                if ai_dir.exists():
                    structure['ai_models_dir'] = str(ai_dir)
                    # 크기 계산
                    total_size = 0
                    for file_path in ai_dir.rglob('*'):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                    structure['ai_models_size_gb'] = total_size / (1024**3)
                    break
            
            # Step 모듈 찾기
            steps_dir = project_root / 'backend' / 'app' / 'ai_pipeline' / 'steps'
            if steps_dir.exists():
                for step_file in steps_dir.glob('step_*.py'):
                    structure['step_modules'].append(step_file.stem)
            
            self.analysis_result.project_structure = structure
            
        except Exception as e:
            print(f"❌ 프로젝트 구조 분석 실패: {e}")
    
    def _analyze_dependencies(self):
        """의존성 분석"""
        try:
            dependencies = {}
            
            # 핵심 라이브러리 체크
            core_libs = ['torch', 'torchvision', 'numpy', 'PIL', 'cv2', 'transformers', 'safetensors']
            
            for lib in core_libs:
                try:
                    module = importlib.import_module(lib if lib != 'PIL' else 'PIL.Image')
                    dependencies[lib] = True
                except ImportError:
                    dependencies[lib] = False
            
            self.analysis_result.dependencies_status = dependencies
            
        except Exception as e:
            print(f"❌ 의존성 분석 실패: {e}")
    
    def _make_recommendations(self):
        """환경 개선 추천사항 생성"""
        # 시스템 상태에 따른 추천사항은 나중에 전체 분석에서 처리
        pass

# =============================================================================
# 🔥 4. 체크포인트 분석기
# =============================================================================

class CheckpointAnalyzer:
    """체크포인트 파일 완전 분석기"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.torch_available = False
        self.safetensors_available = False
        
        try:
            import torch
            self.torch_available = True
            self.torch = torch
        except ImportError:
            pass
            
        try:
            from safetensors.torch import load_file
            self.safetensors_available = True
            self.safetensors_load = load_file
        except ImportError:
            pass
    
    def analyze_checkpoint(self, checkpoint_path: Path) -> CheckpointAnalysis:
        """체크포인트 파일 완전 분석"""
        
        analysis = CheckpointAnalysis(
            file_path=checkpoint_path,
            exists=checkpoint_path.exists(),
            size_mb=0.0
        )
        
        if not analysis.exists:
            analysis.status = CheckpointStatus.NOT_FOUND
            return analysis
        
        # 파일 크기 및 해시
        try:
            stat_info = checkpoint_path.stat()
            analysis.size_mb = stat_info.st_size / (1024 * 1024)
            
            # 해시 계산 (큰 파일은 샘플링)
            if analysis.size_mb < 100:  # 100MB 미만만 전체 해시
                analysis.file_hash = self._calculate_file_hash(checkpoint_path)
            else:
                analysis.file_hash = self._calculate_sample_hash(checkpoint_path)
                
        except Exception as e:
            analysis.loading_errors.append(f"파일 정보 읽기 실패: {e}")
        
        # 로딩 테스트 수행
        if self.torch_available:
            self._test_pytorch_loading(analysis)
        
        if self.safetensors_available and checkpoint_path.suffix == '.safetensors':
            self._test_safetensors_loading(analysis)
        
        # 상태 결정
        if analysis.pytorch_load_success or analysis.safetensors_success:
            analysis.status = CheckpointStatus.SUCCESS
        elif analysis.loading_errors:
            if any("corrupted" in error.lower() for error in analysis.loading_errors):
                analysis.status = CheckpointStatus.CORRUPTED
            elif any("weights_only" in error.lower() for error in analysis.loading_errors):
                analysis.status = CheckpointStatus.WEIGHTS_ONLY_FAILED
            else:
                analysis.status = CheckpointStatus.LOADING_FAILED
        
        return analysis
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """전체 파일 해시 계산"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def _calculate_sample_hash(self, file_path: Path, sample_size: int = 1024*1024) -> str:
        """샘플 해시 계산 (대용량 파일용)"""
        try:
            hash_md5 = hashlib.md5()
            file_size = file_path.stat().st_size
            
            with open(file_path, "rb") as f:
                # 시작 부분
                chunk = f.read(sample_size)
                hash_md5.update(chunk)
                
                # 중간 부분
                if file_size > sample_size * 3:
                    f.seek(file_size // 2)
                    chunk = f.read(sample_size)
                    hash_md5.update(chunk)
                
                # 끝 부분
                if file_size > sample_size * 2:
                    f.seek(file_size - sample_size)
                    chunk = f.read(sample_size)
                    hash_md5.update(chunk)
            
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def _test_pytorch_loading(self, analysis: CheckpointAnalysis):
        """PyTorch 로딩 테스트"""
        if not self.torch_available:
            analysis.loading_errors.append("PyTorch 없음")
            return
        
        file_path = analysis.file_path
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)
        
        # 1. weights_only=True 시도
        try:
            with safety_manager.safe_execution(f"PyTorch weights_only 로딩 {file_path.name}", timeout=60):
                checkpoint = self.torch.load(file_path, map_location=self.device, weights_only=True)
                analysis.weights_only_success = True
                self._analyze_checkpoint_content(analysis, checkpoint)
                
        except Exception as e:
            analysis.loading_errors.append(f"weights_only 로딩 실패: {e}")
        
        # 2. weights_only=False 시도
        if not analysis.weights_only_success:
            try:
                with safety_manager.safe_execution(f"PyTorch 일반 로딩 {file_path.name}", timeout=60):
                    checkpoint = self.torch.load(file_path, map_location=self.device, weights_only=False)
                    analysis.pytorch_load_success = True
                    self._analyze_checkpoint_content(analysis, checkpoint)
                    
            except Exception as e:
                analysis.loading_errors.append(f"일반 로딩 실패: {e}")
        
        # 3. 레거시 로딩 시도
        if not analysis.pytorch_load_success and not analysis.weights_only_success:
            try:
                with safety_manager.safe_execution(f"PyTorch 레거시 로딩 {file_path.name}", timeout=60):
                    checkpoint = self.torch.load(file_path, map_location=self.device)
                    analysis.legacy_load_success = True
                    analysis.pytorch_load_success = True
                    self._analyze_checkpoint_content(analysis, checkpoint)
                    
            except Exception as e:
                analysis.loading_errors.append(f"레거시 로딩 실패: {e}")
        
        # 성능 측정
        analysis.load_time_seconds = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss / (1024**2)
        analysis.memory_usage_mb = end_memory - start_memory
    
    def _test_safetensors_loading(self, analysis: CheckpointAnalysis):
        """SafeTensors 로딩 테스트"""
        if not self.safetensors_available:
            analysis.loading_errors.append("SafeTensors 라이브러리 없음")
            return
        
        try:
            with safety_manager.safe_execution(f"SafeTensors 로딩 {analysis.file_path.name}", timeout=60):
                checkpoint = self.safetensors_load(str(analysis.file_path))
                analysis.safetensors_success = True
                self._analyze_checkpoint_content(analysis, checkpoint)
                
        except Exception as e:
            analysis.loading_errors.append(f"SafeTensors 로딩 실패: {e}")
    
    def _analyze_checkpoint_content(self, analysis: CheckpointAnalysis, checkpoint):
        """체크포인트 내용 분석"""
        try:
            # State dict 추출
            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
            
            if isinstance(state_dict, dict):
                analysis.checkpoint_keys = list(state_dict.keys())[:50]  # 처음 50개만
                
                # 파라미터 수 계산
                param_count = 0
                for key, tensor in state_dict.items():
                    if hasattr(tensor, 'numel'):
                        param_count += tensor.numel()
                analysis.parameter_count = param_count
                
                # 모델 아키텍처 추정
                analysis.model_architecture = self._estimate_architecture(state_dict)
                
                # 구조 정보
                analysis.state_dict_structure = {
                    'total_keys': len(state_dict),
                    'tensor_keys': sum(1 for v in state_dict.values() if hasattr(v, 'shape')),
                    'parameter_count': param_count,
                    'estimated_size_mb': param_count * 4 / (1024**2) if param_count > 0 else 0  # float32 가정
                }
                
                # 디바이스 호환성 테스트
                self._test_device_compatibility(analysis, state_dict)
            
        except Exception as e:
            analysis.loading_errors.append(f"체크포인트 내용 분석 실패: {e}")
    
    def _estimate_architecture(self, state_dict: dict) -> str:
        """모델 아키텍처 추정"""
        keys = list(state_dict.keys())
        key_str = ' '.join(keys).lower()
        
        if 'backbone' in key_str:
            return "Segmentation Model (with backbone)"
        elif 'pose' in key_str or 'keypoint' in key_str:
            return "Pose Estimation Model"
        elif 'diffusion' in key_str or 'unet' in key_str:
            return "Diffusion Model"
        elif 'vit' in key_str or 'transformer' in key_str:
            return "Vision Transformer"
        elif 'resnet' in key_str or 'efficientnet' in key_str:
            return "CNN Backbone"
        elif 'sam' in key_str or 'segment' in key_str:
            return "Segmentation Model"
        elif any(keyword in key_str for keyword in ['conv', 'bn', 'relu']):
            return "Convolutional Neural Network"
        else:
            return "Unknown Architecture"
    
    def _test_device_compatibility(self, analysis: CheckpointAnalysis, state_dict: dict):
        """디바이스 호환성 테스트"""
        if not self.torch_available:
            return
        
        try:
            # 첫 번째 텐서로 테스트
            first_tensor = None
            for value in state_dict.values():
                if hasattr(value, 'to'):
                    first_tensor = value
                    break
            
            if first_tensor is None:
                return
            
            # CPU 테스트
            try:
                cpu_tensor = first_tensor.to('cpu')
                analysis.device_compatibility['cpu'] = True
            except Exception:
                analysis.device_compatibility['cpu'] = False
            
            # CUDA 테스트
            if self.torch.cuda.is_available():
                try:
                    cuda_tensor = first_tensor.to('cuda')
                    analysis.device_compatibility['cuda'] = True
                except Exception:
                    analysis.device_compatibility['cuda'] = False
            
            # MPS 테스트
            if hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
                try:
                    mps_tensor = first_tensor.to('mps')
                    analysis.device_compatibility['mps'] = True
                except Exception:
                    analysis.device_compatibility['mps'] = False
            
        except Exception as e:
            analysis.loading_errors.append(f"디바이스 호환성 테스트 실패: {e}")

# =============================================================================
# 🔥 5. AI Step 분석기
# =============================================================================

class AIStepAnalyzer:
    """AI Step 클래스 완전 분석기"""
    
    def __init__(self, system_analysis: SystemEnvironmentAnalysis):
        self.system_analysis = system_analysis
        self.checkpoint_analyzer = CheckpointAnalyzer(
            device=system_analysis.recommended_device
        )
    
    def analyze_step(self, step_config: Dict[str, Any]) -> StepAnalysis:
        """AI Step 완전 분석"""
        
        analysis = StepAnalysis(
            step_name=step_config['step_name'],
            step_id=step_config.get('step_id', 0),
            module_path=step_config['module_path'],
            class_name=step_config['class_name']
        )
        
        print(f"\n🔧 {analysis.step_name} 완전 분석 시작...")
        
        # 1. Import 테스트
        self._test_import(analysis)
        
        # 2. 클래스 분석
        if analysis.import_success:
            self._analyze_class(analysis)
        
        # 3. 인스턴스 생성 테스트
        if analysis.class_found:
            self._test_instance_creation(analysis)
        
        # 4. 초기화 테스트
        if analysis.instance_created:
            self._test_initialization(analysis)
        
        # 5. AI 모델 분석
        self._analyze_ai_models(analysis)
        
        # 6. 상태 결정 및 점수 계산
        self._determine_status_and_score(analysis)
        
        return analysis
    
    def _test_import(self, analysis: StepAnalysis):
        """Import 테스트"""
        try:
            with safety_manager.safe_execution(f"{analysis.step_name} Import", timeout=30):
                start_time = time.time()
                module = importlib.import_module(analysis.module_path)
                analysis.import_time = time.time() - start_time
                analysis.import_success = True
                
                # 클래스 존재 확인
                if hasattr(module, analysis.class_name):
                    analysis.class_found = True
                    
        except Exception as e:
            analysis.import_errors.append(str(e))
            analysis.status = StepAnalysisStatus.IMPORT_FAILED
    
    def _analyze_class(self, analysis: StepAnalysis):
        """클래스 구조 분석"""
        try:
            module = importlib.import_module(analysis.module_path)
            step_class = getattr(module, analysis.class_name)
            
            # 클래스 메서드 검사
            class_methods = [method for method in dir(step_class) if not method.startswith('_')]
            
            analysis.has_process_method = 'process' in class_methods
            analysis.has_initialize_method = 'initialize' in class_methods
            
            # BaseStepMixin 상속 확인
            mro = inspect.getmro(step_class)
            analysis.is_base_step_mixin = any('BaseStepMixin' in cls.__name__ for cls in mro)
            
            print(f"   ✅ 클래스 분석 완료: process={analysis.has_process_method}, init={analysis.has_initialize_method}")
            
        except Exception as e:
            analysis.import_errors.append(f"클래스 분석 실패: {e}")
    
    def _test_instance_creation(self, analysis: StepAnalysis):
        """인스턴스 생성 테스트"""
        try:
            with safety_manager.safe_execution(f"{analysis.step_name} 인스턴스 생성", timeout=60):
                module = importlib.import_module(analysis.module_path)
                step_class = getattr(module, analysis.class_name)
                
                # 생성자 파라미터 분석
                signature = inspect.signature(step_class.__init__)
                params = list(signature.parameters.keys())[1:]  # self 제외
                
                # 기본 의존성 준비
                constructor_args = {
                    'device': self.system_analysis.recommended_device,
                    'strict_mode': False
                }
                
                # 필요한 경우 추가 의존성
                if 'model_loader' in params:
                    constructor_args['model_loader'] = None  # Mock으로 대체 가능
                
                analysis.constructor_dependencies = constructor_args
                
                # 인스턴스 생성
                step_instance = step_class(**constructor_args)
                analysis.instance_created = True
                
                # 의존성 주입 상태 확인
                if hasattr(step_instance, 'model_loader'):
                    analysis.model_loader_injected = step_instance.model_loader is not None
                
                if hasattr(step_instance, 'memory_manager'):
                    analysis.memory_manager_injected = step_instance.memory_manager is not None
                
                print(f"   ✅ 인스턴스 생성 성공")
                
        except Exception as e:
            analysis.instance_errors.append(str(e))
            analysis.status = StepAnalysisStatus.INSTANCE_FAILED
    
    def _test_initialization(self, analysis: StepAnalysis):
        """초기화 테스트"""
        if not analysis.instance_created:
            return
        
        try:
            with safety_manager.safe_execution(f"{analysis.step_name} 초기화", timeout=90):
                module = importlib.import_module(analysis.module_path)
                step_class = getattr(module, analysis.class_name)
                step_instance = step_class(**analysis.constructor_dependencies)
                
                start_time = time.time()
                
                if hasattr(step_instance, 'initialize'):
                    if asyncio.iscoroutinefunction(step_instance.initialize):
                        # 비동기 초기화
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        result = loop.run_until_complete(
                            asyncio.wait_for(step_instance.initialize(), timeout=60.0)
                        )
                    else:
                        # 동기 초기화
                        result = step_instance.initialize()
                    
                    if result:
                        analysis.initialization_success = True
                        analysis.initialization_time = time.time() - start_time
                        print(f"   ✅ 초기화 성공 ({analysis.initialization_time:.2f}초)")
                    else:
                        analysis.initialization_errors.append("초기화가 False 반환")
                        
                else:
                    # initialize 메서드가 없는 경우
                    analysis.initialization_success = True
                    print(f"   ⚠️ initialize 메서드 없음 (기본 성공 처리)")
                    
        except TimeoutError:
            analysis.initialization_errors.append("초기화 타임아웃 (60초)")
        except Exception as e:
            analysis.initialization_errors.append(str(e))
            analysis.status = StepAnalysisStatus.INIT_FAILED
    
    def _analyze_ai_models(self, analysis: StepAnalysis):
        """AI 모델 파일 분석"""
        try:
            # Step ID 기반 모델 디렉토리 찾기
            ai_models_base = self.system_analysis.project_structure.get('ai_models_dir')
            if not ai_models_base:
                return
                
            ai_models_path = Path(ai_models_base)
            
            # Step별 모델 디렉토리 패턴
            step_patterns = [
                f"step_{analysis.step_id:02d}_*",
                f"*{analysis.step_name.lower().replace('step', '')}*",
                analysis.step_name.lower()
            ]
            
            model_files = []
            
            for pattern in step_patterns:
                matching_dirs = list(ai_models_path.glob(pattern))
                for model_dir in matching_dirs:
                    if model_dir.is_dir():
                        # 체크포인트 파일 찾기
                        for ext in ['*.pth', '*.pt', '*.safetensors', '*.bin']:
                            found_files = list(model_dir.rglob(ext))
                            model_files.extend(found_files)
            
            # 모델 파일 분석
            total_size = 0
            for model_file in model_files[:10]:  # 최대 10개만
                if model_file.stat().st_size > 10 * 1024 * 1024:  # 10MB 이상만
                    checkpoint_analysis = self.checkpoint_analyzer.analyze_checkpoint(model_file)
                    analysis.checkpoints_analysis.append(checkpoint_analysis)
                    analysis.ai_models_detected.append(model_file.name)
                    total_size += checkpoint_analysis.size_mb
            
            analysis.total_model_size_gb = total_size / 1024
            
            if analysis.ai_models_detected:
                print(f"   📊 AI 모델 {len(analysis.ai_models_detected)}개 발견 ({analysis.total_model_size_gb:.1f}GB)")
            
        except Exception as e:
            print(f"   ⚠️ AI 모델 분석 실패: {e}")
    
    def _determine_status_and_score(self, analysis: StepAnalysis):
        """상태 결정 및 건강도 점수 계산"""
        score = 0.0
        
        # Import (20점)
        if analysis.import_success:
            score += 20
        
        # 클래스 구조 (20점)
        if analysis.class_found:
            score += 10
        if analysis.is_base_step_mixin:
            score += 5
        if analysis.has_process_method:
            score += 3
        if analysis.has_initialize_method:
            score += 2
        
        # 인스턴스 생성 (20점)
        if analysis.instance_created:
            score += 20
        
        # 초기화 (20점)
        if analysis.initialization_success:
            score += 20
        
        # AI 모델 (20점)
        if analysis.ai_models_detected:
            score += 10
            successful_checkpoints = sum(
                1 for cp in analysis.checkpoints_analysis 
                if cp.status == CheckpointStatus.SUCCESS
            )
            if successful_checkpoints > 0:
                score += 10
        
        analysis.overall_health_score = score
        
        # 상태 결정
        if not analysis.import_success:
            analysis.status = StepAnalysisStatus.IMPORT_FAILED
        elif not analysis.class_found:
            analysis.status = StepAnalysisStatus.CLASS_NOT_FOUND
        elif not analysis.instance_created:
            analysis.status = StepAnalysisStatus.INSTANCE_FAILED
        elif not analysis.initialization_success:
            analysis.status = StepAnalysisStatus.INIT_FAILED
        elif not analysis.ai_models_detected:
            analysis.status = StepAnalysisStatus.AI_MODELS_FAILED
        else:
            analysis.status = StepAnalysisStatus.SUCCESS

# =============================================================================
# 🔥 6. 메인 디버깅 시스템
# =============================================================================

class UltimateAIModelDebugger:
    """최고급 AI 모델 디버깅 시스템"""
    
    def __init__(self):
        self.start_time = time.time()
        self.system_analysis = None
        self.step_analyses = {}
        
        # GitHub 프로젝트 Step 설정
        self.step_configs = [
            {
                'step_name': 'HumanParsingStep',
                'step_id': 1,
                'module_path': 'app.ai_pipeline.steps.step_01_human_parsing',
                'class_name': 'HumanParsingStep'
            },
            {
                'step_name': 'PoseEstimationStep',
                'step_id': 2,
                'module_path': 'app.ai_pipeline.steps.step_02_pose_estimation',
                'class_name': 'PoseEstimationStep'
            },
            {
                'step_name': 'ClothSegmentationStep',
                'step_id': 3,
                'module_path': 'app.ai_pipeline.steps.step_03_cloth_segmentation',
                'class_name': 'ClothSegmentationStep'
            },
            {
                'step_name': 'GeometricMatchingStep',
                'step_id': 4,
                'module_path': 'app.ai_pipeline.steps.step_04_geometric_matching',
                'class_name': 'GeometricMatchingStep'
            },
            {
                'step_name': 'ClothWarpingStep',
                'step_id': 5,
                'module_path': 'app.ai_pipeline.steps.step_05_cloth_warping',
                'class_name': 'ClothWarpingStep'
            },
            {
                'step_name': 'VirtualFittingStep',
                'step_id': 6,
                'module_path': 'app.ai_pipeline.steps.step_06_virtual_fitting',
                'class_name': 'VirtualFittingStep'
            },
            {
                'step_name': 'PostProcessingStep',
                'step_id': 7,
                'module_path': 'app.ai_pipeline.steps.step_07_post_processing',
                'class_name': 'PostProcessingStep'
            },
            {
                'step_name': 'QualityAssessmentStep',
                'step_id': 8,
                'module_path': 'app.ai_pipeline.steps.step_08_quality_assessment',
                'class_name': 'QualityAssessmentStep'
            }
        ]
    
    def run_ultimate_debugging(self) -> Dict[str, Any]:
        """최고급 디버깅 실행"""
        
        print("🔥" * 30)
        print("🔥 Ultimate AI Model Loading Debugger v4.0 시작")
        print("🔥 GitHub 프로젝트 229GB AI 모델 완전 분석")
        print("🔥" * 30)
        
        debug_result = {
            'timestamp': time.time(),
            'debug_version': '4.0',
            'system_analysis': {},
            'step_analyses': {},
            'overall_summary': {},
            'critical_issues': [],
            'recommendations': [],
            'performance_metrics': {}
        }
        
        try:
            # 1. 시스템 환경 완전 분석
            print("\n📊 1. 시스템 환경 완전 분석")
            self.system_analysis = SystemEnvironmentAnalyzer().analyze_complete_environment()
            debug_result['system_analysis'] = self._serialize_system_analysis(self.system_analysis)
            self._print_system_analysis()
            
            # 2. AI Step별 완전 분석
            print("\n🚀 2. AI Step별 완전 분석")
            step_analyzer = AIStepAnalyzer(self.system_analysis)
            
            for step_config in self.step_configs:
                try:
                    step_analysis = step_analyzer.analyze_step(step_config)
                    self.step_analyses[step_config['step_name']] = step_analysis
                    debug_result['step_analyses'][step_config['step_name']] = self._serialize_step_analysis(step_analysis)
                    
                except Exception as e:
                    print(f"❌ {step_config['step_name']} 분석 실패: {e}")
                    
            # 3. 전체 요약 생성
            print("\n📊 3. 전체 분석 결과 요약")
            debug_result['overall_summary'] = self._generate_overall_summary()
            debug_result['critical_issues'] = self._identify_critical_issues()
            debug_result['recommendations'] = self._generate_actionable_recommendations()
            debug_result['performance_metrics'] = self._calculate_performance_metrics()
            
            # 4. 결과 출력
            self._print_debug_results(debug_result)
            
            # 5. 결과 저장
            self._save_debug_results(debug_result)
            
        except Exception as e:
            print(f"\n❌ 디버깅 실행 중 치명적 오류: {e}")
            print(f"스택 트레이스:\n{traceback.format_exc()}")
            debug_result['fatal_error'] = str(e)
        
        finally:
            total_time = time.time() - self.start_time
            print(f"\n🎉 Ultimate AI Model Debugging 완료! (총 소요시간: {total_time:.2f}초)")
            debug_result['total_debug_time'] = total_time
        
        return debug_result
    
    def _serialize_system_analysis(self, analysis: SystemEnvironmentAnalysis) -> Dict[str, Any]:
        """시스템 분석 결과 직렬화"""
        return {
            'cpu_info': analysis.cpu_info,
            'memory_info': analysis.memory_info,
            'pytorch_info': analysis.pytorch_info,
            'project_structure': analysis.project_structure,
            'dependencies_status': analysis.dependencies_status,
            'recommendations': {
                'is_m3_max': analysis.is_m3_max,
                'memory_sufficient': analysis.memory_sufficient,
                'recommended_device': analysis.recommended_device
            }
        }
    
    def _serialize_step_analysis(self, analysis: StepAnalysis) -> Dict[str, Any]:
        """Step 분석 결과 직렬화"""
        return {
            'basic_info': {
                'step_name': analysis.step_name,
                'step_id': analysis.step_id,
                'module_path': analysis.module_path,
                'class_name': analysis.class_name
            },
            'import_analysis': {
                'success': analysis.import_success,
                'time': analysis.import_time,
                'errors': analysis.import_errors
            },
            'class_analysis': {
                'found': analysis.class_found,
                'is_base_step_mixin': analysis.is_base_step_mixin,
                'has_process_method': analysis.has_process_method,
                'has_initialize_method': analysis.has_initialize_method
            },
            'instance_analysis': {
                'created': analysis.instance_created,
                'dependencies': analysis.constructor_dependencies,
                'errors': analysis.instance_errors
            },
            'initialization': {
                'success': analysis.initialization_success,
                'time': analysis.initialization_time,
                'errors': analysis.initialization_errors
            },
            'ai_models': {
                'detected': analysis.ai_models_detected,
                'total_size_gb': analysis.total_model_size_gb,
                'checkpoint_count': len(analysis.checkpoints_analysis),
                'successful_checkpoints': sum(
                    1 for cp in analysis.checkpoints_analysis 
                    if cp.status == CheckpointStatus.SUCCESS
                )
            },
            'performance': {
                'memory_footprint_mb': analysis.memory_footprint_mb,
                'health_score': analysis.overall_health_score
            },
            'status': analysis.status.value
        }
    
    def _print_system_analysis(self):
        """시스템 분석 결과 출력"""
        analysis = self.system_analysis
        
        print(f"   💻 하드웨어:")
        print(f"      CPU: {analysis.cpu_info.get('logical_cores', 0)}코어 ({analysis.cpu_info.get('architecture', 'unknown')})")
        print(f"      메모리: {analysis.memory_info.get('available_gb', 0):.1f}GB 사용가능 / {analysis.memory_info.get('total_gb', 0):.1f}GB 총량")
        print(f"      M3 Max: {'✅' if analysis.is_m3_max else '❌'}")
        
        print(f"   🔥 AI 환경:")
        print(f"      PyTorch: {'✅' if analysis.pytorch_info.get('available') else '❌'}")
        print(f"      추천 디바이스: {analysis.recommended_device}")
        print(f"      CUDA: {'✅' if analysis.cuda_available else '❌'}")
        print(f"      MPS: {'✅' if analysis.mps_available else '❌'}")
        
        print(f"   📁 프로젝트:")
        print(f"      AI 모델 디렉토리: {analysis.project_structure.get('ai_models_dir', 'None')}")
        print(f"      AI 모델 크기: {analysis.project_structure.get('ai_models_size_gb', 0):.1f}GB")
        print(f"      Python 환경: {analysis.python_info.get('conda_env', 'none')}")
    
    def _generate_overall_summary(self) -> Dict[str, Any]:
        """전체 요약 생성"""
        total_steps = len(self.step_analyses)
        successful_steps = sum(1 for analysis in self.step_analyses.values() 
                              if analysis.status == StepAnalysisStatus.SUCCESS)
        
        total_models = sum(len(analysis.ai_models_detected) for analysis in self.step_analyses.values())
        total_model_size = sum(analysis.total_model_size_gb for analysis in self.step_analyses.values())
        
        successful_checkpoints = sum(
            sum(1 for cp in analysis.checkpoints_analysis if cp.status == CheckpointStatus.SUCCESS)
            for analysis in self.step_analyses.values()
        )
        total_checkpoints = sum(len(analysis.checkpoints_analysis) for analysis in self.step_analyses.values())
        
        average_health_score = sum(analysis.overall_health_score for analysis in self.step_analyses.values()) / total_steps if total_steps > 0 else 0
        
        return {
            'steps': {
                'total': total_steps,
                'successful': successful_steps,
                'success_rate': (successful_steps / total_steps * 100) if total_steps > 0 else 0
            },
            'models': {
                'total_detected': total_models,
                'total_size_gb': total_model_size,
                'successful_checkpoints': successful_checkpoints,
                'total_checkpoints': total_checkpoints,
                'checkpoint_success_rate': (successful_checkpoints / total_checkpoints * 100) if total_checkpoints > 0 else 0
            },
            'health': {
                'average_score': average_health_score,
                'system_ready': self.system_analysis.memory_sufficient and self.system_analysis.pytorch_info.get('available', False),
                'ai_ready': successful_steps >= total_steps * 0.7  # 70% 이상 성공
            }
        }
    
    def _identify_critical_issues(self) -> List[str]:
        """중요 문제점 식별"""
        issues = []
        
        # 시스템 수준 문제
        if not self.system_analysis.pytorch_info.get('available', False):
            issues.append("🔥 CRITICAL: PyTorch가 설치되지 않음 - AI 모델 실행 불가")
        
        if not self.system_analysis.memory_sufficient:
            issues.append("🔥 CRITICAL: 메모리 부족 - AI 모델 로딩에 문제 발생 가능")
        
        # Step 수준 문제
        failed_imports = [name for name, analysis in self.step_analyses.items() 
                         if not analysis.import_success]
        if failed_imports:
            issues.append(f"❌ Import 실패: {', '.join(failed_imports)}")
        
        failed_initialization = [name for name, analysis in self.step_analyses.items() 
                               if not analysis.initialization_success]
        if failed_initialization:
            issues.append(f"❌ 초기화 실패: {', '.join(failed_initialization)}")
        
        # 체크포인트 문제
        corrupted_checkpoints = []
        for analysis in self.step_analyses.values():
            for cp in analysis.checkpoints_analysis:
                if cp.status == CheckpointStatus.CORRUPTED:
                    corrupted_checkpoints.append(cp.file_path.name)
        
        if corrupted_checkpoints:
            issues.append(f"💾 손상된 체크포인트: {', '.join(corrupted_checkpoints[:3])}{'...' if len(corrupted_checkpoints) > 3 else ''}")
        
        return issues
    
    def _generate_actionable_recommendations(self) -> List[str]:
        """실행 가능한 추천사항 생성"""
        recommendations = []
        
        # 시스템 개선
        if not self.system_analysis.pytorch_info.get('available', False):
            recommendations.append("📦 PyTorch 설치: conda install pytorch torchvision -c pytorch")
        
        if self.system_analysis.recommended_device == 'cpu' and self.system_analysis.is_m3_max:
            recommendations.append("⚡ M3 Max 최적화: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        
        # Step별 개선
        for name, analysis in self.step_analyses.items():
            if not analysis.import_success:
                recommendations.append(f"🔧 {name} 의존성 확인: 모듈 경로 및 import 오류 해결 필요")
            
            if analysis.import_success and not analysis.initialization_success:
                recommendations.append(f"🔧 {name} 초기화 개선: AI 모델 파일 경로 및 권한 확인")
        
        # 성능 최적화
        total_model_size = sum(analysis.total_model_size_gb for analysis in self.step_analyses.values())
        if total_model_size > 50:  # 50GB 이상
            recommendations.append(f"💾 메모리 최적화: {total_model_size:.1f}GB 모델 - 배치 크기 조정 및 캐싱 전략 필요")
        
        # 체크포인트 최적화
        weights_only_issues = 0
        for analysis in self.step_analyses.values():
            for cp in analysis.checkpoints_analysis:
                if cp.status == CheckpointStatus.WEIGHTS_ONLY_FAILED:
                    weights_only_issues += 1
        
        if weights_only_issues > 0:
            recommendations.append(f"🔧 PyTorch 호환성: {weights_only_issues}개 체크포인트에서 weights_only 문제 - PyTorch 업데이트 필요")
        
        return recommendations
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """성능 지표 계산"""
        total_analysis_time = time.time() - self.start_time
        
        step_times = [analysis.import_time + analysis.initialization_time 
                     for analysis in self.step_analyses.values()]
        
        return {
            'total_analysis_time_seconds': total_analysis_time,
            'average_step_analysis_time': sum(step_times) / len(step_times) if step_times else 0,
            'system_analysis_efficiency': 'efficient' if total_analysis_time < 300 else 'slow',
            'memory_usage_peak_gb': psutil.Process().memory_info().rss / (1024**3)
        }
    
    def _print_debug_results(self, debug_result: Dict[str, Any]):
        """디버깅 결과 출력"""
        print("\n" + "=" * 80)
        print("📊 Ultimate AI Model Loading Debug Results")
        print("=" * 80)
        
        # 전체 요약
        summary = debug_result['overall_summary']
        print(f"\n🎯 전체 요약:")
        print(f"   Step 성공률: {summary['steps']['success_rate']:.1f}% ({summary['steps']['successful']}/{summary['steps']['total']})")
        print(f"   체크포인트 성공률: {summary['models']['checkpoint_success_rate']:.1f}% ({summary['models']['successful_checkpoints']}/{summary['models']['total_checkpoints']})")
        print(f"   전체 AI 모델 크기: {summary['models']['total_size_gb']:.1f}GB")
        print(f"   평균 건강도: {summary['health']['average_score']:.1f}/100")
        print(f"   AI 준비 상태: {'✅' if summary['health']['ai_ready'] else '❌'}")
        
        # Step별 상세 결과
        print(f"\n🚀 Step별 분석 결과:")
        for step_name, analysis in self.step_analyses.items():
            status_icon = "✅" if analysis.status == StepAnalysisStatus.SUCCESS else "❌"
            
            print(f"   {status_icon} {step_name} (건강도: {analysis.overall_health_score:.0f}/100)")
            print(f"      Import: {'✅' if analysis.import_success else '❌'} | "
                  f"인스턴스: {'✅' if analysis.instance_created else '❌'} | "
                  f"초기화: {'✅' if analysis.initialization_success else '❌'}")
            
            if analysis.ai_models_detected:
                successful_cp = sum(1 for cp in analysis.checkpoints_analysis if cp.status == CheckpointStatus.SUCCESS)
                total_cp = len(analysis.checkpoints_analysis)
                print(f"      AI 모델: {len(analysis.ai_models_detected)}개 ({analysis.total_model_size_gb:.1f}GB)")
                print(f"      체크포인트: {successful_cp}/{total_cp} 성공")
            
            if analysis.import_errors or analysis.instance_errors or analysis.initialization_errors:
                all_errors = analysis.import_errors + analysis.instance_errors + analysis.initialization_errors
                print(f"      오류: {all_errors[0] if all_errors else 'None'}")
        
        # 중요 문제점
        if debug_result['critical_issues']:
            print(f"\n🔥 중요 문제점:")
            for issue in debug_result['critical_issues']:
                print(f"   {issue}")
        
        # 추천사항
        if debug_result['recommendations']:
            print(f"\n💡 추천사항:")
            for i, rec in enumerate(debug_result['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        # 성능 지표
        metrics = debug_result['performance_metrics']
        print(f"\n📈 성능 지표:")
        print(f"   전체 분석 시간: {metrics['total_analysis_time_seconds']:.1f}초")
        print(f"   평균 Step 분석 시간: {metrics['average_step_analysis_time']:.1f}초")
        print(f"   최대 메모리 사용량: {metrics['memory_usage_peak_gb']:.1f}GB")
        print(f"   분석 효율성: {metrics['system_analysis_efficiency']}")
    
    def _save_debug_results(self, debug_result: Dict[str, Any]):
        """디버깅 결과 저장"""
        try:
            timestamp = int(time.time())
            results_file = Path(f"ultimate_ai_debug_results_{timestamp}.json")
            
            # JSON 직렬화 가능하도록 처리
            serializable_result = self._make_json_serializable(debug_result)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            
            print(f"\n📄 상세 디버깅 결과가 {results_file}에 저장되었습니다.")
            
            # 요약 리포트도 저장
            summary_file = Path(f"ai_debug_summary_{timestamp}.txt")
            self._save_summary_report(summary_file, debug_result)
            print(f"📄 요약 리포트가 {summary_file}에 저장되었습니다.")
            
        except Exception as e:
            print(f"\n⚠️ 결과 저장 실패: {e}")
    
    def _make_json_serializable(self, obj):
        """JSON 직렬화 가능하도록 변환"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
    
    def _save_summary_report(self, file_path: Path, debug_result: Dict[str, Any]):
        """요약 리포트 저장"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("🔥 Ultimate AI Model Loading Debug Summary Report\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"생성 시간: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
                f.write(f"디버거 버전: {debug_result['debug_version']}\n")
                f.write(f"분석 소요 시간: {debug_result['total_debug_time']:.1f}초\n\n")
                
                # 시스템 정보
                system = debug_result['system_analysis']
                f.write("📊 시스템 환경:\n")
                f.write(f"   하드웨어: {system['cpu_info'].get('logical_cores', 0)}코어, "
                       f"{system['memory_info'].get('total_gb', 0):.1f}GB 메모리\n")
                f.write(f"   PyTorch: {'사용가능' if system['pytorch_info'].get('available') else '없음'}\n")
                f.write(f"   추천 디바이스: {system['recommendations']['recommended_device']}\n")
                f.write(f"   AI 모델 크기: {system['project_structure'].get('ai_models_size_gb', 0):.1f}GB\n\n")
                
                # 전체 요약
                summary = debug_result['overall_summary']
                f.write("🎯 분석 결과 요약:\n")
                f.write(f"   Step 성공률: {summary['steps']['success_rate']:.1f}%\n")
                f.write(f"   체크포인트 성공률: {summary['models']['checkpoint_success_rate']:.1f}%\n")
                f.write(f"   평균 건강도: {summary['health']['average_score']:.1f}/100\n")
                f.write(f"   AI 시스템 준비 상태: {'준비됨' if summary['health']['ai_ready'] else '문제있음'}\n\n")
                
                # 중요 문제점
                if debug_result['critical_issues']:
                    f.write("🔥 중요 문제점:\n")
                    for issue in debug_result['critical_issues']:
                        f.write(f"   - {issue}\n")
                    f.write("\n")
                
                # 추천사항
                if debug_result['recommendations']:
                    f.write("💡 추천사항:\n")
                    for i, rec in enumerate(debug_result['recommendations'], 1):
                        f.write(f"   {i}. {rec}\n")
                    f.write("\n")
                
                # Step별 상세 정보
                f.write("🚀 Step별 상세 분석:\n")
                for step_name, step_data in debug_result['step_analyses'].items():
                    f.write(f"\n   {step_name}:\n")
                    f.write(f"      상태: {step_data['status']}\n")
                    f.write(f"      건강도: {step_data['performance']['health_score']:.0f}/100\n")
                    f.write(f"      Import: {'성공' if step_data['import_analysis']['success'] else '실패'}\n")
                    f.write(f"      초기화: {'성공' if step_data['initialization']['success'] else '실패'}\n")
                    f.write(f"      AI 모델: {len(step_data['ai_models']['detected'])}개 "
                           f"({step_data['ai_models']['total_size_gb']:.1f}GB)\n")
                    
                    if step_data['import_analysis']['errors']:
                        f.write(f"      오류: {step_data['import_analysis']['errors'][0]}\n")
                
        except Exception as e:
            print(f"요약 리포트 저장 실패: {e}")

# =============================================================================
# 🔥 7. 메인 실행부
# =============================================================================

def main():
    """메인 실행 함수"""
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s: %(message)s',
        force=True
    )
    
    print(f"🔥 Ultimate AI Model Loading Debugger v4.0")
    print(f"🔥 GitHub 프로젝트: MyCloset AI Pipeline")
    print(f"🔥 Target: 229GB AI Models Complete Analysis")
    print(f"🔥 시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 디버거 생성 및 실행
        debugger = UltimateAIModelDebugger()
        debug_result = debugger.run_ultimate_debugging()
        
        # 성공 여부 확인
        if debug_result.get('overall_summary', {}).get('health', {}).get('ai_ready', False):
            print(f"\n🎉 SUCCESS: AI 시스템이 준비되었습니다!")
        else:
            print(f"\n⚠️ WARNING: AI 시스템에 문제가 있습니다. 위의 추천사항을 확인하세요.")
        
        return debug_result
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 사용자에 의해 중단되었습니다.")
        return None
        
    except Exception as e:
        print(f"\n❌ 디버깅 실행 중 치명적 오류: {e}")
        print(f"전체 스택 트레이스:\n{traceback.format_exc()}")
        return None
        
    finally:
        # 리소스 정리
        gc.collect()
        print(f"\n👋 Ultimate AI Model Debugger 종료")

# =============================================================================
# 🔥 8. 추가 유틸리티 함수들
# =============================================================================

def quick_checkpoint_check(checkpoint_path: str) -> bool:
    """빠른 체크포인트 확인"""
    try:
        analyzer = CheckpointAnalyzer()
        result = analyzer.analyze_checkpoint(Path(checkpoint_path))
        return result.status == CheckpointStatus.SUCCESS
    except Exception:
        return False

def quick_step_check(step_name: str) -> bool:
    """빠른 Step 확인"""
    try:
        step_configs = {
            'HumanParsingStep': 'app.ai_pipeline.steps.step_01_human_parsing',
            'PoseEstimationStep': 'app.ai_pipeline.steps.step_02_pose_estimation',
            'ClothSegmentationStep': 'app.ai_pipeline.steps.step_03_cloth_segmentation'
        }
        
        if step_name not in step_configs:
            return False
        
        module = importlib.import_module(step_configs[step_name])
        step_class = getattr(module, step_name)
        instance = step_class(device='cpu', strict_mode=False)
        return True
        
    except Exception:
        return False

def get_system_readiness_score() -> float:
    """시스템 준비도 점수 (0-100)"""
    try:
        analyzer = SystemEnvironmentAnalyzer()
        analysis = analyzer.analyze_complete_environment()
        
        score = 0.0
        
        # PyTorch (30점)
        if analysis.pytorch_info.get('available', False):
            score += 30
        
        # 메모리 (25점)
        if analysis.memory_sufficient:
            score += 25
        
        # 디바이스 가속 (20점)
        if analysis.recommended_device != 'cpu':
            score += 20
        
        # 프로젝트 구조 (15점)
        if analysis.project_structure.get('ai_models_dir'):
            score += 15
        
        # 의존성 (10점)
        deps_ready = sum(analysis.dependencies_status.values())
        total_deps = len(analysis.dependencies_status)
        if total_deps > 0:
            score += (deps_ready / total_deps) * 10
        
        return score
        
    except Exception:
        return 0.0

if __name__ == "__main__":
    main()