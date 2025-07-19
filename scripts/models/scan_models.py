#!/usr/bin/env python3
"""
🔍 MyCloset AI - 수정된 모델 자동 스캔 스크립트 (macOS 최적화)
================================================================

macOS 권한 문제 해결 및 안전한 경로 스캔으로 수정

지원 모델 형식:
- PyTorch: .pth, .pt, .bin
- TensorFlow: .pb, .h5, .tflite  
- ONNX: .onnx
- Safetensors: .safetensors
- Pickle: .pkl, .p
- Caffe: .caffemodel, .prototxt

사용법:
    python scan_models.py                    # 안전한 기본 스캔
    python scan_models.py --safe             # 안전 모드 (권한 문제 방지)
    python scan_models.py --path ./ai_models # 특정 경로 스캔
    python scan_models.py --create-config    # 설정 파일 생성
"""

import os
import sys
import time
import json
import hashlib
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import platform
import glob

# 진행률 표시를 위한 간단한 구현
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.total = len(iterable) if hasattr(iterable, '__len__') else None
            self.desc = kwargs.get('desc', '')
            self.current = 0
            
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.current += 1
                if self.total:
                    print(f"\r{self.desc}: {self.current}/{self.total} ({self.current/self.total*100:.1f}%)", end='', flush=True)
            print()

# ==============================================
# 🎯 모델 정보 데이터 클래스
# ==============================================

@dataclass
class ModelInfo:
    """AI 모델 정보"""
    name: str
    path: str
    size_mb: float
    format: str
    framework: str
    created_time: str
    modified_time: str
    checksum: str
    is_valid: bool
    model_type: str = "unknown"
    step_candidate: str = "unknown"
    confidence: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

# ==============================================
# 🔍 수정된 모델 스캐너 클래스
# ==============================================

class AIModelScanner:
    """AI 모델 자동 스캔 및 분류 (macOS 최적화)"""
    
    def __init__(self, verbose: bool = True, safe_mode: bool = True):
        self.verbose = verbose
        self.safe_mode = safe_mode
        self.found_models: List[ModelInfo] = []
        self.scan_stats = {
            'total_files_scanned': 0,
            'models_found': 0,
            'total_size_gb': 0.0,
            'scan_time': 0.0,
            'errors': []
        }
        
        # 스캔할 파일 확장자
        self.model_extensions = {
            '.pth': 'pytorch',
            '.pt': 'pytorch', 
            '.bin': 'pytorch',
            '.safetensors': 'safetensors',
            '.onnx': 'onnx',
            '.pb': 'tensorflow',
            '.h5': 'tensorflow',
            '.tflite': 'tensorflow',
            '.pkl': 'pickle',
            '.p': 'pickle',
            '.caffemodel': 'caffe',
            '.prototxt': 'caffe'
        }
        
        # MyCloset AI Step별 모델 패턴
        self.step_patterns = {
            'step_01_human_parsing': [
                'human.*parsing', 'graphonomy', 'schp', 'atr', 'lip',
                'parsing', 'segmentation.*human'
            ],
            'step_02_pose_estimation': [
                'pose.*estimation', 'openpose', 'mediapipe', 'pose.*net',
                'body.*pose', 'keypoint', 'skeleton'
            ],
            'step_03_cloth_segmentation': [
                'cloth.*seg', 'u2net', 'sam', 'segment.*anything',
                'mask.*rcnn', 'deeplabv3', 'segmentation.*cloth'
            ],
            'step_04_geometric_matching': [
                'geometric.*matching', 'gmm', 'tps', 'spatial.*transform',
                'warping.*grid', 'flow.*estimation'
            ],
            'step_05_cloth_warping': [
                'cloth.*warp', 'tom', 'viton.*warp', 'deformation',
                'elastic.*transform', 'thin.*plate.*spline'
            ],
            'step_06_virtual_fitting': [
                'virtual.*fitting', 'ootdiffusion', 'stable.*diffusion',
                'diffusion.*unet', 'text2img', 'img2img', 'viton',
                'try.*on', 'outfit'
            ],
            'step_07_post_processing': [
                'post.*process', 'enhancement', 'super.*resolution',
                'srresnet', 'esrgan', 'real.*esrgan', 'upscal',
                'denoise', 'refine'
            ],
            'step_08_quality_assessment': [
                'quality.*assessment', 'clip', 'aesthetic', 'scoring',
                'evaluation', 'metric', 'lpips', 'ssim'
            ]
        }
        
        # 안전한 스캔 경로 (권한 문제 방지)
        self.safe_paths = self._get_safe_scan_paths()
        
    def _get_safe_scan_paths(self) -> List[str]:
        """권한 문제가 없는 안전한 스캔 경로"""
        home = Path.home()
        
        # 기본 안전 경로
        safe_paths = [
            # 현재 프로젝트
            "./ai_models",
            "./models", 
            "./checkpoints",
            "./weights",
            "./pretrained",
            ".",
            
            # 사용자 접근 가능 경로
            str(home / "Downloads"),
            str(home / "Documents"),
            str(home / "Desktop"),
            
            # Python/AI 관련 캐시 (안전함)
            str(home / ".cache" / "huggingface"),
            str(home / ".cache" / "torch"),
            str(home / ".cache" / "transformers"),
            str(home / ".local" / "lib"),
        ]
        
        # macOS 특화 안전 경로
        if platform.system().lower() == "darwin":
            safe_paths.extend([
                str(home / "Library" / "Caches" / "pip"),
                str(home / "Library" / "Caches" / "huggingface"),
                "/opt/homebrew/lib",  # Homebrew 설치 경로
                "/usr/local/lib",     # 기본 라이브러리 경로
            ])
        
        # 존재하고 읽기 가능한 경로만 반환
        valid_paths = []
        for path_str in safe_paths:
            path = Path(path_str)
            try:
                if path.exists() and os.access(path, os.R_OK):
                    valid_paths.append(path_str)
            except (PermissionError, OSError):
                continue
        
        return valid_paths
    
    def scan_system(
        self, 
        custom_paths: List[str] = None,
        deep_scan: bool = False,
        max_workers: int = 2  # macOS에서 안정성을 위해 줄임
    ) -> List[ModelInfo]:
        """시스템 AI 모델 스캔 (안전 모드)"""
        
        print("🔍 AI 모델 스캔 시작...")
        print(f"🖥️ 시스템: {platform.system()} {platform.release()}")
        
        if self.safe_mode:
            print("🛡️ 안전 모드: 권한 문제가 있는 경로는 자동으로 건너뜀")
        
        start_time = time.time()
        
        # 스캔 경로 결정
        if custom_paths:
            # 사용자 지정 경로 검증
            scan_paths = []
            for path_str in custom_paths:
                path = Path(path_str)
                if path.exists() and os.access(path, os.R_OK):
                    scan_paths.append(path)
                else:
                    print(f"⚠️ 접근 불가능한 경로 건너뜀: {path_str}")
        elif deep_scan and not self.safe_mode:
            print("⚠️ 딥 스캔은 권한 문제가 발생할 수 있습니다. --safe 옵션을 권장합니다.")
            scan_paths = self._get_deep_scan_paths()
        else:
            scan_paths = [Path(p) for p in self.safe_paths]
        
        print(f"📂 스캔 경로: {len(scan_paths)}개")
        for i, path in enumerate(scan_paths):
            if i < 5:  # 처음 5개만 표시
                print(f"   - {path}")
            elif i == 5:
                print(f"   ... 외 {len(scan_paths) - 5}개")
        
        # 안전한 스캔 실행
        all_files = []
        for path in scan_paths:
            try:
                files = self._find_model_files_safe(path)
                all_files.extend(files)
                if self.verbose:
                    print(f"📁 {path}: {len(files)}개 파일 발견")
            except Exception as e:
                error_msg = f"경로 스캔 실패 {path}: {e}"
                self.scan_stats['errors'].append(error_msg)
                if self.verbose:
                    print(f"⚠️ {error_msg}")
        
        print(f"\n🔎 총 {len(all_files)}개 모델 파일 발견, 분석 시작...")
        
        if len(all_files) == 0:
            print("❌ 모델 파일을 찾을 수 없습니다.")
            self._suggest_alternatives()
            return []
        
        # 순차 처리 (안정성 우선)
        if len(all_files) > 100 or not self.safe_mode:
            # 병렬 처리
            processed_files = self._process_files_parallel(all_files, max_workers)
        else:
            # 순차 처리 (더 안전)
            processed_files = self._process_files_sequential(all_files)
        
        # 스캔 통계 업데이트
        self.scan_stats['scan_time'] = time.time() - start_time
        self.scan_stats['models_found'] = len(self.found_models)
        self.scan_stats['total_size_gb'] = sum(m.size_mb for m in self.found_models) / 1024
        
        self._print_scan_results()
        return self.found_models
    
    def _process_files_sequential(self, files: List[Path]) -> int:
        """파일 순차 처리 (안전)"""
        processed = 0
        
        for i, file_path in enumerate(files):
            if self.verbose and i % 10 == 0:
                print(f"🔍 처리 중: {i+1}/{len(files)} ({(i+1)/len(files)*100:.1f}%)")
            
            try:
                model_info = self._analyze_model_file(file_path)
                if model_info and model_info.is_valid:
                    self.found_models.append(model_info)
                processed += 1
                self.scan_stats['total_files_scanned'] += 1
            except Exception as e:
                error_msg = f"파일 분석 실패 {file_path}: {e}"
                self.scan_stats['errors'].append(error_msg)
        
        return processed
    
    def _process_files_parallel(self, files: List[Path], max_workers: int) -> int:
        """파일 병렬 처리"""
        processed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._analyze_model_file, file_path): file_path 
                      for file_path in files}
            
            if TQDM_AVAILABLE:
                progress = tqdm(as_completed(futures), total=len(futures), desc="모델 분석")
            else:
                progress = as_completed(futures)
                
            for future in progress:
                try:
                    model_info = future.result()
                    if model_info and model_info.is_valid:
                        self.found_models.append(model_info)
                    processed += 1
                    self.scan_stats['total_files_scanned'] += 1
                except Exception as e:
                    error_msg = f"모델 분석 실패: {e}"
                    self.scan_stats['errors'].append(error_msg)
        
        return processed
    
    def _get_deep_scan_paths(self) -> List[Path]:
        """딥 스캔용 경로 (주의: 권한 문제 가능)"""
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            return [
                Path("/Users"),  # 전체 /를 피하고 Users만
                Path("/opt"),
                Path("/usr/local")
            ]
        elif system == "linux":
            return [Path("/home"), Path("/opt"), Path("/usr")]
        else:
            return [Path.home()]
    
    def _find_model_files_safe(self, root_path: Path) -> List[Path]:
        """안전한 모델 파일 찾기 (권한 오류 방지)"""
        model_files = []
        
        try:
            # 각 확장자별로 glob 패턴 사용 (더 안전)
            for ext in self.model_extensions.keys():
                try:
                    # 재귀적 glob 사용
                    pattern = str(root_path / f"**/*{ext}")
                    files = glob.glob(pattern, recursive=True)
                    
                    for file_str in files:
                        file_path = Path(file_str)
                        if file_path.is_file() and os.access(file_path, os.R_OK):
                            model_files.append(file_path)
                
                except (PermissionError, OSError) as e:
                    if self.verbose:
                        print(f"⚠️ {ext} 파일 검색 중 권한 오류: {e}")
                    continue
                
        except (PermissionError, OSError) as e:
            if self.verbose:
                print(f"⚠️ 경로 접근 권한 오류 {root_path}: {e}")
        
        return model_files
    
    def _analyze_model_file(self, file_path: Path) -> Optional[ModelInfo]:
        """개별 모델 파일 분석 (안전 처리)"""
        try:
            # 파일 접근 권한 확인
            if not os.access(file_path, os.R_OK):
                return None
            
            # 기본 파일 정보
            stat = file_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            
            # 너무 작은 파일은 제외 (0.5MB 미만)
            if size_mb < 0.5:
                return None
            
            # 파일 형식 확인
            suffix = file_path.suffix.lower()
            framework = self.model_extensions.get(suffix, "unknown")
            
            # 체크섬 계산 (안전하게)
            checksum = self._calculate_checksum_safe(file_path, size_mb)
            
            # Step 분류
            step_candidate, confidence = self._classify_model_step(file_path.name.lower())
            
            # 모델 타입 추론
            model_type = self._infer_model_type(file_path, framework)
            
            # 검증 (안전하게)
            is_valid = self._validate_model_file_safe(file_path, framework)
            
            model_info = ModelInfo(
                name=file_path.name,
                path=str(file_path.absolute()),
                size_mb=round(size_mb, 2),
                format=suffix,
                framework=framework,
                created_time=time.ctime(stat.st_ctime),
                modified_time=time.ctime(stat.st_mtime),
                checksum=checksum,
                is_valid=is_valid,
                model_type=model_type,
                step_candidate=step_candidate,
                confidence=confidence,
                metadata={
                    'parent_dir': file_path.parent.name,
                    'depth': len(file_path.parts),
                    'has_config': self._check_config_files_safe(file_path.parent)
                }
            )
            
            return model_info
            
        except (PermissionError, OSError, IOError) as e:
            # 권한 관련 오류는 조용히 무시
            return None
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 파일 분석 실패 {file_path}: {e}")
            return None
    
    def _calculate_checksum_safe(self, file_path: Path, size_mb: float) -> str:
        """안전한 체크섬 계산"""
        try:
            hasher = hashlib.md5()
            
            with open(file_path, 'rb') as f:
                if size_mb > 50:  # 50MB 이상은 샘플링
                    # 처음 512KB
                    chunk = f.read(512 * 1024)
                    if chunk:
                        hasher.update(chunk)
                    
                    # 중간 지점
                    try:
                        f.seek(int(size_mb * 1024 * 512))  # 중간 지점
                        chunk = f.read(512 * 1024)
                        if chunk:
                            hasher.update(chunk)
                    except:
                        pass
                    
                    # 끝 512KB
                    try:
                        f.seek(-512 * 1024, 2)
                        chunk = f.read(512 * 1024)
                        if chunk:
                            hasher.update(chunk)
                    except:
                        pass
                else:
                    # 작은 파일은 전체 읽기
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        hasher.update(chunk)
            
            return hasher.hexdigest()[:12]
            
        except Exception:
            return "unknown"
    
    def _classify_model_step(self, filename: str) -> Tuple[str, float]:
        """파일명으로 Step 분류"""
        best_step = "unknown"
        best_confidence = 0.0
        
        for step_name, patterns in self.step_patterns.items():
            confidence = 0.0
            
            for pattern in patterns:
                # 정규식 패턴 매칭
                import re
                try:
                    if re.search(pattern.replace('.*', '.*?'), filename, re.IGNORECASE):
                        confidence = max(confidence, 0.8)
                    elif pattern.lower() in filename:
                        confidence = max(confidence, 0.6)
                    elif any(word in filename for word in pattern.split('.*') if word):
                        confidence = max(confidence, 0.4)
                except:
                    # 정규식 오류 시 단순 문자열 매칭
                    if pattern.lower() in filename:
                        confidence = max(confidence, 0.5)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_step = step_name
        
        return best_step, best_confidence
    
    def _infer_model_type(self, file_path: Path, framework: str) -> str:
        """모델 타입 추론"""
        filename = file_path.name.lower()
        
        # 딥러닝 아키텍처 패턴
        if any(arch in filename for arch in ['resnet', 'vgg', 'densenet', 'efficientnet']):
            return "cnn_backbone"
        elif any(arch in filename for arch in ['transformer', 'bert', 'gpt', 'clip']):
            return "transformer"
        elif any(arch in filename for arch in ['unet', 'vae', 'diffusion']):
            return "generative"
        elif any(arch in filename for arch in ['pose', 'keypoint', 'openpose']):
            return "pose_estimation"
        elif any(arch in filename for arch in ['segment', 'mask', 'parsing']):
            return "segmentation"
        else:
            return "unknown"
    
    def _validate_model_file_safe(self, file_path: Path, framework: str) -> bool:
        """안전한 모델 파일 유효성 검사"""
        try:
            # 파일 크기 기반 기본 검증
            file_size = file_path.stat().st_size
            if file_size < 1024:  # 1KB 미만은 유효하지 않음
                return False
            
            # 확장자 기반 검증
            suffix = file_path.suffix.lower()
            if suffix in ['.pth', '.pt', '.bin']:
                return self._validate_pytorch_model_safe(file_path)
            elif suffix in ['.pb', '.h5']:
                return self._validate_tensorflow_model_safe(file_path)
            elif suffix == '.onnx':
                return self._validate_onnx_model_safe(file_path)
            else:
                return True  # 다른 형식은 일단 유효로 간주
                
        except Exception:
            return False
    
    def _validate_pytorch_model_safe(self, file_path: Path) -> bool:
        """안전한 PyTorch 모델 검증"""
        try:
            # 헤더만 읽어서 빠른 검증
            with open(file_path, 'rb') as f:
                header = f.read(1024)
            
            # PyTorch/Pickle 매직 바이트 확인
            pytorch_markers = [b'PK', b'\x80', b'PYTORCH', b'PICKLE']
            if any(marker in header for marker in pytorch_markers):
                return True
            
            # torch 모듈이 있으면 실제 로드 시도 (작은 파일만)
            try:
                import torch
                if file_path.stat().st_size < 10 * 1024 * 1024:  # 10MB 미만
                    torch.load(file_path, map_location='cpu')
                    return True
            except ImportError:
                pass
            except Exception:
                pass
            
            return True  # 확장자가 맞으면 일단 유효로 간주
            
        except Exception:
            return False
    
    def _validate_tensorflow_model_safe(self, file_path: Path) -> bool:
        """안전한 TensorFlow 모델 검증"""
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.pb':
                # Protocol Buffer 매직 바이트 확인
                with open(file_path, 'rb') as f:
                    header = f.read(100)
                return len(header) > 10  # 기본적인 크기 확인
                
            elif suffix == '.h5':
                # HDF5 매직 바이트 확인
                with open(file_path, 'rb') as f:
                    header = f.read(8)
                return header.startswith(b'\x89HDF')
                
            return True
            
        except Exception:
            return False
    
    def _validate_onnx_model_safe(self, file_path: Path) -> bool:
        """안전한 ONNX 모델 검증"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(100)
            
            # ONNX 관련 바이트 패턴 확인
            return b'onnx' in header.lower() or len(header) > 50
            
        except Exception:
            return False
    
    def _check_config_files_safe(self, directory: Path) -> bool:
        """안전한 설정 파일 존재 확인"""
        try:
            config_patterns = [
                'config.json', 'config.yaml', 'config.yml',
                'model_config.json', 'tokenizer.json',
                'pytorch_model.bin', 'model.safetensors'
            ]
            
            for pattern in config_patterns:
                config_file = directory / pattern
                if config_file.exists() and os.access(config_file, os.R_OK):
                    return True
            
            return False
            
        except (PermissionError, OSError):
            return False
    
    def _suggest_alternatives(self):
        """모델을 찾을 수 없을 때 대안 제안"""
        print("\n💡 모델을 찾을 수 없습니다. 다음을 시도해보세요:")
        print("   1. 특정 경로 지정: python scan_models.py --path ~/Downloads")
        print("   2. 프로젝트 폴더만: python scan_models.py --path ./ai_models")
        print("   3. Hugging Face 캐시: python scan_models.py --path ~/.cache/huggingface")
        print("   4. 최소 크기 줄이기: 스크립트 내 size_mb < 0.1 로 수정")
        print("\n📁 일반적인 모델 위치:")
        print("   - ./ai_models/")
        print("   - ./models/") 
        print("   - ~/Downloads/")
        print("   - ~/.cache/huggingface/")
        print("   - ~/.cache/torch/")
    
    def _print_scan_results(self):
        """스캔 결과 출력"""
        print(f"\n{'='*60}")
        print(f"🎉 AI 모델 스캔 완료!")
        print(f"{'='*60}")
        
        stats = self.scan_stats
        print(f"📊 스캔 통계:")
        print(f"   - 스캔 시간: {stats['scan_time']:.1f}초")
        print(f"   - 검사한 파일: {stats['total_files_scanned']:,}개")
        print(f"   - 발견한 모델: {stats['models_found']:,}개")
        print(f"   - 총 크기: {stats['total_size_gb']:.2f}GB")
        
        if stats['errors']:
            print(f"   - 경고/오류: {len(stats['errors'])}개")
        
        if len(self.found_models) == 0:
            return
        
        # Step별 분류 결과
        print(f"\n🎯 Step별 모델 분류:")
        step_counts = {}
        for model in self.found_models:
            step = model.step_candidate
            if step not in step_counts:
                step_counts[step] = 0
            step_counts[step] += 1
        
        step_names = {
            'step_01_human_parsing': 'Human Parsing',
            'step_02_pose_estimation': 'Pose Estimation',
            'step_03_cloth_segmentation': 'Cloth Segmentation',
            'step_04_geometric_matching': 'Geometric Matching',
            'step_05_cloth_warping': 'Cloth Warping',
            'step_06_virtual_fitting': 'Virtual Fitting',
            'step_07_post_processing': 'Post Processing',
            'step_08_quality_assessment': 'Quality Assessment'
        }
        
        for step, count in sorted(step_counts.items()):
            if count > 0:
                display_name = step_names.get(step, step)
                print(f"   - {display_name}: {count}개")
        
        # 프레임워크별 통계
        framework_counts = {}
        for model in self.found_models:
            fw = model.framework
            framework_counts[fw] = framework_counts.get(fw, 0) + 1
        
        print(f"\n🔧 프레임워크별 분류:")
        for fw, count in framework_counts.items():
            print(f"   - {fw}: {count}개")
        
        # 상위 모델들 표시
        print(f"\n🏆 발견된 주요 모델들:")
        sorted_models = sorted(self.found_models, key=lambda x: x.size_mb, reverse=True)
        
        for i, model in enumerate(sorted_models[:10]):
            print(f"   {i+1:2d}. {model.name}")
            print(f"       📁 {model.path}")
            print(f"       📊 {model.size_mb:.1f}MB | {model.framework}")
            if model.confidence > 0.5:
                step_display = step_names.get(model.step_candidate, model.step_candidate)
                print(f"       🎯 {step_display} (신뢰도: {model.confidence:.1f})")
            print()
    
    def generate_config_files(self, output_dir: str = "."):
        """설정 파일들 생성"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📝 설정 파일 생성 중... -> {output_path}")
        
        # JSON 요약 파일
        self._generate_json_summary(output_path)
        
        # Python 설정 파일  
        self._generate_python_config(output_path)
        
        # YAML 설정 파일 (PyYAML 있을 때만)
        self._generate_yaml_config(output_path)
        
        # Shell 환경 변수 파일
        self._generate_env_file(output_path)
        
        print(f"✅ 설정 파일 생성 완료!")
    
    def _generate_json_summary(self, output_path: Path):
        """JSON 요약 파일 생성"""
        summary = {
            'scan_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'system': platform.system(),
                'python_version': platform.python_version(),
                'total_models': len(self.found_models),
                'total_size_gb': self.scan_stats['total_size_gb'],
                'scan_time': self.scan_stats['scan_time']
            },
            'models': [asdict(model) for model in self.found_models],
            'statistics': self.scan_stats
        }
        
        json_file = output_path / "discovered_models.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ JSON 요약: {json_file}")
    
    def _generate_python_config(self, output_path: Path):
        """Python 설정 파일 생성"""
        
        config_content = f'''"""
AI 모델 경로 설정 (자동 생성)
Generated by MyCloset AI Model Scanner
스캔 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

from pathlib import Path
from typing import Dict, List, Optional

# 스캔 정보
SCAN_TIMESTAMP = "{time.strftime('%Y-%m-%d %H:%M:%S')}"
TOTAL_MODELS_FOUND = {len(self.found_models)}
TOTAL_SIZE_GB = {self.scan_stats['total_size_gb']:.2f}

# 발견된 모델들
DISCOVERED_MODELS = {{
'''
        
        # Step별 모델 분류
        step_models = {}
        for model in self.found_models:
            step = model.step_candidate
            if step not in step_models:
                step_models[step] = []
            step_models[step].append(model)
        
        for step_name, models in step_models.items():
            config_content += f'    "{step_name}": [\n'
            
            for model in models:
                config_content += f'''        {{
            "name": "{model.name}",
            "path": Path(r"{model.path}"),
            "size_mb": {model.size_mb},
            "framework": "{model.framework}",
            "confidence": {model.confidence:.2f},
            "checksum": "{model.checksum}",
            "model_type": "{model.model_type}"
        }},
'''
            config_content += '    ],\n'
        
        config_content += '''}

def get_models_for_step(step_name: str) -> List[Dict]:
    """특정 Step의 모델 목록 반환"""
    return DISCOVERED_MODELS.get(step_name, [])

def get_best_model_for_step(step_name: str) -> Optional[Dict]:
    """특정 Step의 최적 모델 반환"""
    models = get_models_for_step(step_name)
    if not models:
        return None
    
    # 신뢰도와 크기를 고려해서 최적 모델 선택
    def score_model(m):
        confidence_score = m["confidence"]
        size_score = min(m["size_mb"] / 100, 1.0)  # 100MB를 기준으로 정규화
        return confidence_score * 0.7 + size_score * 0.3
    
    best_model = max(models, key=score_model)
    return best_model

def get_all_model_paths() -> List[str]:
    """모든 모델 경로 목록 반환"""
    paths = []
    for step_models in DISCOVERED_MODELS.values():
        for model in step_models:
            paths.append(str(model["path"]))
    return paths

def validate_model_exists(model_dict: Dict) -> bool:
    """모델 파일 존재 여부 확인"""
    return model_dict["path"].exists()

def list_models_by_framework(framework: str) -> List[Dict]:
    """프레임워크별 모델 목록"""
    result = []
    for step_models in DISCOVERED_MODELS.values():
        for model in step_models:
            if model["framework"] == framework:
                result.append(model)
    return result

def get_largest_models(limit: int = 5) -> List[Dict]:
    """크기가 큰 모델들 반환"""
    all_models = []
    for step_models in DISCOVERED_MODELS.values():
        all_models.extend(step_models)
    
    return sorted(all_models, key=lambda m: m["size_mb"], reverse=True)[:limit]

# 편의 함수들
def print_scan_summary():
    """스캔 요약 출력"""
    print(f"🔍 AI 모델 스캔 결과 (스캔 시간: {{SCAN_TIMESTAMP}})")
    print(f"📊 총 {{TOTAL_MODELS_FOUND}}개 모델 발견 ({{TOTAL_SIZE_GB:.2f}}GB)")
    print()
    
    for step_name, models in DISCOVERED_MODELS.items():
        if models and step_name != "unknown":
            print(f"  🎯 {{step_name}}: {{len(models)}}개")
            for model in models[:3]:  # 최대 3개만 표시
                print(f"     - {{model['name']}} ({{model['size_mb']}}MB)")
            if len(models) > 3:
                print(f"     ... 외 {{len(models)-3}}개")
            print()

if __name__ == "__main__":
    print_scan_summary()
'''
        
        py_file = output_path / "model_paths.py"
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"   ✅ Python 설정: {py_file}")
    
    def _generate_yaml_config(self, output_path: Path):
        """YAML 설정 파일 생성 (선택적)"""
        try:
            import yaml
            
            # Step별 모델 그룹화
            step_models = {}
            for model in self.found_models:
                step = model.step_candidate
                if step not in step_models:
                    step_models[step] = []
                
                step_models[step].append({
                    'name': model.name,
                    'path': model.path,
                    'size_mb': model.size_mb,
                    'framework': model.framework,
                    'confidence': model.confidence,
                    'checksum': model.checksum,
                    'model_type': model.model_type
                })
            
            yaml_config = {
                'scan_info': {
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_models': len(self.found_models),
                    'total_size_gb': round(self.scan_stats['total_size_gb'], 2),
                    'scan_time_seconds': round(self.scan_stats['scan_time'], 1)
                },
                'models_by_step': step_models
            }
            
            yaml_file = output_path / "models_config.yaml"
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
            
            print(f"   ✅ YAML 설정: {yaml_file}")
            
        except ImportError:
            print("   ⚠️ PyYAML이 없어 YAML 파일을 생성할 수 없습니다")
            print("      설치: pip install pyyaml")
    
    def _generate_env_file(self, output_path: Path):
        """환경 변수 파일 생성"""
        
        env_content = f"""#!/bin/bash
# AI 모델 환경 변수 (자동 생성)
# Generated by MyCloset AI Model Scanner on {time.strftime('%Y-%m-%d %H:%M:%S')}

# 스캔 정보
export MYCLOSET_MODELS_SCAN_DATE="{time.strftime('%Y-%m-%d')}"
export MYCLOSET_TOTAL_MODELS="{len(self.found_models)}"
export MYCLOSET_TOTAL_SIZE_GB="{self.scan_stats['total_size_gb']:.2f}"

# Step별 최적 모델 경로
"""
        
        # 각 Step별 최적 모델 경로
        step_models = {}
        for model in self.found_models:
            step = model.step_candidate
            if step not in step_models or model.confidence > step_models[step].confidence:
                step_models[step] = model
        
        for step_name, model in step_models.items():
            if model.confidence > 0.5 and step_name.startswith('step_'):  # 신뢰도 높은 것만
                env_var_name = step_name.upper()
                env_content += f'export {env_var_name}_MODEL_PATH="{model.path}"\n'
                env_content += f'export {env_var_name}_MODEL_SIZE="{model.size_mb}"\n'
                env_content += f'export {env_var_name}_MODEL_FRAMEWORK="{model.framework}"\n\n'
        
        env_content += f'''
# 편의 함수들
mycloset_list_models() {{
    echo "🔍 MyCloset AI 발견된 모델들:"
    echo "   총 $MYCLOSET_TOTAL_MODELS개 모델 ($MYCLOSET_TOTAL_SIZE_GB GB)"
    echo "   스캔일: $MYCLOSET_MODELS_SCAN_DATE"
    echo ""
    env | grep "STEP_.*_MODEL_PATH" | sort
}}

# 사용법 출력
if [[ "${{BASH_SOURCE[0]}}" == "${{0}}" ]]; then
    echo "🔍 MyCloset AI 모델 환경변수 로드됨"
    echo "📊 총 $MYCLOSET_TOTAL_MODELS개 모델 ($MYCLOSET_TOTAL_SIZE_GB GB)"
    echo ""
    echo "💡 사용법:"
    echo "   source {output_path / "models_env.sh"}"
    echo "   mycloset_list_models"
    echo "   echo \\$STEP_01_HUMAN_PARSING_MODEL_PATH"
fi
'''
        
        env_file = output_path / "models_env.sh"
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        # 실행 권한 부여 (Unix 계열)
        try:
            import stat
            env_file.chmod(env_file.stat().st_mode | stat.S_IEXEC)
        except:
            pass
        
        print(f"   ✅ 환경변수 파일: {env_file}")

# ==============================================
# 🚀 CLI 인터페이스
# ==============================================

def main():
    parser = argparse.ArgumentParser(
        description="MyCloset AI 모델 자동 스캔 도구 (macOS 최적화)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python scan_models.py                           # 안전한 기본 스캔
  python scan_models.py --safe                    # 안전 모드 (권장)
  python scan_models.py --path ./ai_models        # 특정 경로 스캔
  python scan_models.py --create-config           # 설정 파일 생성
  python scan_models.py --path ~/Downloads --create-config  # Downloads 스캔 + 설정 생성
        """
    )
    
    parser.add_argument(
        '--path', '-p',
        type=str,
        nargs='+',
        help='스캔할 특정 경로들'
    )
    
    parser.add_argument(
        '--safe',
        action='store_true',
        default=True,
        help='안전 모드 (권한 문제 방지, 기본값)'
    )
    
    parser.add_argument(
        '--unsafe',
        action='store_true',
        help='안전 모드 해제 (권한 문제 발생 가능)'
    )
    
    parser.add_argument(
        '--deep',
        action='store_true',
        help='딥 스캔 (시간 오래 걸림, --unsafe와 함께 사용)'
    )
    
    parser.add_argument(
        '--create-config', '-c',
        action='store_true',
        help='설정 파일들 자동 생성'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./configs',
        help='설정 파일 출력 디렉토리 (기본: ./configs)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=2,
        help='병렬 처리 워커 수 (기본: 2, macOS 안정성)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='조용한 모드 (최소 출력)'
    )
    
    args = parser.parse_args()
    
    # 안전 모드 설정
    safe_mode = args.safe and not args.unsafe
    
    if args.deep and safe_mode:
        print("⚠️ 딥 스캔은 --unsafe 옵션과 함께 사용하세요")
        print("   예: python scan_models.py --deep --unsafe")
        return 1
    
    # 스캐너 초기화
    scanner = AIModelScanner(verbose=not args.quiet, safe_mode=safe_mode)
    
    # 스캔 실행
    try:
        models = scanner.scan_system(
            custom_paths=args.path,
            deep_scan=args.deep,
            max_workers=args.workers
        )
        
        # 설정 파일 생성
        if args.create_config and models:
            scanner.generate_config_files(args.output_dir)
            
            # 사용법 안내
            print(f"\n📋 생성된 설정 파일 사용법:")
            print(f"   Python: from {args.output_dir.replace('./', '')}.model_paths import get_best_model_for_step")
            print(f"   Shell:  source {args.output_dir}/models_env.sh")
            print(f"   JSON:   cat {args.output_dir}/discovered_models.json")
        
        return 0 if models else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())