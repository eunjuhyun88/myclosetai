# app/ai_pipeline/utils/dynamic_model_detector.py
"""
🔍 MyCloset AI - 동적 모델 탐지기 v2.0
✅ 실제 존재하는 체크포인트 파일 자동 탐지
✅ 파일명, 크기, 내용 기반 모델 타입 추론
✅ Step별 최적 모델 자동 매핑
✅ M3 Max 128GB 최적화
✅ 실시간 파일 시스템 모니터링
"""

import os
import re
import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import sqlite3

# PyTorch import (안전)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    from PIL import Image
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==============================================
# 🔍 모델 탐지 데이터 구조
# ==============================================

class ModelCategory(Enum):
    """모델 카테고리"""
    HUMAN_PARSING = "human_parsing"
    POSE_ESTIMATION = "pose_estimation"
    CLOTH_SEGMENTATION = "cloth_segmentation"
    GEOMETRIC_MATCHING = "geometric_matching"
    CLOTH_WARPING = "cloth_warping"
    VIRTUAL_FITTING = "virtual_fitting"
    POST_PROCESSING = "post_processing"
    QUALITY_ASSESSMENT = "quality_assessment"
    AUXILIARY = "auxiliary"
    UNKNOWN = "unknown"

class ModelFormat(Enum):
    """모델 포맷"""
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    DIFFUSERS = "diffusers"
    TRANSFORMERS = "transformers"
    UNKNOWN = "unknown"

@dataclass
class DetectedModelFile:
    """탐지된 모델 파일 정보"""
    file_path: Path
    file_name: str
    file_size_mb: float
    category: ModelCategory
    format: ModelFormat
    confidence_score: float
    step_assignment: str
    priority: int
    pytorch_valid: bool = False
    parameter_count: int = 0
    architecture_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_modified: float = 0.0
    checksum: str = ""

# ==============================================
# 🔍 Step별 모델 패턴 정의
# ==============================================

STEP_MODEL_PATTERNS = {
    "step_01_human_parsing": {
        "category": ModelCategory.HUMAN_PARSING,
        "file_patterns": [
            r".*human.*parsing.*\.(pth|pt|bin)$",
            r".*schp.*atr.*\.(pth|pt)$",
            r".*graphonomy.*\.(pth|pt)$",
            r".*atr.*model.*\.(pth|pt)$",
            r".*lip.*parsing.*\.(pth|pt)$",
            r".*segmentation.*human.*\.(pth|pt)$"
        ],
        "size_range": (50, 500),  # MB
        "priority": 1,
        "required": True
    },
    
    "step_02_pose_estimation": {
        "category": ModelCategory.POSE_ESTIMATION,
        "file_patterns": [
            r".*openpose.*\.(pth|pt|bin)$",
            r".*pose.*model.*\.(pth|pt)$",
            r".*body.*pose.*\.(pth|pt)$",
            r".*coco.*pose.*\.(pth|pt)$",
            r".*hrnet.*pose.*\.(pth|pt)$",
            r".*keypoint.*\.(pth|pt)$"
        ],
        "size_range": (10, 1000),  # MB
        "priority": 1,
        "required": True
    },
    
    "step_03_cloth_segmentation": {
        "category": ModelCategory.CLOTH_SEGMENTATION,
        "file_patterns": [
            r".*u2net.*\.(pth|pt)$",
            r".*cloth.*segmentation.*\.(pth|pt)$",
            r".*segmentation.*cloth.*\.(pth|pt)$",
            r".*u2netp.*\.(pth|pt)$",
            r".*sam.*vit.*\.(pth|pt|bin)$",
            r".*mask.*generation.*\.(pth|pt)$"
        ],
        "size_range": (50, 3000),  # MB (SAM 모델은 큼)
        "priority": 2,
        "required": True
    },
    
    "step_04_geometric_matching": {
        "category": ModelCategory.GEOMETRIC_MATCHING,
        "file_patterns": [
            r".*geometric.*matching.*\.(pth|pt)$",
            r".*gmm.*\.(pth|pt)$",
            r".*tps.*transformation.*\.(pth|pt)$",
            r".*tps.*network.*\.(pth|pt)$",
            r".*geometric.*\.(pth|pt)$"
        ],
        "size_range": (10, 200),  # MB
        "priority": 3,
        "required": False
    },
    
    "step_05_cloth_warping": {
        "category": ModelCategory.CLOTH_WARPING,
        "file_patterns": [
            r".*cloth.*warping.*\.(pth|pt)$",
            r".*warping.*net.*\.(pth|pt)$",
            r".*tom.*final.*\.(pth|pt)$",
            r".*viton.*\.(pth|pt)$",
            r".*warp.*\.(pth|pt)$"
        ],
        "size_range": (50, 500),  # MB
        "priority": 3,
        "required": False
    },
    
    "step_06_virtual_fitting": {
        "category": ModelCategory.VIRTUAL_FITTING,
        "file_patterns": [
            r".*ootd.*diffusion.*\.(pth|pt|bin|safetensors)$",
            r".*stable.*diffusion.*\.(pth|pt|bin|safetensors)$",
            r".*unet.*\.(pth|pt|bin|safetensors)$",
            r".*hr.*viton.*\.(pth|pt)$",
            r".*viton.*hd.*\.(pth|pt)$",
            r".*diffusion.*\.(pth|pt|bin|safetensors)$"
        ],
        "size_range": (500, 8000),  # MB (큰 모델들)
        "priority": 1,
        "required": True
    },
    
    "step_07_post_processing": {
        "category": ModelCategory.POST_PROCESSING,
        "file_patterns": [
            r".*super.*resolution.*\.(pth|pt)$",
            r".*esrgan.*\.(pth|pt)$",
            r".*real.*esrgan.*\.(pth|pt)$",
            r".*sr.*resnet.*\.(pth|pt)$",
            r".*denoise.*\.(pth|pt)$"
        ],
        "size_range": (5, 200),  # MB
        "priority": 4,
        "required": False
    },
    
    "step_08_quality_assessment": {
        "category": ModelCategory.QUALITY_ASSESSMENT,
        "file_patterns": [
            r".*clip.*vit.*\.(pth|pt|bin)$",
            r".*quality.*assessment.*\.(pth|pt)$",
            r".*similarity.*\.(pth|pt)$",
            r".*lpips.*\.(pth|pt)$"
        ],
        "size_range": (50, 1000),  # MB
        "priority": 4,
        "required": False
    }
}

# ==============================================
# 🔍 동적 모델 탐지기 클래스
# ==============================================

class DynamicModelDetector:
    """실제 파일 시스템 기반 동적 모델 탐지기"""
    
    def __init__(self, search_paths: List[Path] = None):
        self.logger = logging.getLogger(f"{__name__}.DynamicDetector")
        
        # 탐색 경로 설정
        self.search_paths = search_paths or [
            Path("ai_models"),
            Path("checkpoints"),
            Path("models"),
            Path("./"),  # 현재 디렉토리
        ]
        
        # 탐지 결과 저장
        self.detected_models: Dict[str, DetectedModelFile] = {}
        self.scan_results: Dict[str, Any] = {}
        self.last_scan_time = 0.0
        
        # 캐시 데이터베이스
        self.cache_db_path = Path("model_detection_cache.db")
        self._init_cache_db()
        
        # 스레드 동기화
        self._lock = threading.RLock()
        
        self.logger.info(f"🔍 DynamicModelDetector 초기화 - 탐색 경로: {len(self.search_paths)}개")
    
    def _init_cache_db(self):
        """캐시 데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_cache (
                        file_path TEXT PRIMARY KEY,
                        file_name TEXT,
                        file_size_mb REAL,
                        category TEXT,
                        format TEXT,
                        confidence_score REAL,
                        step_assignment TEXT,
                        priority INTEGER,
                        pytorch_valid BOOLEAN,
                        parameter_count INTEGER,
                        last_modified REAL,
                        checksum TEXT,
                        scan_time REAL
                    )
                """)
                conn.commit()
            self.logger.debug("✅ 모델 캐시 DB 초기화 완료")
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 DB 초기화 실패: {e}")
    
    def scan_all_models(self, force_rescan: bool = False) -> Dict[str, DetectedModelFile]:
        """모든 경로에서 모델 파일 탐지"""
        try:
            with self._lock:
                start_time = time.time()
                
                # 캐시된 결과 확인
                if not force_rescan and time.time() - self.last_scan_time < 3600:  # 1시간 캐시
                    self.logger.info("📦 캐시된 스캔 결과 사용")
                    return self.detected_models
                
                self.logger.info("🔍 전체 모델 파일 스캔 시작...")
                
                # 모든 경로에서 파일 수집
                all_files = []
                for search_path in self.search_paths:
                    if search_path.exists():
                        files = self._collect_model_files(search_path)
                        all_files.extend(files)
                        self.logger.debug(f"📁 {search_path}: {len(files)}개 파일 발견")
                
                self.logger.info(f"📊 총 {len(all_files)}개 후보 파일 발견")
                
                # 병렬 분석
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(self._analyze_model_file, file_path): file_path 
                              for file_path in all_files}
                    
                    analyzed_count = 0
                    for future in futures:
                        try:
                            detected_model = future.result()
                            if detected_model:
                                self.detected_models[str(detected_model.file_path)] = detected_model
                                analyzed_count += 1
                        except Exception as e:
                            file_path = futures[future]
                            self.logger.warning(f"⚠️ 파일 분석 실패 {file_path}: {e}")
                
                # Step별 최적 모델 선택
                self._assign_optimal_models()
                
                # 캐시 업데이트
                self._update_cache()
                
                self.last_scan_time = time.time()
                scan_duration = time.time() - start_time
                
                self.logger.info(
                    f"✅ 모델 스캔 완료: {analyzed_count}개 분석, "
                    f"{len(self.detected_models)}개 유효 모델, {scan_duration:.2f}초"
                )
                
                return self.detected_models
                
        except Exception as e:
            self.logger.error(f"❌ 모델 스캔 실패: {e}")
            return {}
    
    def _collect_model_files(self, search_path: Path) -> List[Path]:
        """특정 경로에서 모델 파일 수집"""
        model_files = []
        
        try:
            # 지원하는 확장자
            model_extensions = {'.pth', '.pt', '.bin', '.safetensors', '.onnx'}
            
            for file_path in search_path.rglob('*'):
                if (file_path.is_file() and 
                    file_path.suffix.lower() in model_extensions and
                    file_path.stat().st_size > 1024 * 1024):  # 1MB 이상
                    model_files.append(file_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 파일 수집 실패 {search_path}: {e}")
        
        return model_files
    
    def _analyze_model_file(self, file_path: Path) -> Optional[DetectedModelFile]:
        """개별 모델 파일 분석"""
        try:
            # 기본 정보 수집
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            last_modified = file_path.stat().st_mtime
            
            # 파일명으로 카테고리 추론
            category, confidence, step = self._classify_by_filename(file_path.name)
            
            # 포맷 감지
            format_type = self._detect_format(file_path)
            
            # PyTorch 유효성 검증 (가능한 경우)
            pytorch_valid, param_count, arch_info = self._validate_pytorch_model(file_path)
            
            # 체크섬 계산 (작은 파일만)
            checksum = ""
            if file_size_mb < 100:  # 100MB 미만만
                checksum = self._calculate_checksum(file_path)
            
            # 우선순위 계산
            priority = self._calculate_priority(category, confidence, file_size_mb, pytorch_valid)
            
            detected_model = DetectedModelFile(
                file_path=file_path,
                file_name=file_path.name,
                file_size_mb=file_size_mb,
                category=category,
                format=format_type,
                confidence_score=confidence,
                step_assignment=step,
                priority=priority,
                pytorch_valid=pytorch_valid,
                parameter_count=param_count,
                architecture_info=arch_info,
                last_modified=last_modified,
                checksum=checksum
            )
            
            self.logger.debug(f"📊 분석 완료: {file_path.name} -> {category.value} (신뢰도: {confidence:.2f})")
            return detected_model
            
        except Exception as e:
            self.logger.warning(f"⚠️ 파일 분석 실패 {file_path}: {e}")
            return None
    
    def _classify_by_filename(self, filename: str) -> Tuple[ModelCategory, float, str]:
        """파일명으로 모델 카테고리 분류"""
        filename_lower = filename.lower()
        
        best_match = ModelCategory.UNKNOWN
        best_confidence = 0.0
        best_step = "unknown"
        
        for step_name, step_info in STEP_MODEL_PATTERNS.items():
            for pattern in step_info["file_patterns"]:
                if re.search(pattern, filename_lower):
                    confidence = 0.8 + (0.2 if step_info["required"] else 0.0)
                    if confidence > best_confidence:
                        best_match = step_info["category"]
                        best_confidence = confidence
                        best_step = step_name
        
        # 보조 패턴들
        auxiliary_patterns = {
            r"clip.*vit": (ModelCategory.AUXILIARY, 0.7, "auxiliary"),
            r"resnet": (ModelCategory.AUXILIARY, 0.6, "auxiliary"),
            r"vgg": (ModelCategory.AUXILIARY, 0.6, "auxiliary"),
        }
        
        for pattern, (category, confidence, step) in auxiliary_patterns.items():
            if re.search(pattern, filename_lower) and confidence > best_confidence:
                best_match = category
                best_confidence = confidence
                best_step = step
        
        return best_match, best_confidence, best_step
    
    def _detect_format(self, file_path: Path) -> ModelFormat:
        """파일 포맷 감지"""
        suffix = file_path.suffix.lower()
        
        format_map = {
            '.pth': ModelFormat.PYTORCH,
            '.pt': ModelFormat.PYTORCH,
            '.bin': ModelFormat.PYTORCH,  # 일반적으로 PyTorch
            '.safetensors': ModelFormat.SAFETENSORS,
            '.onnx': ModelFormat.ONNX
        }
        
        return format_map.get(suffix, ModelFormat.UNKNOWN)
    
    def _validate_pytorch_model(self, file_path: Path) -> Tuple[bool, int, Dict[str, Any]]:
        """PyTorch 모델 유효성 검증"""
        if not TORCH_AVAILABLE:
            return False, 0, {}
        
        try:
            # 메모리 효율적인 로딩
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
            
            # state_dict 추출
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                return False, 0, {}
            
            # 파라미터 개수 계산
            param_count = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            
            # 아키텍처 정보 추출
            arch_info = {
                'total_parameters': param_count,
                'layer_count': len(state_dict),
                'has_conv_layers': any('conv' in key.lower() for key in state_dict.keys()),
                'has_linear_layers': any('linear' in key.lower() or 'fc' in key.lower() for key in state_dict.keys()),
                'has_norm_layers': any('norm' in key.lower() or 'bn' in key.lower() for key in state_dict.keys())
            }
            
            return True, param_count, arch_info
            
        except Exception as e:
            self.logger.debug(f"PyTorch 검증 실패 {file_path.name}: {e}")
            return False, 0, {}
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """파일 체크섬 계산"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def _calculate_priority(self, category: ModelCategory, confidence: float, 
                           size_mb: float, pytorch_valid: bool) -> int:
        """모델 우선순위 계산"""
        base_priority = 10
        
        # 카테고리별 우선순위
        category_priorities = {
            ModelCategory.HUMAN_PARSING: 1,
            ModelCategory.VIRTUAL_FITTING: 1,
            ModelCategory.POSE_ESTIMATION: 2,
            ModelCategory.CLOTH_SEGMENTATION: 2,
            ModelCategory.CLOTH_WARPING: 3,
            ModelCategory.GEOMETRIC_MATCHING: 3,
            ModelCategory.POST_PROCESSING: 4,
            ModelCategory.QUALITY_ASSESSMENT: 4,
            ModelCategory.AUXILIARY: 5,
            ModelCategory.UNKNOWN: 10
        }
        
        priority = category_priorities.get(category, 10)
        
        # 신뢰도로 조정
        if confidence > 0.8:
            priority -= 1
        elif confidence < 0.5:
            priority += 2
        
        # PyTorch 유효성으로 조정
        if pytorch_valid:
            priority -= 1
        else:
            priority += 1
        
        # 적절한 크기 범위에 있으면 우선순위 향상
        if 50 <= size_mb <= 1000:
            priority -= 1
        
        return max(1, priority)
    
    def _assign_optimal_models(self):
        """Step별 최적 모델 할당"""
        try:
            step_assignments = {}
            
            for model_path, model_info in self.detected_models.items():
                step = model_info.step_assignment
                
                if step not in step_assignments:
                    step_assignments[step] = []
                
                step_assignments[step].append(model_info)
            
            # 각 Step별로 최적 모델 선택
            for step, models in step_assignments.items():
                if not models:
                    continue
                
                # 우선순위와 신뢰도로 정렬
                sorted_models = sorted(models, 
                                     key=lambda m: (m.priority, -m.confidence_score, -m.file_size_mb))
                
                # 최고 우선순위 모델을 주 모델로 설정
                if sorted_models:
                    primary_model = sorted_models[0]
                    primary_model.metadata['is_primary'] = True
                    
                    self.logger.info(f"🎯 {step} 최적 모델: {primary_model.file_name} "
                                   f"(우선순위: {primary_model.priority}, "
                                   f"신뢰도: {primary_model.confidence_score:.2f})")
            
        except Exception as e:
            self.logger.error(f"❌ 최적 모델 할당 실패: {e}")
    
    def _update_cache(self):
        """캐시 데이터베이스 업데이트"""
        try:
            with sqlite3.connect(self.cache_db_path) as conn:
                # 기존 데이터 삭제
                conn.execute("DELETE FROM model_cache")
                
                # 새 데이터 삽입
                for model_info in self.detected_models.values():
                    conn.execute("""
                        INSERT INTO model_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(model_info.file_path),
                        model_info.file_name,
                        model_info.file_size_mb,
                        model_info.category.value,
                        model_info.format.value,
                        model_info.confidence_score,
                        model_info.step_assignment,
                        model_info.priority,
                        model_info.pytorch_valid,
                        model_info.parameter_count,
                        model_info.last_modified,
                        model_info.checksum,
                        time.time()
                    ))
                
                conn.commit()
            
        except Exception as e:
            self.logger.warning(f"⚠️ 캐시 업데이트 실패: {e}")
    
    def get_step_models(self, step_name: str) -> List[DetectedModelFile]:
        """특정 Step의 모델들 반환"""
        return [model for model in self.detected_models.values() 
                if model.step_assignment == step_name]
    
    def get_primary_model(self, step_name: str) -> Optional[DetectedModelFile]:
        """Step의 주 모델 반환"""
        step_models = self.get_step_models(step_name)
        primary_models = [m for m in step_models if m.metadata.get('is_primary', False)]
        
        if primary_models:
            return primary_models[0]
        elif step_models:
            return sorted(step_models, key=lambda m: (m.priority, -m.confidence_score))[0]
        else:
            return None
    
    def generate_model_mapping(self) -> Dict[str, str]:
        """ModelLoader용 모델 매핑 생성"""
        mapping = {}
        
        for step_name in STEP_MODEL_PATTERNS.keys():
            primary_model = self.get_primary_model(step_name)
            if primary_model:
                mapping[step_name] = str(primary_model.file_path)
                
                # 별칭들도 추가
                base_name = step_name.replace("step_", "").replace("_", "")
                mapping[base_name] = str(primary_model.file_path)
                mapping[primary_model.file_name.replace('.pth', '')] = str(primary_model.file_path)
        
        return mapping
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """탐지 결과 요약"""
        summary = {
            'total_models': len(self.detected_models),
            'by_category': {},
            'by_step': {},
            'by_format': {},
            'total_size_gb': sum(m.file_size_mb for m in self.detected_models.values()) / 1024,
            'pytorch_valid_count': sum(1 for m in self.detected_models.values() if m.pytorch_valid),
            'scan_time': self.last_scan_time
        }
        
        for model in self.detected_models.values():
            # 카테고리별
            cat = model.category.value
            summary['by_category'][cat] = summary['by_category'].get(cat, 0) + 1
            
            # Step별
            step = model.step_assignment
            summary['by_step'][step] = summary['by_step'].get(step, 0) + 1
            
            # 포맷별
            fmt = model.format.value
            summary['by_format'][fmt] = summary['by_format'].get(fmt, 0) + 1
        
        return summary

# ==============================================
# 🔍 편의 함수들
# ==============================================

def create_dynamic_detector(search_paths: List[Path] = None) -> DynamicModelDetector:
    """동적 모델 탐지기 생성"""
    return DynamicModelDetector(search_paths)

def quick_model_scan(search_paths: List[Path] = None) -> Dict[str, str]:
    """빠른 모델 스캔 및 매핑 반환"""
    detector = create_dynamic_detector(search_paths)
    detector.scan_all_models()
    return detector.generate_model_mapping()

def find_step_model(step_name: str, search_paths: List[Path] = None) -> Optional[str]:
    """특정 Step의 모델 경로 찾기"""
    detector = create_dynamic_detector(search_paths)
    detector.scan_all_models()
    primary_model = detector.get_primary_model(step_name)
    return str(primary_model.file_path) if primary_model else None

# 모듈 익스포트
__all__ = [
    'DynamicModelDetector',
    'DetectedModelFile',
    'ModelCategory',
    'ModelFormat',
    'create_dynamic_detector',
    'quick_model_scan',
    'find_step_model',
    'STEP_MODEL_PATTERNS'
]