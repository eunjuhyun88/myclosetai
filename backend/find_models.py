#!/usr/bin/env python3
"""
🔥 MyCloset AI 향상된 모델 분석기 v2.0 (실제 데이터 기반)
================================================================================
✅ 실제 스캔 결과(116개 모델, 123.68GB) 기반 최적화
✅ Step별 정확한 분류 및 우선순위
✅ 중복 모델 통합 및 최적화 제안
✅ 메모리 사용량 예측 및 경고
✅ 실제 프로덕션 환경 고려사항
✅ conda 환경 + M3 Max 최적화
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import warnings

@dataclass
class EnhancedModelInfo:
    """향상된 모델 정보 (실제 데이터 구조 기반)"""
    name: str
    path: str
    size_mb: float
    extension: str
    step_category: str
    is_large: bool  # 1GB+
    is_ultra_large: bool  # 5GB+
    is_pytorch_valid: bool = False
    duplicate_group: str = ""
    memory_footprint_estimate: float = 0.0
    production_priority: int = 3  # 1=critical, 2=important, 3=normal, 4=optional
    optimization_suggestion: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedAIModelAnalyzer:
    """향상된 AI 모델 분석기 (실제 스캔 결과 최적화)"""
    
    def __init__(self, search_root: str = "."):
        self.search_root = Path(search_root)
        self.models: List[EnhancedModelInfo] = []
        self.duplicate_groups: Dict[str, List[str]] = defaultdict(list)
        self.memory_estimates: Dict[str, float] = {}
        
        # 실제 데이터 기반 Step 분류 개선
        self.step_patterns = {
            "virtual_fitting": {
                "patterns": ["virtual", "ootd", "diffusion", "stable-diffusion", "sdxl", "v1-5"],
                "priority": 1,  # 최고 우선순위
                "expected_size_range": (1000, 8000),  # 1GB ~ 8GB
                "critical_files": ["v1-5-pruned", "diffusion_pytorch_model"]
            },
            "cloth_segmentation": {
                "patterns": ["cloth", "seg", "sam", "u2net", "RealVis"],
                "priority": 1,
                "expected_size_range": (100, 7000),
                "critical_files": ["sam_vit_h_4b8939", "RealVisXL"]
            },
            "human_parsing": {
                "patterns": ["human", "parsing", "schp", "graphonomy", "atr", "lip"],
                "priority": 2,
                "expected_size_range": (200, 6000),
                "critical_files": ["exp-schp-201908261155-atr", "graphonomy"]
            },
            "pose_estimation": {
                "patterns": ["pose", "openpose", "yolo", "body"],
                "priority": 2,
                "expected_size_range": (10, 1500),
                "critical_files": ["openpose", "yolov8n-pose"]
            },
            "geometric_matching": {
                "patterns": ["geometric", "tps", "gmm", "ViT-L"],
                "priority": 3,
                "expected_size_range": (50, 1500),
                "critical_files": ["tps_network", "ViT-L-14"]
            },
            "post_processing": {
                "patterns": ["post", "enhance", "GFPGAN", "esrgan", "ip-adapter"],
                "priority": 3,
                "expected_size_range": (100, 2000),
                "critical_files": ["GFPGAN", "ip-adapter"]
            },
            "quality_assessment": {
                "patterns": ["quality", "clip", "lpips", "ViT-B"],
                "priority": 4,
                "expected_size_range": (300, 2000),
                "critical_files": ["lpips_vgg", "ViT-B-32"]
            },
            "cloth_warping": {
                "patterns": ["warping", "warp", "tom", "photomaker"],
                "priority": 3,
                "expected_size_range": (500, 1000),
                "critical_files": ["photomaker-v1"]
            }
        }
        
        # PyTorch 설정
        try:
            import torch
            self.torch_available = True
            print("✅ PyTorch 사용 가능")
        except ImportError:
            self.torch_available = False
            print("⚠️ PyTorch 없음 - 기본 분석만 수행")
    
    def scan_and_analyze(self) -> List[EnhancedModelInfo]:
        """완전한 스캔 및 분석"""
        print(f"🔍 향상된 AI 모델 스캔 시작: {self.search_root.absolute()}")
        print("=" * 80)
        
        # 1단계: 기본 스캔
        self._scan_models()
        
        # 2단계: 중복 감지
        self._detect_duplicates()
        
        # 3단계: 메모리 예측
        self._estimate_memory_usage()
        
        # 4단계: 우선순위 및 최적화 제안
        self._analyze_optimization_opportunities()
        
        print(f"✅ 향상된 스캔 완료: {len(self.models)}개 모델 분석")
        return self.models
    
    def _scan_models(self):
        """기본 모델 스캔"""
        extensions = [".pth", ".pt", ".bin", ".safetensors", ".ckpt", ".pkl", ".pickle", ".h5", ".onnx"]
        
        for ext in extensions:
            pattern = f"**/*{ext}"
            for model_path in self.search_root.rglob(pattern):
                if self._should_include_file(model_path):
                    model_info = self._create_enhanced_model_info(model_path)
                    if model_info:
                        self.models.append(model_info)
        
        # 크기순 정렬
        self.models.sort(key=lambda x: x.size_mb, reverse=True)
    
    def _should_include_file(self, file_path: Path) -> bool:
        """파일 포함 여부 결정 (실제 데이터 기준 최적화)"""
        exclude_dirs = ["__pycache__", ".git", "node_modules", ".pytest_cache", "logs"]
        if any(exclude in str(file_path) for exclude in exclude_dirs):
            return False
        
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            return size_mb >= 10.0  # 10MB 이상만
        except:
            return False
    
    def _create_enhanced_model_info(self, file_path: Path) -> EnhancedModelInfo:
        """향상된 모델 정보 생성"""
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Step 카테고리 및 우선순위 감지
            step_category, priority = self._detect_step_and_priority(str(file_path))
            
            # PyTorch 검증
            is_pytorch_valid = False
            metadata = {}
            if self.torch_available and file_path.suffix in [".pth", ".pt"]:
                is_pytorch_valid, metadata = self._validate_pytorch_model(file_path)
            
            return EnhancedModelInfo(
                name=file_path.stem,
                path=str(file_path.relative_to(self.search_root)),
                size_mb=round(size_mb, 2),
                extension=file_path.suffix,
                step_category=step_category,
                is_large=size_mb >= 1000,  # 1GB+
                is_ultra_large=size_mb >= 5000,  # 5GB+
                is_pytorch_valid=is_pytorch_valid,
                production_priority=priority,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"⚠️ 파일 분석 실패: {file_path} - {e}")
            return None
    
    def _detect_step_and_priority(self, file_path: str) -> Tuple[str, int]:
        """Step 카테고리 및 우선순위 감지"""
        file_path_lower = file_path.lower()
        file_name_lower = Path(file_path).name.lower()
        
        for category, info in self.step_patterns.items():
            patterns = info["patterns"]
            
            # 파일명 우선 매칭
            if any(pattern in file_name_lower for pattern in patterns):
                return category, info["priority"]
            
            # 경로 매칭
            if any(pattern in file_path_lower for pattern in patterns):
                return category, info["priority"]
        
        return "unknown", 4
    
    def _validate_pytorch_model(self, file_path: Path) -> Tuple[bool, Dict]:
        """PyTorch 모델 검증 (경고 억제)"""
        try:
            import torch
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
            
            metadata = {
                "type": str(type(checkpoint)),
                "keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
                "tensor_count": 0,
                "total_params": 0
            }
            
            if isinstance(checkpoint, dict):
                for key, value in checkpoint.items():
                    if hasattr(value, 'numel'):
                        metadata["tensor_count"] += 1
                        metadata["total_params"] += value.numel()
            
            return True, metadata
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _detect_duplicates(self):
        """중복 모델 감지 (이름 및 크기 기준)"""
        print("🔍 중복 모델 감지 중...")
        
        # 이름과 크기가 비슷한 모델들 그룹화
        name_size_groups = defaultdict(list)
        
        for model in self.models:
            # 이름 정규화 (버전, 확장자 제거)
            normalized_name = model.name.lower()
            normalized_name = normalized_name.replace('_v1', '').replace('_v2', '').replace('.fp16', '')
            
            # 크기를 100MB 단위로 반올림
            size_bucket = round(model.size_mb / 100) * 100
            
            key = f"{normalized_name}_{size_bucket}"
            name_size_groups[key].append(model)
        
        # 중복 그룹 식별
        for group_key, group_models in name_size_groups.items():
            if len(group_models) > 1:
                self.duplicate_groups[group_key] = [m.path for m in group_models]
                for model in group_models:
                    model.duplicate_group = group_key
    
    def _estimate_memory_usage(self):
        """메모리 사용량 예측"""
        print("💾 메모리 사용량 예측 중...")
        
        for model in self.models:
            # 기본 추정: 파일 크기 * 1.5 (로딩 오버헤드)
            base_estimate = model.size_mb * 1.5
            
            # PyTorch 모델의 경우 더 정확한 추정
            if model.is_pytorch_valid and model.metadata.get("total_params", 0) > 0:
                # 파라미터 수 기반 추정 (fp32 기준)
                params = model.metadata["total_params"]
                param_memory_mb = (params * 4) / (1024 * 1024)  # 4 bytes per param
                model.memory_footprint_estimate = max(base_estimate, param_memory_mb)
            else:
                model.memory_footprint_estimate = base_estimate
    
    def _analyze_optimization_opportunities(self):
        """최적화 기회 분석"""
        print("🎯 최적화 기회 분석 중...")
        
        for model in self.models:
            suggestions = []
            
            # 중복 파일 제안
            if model.duplicate_group:
                suggestions.append("중복 파일 정리 가능")
            
            # 크기 최적화 제안
            if model.is_ultra_large:
                suggestions.append("양자화(fp16) 고려")
            elif model.size_mb > 2000:
                suggestions.append("압축 또는 pruning 고려")
            
            # 사용 빈도 기반 제안
            if model.step_category == "unknown":
                suggestions.append("사용하지 않는 파일일 가능성")
            elif model.production_priority >= 4:
                suggestions.append("선택적 로딩 고려")
            
            model.optimization_suggestion = "; ".join(suggestions) if suggestions else "최적화됨"
    
    def print_enhanced_summary(self):
        """향상된 요약 출력"""
        print("\n" + "=" * 80)
        print("📊 향상된 AI 모델 분석 결과")
        print("=" * 80)
        
        # 전체 통계
        total_count = len(self.models)
        large_count = sum(1 for m in self.models if m.is_large)
        ultra_large_count = sum(1 for m in self.models if m.is_ultra_large)
        total_size_gb = sum(m.size_mb for m in self.models) / 1024
        total_memory_gb = sum(m.memory_footprint_estimate for m in self.models) / 1024
        valid_pytorch = sum(1 for m in self.models if m.is_pytorch_valid)
        duplicate_count = len(self.duplicate_groups)
        
        print(f"📈 전체 통계:")
        print(f"   총 모델 파일: {total_count}개")
        print(f"   대형 모델 (1GB+): {large_count}개")
        print(f"   초대형 모델 (5GB+): {ultra_large_count}개")
        print(f"   디스크 사용량: {total_size_gb:.2f} GB")
        print(f"   예상 메모리 사용량: {total_memory_gb:.2f} GB")
        print(f"   PyTorch 유효: {valid_pytorch}개")
        print(f"   중복 그룹: {duplicate_count}개")
        
        # 우선순위별 분류
        priority_counts = Counter(m.production_priority for m in self.models)
        priority_names = {1: "Critical", 2: "Important", 3: "Normal", 4: "Optional"}
        
        print(f"\n🎯 우선순위별 분류:")
        for priority in sorted(priority_counts.keys()):
            count = priority_counts[priority]
            name = priority_names.get(priority, "Unknown")
            print(f"   {name:10s}: {count:3d}개")
    
    def print_optimization_report(self):
        """최적화 보고서 출력"""
        print(f"\n🔧 최적화 권장사항")
        print("=" * 80)
        
        # 1. 중복 파일 정리
        if self.duplicate_groups:
            print("📁 중복 파일 정리 권장:")
            for group_key, paths in list(self.duplicate_groups.items())[:5]:
                print(f"   그룹 {group_key[:30]}:")
                for path in paths[:3]:  # 상위 3개만 표시
                    print(f"     - {path}")
                if len(paths) > 3:
                    print(f"     ... 외 {len(paths)-3}개")
        
        # 2. 메모리 사용량 경고
        high_memory_models = [m for m in self.models if m.memory_footprint_estimate > 5000]
        if high_memory_models:
            print(f"\n💾 고메모리 모델 (5GB+ 예상):")
            for model in high_memory_models[:5]:
                print(f"   {model.name[:40]:<40} {model.memory_footprint_estimate:>8.1f}MB")
        
        # 3. 최적화 제안 요약
        optimization_counts = Counter()
        for model in self.models:
            if model.optimization_suggestion != "최적화됨":
                for suggestion in model.optimization_suggestion.split("; "):
                    optimization_counts[suggestion] += 1
        
        if optimization_counts:
            print(f"\n🎯 최적화 기회 요약:")
            for suggestion, count in optimization_counts.most_common(5):
                print(f"   {suggestion:<30} {count:3d}개 모델")
        
        # 4. 스토리지 절약 추정
        potential_savings = 0
        for group_models in self.duplicate_groups.values():
            if len(group_models) > 1:
                # 가장 큰 파일만 남기고 나머지 삭제 시 절약량
                models_in_group = [m for m in self.models if m.path in group_models]
                if models_in_group:
                    total_group_size = sum(m.size_mb for m in models_in_group)
                    largest_size = max(m.size_mb for m in models_in_group)
                    potential_savings += total_group_size - largest_size
        
        if potential_savings > 0:
            print(f"\n💾 중복 제거 시 절약 가능: {potential_savings/1024:.2f} GB")
    
    def print_step_production_readiness(self):
        """Step별 프로덕션 준비도 평가"""
        print(f"\n🚀 Step별 프로덕션 준비도")
        print("=" * 80)
        
        step_readiness = {}
        for step_name, step_info in self.step_patterns.items():
            step_models = [m for m in self.models if m.step_category == step_name]
            
            if not step_models:
                readiness = "❌ 모델 없음"
            else:
                critical_files = step_info["critical_files"]
                has_critical = any(
                    any(cf in model.name for cf in critical_files) 
                    for model in step_models
                )
                
                valid_models = [m for m in step_models if m.is_pytorch_valid]
                large_models = [m for m in step_models if m.is_large]
                
                if has_critical and valid_models:
                    readiness = "✅ 준비됨"
                elif has_critical:
                    readiness = "⚠️ 검증 필요"
                else:
                    readiness = "🔧 모델 미완성"
            
            step_readiness[step_name] = readiness
            
            total_size = sum(m.size_mb for m in step_models) / 1024
            print(f"   {step_name.replace('_', ' ').title():<20} {readiness:<12} ({len(step_models):2d}개, {total_size:5.1f}GB)")
    
    def export_enhanced_json(self, output_file: str = "enhanced_ai_models_analysis.json"):
        """향상된 JSON 내보내기"""