#!/usr/bin/env python3
"""
🔍 MyCloset AI - 모델 검색 및 재배치 스크립트
M3 Max conda 환경 최적화 버전

기능:
- 전체 시스템에서 AI 모델 파일 검색
- 중복 파일 탐지 (체크섬 기반)
- 안전한 재배치 (복사본 생성, 순서 번호 자동 추가)
- conda 환경 호환

사용법:
python search_and_relocate_models.py --scan-only     # 검색만
python search_and_relocate_models.py --relocate     # 검색 후 재배치
python search_and_relocate_models.py --target-dir ./ai_models  # 사용자 지정 경로
"""

import os
import sys
import hashlib
import shutil
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import argparse
import re

# 안전한 import (conda 환경 호환)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️ tqdm 없음. 진행률 표시 불가")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

@dataclass
class ModelFile:
    """AI 모델 파일 정보"""
    path: Path
    name: str
    size_mb: float
    extension: str
    checksum: str = ""
    model_type: str = "unknown"
    framework: str = "unknown"
    confidence: float = 0.0
    duplicate_group: int = 0
    metadata: Dict = field(default_factory=dict)

class ModelSearcher:
    """AI 모델 검색기"""
    
    def __init__(self, target_dir: Optional[Path] = None):
        self.target_dir = target_dir or Path.cwd() / "ai_models"
        self.discovered_models: List[ModelFile] = []
        self.duplicate_groups: Dict[str, List[ModelFile]] = defaultdict(list)
        
        # AI 모델 파일 확장자 패턴
        self.model_extensions = {
            '.pth', '.pt', '.bin', '.safetensors', '.ckpt', '.h5', 
            '.pb', '.onnx', '.tflite', '.pkl', '.pickle', '.model',
            '.weights', '.params', '.caffemodel', '.prototxt'
        }
        
        # AI 모델 키워드 패턴
        self.ai_keywords = [
            # Framework 특정
            'pytorch', 'tensorflow', 'torch', 'transformers', 'diffusers',
            'huggingface', 'openai', 'anthropic', 'stability',
            
            # 모델 아키텍처
            'resnet', 'vgg', 'inception', 'mobilenet', 'efficientnet',
            'bert', 'gpt', 'clip', 'vit', 'swin', 'deit',
            'unet', 'vae', 'gan', 'diffusion', 'stable',
            
            # CV 태스크
            'detection', 'segmentation', 'classification', 'pose',
            'parsing', 'openpose', 'yolo', 'rcnn', 'ssd', 'sam',
            'u2net', 'graphonomy', 'schp', 'atr', 'hrnet',
            
            # NLP 태스크  
            'language', 'text', 'embedding', 'tokenizer',
            
            # Virtual Try-on 특화
            'viton', 'tryon', 'ootd', 'garment', 'cloth', 'fashion',
            'warping', 'fitting', 'geometric', 'matching',
            
            # 일반 AI
            'pretrained', 'checkpoint', 'model', 'weights',
            'encoder', 'decoder', 'backbone', 'head'
        ]
        
        # 검색 경로들 (우선순위 순)
        self.search_paths = self._get_search_paths()
        
        print(f"🎯 타겟 디렉토리: {self.target_dir}")
        print(f"🔍 검색 경로: {len(self.search_paths)}개")
        
    def _get_search_paths(self) -> List[Path]:
        """검색 경로 목록 생성"""
        paths = []
        
        # 1. 현재 프로젝트 경로들
        current_dir = Path.cwd()
        project_paths = [
            current_dir / "ai_models",
            current_dir / "backend" / "ai_models", 
            current_dir / "models",
            current_dir / "checkpoints",
            current_dir / "weights",
            current_dir.parent / "ai_models",
        ]
        
        # 2. 시스템 캐시 경로들
        home = Path.home()
        cache_paths = [
            home / ".cache" / "huggingface" / "hub",
            home / ".cache" / "huggingface" / "transformers",
            home / ".cache" / "torch" / "hub",
            home / ".cache" / "torch" / "checkpoints",
            home / ".cache" / "models",
            home / ".torch" / "models",
        ]
        
        # 3. conda 환경 경로들
        conda_paths = []
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            conda_base = os.environ.get('CONDA_PREFIX', home / "anaconda3")
            conda_paths.extend([
                Path(conda_base) / "envs" / conda_env / "lib" / "python3.10" / "site-packages",
                Path(conda_base) / "pkgs",
            ])
        
        # 4. 일반적인 다운로드 경로들
        download_paths = [
            home / "Downloads",
            home / "Desktop",
            home / "Documents" / "AI_Models",
            home / "Documents" / "models",
        ]
        
        # 모든 경로 병합
        all_paths = project_paths + cache_paths + conda_paths + download_paths
        
        # 존재하고 접근 가능한 경로만 필터링
        valid_paths = []
        for path in all_paths:
            try:
                if path.exists() and path.is_dir() and os.access(path, os.R_OK):
                    valid_paths.append(path)
            except (OSError, PermissionError):
                continue
                
        return valid_paths
    
    def _is_ai_model_file(self, file_path: Path) -> Tuple[bool, float]:
        """AI 모델 파일인지 판단 (신뢰도 포함)"""
        try:
            file_name = file_path.name.lower()
            file_stem = file_path.stem.lower()
            path_str = str(file_path).lower()
            
            # 확장자 확인
            if file_path.suffix.lower() not in self.model_extensions:
                return False, 0.0
            
            confidence = 0.1  # 기본 확장자 점수
            
            # 키워드 매칭
            keyword_matches = 0
            for keyword in self.ai_keywords:
                if keyword in file_name or keyword in path_str:
                    keyword_matches += 1
                    confidence += 0.15
            
            # 파일 크기 고려 (AI 모델은 보통 1MB 이상)
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb >= 1.0:
                    confidence += 0.2
                if size_mb >= 10.0:
                    confidence += 0.2
                if size_mb >= 100.0:
                    confidence += 0.3
            except OSError:
                pass
            
            # 경로 기반 추가 점수
            path_keywords = ['models', 'checkpoints', 'weights', 'pretrained', 'hub']
            for path_keyword in path_keywords:
                if path_keyword in path_str:
                    confidence += 0.1
            
            # 특별한 파일명 패턴
            special_patterns = [
                r'.*model.*\.(pth|pt|bin)$',
                r'.*checkpoint.*\.(pth|ckpt)$', 
                r'.*weights.*\.(pth|h5)$',
                r'pytorch_model\.bin$',
                r'.*diffusion.*\.(pth|safetensors)$'
            ]
            
            for pattern in special_patterns:
                if re.match(pattern, file_name):
                    confidence += 0.3
                    break
            
            # 최종 판단 (신뢰도 0.3 이상이면 AI 모델로 간주)
            is_ai_model = confidence >= 0.3
            return is_ai_model, min(confidence, 1.0)
            
        except Exception:
            return False, 0.0
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """파일 체크섬 계산 (중복 탐지용)"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                # 큰 파일을 위해 청크 단위로 읽기
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()[:16]  # 처음 16자리만 사용
        except Exception as e:
            print(f"⚠️ 체크섬 계산 실패 {file_path}: {e}")
            return ""
    
    def _detect_model_type(self, file_path: Path) -> Tuple[str, str]:
        """모델 타입과 프레임워크 추정"""
        file_name = file_path.name.lower()
        path_str = str(file_path).lower()
        
        # 프레임워크 탐지
        framework = "unknown"
        if any(keyword in path_str for keyword in ['pytorch', 'torch', '.pth', '.pt']):
            framework = "pytorch"
        elif any(keyword in path_str for keyword in ['tensorflow', '.pb', '.h5']):
            framework = "tensorflow"
        elif '.onnx' in file_name:
            framework = "onnx"
        elif '.safetensors' in file_name:
            framework = "safetensors"
        
        # 모델 타입 탐지
        model_type = "unknown"
        if any(keyword in file_name for keyword in ['parsing', 'schp', 'atr', 'graphonomy']):
            model_type = "human_parsing"
        elif any(keyword in file_name for keyword in ['pose', 'openpose', 'yolo.*pose']):
            model_type = "pose_estimation"
        elif any(keyword in file_name for keyword in ['u2net', 'segmentation', 'sam']):
            model_type = "segmentation"
        elif any(keyword in file_name for keyword in ['diffusion', 'stable', 'ootd', 'viton']):
            model_type = "virtual_fitting"
        elif any(keyword in file_name for keyword in ['clip', 'vit', 'bert', 'gpt']):
            model_type = "foundation_model"
        elif any(keyword in file_name for keyword in ['resnet', 'mobilenet', 'efficientnet']):
            model_type = "backbone"
        elif any(keyword in file_name for keyword in ['esrgan', 'enhancement']):
            model_type = "post_processing"
        
        return model_type, framework
    
    def search_models(self) -> List[ModelFile]:
        """모든 경로에서 AI 모델 검색"""
        print("🔍 AI 모델 검색 시작...")
        
        all_files = []
        
        # 각 경로에서 파일 수집
        for search_path in self.search_paths:
            print(f"📂 검색 중: {search_path}")
            try:
                for file_path in search_path.rglob("*"):
                    if file_path.is_file():
                        all_files.append(file_path)
            except (PermissionError, OSError) as e:
                print(f"⚠️ 접근 불가: {search_path} - {e}")
                continue
        
        print(f"📊 총 {len(all_files)}개 파일 발견, AI 모델 필터링 중...")
        
        # AI 모델 파일 필터링
        iterator = tqdm(all_files, desc="AI 모델 분석") if TQDM_AVAILABLE else all_files
        
        for file_path in iterator:
            try:
                is_ai, confidence = self._is_ai_model_file(file_path)
                if not is_ai:
                    continue
                
                # 파일 정보 수집
                stat = file_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                
                # 체크섬 계산 (1GB 미만 파일만)
                checksum = ""
                if size_mb < 1024:  # 1GB 미만
                    checksum = self._calculate_checksum(file_path)
                
                model_type, framework = self._detect_model_type(file_path)
                
                model_file = ModelFile(
                    path=file_path,
                    name=file_path.name,
                    size_mb=size_mb,
                    extension=file_path.suffix.lower(),
                    checksum=checksum,
                    model_type=model_type,
                    framework=framework,
                    confidence=confidence,
                    metadata={
                        'modified_time': stat.st_mtime,
                        'relative_path': str(file_path.relative_to(file_path.anchor)),
                    }
                )
                
                self.discovered_models.append(model_file)
                
            except (OSError, PermissionError):
                continue
        
        print(f"✅ {len(self.discovered_models)}개 AI 모델 발견!")
        return self.discovered_models
    
    def find_duplicates(self) -> Dict[str, List[ModelFile]]:
        """중복 파일 찾기 (체크섬 기반)"""
        print("🔍 중복 파일 검색 중...")
        
        checksum_groups = defaultdict(list)
        
        for model in self.discovered_models:
            if model.checksum:  # 체크섬이 있는 파일만
                checksum_groups[model.checksum].append(model)
        
        # 중복이 있는 그룹만 선별
        duplicates = {k: v for k, v in checksum_groups.items() if len(v) > 1}
        
        print(f"📊 중복 그룹: {len(duplicates)}개")
        for checksum, files in duplicates.items():
            print(f"   체크섬 {checksum}: {len(files)}개 파일")
            for file in files:
                print(f"     - {file.path} ({file.size_mb:.1f}MB)")
        
        self.duplicate_groups = duplicates
        return duplicates
    
    def generate_report(self, output_file: Optional[Path] = None) -> Dict:
        """검색 결과 리포트 생성"""
        report = {
            "scan_info": {
                "timestamp": time.time(),
                "search_paths": [str(p) for p in self.search_paths],
                "total_files_found": len(self.discovered_models),
                "total_size_gb": sum(m.size_mb for m in self.discovered_models) / 1024,
                "duplicate_groups": len(self.duplicate_groups)
            },
            "models_by_type": defaultdict(list),
            "models_by_framework": defaultdict(list),
            "models_by_size": {"small": [], "medium": [], "large": []},
            "duplicates": {},
            "all_models": []
        }
        
        # 모델 분류
        for model in self.discovered_models:
            # 타입별
            report["models_by_type"][model.model_type].append({
                "path": str(model.path),
                "name": model.name,
                "size_mb": model.size_mb,
                "confidence": model.confidence
            })
            
            # 프레임워크별
            report["models_by_framework"][model.framework].append({
                "path": str(model.path),
                "name": model.name,
                "size_mb": model.size_mb
            })
            
            # 크기별
            if model.size_mb < 100:
                category = "small"
            elif model.size_mb < 1000:
                category = "medium"  
            else:
                category = "large"
            
            report["models_by_size"][category].append({
                "path": str(model.path),
                "name": model.name,
                "size_mb": model.size_mb
            })
            
            # 전체 목록
            report["all_models"].append({
                "path": str(model.path),
                "name": model.name,
                "size_mb": model.size_mb,
                "type": model.model_type,
                "framework": model.framework,
                "checksum": model.checksum,
                "confidence": model.confidence
            })
        
        # 중복 파일 정보
        for checksum, files in self.duplicate_groups.items():
            report["duplicates"][checksum] = [
                {
                    "path": str(f.path),
                    "name": f.name,
                    "size_mb": f.size_mb
                } for f in files
            ]
        
        # 파일 저장
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 리포트 저장: {output_file}")
        
        return report
    
    def create_relocation_plan(self, copy_mode: bool = True) -> Dict:
        """재배치 계획 생성 - Step별 구조로 개선"""
        print("📋 재배치 계획 생성 중...")
        
        # Step별 타겟 디렉토리 구조 (사용자 요청 반영)
        base_checkpoints = self.target_dir / "app" / "ai_pipeline" / "models" / "checkpoints"
        target_structure = {
            "step_01_human_parsing": base_checkpoints / "step_01_human_parsing",
            "step_02_pose_estimation": base_checkpoints / "step_02_pose_estimation", 
            "step_03_cloth_segmentation": base_checkpoints / "step_03_cloth_segmentation",
            "step_04_geometric_matching": base_checkpoints / "step_04_geometric_matching",
            "step_05_cloth_warping": base_checkpoints / "step_05_cloth_warping",
            "step_06_virtual_fitting": base_checkpoints / "step_06_virtual_fitting",
            "step_07_post_processing": base_checkpoints / "step_07_post_processing", 
            "step_08_quality_assessment": base_checkpoints / "step_08_quality_assessment",
            "misc": base_checkpoints / "misc"
        }
        
        # 디렉토리 생성
        for dir_path in target_structure.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        relocation_plan = {
            "copy_mode": copy_mode,
            "target_structure": {k: str(v) for k, v in target_structure.items()},
            "operations": [],
            "conflicts": [],
            "summary": {
                "total_files": len(self.discovered_models),
                "total_size_gb": sum(m.size_mb for m in self.discovered_models) / 1024,
                "operations_count": 0
            }
        }
        
        # 파일명 충돌 방지용 카운터 (Step별로 관리)
        filename_counters = defaultdict(lambda: defaultdict(int))
        
        for model in self.discovered_models:
            # Step별 타겟 폴더 결정 (모델 타입과 경로 분석)
            target_folder = self._determine_step_folder(model, target_structure)
            
            # 타겟 파일명 생성 (중복 시 순서 번호 추가)
            base_name = model.path.stem
            extension = model.path.suffix
            folder_name = target_folder.name
            
            filename_counters[folder_name][model.name] += 1
            if filename_counters[folder_name][model.name] > 1:
                target_name = f"{base_name}_{filename_counters[folder_name][model.name]:02d}{extension}"
            else:
                target_name = model.name
            
            target_path = target_folder / target_name
            
            operation = {
                "source": str(model.path),
                "target": str(target_path),
                "method": "copy" if copy_mode else "move",
                "size_mb": model.size_mb,
                "model_type": model.model_type,
                "framework": model.framework,
                "reason": f"Size: {model.size_mb:.1f}MB, Type: {model.model_type}"
            }
            
            # 충돌 확인
            if target_path.exists():
                operation["conflict"] = True
                relocation_plan["conflicts"].append(operation)
            else:
                operation["conflict"] = False
                relocation_plan["operations"].append(operation)
        
        relocation_plan["summary"]["operations_count"] = len(relocation_plan["operations"])
        
        return relocation_plan
    
    def _determine_step_folder(self, model: ModelFile, target_structure: Dict[str, Path]) -> Path:
        """모델 타입과 파일명 분석하여 올바른 Step 폴더 결정"""
        file_name = model.path.name.lower()
        path_str = str(model.path).lower()
        
        # Step 01: Human Parsing
        if (model.model_type == "human_parsing" or 
            any(keyword in file_name for keyword in [
                'parsing', 'schp', 'atr', 'lip', 'graphonomy', 'densepose', 
                'exp-schp-201908301523-atr', 'exp-schp-201908261155-lip',
                'segformer_b2_clothes', 'humanparsing'
            ]) or
            any(keyword in path_str for keyword in [
                'human_parsing', 'step_01', 'step_1'
            ])):
            return target_structure["step_01_human_parsing"]
        
        # Step 02: Pose Estimation  
        elif (model.model_type == "pose_estimation" or
              any(keyword in file_name for keyword in [
                  'pose', 'openpose', 'body_pose_model', 'yolov8n-pose',
                  'pose_landmark', 'mediapipe', 'pose_deploy'
              ]) or
              any(keyword in path_str for keyword in [
                  'pose_estimation', 'openpose', 'step_02', 'step_2'
              ])):
            return target_structure["step_02_pose_estimation"]
        
        # Step 03: Cloth Segmentation
        elif (model.model_type == "segmentation" or
              any(keyword in file_name for keyword in [
                  'u2net', 'mobile_sam', 'sam_vit', 'cloth_segmentation',
                  'background_removal', 'segmentation'
              ]) or
              any(keyword in path_str for keyword in [
                  'cloth_segmentation', 'step_03', 'step_3', 'u2net'
              ])):
            return target_structure["step_03_cloth_segmentation"]
        
        # Step 04: Geometric Matching
        elif (any(keyword in file_name for keyword in [
                'gmm', 'geometric', 'matching', 'tps_network', 
                'geometric_matching', 'lightweight_gmm'
              ]) or
              any(keyword in path_str for keyword in [
                  'geometric_matching', 'step_04', 'step_4'
              ])):
            return target_structure["step_04_geometric_matching"]
        
        # Step 05: Cloth Warping
        elif (any(keyword in file_name for keyword in [
                'warping', 'cloth_warping', 'tom_final', 'tps'
              ]) or
              any(keyword in path_str for keyword in [
                  'cloth_warping', 'step_05', 'step_5'
              ])):
            return target_structure["step_05_cloth_warping"]
        
        # Step 06: Virtual Fitting
        elif (model.model_type == "virtual_fitting" or
              any(keyword in file_name for keyword in [
                  'ootd', 'diffusion', 'vton', 'viton', 'unet_vton',
                  'text_encoder', 'vae', 'stable_diffusion', 'ootdiffusion'
              ]) or
              any(keyword in path_str for keyword in [
                  'virtual_fitting', 'ootdiffusion', 'step_06', 'step_6',
                  'stable-diffusion'
              ])):
            return target_structure["step_06_virtual_fitting"]
        
        # Step 07: Post Processing
        elif (model.model_type == "post_processing" or
              any(keyword in file_name for keyword in [
                  'esrgan', 'realesrgan', 'gfpgan', 'codeformer',
                  'enhancement', 'super_resolution', 'post_processing'
              ]) or
              any(keyword in path_str for keyword in [
                  'post_processing', 'step_07', 'step_7'
              ])):
            return target_structure["step_07_post_processing"]
        
        # Step 08: Quality Assessment
        elif (any(keyword in file_name for keyword in [
                'clip', 'quality', 'assessment', 'vit_base', 'vit_large'
              ]) or
              any(keyword in path_str for keyword in [
                  'quality_assessment', 'step_08', 'step_8', 'clip-vit'
              ])):
            return target_structure["step_08_quality_assessment"]
        
        # 기타 분류되지 않은 파일들
        else:
            return target_structure["misc"]
    
    def execute_relocation(self, plan: Dict, dry_run: bool = True) -> bool:
        """재배치 실행"""
        if dry_run:
            print("🔍 DRY RUN - 실제 파일 이동 없음")
        else:
            print("🚀 실제 재배치 시작...")
        
        operations = plan["operations"]
        iterator = tqdm(operations, desc="파일 재배치") if TQDM_AVAILABLE else operations
        
        success_count = 0
        error_count = 0
        
        for operation in iterator:
            source = Path(operation["source"])
            target = Path(operation["target"])
            method = operation["method"]
            
            if dry_run:
                print(f"{'COPY' if method == 'copy' else 'MOVE'}: {source} → {target}")
                success_count += 1
                continue
            
            try:
                # 타겟 디렉토리 생성
                target.parent.mkdir(parents=True, exist_ok=True)
                
                if method == "copy":
                    shutil.copy2(source, target)
                    print(f"✅ 복사 완료: {target.name}")
                else:
                    shutil.move(str(source), str(target))
                    print(f"✅ 이동 완료: {target.name}")
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ 실패: {source} → {target}: {e}")
                error_count += 1
        
        print(f"\n📊 재배치 결과:")
        print(f"   성공: {success_count}개")
        print(f"   실패: {error_count}개")
        
        return error_count == 0

def main():
    parser = argparse.ArgumentParser(description="AI 모델 검색 및 재배치 도구")
    parser.add_argument("--scan-only", action="store_true", help="검색만 수행")
    parser.add_argument("--relocate", action="store_true", help="검색 후 재배치")
    parser.add_argument("--target-dir", type=Path, default=Path.cwd() / "backend", help="타겟 디렉토리")
    parser.add_argument("--dry-run", action="store_true", help="실제 이동 없이 시뮬레이션")
    parser.add_argument("--output", type=Path, default="model_search_report.json", help="리포트 파일명")
    
    args = parser.parse_args()
    
    print("🤖 MyCloset AI 모델 재배치 도구 v2.0")
    print("=" * 50)
    print(f"🎯 타겟: backend/app/ai_pipeline/models/checkpoints/")
    
    # 검색기 초기화
    searcher = ModelSearcher(target_dir=args.target_dir)
    
    # 모델 검색
    models = searcher.search_models()
    if not models:
        print("❌ AI 모델을 찾을 수 없습니다.")
        return
    
    # 중복 찾기
    duplicates = searcher.find_duplicates()
    
    # 리포트 생성
    report = searcher.generate_report(output_file=args.output)
    
    # 결과 요약 출력
    print("\n📊 검색 결과 요약:")
    print(f"   총 모델 수: {len(models)}개")
    print(f"   총 크기: {sum(m.size_mb for m in models) / 1024:.1f}GB")
    print(f"   중복 그룹: {len(duplicates)}개")
    
    print("\n🏷️ 모델 타입별 분포:")
    type_counts = defaultdict(int)
    for model in models:
        type_counts[model.model_type] += 1
    for model_type, count in sorted(type_counts.items()):
        print(f"   {model_type}: {count}개")
    
    print("\n📁 Step별 예상 배치:")
    step_preview = defaultdict(int)
    target_structure = {
        "step_01_human_parsing": Path("step_01_human_parsing"),
        "step_02_pose_estimation": Path("step_02_pose_estimation"),
        "step_03_cloth_segmentation": Path("step_03_cloth_segmentation"),
        "step_04_geometric_matching": Path("step_04_geometric_matching"),
        "step_05_cloth_warping": Path("step_05_cloth_warping"),
        "step_06_virtual_fitting": Path("step_06_virtual_fitting"),
        "step_07_post_processing": Path("step_07_post_processing"),
        "step_08_quality_assessment": Path("step_08_quality_assessment"),
        "misc": Path("misc")
    }
    
    for model in models:
        step_folder = searcher._determine_step_folder(model, target_structure)
        step_preview[step_folder.name] += 1
    
    for step, count in sorted(step_preview.items()):
        print(f"   {step}: {count}개")
    
    # 재배치 수행 (요청된 경우)
    if args.relocate or not args.scan_only:
        print("\n" + "=" * 50)
        plan = searcher.create_relocation_plan(copy_mode=True)  # 항상 복사 모드
        
        print(f"📋 재배치 계획:")
        print(f"   이동할 파일: {len(plan['operations'])}개")
        print(f"   충돌 파일: {len(plan['conflicts'])}개")
        
        if plan['conflicts']:
            print("\n⚠️ 충돌 파일들 (일부 표시):")
            for conflict in plan['conflicts'][:5]:  # 처음 5개만 표시
                print(f"   {conflict['target']}")
        
        # 사용자 확인
        if not args.dry_run:
            confirm = input("\n재배치를 실행하시겠습니까? [y/N]: ")
            if confirm.lower() != 'y':
                print("❌ 재배치 취소")
                return
        
        # 재배치 실행
        success = searcher.execute_relocation(plan, dry_run=args.dry_run)
        
        if success:
            print("✅ 재배치 완료!")
            print(f"\n📁 정리된 구조:")
            print(f"   backend/app/ai_pipeline/models/checkpoints/")
            for step in sorted(step_preview.keys()):
                if step_preview[step] > 0:
                    print(f"   ├── {step}/ ({step_preview[step]}개)")
        else:
            print("⚠️ 일부 오류가 발생했습니다.")

if __name__ == "__main__":
    main()