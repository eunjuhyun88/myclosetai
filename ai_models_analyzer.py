#!/usr/bin/env python3
"""
🔍 AI Models Directory Analyzer
현재 backend/ai_models 디렉토리의 실제 상황을 완전히 분석하는 스크립트

Usage:
    python ai_models_analyzer.py [--scan-only] [--detailed] [--export-json]
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Optional, Tuple, Any
import logging

# 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelFileInfo:
    """모델 파일 정보"""
    name: str
    path: str
    size_mb: float
    extension: str
    step_category: Optional[str]
    is_checkpoint: bool
    is_config: bool
    is_link: bool
    link_target: Optional[str]
    hash_preview: Optional[str]  # 파일 시작 부분의 해시
    potential_duplicates: List[str]
    accessibility: str  # 'readable', 'permission_denied', 'broken_link'
    pytorch_loadable: Optional[bool]
    estimated_parameters: Optional[int]

@dataclass
class DirectoryStats:
    """디렉토리 통계"""
    total_files: int
    total_size_gb: float
    by_extension: Dict[str, int]
    by_step: Dict[str, int]
    duplicate_groups: List[List[str]]
    broken_links: List[str]
    large_files: List[Tuple[str, float]]  # (파일명, 크기GB)
    
class AIModelsAnalyzer:
    """AI 모델 디렉토리 분석기"""
    
    # 지원하는 모델 파일 확장자
    MODEL_EXTENSIONS = {
        '.pth', '.pt',           # PyTorch
        '.safetensors',          # SafeTensors
        '.onnx',                 # ONNX
        '.pkl', '.pickle',       # Pickle
        '.bin',                  # Binary
        '.h5',                   # HDF5
        '.tflite',               # TensorFlow Lite
        '.pb',                   # TensorFlow
        '.engine',               # TensorRT
    }
    
    # 설정 파일 확장자
    CONFIG_EXTENSIONS = {
        '.json', '.yaml', '.yml', '.txt', '.cfg', '.config', '.prototxt'
    }
    
    # 단계별 키워드 (디렉토리/파일명에서 추출)
    STEP_KEYWORDS = {
        'step_01_human_parsing': ['human_parsing', 'parsing', 'segformer', 'schp', 'graphonomy', 'atr', 'lip'],
        'step_02_pose_estimation': ['pose', 'openpose', 'mediapipe', 'hrnet', 'body_pose'],
        'step_03_cloth_segmentation': ['cloth', 'segmentation', 'u2net', 'sam', 'rembg', 'background'],
        'step_04_geometric_matching': ['geometric', 'matching', 'gmm', 'tps'],
        'step_05_cloth_warping': ['warping', 'tom', 'cloth_warping'],
        'step_06_virtual_fitting': ['fitting', 'ootd', 'diffusion', 'unet', 'vae', 'hrviton', 'viton'],
        'step_07_post_processing': ['post', 'enhancement', 'esrgan', 'gfpgan', 'super_resolution'],
        'step_08_quality_assessment': ['quality', 'assessment', 'clip', 'lpips']
    }
    
    def __init__(self, base_path: str = "backend/ai_models"):
        self.base_path = Path(base_path)
        self.files_info: List[ModelFileInfo] = []
        self.directory_stats = DirectoryStats(
            total_files=0,
            total_size_gb=0.0,
            by_extension=Counter(),
            by_step=Counter(),
            duplicate_groups=[],
            broken_links=[],
            large_files=[]
        )
        
    def analyze(self, detailed: bool = False) -> Dict[str, Any]:
        """전체 분석 실행"""
        logger.info(f"🔍 AI 모델 디렉토리 분석 시작: {self.base_path}")
        
        if not self.base_path.exists():
            logger.error(f"❌ 디렉토리가 존재하지 않습니다: {self.base_path}")
            return {}
            
        start_time = time.time()
        
        # 1. 파일 스캔
        self._scan_files(detailed)
        
        # 2. 중복 파일 탐지
        self._detect_duplicates()
        
        # 3. 통계 계산
        self._calculate_stats()
        
        # 4. 결과 정리
        analysis_time = time.time() - start_time
        
        result = {
            'analysis_info': {
                'base_path': str(self.base_path.absolute()),
                'analysis_time_seconds': round(analysis_time, 2),
                'total_files_scanned': len(self.files_info),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'directory_stats': asdict(self.directory_stats),
            'files_by_category': self._categorize_files(),
            'health_report': self._generate_health_report(),
            'recommendations': self._generate_recommendations()
        }
        
        if detailed:
            result['detailed_files'] = [asdict(f) for f in self.files_info]
            
        logger.info(f"✅ 분석 완료 ({analysis_time:.2f}초)")
        return result
    
    def _scan_files(self, detailed: bool = False):
        """파일 스캔"""
        logger.info("📂 파일 스캔 중...")
        
        for root, dirs, files in os.walk(self.base_path):
            root_path = Path(root)
            
            for file_name in files:
                file_path = root_path / file_name
                
                try:
                    file_info = self._analyze_file(file_path, detailed)
                    if file_info:
                        self.files_info.append(file_info)
                        
                except Exception as e:
                    logger.debug(f"파일 분석 실패 {file_path}: {e}")
                    
        logger.info(f"📊 총 {len(self.files_info)}개 파일 발견")
    
    def _analyze_file(self, file_path: Path, detailed: bool = False) -> Optional[ModelFileInfo]:
        """개별 파일 분석"""
        try:
            # 기본 정보
            file_stat = file_path.stat()
            size_mb = file_stat.st_size / (1024 * 1024)
            extension = file_path.suffix.lower()
            
            # 모델 파일 또는 설정 파일인지 확인
            is_model = extension in self.MODEL_EXTENSIONS
            is_config = extension in self.CONFIG_EXTENSIONS
            
            if not (is_model or is_config):
                return None
                
            # 링크 확인
            is_link = file_path.is_symlink()
            link_target = None
            accessibility = 'readable'
            
            if is_link:
                try:
                    link_target = str(file_path.readlink())
                    if not file_path.exists():
                        accessibility = 'broken_link'
                except:
                    accessibility = 'broken_link'
            
            # 접근 권한 확인
            if accessibility == 'readable' and not os.access(file_path, os.R_OK):
                accessibility = 'permission_denied'
            
            # 단계 카테고리 추정
            step_category = self._guess_step_category(file_path)
            
            # 체크포인트 여부
            is_checkpoint = 'checkpoint' in file_path.name.lower() or extension in {'.pth', '.pt', '.safetensors'}
            
            # 해시 미리보기 (detailed 모드에서만)
            hash_preview = None
            if detailed and accessibility == 'readable' and size_mb < 100:  # 100MB 미만만
                hash_preview = self._get_file_hash_preview(file_path)
            
            # PyTorch 로드 가능 여부 (간단 체크)
            pytorch_loadable = None
            estimated_parameters = None
            if detailed and extension in {'.pth', '.pt'} and accessibility == 'readable' and size_mb < 500:
                pytorch_loadable, estimated_parameters = self._test_pytorch_loading(file_path)
            
            return ModelFileInfo(
                name=file_path.name,
                path=str(file_path.relative_to(self.base_path)),
                size_mb=round(size_mb, 2),
                extension=extension,
                step_category=step_category,
                is_checkpoint=is_checkpoint,
                is_config=is_config,
                is_link=is_link,
                link_target=link_target,
                hash_preview=hash_preview,
                potential_duplicates=[],
                accessibility=accessibility,
                pytorch_loadable=pytorch_loadable,
                estimated_parameters=estimated_parameters
            )
            
        except Exception as e:
            logger.debug(f"파일 분석 오류 {file_path}: {e}")
            return None
    
    def _guess_step_category(self, file_path: Path) -> Optional[str]:
        """파일 경로/이름으로 단계 추정"""
        path_str = str(file_path).lower()
        
        for step_name, keywords in self.STEP_KEYWORDS.items():
            if any(keyword in path_str for keyword in keywords):
                return step_name
                
        return None
    
    def _get_file_hash_preview(self, file_path: Path, chunk_size: int = 8192) -> str:
        """파일 시작 부분의 해시 (중복 탐지용)"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                chunk = f.read(chunk_size)
                hasher.update(chunk)
            return hasher.hexdigest()[:16]
        except:
            return None
    
    def _test_pytorch_loading(self, file_path: Path) -> Tuple[bool, Optional[int]]:
        """PyTorch 모델 로딩 테스트"""
        try:
            import torch
            
            # 안전한 로딩 시도
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
            
            # 파라미터 수 추정
            param_count = 0
            if isinstance(checkpoint, dict):
                for key, value in checkpoint.items():
                    if torch.is_tensor(value):
                        param_count += value.numel()
            
            return True, param_count if param_count > 0 else None
            
        except Exception:
            try:
                # weights_only=False로 재시도
                torch.load(file_path, map_location='cpu')
                return True, None
            except:
                return False, None
    
    def _detect_duplicates(self):
        """중복 파일 탐지"""
        logger.info("🔍 중복 파일 탐지 중...")
        
        # 크기와 해시로 그룹화
        size_groups = defaultdict(list)
        hash_groups = defaultdict(list)
        
        for file_info in self.files_info:
            size_groups[file_info.size_mb].append(file_info)
            if file_info.hash_preview:
                hash_groups[file_info.hash_preview].append(file_info)
        
        # 중복 그룹 찾기
        duplicate_groups = []
        
        # 동일한 크기인 파일들 확인
        for size, files in size_groups.items():
            if len(files) > 1 and size > 1:  # 1MB 이상인 파일만
                group_names = [f.name for f in files]
                duplicate_groups.append(group_names)
                
                # 각 파일에 중복 정보 추가
                for file_info in files:
                    file_info.potential_duplicates = [name for name in group_names if name != file_info.name]
        
        self.directory_stats.duplicate_groups = duplicate_groups
        logger.info(f"📊 {len(duplicate_groups)}개 중복 그룹 발견")
    
    def _calculate_stats(self):
        """통계 계산"""
        logger.info("📊 통계 계산 중...")
        
        total_size_gb = 0.0
        extension_count = Counter()
        step_count = Counter()
        broken_links = []
        large_files = []
        
        for file_info in self.files_info:
            total_size_gb += file_info.size_mb / 1024
            extension_count[file_info.extension] += 1
            
            if file_info.step_category:
                step_count[file_info.step_category] += 1
            else:
                step_count['unknown'] += 1
                
            if file_info.accessibility == 'broken_link':
                broken_links.append(file_info.name)
                
            if file_info.size_mb > 1024:  # 1GB 이상
                large_files.append((file_info.name, file_info.size_mb / 1024))
        
        # 큰 파일 순으로 정렬
        large_files.sort(key=lambda x: x[1], reverse=True)
        
        self.directory_stats.total_files = len(self.files_info)
        self.directory_stats.total_size_gb = round(total_size_gb, 2)
        self.directory_stats.by_extension = dict(extension_count)
        self.directory_stats.by_step = dict(step_count)
        self.directory_stats.broken_links = broken_links
        self.directory_stats.large_files = large_files[:20]  # 상위 20개
    
    def _categorize_files(self) -> Dict[str, Any]:
        """파일 카테고리별 분류"""
        categories = {
            'model_files': [],
            'config_files': [],
            'checkpoint_files': [],
            'broken_files': [],
            'large_files': [],
            'unknown_category': []
        }
        
        for file_info in self.files_info:
            if file_info.accessibility == 'broken_link':
                categories['broken_files'].append(file_info.name)
            elif file_info.is_checkpoint:
                categories['checkpoint_files'].append(file_info.name)
            elif file_info.is_config:
                categories['config_files'].append(file_info.name)
            elif file_info.extension in self.MODEL_EXTENSIONS:
                categories['model_files'].append(file_info.name)
            else:
                categories['unknown_category'].append(file_info.name)
                
            if file_info.size_mb > 1024:
                categories['large_files'].append(f"{file_info.name} ({file_info.size_mb/1024:.1f}GB)")
        
        return categories
    
    def _generate_health_report(self) -> Dict[str, Any]:
        """디렉토리 건강 상태 보고서"""
        total_files = len(self.files_info)
        broken_count = len(self.directory_stats.broken_links)
        duplicate_count = sum(len(group) for group in self.directory_stats.duplicate_groups)
        
        # 건강 점수 계산 (0-100)
        health_score = 100
        
        if broken_count > 0:
            health_score -= min(30, broken_count * 5)
            
        if duplicate_count > total_files * 0.3:  # 30% 이상이 중복
            health_score -= 20
            
        if self.directory_stats.total_size_gb > 200:  # 200GB 이상
            health_score -= 10
        
        health_score = max(0, health_score)
        
        return {
            'health_score': health_score,
            'status': 'healthy' if health_score >= 80 else 'needs_attention' if health_score >= 60 else 'critical',
            'issues': {
                'broken_links': broken_count,
                'potential_duplicates': duplicate_count,
                'large_files_count': len(self.directory_stats.large_files),
                'permission_issues': len([f for f in self.files_info if f.accessibility == 'permission_denied'])
            },
            'recommendations_count': len(self._generate_recommendations())
        }
    
    def _generate_recommendations(self) -> List[str]:
        """개선 권장사항"""
        recommendations = []
        
        # 깨진 링크
        if self.directory_stats.broken_links:
            recommendations.append(f"🔗 {len(self.directory_stats.broken_links)}개의 깨진 심볼릭 링크 수정 필요")
        
        # 중복 파일
        if self.directory_stats.duplicate_groups:
            total_duplicates = sum(len(group) for group in self.directory_stats.duplicate_groups)
            recommendations.append(f"📂 {total_duplicates}개의 중복 파일 정리 필요")
        
        # 대용량 파일
        if len(self.directory_stats.large_files) > 10:
            recommendations.append("💾 대용량 파일들의 클라우드 저장소 이동 고려")
        
        # 미분류 파일
        unknown_count = self.directory_stats.by_step.get('unknown', 0)
        if unknown_count > len(self.files_info) * 0.2:
            recommendations.append(f"📋 {unknown_count}개 미분류 파일의 단계별 정리 필요")
        
        # 전체 크기
        if self.directory_stats.total_size_gb > 100:
            recommendations.append(f"💿 전체 용량 {self.directory_stats.total_size_gb:.1f}GB - 용량 최적화 검토 필요")
        
        return recommendations

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Models Directory Analyzer")
    parser.add_argument("--path", default="backend/ai_models", help="분석할 디렉토리 경로")
    parser.add_argument("--detailed", action="store_true", help="상세 분석 (더 오래 걸림)")
    parser.add_argument("--export-json", help="결과를 JSON 파일로 저장")
    parser.add_argument("--scan-only", action="store_true", help="스캔만 수행 (PyTorch 검증 제외)")
    
    args = parser.parse_args()
    
    # 분석 실행
    analyzer = AIModelsAnalyzer(args.path)
    result = analyzer.analyze(detailed=args.detailed and not args.scan_only)
    
    # 결과 출력
    print("\n" + "="*60)
    print("🔍 AI MODELS DIRECTORY ANALYSIS REPORT")
    print("="*60)
    
    # 기본 정보
    info = result['analysis_info']
    print(f"📂 Base Path: {info['base_path']}")
    print(f"⏱️  Analysis Time: {info['analysis_time_seconds']}s")
    print(f"📊 Files Scanned: {info['total_files_scanned']}")
    
    # 디렉토리 통계
    stats = result['directory_stats']
    print(f"\n📈 DIRECTORY STATISTICS")
    print(f"   Total Files: {stats['total_files']}")
    print(f"   Total Size: {stats['total_size_gb']:.2f} GB")
    print(f"   Duplicate Groups: {len(stats['duplicate_groups'])}")
    print(f"   Broken Links: {len(stats['broken_links'])}")
    
    # 파일 형식별 분포
    print(f"\n📄 BY FILE EXTENSION:")
    for ext, count in sorted(stats['by_extension'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {ext}: {count} files")
    
    # 단계별 분포
    print(f"\n🎯 BY AI STEP:")
    for step, count in sorted(stats['by_step'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {step}: {count} files")
    
    # 건강 상태
    health = result['health_report']
    print(f"\n💊 HEALTH REPORT")
    print(f"   Score: {health['health_score']}/100 ({health['status']})")
    print(f"   Issues: {health['issues']}")
    
    # 권장사항
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in result['recommendations']:
        print(f"   • {rec}")
    
    # 대용량 파일 목록
    if stats['large_files']:
        print(f"\n📦 LARGE FILES (>1GB):")
        for name, size_gb in stats['large_files'][:10]:
            print(f"   {name}: {size_gb:.2f} GB")
    
    # JSON 내보내기
    if args.export_json:
        with open(args.export_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 결과를 {args.export_json}에 저장했습니다.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()