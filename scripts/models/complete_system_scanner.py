#!/usr/bin/env python3
"""
MyCloset AI 완전한 시스템 모델 스캐너
전체 시스템의 모든 AI 모델, 체크포인트, 설정 파일을 완전히 스캔
"""

import os
import sys
import json
import hashlib
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import mimetypes
import time

class CompleteSystemModelScanner:
    def __init__(self, deep_scan: bool = False, verbose: bool = True):
        self.project_root = Path.cwd()
        self.deep_scan = deep_scan
        self.verbose = verbose
        self.start_time = time.time()
        
        # 스캔 결과 저장
        self.scan_results = {
            "scan_info": {
                "timestamp": datetime.now().isoformat(),
                "system": platform.system(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
                "project_root": str(self.project_root),
                "deep_scan": deep_scan,
                "scan_duration": 0
            },
            "locations": {},
            "models_by_type": {},
            "duplicates": {},
            "statistics": {},
            "issues": [],
            "recommendations": []
        }
        
        # AI 모델 파일 확장자들 (확장)
        self.model_extensions = {
            # PyTorch
            '.pth': 'pytorch',
            '.pt': 'pytorch', 
            '.bin': 'pytorch_bin',
            '.pkl': 'pickle',
            
            # Safetensors (HuggingFace)
            '.safetensors': 'safetensors',
            
            # ONNX
            '.onnx': 'onnx',
            
            # TensorFlow
            '.pb': 'tensorflow',
            '.h5': 'tensorflow',
            '.tflite': 'tensorflow_lite',
            '.keras': 'keras',
            
            # 체크포인트
            '.ckpt': 'checkpoint',
            '.checkpoint': 'checkpoint',
            '.weights': 'weights',
            '.model': 'generic_model',
            
            # 기타
            '.npz': 'numpy',
            '.npy': 'numpy',
            '.joblib': 'joblib'
        }
        
        # 설정 파일들
        self.config_extensions = {
            '.json': 'json_config',
            '.yaml': 'yaml_config',
            '.yml': 'yaml_config',
            '.toml': 'toml_config',
            '.ini': 'ini_config',
            '.cfg': 'cfg_config',
            '.config': 'config',
            '.txt': 'text_config'
        }
        
        # 제외할 디렉토리 (성능 및 보안)
        self.exclude_dirs = {
            '__pycache__', '.git', '.svn', '.hg',
            'node_modules', '.npm', 'bower_components',
            '.vscode', '.idea', '.vs',
            'venv', 'env', '.env', 'virtualenv',
            '.pytest_cache', '.tox', '.coverage',
            'logs', 'log', 'temp', 'tmp',
            '.cache', 'cache', '.npm',
            'Trash', '.Trash', '.recycle.bin',
            'System Volume Information',
            '$RECYCLE.BIN', 'pagefile.sys', 'hiberfil.sys'
        }
        
        # 시스템 보호 경로 (스캔하지 않음)
        self.protected_paths = {
            '/System', '/Library/System', '/private',
            '/dev', '/proc', '/sys', '/var/log',
            '/etc/shadow', '/etc/passwd'
        }

    def get_all_scan_paths(self) -> List[Path]:
        """스캔할 모든 경로 수집"""
        paths = []
        system = platform.system().lower()
        
        # 1. 현재 프로젝트 (최우선)
        paths.append(self.project_root)
        
        # 2. 홈 디렉토리
        home = Path.home()
        paths.append(home)
        
        # 3. 일반적인 AI 모델 저장 위치들
        common_paths = [
            home / ".cache",
            home / ".cache" / "huggingface",
            home / ".cache" / "torch",
            home / ".cache" / "diffusers",
            home / ".local" / "share",
            home / "Downloads",
            home / "Documents",
            home / "Desktop"
        ]
        
        # 4. 시스템별 추가 경로
        if system == "darwin":  # macOS
            common_paths.extend([
                Path("/Applications"),
                Path("/opt"),
                Path("/usr/local"),
                Path("/opt/homebrew"),
                home / "Library" / "Caches",
                home / "Library" / "Application Support"
            ])
        elif system == "linux":
            common_paths.extend([
                Path("/opt"),
                Path("/usr/local"),
                Path("/usr/share"),
                Path("/var/lib"),
                Path("/home")
            ])
        elif system == "windows":
            common_paths.extend([
                Path("C:/Program Files"),
                Path("C:/Program Files (x86)"),
                Path("C:/Users"),
                home / "AppData"
            ])
        
        # 5. Deep scan인 경우 루트부터 스캔
        if self.deep_scan:
            if system == "darwin":
                paths.append(Path("/"))
            elif system == "linux":
                paths.append(Path("/"))
            elif system == "windows":
                for drive in "CDEFGH":
                    drive_path = Path(f"{drive}:/")
                    if drive_path.exists():
                        paths.append(drive_path)
        
        # 존재하는 경로만 필터링
        valid_paths = []
        for path in paths + common_paths:
            if path.exists() and path.is_dir():
                valid_paths.append(path)
        
        # 중복 제거 (부모-자식 관계 확인)
        unique_paths = []
        for path in valid_paths:
            is_child = False
            for existing in unique_paths:
                try:
                    if path.is_relative_to(existing):
                        is_child = True
                        break
                except (ValueError, OSError):
                    continue
            if not is_child:
                unique_paths.append(path)
        
        return unique_paths

    def is_protected_path(self, path: Path) -> bool:
        """보호된 경로인지 확인"""
        path_str = str(path)
        for protected in self.protected_paths:
            if path_str.startswith(protected):
                return True
        return False

    def should_skip_directory(self, directory: Path) -> bool:
        """디렉토리를 건너뛸지 결정"""
        dir_name = directory.name.lower()
        
        # 제외 디렉토리 확인
        if dir_name in self.exclude_dirs:
            return True
            
        # 보호된 경로 확인
        if self.is_protected_path(directory):
            return True
            
        # 숨겨진 디렉토리 (선택적 제외)
        if dir_name.startswith('.') and dir_name not in {'.cache', '.local'}:
            return True
            
        # 너무 긴 경로 (Windows 호환성)
        if len(str(directory)) > 250:
            return True
            
        return False

    def get_file_info(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """파일 정보 수집"""
        try:
            if not file_path.exists() or not file_path.is_file():
                return None
                
            stat_info = file_path.stat()
            size_bytes = stat_info.st_size
            size_mb = round(size_bytes / (1024 * 1024), 2)
            
            # 너무 작은 파일은 제외 (1KB 미만)
            if size_bytes < 1024:
                return None
            
            # 파일 해시 (중복 탐지용)
            file_hash = self.get_file_hash(file_path)
            
            # 파일 타입 추정
            file_type = self.classify_file_type(file_path)
            
            return {
                "path": str(file_path),
                "name": file_path.name,
                "size_bytes": size_bytes,
                "size_mb": size_mb,
                "size_gb": round(size_mb / 1024, 3),
                "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                "hash": file_hash,
                "extension": file_path.suffix.lower(),
                "framework": self.model_extensions.get(file_path.suffix.lower(), 'unknown'),
                "file_type": file_type,
                "relative_to_home": str(file_path.relative_to(Path.home())) if file_path.is_relative_to(Path.home()) else None,
                "relative_to_project": str(file_path.relative_to(self.project_root)) if file_path.is_relative_to(self.project_root) else None
            }
            
        except (PermissionError, OSError, ValueError) as e:
            if self.verbose:
                print(f"⚠️ 파일 정보 수집 실패 {file_path}: {e}")
            return None

    def get_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산 (중복 탐지용)"""
        try:
            with open(file_path, 'rb') as f:
                # 파일 크기에 따라 읽을 바이트 수 조정
                if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB 이상
                    content = f.read(16384)  # 16KB만 읽기
                else:
                    content = f.read(65536)  # 64KB 읽기
                return hashlib.md5(content).hexdigest()
        except:
            return "unknown"

    def classify_file_type(self, file_path: Path) -> str:
        """파일 타입 분류"""
        name_lower = file_path.name.lower()
        path_lower = str(file_path).lower()
        
        # AI 모델 타입 추정
        if any(keyword in path_lower for keyword in ['clip', 'vit']):
            return 'clip_model'
        elif any(keyword in path_lower for keyword in ['diffusion', 'stable', 'ootd']):
            return 'diffusion_model'
        elif any(keyword in path_lower for keyword in ['pose', 'openpose', 'body']):
            return 'pose_model'
        elif any(keyword in path_lower for keyword in ['segment', 'u2net', 'mask']):
            return 'segmentation_model'
        elif any(keyword in path_lower for keyword in ['parsing', 'human', 'atr']):
            return 'parsing_model'
        elif any(keyword in path_lower for keyword in ['vton', 'viton', 'tryon']):
            return 'virtual_tryon_model'
        elif 'checkpoint' in path_lower or 'ckpt' in name_lower:
            return 'checkpoint'
        elif any(keyword in path_lower for keyword in ['config', 'setup', 'settings']):
            return 'config_file'
        else:
            return 'unknown_model'

    def scan_directory_safe(self, directory: Path) -> List[Dict[str, Any]]:
        """안전한 디렉토리 스캔"""
        found_files = []
        
        try:
            if not directory.exists() or not directory.is_dir():
                return found_files
                
            if self.should_skip_directory(directory):
                return found_files
            
            # 권한 확인
            if not os.access(directory, os.R_OK):
                return found_files
            
            # 재귀적으로 파일 찾기
            for item in directory.iterdir():
                try:
                    if item.is_file():
                        # 모델 파일인지 확인
                        if item.suffix.lower() in self.model_extensions:
                            file_info = self.get_file_info(item)
                            if file_info:
                                found_files.append(file_info)
                        # 설정 파일인지 확인 (AI 관련 키워드 포함 시)
                        elif item.suffix.lower() in self.config_extensions:
                            if any(keyword in str(item).lower() for keyword in 
                                  ['model', 'ai', 'ml', 'torch', 'tensorflow', 'onnx', 'diffusion']):
                                file_info = self.get_file_info(item)
                                if file_info:
                                    found_files.append(file_info)
                    
                    elif item.is_dir() and not self.should_skip_directory(item):
                        # 재귀적으로 하위 디렉토리 스캔
                        sub_files = self.scan_directory_safe(item)
                        found_files.extend(sub_files)
                        
                except (PermissionError, OSError):
                    continue
                    
        except (PermissionError, OSError) as e:
            if self.verbose:
                print(f"⚠️ 디렉토리 스캔 실패 {directory}: {e}")
        
        return found_files

    def run_parallel_scan(self, scan_paths: List[Path]) -> Dict[str, List[Dict]]:
        """병렬 스캔 실행"""
        results = {}
        
        print(f"🔍 {len(scan_paths)}개 위치 병렬 스캔 시작...")
        
        # ThreadPoolExecutor를 사용한 병렬 처리
        with ThreadPoolExecutor(max_workers=min(8, len(scan_paths))) as executor:
            # 각 경로에 대해 스캔 작업 제출
            future_to_path = {
                executor.submit(self.scan_directory_safe, path): path 
                for path in scan_paths
            }
            
            # 완료된 작업들 수집
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    files = future.result()
                    if files:
                        results[str(path)] = files
                        if self.verbose:
                            total_size = sum(f['size_mb'] for f in files)
                            print(f"✅ {path}: {len(files)}개 파일 ({total_size:.1f}MB)")
                    else:
                        if self.verbose:
                            print(f"📂 {path}: 모델 파일 없음")
                            
                except Exception as e:
                    self.scan_results["issues"].append(f"스캔 실패 {path}: {e}")
                    if self.verbose:
                        print(f"❌ {path}: 스캔 실패 - {e}")
        
        return results

    def analyze_scan_results(self, scan_results: Dict[str, List[Dict]]):
        """스캔 결과 분석"""
        print("\n📊 스캔 결과 분석 중...")
        
        all_files = []
        for files in scan_results.values():
            all_files.extend(files)
        
        # 1. 위치별 정리
        self.scan_results["locations"] = scan_results
        
        # 2. 파일 타입별 분류
        type_groups = {}
        for file_info in all_files:
            file_type = file_info["file_type"]
            if file_type not in type_groups:
                type_groups[file_type] = []
            type_groups[file_type].append(file_info)
        
        self.scan_results["models_by_type"] = type_groups
        
        # 3. 중복 파일 탐지
        hash_groups = {}
        for file_info in all_files:
            file_hash = file_info["hash"]
            if file_hash != "unknown":
                if file_hash not in hash_groups:
                    hash_groups[file_hash] = []
                hash_groups[file_hash].append(file_info)
        
        # 중복 파일들만 추출
        duplicates = {h: files for h, files in hash_groups.items() if len(files) > 1}
        self.scan_results["duplicates"] = duplicates
        
        # 4. 통계 계산
        total_files = len(all_files)
        total_size_mb = sum(f["size_mb"] for f in all_files)
        total_size_gb = total_size_mb / 1024
        
        framework_stats = {}
        for file_info in all_files:
            framework = file_info["framework"]
            if framework not in framework_stats:
                framework_stats[framework] = {"count": 0, "size_mb": 0}
            framework_stats[framework]["count"] += 1
            framework_stats[framework]["size_mb"] += file_info["size_mb"]
        
        self.scan_results["statistics"] = {
            "total_files": total_files,
            "total_size_mb": total_size_mb,
            "total_size_gb": total_size_gb,
            "framework_distribution": framework_stats,
            "duplicate_groups": len(duplicates),
            "duplicate_waste_mb": sum(
                sum(f["size_mb"] for f in files[1:])  # 첫 번째 제외하고 나머지가 낭비
                for files in duplicates.values()
            ) if duplicates else 0
        }

    def generate_recommendations(self):
        """정리 권장사항 생성"""
        print("💡 권장사항 생성 중...")
        
        recommendations = []
        stats = self.scan_results["statistics"]
        
        # 1. 중복 파일 정리
        if stats["duplicate_waste_mb"] > 100:  # 100MB 이상 낭비
            recommendations.append({
                "type": "duplicate_cleanup",
                "priority": "high",
                "title": "중복 파일 정리",
                "description": f"{stats['duplicate_groups']}개 그룹의 중복 파일로 {stats['duplicate_waste_mb']:.1f}MB 낭비",
                "potential_savings": f"{stats['duplicate_waste_mb']:.1f}MB",
                "action": "run_duplicate_cleanup"
            })
        
        # 2. 프로젝트 외부 모델들 통합
        external_models = []
        for location, files in self.scan_results["locations"].items():
            if not Path(location).is_relative_to(self.project_root):
                external_models.extend(files)
        
        if external_models:
            external_size = sum(f["size_mb"] for f in external_models)
            recommendations.append({
                "type": "model_consolidation", 
                "priority": "medium",
                "title": "외부 모델 통합",
                "description": f"프로젝트 외부에 {len(external_models)}개 모델 ({external_size:.1f}MB)",
                "action": "move_external_models"
            })
        
        # 3. 대용량 파일 최적화
        large_files = [f for f in self.get_all_files() if f["size_mb"] > 1000]  # 1GB 이상
        if large_files:
            recommendations.append({
                "type": "large_file_optimization",
                "priority": "low", 
                "title": "대용량 파일 최적화",
                "description": f"{len(large_files)}개의 1GB+ 파일 최적화 가능",
                "action": "optimize_large_models"
            })
        
        self.scan_results["recommendations"] = recommendations

    def get_all_files(self) -> List[Dict]:
        """모든 스캔된 파일 목록 반환"""
        all_files = []
        for files in self.scan_results["locations"].values():
            all_files.extend(files)
        return all_files

    def print_detailed_summary(self):
        """상세한 스캔 결과 출력"""
        print("\n" + "="*80)
        print("🎯 MyCloset AI 완전한 시스템 모델 스캔 결과")
        print("="*80)
        
        scan_info = self.scan_results["scan_info"]
        stats = self.scan_results["statistics"]
        
        # 기본 정보
        print(f"🕐 스캔 시간: {scan_info['timestamp']}")
        print(f"⏱️  소요 시간: {scan_info['scan_duration']:.1f}초")
        print(f"💻 시스템: {scan_info['system']} {scan_info['machine']}")
        print(f"📂 프로젝트: {scan_info['project_root']}")
        print(f"🔍 스캔 모드: {'전체 시스템' if scan_info['deep_scan'] else '주요 위치'}")
        
        # 전체 통계
        print(f"\n📊 전체 통계:")
        print(f"  📄 총 파일 수: {stats['total_files']:,}개")
        print(f"  💾 총 크기: {stats['total_size_gb']:.2f}GB ({stats['total_size_mb']:.1f}MB)")
        print(f"  📁 스캔 위치: {len(self.scan_results['locations'])}곳")
        
        # 프레임워크별 분포
        print(f"\n🔧 프레임워크별 분포:")
        for framework, info in sorted(stats['framework_distribution'].items(), 
                                    key=lambda x: x[1]['size_mb'], reverse=True):
            print(f"  - {framework}: {info['count']}개 ({info['size_mb']:.1f}MB)")
        
        # 파일 타입별 분포
        print(f"\n🎯 파일 타입별 분포:")
        for file_type, files in sorted(self.scan_results['models_by_type'].items(),
                                     key=lambda x: len(x[1]), reverse=True):
            total_size = sum(f['size_mb'] for f in files)
            print(f"  - {file_type}: {len(files)}개 ({total_size:.1f}MB)")
        
        # 위치별 분포
        print(f"\n📍 위치별 분포:")
        for location, files in sorted(self.scan_results['locations'].items(),
                                    key=lambda x: len(x[1]), reverse=True):
            total_size = sum(f['size_mb'] for f in files)
            location_display = location if len(location) < 60 else "..." + location[-57:]
            print(f"  - {location_display}: {len(files)}개 ({total_size:.1f}MB)")
        
        # 중복 파일
        if self.scan_results["duplicates"]:
            print(f"\n🔄 중복 파일:")
            print(f"  그룹 수: {stats['duplicate_groups']}개")
            print(f"  절약 가능: {stats['duplicate_waste_mb']:.1f}MB")
        
        # 주요 파일들 (크기 순)
        print(f"\n🏆 주요 파일들 (크기 순 Top 10):")
        all_files = self.get_all_files()
        top_files = sorted(all_files, key=lambda x: x['size_mb'], reverse=True)[:10]
        
        for i, file_info in enumerate(top_files, 1):
            path_display = file_info['path']
            if len(path_display) > 70:
                path_display = "..." + path_display[-67:]
            print(f"  {i:2d}. {file_info['name']}")
            print(f"      📁 {path_display}")
            print(f"      📊 {file_info['size_mb']:.1f}MB | {file_info['framework']} | {file_info['file_type']}")
        
        # 권장사항
        if self.scan_results["recommendations"]:
            print(f"\n💡 권장사항:")
            for rec in self.scan_results["recommendations"]:
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                emoji = priority_emoji.get(rec["priority"], "⚪")
                print(f"  {emoji} {rec['title']}: {rec['description']}")
        
        # 문제사항
        if self.scan_results["issues"]:
            print(f"\n⚠️ 발견된 문제들:")
            for issue in self.scan_results["issues"][:5]:  # 최대 5개
                print(f"  - {issue}")

    def save_results(self, output_file: str = None) -> Path:
        """결과를 JSON 파일로 저장"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"complete_model_scan_{timestamp}.json"
        
        output_path = self.project_root / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.scan_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 상세 결과 저장: {output_path}")
        return output_path

    def run_complete_scan(self) -> Path:
        """완전한 시스템 스캔 실행"""
        print("🚀 MyCloset AI 완전한 시스템 모델 스캔 시작")
        print("="*80)
        
        try:
            # 1. 스캔 경로 수집
            scan_paths = self.get_all_scan_paths()
            print(f"📂 총 {len(scan_paths)}개 위치 스캔 예정")
            
            if self.verbose:
                for i, path in enumerate(scan_paths):
                    print(f"  {i+1:2d}. {path}")
                print()
            
            # 2. 병렬 스캔 실행
            scan_results = self.run_parallel_scan(scan_paths)
            
            # 3. 결과 분석
            self.analyze_scan_results(scan_results)
            
            # 4. 권장사항 생성
            self.generate_recommendations()
            
            # 5. 스캔 시간 기록
            self.scan_results["scan_info"]["scan_duration"] = time.time() - self.start_time
            
            # 6. 결과 출력
            self.print_detailed_summary()
            
            # 7. 결과 저장
            output_file = self.save_results()
            
            print(f"\n✅ 완전한 시스템 스캔 완료!")
            print(f"📊 상세 결과: {output_file}")
            print(f"\n🎯 다음 단계:")
            print("1. 스캔 결과를 확인하여 중복 파일 정리")
            print("2. 외부 모델들을 프로젝트로 이동")
            print("3. 표준 디렉토리 구조 적용")
            
            return output_file
            
        except Exception as e:
            print(f"❌ 스캔 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MyCloset AI 완전한 시스템 모델 스캐너")
    parser.add_argument("--deep", action="store_true", help="전체 시스템 딥 스캔 (시간 오래 걸림)")
    parser.add_argument("--quiet", action="store_true", help="최소 출력 모드")
    parser.add_argument("--output", "-o", help="출력 파일명")
    
    args = parser.parse_args()
    
    # 딥 스캔 경고
    if args.deep:
        print("⚠️  전체 시스템 딥 스캔은 오랜 시간이 걸릴 수 있습니다.")
        response = input("계속하시겠습니까? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("스캔이 취소되었습니다.")
            return
    
    scanner = CompleteSystemModelScanner(
        deep_scan=args.deep,
        verbose=not args.quiet
    )
    
    result_file = scanner.run_complete_scan()
    
    if result_file:
        print(f"\n🎉 스캔 완료! 결과: {result_file}")
    else:
        print("❌ 스캔 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()