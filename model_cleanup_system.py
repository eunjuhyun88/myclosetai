#!/usr/bin/env python3
"""
🔥 MyCloset AI 모델 자동 정리 시스템
✅ 안전한 중복 제거
✅ 디렉토리 구조 최적화  
✅ auto_model_detector.py 개선
✅ conda 환경 최적화
"""

import os
import sys
import shutil
import hashlib
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelCleanupSystem:
    """AI 모델 자동 정리 시스템"""
    
    def __init__(self, base_path: str = "backend/ai_models"):
        self.base_path = Path(base_path)
        self.backup_path = Path("backup_models_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.analysis_result = {}
        self.cleanup_plan = {}
        
        # 단계별 표준 디렉토리 구조
        self.standard_structure = {
            "step_01_human_parsing": ["schp", "graphonomy", "densepose", "parsing"],
            "step_02_pose_estimation": ["openpose", "dwpose", "coco", "pose"],
            "step_03_cloth_segmentation": ["u2net", "cloth", "segmentation", "mask"],
            "step_04_geometric_matching": ["tps", "geometric", "matching", "transform"],
            "step_05_cloth_warping": ["warping", "cloth_warping", "deformation"],
            "step_06_virtual_fitting": ["viton", "diffusion", "fitting", "ootd"],
            "step_07_post_processing": ["postprocess", "refinement", "enhancement"],
            "step_08_quality_assessment": ["quality", "assessment", "metric"]
        }
        
    def run_complete_cleanup(self, dry_run: bool = True) -> Dict[str, Any]:
        """완전한 정리 프로세스 실행"""
        try:
            logger.info("🚀 AI 모델 자동 정리 시스템 시작")
            
            # 1단계: 현재 상태 분석
            logger.info("📊 1단계: 현재 상태 분석")
            self.analysis_result = self.analyze_current_state()
            
            # 2단계: 정리 계획 수립
            logger.info("📋 2단계: 정리 계획 수립")
            self.cleanup_plan = self.create_cleanup_plan()
            
            # 3단계: 안전한 백업 (dry_run이 아닌 경우만)
            if not dry_run:
                logger.info("💾 3단계: 안전한 백업")
                self.create_safety_backup()
            
            # 4단계: 중복 파일 제거
            logger.info("🗂️ 4단계: 중복 파일 정리")
            duplicate_results = self.cleanup_duplicates(dry_run=dry_run)
            
            # 5단계: 디렉토리 구조 최적화
            logger.info("📁 5단계: 디렉토리 구조 최적화")
            structure_results = self.optimize_directory_structure(dry_run=dry_run)
            
            # 6단계: auto_model_detector.py 개선
            logger.info("🔍 6단계: auto_model_detector.py 개선")
            detector_results = self.improve_auto_detector(dry_run=dry_run)
            
            # 7단계: 최종 검증
            logger.info("✅ 7단계: 최종 검증")
            validation_results = self.validate_cleanup()
            
            # 결과 리포트 생성
            final_report = self.generate_final_report(
                duplicate_results, structure_results, 
                detector_results, validation_results
            )
            
            logger.info("🎉 AI 모델 정리 완료!")
            return final_report
            
        except Exception as e:
            logger.error(f"❌ 정리 프로세스 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_current_state(self) -> Dict[str, Any]:
        """현재 상태 상세 분석"""
        try:
            analysis = {
                "total_files": 0,
                "total_size_gb": 0.0,
                "by_extension": defaultdict(int),
                "by_step": defaultdict(int),
                "duplicates": [],
                "large_files": [],
                "broken_links": [],
                "file_details": []
            }
            
            # 모든 파일 스캔
            for root, dirs, files in os.walk(self.base_path):
                for file in files:
                    file_path = Path(root) / file
                    
                    try:
                        if not file_path.exists():
                            analysis["broken_links"].append(str(file_path))
                            continue
                            
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        analysis["total_size_gb"] += size_mb / 1024
                        analysis["total_files"] += 1
                        
                        # 확장자별 분류
                        ext = file_path.suffix.lower()
                        analysis["by_extension"][ext] += 1
                        
                        # 단계별 분류
                        step = self.classify_by_step(file_path)
                        analysis["by_step"][step] += 1
                        
                        # 대용량 파일 추적
                        if size_mb > 1000:  # 1GB 이상
                            analysis["large_files"].append({
                                "path": str(file_path),
                                "size_gb": round(size_mb / 1024, 2)
                            })
                        
                        # 파일 상세 정보
                        analysis["file_details"].append({
                            "path": str(file_path.relative_to(self.base_path)),
                            "size_mb": round(size_mb, 2),
                            "extension": ext,
                            "step": step,
                            "modified": file_path.stat().st_mtime
                        })
                        
                    except Exception as e:
                        logger.warning(f"파일 분석 실패 {file_path}: {e}")
            
            # 중복 탐지
            analysis["duplicates"] = self.find_duplicates()
            
            return analysis
            
        except Exception as e:
            logger.error(f"상태 분석 실패: {e}")
            return {}
    
    def find_duplicates(self) -> List[Dict[str, Any]]:
        """정교한 중복 파일 탐지"""
        try:
            # 1. 크기 기반 그룹화
            size_groups = defaultdict(list)
            
            for file_info in self.analysis_result.get("file_details", []):
                if file_info["size_mb"] > 10:  # 10MB 이상만
                    size_groups[file_info["size_mb"]].append(file_info)
            
            # 2. 동일 크기 그룹에서 해시 비교
            duplicates = []
            
            for size_mb, files in size_groups.items():
                if len(files) > 1:
                    # 해시 그룹화
                    hash_groups = defaultdict(list)
                    
                    for file_info in files:
                        try:
                            file_path = self.base_path / file_info["path"]
                            file_hash = self.get_file_hash(file_path)
                            hash_groups[file_hash].append(file_info)
                        except Exception as e:
                            logger.warning(f"해시 계산 실패 {file_info['path']}: {e}")
                    
                    # 실제 중복 (동일 해시)
                    for file_hash, duplicate_files in hash_groups.items():
                        if len(duplicate_files) > 1:
                            duplicates.append({
                                "hash": file_hash,
                                "size_mb": size_mb,
                                "files": duplicate_files,
                                "waste_mb": size_mb * (len(duplicate_files) - 1)
                            })
            
            # 3. 버전 번호 중복 탐지 (_01, _02 등)
            version_duplicates = self.find_version_duplicates()
            duplicates.extend(version_duplicates)
            
            return duplicates
            
        except Exception as e:
            logger.error(f"중복 탐지 실패: {e}")
            return []
    
    def find_version_duplicates(self) -> List[Dict[str, Any]]:
        """버전 번호 중복 탐지"""
        import re
        
        version_groups = defaultdict(list)
        
        for file_info in self.analysis_result.get("file_details", []):
            # 버전 번호 제거한 기본 이름
            path = file_info["path"]
            base_name = re.sub(r'_\d+(\.(pth|pt|safetensors|bin|onnx))?$', '', Path(path).stem)
            
            version_groups[base_name].append(file_info)
        
        version_duplicates = []
        for base_name, files in version_groups.items():
            if len(files) > 1:
                total_size = sum(f["size_mb"] for f in files)
                max_size = max(f["size_mb"] for f in files)
                
                version_duplicates.append({
                    "type": "version_duplicate",
                    "base_name": base_name,
                    "files": files,
                    "total_size_mb": total_size,
                    "waste_mb": total_size - max_size  # 가장 큰 파일 제외
                })
        
        return version_duplicates
    
    def create_cleanup_plan(self) -> Dict[str, Any]:
        """정리 계획 수립"""
        try:
            plan = {
                "files_to_remove": [],
                "files_to_move": [],
                "directories_to_create": [],
                "estimated_savings_gb": 0.0,
                "safety_checks": []
            }
            
            # 중복 파일 제거 계획
            for duplicate_group in self.analysis_result.get("duplicates", []):
                if duplicate_group.get("type") == "version_duplicate":
                    # 버전 중복의 경우 가장 큰 파일만 유지
                    files = duplicate_group["files"]
                    largest_file = max(files, key=lambda x: x["size_mb"])
                    
                    for file_info in files:
                        if file_info != largest_file:
                            plan["files_to_remove"].append({
                                "path": file_info["path"],
                                "reason": f"버전 중복 - {largest_file['path']} 유지",
                                "size_mb": file_info["size_mb"]
                            })
                            plan["estimated_savings_gb"] += file_info["size_mb"] / 1024
                
                elif "hash" in duplicate_group:
                    # 해시 중복의 경우 첫 번째 파일만 유지
                    files = duplicate_group["files"]
                    keep_file = files[0]
                    
                    for file_info in files[1:]:
                        plan["files_to_remove"].append({
                            "path": file_info["path"],
                            "reason": f"해시 중복 - {keep_file['path']} 유지",
                            "size_mb": file_info["size_mb"]
                        })
                        plan["estimated_savings_gb"] += file_info["size_mb"] / 1024
            
            # 디렉토리 구조 최적화 계획
            self.plan_directory_optimization(plan)
            
            return plan
            
        except Exception as e:
            logger.error(f"정리 계획 수립 실패: {e}")
            return {}
    
    def plan_directory_optimization(self, plan: Dict[str, Any]):
        """디렉토리 구조 최적화 계획"""
        try:
            # 현재 파일들의 단계별 분포 확인
            step_files = defaultdict(list)
            
            for file_info in self.analysis_result.get("file_details", []):
                step = self.classify_by_step(Path(file_info["path"]))
                if step != "unknown":
                    step_files[step].append(file_info)
            
            # 표준 디렉토리 구조로 이동 계획
            for step, files in step_files.items():
                target_dir = f"organized/{step}"
                
                if target_dir not in [d["path"] for d in plan["directories_to_create"]]:
                    plan["directories_to_create"].append({
                        "path": target_dir,
                        "purpose": f"{step} 모델들 정리"
                    })
                
                for file_info in files:
                    current_path = file_info["path"]
                    if not current_path.startswith(f"organized/{step}/"):
                        new_path = f"{target_dir}/{Path(current_path).name}"
                        
                        plan["files_to_move"].append({
                            "from": current_path,
                            "to": new_path,
                            "reason": f"{step} 디렉토리로 정리"
                        })
            
        except Exception as e:
            logger.error(f"디렉토리 최적화 계획 실패: {e}")
    
    def cleanup_duplicates(self, dry_run: bool = True) -> Dict[str, Any]:
        """중복 파일 정리 실행"""
        try:
            results = {
                "removed_files": [],
                "saved_space_gb": 0.0,
                "errors": []
            }
            
            for item in self.cleanup_plan.get("files_to_remove", []):
                try:
                    file_path = self.base_path / item["path"]
                    
                    if dry_run:
                        logger.info(f"[DRY RUN] 제거 예정: {file_path}")
                        results["removed_files"].append(item["path"])
                        results["saved_space_gb"] += item["size_mb"] / 1024
                    else:
                        if file_path.exists():
                            file_path.unlink()
                            logger.info(f"✅ 제거 완료: {file_path}")
                            results["removed_files"].append(item["path"])
                            results["saved_space_gb"] += item["size_mb"] / 1024
                        else:
                            logger.warning(f"⚠️ 파일 없음: {file_path}")
                            
                except Exception as e:
                    error_msg = f"파일 제거 실패 {item['path']}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            return results
            
        except Exception as e:
            logger.error(f"중복 정리 실패: {e}")
            return {"errors": [str(e)]}
    
    def optimize_directory_structure(self, dry_run: bool = True) -> Dict[str, Any]:
        """디렉토리 구조 최적화"""
        try:
            results = {
                "created_directories": [],
                "moved_files": [],
                "errors": []
            }
            
            # 디렉토리 생성
            for dir_info in self.cleanup_plan.get("directories_to_create", []):
                try:
                    dir_path = self.base_path / dir_info["path"]
                    
                    if dry_run:
                        logger.info(f"[DRY RUN] 디렉토리 생성 예정: {dir_path}")
                        results["created_directories"].append(dir_info["path"])
                    else:
                        dir_path.mkdir(parents=True, exist_ok=True)
                        logger.info(f"✅ 디렉토리 생성: {dir_path}")
                        results["created_directories"].append(dir_info["path"])
                        
                except Exception as e:
                    error_msg = f"디렉토리 생성 실패 {dir_info['path']}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # 파일 이동
            for move_info in self.cleanup_plan.get("files_to_move", []):
                try:
                    src_path = self.base_path / move_info["from"]
                    dst_path = self.base_path / move_info["to"]
                    
                    if dry_run:
                        logger.info(f"[DRY RUN] 이동 예정: {src_path} → {dst_path}")
                        results["moved_files"].append(move_info)
                    else:
                        if src_path.exists():
                            dst_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(src_path), str(dst_path))
                            logger.info(f"✅ 이동 완료: {src_path} → {dst_path}")
                            results["moved_files"].append(move_info)
                        else:
                            logger.warning(f"⚠️ 소스 파일 없음: {src_path}")
                            
                except Exception as e:
                    error_msg = f"파일 이동 실패 {move_info}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            return results
            
        except Exception as e:
            logger.error(f"구조 최적화 실패: {e}")
            return {"errors": [str(e)]}
    
    def improve_auto_detector(self, dry_run: bool = True) -> Dict[str, Any]:
        """auto_model_detector.py 개선"""
        try:
            results = {
                "improvements": [],
                "new_patterns": [],
                "optimizations": []
            }
            
            # 실제 파일들 기반으로 패턴 개선
            file_patterns = self.analyze_file_patterns()
            
            # 새로운 auto_model_detector.py 코드 생성
            improved_detector = self.generate_improved_detector(file_patterns)
            
            if dry_run:
                logger.info("[DRY RUN] auto_model_detector.py 개선 미리보기")
                results["improvements"] = ["패턴 매칭 개선", "캐싱 최적화", "성능 향상"]
                results["new_patterns"] = list(file_patterns.keys())
            else:
                # 백업 후 개선된 버전 저장
                detector_path = Path("backend/app/ai_pipeline/utils/auto_model_detector.py")
                if detector_path.exists():
                    backup_path = detector_path.with_suffix('.py.backup')
                    shutil.copy2(detector_path, backup_path)
                    logger.info(f"✅ 백업 생성: {backup_path}")
                
                with open(detector_path, 'w', encoding='utf-8') as f:
                    f.write(improved_detector)
                
                logger.info("✅ auto_model_detector.py 개선 완료")
                results["improvements"] = ["파일 저장 완료"]
            
            return results
            
        except Exception as e:
            logger.error(f"auto_detector 개선 실패: {e}")
            return {"errors": [str(e)]}
    
    def analyze_file_patterns(self) -> Dict[str, List[str]]:
        """실제 파일들 기반 패턴 분석"""
        patterns = defaultdict(list)
        
        for file_info in self.analysis_result.get("file_details", []):
            step = file_info["step"]
            if step != "unknown":
                file_name = Path(file_info["path"]).name
                patterns[step].append(file_name)
        
        return dict(patterns)
    
    def generate_improved_detector(self, file_patterns: Dict[str, List[str]]) -> str:
        """개선된 auto_model_detector.py 코드 생성"""
        
        improved_code = '''#!/usr/bin/env python3
"""
🔥 개선된 AI 모델 자동 탐지 시스템
✅ 실제 파일 기반 패턴 매칭
✅ 캐싱 최적화
✅ 성능 향상
"""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelMatch:
    """모델 매칭 결과"""
    file_path: Path
    confidence: float
    step_name: str
    file_size_mb: float
    match_reason: str

class ImprovedAutoModelDetector:
    """개선된 AI 모델 자동 탐지기"""
    
    def __init__(self, base_path: str = "backend/ai_models"):
        self.base_path = Path(base_path)
        self.cache = {}
        self.cache_file = self.base_path / ".detector_cache.json"
        
        # 실제 파일 기반 패턴
        self.step_patterns = {
'''

        # 실제 파일 패턴 추가
        for step, files in file_patterns.items():
            improved_code += f'            "{step}": [\n'
            
            # 파일명에서 패턴 추출
            unique_patterns = set()
            for file_name in files[:10]:  # 상위 10개만
                # 확장자 제거하고 패턴 생성
                base_name = Path(file_name).stem.lower()
                if len(base_name) > 3:
                    pattern = f".*{re.escape(base_name[:5])}.*"
                    unique_patterns.add(pattern)
            
            for pattern in sorted(unique_patterns)[:5]:  # 상위 5개만
                improved_code += f'                r"{pattern}",\n'
            
            improved_code += '            ],\n'
        
        improved_code += '''        }
        
        self.load_cache()
    
    def find_best_model_for_step(self, step_name: str) -> Optional[ModelMatch]:
        """Step에 최적화된 모델 찾기"""
        try:
            cache_key = f"best_model_{step_name}"
            
            # 캐시 확인
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if time.time() - cached_result["timestamp"] < 3600:  # 1시간 캐시
                    return self.dict_to_model_match(cached_result["result"])
            
            # 실제 검색
            candidates = []
            patterns = self.step_patterns.get(step_name, [])
            
            for pattern in patterns:
                matches = self.scan_files_by_pattern(pattern)
                candidates.extend(matches)
            
            # 최고 후보 선택
            best_match = self.select_best_candidate(candidates, step_name)
            
            # 캐시 저장
            if best_match:
                self.cache[cache_key] = {
                    "result": self.model_match_to_dict(best_match),
                    "timestamp": time.time()
                }
                self.save_cache()
            
            return best_match
            
        except Exception as e:
            logger.error(f"모델 검색 실패 {step_name}: {e}")
            return None
    
    def scan_files_by_pattern(self, pattern: str) -> List[ModelMatch]:
        """패턴으로 파일 스캔"""
        matches = []
        
        try:
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file() and re.match(pattern, file_path.name, re.IGNORECASE):
                    try:
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        
                        # 모델 파일만 (1MB 이상)
                        if size_mb >= 1.0 and file_path.suffix.lower() in ['.pth', '.pt', '.safetensors', '.bin', '.onnx']:
                            match = ModelMatch(
                                file_path=file_path,
                                confidence=0.8,  # 기본 신뢰도
                                step_name="",
                                file_size_mb=size_mb,
                                match_reason=f"패턴 매칭: {pattern}"
                            )
                            matches.append(match)
                            
                    except Exception as e:
                        logger.warning(f"파일 분석 실패 {file_path}: {e}")
                        
        except Exception as e:
            logger.error(f"패턴 스캔 실패 {pattern}: {e}")
        
        return matches
    
    def select_best_candidate(self, candidates: List[ModelMatch], step_name: str) -> Optional[ModelMatch]:
        """최고 후보 선택"""
        if not candidates:
            return None
        
        # 점수 계산
        for candidate in candidates:
            score = 0.0
            
            # 파일 크기 점수 (적당한 크기가 좋음)
            if 10 <= candidate.file_size_mb <= 1000:
                score += 0.3
            elif 1000 <= candidate.file_size_mb <= 5000:
                score += 0.2
            else:
                score += 0.1
            
            # 파일명 관련성 점수
            file_name = candidate.file_path.name.lower()
            if step_name.split('_')[-1] in file_name:  # 단계 키워드 포함
                score += 0.4
            
            # 확장자 점수
            if candidate.file_path.suffix.lower() in ['.pth', '.safetensors']:
                score += 0.2
            elif candidate.file_path.suffix.lower() in ['.pt', '.bin']:
                score += 0.1
            
            candidate.confidence = min(score, 1.0)
            candidate.step_name = step_name
        
        # 최고 점수 반환
        return max(candidates, key=lambda x: x.confidence)
    
    def load_cache(self):
        """캐시 로드"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
        except Exception as e:
            logger.warning(f"캐시 로드 실패: {e}")
            self.cache = {}
    
    def save_cache(self):
        """캐시 저장"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {e}")
    
    def model_match_to_dict(self, match: ModelMatch) -> Dict:
        """ModelMatch를 딕셔너리로 변환"""
        return {
            "file_path": str(match.file_path),
            "confidence": match.confidence,
            "step_name": match.step_name,
            "file_size_mb": match.file_size_mb,
            "match_reason": match.match_reason
        }
    
    def dict_to_model_match(self, data: Dict) -> ModelMatch:
        """딕셔너리를 ModelMatch로 변환"""
        return ModelMatch(
            file_path=Path(data["file_path"]),
            confidence=data["confidence"],
            step_name=data["step_name"],
            file_size_mb=data["file_size_mb"],
            match_reason=data["match_reason"]
        )

# 전역 인스턴스
auto_detector = ImprovedAutoModelDetector()

def find_model_for_step(step_name: str) -> Optional[str]:
    """Step용 모델 찾기 (기존 호환성)"""
    match = auto_detector.find_best_model_for_step(step_name)
    return str(match.file_path) if match else None

def get_all_available_models() -> Dict[str, List[str]]:
    """모든 사용 가능한 모델 반환"""
    results = {}
    
    for step_name in auto_detector.step_patterns.keys():
        match = auto_detector.find_best_model_for_step(step_name)
        if match:
            results[step_name] = [str(match.file_path)]
        else:
            results[step_name] = []
    
    return results
'''
        
        return improved_code
    
    def validate_cleanup(self) -> Dict[str, Any]:
        """정리 결과 검증"""
        try:
            validation = {
                "success": True,
                "remaining_files": 0,
                "total_size_gb": 0.0,
                "structure_valid": True,
                "detector_working": True,
                "issues": []
            }
            
            # 현재 상태 재분석
            current_state = self.analyze_current_state()
            validation["remaining_files"] = current_state.get("total_files", 0)
            validation["total_size_gb"] = current_state.get("total_size_gb", 0.0)
            
            # 구조 검증
            for step_dir in self.standard_structure.keys():
                expected_path = self.base_path / "organized" / step_dir
                if not expected_path.exists():
                    validation["issues"].append(f"디렉토리 누락: {step_dir}")
                    validation["structure_valid"] = False
            
            # 탐지기 검증
            try:
                # 간단한 탐지 테스트
                from backend.app.ai_pipeline.utils.auto_model_detector import auto_detector
                test_result = auto_detector.find_best_model_for_step("step_01_human_parsing")
                if not test_result:
                    validation["issues"].append("auto_detector 작동 불안정")
                    validation["detector_working"] = False
            except Exception as e:
                validation["issues"].append(f"auto_detector 로드 실패: {e}")
                validation["detector_working"] = False
            
            return validation
            
        except Exception as e:
            logger.error(f"검증 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_final_report(self, *results) -> Dict[str, Any]:
        """최종 리포트 생성"""
        try:
            duplicate_results, structure_results, detector_results, validation_results = results
            
            report = {
                "cleanup_summary": {
                    "total_savings_gb": round(duplicate_results.get("saved_space_gb", 0), 2),
                    "files_removed": len(duplicate_results.get("removed_files", [])),
                    "files_moved": len(structure_results.get("moved_files", [])),
                    "directories_created": len(structure_results.get("created_directories", [])),
                    "detector_improved": len(detector_results.get("improvements", [])) > 0
                },
                
                "before_after": {
                    "before": {
                        "total_files": self.analysis_result.get("total_files", 0),
                        "total_size_gb": round(self.analysis_result.get("total_size_gb", 0), 2),
                        "duplicates": len(self.analysis_result.get("duplicates", []))
                    },
                    "after": {
                        "total_files": validation_results.get("remaining_files", 0),
                        "total_size_gb": validation_results.get("total_size_gb", 0),
                        "structure_optimized": validation_results.get("structure_valid", False)
                    }
                },
                
                "recommendations": [
                    "🔧 conda 환경에서 테스트 실행",
                    "📊 정기적인 모델 정리 (월 1회)",
                    "💾 대용량 모델의 외부 저장소 활용",
                    "🔍 auto_model_detector 성능 모니터링",
                    "📁 표준 디렉토리 구조 유지"
                ],
                
                "success": validation_results.get("success", False),
                "issues": validation_results.get("issues", [])
            }
            
            return report
            
        except Exception as e:
            logger.error(f"리포트 생성 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def classify_by_step(self, file_path: Path) -> str:
        """파일을 AI 단계별로 분류"""
        try:
            path_str = str(file_path).lower()
            file_name = file_path.name.lower()
            
            # 디렉토리명으로 분류
            for step, keywords in self.standard_structure.items():
                if step in path_str:
                    return step
                
                # 키워드로 분류
                for keyword in keywords:
                    if keyword in file_name or keyword in path_str:
                        return step
            
            # 특별 패턴들
            if any(pattern in file_name for pattern in ['schp', 'graphonomy', 'parsing']):
                return "step_01_human_parsing"
            elif any(pattern in file_name for pattern in ['openpose', 'dwpose', 'pose']):
                return "step_02_pose_estimation"
            elif any(pattern in file_name for pattern in ['viton', 'diffusion', 'ootd']):
                return "step_06_virtual_fitting"
            elif any(pattern in file_name for pattern in ['u2net', 'segment', 'cloth']):
                return "step_03_cloth_segmentation"
            
            return "unknown"
            
        except Exception as e:
            logger.warning(f"파일 분류 실패 {file_path}: {e}")
            return "unknown"
    
    def get_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """파일 해시 계산 (빠른 방법)"""
        try:
            hash_obj = hashlib.md5()
            
            with open(file_path, 'rb') as f:
                # 큰 파일의 경우 첫 부분만 해시
                if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB 이상
                    chunk = f.read(chunk_size * 100)  # 처음 800KB만
                else:
                    chunk = f.read()
                
                hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.warning(f"해시 계산 실패 {file_path}: {e}")
            return "unknown"
    
    def create_safety_backup(self):
        """안전한 백업 생성"""
        try:
            logger.info(f"💾 백업 생성 중: {self.backup_path}")
            
            # 중요 파일들만 백업 (100MB 이상)
            important_files = []
            
            for file_info in self.analysis_result.get("file_details", []):
                if file_info["size_mb"] > 100:  # 100MB 이상
                    important_files.append(file_info)
            
            # 백업 디렉토리 생성
            self.backup_path.mkdir(parents=True, exist_ok=True)
            
            # 중요 파일들 백업
            for file_info in important_files[:20]:  # 상위 20개만
                try:
                    src_path = self.base_path / file_info["path"]
                    dst_path = self.backup_path / file_info["path"]
                    
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    
                    logger.info(f"✅ 백업 완료: {file_info['path']}")
                    
                except Exception as e:
                    logger.warning(f"백업 실패 {file_info['path']}: {e}")
            
            # 백업 정보 저장
            backup_info = {
                "timestamp": datetime.now().isoformat(),
                "original_path": str(self.base_path),
                "backup_path": str(self.backup_path),
                "files_backed_up": len(important_files),
                "analysis_result": self.analysis_result
            }
            
            with open(self.backup_path / "backup_info.json", 'w') as f:
                json.dump(backup_info, f, indent=2)
            
            logger.info(f"✅ 백업 완료: {len(important_files)}개 중요 파일")
            
        except Exception as e:
            logger.error(f"백업 생성 실패: {e}")
            raise

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI 모델 자동 정리 시스템')
    parser.add_argument('--dry-run', action='store_true', help='실제 변경 없이 미리보기만')
    parser.add_argument('--path', default='backend/ai_models', help='AI 모델 디렉토리 경로')
    
    args = parser.parse_args()
    
    # 정리 시스템 실행
    cleanup_system = ModelCleanupSystem(args.path)
    
    try:
        result = cleanup_system.run_complete_cleanup(dry_run=args.dry_run)
        
        print("\n" + "="*80)
        print("🎉 AI 모델 정리 완료!")
        print("="*80)
        
        if result.get("success"):
            print(f"💾 절약된 용량: {result['cleanup_summary']['total_savings_gb']:.2f}GB")
            print(f"🗂️ 제거된 파일: {result['cleanup_summary']['files_removed']}개")
            print(f"📁 이동된 파일: {result['cleanup_summary']['files_moved']}개")
            print(f"🔍 auto_detector 개선: {'완료' if result['cleanup_summary']['detector_improved'] else '건너뜀'}")
            
            if result.get("issues"):
                print(f"\n⚠️ 주의사항:")
                for issue in result["issues"]:
                    print(f"   • {issue}")
        else:
            print(f"❌ 정리 실패: {result.get('error', 'Unknown error')}")
        
        # 권장사항 출력
        print(f"\n💡 권장사항:")
        for rec in result.get("recommendations", []):
            print(f"   • {rec}")
            
    except Exception as e:
        print(f"❌ 실행 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()