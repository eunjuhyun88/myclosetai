#!/usr/bin/env python3
"""
🔥 수정된 AI 모델 정리 시스템 v2.0
✅ re 모듈 import 오류 해결
✅ 실제 중복 파일 정리 강화
✅ auto_model_detector.py 개선 완료
✅ conda 환경 최적화
"""

import os
import sys
import re  # 🔥 re 모듈 import 추가
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

class FixedModelCleanupSystem:
    """수정된 AI 모델 정리 시스템"""
    
    def __init__(self, base_path: str = "backend/ai_models"):
        self.base_path = Path(base_path)
        self.analysis_result = {}
        
        # 🔥 step_05_cloth_warping 추가
        self.standard_structure = {
            "step_01_human_parsing": ["schp", "graphonomy", "densepose", "parsing"],
            "step_02_pose_estimation": ["openpose", "dwpose", "coco", "pose"],
            "step_03_cloth_segmentation": ["u2net", "cloth", "segmentation", "mask"],
            "step_04_geometric_matching": ["tps", "geometric", "matching", "transform"],
            "step_05_cloth_warping": ["warping", "cloth_warping", "deformation", "tps"],  # 🔥 추가
            "step_06_virtual_fitting": ["viton", "diffusion", "fitting", "ootd"],
            "step_07_post_processing": ["postprocess", "refinement", "enhancement"],
            "step_08_quality_assessment": ["quality", "assessment", "metric"]
        }
    
    def run_advanced_cleanup(self, aggressive_mode: bool = False) -> Dict[str, Any]:
        """고급 정리 모드 실행"""
        try:
            logger.info("🚀 수정된 AI 모델 정리 시스템 v2.0 시작")
            
            # 1단계: 현재 상태 재분석
            logger.info("📊 1단계: 향상된 상태 분석")
            self.analysis_result = self.enhanced_analysis()
            
            # 2단계: 고급 중복 탐지
            logger.info("🔍 2단계: 고급 중복 탐지")
            advanced_duplicates = self.find_advanced_duplicates()
            
            # 3단계: 중복 정리 (aggressive_mode 적용)
            logger.info("🗂️ 3단계: 중복 파일 정리")
            duplicate_results = self.cleanup_advanced_duplicates(
                advanced_duplicates, aggressive=aggressive_mode
            )
            
            # 4단계: step_05_cloth_warping 디렉토리 추가
            logger.info("📁 4단계: 누락된 디렉토리 생성")
            missing_dirs = self.create_missing_directories()
            
            # 5단계: auto_model_detector.py 수정
            logger.info("🔧 5단계: auto_model_detector.py 수정")
            detector_results = self.fix_auto_detector()
            
            # 6단계: 파일 이동 최적화
            logger.info("📋 6단계: 파일 이동 최적화")
            movement_results = self.optimize_file_placement()
            
            # 최종 리포트
            final_report = self.generate_enhanced_report(
                duplicate_results, missing_dirs, detector_results, movement_results
            )
            
            logger.info("🎉 수정된 AI 모델 정리 완료!")
            return final_report
            
        except Exception as e:
            logger.error(f"❌ 정리 프로세스 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def enhanced_analysis(self) -> Dict[str, Any]:
        """향상된 상태 분석"""
        try:
            analysis = {
                "total_files": 0,
                "total_size_gb": 0.0,
                "by_extension": defaultdict(int),
                "by_step": defaultdict(int),
                "file_details": [],
                "large_files": [],
                "potential_duplicates": []
            }
            
            # 더 정교한 파일 스캔
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file():
                    try:
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        analysis["total_size_gb"] += size_mb / 1024
                        analysis["total_files"] += 1
                        
                        ext = file_path.suffix.lower()
                        analysis["by_extension"][ext] += 1
                        
                        step = self.classify_by_step_enhanced(file_path)
                        analysis["by_step"][step] += 1
                        
                        file_info = {
                            "path": str(file_path.relative_to(self.base_path)),
                            "size_mb": round(size_mb, 2),
                            "extension": ext,
                            "step": step,
                            "modified": file_path.stat().st_mtime,
                            "name": file_path.name,
                            "stem": file_path.stem
                        }
                        analysis["file_details"].append(file_info)
                        
                        # 대용량 파일 추적 (100MB 이상)
                        if size_mb > 100:
                            analysis["large_files"].append(file_info)
                            
                    except Exception as e:
                        logger.warning(f"파일 분석 실패 {file_path}: {e}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"향상된 분석 실패: {e}")
            return {}
    
    def find_advanced_duplicates(self) -> List[Dict[str, Any]]:
        """고급 중복 탐지"""
        duplicates = []
        
        try:
            # 1. 파일명 기반 중복 탐지 (더 정교함)
            name_groups = defaultdict(list)
            for file_info in self.analysis_result.get("file_details", []):
                # 버전 번호 제거한 기본 이름
                clean_name = re.sub(r'_\d+$|_v\d+$|_final$|_best$', '', file_info["stem"])
                name_groups[clean_name].append(file_info)
            
            for base_name, files in name_groups.items():
                if len(files) > 1:
                    # 크기로 정렬 (큰 것이 보통 더 좋음)
                    files.sort(key=lambda x: x["size_mb"], reverse=True)
                    
                    duplicates.append({
                        "type": "name_similarity",
                        "base_name": base_name,
                        "files": files,
                        "recommended_keep": files[0],  # 가장 큰 파일
                        "recommended_remove": files[1:],
                        "savings_mb": sum(f["size_mb"] for f in files[1:])
                    })
            
            # 2. 동일 확장자 + 크기 기반 중복
            size_ext_groups = defaultdict(list)
            for file_info in self.analysis_result.get("file_details", []):
                key = f"{file_info['extension']}_{file_info['size_mb']}"
                size_ext_groups[key].append(file_info)
            
            for key, files in size_ext_groups.items():
                if len(files) > 1:
                    duplicates.append({
                        "type": "size_extension_match",
                        "key": key,
                        "files": files,
                        "recommended_keep": files[0],
                        "recommended_remove": files[1:],
                        "savings_mb": sum(f["size_mb"] for f in files[1:])
                    })
            
            # 3. 단계별 과도한 집중 탐지
            step_counts = defaultdict(int)
            for file_info in self.analysis_result.get("file_details", []):
                step_counts[file_info["step"]] += 1
            
            # step_01에 너무 많은 파일이 있는 경우
            if step_counts.get("step_01_human_parsing", 0) > 200:
                step_01_files = [f for f in self.analysis_result["file_details"] 
                               if f["step"] == "step_01_human_parsing"]
                
                # 크기가 작은 파일들 중복 의심
                small_files = [f for f in step_01_files if f["size_mb"] < 10]
                if len(small_files) > 50:
                    duplicates.append({
                        "type": "step_concentration",
                        "step": "step_01_human_parsing",
                        "files": small_files[:30],  # 상위 30개만
                        "recommended_remove": small_files[:20],  # 20개 제거 권장
                        "savings_mb": sum(f["size_mb"] for f in small_files[:20])
                    })
            
            return duplicates
            
        except Exception as e:
            logger.error(f"고급 중복 탐지 실패: {e}")
            return []
    
    def cleanup_advanced_duplicates(
        self, 
        duplicates: List[Dict[str, Any]], 
        aggressive: bool = False
    ) -> Dict[str, Any]:
        """고급 중복 정리"""
        results = {
            "removed_files": [],
            "saved_space_gb": 0.0,
            "errors": []
        }
        
        try:
            for duplicate_group in duplicates:
                # aggressive 모드가 아니면 안전한 것만 제거
                if not aggressive and duplicate_group["type"] == "step_concentration":
                    continue
                
                files_to_remove = duplicate_group.get("recommended_remove", [])
                
                for file_info in files_to_remove:
                    try:
                        file_path = self.base_path / file_info["path"]
                        
                        if file_path.exists():
                            # 백업 생성
                            backup_path = self.base_path / "cleanup_backup" / file_info["path"]
                            backup_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(file_path, backup_path)
                            
                            # 원본 삭제
                            file_path.unlink()
                            
                            results["removed_files"].append(file_info["path"])
                            results["saved_space_gb"] += file_info["size_mb"] / 1024
                            
                            logger.info(f"✅ 중복 파일 제거: {file_path.name}")
                            
                    except Exception as e:
                        error_msg = f"파일 제거 실패 {file_info['path']}: {e}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)
            
            return results
            
        except Exception as e:
            logger.error(f"중복 정리 실패: {e}")
            return {"errors": [str(e)]}
    
    def create_missing_directories(self) -> Dict[str, Any]:
        """누락된 디렉토리 생성"""
        results = {
            "created_directories": [],
            "errors": []
        }
        
        try:
            organized_path = self.base_path / "organized"
            
            for step_name in self.standard_structure.keys():
                step_dir = organized_path / step_name
                
                if not step_dir.exists():
                    step_dir.mkdir(parents=True, exist_ok=True)
                    results["created_directories"].append(step_name)
                    logger.info(f"✅ 디렉토리 생성: {step_name}")
            
            return results
            
        except Exception as e:
            logger.error(f"디렉토리 생성 실패: {e}")
            return {"errors": [str(e)]}
    
    def fix_auto_detector(self) -> Dict[str, Any]:
        """auto_model_detector.py 수정"""
        results = {
            "fixed": False,
            "issues_found": [],
            "fixes_applied": []
        }
        
        try:
            detector_path = Path("backend/app/ai_pipeline/utils/auto_model_detector.py")
            
            if not detector_path.exists():
                results["issues_found"].append("auto_model_detector.py 파일 없음")
                return results
            
            # 파일 읽기
            with open(detector_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 문제점 체크 및 수정
            fixes_needed = []
            
            # 1. re 모듈 import 체크
            if 'import re' not in content:
                fixes_needed.append("re 모듈 import 추가")
                content = "import re\n" + content
            
            # 2. auto_detector 인스턴스 export 체크
            if 'auto_detector = ' not in content:
                fixes_needed.append("auto_detector 인스턴스 추가")
                content += "\n\n# 전역 인스턴스\nauto_detector = ImprovedAutoModelDetector()\n"
            
            # 3. __all__ 체크
            if '__all__' not in content:
                fixes_needed.append("__all__ 추가")
                content += "\n__all__ = ['auto_detector', 'ImprovedAutoModelDetector']\n"
            
            # 수정사항이 있으면 파일 저장
            if fixes_needed:
                # 백업 생성
                backup_path = detector_path.with_suffix('.py.backup')
                shutil.copy2(detector_path, backup_path)
                
                # 수정된 내용 저장
                with open(detector_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                results["fixed"] = True
                results["fixes_applied"] = fixes_needed
                logger.info(f"✅ auto_model_detector.py 수정 완료: {fixes_needed}")
            else:
                results["fixed"] = True
                results["fixes_applied"] = ["이미 정상 상태"]
            
            return results
            
        except Exception as e:
            logger.error(f"auto_detector 수정 실패: {e}")
            return {"errors": [str(e)]}
    
    def optimize_file_placement(self) -> Dict[str, Any]:
        """파일 배치 최적화"""
        results = {
            "moved_files": [],
            "errors": []
        }
        
        try:
            organized_path = self.base_path / "organized"
            
            # step_01에 집중된 파일들 재분배
            step_01_files = [f for f in self.analysis_result.get("file_details", [])
                           if f["step"] == "step_01_human_parsing"]
            
            # 파일명으로 다른 단계로 이동할 수 있는 것들 찾기
            for file_info in step_01_files:
                file_path = self.base_path / file_info["path"]
                
                if not file_path.exists():
                    continue
                
                # 더 적절한 단계 찾기
                better_step = self.find_better_step_placement(file_info["name"])
                
                if better_step and better_step != "step_01_human_parsing":
                    try:
                        target_dir = organized_path / better_step
                        target_dir.mkdir(parents=True, exist_ok=True)
                        
                        target_path = target_dir / file_path.name
                        
                        # 중복 방지
                        if not target_path.exists():
                            shutil.move(str(file_path), str(target_path))
                            results["moved_files"].append({
                                "from": file_info["path"],
                                "to": str(target_path.relative_to(self.base_path)),
                                "reason": f"재분류: {better_step}"
                            })
                            logger.info(f"📋 파일 이동: {file_path.name} → {better_step}")
                    
                    except Exception as e:
                        results["errors"].append(f"파일 이동 실패 {file_info['path']}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"파일 배치 최적화 실패: {e}")
            return {"errors": [str(e)]}
    
    def find_better_step_placement(self, filename: str) -> Optional[str]:
        """파일에 더 적합한 단계 찾기"""
        filename_lower = filename.lower()
        
        # 키워드 기반 매칭
        step_keywords = {
            "step_02_pose_estimation": ["pose", "openpose", "dwpose", "coco", "mpii"],
            "step_03_cloth_segmentation": ["u2net", "segment", "mask", "cloth"],
            "step_04_geometric_matching": ["tps", "geometric", "matching", "transform"],
            "step_05_cloth_warping": ["warp", "deform", "flow"],
            "step_06_virtual_fitting": ["viton", "diffusion", "fitting", "ootd", "stable"],
            "step_07_post_processing": ["enhance", "refine", "post", "super"],
            "step_08_quality_assessment": ["quality", "metric", "eval", "assess"]
        }
        
        for step, keywords in step_keywords.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    return step
        
        return None
    
    def classify_by_step_enhanced(self, file_path: Path) -> str:
        """향상된 단계별 분류"""
        try:
            path_str = str(file_path).lower()
            file_name = file_path.name.lower()
            
            # 디렉토리명으로 분류 (1차)
            for step, keywords in self.standard_structure.items():
                if step in path_str:
                    return step
            
            # 키워드로 분류 (2차)
            for step, keywords in self.standard_structure.items():
                for keyword in keywords:
                    if keyword in file_name or keyword in path_str:
                        return step
            
            # 특별 패턴들 (3차)
            if any(pattern in file_name for pattern in ['schp', 'graphonomy', 'parsing']):
                return "step_01_human_parsing"
            elif any(pattern in file_name for pattern in ['openpose', 'dwpose', 'pose']):
                return "step_02_pose_estimation"
            elif any(pattern in file_name for pattern in ['viton', 'diffusion', 'ootd']):
                return "step_06_virtual_fitting"
            elif any(pattern in file_name for pattern in ['u2net', 'segment', 'cloth']):
                return "step_03_cloth_segmentation"
            elif any(pattern in file_name for pattern in ['warp', 'flow', 'deform']):
                return "step_05_cloth_warping"
            
            return "unknown"
            
        except Exception as e:
            logger.warning(f"파일 분류 실패 {file_path}: {e}")
            return "unknown"
    
    def generate_enhanced_report(self, *results) -> Dict[str, Any]:
        """향상된 최종 리포트"""
        try:
            duplicate_results, missing_dirs, detector_results, movement_results = results
            
            report = {
                "success": True,
                "version": "2.0",
                "cleanup_summary": {
                    "files_removed": len(duplicate_results.get("removed_files", [])),
                    "space_saved_gb": round(duplicate_results.get("saved_space_gb", 0), 2),
                    "directories_created": len(missing_dirs.get("created_directories", [])),
                    "files_moved": len(movement_results.get("moved_files", [])),
                    "detector_fixed": detector_results.get("fixed", False)
                },
                "before_after": {
                    "before": {
                        "total_files": self.analysis_result.get("total_files", 0),
                        "total_size_gb": round(self.analysis_result.get("total_size_gb", 0), 2)
                    },
                    "estimated_after": {
                        "total_files": self.analysis_result.get("total_files", 0) - len(duplicate_results.get("removed_files", [])),
                        "total_size_gb": round(self.analysis_result.get("total_size_gb", 0) - duplicate_results.get("saved_space_gb", 0), 2)
                    }
                },
                "detailed_results": {
                    "duplicates": duplicate_results,
                    "directories": missing_dirs,
                    "detector": detector_results,
                    "movements": movement_results
                },
                "next_steps": [
                    "🔧 수정된 auto_model_detector.py 테스트",
                    "📊 step별 파일 분포 재확인",
                    "💾 백업된 파일들 검토",
                    "🔍 추가 최적화 기회 탐색"
                ]
            }
            
            return report
            
        except Exception as e:
            logger.error(f"리포트 생성 실패: {e}")
            return {"success": False, "error": str(e)}

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='수정된 AI 모델 정리 시스템 v2.0')
    parser.add_argument('--aggressive', action='store_true', help='적극적 정리 모드')
    parser.add_argument('--path', default='backend/ai_models', help='AI 모델 디렉토리 경로')
    
    args = parser.parse_args()
    
    # 수정된 정리 시스템 실행
    cleanup_system = FixedModelCleanupSystem(args.path)
    
    try:
        result = cleanup_system.run_advanced_cleanup(aggressive_mode=args.aggressive)
        
        print("\n" + "="*80)
        print("🎉 수정된 AI 모델 정리 v2.0 완료!")
        print("="*80)
        
        if result.get("success"):
            summary = result["cleanup_summary"]
            print(f"🗂️ 제거된 파일: {summary['files_removed']}개")
            print(f"💾 절약된 용량: {summary['space_saved_gb']:.2f}GB")
            print(f"📁 생성된 디렉토리: {summary['directories_created']}개")
            print(f"📋 이동된 파일: {summary['files_moved']}개")
            print(f"🔧 auto_detector 수정: {'✅' if summary['detector_fixed'] else '❌'}")
            
            before_after = result["before_after"]
            print(f"\n📈 Before → After:")
            print(f"   파일: {before_after['before']['total_files']}개 → {before_after['estimated_after']['total_files']}개")
            print(f"   용량: {before_after['before']['total_size_gb']:.1f}GB → {before_after['estimated_after']['total_size_gb']:.1f}GB")
            
            print(f"\n🔧 다음 단계:")
            for step in result["next_steps"]:
                print(f"   • {step}")
        else:
            print(f"❌ 정리 실패: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ 실행 실패: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()