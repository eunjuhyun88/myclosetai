#!/usr/bin/env python3
"""
🔥 AI 모델 완전 통합 정리 스크립트 v3.0
모든 중복된 .pth 파일들을 backend/app/ai_pipeline/models로 완전 통합
✅ 중복 제거 - 같은 파일은 하나만 유지  
✅ Step별 자동 분류
✅ 파일명 끝부분 기준 스마트 매칭
✅ 안전한 복사 (원본 유지)
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set
import hashlib
import re

class SmartModelConsolidator:
    """스마트 AI 모델 완전 통합 정리기"""
    
    def __init__(self):
        self.project_root = Path("/Users/gimdudeul/MVP/mycloset-ai")
        self.target_path = self.project_root / "backend/app/ai_pipeline/models"
        self.search_paths = [
            self.project_root / "ai_models",
            self.project_root / "backend/ai_models"  
        ]
        
        # Step별 최종 정리 구조 
        self.target_structure = {
            "step_01_human_parsing": [
                r".*schp.*atr.*\.pth$",
                r".*schp.*lip.*\.pth$", 
                r".*exp-schp.*atr.*\.pth$",
                r".*exp-schp.*lip.*\.pth$",
                r".*graphonomy.*\.pth$",
                r".*human.*parsing.*\.pth$",
                r".*atr.*model.*\.pth$",
                r".*lip.*model.*\.pth$"
            ],
            "step_02_pose_estimation": [
                r".*openpose.*\.pth$",
                r".*body.*pose.*model.*\.pth$",
                r".*hand.*pose.*model.*\.pth$"
            ],
            "step_03_cloth_segmentation": [
                r".*u2net.*\.pth$",
                r".*sam.*vit.*\.pth$",
                r".*cloth.*seg.*\.pth$",
                r".*model\.pth$"  # cloth_segmentation 폴더의 model.pth
            ],
            "step_04_geometric_matching": [
                r".*gmm.*\.pth$",
                r".*geometric.*\.pth$"
            ],
            "step_05_cloth_warping": [
                r".*tom.*\.pth$", 
                r".*warp.*\.pth$"
            ],
            "step_06_virtual_fitting": [
                r".*hrviton.*\.pth$",
                r".*ootd.*\.pth$",
                r".*vton.*\.pth$",
                r".*virtual.*fitting.*\.pth$"
            ],
            "step_07_post_processing": [
                r".*codeformer.*\.pth$",
                r".*gfpgan.*\.pth$", 
                r".*real.*esrgan.*\.pth$",
                r".*swinir.*\.pth$",
                r".*enhancer.*\.pth$",
                r".*upscale.*\.pth$"
            ],
            "step_08_quality_assessment": [
                r".*clip.*\.pth$",
                r".*quality.*\.pth$"
            ]
        }
        
        # 결과 추적
        self.results = {
            "found_files": [],
            "moved_files": [],
            "duplicates_removed": [],
            "errors": []
        }
        
    def find_all_pth_files(self) -> List[Path]:
        """모든 .pth 파일 찾기"""
        print("🔍 전체 .pth 파일 스캔 중...")
        
        all_files = []
        for search_path in self.search_paths:
            if search_path.exists():
                pth_files = list(search_path.rglob("*.pth"))
                all_files.extend(pth_files)
                print(f"📂 {search_path}: {len(pth_files)}개 발견")
        
        # 중복 경로 제거 (실제 같은 파일)
        unique_files = []
        seen_paths = set()
        
        for file_path in all_files:
            try:
                resolved_path = file_path.resolve()
                if resolved_path not in seen_paths:
                    seen_paths.add(resolved_path)
                    unique_files.append(file_path)
            except:
                unique_files.append(file_path)
        
        print(f"📊 총 발견: {len(all_files)}개, 고유 파일: {len(unique_files)}개")
        
        # 크기별 정렬 (큰 파일부터 - 더 완전한 모델일 가능성)
        unique_files.sort(key=lambda x: self.get_file_size(x), reverse=True)
        
        return unique_files
    
    def get_file_size(self, file_path: Path) -> int:
        """파일 크기 가져오기"""
        try:
            return file_path.stat().st_size
        except:
            return 0
    
    def get_file_hash(self, file_path: Path) -> str:
        """파일 해시 계산"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return ""
    
    def classify_by_filename(self, file_path: Path) -> Tuple[str, float]:
        """파일명 끝부분 기준으로 Step 분류"""
        filename = file_path.name.lower()
        
        # Step별 패턴 매칭
        for step_name, patterns in self.target_structure.items():
            for pattern in patterns:
                if re.search(pattern, filename):
                    # 파일 크기로 우선순위 결정 (큰 파일이 더 완전)
                    size_mb = self.get_file_size(file_path) / (1024 * 1024)
                    confidence = min(10.0, 5.0 + (size_mb / 100))  # 크기에 따른 가중치
                    return step_name, confidence
        
        # 특별한 경우들
        special_cases = {
            "hrviton_final.pth": ("step_06_virtual_fitting", 10.0),
            "openpose_body.pth": ("step_02_pose_estimation", 10.0), 
            "u2net.pth": ("step_03_cloth_segmentation", 10.0),
            "graphonomy.pth": ("step_01_human_parsing", 10.0)
        }
        
        for special_name, (step, score) in special_cases.items():
            if special_name in filename:
                return step, score
        
        return "misc", 0.0
    
    def remove_duplicates(self, file_list: List[Path]) -> List[Path]:
        """중복 파일 제거 (해시 기준)"""
        print("🔄 중복 파일 제거 중...")
        
        hash_to_file = {}
        unique_files = []
        
        for file_path in file_list:
            file_hash = self.get_file_hash(file_path)
            
            if not file_hash:
                unique_files.append(file_path)  # 해시 실패시 일단 포함
                continue
                
            if file_hash in hash_to_file:
                # 중복 발견 - 더 좋은 위치의 파일 선택
                existing_file = hash_to_file[file_hash]
                
                # backup 폴더에 있는 파일은 제외
                if "backup" in str(file_path).lower():
                    self.results["duplicates_removed"].append({
                        "removed": str(file_path),
                        "kept": str(existing_file),
                        "reason": "backup_file"
                    })
                    continue
                elif "backup" in str(existing_file).lower():
                    # 기존 파일이 백업이면 새 파일로 교체
                    hash_to_file[file_hash] = file_path
                    unique_files = [f for f in unique_files if f != existing_file]
                    unique_files.append(file_path)
                    self.results["duplicates_removed"].append({
                        "removed": str(existing_file),
                        "kept": str(file_path), 
                        "reason": "replaced_backup"
                    })
                else:
                    # 파일 크기로 결정 (큰 파일 우선)
                    if self.get_file_size(file_path) > self.get_file_size(existing_file):
                        hash_to_file[file_hash] = file_path
                        unique_files = [f for f in unique_files if f != existing_file]
                        unique_files.append(file_path)
                        self.results["duplicates_removed"].append({
                            "removed": str(existing_file),
                            "kept": str(file_path),
                            "reason": "larger_file"
                        })
                    else:
                        self.results["duplicates_removed"].append({
                            "removed": str(file_path),
                            "kept": str(existing_file),
                            "reason": "smaller_file"
                        })
            else:
                hash_to_file[file_hash] = file_path
                unique_files.append(file_path)
        
        print(f"🗑️ 중복 제거: {len(file_list) - len(unique_files)}개")
        return unique_files
    
    def copy_file_safely(self, source: Path, target_dir: Path, new_name: str = None) -> bool:
        """안전한 파일 복사 - 중복되면 번호 추가"""
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            
            if new_name:
                base_name = Path(new_name).stem
                extension = Path(new_name).suffix
            else:
                base_name = source.stem
                extension = source.suffix
                
            target_path = target_dir / f"{base_name}{extension}"
            
            # 중복되면 번호 추가 (_01, _02, ...)
            counter = 1
            while target_path.exists():
                target_path = target_dir / f"{base_name}_{counter:02d}{extension}"
                counter += 1
            
            if counter > 1:
                print(f"🔢 이름 변경: {source.name} → {target_path.name}")
            
            shutil.copy2(source, target_path)
            
            # 복사 검증
            if target_path.exists() and target_path.stat().st_size > 0:
                size_mb = target_path.stat().st_size / (1024 * 1024)
                self.results["moved_files"].append({
                    "source": str(source),
                    "target": str(target_path),
                    "size_mb": round(size_mb, 1)
                })
                return True
            else:
                raise Exception("복사 검증 실패")
                
        except Exception as e:
            self.results["errors"].append({
                "file": str(source),
                "error": str(e)
            })
            return False
    
    def consolidate_all_models(self) -> bool:
        """메인 통합 프로세스"""
        print("🚀 AI 모델 완전 통합 시작!")
        print("=" * 60)
        
        # 1. 모든 .pth 파일 찾기
        all_files = self.find_all_pth_files()
        self.results["found_files"] = [str(f) for f in all_files]
        
        if not all_files:
            print("❌ .pth 파일을 찾을 수 없습니다.")
            return False
        
        # 2. 모든 파일 그대로 유지 (중복 제거 안함)
        unique_files = all_files
        print("📋 모든 파일을 그대로 이동합니다 (중복 제거 안함)")
        
        # 3. 대상 디렉토리 준비
        checkpoints_dir = self.target_path / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔄 모든 모델 이동 중... ({len(unique_files)}개)")
        
        step_counts = {}
        misc_count = 0
        
        # 4. 파일별 분류 및 이동
        for file_path in unique_files:
            try:
                step_name, confidence = self.classify_by_filename(file_path)
                
                if step_name == "misc":
                    # 분류되지 않은 파일은 misc 폴더로
                    target_dir = checkpoints_dir / "misc"
                    print(f"❓ 분류 불가: {file_path.name}")
                    misc_count += 1
                else:
                    # Step별 폴더로 이동
                    target_dir = checkpoints_dir / step_name
                    print(f"✅ {step_name}: {file_path.name} ({confidence:.1f})")
                    step_counts[step_name] = step_counts.get(step_name, 0) + 1
                
                # 파일 복사
                self.copy_file_safely(file_path, target_dir)
                
            except Exception as e:
                print(f"❌ 처리 실패 {file_path}: {e}")
                self.results["errors"].append({
                    "file": str(file_path),
                    "error": str(e)
                })
        
        # 5. 결과 출력
        self.print_final_summary(step_counts, misc_count)
        self.save_consolidation_report()
        
        return len(self.results["errors"]) == 0
    
    def print_final_summary(self, step_counts: Dict[str, int], misc_count: int):
        """최종 결과 요약"""
        print("\n" + "=" * 60)
        print("🎉 AI 모델 완전 통합 완료!")
        print("=" * 60)
        
        total_found = len(self.results["found_files"])
        total_moved = len(self.results["moved_files"])
        total_errors = len(self.results["errors"])
        
        print(f"📊 전체 발견: {total_found}개")
        print(f"✅ 모두 이동: {total_moved}개")
        print(f"❌ 오류 발생: {total_errors}개")
        
        if step_counts:
            print(f"\n📁 Step별 정리 결과:")
            for step_name in sorted(step_counts.keys()):
                count = step_counts[step_name]
                print(f"  {step_name}: {count}개")
            
            if misc_count > 0:
                print(f"  misc (분류 불가): {misc_count}개")
        
        print(f"\n📁 최종 정리 위치:")
        print(f"  {self.target_path}/checkpoints/")
        for step_name in sorted(step_counts.keys()):
            print(f"    ├── {step_name}/")
        if misc_count > 0:
            print(f"    └── misc/")
    
    def save_consolidation_report(self):
        """통합 리포트 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.target_path / f"consolidation_report_{timestamp}.json"
        
        report = {
            "timestamp": timestamp,
            "summary": {
                "total_found": len(self.results["found_files"]),
                "total_moved": len(self.results["moved_files"]),
                "duplicates_removed": len(self.results["duplicates_removed"]),
                "errors": len(self.results["errors"])
            },
            "details": self.results
        }
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n📊 상세 리포트: {report_path}")
        except Exception as e:
            print(f"⚠️ 리포트 저장 실패: {e}")

def main():
    """메인 실행"""
    print("🤖 MyCloset AI 모델 완전 통합 정리기 v3.0")
    print("=" * 60)
    
    consolidator = SmartModelConsolidator()
    
    # 경로 확인
    for path in consolidator.search_paths:
        if path.exists():
            print(f"✅ 소스 경로: {path}")
        else:
            print(f"❌ 소스 경로 없음: {path}")
    
    print(f"📁 통합 대상: {consolidator.target_path}")
    
    # 확인
    response = input("\n🔄 모든 .pth 파일을 복사해서 이동하시겠습니까? (중복 제거 안함) (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ 작업이 취소되었습니다.")
        return False
    
    # 실행
    success = consolidator.consolidate_all_models()
    
    if success:
        print("\n🎉 모든 모델이 성공적으로 통합되었습니다!")
        print("💡 이제 중복 없이 깔끔하게 정리된 모델들을 사용하세요!")
    else:
        print("\n⚠️ 일부 오류가 발생했습니다. 리포트를 확인해주세요.")
    
    return success

if __name__ == "__main__":
    main()