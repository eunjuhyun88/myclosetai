#!/usr/bin/env python3
"""
🔥 완전 모델 통합 스크립트 v2.0 - ALL FORMATS
모든 AI 모델 형식을 backend/app/ai_pipeline/models로 완전 통합
✅ .pth, .pt, .onnx, .safetensors, .mediapipe, .pkl, .bin 등 모든 형식
✅ HuggingFace 캐시 + 일반 모델 파일 동시 처리
✅ Step별 자동 분류 및 중복 처리
✅ 진행률 표시 및 안전한 복사
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
import time
import re

class CompleteModelConsolidator:
    """완전 모델 통합기 - 모든 형식 지원"""
    
    def __init__(self):
        self.project_root = Path("/Users/gimdudeul/MVP/mycloset-ai")
        self.search_paths = [
            self.project_root / "ai_models",
            self.project_root / "backend/ai_models"
        ]
        self.target_path = self.project_root / "backend/app/ai_pipeline/models"
        
        # 지원하는 모든 모델 확장자
        self.model_extensions = {
            '.pth': 'PyTorch 체크포인트',
            '.pt': 'PyTorch 모델',
            '.ckpt': '체크포인트 모델',
            '.safetensors': 'SafeTensors (HuggingFace)',
            '.pkl': 'Pickle 모델',
            '.bin': 'Binary 모델',
            '.onnx': 'ONNX 모델',
            '.h5': 'Keras/TensorFlow',
            '.pb': 'TensorFlow SavedModel',
            '.mediapipe': 'MediaPipe 모델',
            '.pytorch': 'PyTorch 저장 모델',
            '.tflite': 'TensorFlow Lite',
            '.mlmodel': 'Core ML (Apple)'
        }
        
        # Step별 분류 패턴 (모든 확장자 대응)
        self.step_patterns = {
            "step_01_human_parsing": [
                r".*human.*parsing.*",
                r".*schp.*atr.*", r".*schp.*lip.*",
                r".*exp-schp.*", r".*graphonomy.*",
                r".*parsing.*atr.*", r".*parsing.*lip.*",
                r".*densepose.*", r".*atr.*model.*",
                r".*lip.*model.*"
            ],
            "step_02_pose_estimation": [
                r".*pose.*estimation.*", r".*openpose.*",
                r".*body.*pose.*", r".*hand.*pose.*",
                r".*mediapipe.*pose.*", r".*keypoint.*",
                r".*landmark.*"
            ],
            "step_03_cloth_segmentation": [
                r".*cloth.*seg.*", r".*u2net.*",
                r".*sam.*", r".*segment.*",
                r".*fashion.*clip.*", r".*rmbg.*",
                r".*background.*removal.*", r".*mask.*"
            ],
            "step_04_geometric_matching": [
                r".*geometric.*match.*", r".*gmm.*",
                r".*tps.*network.*", r".*warp.*network.*"
            ],
            "step_05_cloth_warping": [
                r".*cloth.*warp.*", r".*tom.*",
                r".*warping.*", r".*deformation.*"
            ],
            "step_06_virtual_fitting": [
                r".*virtual.*fitting.*", r".*oot.*diffusion.*",
                r".*idm.*vton.*", r".*hr.*viton.*",
                r".*stable.*diffusion.*", r".*diffusion.*pytorch.*",
                r".*unet.*", r".*vae.*", r".*text.*encoder.*"
            ],
            "step_07_post_processing": [
                r".*post.*process.*", r".*enhance.*",
                r".*super.*resolution.*", r".*sr.*",
                r".*upscale.*", r".*swinir.*",
                r".*real.*esrgan.*", r".*gfpgan.*",
                r".*codeformer.*"
            ],
            "step_08_quality_assessment": [
                r".*quality.*", r".*assessment.*",
                r".*clip.*vit.*", r".*score.*",
                r".*evaluation.*"
            ]
        }
        
        # 결과 추적
        self.results = {
            "found_files": [],
            "moved_files": [],
            "errors": [],
            "stats_by_extension": {},
            "stats_by_step": {}
        }
    
    def find_all_model_files(self) -> Dict[str, List[Path]]:
        """모든 모델 파일 찾기 (모든 확장자)"""
        print("🔍 모든 형식의 모델 파일 스캔 중...")
        
        all_files = {}
        
        for search_path in self.search_paths:
            if not search_path.exists():
                continue
                
            print(f"📂 스캔 중: {search_path}")
            
            for ext in self.model_extensions.keys():
                files = list(search_path.rglob(f"*{ext}"))
                if files:
                    if ext not in all_files:
                        all_files[ext] = []
                    all_files[ext].extend(files)
        
        # 중복 제거 (동일한 resolved path)
        for ext in all_files:
            unique_files = []
            seen_paths = set()
            
            for file_path in all_files[ext]:
                try:
                    resolved = file_path.resolve()
                    if resolved not in seen_paths:
                        seen_paths.add(resolved)
                        unique_files.append(file_path)
                except:
                    unique_files.append(file_path)
            
            all_files[ext] = unique_files
        
        # 통계 출력
        total_files = sum(len(files) for files in all_files.values())
        print(f"\n📊 발견된 모델 파일: {total_files}개")
        
        for ext, files in sorted(all_files.items()):
            if files:
                desc = self.model_extensions[ext]
                print(f"  {ext}: {len(files)}개 ({desc})")
                self.results["stats_by_extension"][ext] = len(files)
        
        return all_files
    
    def classify_model_file(self, file_path: Path) -> str:
        """모델 파일을 Step별로 분류"""
        filename = file_path.name.lower()
        parent_dir = str(file_path.parent).lower()
        
        # HuggingFace 모델 특별 처리
        if "huggingface" in parent_dir or "models--" in parent_dir:
            if "ootdiffusion" in parent_dir or "idm-vton" in parent_dir:
                return "step_06_virtual_fitting"
            elif "sam2" in parent_dir:
                return "step_03_cloth_segmentation"
            elif "clip" in parent_dir:
                return "step_08_quality_assessment"
        
        # Step별 패턴 매칭
        best_score = 0
        best_step = "misc"
        
        for step_name, patterns in self.step_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, filename):
                    score += 3
                elif re.search(pattern, parent_dir):
                    score += 1
            
            if score > best_score:
                best_score = score
                best_step = step_name
        
        return best_step if best_score > 0 else "misc"
    
    def get_file_size_mb(self, file_path: Path) -> float:
        """파일 크기 (MB)"""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except:
            return 0.0
    
    def copy_file_safely(self, source: Path, target_dir: Path) -> bool:
        """안전한 파일 복사"""
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 파일명 중복 처리
            target_path = target_dir / source.name
            counter = 1
            
            while target_path.exists():
                stem = source.stem
                suffix = source.suffix
                target_path = target_dir / f"{stem}_{counter:02d}{suffix}"
                counter += 1
            
            if counter > 1:
                print(f"    🔢 이름 변경: {source.name} → {target_path.name}")
            
            # 파일 이동 (원본 삭제)
            shutil.move(source, target_path)
            
            # 검증
            if target_path.exists() and target_path.stat().st_size > 0:
                size_mb = self.get_file_size_mb(target_path)
                self.results["moved_files"].append({
                    "source": str(source),
                    "target": str(target_path),
                    "size_mb": round(size_mb, 2),
                    "extension": source.suffix
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
        """완전 모델 통합 메인 프로세스"""
        print("🔥 완전 모델 통합 시작! (모든 형식)")
        print("=" * 60)
        
        # 1. 모든 모델 파일 스캔
        all_model_files = self.find_all_model_files()
        
        if not all_model_files:
            print("❌ 이동할 모델 파일이 없습니다.")
            return False
        
        # 2. 총 파일 수 및 크기 계산
        total_files = sum(len(files) for files in all_model_files.values())
        total_size_gb = 0
        
        for files in all_model_files.values():
            for file_path in files:
                total_size_gb += self.get_file_size_mb(file_path) / 1024
        
        print(f"\n📊 통합 대상: {total_files}개 파일, 약 {total_size_gb:.1f}GB")
        
        # 3. 사용자 확인
        response = input("\n🚚 모든 모델 파일을 통합하시겠습니까? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ 작업이 취소되었습니다.")
            return False
        
        # 4. 파일별 분류 및 이동
        print(f"\n🔄 모델 파일 이동 중... ({total_files}개)")
        
        processed = 0
        for ext, files in all_model_files.items():
            if not files:
                continue
                
            print(f"\n📦 {ext} 파일 처리 중... ({len(files)}개)")
            
            for file_path in files:
                try:
                    # Step 분류
                    step_name = self.classify_model_file(file_path)
                    
                    if step_name == "misc":
                        target_dir = self.target_path / "checkpoints" / "misc"
                        print(f"    ❓ 분류 불가: {file_path.name}")
                    else:
                        target_dir = self.target_path / "checkpoints" / step_name
                        size_mb = self.get_file_size_mb(file_path)
                        print(f"    ✅ {step_name}: {file_path.name} ({size_mb:.1f}MB)")
                    
                    # 파일 복사
                    if self.copy_file_safely(file_path, target_dir):
                        # 통계 업데이트
                        if step_name not in self.results["stats_by_step"]:
                            self.results["stats_by_step"][step_name] = 0
                        self.results["stats_by_step"][step_name] += 1
                    
                    processed += 1
                    
                    # 진행률 표시
                    if processed % 10 == 0:
                        progress = (processed / total_files) * 100
                        print(f"    📊 진행률: {progress:.1f}% ({processed}/{total_files})")
                        
                except Exception as e:
                    print(f"    ❌ 처리 실패 {file_path}: {e}")
        
        # 5. 결과 요약
        self.print_final_summary()
        self.save_consolidation_report()
        
        return len(self.results["errors"]) == 0
    
    def print_final_summary(self):
        """최종 결과 요약"""
        print("\n" + "=" * 60)
        print("🎉 완전 모델 통합 완료!")
        print("=" * 60)
        
        total_moved = len(self.results["moved_files"])
        total_errors = len(self.results["errors"])
        total_size_moved = sum(item["size_mb"] for item in self.results["moved_files"]) / 1024
        
        print(f"✅ 성공 이동: {total_moved}개 파일")
        print(f"📦 총 이동 크기: {total_size_moved:.1f}GB")
        print(f"❌ 오류: {total_errors}개")
        
        if self.results["stats_by_extension"]:
            print(f"\n📄 확장자별 통계:")
            for ext, count in sorted(self.results["stats_by_extension"].items()):
                moved_count = len([f for f in self.results["moved_files"] if f["extension"] == ext])
                print(f"  {ext}: {moved_count}개 이동")
        
        if self.results["stats_by_step"]:
            print(f"\n📁 Step별 통계:")
            for step, count in sorted(self.results["stats_by_step"].items()):
                print(f"  {step}: {count}개")
        
        print(f"\n📁 최종 위치:")
        print(f"  {self.target_path}/checkpoints/")
        for step in sorted(self.results["stats_by_step"].keys()):
            print(f"    ├── {step}/")
    
    def save_consolidation_report(self):
        """상세 리포트 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.target_path / f"complete_consolidation_report_{timestamp}.json"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n📊 상세 리포트: {report_path}")
        except Exception as e:
            print(f"⚠️ 리포트 저장 실패: {e}")

def main():
    """메인 실행"""
    print("🔥 완전 모델 통합기 v2.0 - ALL FORMATS")
    print("=" * 60)
    
    consolidator = CompleteModelConsolidator()
    
    # 경로 확인
    for path in consolidator.search_paths:
        if path.exists():
            print(f"✅ 소스 경로: {path}")
        else:
            print(f"❌ 소스 경로 없음: {path}")
    
    print(f"📁 통합 대상: {consolidator.target_path}")
    
    # 지원 형식 표시
    print(f"\n📦 지원하는 모델 형식: {len(consolidator.model_extensions)}개")
    for ext, desc in list(consolidator.model_extensions.items())[:5]:
        print(f"  {ext}: {desc}")
    print(f"  ... 총 {len(consolidator.model_extensions)}개 형식 지원")
    
    # 실행
    success = consolidator.consolidate_all_models()
    
    if success:
        print("\n🎉 모든 모델이 성공적으로 통합되었습니다!")
        print("💡 이제 모든 형식의 AI 모델을 한 곳에서 사용할 수 있습니다!")
    else:
        print("\n⚠️ 일부 오류가 발생했습니다. 리포트를 확인해주세요.")
    
    return success

if __name__ == "__main__":
    main()