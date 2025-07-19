#!/usr/bin/env python3
"""
🤖 HuggingFace 모델 완전 통합 스크립트 v1.0
외부 ai_models/huggingface_cache의 모든 모델을 
backend/app/ai_pipeline/models로 안전하게 이동
✅ 실제 모델 파일들 완전 복사
✅ Step별 자동 분류  
✅ 용량 체크 및 진행률 표시
✅ 안전한 복사 (원본 유지)
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import time

class HuggingFaceModelConsolidator:
    """HuggingFace 모델 완전 통합기"""
    
    def __init__(self):
        self.project_root = Path("/Users/gimdudeul/MVP/mycloset-ai")
        self.source_path = self.project_root / "ai_models" / "huggingface_cache"
        self.target_path = self.project_root / "backend/app/ai_pipeline/models"
        
        # Step별 HuggingFace 모델 매핑
        self.model_mappings = {
            "models--levihsu--OOTDiffusion": {
                "step": "step_06_virtual_fitting",
                "name": "ootdiffusion",
                "priority": "CRITICAL",
                "estimated_size_gb": 15.0,
                "description": "Out-Of-The-Box Diffusion for Virtual Try-On"
            },
            "models--yisol--IDM-VTON": {
                "step": "step_06_virtual_fitting", 
                "name": "idm_vton",
                "priority": "CRITICAL",
                "estimated_size_gb": 8.0,
                "description": "Improving Diffusion Models for Virtual Try-On"
            },
            "models--facebook--sam2-hiera-large": {
                "step": "step_03_cloth_segmentation",
                "name": "sam2_large",
                "priority": "HIGH",
                "estimated_size_gb": 2.0,
                "description": "Segment Anything Model 2 - Large"
            },
            "models--openai--clip-vit-large-patch14-336": {
                "step": "step_08_quality_assessment",
                "name": "clip_vit_large",
                "priority": "HIGH", 
                "estimated_size_gb": 1.5,
                "description": "CLIP Vision Transformer Large"
            },
            "models--patrickjohncyh--fashion-clip": {
                "step": "step_03_cloth_segmentation",
                "name": "fashion_clip",
                "priority": "MEDIUM",
                "estimated_size_gb": 1.0,
                "description": "Fashion-specific CLIP Model"
            },
            "models--stabilityai--stable-diffusion-xl-base-1.0": {
                "step": "step_06_virtual_fitting",
                "name": "stable_diffusion_xl",
                "priority": "HIGH",
                "estimated_size_gb": 5.0,
                "description": "Stable Diffusion XL Base Model"
            }
        }
        
        # 결과 추적
        self.results = {
            "moved_models": [],
            "errors": [],
            "total_size_moved": 0,
            "start_time": None,
            "end_time": None
        }
    
    def analyze_source_models(self) -> Dict[str, Dict]:
        """소스 모델 분석"""
        print("🔍 HuggingFace 모델 분석 중...")
        
        if not self.source_path.exists():
            print(f"❌ 소스 경로가 존재하지 않습니다: {self.source_path}")
            return {}
        
        found_models = {}
        total_estimated_size = 0
        
        for model_dir in self.source_path.iterdir():
            if model_dir.is_dir() and model_dir.name.startswith("models--"):
                model_info = self.model_mappings.get(model_dir.name)
                
                if model_info:
                    # 실제 크기 계산
                    actual_size = self.get_directory_size(model_dir)
                    actual_size_gb = actual_size / (1024**3)
                    
                    found_models[model_dir.name] = {
                        **model_info,
                        "path": model_dir,
                        "actual_size_gb": actual_size_gb,
                        "actual_size_bytes": actual_size
                    }
                    total_estimated_size += actual_size_gb
                    
                    print(f"✅ {model_info['name']}: {actual_size_gb:.1f}GB ({model_info['priority']})")
                else:
                    print(f"❓ 알 수 없는 모델: {model_dir.name}")
        
        print(f"\n📊 총 {len(found_models)}개 모델, 예상 크기: {total_estimated_size:.1f}GB")
        return found_models
    
    def get_directory_size(self, directory: Path) -> int:
        """디렉토리 크기 계산"""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            print(f"⚠️ 크기 계산 실패 {directory.name}: {e}")
        return total_size
    
    def check_disk_space(self, required_gb: float) -> bool:
        """디스크 공간 체크"""
        try:
            stat = os.statvfs(self.target_path)
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            
            print(f"💾 사용 가능 공간: {available_gb:.1f}GB")
            print(f"📦 필요한 공간: {required_gb:.1f}GB")
            
            if available_gb > required_gb * 1.2:  # 20% 여유 공간
                print("✅ 충분한 디스크 공간")
                return True
            else:
                print("❌ 디스크 공간 부족")
                return False
        except Exception as e:
            print(f"⚠️ 디스크 공간 체크 실패: {e}")
            return True  # 에러시 진행
    
    def copy_model_with_progress(self, source: Path, target: Path, model_name: str, size_gb: float) -> bool:
        """진행률 표시와 함께 모델 복사"""
        try:
            target.mkdir(parents=True, exist_ok=True)
            
            print(f"\n🚚 {model_name} 복사 중... ({size_gb:.1f}GB)")
            start_time = time.time()
            
            # 파일별 복사 (진행률 표시)
            total_files = sum(1 for _ in source.rglob("*") if _.is_file())
            copied_files = 0
            
            for file_path in source.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(source)
                    target_file = target / relative_path
                    
                    # 디렉토리 생성
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 파일 복사
                    shutil.copy2(file_path, target_file)
                    
                    copied_files += 1
                    progress = (copied_files / total_files) * 100
                    
                    # 진행률 표시 (10% 단위)
                    if copied_files % max(1, total_files // 10) == 0:
                        elapsed = time.time() - start_time
                        print(f"  📊 {progress:.0f}% 완료 ({copied_files}/{total_files} 파일, {elapsed:.1f}초)")
            
            elapsed = time.time() - start_time
            print(f"  ✅ 완료! ({elapsed:.1f}초)")
            
            return True
            
        except Exception as e:
            print(f"  ❌ 복사 실패: {e}")
            return False
    
    def consolidate_huggingface_models(self) -> bool:
        """메인 통합 프로세스"""
        print("🤖 HuggingFace 모델 완전 통합 시작!")
        print("=" * 60)
        
        self.results["start_time"] = datetime.now()
        
        # 1. 소스 모델 분석
        found_models = self.analyze_source_models()
        
        if not found_models:
            print("❌ 이동할 HuggingFace 모델이 없습니다.")
            return False
        
        # 2. 디스크 공간 체크
        total_size_gb = sum(model["actual_size_gb"] for model in found_models.values())
        
        if not self.check_disk_space(total_size_gb):
            print("❌ 디스크 공간 부족으로 중단됩니다.")
            return False
        
        # 3. 사용자 확인
        print(f"\n📋 이동할 모델들:")
        for model_dir, info in found_models.items():
            print(f"  🤖 {info['name']}: {info['actual_size_gb']:.1f}GB → {info['step']}/")
        
        response = input(f"\n🚚 총 {total_size_gb:.1f}GB의 모델을 이동하시겠습니까? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ 작업이 취소되었습니다.")
            return False
        
        # 4. 모델별 이동 실행
        success_count = 0
        
        for model_dir, info in found_models.items():
            try:
                # 대상 경로 설정
                target_model_dir = self.target_path / "downloads" / info["name"]
                
                # 모델 복사
                if self.copy_model_with_progress(
                    info["path"], 
                    target_model_dir, 
                    info["name"], 
                    info["actual_size_gb"]
                ):
                    success_count += 1
                    self.results["moved_models"].append({
                        "name": info["name"],
                        "step": info["step"], 
                        "size_gb": info["actual_size_gb"],
                        "source": str(info["path"]),
                        "target": str(target_model_dir)
                    })
                    self.results["total_size_moved"] += info["actual_size_gb"]
                else:
                    self.results["errors"].append({
                        "model": info["name"],
                        "error": "복사 실패"
                    })
                    
            except Exception as e:
                print(f"❌ {info['name']} 처리 실패: {e}")
                self.results["errors"].append({
                    "model": info["name"],
                    "error": str(e)
                })
        
        # 5. 결과 요약
        self.results["end_time"] = datetime.now()
        self.print_final_summary(success_count, len(found_models))
        self.save_consolidation_report()
        
        return success_count == len(found_models)
    
    def print_final_summary(self, success_count: int, total_count: int):
        """최종 결과 요약"""
        total_time = (self.results["end_time"] - self.results["start_time"]).total_seconds()
        
        print("\n" + "=" * 60)
        print("🎉 HuggingFace 모델 통합 완료!")
        print("=" * 60)
        print(f"✅ 성공: {success_count}/{total_count}개 모델")
        print(f"📦 이동된 용량: {self.results['total_size_moved']:.1f}GB")
        print(f"⏱️ 총 소요 시간: {total_time:.1f}초")
        print(f"❌ 실패: {len(self.results['errors'])}개")
        
        if self.results["moved_models"]:
            print(f"\n📁 이동된 모델들:")
            for model in self.results["moved_models"]:
                print(f"  ✅ {model['name']}: {model['size_gb']:.1f}GB → {model['step']}/")
        
        print(f"\n📁 최종 위치:")
        print(f"  {self.target_path}/downloads/")
        for model in self.results["moved_models"]:
            print(f"    ├── {model['name']}/")
    
    def save_consolidation_report(self):
        """통합 리포트 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.target_path / f"huggingface_consolidation_report_{timestamp}.json"
        
        # 시간 정보를 문자열로 변환
        report_data = {
            **self.results,
            "start_time": self.results["start_time"].isoformat(),
            "end_time": self.results["end_time"].isoformat()
        }
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print(f"\n📊 상세 리포트: {report_path}")
        except Exception as e:
            print(f"⚠️ 리포트 저장 실패: {e}")

def main():
    """메인 실행"""
    print("🤖 HuggingFace 모델 완전 통합기 v1.0")
    print("=" * 60)
    
    consolidator = HuggingFaceModelConsolidator()
    
    # 경로 확인
    if consolidator.source_path.exists():
        print(f"✅ 소스 경로: {consolidator.source_path}")
    else:
        print(f"❌ 소스 경로 없음: {consolidator.source_path}")
        return False
    
    print(f"📁 통합 대상: {consolidator.target_path}")
    
    # 실행
    success = consolidator.consolidate_huggingface_models()
    
    if success:
        print("\n🎉 모든 HuggingFace 모델이 성공적으로 통합되었습니다!")
        print("💡 이제 파이프라인에서 모든 모델을 직접 사용할 수 있습니다!")
    else:
        print("\n⚠️ 일부 오류가 발생했습니다. 리포트를 확인해주세요.")
    
    return success

if __name__ == "__main__":
    main()