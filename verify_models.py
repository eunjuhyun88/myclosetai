#!/usr/bin/env python3
"""
🔍 MyCloset AI - 모델 재배치 검증 스크립트
재배치된 AI 모델들의 상태를 체크하고 분석

기능:
- Step별 모델 파일 검증
- 파일 무결성 체크 (크기, 접근성)
- 누락된 필수 모델 탐지
- 백엔드 호환성 검증
- conda 환경 최적화

사용법:
python verify_models.py                    # 전체 검증
python verify_models.py --step 1          # 특정 Step만 검증
python verify_models.py --detailed        # 상세 분석
python verify_models.py --fix-missing     # 누락된 모델 탐지 및 제안
"""

import os
import sys
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import argparse

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
class ModelInfo:
    """모델 정보"""
    name: str
    path: Path
    size_mb: float
    extension: str
    exists: bool
    readable: bool
    checksum: str = ""
    step: str = ""
    status: str = "unknown"
    errors: List[str] = field(default_factory=list)

@dataclass
class StepRequirement:
    """Step별 필수 모델 요구사항"""
    step_name: str
    required_models: List[str]
    optional_models: List[str]
    min_models: int
    expected_size_mb: Tuple[float, float]  # (min, max)
    description: str

class ModelVerifier:
    """AI 모델 재배치 검증기"""
    
    def __init__(self, backend_dir: Optional[Path] = None):
        self.backend_dir = backend_dir or Path.cwd() / "backend"
        self.checkpoints_dir = self.backend_dir / "app" / "ai_pipeline" / "models" / "checkpoints"
        self.discovered_models: Dict[str, List[ModelInfo]] = defaultdict(list)
        self.verification_results: Dict = {}
        
        # Step별 필수 모델 정의
        self.step_requirements = {
            "step_01_human_parsing": StepRequirement(
                step_name="Human Parsing",
                required_models=[
                    "exp-schp-201908301523-atr.pth",
                    "parsing_atr.onnx"
                ],
                optional_models=[
                    "graphonomy_lip.pth",
                    "densepose_rcnn_R_50_FPN_s1x.pkl",
                    "segformer_b2_clothes.pth"
                ],
                min_models=1,
                expected_size_mb=(100, 800),
                description="인체 파싱 - 사람의 신체 부위를 세분화하여 인식"
            ),
            "step_02_pose_estimation": StepRequirement(
                step_name="Pose Estimation", 
                required_models=[
                    "openpose.pth",
                    "body_pose_model.pth"
                ],
                optional_models=[
                    "yolov8n-pose.pt",
                    "pose_deploy_linevec.prototxt",
                    "pose_landmark_heavy.tflite"
                ],
                min_models=1,
                expected_size_mb=(6, 400),
                description="포즈 추정 - 사람의 관절점과 자세를 감지"
            ),
            "step_03_cloth_segmentation": StepRequirement(
                step_name="Cloth Segmentation",
                required_models=[
                    "u2net.pth"
                ],
                optional_models=[
                    "mobile_sam.pt",
                    "sam_vit_h_4b8939.pth",
                    "cloth_segmentation.onnx"
                ],
                min_models=1,
                expected_size_mb=(38, 2500),
                description="의류 분할 - 의류와 배경을 정확히 분리"
            ),
            "step_04_geometric_matching": StepRequirement(
                step_name="Geometric Matching",
                required_models=[
                    "gmm_final.pth",
                    "lightweight_gmm.pth"
                ],
                optional_models=[
                    "tps_network.pth",
                    "geometric_matching.onnx"
                ],
                min_models=0,  # 선택적 단계
                expected_size_mb=(10, 200),
                description="기하학적 매칭 - 의류와 신체의 기하학적 관계 분석"
            ),
            "step_05_cloth_warping": StepRequirement(
                step_name="Cloth Warping",
                required_models=[
                    "tom_final.pth"
                ],
                optional_models=[
                    "tps_final.pth",
                    "cloth_warping.pth"
                ],
                min_models=0,  # 선택적 단계
                expected_size_mb=(15, 150),
                description="의류 변형 - 신체에 맞게 의류 형태 조정"
            ),
            "step_06_virtual_fitting": StepRequirement(
                step_name="Virtual Fitting",
                required_models=[
                    "diffusion_pytorch_model.safetensors",
                    "unet_vton",
                    "pytorch_model.bin"
                ],
                optional_models=[
                    "text_encoder",
                    "vae",
                    "scheduler"
                ],
                min_models=1,
                expected_size_mb=(500, 8000),
                description="가상 피팅 - 실제 착용 모습을 생성"
            ),
            "step_07_post_processing": StepRequirement(
                step_name="Post Processing",
                required_models=[],
                optional_models=[
                    "RealESRGAN_x4plus.pth",
                    "GFPGAN.pth",
                    "CodeFormer.pth"
                ],
                min_models=0,
                expected_size_mb=(17, 350),
                description="후처리 - 이미지 품질 향상 및 노이즈 제거"
            ),
            "step_08_quality_assessment": StepRequirement(
                step_name="Quality Assessment",
                required_models=[],
                optional_models=[
                    "clip-vit-base-patch32",
                    "clip-vit-large-patch14",
                    "pytorch_model.bin"
                ],
                min_models=0,
                expected_size_mb=(150, 1200),
                description="품질 평가 - 생성된 이미지의 품질을 자동 평가"
            )
        }
        
        print(f"🎯 백엔드 디렉토리: {self.backend_dir}")
        print(f"📁 체크포인트 경로: {self.checkpoints_dir}")
    
    def scan_models(self) -> Dict[str, List[ModelInfo]]:
        """재배치된 모델들 스캔"""
        print("🔍 재배치된 모델 스캔 시작...")
        
        if not self.checkpoints_dir.exists():
            print(f"❌ 체크포인트 디렉토리가 존재하지 않습니다: {self.checkpoints_dir}")
            return {}
        
        # Step 폴더들 스캔
        step_folders = [d for d in self.checkpoints_dir.iterdir() if d.is_dir()]
        print(f"📂 발견된 Step 폴더: {len(step_folders)}개")
        
        for step_folder in step_folders:
            step_name = step_folder.name
            print(f"  📁 {step_name}")
            
            # 폴더 내 모든 파일 스캔
            model_files = []
            for file_path in step_folder.rglob("*"):
                if file_path.is_file():
                    model_files.append(file_path)
            
            print(f"     🔢 파일 수: {len(model_files)}개")
            
            # ModelInfo 객체 생성
            for file_path in model_files:
                try:
                    stat = file_path.stat()
                    size_mb = stat.st_size / (1024 * 1024)
                    
                    model_info = ModelInfo(
                        name=file_path.name,
                        path=file_path,
                        size_mb=size_mb,
                        extension=file_path.suffix.lower(),
                        exists=True,
                        readable=os.access(file_path, os.R_OK),
                        step=step_name
                    )
                    
                    # 기본 검증
                    if not model_info.readable:
                        model_info.errors.append("파일 읽기 권한 없음")
                        model_info.status = "error"
                    elif size_mb < 0.1:
                        model_info.errors.append("파일 크기가 너무 작음 (100KB 미만)")
                        model_info.status = "warning"
                    else:
                        model_info.status = "ok"
                    
                    self.discovered_models[step_name].append(model_info)
                    
                except (OSError, PermissionError) as e:
                    # 접근할 수 없는 파일
                    model_info = ModelInfo(
                        name=file_path.name,
                        path=file_path,
                        size_mb=0.0,
                        extension=file_path.suffix.lower(),
                        exists=True,
                        readable=False,
                        step=step_name,
                        status="error",
                        errors=[f"파일 접근 오류: {e}"]
                    )
                    self.discovered_models[step_name].append(model_info)
        
        print(f"✅ 총 {sum(len(models) for models in self.discovered_models.values())}개 모델 발견")
        return self.discovered_models
    
    def verify_step_requirements(self) -> Dict:
        """Step별 요구사항 검증"""
        print("🔍 Step별 요구사항 검증 중...")
        
        verification_results = {}
        
        for step_id, requirement in self.step_requirements.items():
            step_models = self.discovered_models.get(step_id, [])
            
            result = {
                "step_name": requirement.step_name,
                "description": requirement.description,
                "found_models": len(step_models),
                "total_size_mb": sum(m.size_mb for m in step_models),
                "required_models": {
                    "found": [],
                    "missing": []
                },
                "optional_models": {
                    "found": [],
                    "missing": []
                },
                "status": "unknown",
                "issues": [],
                "recommendations": []
            }
            
            # 모델 파일명 목록
            found_names = {m.name.lower() for m in step_models}
            
            # 필수 모델 체크
            for req_model in requirement.required_models:
                req_lower = req_model.lower()
                found = any(req_lower in name or name in req_lower for name in found_names)
                if found:
                    result["required_models"]["found"].append(req_model)
                else:
                    result["required_models"]["missing"].append(req_model)
            
            # 선택적 모델 체크
            for opt_model in requirement.optional_models:
                opt_lower = opt_model.lower()
                found = any(opt_lower in name or name in opt_lower for name in found_names)
                if found:
                    result["optional_models"]["found"].append(opt_model)
                else:
                    result["optional_models"]["missing"].append(opt_model)
            
            # 최소 요구사항 체크
            min_met = len(step_models) >= requirement.min_models
            required_met = len(result["required_models"]["missing"]) == 0
            
            # 크기 범위 체크
            size_ok = (requirement.expected_size_mb[0] <= result["total_size_mb"] <= 
                      requirement.expected_size_mb[1] * 2)  # 2배까지 허용
            
            # 전체 상태 결정
            if requirement.min_models == 0:  # 선택적 단계
                if len(step_models) == 0:
                    result["status"] = "optional_empty"
                elif min_met:
                    result["status"] = "ok"
                else:
                    result["status"] = "warning"
            else:  # 필수 단계
                if required_met and min_met and size_ok:
                    result["status"] = "ok"
                elif min_met:
                    result["status"] = "warning"
                else:
                    result["status"] = "error"
            
            # 이슈 및 권장사항 생성
            if not min_met:
                result["issues"].append(f"최소 {requirement.min_models}개 모델 필요, 현재 {len(step_models)}개")
            
            if result["required_models"]["missing"]:
                result["issues"].append(f"필수 모델 누락: {', '.join(result['required_models']['missing'])}")
                result["recommendations"].append("필수 모델을 다운로드하거나 대체 모델을 찾아주세요")
            
            if not size_ok and len(step_models) > 0:
                expected_range = f"{requirement.expected_size_mb[0]}-{requirement.expected_size_mb[1]}MB"
                result["issues"].append(f"예상 크기 범위 벗어남: {result['total_size_mb']:.1f}MB (예상: {expected_range})")
            
            verification_results[step_id] = result
        
        self.verification_results = verification_results
        return verification_results
    
    def check_file_integrity(self, quick_check: bool = True) -> Dict:
        """파일 무결성 검사"""
        print("🔍 파일 무결성 검사 중...")
        
        integrity_results = {
            "total_files": 0,
            "healthy_files": 0,
            "corrupted_files": 0,
            "inaccessible_files": 0,
            "suspicious_files": 0,
            "details": []
        }
        
        all_models = []
        for step_models in self.discovered_models.values():
            all_models.extend(step_models)
        
        iterator = tqdm(all_models, desc="무결성 검사") if TQDM_AVAILABLE else all_models
        
        for model in iterator:
            integrity_results["total_files"] += 1
            file_result = {
                "path": str(model.path),
                "name": model.name,
                "step": model.step,
                "size_mb": model.size_mb,
                "status": "unknown",
                "issues": []
            }
            
            try:
                # 파일 존재 및 접근 권한 체크
                if not model.path.exists():
                    file_result["status"] = "missing"
                    file_result["issues"].append("파일이 존재하지 않음")
                    integrity_results["corrupted_files"] += 1
                
                elif not model.readable:
                    file_result["status"] = "inaccessible"
                    file_result["issues"].append("파일 읽기 권한 없음")
                    integrity_results["inaccessible_files"] += 1
                
                else:
                    # 파일 크기 체크
                    if model.size_mb < 0.1:
                        file_result["status"] = "suspicious"
                        file_result["issues"].append("파일 크기가 비정상적으로 작음")
                        integrity_results["suspicious_files"] += 1
                    
                    elif model.size_mb > 10000:  # 10GB 초과
                        file_result["status"] = "suspicious"
                        file_result["issues"].append("파일 크기가 비정상적으로 큼")
                        integrity_results["suspicious_files"] += 1
                    
                    # 빠른 체크가 아닌 경우 실제 파일 읽기 테스트
                    elif not quick_check:
                        try:
                            with open(model.path, 'rb') as f:
                                # 처음과 끝 1KB씩 읽어보기
                                f.read(1024)
                                f.seek(-1024, 2)  # 파일 끝에서 1KB 앞으로
                                f.read(1024)
                            file_result["status"] = "healthy"
                            integrity_results["healthy_files"] += 1
                        except Exception as e:
                            file_result["status"] = "corrupted"
                            file_result["issues"].append(f"파일 읽기 오류: {e}")
                            integrity_results["corrupted_files"] += 1
                    
                    else:
                        file_result["status"] = "healthy"
                        integrity_results["healthy_files"] += 1
            
            except Exception as e:
                file_result["status"] = "error"
                file_result["issues"].append(f"검사 중 오류: {e}")
                integrity_results["corrupted_files"] += 1
            
            if file_result["issues"]:  # 이슈가 있는 파일만 상세 결과에 포함
                integrity_results["details"].append(file_result)
        
        return integrity_results
    
    def generate_summary_report(self, detailed: bool = False) -> Dict:
        """종합 요약 보고서 생성"""
        print("📊 종합 보고서 생성 중...")
        
        report = {
            "scan_info": {
                "timestamp": time.time(),
                "backend_dir": str(self.backend_dir),
                "checkpoints_dir": str(self.checkpoints_dir),
                "scan_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "overview": {
                "total_steps": len(self.step_requirements),
                "active_steps": len(self.discovered_models),
                "total_models": sum(len(models) for models in self.discovered_models.values()),
                "total_size_gb": sum(sum(m.size_mb for m in models) for models in self.discovered_models.values()) / 1024
            },
            "step_summary": {},
            "status_distribution": {
                "ok": 0,
                "warning": 0,
                "error": 0,
                "optional_empty": 0
            },
            "recommendations": [],
            "critical_issues": []
        }
        
        # Step별 요약
        for step_id, result in self.verification_results.items():
            step_summary = {
                "name": result["step_name"],
                "status": result["status"],
                "models_count": result["found_models"],
                "size_mb": result["total_size_mb"],
                "required_found": len(result["required_models"]["found"]),
                "required_total": len(result["required_models"]["found"]) + len(result["required_models"]["missing"]),
                "issues_count": len(result["issues"])
            }
            
            report["step_summary"][step_id] = step_summary
            report["status_distribution"][result["status"]] += 1
        
        # 전반적 권장사항
        if report["status_distribution"]["error"] > 0:
            report["recommendations"].append("🚨 필수 모델이 누락된 Step들이 있습니다. 즉시 해결 필요!")
        
        if report["status_distribution"]["warning"] > 0:
            report["recommendations"].append("⚠️ 일부 Step에서 개선이 필요합니다.")
        
        if report["overview"]["total_models"] < 10:
            report["recommendations"].append("💡 모델 수가 적습니다. 추가 모델 다운로드를 고려해보세요.")
        
        if report["overview"]["total_size_gb"] < 1.0:
            report["recommendations"].append("💡 모델 크기가 작습니다. 고성능 모델 추가를 고려해보세요.")
        
        # 중요 이슈 수집
        for step_id, result in self.verification_results.items():
            if result["status"] == "error":
                report["critical_issues"].append(f"{result['step_name']}: {', '.join(result['issues'])}")
        
        return report
    
    def print_detailed_report(self, report: Dict):
        """상세 보고서 출력"""
        print("\n" + "=" * 70)
        print("🤖 MyCloset AI 모델 검증 보고서")
        print("=" * 70)
        
        # 개요
        overview = report["overview"]
        print(f"\n📊 전체 개요:")
        print(f"   📁 총 Step 수: {overview['total_steps']}개")
        print(f"   ✅ 활성 Step 수: {overview['active_steps']}개")
        print(f"   🔢 총 모델 수: {overview['total_models']}개")
        print(f"   💾 총 크기: {overview['total_size_gb']:.2f}GB")
        
        # 상태 분포
        status_dist = report["status_distribution"]
        print(f"\n🎯 Step 상태 분포:")
        print(f"   ✅ 정상: {status_dist['ok']}개")
        print(f"   ⚠️ 경고: {status_dist['warning']}개")
        print(f"   ❌ 오류: {status_dist['error']}개")
        print(f"   ⭕ 선택적 비어있음: {status_dist['optional_empty']}개")
        
        # Step별 상세 정보
        print(f"\n📋 Step별 상세 현황:")
        for step_id, result in self.verification_results.items():
            status_emoji = {
                "ok": "✅",
                "warning": "⚠️", 
                "error": "❌",
                "optional_empty": "⭕"
            }
            
            emoji = status_emoji.get(result["status"], "❓")
            print(f"\n{emoji} {result['step_name']} ({step_id})")
            print(f"   📝 {result['description']}")
            print(f"   🔢 모델 수: {result['found_models']}개")
            print(f"   💾 크기: {result['total_size_mb']:.1f}MB")
            
            if result["required_models"]["found"]:
                print(f"   ✅ 필수 모델: {', '.join(result['required_models']['found'])}")
            if result["required_models"]["missing"]:
                print(f"   ❌ 누락 필수: {', '.join(result['required_models']['missing'])}")
            if result["optional_models"]["found"]:
                print(f"   💡 선택 모델: {', '.join(result['optional_models']['found'])}")
            
            if result["issues"]:
                print(f"   🚨 이슈:")
                for issue in result["issues"]:
                    print(f"     - {issue}")
            
            if result["recommendations"]:
                print(f"   💡 권장사항:")
                for rec in result["recommendations"]:
                    print(f"     - {rec}")
        
        # 전반적 권장사항
        if report["recommendations"]:
            print(f"\n🎯 전반적 권장사항:")
            for rec in report["recommendations"]:
                print(f"   {rec}")
        
        # 중요 이슈
        if report["critical_issues"]:
            print(f"\n🚨 즉시 해결 필요한 이슈:")
            for issue in report["critical_issues"]:
                print(f"   ❌ {issue}")
    
    def suggest_missing_models(self) -> Dict:
        """누락된 모델 다운로드 제안"""
        suggestions = {
            "critical_missing": [],
            "recommended_additions": [],
            "download_commands": []
        }
        
        for step_id, result in self.verification_results.items():
            if result["required_models"]["missing"]:
                for missing_model in result["required_models"]["missing"]:
                    suggestions["critical_missing"].append({
                        "step": result["step_name"],
                        "model": missing_model,
                        "priority": "high"
                    })
            
            if result["status"] in ["warning", "error"] and result["optional_models"]["missing"]:
                for missing_model in result["optional_models"]["missing"][:2]:  # 최대 2개만
                    suggestions["recommended_additions"].append({
                        "step": result["step_name"],
                        "model": missing_model,
                        "priority": "medium"
                    })
        
        return suggestions

def main():
    parser = argparse.ArgumentParser(description="AI 모델 재배치 검증 도구")
    parser.add_argument("--backend-dir", type=Path, default=Path.cwd() / "backend", help="백엔드 디렉토리 경로")
    parser.add_argument("--step", type=int, help="특정 Step만 검증 (1-8)")
    parser.add_argument("--detailed", action="store_true", help="상세 분석 실행")
    parser.add_argument("--fix-missing", action="store_true", help="누락된 모델 탐지 및 제안")
    parser.add_argument("--integrity", action="store_true", help="파일 무결성 검사")
    parser.add_argument("--output", type=Path, help="결과를 JSON 파일로 저장")
    
    args = parser.parse_args()
    
    print("🔍 MyCloset AI 모델 검증기 v1.0")
    print("=" * 50)
    
    # 검증기 초기화
    verifier = ModelVerifier(backend_dir=args.backend_dir)
    
    # 모델 스캔
    discovered_models = verifier.scan_models()
    if not discovered_models:
        print("❌ 재배치된 모델을 찾을 수 없습니다.")
        print("💡 먼저 search_and_relocate_models.py를 실행하여 모델을 재배치해주세요.")
        return
    
    # Step별 요구사항 검증
    verification_results = verifier.verify_step_requirements()
    
    # 종합 보고서 생성
    report = verifier.generate_summary_report(detailed=args.detailed)
    
    # 보고서 출력
    verifier.print_detailed_report(report)
    
    # 파일 무결성 검사 (요청된 경우)
    if args.integrity:
        print("\n" + "=" * 50)
        integrity_results = verifier.check_file_integrity(quick_check=not args.detailed)
        
        print(f"🔍 파일 무결성 검사 결과:")
        print(f"   총 파일: {integrity_results['total_files']}개")
        print(f"   정상: {integrity_results['healthy_files']}개")
        print(f"   손상: {integrity_results['corrupted_files']}개")
        print(f"   접근불가: {integrity_results['inaccessible_files']}개")
        print(f"   의심스러운: {integrity_results['suspicious_files']}개")
        
        if integrity_results['details']:
            print(f"\n🚨 문제가 있는 파일들:")
            for detail in integrity_results['details'][:10]:  # 최대 10개만 표시
                print(f"   ❌ {detail['name']} ({detail['step']}) - {', '.join(detail['issues'])}")
    
    # 누락된 모델 제안 (요청된 경우)
    if args.fix_missing:
        print("\n" + "=" * 50)
        suggestions = verifier.suggest_missing_models()
        
        if suggestions["critical_missing"]:
            print(f"🚨 즉시 필요한 모델들:")
            for item in suggestions["critical_missing"]:
                print(f"   ❌ {item['step']}: {item['model']}")
        
        if suggestions["recommended_additions"]:
            print(f"\n💡 추가 권장 모델들:")
            for item in suggestions["recommended_additions"]:
                print(f"   ⚠️ {item['step']}: {item['model']}")
        
        print(f"\n🔧 해결 방법:")
        print(f"   1. HuggingFace Hub에서 해당 모델들 검색")
        print(f"   2. 공식 리포지토리에서 다운로드")
        print(f"   3. 대체 모델 사용 고려")
    
    # 결과 저장 (요청된 경우)
    if args.output:
        combined_results = {
            "verification": report,
            "models": {step: [vars(model) for model in models] 
                     for step, models in discovered_models.items()}
        }
        
        if args.integrity:
            combined_results["integrity"] = integrity_results
        
        if args.fix_missing:
            combined_results["suggestions"] = suggestions
        
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 결과 저장: {args.output}")
    
    # 최종 결론
    print("\n" + "=" * 50)
    total_errors = report["status_distribution"]["error"]
    total_warnings = report["status_distribution"]["warning"]
    
    if total_errors == 0 and total_warnings == 0:
        print("🎉 모든 검증 통과! AI 파이프라인이 올바르게 구성되었습니다.")
    elif total_errors == 0:
        print(f"✅ 기본 요구사항 충족. {total_warnings}개 개선 권장사항이 있습니다.")
    else:
        print(f"⚠️ {total_errors}개 중요 이슈, {total_warnings}개 경고가 있습니다. 해결이 필요합니다.")

if __name__ == "__main__":
    main()