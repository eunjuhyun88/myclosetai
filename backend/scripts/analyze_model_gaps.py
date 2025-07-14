
#!/usr/bin/env python3
"""
🔍 8단계 파이프라인 모델 갭 분석
현재 보유 vs 필요 모델 분석 및 다운로드 우선순위 제안
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelRequirement:
    """모델 요구사항"""
    step: str
    step_name: str
    required_models: List[str]
    alternative_models: List[str]
    current_status: str  # "available", "missing", "partial"
    priority: int  # 1=필수, 2=중요, 3=권장
    size_estimate_gb: float

class PipelineModelGapAnalyzer:
    """8단계 파이프라인 모델 갭 분석기"""
    
    def __init__(self):
        self.pipeline_requirements = self._define_pipeline_requirements()
        self.current_models = self._load_current_models()
        
    def _define_pipeline_requirements(self) -> Dict[str, ModelRequirement]:
        """8단계 파이프라인 모델 요구사항 정의"""
        return {
            "step_01": ModelRequirement(
                step="step_01_human_parsing",
                step_name="인체 파싱 (Human Parsing)",
                required_models=[
                    "Graphonomy ATR weights",
                    "Graphonomy LIP weights", 
                    "Human parsing model weights"
                ],
                alternative_models=[
                    "Self-Correction-Human-Parsing",
                    "MediaPipe Selfie Segmentation",
                    "DeepLabV3+ Human"
                ],
                current_status="partial",
                priority=1,
                size_estimate_gb=0.5
            ),
            
            "step_02": ModelRequirement(
                step="step_02_pose_estimation", 
                step_name="포즈 추정 (Pose Estimation)",
                required_models=[
                    "OpenPose Body Model",
                    "OpenPose pose weights"
                ],
                alternative_models=[
                    "MediaPipe Pose",
                    "PoseNet",
                    "DensePose"
                ],
                current_status="available",
                priority=1,
                size_estimate_gb=0.2
            ),
            
            "step_03": ModelRequirement(
                step="step_03_cloth_segmentation",
                step_name="의류 세그멘테이션 (Cloth Segmentation)", 
                required_models=[
                    "U²-Net pretrained weights",
                    "U²-Net human segmentation", 
                    "Cloth-specific segmentation model"
                ],
                alternative_models=[
                    "RemBG",
                    "SAM (Segment Anything)",
                    "DeepLabV3+ clothing"
                ],
                current_status="missing",
                priority=1,
                size_estimate_gb=0.3
            ),
            
            "step_04": ModelRequirement(
                step="step_04_geometric_matching",
                step_name="기하학적 매칭 (Geometric Matching)",
                required_models=[
                    "HR-VITON GMM weights",
                    "Geometric Matching Module",
                    "TPS transformation model"
                ],
                alternative_models=[
                    "VITON GMM",
                    "CP-VTON matching",
                    "Custom geometric matching"
                ],
                current_status="missing",
                priority=2,
                size_estimate_gb=0.1
            ),
            
            "step_05": ModelRequirement(
                step="step_05_cloth_warping",
                step_name="의류 워핑 (Cloth Warping)",
                required_models=[
                    "HR-VITON TOM weights",
                    "Try-On Module weights",
                    "Cloth warping model"
                ],
                alternative_models=[
                    "VITON TOM",
                    "CP-VTON warping",
                    "PF-AFN warping"
                ],
                current_status="missing",
                priority=2,
                size_estimate_gb=0.2
            ),
            
            "step_06": ModelRequirement(
                step="step_06_virtual_fitting",
                step_name="가상 피팅 생성 (Virtual Fitting)",
                required_models=[
                    "Stable Diffusion v1.5",
                    "OOTDiffusion weights",
                    "ControlNet pose"
                ],
                alternative_models=[
                    "SDXL",
                    "HR-VITON full",
                    "ACGPN"
                ],
                current_status="available",
                priority=1,
                size_estimate_gb=10.0
            ),
            
            "step_07": ModelRequirement(
                step="step_07_post_processing",
                step_name="후처리 (Post Processing)",
                required_models=[
                    "Real-ESRGAN weights",
                    "Face enhancement model",
                    "Upscaling model"
                ],
                alternative_models=[
                    "GFPGAN",
                    "CodeFormer",
                    "SwinIR"
                ],
                current_status="missing",
                priority=3,
                size_estimate_gb=0.1
            ),
            
            "step_08": ModelRequirement(
                step="step_08_quality_assessment",
                step_name="품질 평가 (Quality Assessment)",
                required_models=[
                    "LPIPS model",
                    "FID calculation model",
                    "Quality metrics"
                ],
                alternative_models=[
                    "CLIP quality assessment",
                    "Custom metrics",
                    "SSIM/PSNR only"
                ],
                current_status="missing",
                priority=3,
                size_estimate_gb=0.1
            )
        }
    
    def _load_current_models(self) -> Dict[str, Any]:
        """현재 보유 모델 정보 로드"""
        try:
            from app.core.optimized_model_paths import ANALYZED_MODELS
            return ANALYZED_MODELS
        except ImportError:
            logger.warning("optimized_model_paths를 로드할 수 없음")
            return {}
    
    def analyze_gaps(self) -> Dict[str, Any]:
        """모델 갭 분석"""
        logger.info("🔍 8단계 파이프라인 모델 갭 분석 시작...")
        
        analysis_result = {
            "pipeline_completeness": {},
            "missing_models": [],
            "download_priorities": [],
            "current_coverage": {},
            "recommendations": []
        }
        
        total_steps = len(self.pipeline_requirements)
        ready_steps = 0
        
        for step_key, requirement in self.pipeline_requirements.items():
            step_analysis = self._analyze_step_requirement(requirement)
            analysis_result["pipeline_completeness"][step_key] = step_analysis
            
            if step_analysis["status"] == "ready":
                ready_steps += 1
            elif step_analysis["status"] == "missing":
                analysis_result["missing_models"].extend(step_analysis["missing_models"])
        
        # 전체 완성도 계산
        completeness = ready_steps / total_steps
        analysis_result["overall_completeness"] = completeness
        
        # 다운로드 우선순위 생성
        analysis_result["download_priorities"] = self._generate_download_priorities()
        
        # 권장사항 생성
        analysis_result["recommendations"] = self._generate_recommendations(completeness)
        
        self._display_analysis_results(analysis_result)
        
        return analysis_result
    
    def _analyze_step_requirement(self, requirement: ModelRequirement) -> Dict[str, Any]:
        """개별 단계 요구사항 분석"""
        step_analysis = {
            "step": requirement.step,
            "step_name": requirement.step_name,
            "status": "missing",
            "available_models": [],
            "missing_models": [],
            "alternatives_available": [],
            "priority": requirement.priority
        }
        
        # 현재 모델과 매칭
        available_count = 0
        total_required = len(requirement.required_models)
        
        for required_model in requirement.required_models:
            found = False
            for current_model_key, current_model_info in self.current_models.items():
                if self._is_model_match(required_model, current_model_key, current_model_info):
                    step_analysis["available_models"].append(current_model_key)
                    available_count += 1
                    found = True
                    break
            
            if not found:
                step_analysis["missing_models"].append(required_model)
        
        # 대체 모델 확인
        for alt_model in requirement.alternative_models:
            for current_model_key, current_model_info in self.current_models.items():
                if self._is_model_match(alt_model, current_model_key, current_model_info):
                    step_analysis["alternatives_available"].append(current_model_key)
        
        # 상태 결정
        if available_count >= total_required:
            step_analysis["status"] = "ready"
        elif available_count > 0 or step_analysis["alternatives_available"]:
            step_analysis["status"] = "partial"
        else:
            step_analysis["status"] = "missing"
        
        return step_analysis
    
    def _is_model_match(self, required_model: str, current_key: str, current_info: Dict) -> bool:
        """모델 매칭 확인"""
        required_lower = required_model.lower()
        current_key_lower = current_key.lower()
        current_name_lower = current_info.get("name", "").lower()
        
        # 키워드 매칭
        keywords_map = {
            "stable diffusion": ["stable", "diffusion"],
            "ootdiffusion": ["ootd", "diffusion"],
            "graphonomy": ["graphonomy"],
            "openpose": ["openpose", "pose"],
            "u2net": ["u2net", "background"],
            "human parsing": ["human", "parsing"],
            "gmm": ["gmm", "geometric"],
            "tom": ["tom", "try-on"],
            "esrgan": ["esrgan", "upscal"],
            "clip": ["clip", "vit"]
        }
        
        for pattern, keywords in keywords_map.items():
            if pattern in required_lower:
                if any(keyword in current_key_lower or keyword in current_name_lower 
                      for keyword in keywords):
                    return current_info.get("ready", False)
        
        return False
    
    def _generate_download_priorities(self) -> List[Dict[str, Any]]:
        """다운로드 우선순위 생성"""
        priorities = [
            # 우선순위 1 (필수)
            {
                "model": "U²-Net Human Segmentation",
                "step": "step_03_cloth_segmentation", 
                "reason": "의류 세그멘테이션 필수 - 파이프라인 핵심",
                "url": "https://drive.google.com/uc?id=1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ",
                "filename": "u2net.pth",
                "size_gb": 0.176,
                "priority": 1
            },
            {
                "model": "Graphonomy ATR weights",
                "step": "step_01_human_parsing",
                "reason": "인체 파싱 정확도 향상",
                "url": "https://drive.google.com/uc?id=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP",
                "filename": "graphonomy_atr.pth", 
                "size_gb": 0.178,
                "priority": 1
            },
            {
                "model": "Graphonomy LIP weights",
                "step": "step_01_human_parsing",
                "reason": "인체 파싱 대체 모델",
                "url": "https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH",
                "filename": "graphonomy_lip.pth",
                "size_gb": 0.178,
                "priority": 1
            },
            
            # 우선순위 2 (중요)
            {
                "model": "HR-VITON GMM weights",
                "step": "step_04_geometric_matching",
                "reason": "기하학적 매칭 정확도",
                "url": "https://drive.google.com/uc?id=1WJkwlCJXFWsEgdNGWSoXDhpqtNmwcaVY",
                "filename": "gmm_final.pth",
                "size_gb": 0.045,
                "priority": 2
            },
            {
                "model": "HR-VITON TOM weights", 
                "step": "step_05_cloth_warping",
                "reason": "의류 워핑 품질 향상",
                "url": "https://drive.google.com/uc?id=1YJU5kNNL8Y-CqaXq-hOjJlh2hZ3s2qY",
                "filename": "tom_final.pth",
                "size_gb": 0.089,
                "priority": 2
            },
            
            # 우선순위 3 (권장)
            {
                "model": "Real-ESRGAN",
                "step": "step_07_post_processing",
                "reason": "이미지 품질 향상",
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                "filename": "RealESRGAN_x4plus.pth",
                "size_gb": 0.067,
                "priority": 3
            }
        ]
        
        return sorted(priorities, key=lambda x: x["priority"])
    
    def _generate_recommendations(self, completeness: float) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        if completeness < 0.3:
            recommendations.append("🚨 파이프라인 완성도가 30% 미만입니다. 즉시 필수 모델 다운로드가 필요합니다.")
        elif completeness < 0.6:
            recommendations.append("⚠️ 파이프라인이 부분적으로 작동 가능합니다. 핵심 모델들을 우선 다운로드하세요.")
        else:
            recommendations.append("✅ 파이프라인이 대부분 준비되었습니다. 성능 향상 모델들을 추가하세요.")
        
        recommendations.extend([
            "🎯 1순위: U²-Net (의류 세그멘테이션) - 파이프라인 핵심 단계",
            "🎯 2순위: Graphonomy weights (인체 파싱) - 정확도 향상", 
            "💡 대체안: MediaPipe + RemBG로 빠른 프로토타입 구현 가능",
            "🔧 단계적 접근: 필수 모델(1-2GB) → 성능 모델(추가 1GB) → 고급 모델"
        ])
        
        return recommendations
    
    def _display_analysis_results(self, analysis: Dict[str, Any]):
        """분석 결과 출력"""
        logger.info("📊 8단계 파이프라인 모델 갭 분석 결과")
        logger.info("=" * 60)
        
        completeness = analysis["overall_completeness"]
        logger.info(f"🎯 전체 완성도: {completeness:.1%}")
        
        logger.info(f"\n📋 단계별 상태:")
        for step_key, step_analysis in analysis["pipeline_completeness"].items():
            status_emoji = {
                "ready": "✅",
                "partial": "⚠️", 
                "missing": "❌"
            }
            emoji = status_emoji.get(step_analysis["status"], "❓")
            logger.info(f"   {emoji} {step_analysis['step_name']}: {step_analysis['status']}")
            
            if step_analysis["available_models"]:
                logger.info(f"      보유: {', '.join(step_analysis['available_models'][:2])}")
            if step_analysis["missing_models"]:
                logger.info(f"      부족: {', '.join(step_analysis['missing_models'][:2])}")
        
        logger.info(f"\n🚀 다운로드 우선순위 (상위 5개):")
        for i, priority_item in enumerate(analysis["download_priorities"][:5], 1):
            logger.info(f"   {i}. {priority_item['model']} ({priority_item['size_gb']:.2f}GB)")
            logger.info(f"      → {priority_item['reason']}")
        
        logger.info(f"\n💡 권장사항:")
        for rec in analysis["recommendations"]:
            logger.info(f"   {rec}")
    
    def create_download_script(self, analysis: Dict[str, Any]):
        """부족한 모델 다운로드 스크립트 생성"""
        logger.info("📝 부족한 모델 다운로드 스크립트 생성 중...")
        
        script_content = '''#!/usr/bin/env python3
"""
🔥 MyCloset AI - 필수 모델 자동 다운로드
8단계 파이프라인 완성을 위한 우선순위 기반 다운로드
"""

import os
import sys
import gdown
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_priority_models():
    """우선순위 기반 필수 모델 다운로드"""
    
    print("🔥 MyCloset AI - 필수 모델 다운로드")
    print("=" * 50)
    
    base_dir = Path("ai_models/checkpoints")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 우선순위 모델들
    models = [
'''
        
        # 우선순위 모델들 추가
        for model in analysis["download_priorities"][:6]:  # 상위 6개
            script_content += f'''        {{
            "name": "{model['model']}",
            "step": "{model['step']}",
            "url": "{model['url']}", 
            "filename": "{model['filename']}",
            "size_gb": {model['size_gb']},
            "priority": {model['priority']},
            "reason": "{model['reason']}"
        }},
'''
        
        script_content += '''    ]
    
    total_size = sum(model["size_gb"] for model in models)
    logger.info(f"📦 다운로드 예정: {len(models)}개 모델 ({total_size:.2f}GB)")
    
    success_count = 0
    
    for i, model in enumerate(models, 1):
        logger.info(f"\\n[{i}/{len(models)}] {model['name']} 다운로드 중...")
        logger.info(f"   이유: {model['reason']}")
        logger.info(f"   크기: {model['size_gb']:.2f}GB")
        
        # 단계별 디렉토리 생성
        step_dir = base_dir / model["step"]
        step_dir.mkdir(exist_ok=True)
        output_path = step_dir / model["filename"]
        
        # 이미 존재하는지 확인
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 10:  # 10MB 이상이면 이미 다운로드된 것으로 간주
                logger.info(f"   ✅ 이미 존재함: {model['name']} ({file_size_mb:.1f}MB)")
                success_count += 1
                continue
        
        try:
            if "drive.google.com" in model["url"]:
                # Google Drive 다운로드
                success = gdown.download(model["url"], str(output_path), quiet=False)
                if success:
                    logger.info(f"   ✅ {model['name']} 다운로드 완료")
                    success_count += 1
                else:
                    logger.error(f"   ❌ {model['name']} 다운로드 실패")
            else:
                logger.info(f"   ⚠️ 수동 다운로드 필요: {model['url']}")
                logger.info(f"      다운로드 후 {output_path}에 저장하세요")
                
        except Exception as e:
            logger.error(f"   ❌ {model['name']} 다운로드 실패: {e}")
    
    logger.info(f"\\n🎉 다운로드 완료: {success_count}/{len(models)}개")
    
    if success_count >= len(models) * 0.8:  # 80% 이상 성공
        logger.info("✅ 필수 모델 다운로드 성공! 이제 파이프라인을 테스트할 수 있습니다.")
        logger.info("\\n🚀 다음 단계:")
        logger.info("   python scripts/analyze_checkpoints.py  # 모델 재스캔")
        logger.info("   python scripts/test_loaded_models.py   # 파이프라인 테스트")
    else:
        logger.warning(f"⚠️ 일부 모델 다운로드 실패. 수동 다운로드가 필요할 수 있습니다.")

if __name__ == "__main__":
    try:
        # gdown 설치 확인
        import gdown
    except ImportError:
        print("❌ gdown이 설치되지 않았습니다.")
        print("설치: pip install gdown")
        sys.exit(1)
    
    download_priority_models()
'''
        
        # 스크립트 파일 저장
        script_path = Path("scripts/download_missing_models.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 실행 권한 추가
        script_path.chmod(0o755)
        
        logger.info(f"✅ 다운로드 스크립트 생성: {script_path}")

def main():
    """메인 함수"""
    print("🔍 MyCloset AI - 8단계 파이프라인 모델 갭 분석")
    print("=" * 60)
    
    analyzer = PipelineModelGapAnalyzer()
    analysis_result = analyzer.analyze_gaps()
    
    # 다운로드 스크립트 생성
    analyzer.create_download_script(analysis_result)
    
    # 결과 저장
    result_path = Path("ai_models/pipeline_gap_analysis.json")
    result_path.parent.mkdir(exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 완성도: {analysis_result['overall_completeness']:.1%}")
    print(f"📁 분석 결과: {result_path}")
    print(f"📝 다운로드 스크립트: scripts/download_missing_models.py")
    
    total_download_size = sum(item["size_gb"] for item in analysis_result["download_priorities"][:5])
    print(f"💾 필수 다운로드 크기: {total_download_size:.2f}GB")
    
    print(f"\n🚀 다음 실행:")
    print(f"   python scripts/download_missing_models.py")
    
    return True

if __name__ == "__main__":
    main()
