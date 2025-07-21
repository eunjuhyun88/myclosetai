# scripts/analyze_model_files.py
"""
🔥 MyCloset AI - 실제 모델 파일 분석 및 최적 매핑 생성
370GB 모델 파일들을 분석하여 각 Step에 최적 모델 매핑
"""

import os
import sys
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelFileAnalyzer:
    """AI 모델 파일 분석기"""
    
    def __init__(self, backend_dir: str = "backend"):
        self.backend_dir = Path(backend_dir)
        self.ai_models_dir = self.backend_dir / "ai_models"
        
        # Step별 키워드 매핑
        self.step_keywords = {
            "step_01_human_parsing": [
                "schp", "atr", "lip", "graphonomy", "parsing", "human", 
                "densepose", "segment", "body"
            ],
            "step_02_pose_estimation": [
                "pose", "openpose", "body_pose", "hand_pose", "joint",
                "keypoint", "skeleton", "coco", "mpii"
            ],
            "step_03_cloth_segmentation": [
                "u2net", "cloth", "segment", "sam", "mask", "background",
                "removal", "matting", "rmbg"
            ],
            "step_04_geometric_matching": [
                "geometric", "gmm", "matching", "tps", "transformation",
                "warping", "correspondence", "alignment"
            ],
            "step_05_cloth_warping": [
                "warp", "tps", "transformation", "deformation", "grid",
                "flow", "displacement", "thin_plate"
            ],
            "step_06_virtual_fitting": [
                "viton", "hrviton", "ootd", "diffusion", "fitting", "try_on",
                "virtual", "vae", "unet", "stable_diffusion"
            ],
            "step_07_post_processing": [
                "esrgan", "super_resolution", "enhance", "gfpgan", "codeformer",
                "real_esrgan", "swinir", "restoration", "upscale"
            ],
            "step_08_quality_assessment": [
                "lpips", "quality", "metric", "score", "assessment",
                "clip", "vgg", "resnet", "feature"
            ]
        }
        
        # 모델 우선순위 (파일명 패턴)
        self.priority_patterns = {
            "step_01_human_parsing": [
                "exp-schp-201908301523-atr.pth",  # 최고 우선순위
                "exp-schp-201908261155-lip.pth",
                "graphonomy.pth",
                "schp_atr.pth"
            ],
            "step_02_pose_estimation": [
                "body_pose_model.pth",
                "openpose.pth", 
                "openpose_05.pth",
                "hand_pose_model.pth"
            ],
            "step_03_cloth_segmentation": [
                "u2net.pth",
                "u2net_backup.pth",
                "sam_vit_h_4b8939.pth"
            ],
            "step_06_virtual_fitting": [
                "hrviton_final.pth",
                "diffusion_pytorch_model.bin",
                "ootd_hd_unet.bin",
                "vae_model.bin"
            ]
        }
        
    def get_file_size_mb(self, file_path: Path) -> float:
        """파일 크기를 MB로 반환"""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except:
            return 0.0
            
    def classify_model_file(self, file_path: Path) -> Tuple[str, float]:
        """파일을 적절한 Step으로 분류하고 신뢰도 반환"""
        filename = file_path.name.lower()
        parent_dir = file_path.parent.name.lower()
        
        step_scores = defaultdict(float)
        
        # 1. 디렉토리명으로 우선 분류
        if parent_dir.startswith("step_"):
            step_scores[parent_dir] += 10.0
            
        # 2. 키워드 매칭
        for step_name, keywords in self.step_keywords.items():
            for keyword in keywords:
                if keyword in filename or keyword in parent_dir:
                    step_scores[step_name] += 1.0
                    
        # 3. 우선순위 패턴 매칭
        for step_name, patterns in self.priority_patterns.items():
            for pattern in patterns:
                if pattern.lower() in filename:
                    step_scores[step_name] += 5.0
                    
        if not step_scores:
            return "unknown", 0.0
            
        best_step = max(step_scores.items(), key=lambda x: x[1])
        return best_step[0], best_step[1]
        
    def analyze_all_models(self) -> Dict:
        """모든 모델 파일 분석"""
        logger.info("🔍 AI 모델 파일 전체 분석 시작...")
        
        results = {
            "total_files": 0,
            "total_size_gb": 0.0,
            "by_step": defaultdict(list),
            "by_extension": defaultdict(int),
            "recommendations": {}
        }
        
        # 모든 AI 모델 파일 탐색
        for ext in ["*.pth", "*.bin", "*.pkl"]:
            for model_file in self.ai_models_dir.rglob(ext):
                if "cleanup_backup" in str(model_file):
                    continue  # 백업 파일 제외
                    
                size_mb = self.get_file_size_mb(model_file)
                step_name, confidence = self.classify_model_file(model_file)
                
                file_info = {
                    "path": str(model_file.relative_to(self.ai_models_dir)),
                    "name": model_file.name,
                    "size_mb": round(size_mb, 2),
                    "extension": model_file.suffix,
                    "confidence": round(confidence, 2),
                    "parent_dir": model_file.parent.name
                }
                
                results["by_step"][step_name].append(file_info)
                results["by_extension"][model_file.suffix] += 1
                results["total_files"] += 1
                results["total_size_gb"] += size_mb / 1024
                
        results["total_size_gb"] = round(results["total_size_gb"], 2)
        
        # Step별 추천 모델 선정
        self.generate_recommendations(results)
        
        return results
        
    def generate_recommendations(self, results: Dict):
        """각 Step별 추천 모델 생성"""
        logger.info("🎯 Step별 최적 모델 추천 생성 중...")
        
        for step_name, files in results["by_step"].items():
            if step_name == "unknown" or not files:
                continue
                
            # 신뢰도와 크기로 정렬
            sorted_files = sorted(files, key=lambda x: (-x["confidence"], -x["size_mb"]))
            
            recommendations = {
                "primary": None,
                "alternatives": [],
                "total_files": len(files),
                "total_size_mb": sum(f["size_mb"] for f in files)
            }
            
            # 기본 모델 선정
            if sorted_files:
                recommendations["primary"] = sorted_files[0]
                recommendations["alternatives"] = sorted_files[1:min(4, len(sorted_files))]
                
            results["recommendations"][step_name] = recommendations
            
    def create_optimized_config(self, results: Dict) -> Dict:
        """최적화된 모델 설정 생성"""
        logger.info("⚙️ 최적화된 모델 설정 생성 중...")
        
        config = {
            "version": "1.0",
            "generated_by": "ModelFileAnalyzer",
            "total_models": results["total_files"],
            "total_size_gb": results["total_size_gb"],
            "step_configs": {}
        }
        
        for step_name, rec in results["recommendations"].items():
            if step_name == "unknown" or not rec["primary"]:
                continue
                
            step_config = {
                "enabled": True,
                "primary_model": {
                    "path": rec["primary"]["path"],
                    "name": rec["primary"]["name"],
                    "size_mb": rec["primary"]["size_mb"],
                    "confidence": rec["primary"]["confidence"]
                },
                "alternative_models": [
                    {
                        "path": alt["path"], 
                        "name": alt["name"],
                        "size_mb": alt["size_mb"]
                    } for alt in rec["alternatives"]
                ],
                "total_available": rec["total_files"]
            }
            
            config["step_configs"][step_name] = step_config
            
        return config
        
    def save_analysis_results(self, results: Dict, config: Dict):
        """분석 결과 저장"""
        logger.info("💾 분석 결과 저장 중...")
        
        # 결과 디렉토리 생성
        output_dir = self.backend_dir / "analysis_results"
        output_dir.mkdir(exist_ok=True)
        
        # 상세 분석 결과
        with open(output_dir / "model_analysis_detailed.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        # 최적화 설정
        with open(output_dir / "optimized_model_config.json", "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        # 요약 리포트 생성
        self.generate_summary_report(results, output_dir)
        
        logger.info(f"✅ 분석 결과 저장 완료: {output_dir}")
        
    def generate_summary_report(self, results: Dict, output_dir: Path):
        """요약 리포트 생성"""
        report_file = output_dir / "analysis_summary.md"
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# MyCloset AI 모델 분석 리포트\n\n")
            f.write(f"## 📊 전체 현황\n")
            f.write(f"- **총 모델 파일**: {results['total_files']}개\n")
            f.write(f"- **총 크기**: {results['total_size_gb']:.1f}GB\n")
            f.write(f"- **확장자별 분포**: {dict(results['by_extension'])}\n\n")
            
            f.write("## 🎯 Step별 추천 모델\n\n")
            for step_name, rec in results["recommendations"].items():
                if step_name == "unknown" or not rec["primary"]:
                    continue
                    
                f.write(f"### {step_name}\n")
                f.write(f"- **추천 모델**: {rec['primary']['name']}\n")
                f.write(f"- **크기**: {rec['primary']['size_mb']:.1f}MB\n")
                f.write(f"- **신뢰도**: {rec['primary']['confidence']:.1f}\n")
                f.write(f"- **전체 옵션**: {rec['total_files']}개\n\n")
                
        logger.info(f"📋 요약 리포트 생성: {report_file}")
        
    def run(self):
        """전체 분석 프로세스 실행"""
        logger.info("🚀 AI 모델 파일 분석 시작")
        
        if not self.ai_models_dir.exists():
            logger.error(f"❌ AI 모델 디렉토리 없음: {self.ai_models_dir}")
            return False
            
        # 1. 모든 모델 분석
        results = self.analyze_all_models()
        
        # 2. 최적화 설정 생성
        config = self.create_optimized_config(results)
        
        # 3. 결과 저장
        self.save_analysis_results(results, config)
        
        # 4. 요약 출력
        logger.info("🎉 분석 완료!")
        logger.info(f"📊 총 {results['total_files']}개 파일, {results['total_size_gb']:.1f}GB")
        
        for step_name, rec in results["recommendations"].items():
            if step_name != "unknown" and rec["primary"]:
                logger.info(f"🎯 {step_name}: {rec['primary']['name']} ({rec['primary']['size_mb']:.1f}MB)")
                
        return True

if __name__ == "__main__":
    analyzer = ModelFileAnalyzer()
    success = analyzer.run()
    sys.exit(0 if success else 1)