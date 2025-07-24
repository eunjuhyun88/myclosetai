#!/usr/bin/env python3
"""
🔍 체크포인트 파일 특화 탐지 및 검증 스크립트
✅ .pth, .pt 파일 전문 탐지
✅ 실제 PyTorch 체크포인트 내용 검증
✅ 누락된 모델 다운로드 가이드
✅ M3 Max 최적화
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

# PyTorch 체크포인트 검증용
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CheckpointInfo:
    """체크포인트 파일 정보"""
    file_path: Path
    file_size_mb: float
    model_type: str
    confidence: float
    pytorch_valid: bool
    contains_state_dict: bool
    parameter_count: int
    layers_info: Dict[str, Any]
    architecture_info: Dict[str, Any]
    checksum: str

class CheckpointFinder:
    """체크포인트 파일 전문 탐지기"""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.backend_dir = self.project_root
        self.checkpoints: Dict[str, List[CheckpointInfo]] = {}
        
        # 체크포인트 패턴 (더 구체적)
        self.checkpoint_patterns = {
            "human_parsing": {
                "keywords": ["human", "parsing", "atr", "schp", "graphonomy", "segmentation"],
                "expected_size_range": (50, 500),  # MB
                "required_layers": ["backbone", "classifier", "conv"]
            },
            "pose_estimation": {
                "keywords": ["pose", "openpose", "body", "keypoint", "coco"],
                "expected_size_range": (10, 1000),
                "required_layers": ["stage", "paf", "heatmap"]
            },
            "cloth_segmentation": {
                "keywords": ["u2net", "cloth", "segmentation", "mask"],
                "expected_size_range": (10, 200),
                "required_layers": ["encoder", "decoder", "side"]
            },
            "geometric_matching": {
                "keywords": ["gmm", "geometric", "matching", "tps"],
                "expected_size_range": (1, 100),
                "required_layers": ["extractor", "regression"]
            },
            "cloth_warping": {
                "keywords": ["tom", "warping", "viton", "try_on"],
                "expected_size_range": (10, 200),
                "required_layers": ["generator", "unet"]
            },
            "virtual_fitting": {
                "keywords": ["hrviton", "viton_hd", "hr_viton"],
                "expected_size_range": (100, 2000),
                "required_layers": ["unet", "vae", "text_encoder"]
            }
        }
        
        logger.info(f"🔍 체크포인트 전문 탐지기 초기화 - 루트: {self.project_root}")

    def find_all_checkpoints(self) -> Dict[str, List[CheckpointInfo]]:
        """모든 체크포인트 파일 탐지"""
        logger.info("🔍 체크포인트 파일 전체 탐지 시작...")
        
        # 검색 경로들
        search_paths = self._get_checkpoint_search_paths()
        
        all_checkpoint_files = []
        
        # .pth, .pt 파일들 수집
        for search_path in search_paths:
            if search_path.exists():
                logger.info(f"📁 체크포인트 검색 중: {search_path}")
                checkpoint_files = self._find_checkpoint_files(search_path)
                all_checkpoint_files.extend(checkpoint_files)
                logger.info(f"   └─ 발견: {len(checkpoint_files)}개 체크포인트")
        
        logger.info(f"📊 총 체크포인트 파일: {len(all_checkpoint_files)}개")
        
        # 각 파일 분석
        for checkpoint_file in all_checkpoint_files:
            try:
                checkpoint_info = self._analyze_checkpoint(checkpoint_file)
                if checkpoint_info:
                    model_type = checkpoint_info.model_type
                    if model_type not in self.checkpoints:
                        self.checkpoints[model_type] = []
                    self.checkpoints[model_type].append(checkpoint_info)
                    
            except Exception as e:
                logger.debug(f"체크포인트 분석 실패 {checkpoint_file}: {e}")
        
        self._print_checkpoint_summary()
        return self.checkpoints

    def _get_checkpoint_search_paths(self) -> List[Path]:
        """체크포인트 검색 경로"""
        paths = []
        
        # 프로젝트 내부
        project_paths = [
            self.backend_dir / "ai_models",
            self.backend_dir / "checkpoints", 
            self.backend_dir / "weights",
            self.backend_dir / "models",
            self.backend_dir / "pretrained",
            self.backend_dir / "app" / "ai_pipeline" / "models",
            self.backend_dir / ".." / "ai_models",
        ]
        
        # 사용자 캐시
        home = Path.home()
        cache_paths = [
            home / ".cache" / "torch" / "hub",
            home / ".cache" / "huggingface" / "hub",
            home / "Downloads",
            home / "Desktop",
            home / "Documents",
        ]
        
        # conda 환경
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_paths = [
                Path(conda_prefix) / "lib" / "python3.11" / "site-packages",
                Path(conda_prefix) / "share",
            ]
            cache_paths.extend(conda_paths)
        
        all_paths = project_paths + cache_paths
        return [p for p in all_paths if p.exists()]

    def _find_checkpoint_files(self, directory: Path, max_depth: int = 5) -> List[Path]:
        """디렉토리에서 체크포인트 파일 찾기"""
        checkpoint_files = []
        
        try:
            for item in directory.rglob("*"):
                if item.is_file() and item.suffix.lower() in ['.pth', '.pt']:
                    # 파일 크기 확인 (최소 1MB)
                    try:
                        size_mb = item.stat().st_size / (1024 * 1024)
                        if size_mb >= 1.0:
                            checkpoint_files.append(item)
                    except:
                        continue
        except Exception as e:
            logger.debug(f"디렉토리 스캔 오류 {directory}: {e}")
        
        return checkpoint_files

    def _analyze_checkpoint(self, checkpoint_file: Path) -> Optional[CheckpointInfo]:
        """체크포인트 파일 상세 분석"""
        try:
            file_size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
            
            # PyTorch 체크포인트 검증
            pytorch_valid = False
            contains_state_dict = False
            parameter_count = 0
            layers_info = {}
            architecture_info = {}
            
            if TORCH_AVAILABLE:
                try:
                    # 안전한 로딩
                    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=True)
                    pytorch_valid = True
                    
                    # state_dict 확인
                    if isinstance(checkpoint, dict):
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                            contains_state_dict = True
                        elif 'model' in checkpoint:
                            state_dict = checkpoint['model']
                            contains_state_dict = True
                        else:
                            # checkpoint 자체가 state_dict인 경우
                            state_dict = checkpoint
                            contains_state_dict = True
                        
                        # 파라미터 수 계산
                        if isinstance(state_dict, dict):
                            for key, tensor in state_dict.items():
                                if torch.is_tensor(tensor):
                                    parameter_count += tensor.numel()
                            
                            # 레이어 정보 추출
                            layers_info = self._extract_layer_info(state_dict)
                        
                        # 아키텍처 정보 추출
                        architecture_info = self._extract_architecture_info(checkpoint, checkpoint_file.name)
                    
                except Exception as e:
                    logger.debug(f"PyTorch 로딩 실패 {checkpoint_file}: {e}")
            
            # 모델 타입 분류
            model_type = self._classify_checkpoint(checkpoint_file, file_size_mb, layers_info)
            confidence = self._calculate_confidence(checkpoint_file, file_size_mb, layers_info, model_type)
            
            if confidence < 0.3:
                return None
            
            # 체크섬 계산
            checksum = self._calculate_checksum(checkpoint_file)
            
            return CheckpointInfo(
                file_path=checkpoint_file,
                file_size_mb=file_size_mb,
                model_type=model_type,
                confidence=confidence,
                pytorch_valid=pytorch_valid,
                contains_state_dict=contains_state_dict,
                parameter_count=parameter_count,
                layers_info=layers_info,
                architecture_info=architecture_info,
                checksum=checksum
            )
            
        except Exception as e:
            logger.debug(f"체크포인트 분석 실패 {checkpoint_file}: {e}")
            return None

    def _extract_layer_info(self, state_dict: Dict) -> Dict[str, Any]:
        """레이어 정보 추출"""
        layer_info = {
            "total_layers": len(state_dict),
            "layer_types": {},
            "key_patterns": set()
        }
        
        for key in state_dict.keys():
            # 레이어 타입 분류
            if 'conv' in key.lower():
                layer_info["layer_types"]["conv"] = layer_info["layer_types"].get("conv", 0) + 1
            elif 'linear' in key.lower() or 'fc' in key.lower():
                layer_info["layer_types"]["linear"] = layer_info["layer_types"].get("linear", 0) + 1
            elif 'bn' in key.lower() or 'batch' in key.lower():
                layer_info["layer_types"]["batch_norm"] = layer_info["layer_types"].get("batch_norm", 0) + 1
            elif 'attention' in key.lower() or 'attn' in key.lower():
                layer_info["layer_types"]["attention"] = layer_info["layer_types"].get("attention", 0) + 1
            
            # 키 패턴 추출
            key_parts = key.split('.')
            if len(key_parts) > 1:
                layer_info["key_patterns"].add(key_parts[0])
        
        layer_info["key_patterns"] = list(layer_info["key_patterns"])
        return layer_info

    def _extract_architecture_info(self, checkpoint: Dict, filename: str) -> Dict[str, Any]:
        """아키텍처 정보 추출"""
        arch_info = {
            "filename": filename,
            "metadata": {}
        }
        
        # 메타데이터 추출
        meta_keys = ['arch', 'model_name', 'version', 'epoch', 'config']
        for key in meta_keys:
            if key in checkpoint:
                arch_info["metadata"][key] = str(checkpoint[key])
        
        return arch_info

    def _classify_checkpoint(self, file_path: Path, file_size_mb: float, layers_info: Dict) -> str:
        """체크포인트 파일 분류"""
        file_name = file_path.name.lower()
        file_path_str = str(file_path).lower()
        
        best_type = "unknown"
        best_score = 0
        
        for model_type, pattern_info in self.checkpoint_patterns.items():
            score = 0
            
            # 키워드 매칭
            for keyword in pattern_info["keywords"]:
                if keyword in file_name:
                    score += 15
                elif keyword in file_path_str:
                    score += 8
            
            # 파일 크기 확인
            min_size, max_size = pattern_info["expected_size_range"]
            if min_size <= file_size_mb <= max_size:
                score += 10
            
            # 레이어 패턴 확인
            if layers_info and "key_patterns" in layers_info:
                required_layers = pattern_info.get("required_layers", [])
                for required_layer in required_layers:
                    for pattern in layers_info["key_patterns"]:
                        if required_layer in pattern.lower():
                            score += 5
                            break
            
            if score > best_score:
                best_score = score
                best_type = model_type
        
        return best_type

    def _calculate_confidence(self, file_path: Path, file_size_mb: float, layers_info: Dict, model_type: str) -> float:
        """신뢰도 계산"""
        if model_type == "unknown":
            return 0.0
        
        confidence = 0.3  # 기본 신뢰도
        
        # PyTorch 유효성
        if layers_info:
            confidence += 0.2
        
        # 파일 크기 적정성
        if model_type in self.checkpoint_patterns:
            min_size, max_size = self.checkpoint_patterns[model_type]["expected_size_range"]
            if min_size <= file_size_mb <= max_size:
                confidence += 0.3
        
        # 레이어 구조 적정성
        if layers_info and "layer_types" in layers_info:
            if len(layers_info["layer_types"]) > 1:
                confidence += 0.2
        
        return min(confidence, 1.0)

    def _calculate_checksum(self, file_path: Path) -> str:
        """체크섬 계산"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                # 처음 1MB만 읽어서 빠르게 체크섬 계산
                chunk = f.read(1024 * 1024)
                hash_md5.update(chunk)
            return hash_md5.hexdigest()[:12]
        except:
            return "unknown"

    def _print_checkpoint_summary(self):
        """체크포인트 탐지 결과 요약"""
        logger.info("=" * 70)
        logger.info("🎯 체크포인트 파일 탐지 결과")
        logger.info("=" * 70)
        
        total_checkpoints = sum(len(checkpoints) for checkpoints in self.checkpoints.values())
        logger.info(f"📊 총 유효 체크포인트: {total_checkpoints}개")
        
        for model_type, checkpoints in self.checkpoints.items():
            if checkpoints:
                logger.info(f"\n📁 {model_type}:")
                for i, checkpoint in enumerate(checkpoints[:3]):  # 상위 3개만 표시
                    logger.info(f"  {i+1}. {checkpoint.file_path.name}")
                    logger.info(f"     크기: {checkpoint.file_size_mb:.1f}MB")
                    logger.info(f"     신뢰도: {checkpoint.confidence:.2f}")
                    logger.info(f"     파라미터: {checkpoint.parameter_count:,}개")
                    logger.info(f"     유효: {'✅' if checkpoint.pytorch_valid else '❌'}")
                
                if len(checkpoints) > 3:
                    logger.info(f"  ... 외 {len(checkpoints) - 3}개")

    def generate_missing_models_report(self) -> Dict[str, Any]:
        """누락된 모델 리포트 생성"""
        required_models = [
            "human_parsing", "pose_estimation", "cloth_segmentation",
            "geometric_matching", "cloth_warping", "virtual_fitting"
        ]
        
        found_models = set(self.checkpoints.keys())
        missing_models = [model for model in required_models if model not in found_models]
        
        report = {
            "total_required": len(required_models),
            "found_count": len(found_models),
            "missing_count": len(missing_models),
            "found_models": list(found_models),
            "missing_models": missing_models,
            "download_suggestions": {}
        }
        
        # 다운로드 제안
        download_suggestions = {
            "human_parsing": {
                "name": "Graphonomy Human Parsing",
                "url": "https://github.com/Gaoyiminggithub/Graphonomy",
                "files": ["graphonomy.pth", "exp-schp-201908261155-pascal-person-part.pth"]
            },
            "pose_estimation": {
                "name": "OpenPose Body Pose",
                "url": "https://github.com/CMU-Perceptual-Computing-Lab/openpose",
                "files": ["body_pose_model.pth", "pose_model.pth"]
            },
            "cloth_segmentation": {
                "name": "U²-Net Cloth Segmentation", 
                "url": "https://github.com/xuebinqin/U-2-Net",
                "files": ["u2net.pth", "u2netp.pth"]
            },
            "geometric_matching": {
                "name": "CP-VTON Geometric Matching",
                "url": "https://github.com/sergeywong/cp-vton",
                "files": ["gmm_final.pth", "gmm_traintest_final.pth"]
            },
            "cloth_warping": {
                "name": "TOM (Try-On Module)",
                "url": "https://github.com/sergeywong/cp-vton", 
                "files": ["tom_final.pth", "tom_traintest_final.pth"]
            },
            "virtual_fitting": {
                "name": "HR-VITON",
                "url": "https://github.com/sangyun884/HR-VITON",
                "files": ["hrviton_final.pth", "hr_viton.pth"]
            }
        }
        
        for missing_model in missing_models:
            if missing_model in download_suggestions:
                report["download_suggestions"][missing_model] = download_suggestions[missing_model]
        
        return report

    def create_checkpoint_relocate_plan(self) -> Dict[str, Any]:
        """체크포인트 재배치 계획 생성"""
        plan = {
            "actions": [],
            "summary": {}
        }
        
        target_mapping = {
            "human_parsing": "ai_models/checkpoints/step_01_human_parsing/graphonomy.pth",
            "pose_estimation": "ai_models/checkpoints/step_02_pose_estimation/openpose.pth",
            "cloth_segmentation": "ai_models/checkpoints/step_03_cloth_segmentation/u2net.pth",
            "geometric_matching": "ai_models/checkpoints/gmm_final.pth", 
            "cloth_warping": "ai_models/checkpoints/tom_final.pth",
            "virtual_fitting": "ai_models/checkpoints/hrviton_final.pth"
        }
        
        for model_type, checkpoints in self.checkpoints.items():
            if checkpoints and model_type in target_mapping:
                # 최고 신뢰도 체크포인트 선택
                best_checkpoint = max(checkpoints, key=lambda c: c.confidence)
                
                plan["actions"].append({
                    "model_type": model_type,
                    "source": str(best_checkpoint.file_path),
                    "target": target_mapping[model_type],
                    "size_mb": best_checkpoint.file_size_mb,
                    "confidence": best_checkpoint.confidence,
                    "action": "symlink" if best_checkpoint.file_size_mb > 100 else "copy"
                })
        
        plan["summary"] = {
            "total_actions": len(plan["actions"]),
            "total_size_gb": sum(action["size_mb"] for action in plan["actions"]) / 1024
        }
        
        return plan

def main():
    """메인 실행 함수"""
    logger.info("=" * 70)
    logger.info("🔍 체크포인트 파일 특화 탐지 시작")
    logger.info("=" * 70)
    
    # 체크포인트 탐지
    finder = CheckpointFinder()
    checkpoints = finder.find_all_checkpoints()
    
    if not checkpoints:
        logger.error("❌ 체크포인트 파일을 찾을 수 없습니다")
        return False
    
    # 누락된 모델 분석
    missing_report = finder.generate_missing_models_report()
    
    logger.info("\n" + "=" * 70)
    logger.info("📋 누락된 모델 분석 결과")
    logger.info("=" * 70)
    logger.info(f"📊 필요한 모델: {missing_report['total_required']}개")
    logger.info(f"✅ 발견된 모델: {missing_report['found_count']}개")
    logger.info(f"❌ 누락된 모델: {missing_report['missing_count']}개")
    
    if missing_report["missing_models"]:
        logger.info("\n❌ 누락된 모델들:")
        for missing_model in missing_report["missing_models"]:
            logger.info(f"  - {missing_model}")
            if missing_model in missing_report["download_suggestions"]:
                suggestion = missing_report["download_suggestions"][missing_model]
                logger.info(f"    다운로드: {suggestion['name']}")
                logger.info(f"    URL: {suggestion['url']}")
                logger.info(f"    파일: {', '.join(suggestion['files'])}")
    
    # 재배치 계획
    relocate_plan = finder.create_checkpoint_relocate_plan()
    
    logger.info("\n" + "=" * 70)
    logger.info("🚀 체크포인트 재배치 계획")
    logger.info("=" * 70)
    logger.info(f"📊 재배치 대상: {relocate_plan['summary']['total_actions']}개")
    logger.info(f"💾 총 크기: {relocate_plan['summary']['total_size_gb']:.2f}GB")
    
    for action in relocate_plan["actions"]:
        logger.info(f"\n📁 {action['model_type']}:")
        logger.info(f"  소스: {action['source']}")
        logger.info(f"  타겟: {action['target']}")
        logger.info(f"  크기: {action['size_mb']:.1f}MB")
        logger.info(f"  액션: {action['action']}")
    
    # 결과 파일 저장
    results = {
        "checkpoints": {
            model_type: [
                {
                    "file_path": str(cp.file_path),
                    "file_size_mb": cp.file_size_mb,
                    "confidence": cp.confidence,
                    "pytorch_valid": cp.pytorch_valid,
                    "parameter_count": cp.parameter_count
                }
                for cp in checkpoints_list
            ]
            for model_type, checkpoints_list in checkpoints.items()
        },
        "missing_report": missing_report,
        "relocate_plan": relocate_plan
    }
    
    results_file = Path("checkpoint_analysis_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ 상세 결과 저장: {results_file}")
    
    return len(checkpoints) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)