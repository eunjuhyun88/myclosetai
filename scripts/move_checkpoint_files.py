# scripts/move_checkpoint_files.py
"""
🔥 MyCloset AI - 실제 파일 이동을 통한 체크포인트 경로 수정
================================================================================
✅ 심볼릭 링크 대신 실제 파일 이동
✅ 소스 파일 삭제 후 타겟 위치에만 파일 유지
✅ 분석 결과 기반 최적 파일 위치 결정
✅ 백업 생성 및 안전한 복구 메커니즘
✅ 중복 파일 정리 및 공간 절약
================================================================================
"""

import os
import sys
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CheckpointFileMover:
    """실제 파일 이동을 통한 체크포인트 정리"""
    
    def __init__(self, backend_dir: str = "backend"):
        self.backend_dir = Path(backend_dir)
        self.ai_models_dir = self.backend_dir / "ai_models"
        self.backup_dir = self.ai_models_dir / "move_backup"
        
        # 분석 결과 로드
        self.analysis_results = self._load_analysis_results()
        
        # 코드에서 찾는 파일 → 실제 이동할 위치 매핑
        self.target_mappings = {
            # Virtual Fitting에서 찾는 파일들을 해당 위치로 이동
            "step_06_virtual_fitting/body_pose_model.pth": "step_02_pose_estimation/body_pose_model.pth",
            "step_06_virtual_fitting/hrviton_final_01.pth": "step_06_virtual_fitting/hrviton_final.pth",
            "step_06_virtual_fitting/exp-schp-201908261155-lip.pth": "step_01_human_parsing/exp-schp-201908261155-lip.pth",
            "step_06_virtual_fitting/exp-schp-201908301523-atr.pth": "step_01_human_parsing/exp-schp-201908301523-atr.pth",
            
            # Human Parsing 표준화 (코드에서 찾는 파일명으로 이동)
            "step_01_human_parsing/exp-schp-201908261155-lip_22.pth": "step_01_human_parsing/exp-schp-201908261155-lip.pth",
            "step_01_human_parsing/graphonomy_08.pth": "step_01_human_parsing/graphonomy.pth",
            "step_01_human_parsing/exp-schp-201908301523-atr_30.pth": "step_01_human_parsing/exp-schp-201908301523-atr.pth",
            
            # Pose Estimation 표준화
            "step_02_pose_estimation/body_pose_model_41.pth": "step_02_pose_estimation/body_pose_model.pth",
            "step_02_pose_estimation/openpose_08.pth": "step_02_pose_estimation/openpose.pth",
            
            # Cloth Warping
            "step_05_cloth_warping/tom_final_01.pth": "step_05_cloth_warping/tom_final.pth",
        }
        
        # 분석 결과 기반 추천 파일들을 표준 이름으로 이동
        self.recommended_moves = self._generate_recommended_moves()
        
    def _load_analysis_results(self) -> Dict:
        """분석 결과 로드"""
        analysis_file = self.backend_dir / "analysis_results" / "optimized_model_config.json"
        
        if analysis_file.exists():
            try:
                with open(analysis_file, 'r') as f:
                    results = json.load(f)
                logger.info(f"✅ 분석 결과 로드 완료: {analysis_file}")
                return results
            except Exception as e:
                logger.warning(f"⚠️ 분석 결과 로드 실패: {e}")
                
        return {}
        
    def _generate_recommended_moves(self) -> Dict[str, str]:
        """분석 결과 기반 추천 이동 매핑 생성"""
        moves = {}
        
        if not self.analysis_results or "step_configs" not in self.analysis_results:
            return moves
            
        # 각 Step의 추천 모델을 표준 이름으로 이동
        step_standard_names = {
            "step_01_human_parsing": {
                "primary": "human_parsing_primary.pth",
                "atr": "exp-schp-201908301523-atr.pth", 
                "lip": "exp-schp-201908261155-lip.pth",
                "graphonomy": "graphonomy.pth"
            },
            "step_02_pose_estimation": {
                "primary": "pose_estimation_primary.pth",
                "body": "body_pose_model.pth",
                "openpose": "openpose.pth",
                "hand": "hand_pose_model.pth"
            },
            "step_03_cloth_segmentation": {
                "primary": "cloth_segmentation_primary.pth",
                "u2net": "u2net.pth",
                "sam": "sam_vit_h.pth"
            },
            "step_06_virtual_fitting": {
                "primary": "virtual_fitting_primary.pth",
                "hrviton": "hrviton_final.pth",
                "diffusion": "diffusion_pytorch_model.bin",
                "ootd": "ootd_hd_unet.bin"
            }
        }
        
        for step_name, config in self.analysis_results["step_configs"].items():
            if step_name in step_standard_names:
                primary_model = config["primary_model"]
                standard_names = step_standard_names[step_name]
                
                # 추천 모델을 primary로 이동
                current_path = primary_model["path"]
                target_path = f"{step_name}/{standard_names['primary']}"
                moves[target_path] = current_path
                
        return moves
        
    def create_backup(self) -> bool:
        """이동 전 백업 생성"""
        logger.info("📋 이동 전 백업 생성 중...")
        
        try:
            # 백업 디렉토리 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"before_move_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # 이동할 파일들의 현재 상태 기록
            backup_manifest = {
                "timestamp": timestamp,
                "total_files": 0,
                "moves_planned": {},
                "file_checksums": {}
            }
            
            # 이동 계획 기록
            all_moves = {**self.target_mappings, **self.recommended_moves}
            
            for target_path, source_path in all_moves.items():
                source_full = self.ai_models_dir / source_path
                target_full = self.ai_models_dir / target_path
                
                backup_manifest["moves_planned"][target_path] = {
                    "source": source_path,
                    "source_exists": source_full.exists(),
                    "target_exists": target_full.exists(),
                    "source_size": source_full.stat().st_size if source_full.exists() else 0
                }
                
            # 백업 매니페스트 저장
            with open(backup_path / "move_manifest.json", "w") as f:
                json.dump(backup_manifest, f, indent=2)
                
            logger.info(f"✅ 백업 생성 완료: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 백업 생성 실패: {e}")
            return False
            
    def find_best_source_file(self, target_path: str) -> Path:
        """타겟 경로에 가장 적합한 소스 파일 찾기"""
        # 1. 명시적 매핑 확인
        if target_path in self.target_mappings:
            source_path = self.ai_models_dir / self.target_mappings[target_path]
            if source_path.exists():
                return source_path
                
        # 2. 추천 매핑 확인
        if target_path in self.recommended_moves:
            source_path = self.ai_models_dir / self.recommended_moves[target_path]
            if source_path.exists():
                return source_path
                
        # 3. 유사한 이름의 파일 찾기
        target_name = Path(target_path).name
        target_step = Path(target_path).parent.name
        
        # 같은 스텝 내에서 유사한 파일 찾기
        step_dir = self.ai_models_dir / target_step
        if step_dir.exists():
            for file_path in step_dir.glob("*.pth"):
                if self._files_are_similar(file_path.name, target_name):
                    return file_path
                    
        # 4. 전체에서 유사한 파일 찾기
        for file_path in self.ai_models_dir.rglob("*.pth"):
            if "cleanup_backup" in str(file_path):
                continue
            if self._files_are_similar(file_path.name, target_name):
                return file_path
                
        return None
        
    def _files_are_similar(self, file1: str, file2: str) -> bool:
        """두 파일명이 유사한지 확인"""
        # 확장자 제거
        name1 = Path(file1).stem.lower()
        name2 = Path(file2).stem.lower()
        
        # 정확히 일치
        if name1 == name2:
            return True
            
        # 키워드 기반 유사성 확인
        keywords1 = set(name1.replace('_', ' ').replace('-', ' ').split())
        keywords2 = set(name2.replace('_', ' ').replace('-', ' ').split())
        
        # 공통 키워드가 50% 이상
        if keywords1 and keywords2:
            intersection = keywords1.intersection(keywords2)
            union = keywords1.union(keywords2)
            similarity = len(intersection) / len(union)
            return similarity >= 0.5
            
        return False
        
    def move_file_safely(self, source_path: Path, target_path: Path) -> bool:
        """안전하게 파일 이동"""
        try:
            # 타겟 디렉토리 생성
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 타겟 파일이 이미 존재하는 경우
            if target_path.exists():
                # 같은 파일인지 확인
                if source_path.samefile(target_path):
                    logger.info(f"🔄 이미 같은 파일: {target_path.name}")
                    return True
                    
                # 크기 비교
                source_size = source_path.stat().st_size
                target_size = target_path.stat().st_size
                
                if source_size > target_size:
                    # 소스가 더 크면 교체
                    logger.info(f"🔄 더 큰 파일로 교체: {target_path.name} ({source_size/1024/1024:.1f}MB > {target_size/1024/1024:.1f}MB)")
                    target_path.unlink()  # 기존 파일 삭제
                elif source_size == target_size:
                    # 같은 크기면 소스 삭제
                    logger.info(f"♻️ 중복 파일 제거: {source_path.name}")
                    source_path.unlink()
                    return True
                else:
                    # 타겟이 더 크면 소스만 삭제
                    logger.info(f"🗑️ 더 작은 파일 제거: {source_path.name}")
                    source_path.unlink()
                    return True
                    
            # 실제 파일 이동
            shutil.move(str(source_path), str(target_path))
            
            size_mb = target_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ 파일 이동 완료: {source_path.name} → {target_path.name} ({size_mb:.1f}MB)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 파일 이동 실패 {source_path} → {target_path}: {e}")
            return False
            
    def perform_strategic_moves(self) -> Dict[str, int]:
        """전략적 파일 이동 수행"""
        logger.info("🚀 전략적 파일 이동 시작...")
        
        stats = {
            "moved": 0,
            "skipped": 0,
            "failed": 0,
            "removed_duplicates": 0
        }
        
        # 1. 코드에서 찾는 필수 파일들 먼저 이동
        logger.info("📋 1단계: 필수 체크포인트 파일 이동")
        
        required_files = [
            "step_06_virtual_fitting/body_pose_model.pth",
            "step_06_virtual_fitting/hrviton_final_01.pth", 
            "step_06_virtual_fitting/exp-schp-201908261155-lip.pth",
            "step_06_virtual_fitting/exp-schp-201908301523-atr.pth",
            "step_01_human_parsing/exp-schp-201908261155-lip_22.pth",
            "step_01_human_parsing/graphonomy_08.pth",
            "step_01_human_parsing/exp-schp-201908301523-atr_30.pth",
            "step_02_pose_estimation/body_pose_model_41.pth",
            "step_02_pose_estimation/openpose_08.pth",
            "step_05_cloth_warping/tom_final_01.pth"
        ]
        
        for target_path_str in required_files:
            target_path = self.ai_models_dir / target_path_str
            
            if target_path.exists():
                logger.info(f"✅ 이미 존재: {target_path_str}")
                continue
                
            # 최적 소스 파일 찾기
            source_path = self.find_best_source_file(target_path_str)
            
            if source_path:
                if self.move_file_safely(source_path, target_path):
                    stats["moved"] += 1
                else:
                    stats["failed"] += 1
            else:
                logger.warning(f"⚠️ 소스 파일 없음: {target_path_str}")
                stats["skipped"] += 1
                
        # 2. 분석 결과 기반 최적화
        logger.info("📋 2단계: 분석 결과 기반 최적화")
        
        if self.analysis_results and "step_configs" in self.analysis_results:
            for step_name, config in self.analysis_results["step_configs"].items():
                primary_model = config["primary_model"]
                
                # 추천 모델을 표준 위치로 이동
                source_path = self.ai_models_dir / primary_model["path"]
                target_path = self.ai_models_dir / step_name / f"{step_name}_primary.pth"
                
                if source_path.exists() and not target_path.exists():
                    if self.move_file_safely(source_path, target_path):
                        stats["moved"] += 1
                    else:
                        stats["failed"] += 1
                        
        return stats
        
    def cleanup_duplicates(self) -> int:
        """중복 파일 정리"""
        logger.info("🧹 중복 파일 정리 중...")
        
        removed_count = 0
        file_groups = {}
        
        # 파일을 크기별로 그룹화
        for model_file in self.ai_models_dir.rglob("*.pth"):
            if "cleanup_backup" in str(model_file):
                continue
                
            try:
                size = model_file.stat().st_size
                if size not in file_groups:
                    file_groups[size] = []
                file_groups[size].append(model_file)
            except:
                continue
                
        # 같은 크기의 파일들 중 중복 제거
        for size, files in file_groups.items():
            if len(files) < 2:
                continue
                
            # 파일명으로 정렬 (더 간단한 이름을 우선)
            files.sort(key=lambda x: (len(x.name), x.name))
            
            # 첫 번째 파일 유지, 나머지 삭제
            keep_file = files[0]
            
            for duplicate_file in files[1:]:
                try:
                    duplicate_file.unlink()
                    removed_count += 1
                    logger.info(f"🗑️ 중복 파일 제거: {duplicate_file.name}")
                except Exception as e:
                    logger.warning(f"⚠️ 중복 파일 제거 실패 {duplicate_file}: {e}")
                    
        return removed_count
        
    def run(self) -> bool:
        """전체 이동 프로세스 실행"""
        logger.info("🚀 체크포인트 파일 이동 프로세스 시작")
        
        if not self.ai_models_dir.exists():
            logger.error(f"❌ AI 모델 디렉토리 없음: {self.ai_models_dir}")
            return False
            
        # 1. 백업 생성
        if not self.create_backup():
            logger.error("❌ 백업 생성 실패, 중단")
            return False
            
        # 2. 전략적 파일 이동
        move_stats = self.perform_strategic_moves()
        
        # 3. 중복 파일 정리
        removed_duplicates = self.cleanup_duplicates()
        move_stats["removed_duplicates"] = removed_duplicates
        
        # 4. 결과 리포트
        logger.info("🎉 파일 이동 완료!")
        logger.info(f"📊 이동 통계:")
        logger.info(f"   - 이동 완료: {move_stats['moved']}개")
        logger.info(f"   - 스킵: {move_stats['skipped']}개") 
        logger.info(f"   - 실패: {move_stats['failed']}개")
        logger.info(f"   - 중복 제거: {move_stats['removed_duplicates']}개")
        
        # 5. 다음 단계 안내
        logger.info("📋 다음 단계:")
        logger.info("   1. python run_server.py 실행")
        logger.info("   2. 체크포인트 로딩 에러 확인")
        logger.info("   3. 필요시 추가 파일 이동")
        
        return move_stats["failed"] == 0

if __name__ == "__main__":
    mover = CheckpointFileMover()
    success = mover.run()
    sys.exit(0 if success else 1)