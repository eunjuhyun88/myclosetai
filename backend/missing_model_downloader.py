#!/usr/bin/env python3
"""
🔥 MyCloset AI - 누락 모델 자동 다운로드 및 설정 시스템
===============================================================================
✅ Step 01-05 필수 모델 자동 다운로드
✅ conda 환경 최적화
✅ 실제 파일 구조 기반 경로 설정
✅ M3 Max 128GB 메모리 최적화
✅ 체크포인트 검증 및 설정
===============================================================================
"""

import os
import sys
import requests
import hashlib
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_download.log')
    ]
)
logger = logging.getLogger(__name__)

# ==============================================
# 🔥 1. 누락 모델 정의 (실제 필요한 모델들)
# ==============================================

@dataclass
class ModelInfo:
    """모델 정보"""
    name: str
    step_name: str
    file_name: str
    download_url: str
    file_size_mb: float
    sha256_hash: Optional[str]
    description: str
    priority: int  # 1=필수, 2=권장, 3=선택
    target_path: str

# 🔥 실제 필요한 누락 모델들
MISSING_MODELS = {
    # Step 01: Human Parsing (필수)
    "human_parsing_atr": ModelInfo(
        name="human_parsing_atr",
        step_name="HumanParsingStep", 
        file_name="exp-schp-201908301523-atr.pth",
        download_url="https://drive.google.com/uc?id=1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH",
        file_size_mb=255.1,
        sha256_hash=None,
        description="Self-Correction Human Parsing (SCHP) ATR 모델 - 인체 파싱",
        priority=1,
        target_path="step_01_human_parsing"
    ),
    
    "graphonomy_lip": ModelInfo(
        name="graphonomy_lip",
        step_name="HumanParsingStep",
        file_name="graphonomy_lip.pth",
        download_url="https://drive.google.com/uc?id=1P_jkP5fZ8-sj8BIb4oJzZfHk7uJyuWi-",
        file_size_mb=255.1,
        sha256_hash=None,
        description="Graphonomy LIP 모델 - 고품질 인체 파싱",
        priority=2,
        target_path="step_01_human_parsing"
    ),
    
    # Step 02: Pose Estimation (필수)
    "openpose_body": ModelInfo(
        name="openpose_body",
        step_name="PoseEstimationStep",
        file_name="body_pose_model.pth",
        download_url="https://www.dropbox.com/s/llpxd14is7gyj0z/body_pose_model.pth",
        file_size_mb=199.6,
        sha256_hash=None,
        description="OpenPose Body 모델 - 18-키포인트 포즈 추정",
        priority=1,
        target_path="step_02_pose_estimation"
    ),
    
    "openpose_full": ModelInfo(
        name="openpose_full",
        step_name="PoseEstimationStep", 
        file_name="openpose.pth",
        download_url="https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases/download/v1.7.0/pose_coco.caffemodel",
        file_size_mb=200.0,
        sha256_hash=None,
        description="OpenPose Full 모델 - 전신 포즈 추정",
        priority=1,
        target_path="step_02_pose_estimation"
    ),
    
    # Step 03: Cloth Segmentation (필수)
    "u2net_cloth": ModelInfo(
        name="u2net_cloth",
        step_name="ClothSegmentationStep",
        file_name="u2net.pth", 
        download_url="https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth",
        file_size_mb=168.1,
        sha256_hash="347c3d51b944ce1cd92e8e3de4f35734e1a3cf82e7ecb86b9fcbb695d9a94e33",
        description="U²-Net 모델 - 의류 세그멘테이션",
        priority=1,
        target_path="step_03_cloth_segmentation"
    ),
    
    # Step 04: Geometric Matching (권장)
    "geometric_matching": ModelInfo(
        name="geometric_matching",
        step_name="GeometricMatchingStep",
        file_name="gmm.pth",
        download_url="https://github.com/sergeywong/cp-vton/releases/download/checkpoint/gmm_final.pth",
        file_size_mb=50.0,
        sha256_hash=None,
        description="Geometric Matching Module - CP-VTON",
        priority=2,
        target_path="step_04_geometric_matching"
    ),
    
    # Step 05: Cloth Warping (권장)
    "cloth_warping_tom": ModelInfo(
        name="cloth_warping_tom",
        step_name="ClothWarpingStep",
        file_name="tom.pth",
        download_url="https://github.com/sergeywong/cp-vton/releases/download/checkpoint/tom_final.pth",
        file_size_mb=100.0,
        sha256_hash=None,
        description="Try-On Module - CP-VTON 의류 워핑",
        priority=2,
        target_path="step_05_cloth_warping"
    ),
    
    # 추가 유틸리티 모델들
    "sam_vit_h": ModelInfo(
        name="sam_vit_h", 
        step_name="ClothSegmentationStep",
        file_name="sam_vit_h_4b8939.pth",
        download_url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        file_size_mb=2445.7,
        sha256_hash="a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
        description="Segment Anything Model (SAM) ViT-H - 고정밀 세그멘테이션",
        priority=3,
        target_path="step_03_cloth_segmentation"
    )
}

# ==============================================
# 🔥 2. 프로젝트 경로 관리
# ==============================================

class ProjectPathManager:
    """프로젝트 경로 관리자"""
    
    def __init__(self):
        self.project_root = self._find_project_root()
        self.backend_dir = self.project_root / "backend"
        self.ai_models_dir = self.backend_dir / "ai_models"
        
    def _find_project_root(self) -> Path:
        """프로젝트 루트 찾기"""
        current = Path(__file__).resolve()
        
        # backend 폴더 찾기
        for _ in range(10):
            if current.name == 'backend':
                return current.parent  # mycloset-ai 루트
            if current.parent == current:
                break
            current = current.parent
        
        # 현재 경로에서 추측
        current = Path.cwd()
        if 'mycloset-ai' in str(current):
            while current.name != 'mycloset-ai' and current.parent != current:
                current = current.parent
            return current
        
        # 기본 경로
        return Path.home() / "MVP" / "mycloset-ai"
    
    def get_step_dir(self, step_path: str) -> Path:
        """Step별 디렉토리 경로"""
        return self.ai_models_dir / step_path
    
    def ensure_directories(self):
        """필요한 디렉토리 생성"""
        dirs_to_create = [
            self.ai_models_dir,
            self.ai_models_dir / "step_01_human_parsing",
            self.ai_models_dir / "step_02_pose_estimation", 
            self.ai_models_dir / "step_03_cloth_segmentation",
            self.ai_models_dir / "step_04_geometric_matching",
            self.ai_models_dir / "step_05_cloth_warping",
            self.ai_models_dir / "step_06_virtual_fitting",  # 이미 존재
            self.ai_models_dir / "step_07_post_processing",
            self.ai_models_dir / "step_08_quality_assessment"  # 이미 존재
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ 디렉토리 생성: {dir_path}")

# ==============================================
# 🔥 3. 모델 다운로더
# ==============================================

class ModelDownloader:
    """모델 다운로드 관리자"""
    
    def __init__(self, path_manager: ProjectPathManager):
        self.path_manager = path_manager
        self.download_stats = {
            "total_models": 0,
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "total_size_mb": 0
        }
    
    def download_file_with_progress(self, url: str, file_path: Path, expected_size_mb: float) -> bool:
        """파일 다운로드 (진행률 표시)"""
        try:
            logger.info(f"📥 다운로드 시작: {file_path.name} ({expected_size_mb:.1f}MB)")
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 진행률 출력
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r  진행률: {progress:.1f}% ({downloaded/(1024*1024):.1f}MB)", end='', flush=True)
            
            print()  # 새 줄
            
            # 파일 크기 확인
            actual_size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ 다운로드 완료: {file_path.name} ({actual_size_mb:.1f}MB)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 다운로드 실패 {file_path.name}: {e}")
            return False
    
    def verify_file_hash(self, file_path: Path, expected_hash: str) -> bool:
        """파일 해시 검증"""
        if not expected_hash:
            return True  # 해시가 없으면 패스
            
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            
            actual_hash = sha256_hash.hexdigest()
            if actual_hash.lower() == expected_hash.lower():
                logger.info(f"✅ 해시 검증 성공: {file_path.name}")
                return True
            else:
                logger.error(f"❌ 해시 불일치: {file_path.name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 해시 검증 실패: {e}")
            return False
    
    def download_model(self, model_info: ModelInfo) -> bool:
        """개별 모델 다운로드"""
        target_dir = self.path_manager.get_step_dir(model_info.target_path)
        target_file = target_dir / model_info.file_name
        
        # 이미 존재하는지 확인
        if target_file.exists():
            actual_size_mb = target_file.stat().st_size / (1024 * 1024)
            if abs(actual_size_mb - model_info.file_size_mb) < 10:  # 10MB 오차 허용
                logger.info(f"⏭️ 이미 존재: {model_info.file_name} ({actual_size_mb:.1f}MB)")
                self.download_stats["skipped"] += 1
                return True
        
        # 디렉토리 생성
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 다운로드 시도
        success = self.download_file_with_progress(
            model_info.download_url, 
            target_file, 
            model_info.file_size_mb
        )
        
        if success:
            # 해시 검증
            if model_info.sha256_hash:
                if not self.verify_file_hash(target_file, model_info.sha256_hash):
                    logger.error(f"❌ {model_info.file_name} 해시 검증 실패")
                    target_file.unlink()  # 삭제
                    success = False
            
            if success:
                self.download_stats["downloaded"] += 1
                self.download_stats["total_size_mb"] += model_info.file_size_mb
                logger.info(f"🎉 {model_info.file_name} 다운로드 성공!")
        
        if not success:
            self.download_stats["failed"] += 1
        
        return success

# ==============================================
# 🔥 4. conda 환경 설정 관리자
# ==============================================

class CondaEnvironmentManager:
    """conda 환경 설정 관리자"""
    
    def __init__(self):
        self.conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
        self.conda_prefix = os.environ.get('CONDA_PREFIX', '')
        self.is_m3_max = self._detect_m3_max()
    
    def _detect_m3_max(self) -> bool:
        """M3 Max 감지"""
        try:
            import platform
            if platform.processor() == 'arm' and platform.system() == 'Darwin':
                # macOS ARM64
                with open('/proc/cpuinfo', 'r') as f:
                    cpu_info = f.read()
                    return 'Apple M3 Max' in cpu_info
        except:
            pass
        
        # 간단한 ARM64 macOS 감지
        import os
        return 'arm64' in str(os.uname()) if hasattr(os, 'uname') else False
    
    def check_conda_environment(self) -> Dict[str, Any]:
        """conda 환경 체크"""
        info = {
            "conda_active": bool(self.conda_env),
            "conda_env": self.conda_env,
            "conda_prefix": self.conda_prefix,
            "is_m3_max": self.is_m3_max,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "required_packages": [],
            "missing_packages": []
        }
        
        # 필수 패키지 체크
        required_packages = {
            'torch': 'PyTorch',
            'torchvision': 'TorchVision', 
            'numpy': 'NumPy',
            'pillow': 'PIL',
            'opencv-python': 'OpenCV',
            'requests': 'Requests'
        }
        
        for package, display_name in required_packages.items():
            try:
                __import__(package.replace('-', '_'))
                info["required_packages"].append(display_name)
            except ImportError:
                info["missing_packages"].append(display_name)
        
        return info
    
    def print_conda_setup_guide(self):
        """conda 설정 가이드 출력"""
        print("""
🐍 MyCloset AI - conda 환경 설정 가이드 (필수 모델용)

# 1. conda 환경 생성
conda create -n mycloset-ai python=3.9 -y
conda activate mycloset-ai

# 2. PyTorch 설치 (M3 Max 최적화)
conda install pytorch torchvision torchaudio -c pytorch -y

# 3. 기본 라이브러리 설치
conda install numpy pillow opencv -c conda-forge -y
pip install requests tqdm

# 4. AI 모델 라이브러리 (선택사항)
pip install transformers diffusers rembg

# 5. 검증
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'MPS 사용가능: {torch.backends.mps.is_available()}')"

# 6. 모델 다운로드 실행
python missing_model_downloader.py --download-all
        """)

# ==============================================
# 🔥 5. 메인 관리자 클래스
# ==============================================

class MissingModelManager:
    """누락 모델 통합 관리자"""
    
    def __init__(self):
        self.path_manager = ProjectPathManager()
        self.downloader = ModelDownloader(self.path_manager)
        self.conda_manager = CondaEnvironmentManager()
        
        logger.info(f"🏠 프로젝트 루트: {self.path_manager.project_root}")
        logger.info(f"🤖 AI 모델 경로: {self.path_manager.ai_models_dir}")
    
    def analyze_current_state(self) -> Dict[str, Any]:
        """현재 상태 분석"""
        state = {
            "project_info": {
                "root_path": str(self.path_manager.project_root),
                "ai_models_path": str(self.path_manager.ai_models_dir),
                "ai_models_exists": self.path_manager.ai_models_dir.exists()
            },
            "conda_info": self.conda_manager.check_conda_environment(),
            "existing_models": self._scan_existing_models(),
            "missing_models": self._identify_missing_models(),
            "recommendations": []
        }
        
        # 권장사항 생성
        if not state["conda_info"]["conda_active"]:
            state["recommendations"].append("conda 환경 활성화 필요")
        
        if state["conda_info"]["missing_packages"]:
            state["recommendations"].append(f"패키지 설치 필요: {', '.join(state['conda_info']['missing_packages'])}")
        
        if state["missing_models"]:
            total_missing_size = sum(MISSING_MODELS[m]["file_size_mb"] for m in state["missing_models"])
            state["recommendations"].append(f"모델 다운로드 필요: {len(state['missing_models'])}개 ({total_missing_size:.1f}MB)")
        
        return state
    
    def _scan_existing_models(self) -> Dict[str, List[str]]:
        """기존 모델 스캔"""
        existing = {}
        
        if not self.path_manager.ai_models_dir.exists():
            return existing
        
        # 각 Step 폴더 스캔
        for step_dir in self.path_manager.ai_models_dir.iterdir():
            if step_dir.is_dir():
                step_name = step_dir.name
                models = []
                
                for model_file in step_dir.rglob('*'):
                    if model_file.is_file() and model_file.suffix in ['.pth', '.pt', '.bin', '.ckpt', '.safetensors']:
                        size_mb = model_file.stat().st_size / (1024 * 1024)
                        models.append(f"{model_file.name} ({size_mb:.1f}MB)")
                
                if models:
                    existing[step_name] = models
        
        return existing
    
    def _identify_missing_models(self) -> List[str]:
        """누락된 모델 식별"""
        missing = []
        
        for model_key, model_info in MISSING_MODELS.items():
            target_dir = self.path_manager.get_step_dir(model_info.target_path)
            target_file = target_dir / model_info.file_name
            
            if not target_file.exists():
                missing.append(model_key)
        
        return missing
    
    def print_status_report(self):
        """상태 리포트 출력"""
        state = self.analyze_current_state()
        
        print("="*70)
        print("🔍 MyCloset AI - 모델 상태 분석")
        print("="*70)
        
        # 프로젝트 정보
        print(f"🏠 프로젝트 루트: {state['project_info']['root_path']}")
        print(f"🤖 AI 모델 경로: {state['project_info']['ai_models_path']}")
        print(f"   존재 여부: {'✅' if state['project_info']['ai_models_exists'] else '❌'}")
        
        # conda 환경
        conda = state['conda_info']
        print(f"\n🐍 conda 환경:")
        print(f"   활성화: {'✅' if conda['conda_active'] else '❌'}")
        print(f"   환경명: {conda['conda_env'] or 'None'}")
        print(f"   Python: {conda['python_version']}")
        print(f"   M3 Max: {'✅' if conda['is_m3_max'] else '❌'}")
        
        if conda['missing_packages']:
            print(f"   ⚠️ 누락 패키지: {', '.join(conda['missing_packages'])}")
        
        # 기존 모델
        print(f"\n📦 기존 모델:")
        if state['existing_models']:
            for step, models in state['existing_models'].items():
                print(f"   {step}:")
                for model in models:
                    print(f"     - {model}")
        else:
            print("   없음")
        
        # 누락 모델
        print(f"\n❌ 누락 모델 ({len(state['missing_models'])}개):")
        if state['missing_models']:
            total_size = 0
            for model_key in state['missing_models']:
                model_info = MISSING_MODELS[model_key]
                priority_emoji = "🔥" if model_info.priority == 1 else "⭐" if model_info.priority == 2 else "💡"
                print(f"   {priority_emoji} {model_info.file_name} ({model_info.file_size_mb:.1f}MB) - {model_info.description}")
                total_size += model_info.file_size_mb
            print(f"   📊 총 다운로드 크기: {total_size:.1f}MB ({total_size/1024:.1f}GB)")
        else:
            print("   없음 - 모든 모델이 준비됨! ✅")
        
        # 권장사항
        if state['recommendations']:
            print(f"\n💡 권장사항:")
            for i, rec in enumerate(state['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print("="*70)
    
    def download_missing_models(self, priority_filter: Optional[int] = None) -> bool:
        """누락된 모델 다운로드"""
        missing_models = self._identify_missing_models()
        
        if not missing_models:
            logger.info("✅ 모든 모델이 이미 존재합니다!")
            return True
        
        # 우선순위 필터링
        if priority_filter:
            missing_models = [m for m in missing_models if MISSING_MODELS[m].priority <= priority_filter]
        
        if not missing_models:
            logger.info(f"✅ 우선순위 {priority_filter} 이하 모델들이 모두 존재합니다!")
            return True
        
        logger.info(f"📥 {len(missing_models)}개 모델 다운로드 시작...")
        
        # 디렉토리 생성
        self.path_manager.ensure_directories()
        
        # 다운로드 실행
        success_count = 0
        for model_key in missing_models:
            model_info = MISSING_MODELS[model_key]
            
            if self.downloader.download_model(model_info):
                success_count += 1
            
            # 메모리 정리 (대용량 파일 처리 후)
            if model_info.file_size_mb > 500:
                import gc
                gc.collect()
        
        # 결과 리포트
        stats = self.downloader.download_stats
        print("\n" + "="*50)
        print("📊 다운로드 결과:")
        print(f"   ✅ 성공: {stats['downloaded']}개")
        print(f"   ⏭️ 건너뜀: {stats['skipped']}개") 
        print(f"   ❌ 실패: {stats['failed']}개")
        print(f"   📁 총 다운로드: {stats['total_size_mb']:.1f}MB")
        print("="*50)
        
        return stats['failed'] == 0
    
    def create_model_config(self) -> Dict[str, Any]:
        """모델 설정 파일 생성"""
        config = {
            "version": "1.0",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "project_root": str(self.path_manager.project_root),
            "ai_models_path": str(self.path_manager.ai_models_dir),
            "models": {}
        }
        
        # 각 모델 정보 추가
        for model_key, model_info in MISSING_MODELS.items():
            target_path = self.path_manager.get_step_dir(model_info.target_path) / model_info.file_name
            
            config["models"][model_key] = {
                "name": model_info.name,
                "step_name": model_info.step_name,
                "file_name": model_info.file_name,
                "file_path": str(target_path),
                "file_size_mb": model_info.file_size_mb,
                "description": model_info.description,
                "priority": model_info.priority,
                "exists": target_path.exists(),
                "file_size_actual": target_path.stat().st_size / (1024 * 1024) if target_path.exists() else 0
            }
        
        # 설정 파일 저장
        config_path = self.path_manager.ai_models_dir / "model_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 모델 설정 파일 생성: {config_path}")
        return config

# ==============================================
# 🔥 6. CLI 인터페이스
# ==============================================

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MyCloset AI - 누락 모델 다운로드 도구')
    parser.add_argument('--status', action='store_true', help='현재 상태만 확인')
    parser.add_argument('--download-all', action='store_true', help='모든 누락 모델 다운로드')
    parser.add_argument('--download-priority', type=int, choices=[1, 2, 3], 
                       help='지정된 우선순위 이하 모델만 다운로드 (1=필수, 2=권장, 3=선택)')
    parser.add_argument('--conda-guide', action='store_true', help='conda 설정 가이드 출력')
    parser.add_argument('--create-config', action='store_true', help='모델 설정 파일 생성')
    
    args = parser.parse_args()
    
    # 관리자 초기화
    manager = MissingModelManager()
    
    # conda 가이드
    if args.conda_guide:
        manager.conda_manager.print_conda_setup_guide()
        return
    
    # 상태 확인
    manager.print_status_report()
    
    if args.status:
        return
    
    # 설정 파일 생성
    if args.create_config:
        manager.create_model_config()
        print("✅ 모델 설정 파일 생성 완료")
        return
    
    # 다운로드 실행
    if args.download_all:
        success = manager.download_missing_models()
        if success:
            print("🎉 모든 모델 다운로드 완료!")
            manager.create_model_config()
        else:
            print("⚠️ 일부 모델 다운로드 실패")
            return 1
    
    elif args.download_priority:
        success = manager.download_missing_models(priority_filter=args.download_priority)
        if success:
            print(f"🎉 우선순위 {args.download_priority} 모델 다운로드 완료!")
            manager.create_model_config()
        else:
            print("⚠️ 일부 모델 다운로드 실패")
            return 1
    
    else:
        print("\n💡 다음 명령어로 모델을 다운로드하세요:")
        print("   python missing_model_downloader.py --download-priority 1  # 필수 모델만")
        print("   python missing_model_downloader.py --download-all         # 모든 모델")
        print("   python missing_model_downloader.py --conda-guide          # conda 설정 가이드")

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code or 0)
    except KeyboardInterrupt:
        print("\n⚠️ 사용자가 중단했습니다.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)