#!/usr/bin/env python3
"""
🔥 MyCloset AI - GitHub 실제 AI 추론 테스터 v4.0
================================================================================
✅ GitHub 프로젝트 구조 완전 호환 (backend/app/ai_pipeline/)
✅ StepFactory v11.0 + BaseStepMixin v19.2 패턴 사용
✅ DI Container 기반 의존성 주입
✅ 실제 229GB AI 모델 체크포인트 활용
✅ M3 Max 128GB 메모리 최적화
✅ conda 환경 mycloset-ai-clean 자동 감지
✅ 실제 추론 실행 → 결과 이미지 생성 → 정확한 성능 측정
✅ GitHub Actions CI/CD 준비
================================================================================
"""

import os
import sys
import time
import gc
import warnings
import asyncio
import base64
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json

# GitHub 프로젝트 구조 자동 감지
PROJECT_ROOT = Path(__file__).resolve()

# mycloset-ai/backend 구조 감지
while PROJECT_ROOT.name != 'mycloset-ai' and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

if PROJECT_ROOT.name == 'mycloset-ai':
    BACKEND_ROOT = PROJECT_ROOT / "backend"
    if BACKEND_ROOT.exists() and (BACKEND_ROOT / "app").exists():
        sys.path.insert(0, str(BACKEND_ROOT))
        print(f"✅ GitHub 프로젝트 루트 감지: {PROJECT_ROOT}")
        print(f"✅ Backend 루트 설정: {BACKEND_ROOT}")
    else:
        print("❌ GitHub backend/app 구조를 찾을 수 없습니다")
        sys.exit(1)
else:
    print("❌ mycloset-ai 프로젝트 루트를 찾을 수 없습니다")
    sys.exit(1)

# 환경 최적화
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class GitHubInferenceTestType(Enum):
    QUICK = "quick"          # 1개 Step (Human Parsing)
    STANDARD = "standard"    # 3개 핵심 Step
    FULL_PIPELINE = "full"   # 전체 파이프라인
    STRESS_TEST = "stress"   # 스트레스 테스트

@dataclass
class GitHubInferenceResult:
    step_name: str
    success: bool
    inference_time: float
    memory_used_mb: float
    model_loaded: bool = False
    ai_inference_executed: bool = False
    result_saved: bool = False
    output_path: Optional[str] = None
    confidence_score: Optional[float] = None
    error_message: Optional[str] = None
    step_factory_used: bool = False
    di_container_used: bool = False
    basestepmixin_compatible: bool = False

class GitHubAIInferenceTester:
    """GitHub 구조 기반 실제 AI 추론 테스터"""
    
    def __init__(self, test_type: GitHubInferenceTestType = GitHubInferenceTestType.STANDARD):
        self.test_type = test_type
        self.project_root = PROJECT_ROOT
        self.backend_root = BACKEND_ROOT
        self.results_dir = self.backend_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # GitHub 컴포넌트들
        self.step_factory = None
        self.di_container = None
        self.model_loader = None
        self.pipeline_manager = None
        
        # 테스트 결과
        self.inference_results: List[GitHubInferenceResult] = []
        self.total_inference_time = 0.0
        self.peak_memory_mb = 0.0
        
        # conda 환경 확인
        self._check_conda_environment()
        
        print(f"🚀 GitHub MyCloset AI 실제 추론 테스터 v4.0 시작")
        print(f"📁 프로젝트: {self.project_root}")
        print(f"📁 백엔드: {self.backend_root}")
        print(f"🧪 테스트 모드: {test_type.value}")
        
        self._initialize_github_components()
        self._prepare_test_images()
    
    def _check_conda_environment(self):
        """conda 환경 확인"""
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'none')
        if conda_env == 'mycloset-ai-clean':
            print(f"✅ conda 환경 확인: {conda_env}")
        else:
            print(f"⚠️ conda 환경 확인: {conda_env} (권장: mycloset-ai-clean)")
    
    def _initialize_github_components(self):
        """GitHub 프로젝트 컴포넌트 초기화"""
        try:
            print("🔧 GitHub 컴포넌트 초기화 중...")
            
            # StepFactory 초기화 (GitHub 패턴)
            try:
                from app.ai_pipeline.factories.step_factory import get_global_step_factory
                self.step_factory = get_global_step_factory()
                if self.step_factory:
                    print("✅ StepFactory v11.0 초기화 완료")
                else:
                    print("⚠️ StepFactory 초기화 실패")
            except ImportError as e:
                print(f"⚠️ StepFactory import 실패: {e}")
            
            # DI Container 초기화 (GitHub 패턴)
            try:
                from app.core.di_container import get_global_di_container
                self.di_container = get_global_di_container()
                if self.di_container:
                    print("✅ DI Container 초기화 완료")
                else:
                    print("⚠️ DI Container 초기화 실패")
            except ImportError as e:
                print(f"⚠️ DI Container import 실패: {e}")
            
            # ModelLoader 초기화 (GitHub 패턴)
            try:
                from app.ai_pipeline.utils.model_loader import get_global_model_loader
                self.model_loader = get_global_model_loader()
                if self.model_loader:
                    print("✅ ModelLoader v5.1 초기화 완료")
                else:
                    print("⚠️ ModelLoader 초기화 실패")
            except ImportError as e:
                print(f"⚠️ ModelLoader import 실패: {e}")
            
            # PipelineManager 초기화 (선택적)
            try:
                from app.ai_pipeline.pipeline_manager import PipelineManager
                self.pipeline_manager = PipelineManager()
                print("✅ PipelineManager 초기화 완료")
            except ImportError as e:
                print(f"⚠️ PipelineManager import 실패: {e}")
            
            if not any([self.step_factory, self.model_loader]):
                raise Exception("핵심 컴포넌트 초기화 실패")
                
        except Exception as e:
            print(f"❌ GitHub 컴포넌트 초기화 실패: {e}")
            raise
    
    def _prepare_test_images(self):
        """테스트용 실제 이미지 생성 (GitHub 표준)"""
        print("🖼️ GitHub 테스트 이미지 준비 중...")
        
        # 512x512 고품질 테스트 인물 이미지
        person_img = Image.new('RGB', (512, 512), color='lightsteelblue')
        draw = ImageDraw.Draw(person_img)
        
        # 사실적인 인물 형태
        # 머리
        draw.ellipse([180, 30, 332, 180], fill='peachpuff', outline='saddlebrown', width=2)
        # 목
        draw.rectangle([236, 180, 276, 210], fill='peachpuff', outline='saddlebrown')
        # 상체 (티셔츠 영역)
        draw.rectangle([200, 210, 312, 380], fill='lightcoral', outline='darkred', width=2)
        # 팔
        draw.rectangle([160, 230, 200, 320], fill='peachpuff', outline='saddlebrown', width=2)
        draw.rectangle([312, 230, 352, 320], fill='peachpuff', outline='saddlebrown', width=2)
        # 하체 (바지 영역)
        draw.rectangle([220, 380, 292, 480], fill='navy', outline='darkblue', width=2)
        
        # GitHub 테스트 라벨
        try:
            font = ImageFont.load_default()
            draw.text((160, 10), "GitHub Test Person", fill='black', font=font)
            draw.text((200, 490), "MyCloset AI", fill='black', font=font)
        except:
            draw.text((160, 10), "GitHub Test Person", fill='black')
            draw.text((200, 490), "MyCloset AI", fill='black')
        
        person_path = self.results_dir / "github_test_person.jpg"
        person_img.save(person_path, quality=95)
        
        # 512x512 고품질 의류 이미지
        cloth_img = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(cloth_img)
        
        # 상세한 티셔츠 디자인
        # 메인 몸통
        draw.rectangle([130, 80, 382, 420], fill='crimson', outline='darkred', width=3)
        # 소매 (더 사실적)
        draw.rectangle([80, 100, 130, 220], fill='crimson', outline='darkred', width=2)
        draw.rectangle([382, 100, 432, 220], fill='crimson', outline='darkred', width=2)
        # 목선 (라운드 넥)
        draw.arc([180, 60, 332, 120], 0, 180, fill='darkred', width=4)
        # 디자인 요소
        draw.rectangle([180, 200, 332, 240], fill='white', outline='darkred', width=2)
        
        try:
            font = ImageFont.load_default()
            draw.text((180, 30), "GitHub Test Cloth", fill='black', font=font)
            draw.text((210, 210), "AI Fashion", fill='darkred', font=font)
            draw.text((200, 460), "MyCloset", fill='darkred', font=font)
        except:
            draw.text((180, 30), "GitHub Test Cloth", fill='black')
            draw.text((210, 210), "AI Fashion", fill='darkred')
            draw.text((200, 460), "MyCloset", fill='darkred')
        
        cloth_path = self.results_dir / "github_test_cloth.jpg"
        cloth_img.save(cloth_path, quality=95)
        
        self.test_person_path = person_path
        self.test_cloth_path = cloth_path
        
        print(f"✅ GitHub 테스트 이미지 준비 완료:")
        print(f"   인물: {person_path}")
        print(f"   의류: {cloth_path}")
    
    def _monitor_memory(self) -> float:
        """메모리 사용량 모니터링 (M3 Max 최적화)"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            return memory_mb
        except:
            return 0.0
    
    def _load_image_data(self, image_path: Path) -> Dict[str, Any]:
        """GitHub 표준 이미지 데이터 로딩"""
        try:
            pil_image = Image.open(image_path).convert('RGB')
            np_array = np.array(pil_image)
            
            # Base64 인코딩
            import io
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=95)
            base64_str = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                'pil_image': pil_image,
                'numpy_array': np_array,
                'base64_string': base64_str,
                'image_path': str(image_path),
                'width': pil_image.width,
                'height': pil_image.height,
                'shape': np_array.shape,
                'format': 'RGB',
                'quality': 'high'
            }
        except Exception as e:
            print(f"❌ 이미지 데이터 로딩 실패 {image_path}: {e}")
            return {}
    
    def test_github_step_inference(self, step_name: str) -> GitHubInferenceResult:
        """GitHub Step 실제 추론 테스트"""
        print(f"🧠 GitHub {step_name} 실제 추론 테스트 시작...")
        
        start_time = time.time()
        memory_before = self._monitor_memory()
        
        try:
            # Step 인스턴스 생성 (GitHub StepFactory 패턴)
            step_instance = None
            step_factory_used = False
            di_container_used = False
            
            if self.step_factory:
                try:
                    # StepFactory를 통한 생성
                    step_instance = self.step_factory.create_step(step_name)
                    step_factory_used = True
                    print(f"  ✅ StepFactory로 {step_name} 생성 성공")
                except Exception as e:
                    print(f"  ⚠️ StepFactory 생성 실패: {e}")
            
            # 폴백: 직접 import
            if not step_instance:
                step_instance = self._create_step_directly(step_name)
                if step_instance:
                    print(f"  ✅ 직접 import로 {step_name} 생성 성공")
            
            if not step_instance:
                return GitHubInferenceResult(
                    step_name=step_name,
                    success=False,
                    inference_time=time.time() - start_time,
                    memory_used_mb=0.0,
                    error_message="Step 인스턴스 생성 실패"
                )
            
            # BaseStepMixin 호환성 검증
            basestepmixin_compatible = self._check_basestepmixin_compatibility(step_instance)
            
            # DI Container 의존성 주입 (선택적)
            if self.di_container and hasattr(step_instance, 'inject_dependencies'):
                try:
                    step_instance.inject_dependencies(self.di_container)
                    di_container_used = True
                    print(f"  ✅ DI Container 의존성 주입 완료")
                except Exception as e:
                    print(f"  ⚠️ DI Container 주입 실패: {e}")
            
            # 테스트 데이터 준비
            person_data = self._load_image_data(self.test_person_path)
            cloth_data = self._load_image_data(self.test_cloth_path)
            
            if not person_data or not cloth_data:
                return GitHubInferenceResult(
                    step_name=step_name,
                    success=False,
                    inference_time=time.time() - start_time,
                    memory_used_mb=0.0,
                    error_message="테스트 이미지 로딩 실패"
                )
            
            # GitHub Step별 입력 데이터 구성
            input_data = self._prepare_github_step_input(step_name, person_data, cloth_data)
            
            print(f"  📊 입력 데이터: {list(input_data.keys())}")
            
            # 모델 로딩 확인
            model_loaded = self._check_model_loading(step_instance)
            
            # 실제 AI 추론 실행
            result = None
            ai_inference_executed = False
            
            if hasattr(step_instance, 'process'):
                try:
                    if asyncio.iscoroutinefunction(step_instance.process):
                        # 비동기 실행
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(step_instance.process(**input_data))
                        loop.close()
                    else:
                        # 동기 실행
                        result = step_instance.process(**input_data)
                    
                    ai_inference_executed = True
                    print(f"  🧠 AI 추론 실행 완료")
                    
                except Exception as process_e:
                    print(f"  ⚠️ process 실행 실패: {process_e}")
                    # 폴백: 다른 메서드 시도
                    if hasattr(step_instance, 'run_inference'):
                        try:
                            result = step_instance.run_inference(input_data)
                            ai_inference_executed = True
                            print(f"  🧠 run_inference로 AI 추론 실행 완료")
                        except Exception as run_e:
                            print(f"  ❌ run_inference 실행 실패: {run_e}")
            
            inference_time = time.time() - start_time
            memory_after = self._monitor_memory()
            memory_used = memory_after - memory_before
            
            # 결과 분석
            success = result is not None and ai_inference_executed
            result_saved = False
            output_path = None
            confidence_score = None
            
            if success and isinstance(result, dict):
                # 결과 저장
                output_path = self._save_github_result(step_name, result)
                result_saved = output_path is not None
                
                # 신뢰도 점수 추출
                confidence_score = result.get('confidence', result.get('confidence_score'))
                
                print(f"  ✅ GitHub 추론 성공: {inference_time:.2f}s, {memory_used:.1f}MB")
                if confidence_score:
                    print(f"  📊 신뢰도: {confidence_score:.3f}")
            else:
                print(f"  ❌ GitHub 추론 실패 또는 결과 없음")
            
            return GitHubInferenceResult(
                step_name=step_name,
                success=success,
                inference_time=inference_time,
                memory_used_mb=memory_used,
                model_loaded=model_loaded,
                ai_inference_executed=ai_inference_executed,
                result_saved=result_saved,
                output_path=output_path,
                confidence_score=confidence_score,
                step_factory_used=step_factory_used,
                di_container_used=di_container_used,
                basestepmixin_compatible=basestepmixin_compatible
            )
            
        except Exception as e:
            inference_time = time.time() - start_time
            memory_after = self._monitor_memory()
            
            print(f"  ❌ {step_name} GitHub 추론 오류: {e}")
            
            return GitHubInferenceResult(
                step_name=step_name,
                success=False,
                inference_time=inference_time,
                memory_used_mb=memory_after - memory_before,
                error_message=str(e)
            )
    
    def _create_step_directly(self, step_name: str):
        """직접 Step 생성 (폴백)"""
        try:
            # GitHub Step 매핑
            step_mappings = {
                "step_01_human_parsing": ("app.ai_pipeline.steps.step_01_human_parsing", "HumanParsingStep"),
                "step_02_pose_estimation": ("app.ai_pipeline.steps.step_02_pose_estimation", "PoseEstimationStep"),
                "step_03_cloth_segmentation": ("app.ai_pipeline.steps.step_03_cloth_segmentation", "ClothSegmentationStep"),
                "step_04_geometric_matching": ("app.ai_pipeline.steps.step_04_geometric_matching", "GeometricMatchingStep"),
                "step_05_cloth_warping": ("app.ai_pipeline.steps.step_05_cloth_warping", "ClothWarpingStep"),
                "step_06_virtual_fitting": ("app.ai_pipeline.steps.step_06_virtual_fitting", "VirtualFittingStep"),
                "step_07_post_processing": ("app.ai_pipeline.steps.step_07_post_processing", "PostProcessingStep"),
                "step_08_quality_assessment": ("app.ai_pipeline.steps.step_08_quality_assessment", "QualityAssessmentStep")
            }
            
            if step_name not in step_mappings:
                return None
            
            module_path, class_name = step_mappings[step_name]
            
            import importlib
            module = importlib.import_module(module_path)
            step_class = getattr(module, class_name)
            
            # BaseStepMixin 호환 kwargs (안전한 초기화)
            step_kwargs = {
                'step_name': step_name,
                'device': 'cpu'
            }
            
            # 선택적 파라미터 추가 (안전하게)
            if self.model_loader:
                try:
                    step_instance = step_class(**step_kwargs)
                    if hasattr(step_instance, 'set_model_loader'):
                        step_instance.set_model_loader(self.model_loader)
                    return step_instance
                except Exception as e:
                    print(f"  ⚠️ ModelLoader 주입 실패, 기본 생성 시도: {e}")
            
            return step_class(**step_kwargs)
            
        except Exception as e:
            print(f"  ❌ 직접 Step 생성 실패: {e}")
            return None
    
    def _check_basestepmixin_compatibility(self, step_instance) -> bool:
        """BaseStepMixin v19.2 호환성 검증"""
        try:
            mro_names = [cls.__name__ for cls in step_instance.__class__.__mro__]
            return 'BaseStepMixin' in mro_names
        except:
            return False
    
    def _check_model_loading(self, step_instance) -> bool:
        """모델 로딩 상태 확인"""
        try:
            # 다양한 모델 로딩 확인 방법
            if hasattr(step_instance, 'models_loaded'):
                return getattr(step_instance, 'models_loaded', False)
            elif hasattr(step_instance, 'model_loader') and step_instance.model_loader:
                return True
            elif hasattr(step_instance, '_models') and step_instance._models:
                return len(step_instance._models) > 0
            else:
                return False
        except:
            return False
    
    def _prepare_github_step_input(self, step_name: str, person_data: Dict, cloth_data: Dict) -> Dict[str, Any]:
        """GitHub Step별 입력 데이터 준비"""
        base_input = {
            'person_image': person_data['pil_image'],
            'clothing_image': cloth_data['pil_image'],
            'user_id': 'github_test_user',
            'session_id': f'github_test_session_{int(time.time())}',
            'github_test_mode': True
        }
        
        # Step별 특화 입력 (GitHub 표준)
        if 'human_parsing' in step_name:
            return {
                **base_input,
                'image': person_data['pil_image'],
                'parsing_type': 'full_body',
                'output_format': 'mask'
            }
        elif 'pose' in step_name:
            return {
                **base_input,
                'image': person_data['pil_image'],
                'keypoint_format': 'coco_17',
                'confidence_threshold': 0.3
            }
        elif 'segmentation' in step_name:
            return {
                **base_input,
                'target_image': cloth_data['pil_image'],
                'segmentation_type': 'cloth',
                'use_sam': True
            }
        elif 'geometric' in step_name:
            return {
                **base_input,
                'source_cloth': cloth_data['pil_image'],
                'person_mask': person_data['numpy_array'],
                'matching_algorithm': 'tps'
            }
        elif 'warping' in step_name:
            return {
                **base_input,
                'cloth_image': cloth_data['pil_image'],
                'pose_keypoints': person_data['numpy_array'],
                'warping_method': 'dense_flow'
            }
        elif 'virtual_fitting' in step_name:
            return {
                **base_input,
                'cloth_type': 'upper_body',
                'fitting_mode': 'realistic',
                'quality_level': 'high'
            }
        elif 'post_processing' in step_name:
            return {
                **base_input,
                'enhance_quality': True,
                'remove_artifacts': True
            }
        elif 'quality' in step_name:
            return {
                **base_input,
                'assessment_metrics': ['fid', 'lpips', 'ssim'],
                'reference_image': person_data['pil_image']
            }
        else:
            return base_input
    
    def _save_github_result(self, step_name: str, result: Dict[str, Any]) -> Optional[str]:
        """GitHub 추론 결과 저장"""
        try:
            output_dir = self.results_dir / f"github_{step_name}"
            output_dir.mkdir(exist_ok=True)
            
            timestamp = int(time.time())
            
            # 이미지 결과 저장
            if 'result' in result:
                result_data = result['result']
                
                # Base64 이미지
                if isinstance(result_data, str) and ('base64' in result_data or result_data.startswith('data:image')):
                    try:
                        if ',' in result_data:
                            result_data = result_data.split(',')[1]
                        image_data = base64.b64decode(result_data)
                        
                        output_path = output_dir / f"github_result_{timestamp}.jpg"
                        with open(output_path, 'wb') as f:
                            f.write(image_data)
                        
                        return str(output_path)
                    except Exception as e:
                        print(f"  ⚠️ Base64 이미지 저장 실패: {e}")
                
                # NumPy 배열
                elif hasattr(result_data, 'shape'):
                    try:
                        if len(result_data.shape) >= 2:
                            if result_data.max() <= 1.0:
                                result_data = (result_data * 255).astype(np.uint8)
                            
                            if len(result_data.shape) == 3:
                                image = Image.fromarray(result_data)
                            elif len(result_data.shape) == 2:
                                image = Image.fromarray(result_data, mode='L')
                            else:
                                return None
                            
                            output_path = output_dir / f"github_result_{timestamp}.jpg"
                            image.save(output_path, quality=95)
                            
                            return str(output_path)
                    except Exception as e:
                        print(f"  ⚠️ NumPy 이미지 저장 실패: {e}")
            
            # JSON 메타데이터 저장
            json_path = output_dir / f"github_result_{timestamp}.json"
            try:
                json_result = {}
                for key, value in result.items():
                    if hasattr(value, 'tolist'):
                        json_result[key] = value.tolist()
                    elif hasattr(value, 'shape'):
                        json_result[key] = f"tensor_shape_{value.shape}"
                    elif isinstance(value, (str, int, float, bool, list, dict)):
                        json_result[key] = value
                    else:
                        json_result[key] = str(value)
                
                json_result['github_test_metadata'] = {
                    'timestamp': timestamp,
                    'step_name': step_name,
                    'test_version': '4.0'
                }
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_result, f, indent=2, ensure_ascii=False)
                
                return str(json_path)
            except Exception as e:
                print(f"  ⚠️ JSON 저장 실패: {e}")
            
            return None
            
        except Exception as e:
            print(f"  ⚠️ GitHub 결과 저장 실패: {e}")
            return None
    
    def run_github_inference_tests(self) -> List[GitHubInferenceResult]:
        """GitHub 추론 테스트 실행"""
        print(f"\n🧠 GitHub 실제 AI 추론 테스트 시작 (모드: {self.test_type.value})")
        print("=" * 60)
        
        # 테스트할 Step들 선택 (GitHub 표준)
        if self.test_type == GitHubInferenceTestType.QUICK:
            test_steps = ["step_01_human_parsing"]
        elif self.test_type == GitHubInferenceTestType.STANDARD:
            test_steps = [
                "step_01_human_parsing",
                "step_03_cloth_segmentation", 
                "step_06_virtual_fitting"
            ]
        elif self.test_type == GitHubInferenceTestType.FULL_PIPELINE:
            test_steps = [
                "step_01_human_parsing",
                "step_02_pose_estimation",
                "step_03_cloth_segmentation",
                "step_04_geometric_matching",
                "step_05_cloth_warping",
                "step_06_virtual_fitting"
            ]
        else:  # STRESS_TEST
            test_steps = [
                "step_01_human_parsing",
                "step_02_pose_estimation",
                "step_03_cloth_segmentation",
                "step_04_geometric_matching",
                "step_05_cloth_warping",
                "step_06_virtual_fitting",
                "step_07_post_processing",
                "step_08_quality_assessment"
            ]
        
        results = []
        
        for i, step_name in enumerate(test_steps, 1):
            print(f"\n[{i}/{len(test_steps)}] GitHub {step_name} 추론 테스트")
            
            result = self.test_github_step_inference(step_name)
            results.append(result)
            self.inference_results.append(result)
            
            if result.success:
                self.total_inference_time += result.inference_time
                print(f"  ✅ 성공: {result.inference_time:.2f}s")
                if result.result_saved:
                    print(f"  💾 결과 저장: {result.output_path}")
            else:
                print(f"  ❌ 실패: {result.error_message}")
            
            # 메모리 정리 (M3 Max 최적화)
            gc.collect()
            if hasattr(gc, 'set_debug'):
                gc.set_debug(0)
        
        return results
    
    def generate_github_report(self):
        """GitHub 추론 테스트 리포트 생성"""
        print("\n" + "=" * 80)
        print("🧠 GitHub MyCloset AI 실제 추론 테스트 결과 리포트")
        print("=" * 80)
        
        if not self.inference_results:
            print("❌ 테스트 결과가 없습니다.")
            return
        
        # 통계 계산
        total_tests = len(self.inference_results)
        successful_tests = sum(1 for r in self.inference_results if r.success)
        success_rate = (successful_tests / total_tests) * 100
        
        avg_inference_time = self.total_inference_time / successful_tests if successful_tests > 0 else 0
        total_memory_used = sum(r.memory_used_mb for r in self.inference_results)
        
        # GitHub 특화 통계
        step_factory_used_count = sum(1 for r in self.inference_results if r.step_factory_used)
        di_container_used_count = sum(1 for r in self.inference_results if r.di_container_used)
        basestepmixin_compatible_count = sum(1 for r in self.inference_results if r.basestepmixin_compatible)
        model_loaded_count = sum(1 for r in self.inference_results if r.model_loaded)
        ai_inference_count = sum(1 for r in self.inference_results if r.ai_inference_executed)
        
        print(f"📊 GitHub MyCloset AI 전체 통계:")
        print(f"   프로젝트: {self.project_root}")
        print(f"   백엔드: {self.backend_root}")
        print(f"   테스트 모드: {self.test_type.value.upper()}")
        print(f"   총 테스트: {total_tests}개")
        print(f"   성공: {successful_tests}개")
        print(f"   실패: {total_tests - successful_tests}개")
        print(f"   성공률: {success_rate:.1f}%")
        print(f"   총 추론 시간: {self.total_inference_time:.2f}초")
        print(f"   평균 추론 시간: {avg_inference_time:.2f}초/Step")
        print(f"   총 메모리 사용: {total_memory_used:.1f}MB")
        print(f"   피크 메모리: {self.peak_memory_mb:.1f}MB")
        
        print(f"\n🔧 GitHub 컴포넌트 사용 통계:")
        print(f"   StepFactory 사용: {step_factory_used_count}/{total_tests}개")
        print(f"   DI Container 사용: {di_container_used_count}/{total_tests}개")
        print(f"   BaseStepMixin 호환: {basestepmixin_compatible_count}/{total_tests}개")
        print(f"   모델 로딩 성공: {model_loaded_count}/{total_tests}개")
        print(f"   AI 추론 실행: {ai_inference_count}/{total_tests}개")
        
        # 상세 결과
        print(f"\n📋 GitHub Step별 상세 결과:")
        for result in self.inference_results:
            status = "✅" if result.success else "❌"
            time_info = f"({result.inference_time:.2f}s)" if result.inference_time > 0 else ""
            memory_info = f"[{result.memory_used_mb:.1f}MB]" if result.memory_used_mb > 0 else ""
            
            components = []
            if result.step_factory_used:
                components.append("StepFactory")
            if result.di_container_used:
                components.append("DI")
            if result.basestepmixin_compatible:
                components.append("BaseStepMixin")
            if result.model_loaded:
                components.append("Model")
            if result.ai_inference_executed:
                components.append("AI")
            
            component_info = f"[{'/'.join(components)}]" if components else "[직접생성]"
            
            print(f"  {status} {result.step_name}: ", end="")
            
            if result.success:
                confidence_info = f"신뢰도: {result.confidence_score:.3f}" if result.confidence_score else "추론 완료"
                output_info = "결과 저장됨" if result.result_saved else "메타데이터만"
                print(f"{confidence_info}, {output_info} {component_info} {time_info} {memory_info}")
            else:
                print(f"{result.error_message} {component_info} {time_info}")
        
        # 저장된 결과 파일들
        saved_results = [r for r in self.inference_results if r.result_saved]
        if saved_results:
            print(f"\n💾 GitHub 저장된 결과 파일:")
            for result in saved_results:
                if result.output_path:
                    print(f"   {result.step_name}: {result.output_path}")
        
        # GitHub CI/CD 호환성 검증
        github_compatibility_score = self._calculate_github_compatibility()
        
        # 최종 결론
        print(f"\n🎯 GitHub MyCloset AI 최종 결론:")
        if success_rate >= 90 and github_compatibility_score >= 80:
            print("   🚀 MyCloset AI가 GitHub 환경에서 완벽하게 실제 추론을 수행합니다!")
            print("   🌟 모든 Step이 GitHub 표준에 맞춰 정상적으로 AI 추론을 실행하고 있습니다.")
            print("   🔧 StepFactory, DI Container, BaseStepMixin 패턴이 완벽하게 작동합니다.")
        elif success_rate >= 70 and github_compatibility_score >= 60:
            print("   ✅ MyCloset AI가 GitHub 환경에서 대부분의 추론을 성공적으로 수행합니다!")
            print("   🔧 일부 Step에 문제가 있지만 핵심 기능은 GitHub 표준에 맞춰 작동합니다.")
        elif success_rate >= 50:
            print("   ⚠️ MyCloset AI GitHub 추론에 부분적 문제가 있습니다.")
            print("   🛠️ 실패한 Step들의 GitHub 호환성과 모델 로딩 상태를 확인하세요.")
        else:
            print("   ❌ MyCloset AI GitHub 추론에 심각한 문제가 있습니다.")
            print("   🚨 GitHub 프로젝트 구조, 모델 파일, conda 환경을 전면 점검하세요.")
        
        print(f"\n📈 GitHub 호환성 점수: {github_compatibility_score:.1f}/100")
        print("=" * 80)
    
    def _calculate_github_compatibility(self) -> float:
        """GitHub 호환성 점수 계산"""
        if not self.inference_results:
            return 0.0
        
        total_tests = len(self.inference_results)
        scores = []
        
        for result in self.inference_results:
            score = 0
            
            # 기본 성공 (40점)
            if result.success:
                score += 40
            
            # GitHub 컴포넌트 사용 (각 10점)
            if result.step_factory_used:
                score += 10
            if result.di_container_used:
                score += 10
            if result.basestepmixin_compatible:
                score += 10
            
            # AI 기능 (각 10점)
            if result.model_loaded:
                score += 10
            if result.ai_inference_executed:
                score += 10
            
            # 결과 저장 (10점)
            if result.result_saved:
                score += 10
            
            scores.append(score)
        
        return sum(scores) / total_tests

def quick_github_test():
    """빠른 GitHub AI 추론 테스트"""
    print("⚡ GitHub MyCloset AI 빠른 추론 테스트...")
    tester = GitHubAIInferenceTester(GitHubInferenceTestType.QUICK)
    results = tester.run_github_inference_tests()
    tester.generate_github_report()
    return len([r for r in results if r.success]) > 0

def standard_github_test():
    """표준 GitHub AI 추론 테스트"""
    print("🔍 GitHub MyCloset AI 표준 추론 테스트...")
    tester = GitHubAIInferenceTester(GitHubInferenceTestType.STANDARD)
    results = tester.run_github_inference_tests()
    tester.generate_github_report()
    return results

def full_pipeline_github_test():
    """전체 파이프라인 GitHub 추론 테스트"""
    print("🚀 GitHub MyCloset AI 전체 파이프라인 추론 테스트...")
    tester = GitHubAIInferenceTester(GitHubInferenceTestType.FULL_PIPELINE)
    results = tester.run_github_inference_tests()
    tester.generate_github_report()
    return results

def main():
    """메인 실행 함수"""
    print("🔥 GitHub MyCloset AI 실제 AI 추론 테스터 v4.0")
    print("=" * 60)
    print("GitHub 호환 추론 테스트 모드를 선택하세요:")
    print("1. 빠른 테스트 (1개 Step - Human Parsing)")
    print("2. 표준 테스트 (3개 핵심 Step)")
    print("3. 전체 파이프라인 (6개 Step)")
    print("4. 스트레스 테스트 (8개 전체 Step)")
    
    choice = input("선택 (1/2/3/4): ").strip()
    
    if choice == "1":
        return quick_github_test()
    elif choice == "3":
        return full_pipeline_github_test()
    elif choice == "4":
        tester = GitHubAIInferenceTester(GitHubInferenceTestType.STRESS_TEST)
        results = tester.run_github_inference_tests()
        tester.generate_github_report()
        return results
    else:
        return standard_github_test()

if __name__ == "__main__":
    main()