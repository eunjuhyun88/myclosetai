#!/usr/bin/env python3
"""
🔥 MyCloset AI - 자동 프리미엄 모델 연동 스크립트 v2.0
===============================================================================
✅ 기존 Step 구현체들에 최고급 AI 모델 자동 연동
✅ ModelLoader와 완벽 호환
✅ conda 환경 우선 최적화
✅ M3 Max 128GB 완전 활용

실행: python auto_premium_model_integration.py
"""

import sys
import os
import asyncio
import logging
import torch
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def integrate_premium_models_to_existing_steps():
    """기존 Step 구현체들에 프리미엄 모델 자동 연동"""
    
    print("🔥 MyCloset AI Premium 모델 자동 연동 시작...")
    print("="*60)
    
    try:
        # 1. ModelLoader 가져오기
        from app.ai_pipeline.utils.model_loader import get_global_model_loader
        model_loader = get_global_model_loader()
        
        print("✅ ModelLoader 가져오기 성공")
        
        # 2. Premium 모델 선택기 가져오기
        sys.path.append('.')
        from premium_ai_model_mapping import PremiumAIModelSelector, PREMIUM_AI_MODELS_BY_STEP
        
        # M3 Max 128GB 메모리 활용
        selector = PremiumAIModelSelector(available_memory_gb=128.0)
        selected_models = selector.select_best_models_for_all_steps()
        
        print(f"✅ 프리미엄 모델 선택 완료: {len(selected_models)}개")
        
        # 3. 각 Step 구현체에 프리미엄 모델 연동
        integration_results = {}
        
        # Step 01: Human Parsing
        result = await integrate_step_01_premium(model_loader, selected_models)
        integration_results["HumanParsingStep"] = result
        
        # Step 02: Pose Estimation  
        result = await integrate_step_02_premium(model_loader, selected_models)
        integration_results["PoseEstimationStep"] = result
        
        # Step 03: Cloth Segmentation
        result = await integrate_step_03_premium(model_loader, selected_models)
        integration_results["ClothSegmentationStep"] = result
        
        # Step 06: Virtual Fitting (핵심!)
        result = await integrate_step_06_premium(model_loader, selected_models)
        integration_results["VirtualFittingStep"] = result
        
        # Step 08: Quality Assessment
        result = await integrate_step_08_premium(model_loader, selected_models)
        integration_results["QualityAssessmentStep"] = result
        
        # 결과 출력
        print("\n🎉 프리미엄 모델 연동 결과:")
        print("="*60)
        
        success_count = 0
        for step_name, result in integration_results.items():
            status = "✅" if result["success"] else "❌"
            print(f"{status} {step_name}: {result['message']}")
            if result["success"]:
                success_count += 1
                if "model_info" in result:
                    info = result["model_info"]
                    print(f"    📦 모델: {info['name']}")
                    print(f"    📊 파라미터: {info['parameters']:,}개")
                    print(f"    💾 메모리: {info['memory_gb']:.1f}GB")
        
        print(f"\n📈 총 연동 성공: {success_count}/{len(integration_results)}개")
        
        if success_count > 0:
            print("\n🚀 다음 단계: FastAPI 서버 실행")
            print("cd backend && python -m app.main")
        
        return integration_results
        
    except Exception as e:
        logger.error(f"❌ 프리미엄 모델 연동 실패: {e}")
        return {"error": str(e)}

async def integrate_step_01_premium(model_loader, selected_models):
    """Step 01 Human Parsing에 프리미엄 모델 연동"""
    try:
        if "HumanParsingStep" not in selected_models:
            return {"success": False, "message": "선택된 프리미엄 모델 없음"}
        
        premium_model = selected_models["HumanParsingStep"]
        
        print(f"\n🔄 Step 01 프리미엄 모델 연동: {premium_model.name}")
        
        # Step 01 구현체 가져오기
        from app.ai_pipeline.steps.step_01_human_parsing import HumanParsingStep
        
        # 프리미엄 모델 로드
        model_path = premium_model.file_path
        if not os.path.exists(model_path):
            return {"success": False, "message": f"모델 파일 없음: {model_path}"}
        
        # SCHP 모델 로딩 (66M 파라미터)
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # ModelLoader에 등록
        model_loader.register_premium_model(
            step_class="HumanParsingStep",
            model_name=premium_model.name,
            model_checkpoint=checkpoint,
            model_info={
                "parameters": premium_model.parameters,
                "performance_score": premium_model.performance_score,
                "memory_requirement_gb": premium_model.memory_requirement_gb
            }
        )
        
        return {
            "success": True,
            "message": "SCHP 프리미엄 모델 연동 성공",
            "model_info": {
                "name": premium_model.name,
                "parameters": premium_model.parameters,
                "memory_gb": premium_model.memory_requirement_gb
            }
        }
        
    except Exception as e:
        return {"success": False, "message": f"연동 실패: {e}"}

async def integrate_step_02_premium(model_loader, selected_models):
    """Step 02 Pose Estimation에 프리미엄 모델 연동"""
    try:
        if "PoseEstimationStep" not in selected_models:
            return {"success": False, "message": "선택된 프리미엄 모델 없음"}
        
        premium_model = selected_models["PoseEstimationStep"]
        
        print(f"\n🔄 Step 02 프리미엄 모델 연동: {premium_model.name}")
        
        # OpenPose 모델 로딩 (52M 파라미터)
        model_path = premium_model.file_path
        if not os.path.exists(model_path):
            return {"success": False, "message": f"모델 파일 없음: {model_path}"}
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # ModelLoader에 등록
        model_loader.register_premium_model(
            step_class="PoseEstimationStep", 
            model_name=premium_model.name,
            model_checkpoint=checkpoint,
            model_info={
                "parameters": premium_model.parameters,
                "performance_score": premium_model.performance_score,
                "keypoints": 25  # COCO 25개 키포인트
            }
        )
        
        return {
            "success": True,
            "message": "OpenPose 프리미엄 모델 연동 성공",
            "model_info": {
                "name": premium_model.name,
                "parameters": premium_model.parameters,
                "memory_gb": premium_model.memory_requirement_gb
            }
        }
        
    except Exception as e:
        return {"success": False, "message": f"연동 실패: {e}"}

async def integrate_step_03_premium(model_loader, selected_models):
    """Step 03 Cloth Segmentation에 프리미엄 모델 연동"""
    try:
        if "ClothSegmentationStep" not in selected_models:
            return {"success": False, "message": "선택된 프리미엄 모델 없음"}
        
        premium_model = selected_models["ClothSegmentationStep"]
        
        print(f"\n🔄 Step 03 프리미엄 모델 연동: {premium_model.name}")
        
        # SAM ViT-H 모델 로딩 (641M 파라미터!)
        model_path = premium_model.file_path
        if not os.path.exists(model_path):
            return {"success": False, "message": f"모델 파일 없음: {model_path}"}
        
        # SAM 특화 로딩
        try:
            # SAM 패키지가 있으면 사용
            from segment_anything import sam_model_registry, SamPredictor
            sam_model = sam_model_registry["vit_h"](checkpoint=model_path)
            sam_predictor = SamPredictor(sam_model)
            
            # ModelLoader에 등록
            model_loader.register_premium_model(
                step_class="ClothSegmentationStep",
                model_name=premium_model.name, 
                model_checkpoint={"sam_model": sam_model, "sam_predictor": sam_predictor},
                model_info={
                    "parameters": premium_model.parameters,
                    "performance_score": premium_model.performance_score,
                    "model_type": "SAM_ViT_H"
                }
            )
            
        except ImportError:
            # SAM 패키지 없으면 일반 로딩
            checkpoint = torch.load(model_path, map_location='cpu')
            model_loader.register_premium_model(
                step_class="ClothSegmentationStep",
                model_name=premium_model.name,
                model_checkpoint=checkpoint,
                model_info={
                    "parameters": premium_model.parameters,
                    "performance_score": premium_model.performance_score
                }
            )
        
        return {
            "success": True,
            "message": "SAM ViT-H 프리미엄 모델 연동 성공",
            "model_info": {
                "name": premium_model.name,
                "parameters": premium_model.parameters,
                "memory_gb": premium_model.memory_requirement_gb
            }
        }
        
    except Exception as e:
        return {"success": False, "message": f"연동 실패: {e}"}

async def integrate_step_06_premium(model_loader, selected_models):
    """Step 06 Virtual Fitting에 프리미엄 모델 연동 (핵심!)"""
    try:
        if "VirtualFittingStep" not in selected_models:
            return {"success": False, "message": "선택된 프리미엄 모델 없음"}
        
        premium_model = selected_models["VirtualFittingStep"]
        
        print(f"\n🔄 Step 06 프리미엄 모델 연동: {premium_model.name}")
        print(f"    🔥 핵심 가상피팅 모델 - {premium_model.parameters:,} 파라미터!")
        
        # OOTDiffusion HD 모델 로딩 (859M 파라미터!)
        model_path = premium_model.file_path
        model_dir = os.path.dirname(model_path)
        
        if not os.path.exists(model_path):
            return {"success": False, "message": f"모델 파일 없음: {model_path}"}
        
        try:
            # Diffusers로 UNet 로딩
            from diffusers import UNet2DConditionModel
            unet_model = UNet2DConditionModel.from_pretrained(
                model_dir,
                torch_dtype=torch.float16,  # M3 Max 메모리 최적화
                variant="fp16",
                use_safetensors=True
            )
            
            # ModelLoader에 등록
            model_loader.register_premium_model(
                step_class="VirtualFittingStep",
                model_name=premium_model.name,
                model_checkpoint={"unet": unet_model},
                model_info={
                    "parameters": premium_model.parameters,
                    "performance_score": premium_model.performance_score,
                    "model_type": "OOTDiffusion_HD",
                    "resolution": "1024px"
                }
            )
            
        except ImportError:
            # Diffusers 없으면 safetensors로 직접 로딩
            import safetensors.torch
            checkpoint = safetensors.torch.load_file(model_path)
            
            model_loader.register_premium_model(
                step_class="VirtualFittingStep",
                model_name=premium_model.name,
                model_checkpoint=checkpoint,
                model_info={
                    "parameters": premium_model.parameters,
                    "performance_score": premium_model.performance_score
                }
            )
        
        return {
            "success": True,
            "message": "OOTDiffusion HD 프리미엄 모델 연동 성공",
            "model_info": {
                "name": premium_model.name,
                "parameters": premium_model.parameters,
                "memory_gb": premium_model.memory_requirement_gb
            }
        }
        
    except Exception as e:
        return {"success": False, "message": f"연동 실패: {e}"}

async def integrate_step_08_premium(model_loader, selected_models):
    """Step 08 Quality Assessment에 프리미엄 모델 연동"""
    try:
        if "QualityAssessmentStep" not in selected_models:
            return {"success": False, "message": "선택된 프리미엄 모델 없음"}
        
        premium_model = selected_models["QualityAssessmentStep"]
        
        print(f"\n🔄 Step 08 프리미엄 모델 연동: {premium_model.name}")
        
        # CLIP ViT-L 모델 로딩 (782M 파라미터!)
        model_path = premium_model.file_path
        if not os.path.exists(model_path):
            return {"success": False, "message": f"모델 파일 없음: {model_path}"}
        
        try:
            # OpenCLIP으로 로딩
            import open_clip
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14',
                pretrained=model_path
            )
            
            # ModelLoader에 등록
            model_loader.register_premium_model(
                step_class="QualityAssessmentStep",
                model_name=premium_model.name,
                model_checkpoint={
                    "clip_model": clip_model,
                    "clip_preprocess": clip_preprocess
                },
                model_info={
                    "parameters": premium_model.parameters,
                    "performance_score": premium_model.performance_score,
                    "model_type": "CLIP_ViT_L"
                }
            )
            
        except ImportError:
            # OpenCLIP 없으면 일반 로딩
            checkpoint = torch.load(model_path, map_location='cpu')
            model_loader.register_premium_model(
                step_class="QualityAssessmentStep",
                model_name=premium_model.name,
                model_checkpoint=checkpoint,
                model_info={
                    "parameters": premium_model.parameters,
                    "performance_score": premium_model.performance_score
                }
            )
        
        return {
            "success": True,
            "message": "CLIP ViT-L 프리미엄 모델 연동 성공",
            "model_info": {
                "name": premium_model.name,
                "parameters": premium_model.parameters,
                "memory_gb": premium_model.memory_requirement_gb
            }
        }
        
    except Exception as e:
        return {"success": False, "message": f"연동 실패: {e}"}

# ModelLoader에 프리미엄 모델 등록 메서드 추가
def add_premium_model_registration_to_model_loader():
    """ModelLoader에 프리미엄 모델 등록 기능 추가"""
    
    registration_code = '''
def register_premium_model(self, step_class: str, model_name: str, model_checkpoint: Any, model_info: Dict[str, Any]):
    """프리미엄 모델을 ModelLoader에 등록"""
    try:
        # 프리미엄 모델 저장소 초기화
        if not hasattr(self, '_premium_models'):
            self._premium_models = {}
        
        if step_class not in self._premium_models:
            self._premium_models[step_class] = {}
        
        # 프리미엄 모델 등록
        self._premium_models[step_class][model_name] = {
            "checkpoint": model_checkpoint,
            "info": model_info,
            "loaded_at": time.time()
        }
        
        # available_models에도 추가
        self.available_models[model_name] = {
            "name": model_name,
            "file_path": "premium_model",
            "size_mb": model_info.get("memory_requirement_gb", 0) * 1024,
            "step_class": step_class,
            "priority": 100,  # 최고 우선순위
            "loaded": True,
            "premium": True
        }
        
        self.logger.info(f"✅ 프리미엄 모델 등록 성공: {model_name} ({step_class})")
        return True
        
    except Exception as e:
        self.logger.error(f"❌ 프리미엄 모델 등록 실패: {e}")
        return False

def get_premium_model(self, step_class: str, model_name: str = None):
    """등록된 프리미엄 모델 가져오기"""
    if not hasattr(self, '_premium_models') or step_class not in self._premium_models:
        return None
    
    step_models = self._premium_models[step_class]
    
    if model_name:
        return step_models.get(model_name, {}).get("checkpoint")
    else:
        # 첫 번째 프리미엄 모델 반환
        if step_models:
            first_model = next(iter(step_models.values()))
            return first_model.get("checkpoint")
    
    return None
'''
    
    print("🔧 ModelLoader에 프리미엄 모델 등록 기능 추가 코드:")
    print("="*50)
    print(registration_code)
    
    return registration_code

async def main():
    """메인 실행 함수"""
    print("🚀 MyCloset AI Premium 모델 자동 연동 시작!")
    
    # 1. ModelLoader에 프리미엄 기능 추가
    add_premium_model_registration_to_model_loader()
    
    # 2. 프리미엄 모델들 자동 연동
    results = await integrate_premium_models_to_existing_steps()
    
    # 3. 결과 출력
    if "error" not in results:
        success_count = sum(1 for r in results.values() if r.get("success", False))
        print(f"\n🎉 프리미엄 모델 연동 완료: {success_count}개 성공!")
        
        if success_count > 0:
            print("\n🔥 다음 단계: 실제 AI 시스템 실행")
            print("cd backend")
            print("python -m app.main")
    else:
        print(f"❌ 연동 실패: {results['error']}")

if __name__ == "__main__":
    # conda 환경 확인
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'none')
    print(f"🐍 현재 conda 환경: {conda_env}")
    
    if conda_env != 'mycloset-ai-clean':
        print("⚠️ 권장: conda activate mycloset-ai-clean")
    
    # 비동기 실행
    asyncio.run(main())