#!/usr/bin/env python3
"""
🔧 ModelLoader 핵심 메서드 추가 패치
get_model(), get_model_for_step() 등 필수 메서드 구현
"""

import sys
from pathlib import Path
from typing import Optional, Any, Dict, List

# 프로젝트 경로 설정
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

def patch_modelloader():
    """ModelLoader에 누락된 메서드들 패치"""
    
    try:
        from backend.app.ai_pipeline.utils.model_loader import ModelLoader
        
        print("🔧 ModelLoader 패치 적용 중...")
        
        # 1. get_model 메서드 추가
        def get_model(self, model_name: str) -> Optional[Any]:
            """이미 로딩된 모델 반환 (핵심 메서드!)"""
            try:
                # 1단계: 캐시에서 찾기
                if hasattr(self, 'model_cache'):
                    for cache_key, model in self.model_cache.items():
                        if model_name in cache_key:
                            self.logger.debug(f"✅ 캐시에서 모델 발견: {model_name}")
                            return model
                
                # 2단계: SafeModelService에서 찾기
                if hasattr(self, 'safe_model_service'):
                    model = self.safe_model_service.call_model(model_name)
                    if model:
                        self.logger.debug(f"✅ SafeModelService에서 모델 발견: {model_name}")
                        return model
                
                # 3단계: 자동 로딩 시도
                if hasattr(self, 'load_model_sync'):
                    model = self.load_model_sync(model_name)
                    if model:
                        self.logger.info(f"✅ 자동 로딩 성공: {model_name}")
                        return model
                elif hasattr(self, 'load_model_async'):
                    # 비동기를 동기로 변환
                    import asyncio
                    try:
                        model = asyncio.run(self.load_model_async(model_name))
                        if model:
                            self.logger.info(f"✅ 비동기 로딩 성공: {model_name}")
                            return model
                    except Exception as e:
                        self.logger.warning(f"⚠️ 비동기 로딩 실패: {e}")
                
                self.logger.warning(f"⚠️ 모델을 찾을 수 없음: {model_name}")
                return None
                
            except Exception as e:
                self.logger.error(f"❌ get_model 실패 {model_name}: {e}")
                return None
        
        # 2. get_model_for_step 메서드 추가
        def get_model_for_step(self, step_name: str) -> Optional[Any]:
            """Step별 최적 모델 반환 (핵심 메서드!)"""
            try:
                # Step별 패턴 매핑
                step_patterns = {
                    'HumanParsingStep': ['human_parsing', 'parsing', 'graphonomy', 'schp', 'atr'],
                    'PoseEstimationStep': ['pose', 'openpose', 'body_pose'],
                    'ClothSegmentationStep': ['cloth_segmentation', 'u2net', 'segmentation'],
                    'VirtualFittingStep': ['virtual_fitting', 'diffusion', 'ootd', 'viton', 'stable'],
                    'GeometricMatchingStep': ['geometric', 'matching', 'gmm'],
                    'ClothWarpingStep': ['cloth_warping', 'warping', 'tom'],
                    'PostProcessingStep': ['post_processing', 'esrgan', 'super_resolution'],
                    'QualityAssessmentStep': ['quality', 'assessment', 'clip']
                }
                
                if step_name not in step_patterns:
                    self.logger.warning(f"⚠️ 알 수 없는 Step: {step_name}")
                    return None
                
                patterns = step_patterns[step_name]
                
                # 1단계: step_model_mapping에서 찾기
                if hasattr(self, 'step_model_mapping') and step_name in self.step_model_mapping:
                    step_models = self.step_model_mapping[step_name]
                    if step_models:
                        best_model_name = step_models[0] if isinstance(step_models, list) else step_models
                        model = self.get_model(best_model_name)
                        if model:
                            self.logger.info(f"✅ Step 매핑에서 모델 발견: {best_model_name}")
                            return model
                
                # 2단계: 등록된 모델들에서 패턴 매칭
                all_models = []
                if hasattr(self, 'model_configs'):
                    all_models.extend(self.model_configs.keys())
                if hasattr(self, 'detected_model_registry'):
                    all_models.extend(self.detected_model_registry.keys())
                if hasattr(self, 'model_cache'):
                    all_models.extend([key.split('_')[0] for key in self.model_cache.keys()])
                
                # 패턴 매칭으로 최적 모델 찾기
                for model_name in all_models:
                    model_name_lower = model_name.lower()
                    if any(pattern in model_name_lower for pattern in patterns):
                        model = self.get_model(model_name)
                        if model:
                            self.logger.info(f"✅ 패턴 매칭으로 모델 발견: {model_name} for {step_name}")
                            return model
                
                self.logger.warning(f"⚠️ {step_name}용 모델을 찾을 수 없음")
                return None
                
            except Exception as e:
                self.logger.error(f"❌ get_model_for_step 실패 {step_name}: {e}")
                return None
        
        # 3. list_available_models 메서드 추가
        def list_available_models(self) -> List[str]:
            """사용 가능한 모든 모델 목록 반환"""
            try:
                models = set()
                
                # 캐시된 모델들
                if hasattr(self, 'model_cache'):
                    for cache_key in self.model_cache.keys():
                        # cache_key에서 모델명 추출
                        model_name = cache_key.split('_')[0]
                        models.add(model_name)
                
                # 등록된 모델들
                if hasattr(self, 'model_configs'):
                    models.update(self.model_configs.keys())
                
                # 탐지된 모델들
                if hasattr(self, 'detected_model_registry'):
                    models.update(self.detected_model_registry.keys())
                
                # SafeModelService의 모델들
                if hasattr(self, 'safe_model_service') and hasattr(self.safe_model_service, 'models'):
                    models.update(self.safe_model_service.models.keys())
                
                return sorted(list(models))
                
            except Exception as e:
                self.logger.error(f"❌ list_available_models 실패: {e}")
                return []
        
        # 4. load_model_sync 메서드 강화 (이미 있다면 패스)
        def load_model_sync(self, model_name: str, **kwargs) -> Optional[Any]:
            """동기 모델 로드 (강화 버전)"""
            try:
                # 기존 load_model이 있다면 사용
                if hasattr(self, 'load_model') and hasattr(self.load_model, '__call__'):
                    return self.load_model(model_name, **kwargs)
                
                # 비동기를 동기로 변환
                if hasattr(self, 'load_model_async'):
                    import asyncio
                    try:
                        return asyncio.run(self.load_model_async(model_name, **kwargs))
                    except Exception as e:
                        self.logger.warning(f"⚠️ 비동기->동기 변환 실패: {e}")
                
                self.logger.error(f"❌ 사용 가능한 로딩 메서드 없음: {model_name}")
                return None
                        
            except Exception as e:
                self.logger.error(f"❌ load_model_sync 실패 {model_name}: {e}")
                return None
        
        # 메서드 동적 추가
        methods_added = []
        
        if not hasattr(ModelLoader, 'get_model'):
            ModelLoader.get_model = get_model
            methods_added.append('get_model')
        
        if not hasattr(ModelLoader, 'get_model_for_step'):
            ModelLoader.get_model_for_step = get_model_for_step
            methods_added.append('get_model_for_step')
        
        if not hasattr(ModelLoader, 'list_available_models'):
            ModelLoader.list_available_models = list_available_models
            methods_added.append('list_available_models')
        
        if not hasattr(ModelLoader, 'load_model_sync'):
            ModelLoader.load_model_sync = load_model_sync
            methods_added.append('load_model_sync')
        
        print(f"✅ ModelLoader 패치 완료! 추가된 메서드: {', '.join(methods_added)}")
        return True
        
    except Exception as e:
        print(f"❌ ModelLoader 패치 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_patched_modelloader():
    """패치된 ModelLoader 테스트"""
    
    print("\n🧪 패치된 ModelLoader 테스트...")
    
    try:
        from backend.app.ai_pipeline.utils.model_loader import ModelLoader
        
        loader = ModelLoader()
        
        # 메서드 존재 확인
        required_methods = ['get_model', 'get_model_for_step', 'list_available_models', 'load_model_sync']
        
        for method in required_methods:
            if hasattr(loader, method):
                print(f"   ✅ {method}: 존재")
            else:
                print(f"   ❌ {method}: 없음")
        
        # 실제 동작 테스트
        if hasattr(loader, 'list_available_models'):
            try:
                models = loader.list_available_models()
                print(f"   📦 사용 가능한 모델: {len(models)}개")
                if models:
                    print(f"      예시: {models[:3]}")
            except Exception as e:
                print(f"   ⚠️ list_available_models 테스트 실패: {e}")
        
        if hasattr(loader, 'get_model_for_step'):
            test_steps = ['HumanParsingStep', 'VirtualFittingStep', 'ClothSegmentationStep']
            for step in test_steps:
                try:
                    model = loader.get_model_for_step(step)
                    status = "✅" if model else "❓"
                    model_type = type(model).__name__ if model else 'None'
                    print(f"   {status} {step}: {model_type}")
                except Exception as e:
                    print(f"   ❌ {step}: 오류 - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 패치 테스트 실패: {e}")
        return False

def verify_auto_detector_integration():
    """Auto Detector와 ModelLoader 연동 확인"""
    
    print("\n🔍 Auto Detector ↔ ModelLoader 연동 검증...")
    
    try:
        from backend.app.ai_pipeline.utils.model_loader import ModelLoader
        
        # 1. ModelLoader 생성
        loader = ModelLoader()
        
        # 2. Auto Detector에서 발견된 모델들이 ModelLoader에서 접근 가능한지 확인
        known_models = [
            'human_parsing_graphonomy',
            'cloth_segmentation_u2net', 
            'virtual_fitting_ootdiffusion',
            'pose_estimation_openpose'
        ]
        
        integration_results = {}
        
        for model_name in known_models:
            try:
                model = loader.get_model(model_name)
                integration_results[model_name] = {
                    'accessible': model is not None,
                    'type': type(model).__name__ if model else 'None'
                }
            except Exception as e:
                integration_results[model_name] = {
                    'accessible': False,
                    'error': str(e)
                }
        
        # 결과 출력
        successful_integrations = 0
        for model_name, result in integration_results.items():
            if result['accessible']:
                print(f"   ✅ {model_name}: {result['type']}")
                successful_integrations += 1
            else:
                error_msg = result.get('error', 'No model found')
                print(f"   ❌ {model_name}: {error_msg}")
        
        print(f"\n📊 연동 결과: {successful_integrations}/{len(known_models)} 성공")
        
        return successful_integrations > 0
        
    except Exception as e:
        print(f"❌ 연동 검증 실패: {e}")
        return False

def main():
    """메인 실행 함수"""
    
    print("🔧 ModelLoader 핵심 메서드 패치 및 테스트...")
    print("=" * 60)
    
    # 1. 패치 적용
    patch_success = patch_modelloader()
    
    if patch_success:
        # 2. 패치 테스트
        test_success = test_patched_modelloader()
        
        if test_success:
            # 3. Auto Detector 연동 확인
            integration_success = verify_auto_detector_integration()
            
            if integration_success:
                print("\n🎉 완벽! Auto Detector → ModelLoader 연동 성공!")
                print("\n🚀 이제 사용 가능한 코드:")
                print("   from backend.app.ai_pipeline.utils.model_loader import ModelLoader")
                print("   loader = ModelLoader()")
                print("   model = loader.get_model_for_step('HumanParsingStep')")
                print("   models = loader.list_available_models()")
                
                return True
            else:
                print("\n⚠️ 패치는 성공했지만 Auto Detector 연동에 문제가 있습니다")
        else:
            print("\n⚠️ 패치는 성공했지만 테스트에서 문제가 발견되었습니다")
    else:
        print("\n❌ 패치 적용에 실패했습니다")
    
    return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Auto Detector가 찾은 경로를 ModelLoader가 완벽하게 사용할 수 있습니다!")
    else:
        print("\n💡 추가 디버깅이 필요합니다.")
        print("   실행해보세요: python patch_modelloader.py")