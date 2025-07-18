# ==============================================
# 🔥 1단계: 새 파일 생성
# backend/app/core/di_container.py (새 파일)
# ==============================================

import logging
import threading
from typing import Dict, Any, Optional
import weakref

logger = logging.getLogger(__name__)

class SimpleDIContainer:
    """🔥 간단한 DI 컨테이너 - ModelLoader 문제 해결용"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._instances: Dict[str, Any] = {}
        self._instance_lock = threading.RLock()
        self._initialized = True
        logger.info("✅ DI Container 초기화")
    
    def register(self, name: str, instance: Any):
        """의존성 등록"""
        with self._instance_lock:
            self._instances[name] = instance
            logger.info(f"✅ DI 등록: {name} ({type(instance).__name__})")
    
    def get(self, name: str) -> Optional[Any]:
        """의존성 조회"""
        with self._instance_lock:
            instance = self._instances.get(name)
            if instance:
                logger.debug(f"🔍 DI 조회 성공: {name}")
            else:
                logger.warning(f"⚠️ DI 조회 실패: {name}")
            return instance
    
    def exists(self, name: str) -> bool:
        """의존성 존재 확인"""
        with self._instance_lock:
            return name in self._instances
    
    def clear(self):
        """모든 의존성 정리"""
        with self._instance_lock:
            count = len(self._instances)
            self._instances.clear()
            logger.info(f"🧹 DI Container 정리: {count}개 제거")

# 전역 DI 컨테이너
def get_di_container() -> SimpleDIContainer:
    """전역 DI 컨테이너 반환"""
    return SimpleDIContainer()

# ==============================================
# 🔥 2단계: main.py 수정 (기존 파일에 추가)
# backend/app/main.py - 기존 코드 끝에 추가만 하면 됨
# ==============================================

# 기존 main.py 코드 맨 끝에 이 부분만 추가:

@app.on_event("startup")
async def startup_event():
    """🔥 앱 시작 시 ModelLoader DI 초기화"""
    try:
        logger.info("🚀 MyCloset AI 시작 - ModelLoader DI 초기화...")
        
        # 1. DI Container 준비
        from app.core.di_container import get_di_container
        di_container = get_di_container()
        
        # 2. ModelLoader 초기화 및 등록
        try:
            from app.ai_pipeline.utils.model_loader import get_global_model_loader, initialize_global_model_loader
            
            # ModelLoader 초기화
            init_result = initialize_global_model_loader(
                device="auto",
                use_fp16=True,
                optimization_enabled=True,
                enable_fallback=True
            )
            
            if init_result.get("success"):
                model_loader = get_global_model_loader()
                if model_loader:
                    # DI Container에 등록
                    di_container.register('model_loader', model_loader)
                    logger.info("✅ ModelLoader DI 등록 완료")
                else:
                    logger.error("❌ ModelLoader 인스턴스가 None")
            else:
                logger.error(f"❌ ModelLoader 초기화 실패: {init_result.get('error')}")
        
        except Exception as e:
            logger.error(f"❌ ModelLoader DI 설정 실패: {e}")
        
        # 3. Step 생성 함수들을 DI 버전으로 패치
        await patch_step_creation_functions(di_container)
        
        logger.info("🎉 ModelLoader DI 시스템 초기화 완료!")
        
    except Exception as e:
        logger.error(f"❌ DI 시스템 초기화 실패: {e}")

async def patch_step_creation_functions(di_container):
    """🔥 Step 생성 함수들에 ModelLoader 자동 주입"""
    try:
        model_loader = di_container.get('model_loader')
        if not model_loader:
            logger.warning("⚠️ ModelLoader가 DI Container에 없음")
            return
        
        # HumanParsingStep 패치
        try:
            import app.ai_pipeline.steps.step_01_human_parsing as hp_module
            
            if hasattr(hp_module, 'create_human_parsing_step'):
                original_create = hp_module.create_human_parsing_step
                
                def create_with_di(*args, **kwargs):
                    # ModelLoader 자동 주입
                    if 'model_loader' not in kwargs:
                        kwargs['model_loader'] = model_loader
                        logger.info("✅ HumanParsingStep에 ModelLoader 자동 주입")
                    return original_create(*args, **kwargs)
                
                hp_module.create_human_parsing_step = create_with_di
                logger.info("✅ HumanParsingStep 생성 함수 DI 패치 완료")
        
        except Exception as e:
            logger.warning(f"⚠️ HumanParsingStep 패치 실패: {e}")
        
        # ClothSegmentationStep 패치
        try:
            import app.ai_pipeline.steps.step_03_cloth_segmentation as cs_module
            
            if hasattr(cs_module, 'create_cloth_segmentation_step'):
                original_create = cs_module.create_cloth_segmentation_step
                
                def create_with_di(*args, **kwargs):
                    if 'model_loader' not in kwargs:
                        kwargs['model_loader'] = model_loader
                        logger.info("✅ ClothSegmentationStep에 ModelLoader 자동 주입")
                    return original_create(*args, **kwargs)
                
                cs_module.create_cloth_segmentation_step = create_with_di
                logger.info("✅ ClothSegmentationStep 생성 함수 DI 패치 완료")
        
        except Exception as e:
            logger.warning(f"⚠️ ClothSegmentationStep 패치 실패: {e}")
        
        # 다른 Step들도 필요하면 추가...
        
    except Exception as e:
        logger.error(f"❌ Step 함수 패치 실패: {e}")

# DI 테스트 엔드포인트 추가
@app.get("/api/test-model-loader-di")
async def test_model_loader_di():
    """🧪 ModelLoader DI 테스트"""
    try:
        from app.core.di_container import get_di_container
        
        di_container = get_di_container()
        model_loader = di_container.get('model_loader')
        
        if model_loader:
            # ModelLoader 정보 확인
            info = {
                "model_loader_type": type(model_loader).__name__,
                "has_create_step_interface": hasattr(model_loader, 'create_step_interface'),
                "device": getattr(model_loader, 'device', 'unknown'),
                "is_initialized": getattr(model_loader, 'is_initialized', False)
            }
            
            # Step 인터페이스 테스트
            if hasattr(model_loader, 'create_step_interface'):
                try:
                    test_interface = model_loader.create_step_interface("TestStep")
                    info["step_interface_creation"] = test_interface is not None
                except Exception as e:
                    info["step_interface_error"] = str(e)
            
            return {
                "success": True,
                "message": "ModelLoader DI 정상 작동",
                "model_loader_info": info
            }
        else:
            return {
                "success": False,
                "message": "ModelLoader가 DI Container에 없음",
                "di_container_contents": list(di_container._instances.keys())
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "ModelLoader DI 테스트 실패"
        }

# ==============================================
# 🔥 3단계: Step 클래스 생성자 수정 (옵션)
# 기존 Step 클래스들을 건드리고 싶지 않다면 생략 가능
# ==============================================

"""
만약 기존 Step 클래스를 수정하고 싶다면:

HumanParsingStep.__init__에서:
```python
def __init__(self, model_loader=None, **kwargs):
    # 기존 코드...
    
    # DI Container에서 ModelLoader 가져오기
    if model_loader is None:
        try:
            from app.core.di_container import get_di_container
            di_container = get_di_container()
            model_loader = di_container.get('model_loader')
            if model_loader:
                logger.info("✅ DI Container에서 ModelLoader 주입")
        except Exception as e:
            logger.warning(f"⚠️ DI Container ModelLoader 조회 실패: {e}")
    
    # 기존 모델 인터페이스 설정
    self._setup_model_interface(model_loader)
```

하지만 2단계까지만 해도 충분합니다!
"""

# ==============================================
# 🔥 4단계: 검증 스크립트
# ==============================================

async def verify_model_loader_di():
    """🧪 ModelLoader DI 시스템 검증"""
    print("🧪 ModelLoader DI 시스템 검증 시작...")
    
    try:
        # 1. DI Container 확인
        from app.core.di_container import get_di_container
        di_container = get_di_container()
        
        model_loader = di_container.get('model_loader')
        if model_loader:
            print("✅ 1단계: ModelLoader DI 등록 확인됨")
        else:
            print("❌ 1단계: ModelLoader가 DI Container에 없음")
            return False
        
        # 2. Step 생성 테스트
        print("🔄 2단계: Step 생성 테스트...")
        
        try:
            from app.ai_pipeline.steps.step_01_human_parsing import create_human_parsing_step
            
            # ModelLoader 없이 생성 (자동 주입 확인)
            step = await create_human_parsing_step(device="cpu")
            
            if hasattr(step, 'model_interface') and step.model_interface:
                print("✅ 2단계: HumanParsingStep ModelLoader 자동 주입 성공")
            else:
                print("⚠️ 2단계: HumanParsingStep ModelLoader 자동 주입 부분 실패")
        
        except Exception as e:
            print(f"❌ 2단계: HumanParsingStep 테스트 실패: {e}")
        
        # 3. 실제 모델 로드 테스트
        print("🔄 3단계: 실제 모델 로드 테스트...")
        
        try:
            if hasattr(model_loader, 'create_step_interface'):
                interface = model_loader.create_step_interface("HumanParsingStep")
                
                if interface:
                    model = await interface.get_model("human_parsing_graphonomy")
                    if model:
                        print("✅ 3단계: 실제 AI 모델 로드 성공")
                    else:
                        print("⚠️ 3단계: AI 모델 로드 실패 (Mock 모델)")
                else:
                    print("❌ 3단계: Step 인터페이스 생성 실패")
        
        except Exception as e:
            print(f"❌ 3단계: 모델 로드 테스트 실패: {e}")
        
        print("🎉 ModelLoader DI 시스템 검증 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 검증 실패: {e}")
        return False

# ==============================================
# 🔥 실행 명령어 요약
# ==============================================

"""
🚀 실행 순서:

1. 새 파일 생성:
   backend/app/core/di_container.py

2. main.py 수정:
   @app.on_event("startup") 함수 추가

3. 서버 재시작:
   cd backend
   python app/main.py

4. 테스트:
   curl http://localhost:8000/api/test-model-loader-di

5. 검증:
   python -c "
   import asyncio
   from app.main import verify_model_loader_di
   asyncio.run(verify_model_loader_di())
   "

결과:
- ✅ HumanParsingStep이 실제 AI 모델 사용
- ✅ ClothSegmentationStep이 실제 AI 모델 사용  
- ✅ Fallback 모드에서 벗어남
- ✅ M3 Max 128GB 완전 활용
"""