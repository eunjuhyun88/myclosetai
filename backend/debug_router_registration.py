#!/usr/bin/env python3
"""
라우터 등록 문제 진단 스크립트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def debug_router_registration():
    """라우터 등록 문제 진단"""
    print("🔧 라우터 등록 문제 진단")
    print("=" * 60)
    
    # 1. step_routes 직접 import 테스트
    print("\n1. step_routes 직접 import 테스트")
    try:
        from app.api.step_routes import router as step_router
        print("✅ step_routes 라우터 import 성공")
        
        # 라우터 상태 확인
        if hasattr(step_router, 'routes'):
            route_count = len(step_router.routes)
            print(f"✅ step_router에 {route_count}개 엔드포인트 확인됨")
            
            # 주요 엔드포인트 확인
            human_parsing_found = False
            for route in step_router.routes:
                if hasattr(route, 'path') and hasattr(route, 'methods'):
                    if '/3/human-parsing' in route.path:
                        print(f"✅ /3/human-parsing 엔드포인트 확인됨: {route.path} [{', '.join(route.methods)}]")
                        human_parsing_found = True
            
            if not human_parsing_found:
                print("❌ /3/human-parsing 엔드포인트를 찾을 수 없음!")
                
        else:
            print("❌ step_router에 routes 속성이 없음")
            
    except Exception as e:
        print(f"❌ step_routes import 실패: {e}")
    
    # 2. AVAILABLE_ROUTERS 확인
    print("\n2. AVAILABLE_ROUTERS 확인")
    try:
        from app.api import AVAILABLE_ROUTERS
        print(f"✅ AVAILABLE_ROUTERS 키들: {list(AVAILABLE_ROUTERS.keys())}")
        
        if 'step_routes' in AVAILABLE_ROUTERS:
            step_router = AVAILABLE_ROUTERS['step_routes']
            print(f"✅ step_routes 라우터 발견: {type(step_router)}")
            
            if hasattr(step_router, 'routes'):
                route_count = len(step_router.routes)
                print(f"✅ step_router에 {route_count}개 엔드포인트 확인됨")
            else:
                print("❌ step_router에 routes 속성이 없음")
        else:
            print("❌ step_routes가 AVAILABLE_ROUTERS에 없음!")
            
    except Exception as e:
        print(f"❌ AVAILABLE_ROUTERS 확인 실패: {e}")
    
    # 3. register_routers 함수 테스트
    print("\n3. register_routers 함수 테스트")
    try:
        from app.api import register_routers
        
        # Mock FastAPI app 생성
        from fastapi import FastAPI
        mock_app = FastAPI()
        
        # 라우터 등록 테스트
        registered_count = register_routers(mock_app)
        print(f"✅ register_routers 실행 완료: {registered_count}개 라우터 등록됨")
        
        # 등록된 라우터 확인
        if hasattr(mock_app, 'routes'):
            print(f"✅ mock_app에 {len(mock_app.routes)}개 라우터 등록됨")
            
            # /api/step 경로 확인
            step_routes_found = False
            for route in mock_app.routes:
                if hasattr(route, 'path') and '/api/step' in route.path:
                    print(f"✅ /api/step 경로 발견: {route.path}")
                    step_routes_found = True
            
            if not step_routes_found:
                print("❌ /api/step 경로를 찾을 수 없음!")
                
    except Exception as e:
        print(f"❌ register_routers 테스트 실패: {e}")
    
    # 4. Central Hub Container 확인
    print("\n4. Central Hub Container 확인")
    try:
        from app.api import _get_central_hub_container
        container = _get_central_hub_container()
        
        if container:
            print("✅ Central Hub Container 발견")
        else:
            print("⚠️ Central Hub Container 없음 (정상일 수 있음)")
            
    except Exception as e:
        print(f"❌ Central Hub Container 확인 실패: {e}")
    
    print("\n" + "=" * 60)
    print("📊 진단 결과 요약:")
    print("✅ step_routes 라우터가 정상적으로 import되어야 합니다")
    print("✅ AVAILABLE_ROUTERS에 step_routes가 포함되어야 합니다")
    print("✅ register_routers 함수가 정상적으로 실행되어야 합니다")
    print("✅ /api/step 경로가 등록되어야 합니다")

if __name__ == "__main__":
    debug_router_registration() 