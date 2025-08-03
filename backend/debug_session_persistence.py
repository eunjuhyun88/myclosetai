#!/usr/bin/env python3
"""
세션 지속성 문제 진단 및 해결 스크립트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def debug_session_persistence():
    """세션 지속성 문제 진단"""
    print("🔧 세션 지속성 문제 진단")
    print("=" * 60)
    
    # 1. 세션 매니저 상태 확인
    print("\n1. 세션 매니저 상태 확인")
    try:
        from app.api.step_routes import _get_or_create_global_session_manager
        session_manager = _get_or_create_global_session_manager()
        
        if session_manager:
            print(f"✅ 세션 매니저 생성 성공")
            print(f"📊 세션 수: {len(session_manager.sessions)}")
            print(f"🔑 세션 키들: {list(session_manager.sessions.keys())}")
            
            # 세션 구조 확인
            if session_manager.sessions:
                sample_session_id = list(session_manager.sessions.keys())[0]
                sample_session = session_manager.sessions[sample_session_id]
                print(f"🔍 샘플 세션 구조: {sample_session}")
        else:
            print("❌ 세션 매니저 생성 실패")
            
    except Exception as e:
        print(f"❌ 세션 매니저 확인 실패: {e}")
    
    # 2. 세션 생성 테스트
    print("\n2. 세션 생성 테스트")
    try:
        import asyncio
        from PIL import Image
        import numpy as np
        
        # 테스트 이미지 생성
        test_person_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        test_clothing_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        
        async def test_session_creation():
            session_manager = _get_or_create_global_session_manager()
            if session_manager:
                # 세션 생성
                session_id = await session_manager.create_session(
                    person_image=test_person_img,
                    clothing_image=test_clothing_img,
                    measurements={}
                )
                print(f"✅ 테스트 세션 생성: {session_id}")
                
                # 세션 상태 확인
                status = await session_manager.get_session_status(session_id)
                print(f"🔍 세션 상태: {status.get('status', 'unknown')}")
                
                # 세션 이미지 확인
                try:
                    person_img, clothing_img = await session_manager.get_session_images(session_id)
                    print(f"✅ 세션 이미지 로드 성공: {person_img.size}, {clothing_img.size}")
                except Exception as e:
                    print(f"❌ 세션 이미지 로드 실패: {e}")
                
                return session_id
            return None
        
        # 비동기 테스트 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        test_session_id = loop.run_until_complete(test_session_creation())
        loop.close()
        
        if test_session_id:
            print(f"✅ 세션 생성 테스트 완료: {test_session_id}")
        else:
            print("❌ 세션 생성 테스트 실패")
            
    except Exception as e:
        print(f"❌ 세션 생성 테스트 실패: {e}")
    
    # 3. 세션 지속성 테스트
    print("\n3. 세션 지속성 테스트")
    try:
        session_manager = _get_or_create_global_session_manager()
        if session_manager and session_manager.sessions:
            # 기존 세션들 확인
            existing_sessions = list(session_manager.sessions.keys())
            print(f"📊 기존 세션 수: {len(existing_sessions)}")
            
            # 세션 지속성 확인
            for session_id in existing_sessions[:3]:  # 처음 3개만 테스트
                print(f"🔍 세션 지속성 확인: {session_id}")
                
                # 세션 존재 여부 확인
                if session_id in session_manager.sessions:
                    session = session_manager.sessions[session_id]
                    print(f"  ✅ 세션 존재: {session.get('status', 'unknown')}")
                    
                    # 데이터 존재 여부 확인
                    if 'data' in session:
                        data_keys = list(session['data'].keys())
                        print(f"  📋 데이터 키들: {data_keys}")
                        
                        # 이미지 데이터 확인
                        if 'original_person_image' in session['data']:
                            img_data = session['data']['original_person_image']
                            print(f"  🖼️ 사람 이미지 데이터 길이: {len(img_data) if img_data else 0}")
                        else:
                            print(f"  ⚠️ 사람 이미지 데이터 없음")
                            
                        if 'original_clothing_image' in session['data']:
                            img_data = session['data']['original_clothing_image']
                            print(f"  👕 의류 이미지 데이터 길이: {len(img_data) if img_data else 0}")
                        else:
                            print(f"  ⚠️ 의류 이미지 데이터 없음")
                    else:
                        print(f"  ⚠️ 세션 데이터 없음")
                else:
                    print(f"  ❌ 세션 없음")
                    
    except Exception as e:
        print(f"❌ 세션 지속성 테스트 실패: {e}")
    
    # 4. 문제 해결 방안 제시
    print("\n4. 문제 해결 방안")
    print("🔧 세션 손실 문제 해결 방안:")
    print("  1. 세션 매니저 싱글톤 패턴 강화")
    print("  2. 세션 데이터 백업 메커니즘 추가")
    print("  3. 세션 만료 시간 연장")
    print("  4. 세션 복구 로직 구현")
    print("  5. 메모리 최적화 시 세션 보호")

if __name__ == "__main__":
    debug_session_persistence() 