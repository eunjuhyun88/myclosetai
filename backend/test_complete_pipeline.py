#!/usr/bin/env python3
"""
MyCloset AI - 완전한 4-6단계 파이프라인 테스트
Step 4: 기하학적 매칭 → Step 5: 의류 워핑 → Step 6: 가상 피팅
"""

import asyncio
import numpy as np
import requests
import base64
import json
import time
from PIL import Image
from io import BytesIO

class CompletePipelineTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.session = requests.Session()
    
    def create_test_images(self):
        """테스트용 이미지 생성"""
        # 사람 이미지 (512x512)
        person_img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
        # 상체 영역 밝게
        person_img[100:350, 150:350] = np.random.randint(150, 255, (250, 200, 3), dtype=np.uint8)
        
        # 의류 이미지 (512x512)
        cloth_img = np.random.randint(80, 220, (512, 512, 3), dtype=np.uint8)
        # 의류 패턴
        cloth_img[50:450, 50:450] = [100, 149, 237]  # 콘플라워 블루
        
        return person_img, cloth_img
    
    def numpy_to_base64(self, img_array):
        """numpy 배열을 base64로 변환"""
        img = Image.fromarray(img_array)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def test_health_check(self):
        """서버 상태 확인"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            print(f"🏥 헬스체크: {response.status_code} - {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ 헬스체크 실패: {e}")
            return False
    
    def test_step_04_geometric_matching(self, person_img, cloth_img):
        """Step 4: 기하학적 매칭 테스트"""
        print("\n🎯 Step 4: 기하학적 매칭 테스트")
        
        try:
            # Step 4 API 호출
            payload = {
                "person_image": self.numpy_to_base64(person_img),
                "clothing_image": self.numpy_to_base64(cloth_img),
                "quality_level": "balanced",
                "method": "neural_tps"
            }
            
            response = self.session.post(
                f"{self.base_url}/api/step/4/geometric-matching",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Step 4 성공: 신뢰도 {result.get('confidence', 0):.3f}")
                print(f"   처리시간: {result.get('processing_time', 0):.2f}초")
                print(f"   키포인트: {len(result.get('source_keypoints', []))}개")
                return result
            else:
                print(f"❌ Step 4 실패: {response.status_code}")
                print(f"   응답: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"❌ Step 4 예외: {e}")
            return None
    
    def test_step_05_cloth_warping(self, person_img, cloth_img, step4_result=None):
        """Step 5: 의류 워핑 테스트"""
        print("\n🌊 Step 5: 의류 워핑 테스트")
        
        try:
            payload = {
                "cloth_image": self.numpy_to_base64(cloth_img),
                "person_image": self.numpy_to_base64(person_img),
                "fabric_type": "cotton",
                "clothing_type": "shirt",
                "warping_method": "hybrid",
                "quality_level": "high"
            }
            
            # Step 4 결과가 있으면 포함
            if step4_result:
                payload["geometric_data"] = {
                    "source_keypoints": step4_result.get("source_keypoints", []),
                    "target_keypoints": step4_result.get("target_keypoints", []),
                    "transformation_matrix": step4_result.get("transformation_matrix")
                }
            
            response = self.session.post(
                f"{self.base_url}/api/step/5/cloth-warping",
                json=payload,
                timeout=45
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Step 5 성공: 품질 {result.get('quality_score', 0):.3f}")
                print(f"   처리시간: {result.get('processing_time', 0):.2f}초")
                print(f"   워핑방법: {result.get('metadata', {}).get('warping_method_used', 'N/A')}")
                return result
            else:
                print(f"❌ Step 5 실패: {response.status_code}")
                print(f"   응답: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"❌ Step 5 예외: {e}")
            return None
    
    def test_step_06_virtual_fitting(self, person_img, cloth_img, step5_result=None):
        """Step 6: 가상 피팅 테스트"""
        print("\n🎭 Step 6: 가상 피팅 테스트")
        
        try:
            payload = {
                "person_image": self.numpy_to_base64(person_img),
                "cloth_image": self.numpy_to_base64(cloth_img),
                "fabric_type": "cotton",
                "clothing_type": "shirt",
                "fitting_method": "hybrid",
                "quality_enhancement": True,
                "enable_visualization": True
            }
            
            # Step 5 결과가 있으면 포함
            if step5_result:
                payload["warped_cloth_data"] = {
                    "warped_image": step5_result.get("warped_cloth_image"),
                    "quality_score": step5_result.get("quality_score", 0)
                }
            
            response = self.session.post(
                f"{self.base_url}/api/step/6/virtual-fitting",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Step 6 성공: 전체점수 {result.get('overall_score', 0):.3f}")
                print(f"   처리시간: {result.get('processing_time', 0):.2f}초")
                print(f"   피팅방법: {result.get('metadata', {}).get('fitting_method', 'N/A')}")
                print(f"   시각화: {'생성됨' if result.get('visualization') else '없음'}")
                return result
            else:
                print(f"❌ Step 6 실패: {response.status_code}")
                print(f"   응답: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"❌ Step 6 예외: {e}")
            return None
    
    def test_complete_pipeline(self, person_img, cloth_img):
        """완전한 파이프라인 테스트"""
        print("\n🚀 완전한 4-6단계 파이프라인 테스트")
        
        try:
            payload = {
                "person_image": self.numpy_to_base64(person_img),
                "clothing_image": self.numpy_to_base64(cloth_img),
                "fabric_type": "cotton",
                "clothing_type": "shirt",
                "quality_level": "balanced",
                "enable_all_steps": True,
                "enable_visualization": True
            }
            
            response = self.session.post(
                f"{self.base_url}/api/pipeline/complete",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 완전 파이프라인 성공!")
                print(f"   전체 처리시간: {result.get('total_processing_time', 0):.2f}초")
                
                # 각 단계별 결과 확인
                steps = result.get('steps_results', {})
                for step_name, step_result in steps.items():
                    if step_result.get('success'):
                        print(f"   ✅ {step_name}: {step_result.get('confidence', 0):.3f}")
                    else:
                        print(f"   ❌ {step_name}: {step_result.get('error', 'Unknown')}")
                
                return result
            else:
                print(f"❌ 완전 파이프라인 실패: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ 완전 파이프라인 예외: {e}")
            return None
    
    def run_full_test(self):
        """전체 테스트 실행"""
        print("🔥 MyCloset AI Step 4-6 완전한 파이프라인 테스트 시작!")
        print("=" * 60)
        
        # 1. 서버 상태 확인
        if not self.test_health_check():
            print("❌ 서버 연결 실패. 서버가 실행 중인지 확인하세요.")
            return
        
        # 2. 테스트 이미지 생성
        print("\n📸 테스트 이미지 생성 중...")
        person_img, cloth_img = self.create_test_images()
        print(f"   사람 이미지: {person_img.shape}")
        print(f"   의류 이미지: {cloth_img.shape}")
        
        # 3. 개별 Step 테스트
        step4_result = self.test_step_04_geometric_matching(person_img, cloth_img)
        step5_result = self.test_step_05_cloth_warping(person_img, cloth_img, step4_result)
        step6_result = self.test_step_06_virtual_fitting(person_img, cloth_img, step5_result)
        
        # 4. 완전한 파이프라인 테스트
        complete_result = self.test_complete_pipeline(person_img, cloth_img)
        
        # 5. 결과 요약
        print("\n" + "=" * 60)
        print("📊 테스트 결과 요약")
        print("=" * 60)
        
        tests = [
            ("Step 4 (기하학적 매칭)", step4_result),
            ("Step 5 (의류 워핑)", step5_result),
            ("Step 6 (가상 피팅)", step6_result),
            ("완전한 파이프라인", complete_result)
        ]
        
        success_count = 0
        for test_name, result in tests:
            if result and result.get('success', False):
                print(f"✅ {test_name}: 성공")
                success_count += 1
            else:
                print(f"❌ {test_name}: 실패")
        
        print(f"\n🎯 성공률: {success_count}/{len(tests)} ({success_count/len(tests)*100:.1f}%)")
        
        if success_count == len(tests):
            print("🎉 모든 테스트 성공! 파이프라인이 완벽하게 작동합니다!")
        elif success_count >= len(tests) // 2:
            print("⚠️ 일부 테스트 성공. 누락된 라이브러리를 설치하세요.")
        else:
            print("❌ 대부분 테스트 실패. 서버 설정을 확인하세요.")

def main():
    """메인 실행 함수"""
    tester = CompletePipelineTester()
    tester.run_full_test()

if __name__ == "__main__":
    main()