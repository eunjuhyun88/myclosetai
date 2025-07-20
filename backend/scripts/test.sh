#!/usr/bin/env python3
"""
🧪 MyCloset AI Backend API 테스트 스크립트
백엔드만 단독으로 테스트할 수 있는 완전한 테스트 도구
"""

import requests
import json
import time
import sys
from pathlib import Path
from PIL import Image
import io
import base64

class BackendTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
        
    def print_result(self, test_name, response):
        """테스트 결과 출력"""
        print(f"\n🧪 {test_name}")
        print(f"Status: {response.status_code}")
        try:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            # 세션 ID 추출
            if 'session_id' in data:
                self.session_id = data['session_id']
                print(f"📝 세션 ID 저장됨: {self.session_id}")
            elif 'details' in data and 'session_id' in data['details']:
                self.session_id = data['details']['session_id']
                print(f"📝 세션 ID 저장됨: {self.session_id}")
                
        except:
            print(f"Response: {response.text}")
        print("-" * 50)
    
    def create_test_images(self):
        """테스트용 이미지 생성"""
        print("🖼️ 테스트 이미지 생성 중...")
        
        # 사용자 이미지 (파란색)
        person_img = Image.new('RGB', (512, 768), color=(135, 206, 235))
        person_buffer = io.BytesIO()
        person_img.save(person_buffer, format='JPEG', quality=85)
        person_data = person_buffer.getvalue()
        
        # 의류 이미지 (빨간색)
        clothing_img = Image.new('RGB', (512, 768), color=(240, 128, 128))
        clothing_buffer = io.BytesIO()
        clothing_img.save(clothing_buffer, format='JPEG', quality=85)
        clothing_data = clothing_buffer.getvalue()
        
        print("✅ 테스트 이미지 생성 완료")
        return person_data, clothing_data
    
    def test_basic_endpoints(self):
        """기본 엔드포인트 테스트"""
        print("🔧 기본 엔드포인트 테스트")
        
        # 루트 엔드포인트
        response = requests.get(f"{self.base_url}/")
        self.print_result("루트 엔드포인트", response)
        
        # 헬스체크
        response = requests.get(f"{self.base_url}/api/health")
        self.print_result("헬스체크", response)
        
        # 시스템 정보
        response = requests.get(f"{self.base_url}/api/system/info")
        self.print_result("시스템 정보", response)
    
    def test_step_1_image_validation(self, person_data, clothing_data):
        """1단계: 이미지 검증 테스트"""
        files = {
            'person_image': ('test_person.jpg', person_data, 'image/jpeg'),
            'clothing_image': ('test_clothing.jpg', clothing_data, 'image/jpeg')
        }
        
        response = requests.post(f"{self.base_url}/api/step/1/upload-validation", files=files)
        self.print_result("1단계: 이미지 검증", response)
        return response.status_code == 200
    
    def test_step_2_measurements(self):
        """2단계: 측정값 검증 테스트"""
        data = {
            'height': 170,
            'weight': 65,
            'session_id': self.session_id or 'test_session_123'
        }
        
        response = requests.post(f"{self.base_url}/api/step/2/measurements-validation", data=data)
        self.print_result("2단계: 측정값 검증", response)
        return response.status_code == 200
    
    def test_ai_steps(self):
        """3-8단계: AI 처리 단계 테스트"""
        if not self.session_id:
            print("⚠️ 세션 ID가 없습니다. 1-2단계를 먼저 실행하세요.")
            return
        
        steps = [
            (3, "human-parsing", "인체 파싱"),
            (4, "pose-estimation", "포즈 추정"),
            (5, "clothing-analysis", "의류 분석"),
            (6, "geometric-matching", "기하학적 매칭"),
            (7, "virtual-fitting", "가상 피팅"),
            (8, "result-analysis", "결과 분석")
        ]
        
        for step_id, endpoint, name in steps:
            data = {'session_id': self.session_id}
            url = f"{self.base_url}/api/step/{step_id}/{endpoint}"
            
            response = requests.post(url, data=data)
            self.print_result(f"{step_id}단계: {name}", response)
            
            if response.status_code != 200:
                print(f"❌ {step_id}단계 실패, 다음 단계 건너뛰기")
                break
            
            # 단계별 딜레이 (실제 처리 시뮬레이션)
            time.sleep(1)
    
    def test_complete_pipeline(self, person_data, clothing_data):
        """전체 파이프라인 테스트"""
        print("\n🚀 전체 파이프라인 테스트")
        
        files = {
            'person_image': ('test_person.jpg', person_data, 'image/jpeg'),
            'clothing_image': ('test_clothing.jpg', clothing_data, 'image/jpeg')
        }
        
        data = {
            'height': 175,
            'weight': 70,
            'session_id': 'complete_test_' + str(int(time.time()))
        }
        
        response = requests.post(f"{self.base_url}/api/pipeline/complete", files=files, data=data)
        self.print_result("전체 파이프라인", response)
        return response.status_code == 200
    
    def test_legacy_api(self, person_data, clothing_data):
        """레거시 API 테스트"""
        print("\n🔄 레거시 API 테스트")
        
        files = {
            'person_image': ('test_person.jpg', person_data, 'image/jpeg'),
            'clothing_image': ('test_clothing.jpg', clothing_data, 'image/jpeg')
        }
        
        data = {
            'height': 172,
            'weight': 68
        }
        
        response = requests.post(f"{self.base_url}/api/virtual-tryon", files=files, data=data)
        self.print_result("레거시 가상 피팅", response)
        return response.status_code == 200
    
    def test_memory_optimization(self):
        """메모리 최적화 테스트"""
        print("\n💾 메모리 최적화 테스트")
        
        response = requests.post(f"{self.base_url}/api/optimize-memory")
        self.print_result("메모리 최적화", response)
        return response.status_code == 200
    
    def test_pipeline_status(self):
        """파이프라인 상태 조회 테스트"""
        if not self.session_id:
            print("⚠️ 세션 ID가 없어 상태 조회를 건너뜁니다.")
            return
        
        print(f"\n📊 파이프라인 상태 조회 (세션: {self.session_id})")
        
        response = requests.get(f"{self.base_url}/api/pipeline/status/{self.session_id}")
        self.print_result("파이프라인 상태", response)
        return response.status_code == 200
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🎯 MyCloset AI Backend 전체 테스트 시작")
        print("=" * 60)
        
        # 테스트 이미지 생성
        person_data, clothing_data = self.create_test_images()
        
        # 기본 엔드포인트 테스트
        self.test_basic_endpoints()
        
        # 단계별 테스트
        print("\n📋 8단계 파이프라인 순차 테스트")
        if self.test_step_1_image_validation(person_data, clothing_data):
            if self.test_step_2_measurements():
                self.test_ai_steps()
                self.test_pipeline_status()
        
        # 통합 테스트
        self.test_complete_pipeline(person_data, clothing_data)
        
        # 레거시 API 테스트
        self.test_legacy_api(person_data, clothing_data)
        
        # 메모리 최적화 테스트
        self.test_memory_optimization()
        
        print("\n🎉 모든 테스트 완료!")
        print("📖 API 문서: http://localhost:8000/docs")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MyCloset AI Backend API 테스트")
    parser.add_argument("--url", default="http://localhost:8000", help="백엔드 서버 URL")
    parser.add_argument("--test", choices=['all', 'basic', 'pipeline', 'complete', 'legacy'], 
                        default='all', help="실행할 테스트 종류")
    
    args = parser.parse_args()
    
    tester = BackendTester(args.url)
    
    # 서버 연결 확인
    try:
        response = requests.get(f"{args.url}/api/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ 서버 연결 실패: {response.status_code}")
            sys.exit(1)
        print(f"✅ 서버 연결 확인: {args.url}")
    except requests.exceptions.RequestException as e:
        print(f"❌ 서버에 연결할 수 없습니다: {e}")
        print("💡 백엔드 서버가 실행 중인지 확인하세요: python run_server.py")
        sys.exit(1)
    
    # 테스트 실행
    if args.test == 'all':
        tester.run_all_tests()
    elif args.test == 'basic':
        tester.test_basic_endpoints()
    elif args.test == 'pipeline':
        person_data, clothing_data = tester.create_test_images()
        if tester.test_step_1_image_validation(person_data, clothing_data):
            if tester.test_step_2_measurements():
                tester.test_ai_steps()
    elif args.test == 'complete':
        person_data, clothing_data = tester.create_test_images()
        tester.test_complete_pipeline(person_data, clothing_data)
    elif args.test == 'legacy':
        person_data, clothing_data = tester.create_test_images()
        tester.test_legacy_api(person_data, clothing_data)

if __name__ == "__main__":
    main()