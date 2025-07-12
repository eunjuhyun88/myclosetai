#!/usr/bin/env python3
"""기본 Virtual Try-On 구현"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Optional

class BasicVirtualTryOn:
    def __init__(self):
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        print("✅ Virtual Try-On 모델 초기화 완료")
    
    def create_test_images(self):
        """테스트 이미지 생성"""
        import os
        os.makedirs("data/test", exist_ok=True)
        
        # 사람 이미지 (더미)
        person = np.ones((600, 400, 3), dtype=np.uint8) * 200
        cv2.rectangle(person, (150, 100), (250, 400), (100, 100, 100), -1)
        cv2.putText(person, "PERSON", (140, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite("data/test/person.jpg", person)
        
        # 의류 이미지 (더미)
        cloth = np.ones((300, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(cloth, (50, 50), (250, 250), (0, 100, 200), -1)
        cv2.putText(cloth, "SHIRT", (100, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite("data/test/cloth.jpg", cloth)
        
        print("✅ 테스트 이미지 생성 완료!")
        return "data/test/person.jpg", "data/test/cloth.jpg"
    
    def process(self, person_path: str, cloth_path: str) -> Optional[np.ndarray]:
        """가상 피팅 처리"""
        # 이미지 로드
        person_img = cv2.imread(person_path)
        cloth_img = cv2.imread(cloth_path)
        
        if person_img is None or cloth_img is None:
            print("❌ 이미지를 로드할 수 없습니다")
            return None
        
        print("🔄 가상 피팅 처리 중...")
        
        # 간단한 오버레이 (실제로는 더 복잡한 처리 필요)
        h, w = person_img.shape[:2]
        
        # 의류 크기 조정
        cloth_resized = cv2.resize(cloth_img, (200, 250))
        
        # 위치 계산 (가슴 중앙)
        x = w // 2 - 100
        y = 120
        
        # 결과 이미지 생성
        result = person_img.copy()
        
        # 의류 오버레이
        roi = result[y:y+250, x:x+200]
        alpha = 0.8
        blended = cv2.addWeighted(roi, 1-alpha, cloth_resized, alpha, 0)
        result[y:y+250, x:x+200] = blended
        
        return result

# 테스트 실행
if __name__ == "__main__":
    print("🚀 Virtual Try-On 테스트 시작")
    
    model = BasicVirtualTryOn()
    
    # 테스트 이미지 생성
    person_path, cloth_path = model.create_test_images()
    
    # 처리
    result = model.process(person_path, cloth_path)
    
    if result is not None:
        # 결과 저장
        import os
        os.makedirs("results", exist_ok=True)
        cv2.imwrite("results/test_result.jpg", result)
        print("✅ 완료! results/test_result.jpg 확인하세요")
        
        # 비교 이미지 생성
        person = cv2.imread(person_path)
        cloth = cv2.imread(cloth_path)
        cloth_display = cv2.resize(cloth, (200, 300))
        
        # 가로로 나열
        comparison = np.hstack([
            cv2.resize(person, (300, 400)),
            np.ones((400, 50, 3), dtype=np.uint8) * 255,
            cv2.copyMakeBorder(cloth_display, 50, 50, 50, 50, 
                             cv2.BORDER_CONSTANT, value=[255,255,255]),
            np.ones((400, 50, 3), dtype=np.uint8) * 255,
            cv2.resize(result, (300, 400))
        ])
        
        cv2.imwrite("results/comparison.jpg", comparison)
        print("📊 비교 이미지: results/comparison.jpg")
