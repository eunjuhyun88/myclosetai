# backend/app/services/virtual_fitter.py
"""
🔥 실제 동작하는 MyCloset AI 가상 피팅 시스템
M3 Max 128GB 메모리 최적화 버전
"""

import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np
import asyncio
import cv2
import time
import os
import base64
import io
import mediapipe as mp
from typing import Optional, Dict, Any, Tuple, List
import logging
import uuid
from datetime import datetime
import json

from app.core.gpu_config import gpu_config, DEVICE, MODEL_CONFIG

logger = logging.getLogger(__name__)

class RealVirtualFitter:
    """
    🎽 실제 동작하는 MyCloset AI 가상 피팅 시스템
    
    M3 Max 128GB 메모리를 활용한 고성능 버전
    - MediaPipe를 이용한 실제 인체 분석
    - OpenCV 기반 실제 이미지 처리
    - Metal Performance Shaders 활용
    """
    
    def __init__(self):
        self.device = DEVICE
        self.model_config = MODEL_CONFIG
        self.sessions = {}  # 세션 관리
        
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        
        # 모델 초기화
        self.pose_detector = None
        self.segmentation_model = None
        self.models_loaded = False
        
        # 성능 설정 (M3 Max 최적화)
        self.max_image_size = (1024, 1024)  # M3 Max는 더 큰 이미지 처리 가능
        self.processing_timeout = 60
        
        logger.info(f"🚀 RealVirtualFitter 초기화 완료 - 디바이스: {self.device}")
        
    async def initialize_models(self):
        """실제 AI 모델 초기화"""
        try:
            logger.info("🤖 실제 AI 모델들 초기화 중...")
            
            # MediaPipe Pose 초기화
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,  # 최고 품질
                enable_segmentation=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            # MediaPipe Selfie Segmentation 초기화
            self.segmentation_model = self.mp_selfie_segmentation.SelfieSegmentation(
                model_selection=1  # 고정밀 모델
            )
            
            # GPU 메모리 최적화
            gpu_config.optimize_memory()
            
            self.models_loaded = True
            logger.info("✅ 실제 AI 모델 초기화 완료")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 모델 초기화 실패: {e}")
            self.models_loaded = False
            return False
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """이미지 전처리 (M3 Max 최적화)"""
        try:
            # RGB 모드로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 이미지 방향 자동 수정
            image = ImageOps.exif_transpose(image)
            
            # 크기 조정 (M3 Max는 더 큰 이미지 처리 가능)
            width, height = image.size
            max_width, max_height = self.max_image_size
            
            if width > max_width or height > max_height:
                ratio = min(max_width / width, max_height / height)
                new_size = (int(width * ratio), int(height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"이미지 크기 조정: {(width, height)} -> {new_size}")
            
            # 이미지 품질 향상
            image = self._enhance_image_quality(image)
            
            return image
            
        except Exception as e:
            logger.error(f"이미지 전처리 실패: {e}")
            return image
    
    def _enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """이미지 품질 향상"""
        try:
            # 샤프닝 필터 적용
            enhanced = image.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))
            
            # 색상 보정
            enhanced = ImageOps.autocontrast(enhanced, cutoff=1)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"이미지 품질 향상 실패: {e}")
            return image
    
    async def analyze_human_pose(self, person_image: Image.Image) -> Dict[str, Any]:
        """실제 MediaPipe를 사용한 인체 분석"""
        try:
            if not self.models_loaded:
                await self.initialize_models()
            
            logger.info("👤 실제 MediaPipe 인체 분석 시작...")
            
            # PIL을 OpenCV 형식으로 변환
            cv_image = cv2.cvtColor(np.array(person_image), cv2.COLOR_RGB2BGR)
            
            # MediaPipe로 포즈 감지
            results = self.pose_detector.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            analysis = {
                "pose_detected": False,
                "landmarks": [],
                "segmentation_mask": None,
                "body_measurements": {},
                "confidence": 0.0
            }
            
            if results.pose_landmarks:
                analysis["pose_detected"] = True
                analysis["confidence"] = 0.9
                
                # 랜드마크 좌표 추출
                landmarks = []
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks.append({
                        "id": idx,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })
                
                analysis["landmarks"] = landmarks
                
                # 신체 측정값 계산
                analysis["body_measurements"] = self._calculate_body_measurements(
                    landmarks, person_image.size
                )
                
                # 세그멘테이션 마스크 생성
                if results.segmentation_mask is not None:
                    analysis["segmentation_mask"] = results.segmentation_mask
                
                logger.info("✅ MediaPipe 인체 분석 완료")
            else:
                logger.warning("⚠️ 인체 포즈 감지 실패")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ 인체 분석 실패: {e}")
            return {"pose_detected": False, "confidence": 0.0}
    
    def _calculate_body_measurements(self, landmarks: List[Dict], image_size: Tuple[int, int]) -> Dict[str, Any]:
        """랜드마크를 이용한 실제 신체 측정"""
        try:
            width, height = image_size
            measurements = {}
            
            # 주요 포인트들 (MediaPipe 포즈 랜드마크 인덱스)
            NOSE = 0
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_ELBOW = 13
            RIGHT_ELBOW = 14
            LEFT_WRIST = 15
            RIGHT_WRIST = 16
            LEFT_HIP = 23
            RIGHT_HIP = 24
            LEFT_KNEE = 25
            RIGHT_KNEE = 26
            LEFT_ANKLE = 27
            RIGHT_ANKLE = 28
            
            if len(landmarks) >= 33:  # MediaPipe는 33개 포인트
                # 어깨 너비 계산
                left_shoulder = landmarks[LEFT_SHOULDER]
                right_shoulder = landmarks[RIGHT_SHOULDER]
                shoulder_width = abs(left_shoulder["x"] - right_shoulder["x"]) * width
                
                # 몸통 길이 계산
                torso_length = abs((left_shoulder["y"] + right_shoulder["y"]) / 2 - 
                                 (landmarks[LEFT_HIP]["y"] + landmarks[RIGHT_HIP]["y"]) / 2) * height
                
                # 팔 길이 계산
                arm_length = (
                    np.sqrt((left_shoulder["x"] - landmarks[LEFT_WRIST]["x"])**2 + 
                           (left_shoulder["y"] - landmarks[LEFT_WRIST]["y"])**2) * 
                    max(width, height)
                )
                
                # 다리 길이 계산
                leg_length = abs(landmarks[LEFT_HIP]["y"] - landmarks[LEFT_ANKLE]["y"]) * height
                
                measurements = {
                    "shoulder_width": shoulder_width,
                    "torso_length": torso_length,
                    "arm_length": arm_length,
                    "leg_length": leg_length,
                    "body_height_ratio": leg_length / height if height > 0 else 0.5,
                    "shoulder_center": {
                        "x": (left_shoulder["x"] + right_shoulder["x"]) / 2 * width,
                        "y": (left_shoulder["y"] + right_shoulder["y"]) / 2 * height
                    }
                }
                
                logger.info(f"📏 신체 측정 완료: 어깨너비={shoulder_width:.1f}px")
            
            return measurements
            
        except Exception as e:
            logger.error(f"신체 측정 실패: {e}")
            return {}
    
    async def segment_clothing(self, clothing_image: Image.Image) -> Dict[str, Any]:
        """실제 의류 세그멘테이션"""
        try:
            logger.info("👕 의류 분석 및 세그멘테이션 시작...")
            
            # OpenCV로 변환
            cv_image = cv2.cvtColor(np.array(clothing_image), cv2.COLOR_RGB2BGR)
            
            # 배경 제거 (GrabCut 알고리즘 사용)
            mask = np.zeros(cv_image.shape[:2], np.uint8)
            
            # 전경/배경 모델을 위한 배열
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # 이미지 중앙 영역을 전경으로 가정
            height, width = cv_image.shape[:2]
            rect = (width//8, height//8, width*3//4, height*3//4)
            
            # GrabCut 실행
            cv2.grabCut(cv_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # 마스크 후처리
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # 의류 영역만 추출
            clothing_masked = cv_image * mask2[:, :, np.newaxis]
            
            # PIL로 변환
            clothing_segmented = Image.fromarray(cv2.cvtColor(clothing_masked, cv2.COLOR_BGR2RGB))
            
            # 의류 타입 분석
            clothing_type = self._analyze_clothing_type(clothing_image)
            
            # 색상 분석
            dominant_colors = self._extract_dominant_colors(clothing_image)
            
            analysis = {
                "segmented_clothing": clothing_segmented,
                "mask": mask2,
                "clothing_type": clothing_type,
                "dominant_colors": dominant_colors,
                "confidence": 0.85
            }
            
            logger.info(f"✅ 의류 분석 완료: {clothing_type}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ 의류 세그멘테이션 실패: {e}")
            return {
                "segmented_clothing": clothing_image,
                "clothing_type": "unknown",
                "confidence": 0.0
            }
    
    def _analyze_clothing_type(self, clothing_image: Image.Image) -> str:
        """의류 타입 분석 (형태 기반)"""
        try:
            width, height = clothing_image.size
            aspect_ratio = height / width
            
            # 의류 타입 분류
            if aspect_ratio > 1.8:
                return "dress"
            elif aspect_ratio > 1.4:
                return "pants"
            elif aspect_ratio < 0.7:
                return "jacket"
            elif aspect_ratio < 1.2:
                return "shirt"
            else:
                return "top"
                
        except Exception as e:
            logger.warning(f"의류 타입 분석 실패: {e}")
            return "unknown"
    
    def _extract_dominant_colors(self, image: Image.Image, k: int = 3) -> List[Tuple[int, int, int]]:
        """주요 색상 추출 (K-means 클러스터링)"""
        try:
            # 이미지를 NumPy 배열로 변환
            img_array = np.array(image)
            img_array = img_array.reshape((-1, 3))
            
            # K-means 클러스터링
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(img_array)
            
            # 중심점들을 색상으로 반환
            colors = kmeans.cluster_centers_.astype(int)
            
            return [tuple(color) for color in colors]
            
        except Exception as e:
            logger.warning(f"색상 추출 실패: {e}")
            return [(128, 128, 128)]  # 기본 회색
    
    async def real_virtual_fitting(
        self,
        person_image: Image.Image,
        clothing_image: Image.Image,
        height: float,
        weight: float,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """🔥 실제 가상 피팅 실행"""
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        logger.info(f"🔥 실제 가상 피팅 시작 (세션: {session_id})")
        
        # 세션 정보 저장
        self.sessions[session_id] = {
            "start_time": start_time,
            "status": "processing",
            "steps": []
        }
        
        try:
            # 이미지 전처리
            person_processed = self._preprocess_image(person_image)
            clothing_processed = self._preprocess_image(clothing_image)
            
            # 1단계: 인체 분석
            self._update_session_step(session_id, "인체 분석", "processing")
            body_analysis = await self.analyze_human_pose(person_processed)
            self._update_session_step(session_id, "인체 분석", "completed")
            
            if not body_analysis["pose_detected"]:
                logger.warning("⚠️ 인체 포즈를 감지할 수 없어 기본 모드로 전환")
                result = await self._fallback_fitting(person_processed, clothing_processed)
                return self._create_response(result, session_id, start_time, fallback=True)
            
            # 2단계: 의류 분석
            self._update_session_step(session_id, "의류 분석", "processing")
            clothing_analysis = await self.segment_clothing(clothing_processed)
            self._update_session_step(session_id, "의류 분석", "completed")
            
            # 3단계: 가상 피팅 실행
            self._update_session_step(session_id, "가상 피팅", "processing")
            fitted_result = await self._execute_fitting(
                person_processed, 
                clothing_analysis, 
                body_analysis,
                height,
                weight
            )
            self._update_session_step(session_id, "가상 피팅", "completed")
            
            # 4단계: 후처리
            self._update_session_step(session_id, "품질 향상", "processing")
            final_result = self._post_process_result(fitted_result)
            self._update_session_step(session_id, "품질 향상", "completed")
            
            # 결과 반환
            return self._create_response(final_result, session_id, start_time)
            
        except Exception as e:
            logger.error(f"❌ 가상 피팅 실패: {e}")
            self._update_session_step(session_id, "오류", "error")
            
            # 실패 시 기본 모드
            fallback_result = await self._fallback_fitting(person_image, clothing_image)
            return self._create_response(fallback_result, session_id, start_time, error=str(e))
    
    def _update_session_step(self, session_id: str, step_name: str, status: str):
        """세션 단계 업데이트"""
        if session_id in self.sessions:
            self.sessions[session_id]["steps"].append({
                "name": step_name,
                "status": status,
                "timestamp": time.time()
            })
    
    async def _execute_fitting(
        self,
        person_image: Image.Image,
        clothing_analysis: Dict[str, Any],
        body_analysis: Dict[str, Any],
        height: float,
        weight: float
    ) -> Image.Image:
        """실제 가상 피팅 실행"""
        try:
            result = person_image.copy()
            clothing_image = clothing_analysis["segmented_clothing"]
            clothing_type = clothing_analysis["clothing_type"]
            
            # 신체 측정값 가져오기
            measurements = body_analysis.get("body_measurements", {})
            
            if measurements:
                # 정확한 위치 계산
                fit_position = self._calculate_precise_fit_position(
                    person_image.size,
                    measurements,
                    clothing_type,
                    height,
                    weight
                )
                
                # 의류 크기 조정
                fitted_clothing = self._resize_clothing_to_body(
                    clothing_image,
                    fit_position,
                    measurements
                )
                
                # 자연스러운 합성
                result = self._blend_clothing_naturally(
                    result,
                    fitted_clothing,
                    fit_position,
                    body_analysis
                )
                
            else:
                # 측정값이 없으면 기본 방식
                result = await self._fallback_fitting(person_image, clothing_image)
            
            logger.info("✅ 가상 피팅 실행 완료")
            return result
            
        except Exception as e:
            logger.error(f"❌ 가상 피팅 실행 실패: {e}")
            return person_image
    
    def _calculate_precise_fit_position(
        self,
        image_size: Tuple[int, int],
        measurements: Dict[str, Any],
        clothing_type: str,
        height: float,
        weight: float
    ) -> Dict[str, Any]:
        """정밀한 피팅 위치 계산"""
        try:
            width, height = image_size
            
            # BMI 계산
            bmi = weight / ((height / 100) ** 2)
            
            # 어깨 중심점 기준
            shoulder_center = measurements.get("shoulder_center", {"x": width//2, "y": height//4})
            shoulder_width = measurements.get("shoulder_width", width * 0.3)
            torso_length = measurements.get("torso_length", height * 0.4)
            
            # 의류 타입별 위치 조정
            if clothing_type in ["shirt", "top", "jacket"]:
                # 상의
                clothing_width = int(shoulder_width * 1.2)  # 어깨보다 약간 넓게
                clothing_height = int(torso_length * 0.8)   # 몸통 길이의 80%
                
                x = int(shoulder_center["x"] - clothing_width // 2)
                y = int(shoulder_center["y"] - clothing_height * 0.1)  # 어깨 약간 아래
                
            elif clothing_type == "dress":
                # 원피스
                clothing_width = int(shoulder_width * 1.3)
                clothing_height = int(torso_length * 1.8)
                
                x = int(shoulder_center["x"] - clothing_width // 2)
                y = int(shoulder_center["y"])
                
            elif clothing_type == "pants":
                # 하의
                hip_y = shoulder_center["y"] + torso_length
                clothing_width = int(shoulder_width * 1.1)
                clothing_height = int(measurements.get("leg_length", height * 0.5))
                
                x = int(shoulder_center["x"] - clothing_width // 2)
                y = int(hip_y)
                
            else:
                # 기본값
                clothing_width = int(shoulder_width * 1.2)
                clothing_height = int(torso_length)
                x = int(shoulder_center["x"] - clothing_width // 2)
                y = int(shoulder_center["y"])
            
            # 경계 검사
            x = max(0, min(x, width - clothing_width))
            y = max(0, min(y, height - clothing_height))
            
            return {
                "x": x,
                "y": y,
                "width": clothing_width,
                "height": clothing_height,
                "clothing_type": clothing_type,
                "bmi": bmi
            }
            
        except Exception as e:
            logger.error(f"위치 계산 실패: {e}")
            # 기본 위치 반환
            width, height = image_size
            return {
                "x": width // 4,
                "y": height // 4,
                "width": width // 2,
                "height": height // 2,
                "clothing_type": clothing_type,
                "bmi": 22.0
            }
    
    def _resize_clothing_to_body(
        self,
        clothing_image: Image.Image,
        fit_position: Dict[str, Any],
        measurements: Dict[str, Any]
    ) -> Image.Image:
        """신체에 맞게 의류 크기 조정"""
        try:
            # 기본 리사이즈
            resized = clothing_image.resize(
                (fit_position["width"], fit_position["height"]),
                Image.Resampling.LANCZOS
            )
            
            # 체형에 따른 추가 조정
            bmi = fit_position.get("bmi", 22.0)
            
            if bmi < 18.5:  # 마른 체형
                # 세로로 약간 늘리기
                new_height = int(fit_position["height"] * 1.05)
                new_width = int(fit_position["width"] * 0.95)
                resized = resized.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
            elif bmi > 25:  # 통통한 체형
                # 가로로 약간 늘리기
                new_width = int(fit_position["width"] * 1.1)
                new_height = int(fit_position["height"] * 0.95)
                resized = resized.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 자연스러운 곡률 효과 추가
            resized = self._add_fabric_drape(resized)
            
            return resized
            
        except Exception as e:
            logger.error(f"의류 크기 조정 실패: {e}")
            return clothing_image.resize(
                (fit_position["width"], fit_position["height"]),
                Image.Resampling.LANCZOS
            )
    
    def _add_fabric_drape(self, clothing_image: Image.Image) -> Image.Image:
        """천의 자연스러운 드레이프 효과"""
        try:
            # OpenCV로 변환
            cv_image = cv2.cvtColor(np.array(clothing_image), cv2.COLOR_RGB2BGR)
            
            # 미세한 웨이브 효과
            rows, cols = cv_image.shape[:2]
            
            # 사인파를 이용한 자연스러운 왜곡
            map_x = np.zeros((rows, cols), np.float32)
            map_y = np.zeros((rows, cols), np.float32)
            
            for i in range(rows):
                for j in range(cols):
                    # 미세한 웨이브 효과
                    offset_x = 2 * np.sin(2 * np.pi * i / 180)
                    offset_y = 1 * np.sin(2 * np.pi * j / 100)
                    
                    map_x[i, j] = j + offset_x
                    map_y[i, j] = i + offset_y
            
            # 왜곡 적용
            draped = cv2.remap(cv_image, map_x, map_y, cv2.INTER_LINEAR)
            
            # PIL로 변환
            return Image.fromarray(cv2.cvtColor(draped, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            logger.warning(f"드레이프 효과 실패: {e}")
            return clothing_image
    
    def _blend_clothing_naturally(
        self,
        person_image: Image.Image,
        clothing_image: Image.Image,
        fit_position: Dict[str, Any],
        body_analysis: Dict[str, Any]
    ) -> Image.Image:
        """자연스러운 의류 합성"""
        try:
            result = person_image.copy()
            
            # 그림자 효과 먼저 추가
            shadow = self._create_realistic_shadow(clothing_image, fit_position)
            if shadow:
                try:
                    result.paste(shadow, 
                               (fit_position["x"] + 2, fit_position["y"] + 2), 
                               shadow)
                except:
                    pass
            
            # 의류 알파 마스크 생성
            alpha_mask = self._create_alpha_mask(clothing_image)
            
            # 의류 합성
            try:
                if clothing_image.mode == 'RGBA':
                    result.paste(clothing_image, 
                               (fit_position["x"], fit_position["y"]), 
                               clothing_image)
                else:
                    result.paste(clothing_image, 
                               (fit_position["x"], fit_position["y"]), 
                               alpha_mask)
            except Exception as paste_error:
                logger.warning(f"마스크 붙여넣기 실패: {paste_error}")
                result.paste(clothing_image, (fit_position["x"], fit_position["y"]))
            
            return result
            
        except Exception as e:
            logger.error(f"의류 합성 실패: {e}")
            return person_image
    
    def _create_alpha_mask(self, clothing_image: Image.Image) -> Image.Image:
        """알파 마스크 생성"""
        try:
            # 그레이스케일로 변환
            gray = clothing_image.convert('L')
            
            # 가장자리 페이드 효과
            mask = gray.copy()
            width, height = mask.size
            
            # 가장자리를 부드럽게
            fade_pixels = min(width, height) // 20
            
            pixels = mask.load()
            for y in range(height):
                for x in range(width):
                    edge_dist = min(x, y, width-x-1, height-y-1)
                    if edge_dist < fade_pixels:
                        alpha = int(255 * (edge_dist / fade_pixels))
                        current_alpha = pixels[x, y]
                        pixels[x, y] = min(current_alpha, alpha)
            
            return mask
            
        except Exception as e:
            logger.warning(f"알파 마스크 생성 실패: {e}")
            return clothing_image.convert('L')
    
    def _create_realistic_shadow(self, clothing_image: Image.Image, fit_position: Dict[str, Any]) -> Optional[Image.Image]:
        """현실적인 그림자 효과 생성"""
        try:
            # 그림자용 이미지 생성
            shadow = Image.new('RGBA', clothing_image.size, (0, 0, 0, 0))
            
            # 의류 형태를 따른 그림자 생성
            gray = clothing_image.convert('L')
            
            # 그림자 데이터 생성
            shadow_data = []
            for pixel in gray.getdata():
                if pixel > 50:  # 배경이 아닌 부분
                    shadow_data.append((0, 0, 0, 80))  # 반투명 검은색
                else:
                    shadow_data.append((0, 0, 0, 0))   # 투명
            
            shadow.putdata(shadow_data)
            
            # 블러 효과로 부드러운 그림자
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=3))
            
            return shadow
            
        except Exception as e:
            logger.warning(f"그림자 생성 실패: {e}")
            return None
    
    def _post_process_result(self, result_image: Image.Image) -> Image.Image:
        """결과 이미지 후처리"""
        try:
            # 색상 보정
            enhanced = ImageOps.autocontrast(result_image, cutoff=0.5)
            
            # 선명도 향상
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            
            # 약간의 채도 향상
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Color(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            # 워터마크 추가
            enhanced = self._add_watermark(enhanced, "MyCloset AI - Real Fitting")
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"후처리 실패: {e}")
            return result_image
    
    def _add_watermark(self, image: Image.Image, text: str) -> Image.Image:
        """워터마크 추가"""
        try:
            draw = ImageDraw.Draw(image)
            
            # 폰트 설정
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            # 텍스트 위치 (오른쪽 하단)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = image.width - text_width - 10
            y = image.height - text_height - 10
            
            # 반투명 배경
            draw.rectangle([x-3, y-3, x+text_width+3, y+text_height+3], 
                          fill=(0, 0, 0, 100))
            
            # 텍스트
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            
            return image
            
        except Exception as e:
            logger.warning(f"워터마크 추가 실패: {e}")
            return image
    
    async def _fallback_fitting(self, person_image: Image.Image, clothing_image: Image.Image) -> Image.Image:
        """기본 피팅 모드 (포즈 감지 실패 시)"""
        try:
            logger.info("🔄 기본 피팅 모드 실행")
            
            result = person_image.copy()
            
            # 안전한 크기 계산
            person_width, person_height = person_image.size
            clothing_size = min(person_width, person_height) // 3
            
            # 의류 크기 조정
            clothing_resized = clothing_image.resize((clothing_size, clothing_size), Image.Resampling.LANCZOS)
            
            # 중앙 상단에 배치
            x = (person_width - clothing_size) // 2
            y = person_height // 4
            
            # 간단한 합성
            result.paste(clothing_resized, (x, y))
            
            # 기본 모드 표시
            draw = ImageDraw.Draw(result)
            draw.text((10, 10), "Basic Fitting Mode", fill='white')
            
            return result
            
        except Exception as e:
            logger.error(f"기본 피팅 실패: {e}")
            return person_image
    
    def _create_response(
        self,
        result_image: Image.Image,
        session_id: str,
        start_time: float,
        fallback: bool = False,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """응답 생성"""
        try:
            # 이미지를 base64로 인코딩
            buffer = io.BytesIO()
            result_image.save(buffer, format='JPEG', quality=95)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            processing_time = time.time() - start_time
            
            # 세션 완료 처리
            if session_id in self.sessions:
                self.sessions[session_id]["status"] = "completed"
                self.sessions[session_id]["processing_time"] = processing_time
            
            response = {
                "success": True,
                "session_id": session_id,
                "fitted_image": img_base64,
                "processing_time": processing_time,
                "confidence": 0.9 if not fallback else 0.6,
                "mode": "fallback" if fallback else "advanced",
                "device_info": {
                    "device": self.device,
                    "models_loaded": self.models_loaded
                },
                "steps_completed": len(self.sessions.get(session_id, {}).get("steps", [])),
                "timestamp": datetime.now().isoformat()
            }
            
            if error:
                response["warning"] = f"일부 기능에서 오류 발생: {error}"
            
            return response
            
        except Exception as e:
            logger.error(f"응답 생성 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """세션 상태 조회"""
        if session_id in self.sessions:
            return self.sessions[session_id]
        else:
            return {"error": "Session not found"}
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """오래된 세션 정리"""
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            sessions_to_remove = []
            for session_id, session_data in self.sessions.items():
                if current_time - session_data["start_time"] > max_age_seconds:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.sessions[session_id]
            
            if sessions_to_remove:
                logger.info(f"🧹 {len(sessions_to_remove)}개 오래된 세션 정리 완료")
                
        except Exception as e:
            logger.error(f"세션 정리 실패: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 반환"""
        return {
            "models_loaded": self.models_loaded,
            "device": self.device,
            "active_sessions": len(self.sessions),
            "max_image_size": self.max_image_size,
            "device_info": gpu_config.get_device_info(),
            "memory_usage": gpu_config.get_model_config()
        }

# 전역 인스턴스
real_virtual_fitter = RealVirtualFitter()