# backend/app/services/real_virtual_fitting.py
"""
실제 가상 피팅 통합 서비스
인체 분석 → 3D 모델링 → 물리 시뮬레이션 → 렌더링
"""

import asyncio
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from typing import Dict, Any, Tuple
import logging

from app.services.human_analysis import human_analyzer
from app.services.clothing_3d_modeling import clothing_3d_modeler

logger = logging.getLogger(__name__)

class RealVirtualFittingService:
    """실제 가상 피팅 서비스"""
    
    def __init__(self):
        self.processing_steps = [
            "인체 분석 및 분할",
            "3D 포즈 추정", 
            "의류 3D 모델링",
            "체형별 변형",
            "물리 시뮬레이션",
            "조명 및 렌더링",
            "후처리 및 합성"
        ]
        
    async def process_virtual_fitting(
        self,
        person_image: Image.Image,
        clothing_image: Image.Image,
        height: float,
        weight: float,
        clothing_type: str = "shirt",
        quality_level: str = "high"
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """실제 가상 피팅 처리"""
        
        logger.info("🎯 실제 가상 피팅 프로세스 시작")
        start_time = time.time()
        
        processing_info = {
            "steps_completed": [],
            "processing_times": {},
            "quality_metrics": {},
            "errors": []
        }
        
        try:
            # 1단계: 인체 분석 및 분할
            step_start = time.time()
            logger.info("👤 1단계: 인체 분석 시작...")
            
            body_analysis = await human_analyzer.analyze_human_body(person_image)
            
            if not body_analysis['pose_landmarks']['detected']:
                raise ValueError("인체 포즈를 검출할 수 없습니다")
            
            processing_info["steps_completed"].append("인체 분석")
            processing_info["processing_times"]["human_analysis"] = time.time() - step_start
            
            # 2단계: 의류 3D 모델링
            step_start = time.time()
            logger.info("👕 2단계: 의류 3D 모델링 시작...")
            
            clothing_mesh = await clothing_3d_modeler.create_clothing_mesh(
                clothing_image, clothing_type, "cotton"
            )
            
            processing_info["steps_completed"].append("의류 3D 모델링")
            processing_info["processing_times"]["clothing_modeling"] = time.time() - step_start
            
            # 3단계: 체형별 의류 변형
            step_start = time.time()
            logger.info("🔧 3단계: 체형별 의류 변형 시작...")
            
            fitted_clothing = await clothing_3d_modeler.fit_clothing_to_body(
                clothing_mesh,
                body_analysis,
                body_analysis['body_measurements']
            )
            
            processing_info["steps_completed"].append("체형별 변형")
            processing_info["processing_times"]["body_fitting"] = time.time() - step_start
            
            # 4단계: 고품질 렌더링 및 합성
            step_start = time.time()
            logger.info("🎨 4단계: 고품질 렌더링 시작...")
            
            result_image = await self._render_and_composite(
                person_image,
                fitted_clothing,
                body_analysis,
                quality_level
            )
            
            processing_info["steps_completed"].append("렌더링 및 합성")
            processing_info["processing_times"]["rendering"] = time.time() - step_start
            
            # 5단계: 품질 분석 및 후처리
            step_start = time.time()
            logger.info("✨ 5단계: 품질 분석 및 후처리...")
            
            final_image, quality_metrics = await self._post_process_and_analyze(
                result_image, body_analysis, fitted_clothing
            )
            
            processing_info["quality_metrics"] = quality_metrics
            processing_info["processing_times"]["post_processing"] = time.time() - step_start
            
            # 전체 처리 시간
            total_time = time.time() - start_time
            processing_info["total_processing_time"] = total_time
            processing_info["success"] = True
            
            logger.info(f"✅ 실제 가상 피팅 완료 ({total_time:.2f}초)")
            
            return final_image, processing_info
            
        except Exception as e:
            error_msg = f"가상 피팅 처리 실패: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            processing_info["errors"].append(error_msg)
            processing_info["success"] = False
            
            # 에러 발생시 기본 합성 이미지 반환
            fallback_image = await self._create_fallback_result(person_image, clothing_image)
            
            return fallback_image, processing_info
    
    async def _render_and_composite(
        self,
        person_image: Image.Image,
        fitted_clothing: Dict[str, Any],
        body_analysis: Dict[str, Any],
        quality_level: str
    ) -> Image.Image:
        """고품질 렌더링 및 합성"""
        
        # 베이스 이미지
        result = person_image.copy()
        width, height = result.size
        
        # 1. 의류 영역 렌더링
        clothing_region = self._render_3d_clothing(
            fitted_clothing['fitted_mesh'],
            fitted_clothing['adjusted_texture'],
            width, height
        )
        
        # 2. 조명 분석 및 적용
        lighting_info = self._analyze_lighting(person_image, body_analysis)
        lit_clothing = self._apply_realistic_lighting(clothing_region, lighting_info)
        
        # 3. 그림자 생성
        shadow_map = self._generate_realistic_shadows(
            fitted_clothing['fitted_mesh'],
            lighting_info,
            width, height
        )
        
        # 4. 오클루전 처리 (가려지는 부분)
        occlusion_mask = self._calculate_occlusion(
            body_analysis['clothing_regions'],
            fitted_clothing['fitted_mesh']
        )
        
        # 5. 고급 블렌딩
        composited = self._advanced_composite(
            result, lit_clothing, shadow_map, occlusion_mask
        )
        
        # 6. 품질 레벨에 따른 추가 처리
        if quality_level == "high":
            composited = self._apply_high_quality_effects(composited)
        elif quality_level == "ultra":
            composited = self._apply_ultra_quality_effects(composited)
        
        return composited
    
    def _render_3d_clothing(
        self, 
        mesh: Dict[str, np.ndarray], 
        texture: np.ndarray,
        width: int, 
        height: int
    ) -> Image.Image:
        """3D 의류 메쉬를 2D 이미지로 렌더링"""
        
        # 3D 메쉬를 2D로 투영
        vertices = mesh['vertices']
        faces = mesh['faces']
        
        # 렌더링 버퍼 생성
        render_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        z_buffer = np.full((height, width), -np.inf)
        
        # 각 면(face)을 렌더링
        for face in faces:
            if len(face) >= 3:
                # 삼각형 정점
                v0, v1, v2 = vertices[face[:3]]
                
                # 2D 좌표로 변환
                p0 = (int(v0[0]), int(v0[1]))
                p1 = (int(v1[0]), int(v1[1]))
                p2 = (int(v2[0]), int(v2[1]))
                
                # 삼각형 래스터화 및 텍스처 매핑
                self._rasterize_triangle(
                    render_buffer, z_buffer, texture,
                    p0, p1, p2, v0[2], v1[2], v2[2]
                )
        
        return Image.fromarray(render_buffer)
    
    def _rasterize_triangle(
        self,
        render_buffer: np.ndarray,
        z_buffer: np.ndarray,
        texture: np.ndarray,
        p0: Tuple[int, int],
        p1: Tuple[int, int], 
        p2: Tuple[int, int],
        z0: float, z1: float, z2: float
    ):
        """삼각형 래스터화"""
        
        # 바운딩 박스 계산
        min_x = max(0, min(p0[0], p1[0], p2[0]))
        max_x = min(render_buffer.shape[1] - 1, max(p0[0], p1[0], p2[0]))
        min_y = max(0, min(p0[1], p1[1], p2[1]))
        max_y = min(render_buffer.shape[0] - 1, max(p0[1], p1[1], p2[1]))
        
        # 삼각형 내부 픽셀들 처리
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                # 무게중심 좌표 계산
                barycentric = self._calculate_barycentric(
                    (x, y), p0, p1, p2
                )
                
                if all(coord >= 0 for coord in barycentric):
                    # 픽셀이 삼각형 내부에 있음
                    alpha, beta, gamma = barycentric
                    
                    # 깊이 보간
                    z = alpha * z0 + beta * z1 + gamma * z2
                    
                    # Z-버퍼 테스트
                    if z > z_buffer[y, x]:
                        z_buffer[y, x] = z
                        
                        # 텍스처 좌표 계산
                        tex_x = int((alpha * 0.1 + beta * 0.9 + gamma * 0.5) * texture.shape[1])
                        tex_y = int((alpha * 0.1 + beta * 0.1 + gamma * 0.9) * texture.shape[0])
                        
                        tex_x = np.clip(tex_x, 0, texture.shape[1] - 1)
                        tex_y = np.clip(tex_y, 0, texture.shape[0] - 1)
                        
                        # 텍스처 색상 적용
                        render_buffer[y, x] = texture[tex_y, tex_x]
    
    def _calculate_barycentric(
        self, 
        point: Tuple[int, int],
        p0: Tuple[int, int],
        p1: Tuple[int, int], 
        p2: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """무게중심 좌표 계산"""
        
        x, y = point
        x0, y0 = p0
        x1, y1 = p1
        x2, y2 = p2
        
        # 삼각형 면적 계산
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        
        if abs(area) < 1e-10:
            return (0.0, 0.0, 0.0)
        
        # 무게중심 좌표
        alpha = ((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)) / area
        beta = ((x2 - x) * (y0 - y) - (x0 - x) * (y2 - y)) / area
        gamma = 1.0 - alpha - beta
        
        return (alpha, beta, gamma)
    
    def _analyze_lighting(self, person_image: Image.Image, body_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """조명 분석"""
        
        # 이미지에서 조명 방향 및 강도 분석
        cv_image = cv2.cvtColor(np.array(person_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 그라디언트 분석으로 조명 방향 추정
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 평균 그라디언트 방향
        avg_grad_x = np.mean(grad_x)
        avg_grad_y = np.mean(grad_y)
        
        # 조명 방향 (정규화)
        light_direction = np.array([avg_grad_x, avg_grad_y, 50])  # z축 기본값
        light_direction = light_direction / np.linalg.norm(light_direction)
        
        # 조명 강도 (밝기 분석)
        brightness = np.mean(gray)
        light_intensity = brightness / 255.0
        
        # 그림자 방향 (조명의 반대)
        shadow_direction = -light_direction
        
        return {
            'direction': light_direction,
            'intensity': light_intensity,
            'shadow_direction': shadow_direction,
            'ambient_light': 0.3,  # 환경광
            'color_temperature': 5500  # 색온도 (K)
        }
    
    def _apply_realistic_lighting(self, clothing_image: Image.Image, lighting_info: Dict[str, Any]) -> Image.Image:
        """사실적인 조명 적용"""
        
        cv_image = cv2.cvtColor(np.array(clothing_image), cv2.COLOR_RGB2BGR)
        
        # 조명 방향과 강도 적용
        light_direction = lighting_info['direction']
        light_intensity = lighting_info['intensity']
        
        # 법선 맵 생성 (단순화)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        normal_map = self._generate_normal_map(gray)
        
        # 각 픽셀에 조명 계산 적용
        lit_image = cv_image.copy().astype(np.float32)
        
        for y in range(cv_image.shape[0]):
            for x in range(cv_image.shape[1]):
                if np.any(cv_image[y, x] > 0):  # 투명하지 않은 픽셀만
                    # 법선 벡터
                    normal = normal_map[y, x]
                    
                    # 램버트 조명 모델
                    dot_product = max(0, np.dot(normal, light_direction))
                    lighting_factor = lighting_info['ambient_light'] + light_intensity * dot_product
                    
                    # 조명 적용
                    lit_image[y, x] *= lighting_factor
        
        # 클리핑 및 타입 변환
        lit_image = np.clip(lit_image, 0, 255).astype(np.uint8)
        
        return Image.fromarray(cv2.cvtColor(lit_image, cv2.COLOR_BGR2RGB))
    
    def _generate_normal_map(self, gray_image: np.ndarray) -> np.ndarray:
        """그레이스케일 이미지에서 법선 맵 생성"""
        
        # 소벨 필터로 그라디언트 계산
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 법선 벡터 계산
        normal_map = np.zeros((gray_image.shape[0], gray_image.shape[1], 3))
        
        for y in range(gray_image.shape[0]):
            for x in range(gray_image.shape[1]):
                # 법선 벡터 (x, y, z)
                nx = -grad_x[y, x] / 255.0
                ny = -grad_y[y, x] / 255.0
                nz = 1.0
                
                # 정규화
                length = np.sqrt(nx*nx + ny*ny + nz*nz)
                if length > 0:
                    normal_map[y, x] = [nx/length, ny/length, nz/length]
                else:
                    normal_map[y, x] = [0, 0, 1]
        
        return normal_map
    
    def _generate_realistic_shadows(
        self, 
        mesh: Dict[str, np.ndarray], 
        lighting_info: Dict[str, Any],
        width: int, 
        height: int
    ) -> np.ndarray:
        """사실적인 그림자 생성"""
        
        shadow_map = np.zeros((height, width), dtype=np.float32)
        
        vertices = mesh['vertices']
        light_direction = lighting_info['direction']
        
        # 각 정점에서 그림자 투영
        for vertex in vertices:
            # 그림자 투영 계산
            shadow_point = vertex - light_direction * vertex[2]
            
            # 2D 좌표로 변환
            shadow_x = int(shadow_point[0])
            shadow_y = int(shadow_point[1])
            
            # 그림자 맵에 그림자 추가
            if 0 <= shadow_x < width and 0 <= shadow_y < height:
                # 거리에 따른 그림자 강도
                distance = np.linalg.norm(shadow_point - vertex)
                shadow_intensity = max(0, 1.0 - distance / 100.0)
                
                # 가우시안 블러로 부드러운 그림자
                self._add_soft_shadow(shadow_map, shadow_x, shadow_y, shadow_intensity)
        
        # 그림자 맵 후처리
        shadow_map = cv2.GaussianBlur(shadow_map, (15, 15), 5)
        shadow_map = np.clip(shadow_map, 0, 0.7)  # 최대 그림자 강도 제한
        
        return shadow_map
    
    def _add_soft_shadow(self, shadow_map: np.ndarray, x: int, y: int, intensity: float):
        """부드러운 그림자 추가"""
        
        radius = 10
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                sx, sy = x + dx, y + dy
                if 0 <= sx < shadow_map.shape[1] and 0 <= sy < shadow_map.shape[0]:
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance <= radius:
                        falloff = max(0, 1.0 - distance / radius)
                        shadow_map[sy, sx] = max(shadow_map[sy, sx], intensity * falloff)
    
    def _calculate_occlusion(
        self, 
        clothing_regions: Dict[str, Dict], 
        fitted_mesh: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """오클루전(가려짐) 계산"""
        
        # 기본 마스크 (전체 투명)
        occlusion_mask = np.ones((512, 512), dtype=np.float32)
        
        # 의류가 덮는 영역은 원본 신체 부위를 가림
        if 'upper_body' in clothing_regions:
            region = clothing_regions['upper_body']
            if 'mask' in region:
                mask = region['mask']
                if mask.shape[:2] == occlusion_mask.shape:
                    # 의류 영역은 가려짐 (0에 가까운 값)
                    occlusion_mask[mask > 0] = 0.1
        
        return occlusion_mask
    
    def _advanced_composite(
        self,
        base_image: Image.Image,
        clothing_image: Image.Image,
        shadow_map: np.ndarray,
        occlusion_mask: np.ndarray
    ) -> Image.Image:
        """고급 합성"""
        
        base_array = np.array(base_image).astype(np.float32)
        clothing_array = np.array(clothing_image).astype(np.float32)
        
        # 이미지 크기 맞추기
        h, w = base_array.shape[:2]
        clothing_resized = cv2.resize(clothing_array, (w, h))
        shadow_resized = cv2.resize(shadow_map, (w, h))
        occlusion_resized = cv2.resize(occlusion_mask, (w, h))
        
        # 합성 결과
        result = base_array.copy()
        
        # 의류가 있는 영역 확인
        clothing_mask = np.any(clothing_resized > 0, axis=2)
        
        for y in range(h):
            for x in range(w):
                if clothing_mask[y, x]:
                    # 의류 픽셀이 있는 경우
                    clothing_color = clothing_resized[y, x]
                    base_color = base_array[y, x]
                    
                    # 오클루전 적용
                    occlusion_factor = occlusion_resized[y, x]
                    
                    # 의류와 기존 이미지 블렌딩
                    alpha = 0.9  # 의류 불투명도
                    blended = alpha * clothing_color + (1 - alpha) * base_color * occlusion_factor
                    
                    # 그림자 적용
                    shadow_factor = 1.0 - shadow_resized[y, x]
                    result[y, x] = blended * shadow_factor
                else:
                    # 의류가 없는 영역에는 그림자만 적용
                    shadow_factor = 1.0 - shadow_resized[y, x] * 0.3  # 약한 그림자
                    result[y, x] = base_array[y, x] * shadow_factor
        
        # 클리핑 및 타입 변환
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def _apply_high_quality_effects(self, image: Image.Image) -> Image.Image:
        """고품질 효과 적용"""
        
        # 1. 안티앨리어싱
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        smooth = cv2.bilateralFilter(cv_image, 9, 75, 75)
        
        # 2. 색상 보정
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 밝기 조정
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. 선명도 향상
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel * 0.1)
        
        return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    
    def _apply_ultra_quality_effects(self, image: Image.Image) -> Image.Image:
        """초고품질 효과 적용"""
        
        # 고품질 효과 먼저 적용
        high_quality = self._apply_high_quality_effects(image)
        
        cv_image = cv2.cvtColor(np.array(high_quality), cv2.COLOR_RGB2BGR)
        
        # 1. 고급 노이즈 제거
        denoised = cv2.fastNlMeansDenoisingColored(cv_image, None, 10, 10, 7, 21)
        
        # 2. 에지 보존 스무딩
        smooth = cv2.edgePreservingFilter(denoised, flags=1, sigma_s=50, sigma_r=0.4)
        
        # 3. HDR 톤 매핑 효과
        hdr_effect = cv2.createTonemap(gamma=1.2)
        hdr_mapped = hdr_effect.process(smooth.astype(np.float32) / 255.0)
        hdr_mapped = np.clip(hdr_mapped * 255, 0, 255).astype(np.uint8)
        
        # 4. 미세 디테일 향상
        detail_enhanced = cv2.detailEnhance(hdr_mapped, sigma_s=10, sigma_r=0.15)
        
        return Image.fromarray(cv2.cvtColor(detail_enhanced, cv2.COLOR_BGR2RGB))
    
    async def _post_process_and_analyze(
        self,
        result_image: Image.Image,
        body_analysis: Dict[str, Any],
        fitted_clothing: Dict[str, Any]
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """후처리 및 품질 분석"""
        
        # 1. 최종 이미지 후처리
        final_image = self._final_post_processing(result_image)
        
        # 2. 품질 메트릭 계산
        quality_metrics = {
            'fit_quality': fitted_clothing.get('fit_quality', 0.0),
            'pose_confidence': body_analysis.get('confidence', 0.0),
            'lighting_realism': self._calculate_lighting_realism(final_image),
            'overall_quality': 0.0
        }
        
        # 전체 품질 점수
        quality_metrics['overall_quality'] = (
            quality_metrics['fit_quality'] * 0.4 +
            quality_metrics['pose_confidence'] * 0.3 +
            quality_metrics['lighting_realism'] * 0.3
        )
        
        return final_image, quality_metrics
    
    def _final_post_processing(self, image: Image.Image) -> Image.Image:
        """최종 후처리"""
        
        # 1. 색상 균형 조정
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 2. 대비 향상
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. 최종 선명화
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel * 0.1)
        
        return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    
    def _calculate_lighting_realism(self, image: Image.Image) -> float:
        """조명 사실성 계산"""
        
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 그라디언트 분석
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 조명의 일관성 측정 (그라디언트 분산의 역수)
        consistency = 1.0 / (1.0 + np.var(gradient_magnitude))
        
        return min(1.0, consistency * 2.0)
    
    async def _create_fallback_result(
        self, 
        person_image: Image.Image, 
        clothing_image: Image.Image
    ) -> Image.Image:
        """에러 발생시 기본 결과 생성"""
        
        logger.info("⚠️ 기본 합성 모드로 대체")
        
        result = person_image.copy()
        
        # 간단한 오버레이
        clothing_resized = clothing_image.resize((200, 200))
        result.paste(clothing_resized, (150, 100), clothing_resized)
        
        # 에러 표시
        draw = ImageDraw.Draw(result)
        draw.text((10, 10), "Processing Error - Basic Mode", fill='red')
        
        return result

# 전역 실제 가상 피팅 서비스
real_virtual_fitting = RealVirtualFittingService()