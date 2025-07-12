# backend/app/services/clothing_3d_modeling.py
"""
의류 3D 모델링 및 변형
- 의류 메쉬 생성
- 체형에 맞는 3D 변형
- 물리 시뮬레이션
- 텍스처 매핑
"""

import numpy as np
import cv2
from PIL import Image
import torch
import scipy.spatial.distance as dist
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class Clothing3DModeler:
    """의류 3D 모델링 및 변형"""
    
    def __init__(self):
        # 3D 메쉬 관련 설정
        self.mesh_resolution = 64
        self.texture_size = 512
        
        # 물리 시뮬레이션 파라미터
        self.fabric_properties = {
            'cotton': {'stiffness': 0.3, 'stretch': 0.1, 'drape': 0.7},
            'denim': {'stiffness': 0.8, 'stretch': 0.05, 'drape': 0.3},
            'silk': {'stiffness': 0.1, 'stretch': 0.2, 'drape': 0.9},
            'leather': {'stiffness': 0.9, 'stretch': 0.02, 'drape': 0.2}
        }
    
    async def create_clothing_mesh(
        self, 
        clothing_image: Image.Image,
        clothing_type: str = "shirt",
        fabric_type: str = "cotton"
    ) -> Dict[str, Any]:
        """의류 3D 메쉬 생성"""
        
        logger.info(f"🎽 의류 3D 메쉬 생성: {clothing_type}")
        
        try:
            # 1. 의류 윤곽선 추출
            contour = self._extract_clothing_contour(clothing_image)
            
            # 2. 2D에서 3D 메쉬로 변환
            mesh_3d = self._generate_3d_mesh(contour, clothing_type)
            
            # 3. 텍스처 추출 및 매핑
            texture = self._extract_texture(clothing_image, contour)
            
            # 4. 물리 속성 적용
            physics_properties = self.fabric_properties.get(fabric_type, self.fabric_properties['cotton'])
            
            return {
                'mesh': mesh_3d,
                'texture': texture,
                'contour': contour,
                'physics': physics_properties,
                'clothing_type': clothing_type,
                'fabric_type': fabric_type
            }
            
        except Exception as e:
            logger.error(f"❌ 의류 메쉬 생성 실패: {e}")
            raise
    
    def _extract_clothing_contour(self, clothing_image: Image.Image) -> np.ndarray:
        """의류 윤곽선 추출"""
        
        # PIL을 OpenCV로 변환
        cv_image = cv2.cvtColor(np.array(clothing_image), cv2.COLOR_RGB2BGR)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 가우시안 블러
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 적응적 임계값
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 윤곽선 선택 (의류 본체)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 윤곽선 단순화
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            return simplified_contour.reshape(-1, 2)
        
        # 윤곽선을 찾지 못한 경우 기본 사각형 반환
        h, w = cv_image.shape[:2]
        return np.array([[w//4, h//4], [3*w//4, h//4], [3*w//4, 3*h//4], [w//4, 3*h//4]])
    
    def _generate_3d_mesh(self, contour: np.ndarray, clothing_type: str) -> Dict[str, np.ndarray]:
        """2D 윤곽선에서 3D 메쉬 생성"""
        
        # 의류 타입별 3D 형태 정의
        depth_profiles = {
            'shirt': self._create_shirt_depth_profile,
            'pants': self._create_pants_depth_profile,
            'dress': self._create_dress_depth_profile,
            'jacket': self._create_jacket_depth_profile
        }
        
        depth_func = depth_profiles.get(clothing_type, depth_profiles['shirt'])
        
        # 3D 좌표 생성
        vertices_3d = []
        faces = []
        normals = []
        
        # 윤곽선을 기반으로 메쉬 정점 생성
        num_points = len(contour)
        
        for i, point in enumerate(contour):
            x, y = point
            
            # 깊이 계산 (의류 타입에 따라)
            depth = depth_func(x, y, contour)
            
            # 3D 좌표
            vertices_3d.append([x, y, depth])
            
            # 법선 벡터 계산
            normal = self._calculate_normal(i, contour, depth)
            normals.append(normal)
        
        # 면(face) 생성 - 삼각형 메쉬
        for i in range(num_points - 2):
            faces.append([0, i + 1, i + 2])
        
        return {
            'vertices': np.array(vertices_3d),
            'faces': np.array(faces),
            'normals': np.array(normals),
            'uv_coords': self._generate_uv_coordinates(contour)
        }
    
    def _create_shirt_depth_profile(self, x: float, y: float, contour: np.ndarray) -> float:
        """셔츠 깊이 프로필"""
        
        # 가슴 부분이 가장 돌출
        center_x = np.mean(contour[:, 0])
        center_y = np.mean(contour[:, 1])
        
        # 중앙에서의 거리
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.max([np.sqrt((pt[0] - center_x)**2 + (pt[1] - center_y)**2) for pt in contour])
        
        # 가슴 부분 (상단 1/3)이 가장 돌출되도록
        if y < center_y:  # 상단부
            depth = 20 * (1 - dist_from_center / max_dist) * 1.5
        else:  # 하단부
            depth = 15 * (1 - dist_from_center / max_dist)
        
        return max(0, depth)
    
    def _create_pants_depth_profile(self, x: float, y: float, contour: np.ndarray) -> float:
        """바지 깊이 프로필"""
        
        # 엉덩이와 허벅지 부분이 돌출
        center_x = np.mean(contour[:, 0])
        center_y = np.mean(contour[:, 1])
        
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.max([np.sqrt((pt[0] - center_x)**2 + (pt[1] - center_y)**2) for pt in contour])
        
        # 상단(엉덩이)과 중간(허벅지)이 돌출
        if y < center_y + 50:  # 상단부
            depth = 25 * (1 - dist_from_center / max_dist)
        else:  # 하단부 (종아리)
            depth = 12 * (1 - dist_from_center / max_dist)
        
        return max(0, depth)
    
    def _create_dress_depth_profile(self, x: float, y: float, contour: np.ndarray) -> float:
        """드레스 깊이 프로필"""
        # 셔츠와 유사하지만 더 넓은 하단부
        return self._create_shirt_depth_profile(x, y, contour) * 0.8
    
    def _create_jacket_depth_profile(self, x: float, y: float, contour: np.ndarray) -> float:
        """재킷 깊이 프로필"""
        # 셔츠보다 두꺼운 프로필
        return self._create_shirt_depth_profile(x, y, contour) * 1.3
    
    def _calculate_normal(self, index: int, contour: np.ndarray, depth: float) -> np.ndarray:
        """법선 벡터 계산"""
        
        num_points = len(contour)
        
        # 이전, 현재, 다음 점
        prev_idx = (index - 1) % num_points
        next_idx = (index + 1) % num_points
        
        prev_point = contour[prev_idx]
        curr_point = contour[index]
        next_point = contour[next_idx]
        
        # 접선 벡터
        tangent = next_point - prev_point
        tangent = tangent / np.linalg.norm(tangent)
        
        # 법선 벡터 (2D에서 90도 회전)
        normal_2d = np.array([-tangent[1], tangent[0]])
        
        # 3D 법선 벡터 (z축 성분 추가)
        normal_3d = np.array([normal_2d[0], normal_2d[1], 0.1])
        normal_3d = normal_3d / np.linalg.norm(normal_3d)
        
        return normal_3d
    
    def _generate_uv_coordinates(self, contour: np.ndarray) -> np.ndarray:
        """UV 좌표 생성 (텍스처 매핑용)"""
        
        # 윤곽선을 [0, 1] 범위로 정규화
        min_x, min_y = np.min(contour, axis=0)
        max_x, max_y = np.max(contour, axis=0)
        
        uv_coords = []
        for point in contour:
            u = (point[0] - min_x) / (max_x - min_x)
            v = (point[1] - min_y) / (max_y - min_y)
            uv_coords.append([u, v])
        
        return np.array(uv_coords)
    
    def _extract_texture(self, clothing_image: Image.Image, contour: np.ndarray) -> np.ndarray:
        """의류 텍스처 추출"""
        
        # 윤곽선 영역만 추출
        mask = np.zeros(clothing_image.size[::-1], dtype=np.uint8)
        cv2.fillPoly(mask, [contour.astype(np.int32)], 255)
        
        # 마스크 적용
        clothing_array = np.array(clothing_image)
        masked_clothing = cv2.bitwise_and(clothing_array, clothing_array, mask=mask)
        
        # 정사각형 텍스처로 리사이즈
        texture = cv2.resize(masked_clothing, (self.texture_size, self.texture_size))
        
        return texture
    
    async def fit_clothing_to_body(
        self,
        clothing_mesh: Dict[str, Any],
        body_analysis: Dict[str, Any],
        body_measurements: Dict[str, float]
    ) -> Dict[str, Any]:
        """체형에 맞게 의류 변형"""
        
        logger.info("👤 체형에 맞는 의류 변형 시작")
        
        try:
            # 1. 체형 분석 데이터 추출
            clothing_regions = body_analysis['clothing_regions']
            pose_3d = body_analysis['pose_3d']
            
            # 2. 의류 타입에 따른 피팅 영역 선택
            clothing_type = clothing_mesh['clothing_type']
            target_region = self._select_target_region(clothing_type, clothing_regions)
            
            # 3. 메쉬 변형 계산
            deformed_mesh = self._deform_mesh_to_body(
                clothing_mesh['mesh'], 
                target_region,
                body_measurements,
                clothing_mesh['physics']
            )
            
            # 4. 물리 시뮬레이션 적용
            simulated_mesh = self._apply_physics_simulation(
                deformed_mesh, 
                clothing_mesh['physics'],
                pose_3d
            )
            
            # 5. 텍스처 조정
            adjusted_texture = self._adjust_texture_mapping(
                clothing_mesh['texture'],
                simulated_mesh,
                clothing_mesh['mesh']
            )
            
            return {
                'fitted_mesh': simulated_mesh,
                'adjusted_texture': adjusted_texture,
                'fit_quality': self._calculate_fit_quality(simulated_mesh, target_region),
                'deformation_info': {
                    'scale_factor': self._calculate_scale_factor(clothing_mesh['mesh'], simulated_mesh),
                    'stress_points': self._identify_stress_points(simulated_mesh)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 의류 피팅 실패: {e}")
            raise
    
    def _select_target_region(self, clothing_type: str, clothing_regions: Dict[str, Dict]) -> Dict[str, Any]:
        """의류 타입에 따른 타겟 영역 선택"""
        
        region_mapping = {
            'shirt': 'upper_body',
            'pants': 'lower_body', 
            'dress': 'upper_body',  # 전신이지만 상체 기준
            'jacket': 'upper_body'
        }
        
        region_key = region_mapping.get(clothing_type, 'upper_body')
        return clothing_regions.get(region_key, {})
    
    def _deform_mesh_to_body(
        self,
        mesh: Dict[str, np.ndarray],
        target_region: Dict[str, Any],
        body_measurements: Dict[str, float],
        physics_props: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """메쉬를 체형에 맞게 변형"""
        
        vertices = mesh['vertices'].copy()
        
        if not target_region or 'bounds' not in target_region:
            return mesh
        
        bounds = target_region['bounds']
        
        # 스케일링 팩터 계산
        mesh_width = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
        mesh_height = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        
        target_width = bounds['width']
        target_height = bounds['height']
        
        scale_x = target_width / mesh_width if mesh_width > 0 else 1.0
        scale_y = target_height / mesh_height if mesh_height > 0 else 1.0
        
        # 체형 측정치 기반 조정
        if 'shoulder_width' in body_measurements:
            shoulder_scale = body_measurements['shoulder_width'] / mesh_width if mesh_width > 0 else 1.0
            scale_x = (scale_x + shoulder_scale) / 2
        
        # 스케일링 적용
        vertices[:, 0] *= scale_x
        vertices[:, 1] *= scale_y
        
        # 위치 조정
        vertices[:, 0] += bounds['x']
        vertices[:, 1] += bounds['y']
        
        # 물리 속성을 고려한 추가 변형
        stretch_factor = physics_props.get('stretch', 0.1)
        vertices = self._apply_stretch_deformation(vertices, stretch_factor)
        
        return {
            'vertices': vertices,
            'faces': mesh['faces'],
            'normals': self._recalculate_normals(vertices, mesh['faces']),
            'uv_coords': mesh['uv_coords']
        }
    
    def _apply_stretch_deformation(self, vertices: np.ndarray, stretch_factor: float) -> np.ndarray:
        """신축성을 고려한 변형"""
        
        # 중심점 계산
        center = np.mean(vertices, axis=0)
        
        # 중심에서의 거리에 따른 신축
        for i, vertex in enumerate(vertices):
            direction = vertex - center
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                # 거리에 비례한 신축 적용
                stretch_amount = stretch_factor * distance / 100
                vertices[i] = center + direction * (1 + stretch_amount)
        
        return vertices
    
    def _apply_physics_simulation(
        self,
        mesh: Dict[str, np.ndarray],
        physics_props: Dict[str, float],
        pose_3d: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """물리 시뮬레이션 적용"""
        
        vertices = mesh['vertices'].copy()
        
        # 중력 효과
        gravity_effect = physics_props.get('drape', 0.5)
        vertices[:, 1] += gravity_effect * 2  # y축이 아래 방향
        
        # 바람/움직임 효과 (포즈 기반)
        if 'angles' in pose_3d:
            movement_factor = self._calculate_movement_factor(pose_3d['angles'])
            vertices = self._apply_movement_deformation(vertices, movement_factor)
        
        # 주름 효과
        stiffness = physics_props.get('stiffness', 0.5)
        vertices = self._add_wrinkle_effect(vertices, stiffness)
        
        return {
            'vertices': vertices,
            'faces': mesh['faces'],
            'normals': self._recalculate_normals(vertices, mesh['faces']),
            'uv_coords': mesh['uv_coords']
        }
    
    def _calculate_movement_factor(self, joint_angles: Dict[str, float]) -> float:
        """관절 각도에 따른 움직임 팩터 계산"""
        
        # 팔꿈치, 무릎 각도를 기반으로 움직임 정도 계산
        movement = 0.0
        
        if 'left_elbow' in joint_angles:
            elbow_angle = joint_angles['left_elbow']
            # 각도가 클수록 (팔을 굽힐수록) 움직임 증가
            movement += abs(elbow_angle - 180) / 180
        
        return min(movement, 1.0)
    
    def _apply_movement_deformation(self, vertices: np.ndarray, movement_factor: float) -> np.ndarray:
        """움직임에 따른 변형"""
        
        if movement_factor < 0.1:
            return vertices
        
        # 움직임에 따른 미세한 변형 추가
        noise = np.random.normal(0, movement_factor * 0.5, vertices.shape)
        vertices += noise
        
        return vertices
    
    def _add_wrinkle_effect(self, vertices: np.ndarray, stiffness: float) -> np.ndarray:
        """주름 효과 추가"""
        
        # 낮은 강성일수록 주름이 많음
        wrinkle_intensity = 1.0 - stiffness
        
        if wrinkle_intensity < 0.1:
            return vertices
        
        # 주름은 주로 z축 방향의 미세한 변화
        wrinkle_noise = np.random.normal(0, wrinkle_intensity * 0.3, (len(vertices), 1))
        vertices[:, 2:3] += wrinkle_noise
        
        return vertices
    
    def _recalculate_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """법선 벡터 재계산"""
        
        normals = np.zeros_like(vertices)
        
        for face in faces:
            if len(face) >= 3:
                v0, v1, v2 = vertices[face[:3]]
                
                # 외적으로 법선 계산
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                
                # 정규화
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                
                # 면에 속한 정점들에 법선 누적
                for vertex_idx in face[:3]:
                    normals[vertex_idx] += normal
        
        # 최종 정규화
        for i in range(len(normals)):
            norm = np.linalg.norm(normals[i])
            if norm > 0:
                normals[i] = normals[i] / norm
        
        return normals
    
    def _adjust_texture_mapping(
        self,
        original_texture: np.ndarray,
        new_mesh: Dict[str, np.ndarray],
        original_mesh: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """변형된 메쉬에 맞게 텍스처 조정"""
        
        # 메쉬 변형에 따른 텍스처 스트레칭 보정
        scale_factor = self._calculate_scale_factor(original_mesh, new_mesh)
        
        # 텍스처 보정 적용
        if scale_factor['x'] != 1.0 or scale_factor['y'] != 1.0:
            h, w = original_texture.shape[:2]
            new_w = int(w * scale_factor['x'])
            new_h = int(h * scale_factor['y'])
            
            adjusted_texture = cv2.resize(original_texture, (new_w, new_h))
            
            # 원래 크기로 패딩 또는 크롭
            if new_w > w or new_h > h:
                # 크롭
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                adjusted_texture = adjusted_texture[start_y:start_y+h, start_x:start_x+w]
            else:
                # 패딩
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                adjusted_texture = cv2.copyMakeBorder(
                    adjusted_texture, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x,
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
            
            return adjusted_texture
        
        return original_texture
    
    def _calculate_scale_factor(
        self, 
        original_mesh: Dict[str, np.ndarray], 
        new_mesh: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """메쉬 변형의 스케일 팩터 계산"""
        
        orig_vertices = original_mesh['vertices']
        new_vertices = new_mesh['vertices']
        
        orig_bounds = {
            'min_x': np.min(orig_vertices[:, 0]),
            'max_x': np.max(orig_vertices[:, 0]),
            'min_y': np.min(orig_vertices[:, 1]),
            'max_y': np.max(orig_vertices[:, 1])
        }
        
        new_bounds = {
            'min_x': np.min(new_vertices[:, 0]),
            'max_x': np.max(new_vertices[:, 0]),
            'min_y': np.min(new_vertices[:, 1]),
            'max_y': np.max(new_vertices[:, 1])
        }
        
        scale_x = (new_bounds['max_x'] - new_bounds['min_x']) / (orig_bounds['max_x'] - orig_bounds['min_x'])
        scale_y = (new_bounds['max_y'] - new_bounds['min_y']) / (orig_bounds['max_y'] - orig_bounds['min_y'])
        
        return {'x': scale_x, 'y': scale_y}
    
    def _calculate_fit_quality(self, mesh: Dict[str, np.ndarray], target_region: Dict[str, Any]) -> float:
        """피팅 품질 계산"""
        
        if not target_region or 'area' not in target_region:
            return 0.5
        
        # 메쉬 영역과 타겟 영역의 매칭도
        mesh_area = len(mesh['vertices'])  # 간단한 추정
        target_area = target_region['area']
        
        if target_area > 0:
            area_ratio = min(mesh_area / target_area, target_area / mesh_area)
            return area_ratio
        
        return 0.5
    
    def _identify_stress_points(self, mesh: Dict[str, np.ndarray]) -> List[int]:
        """스트레스 포인트 식별"""
        
        vertices = mesh['vertices']
        stress_points = []
        
        # 곡률이 높은 지점을 스트레스 포인트로 식별
        for i in range(len(vertices)):
            neighbors = self._find_neighbor_vertices(i, mesh['faces'])
            if len(neighbors) >= 3:
                curvature = self._calculate_curvature(i, neighbors, vertices)
                if curvature > 0.1:  # 임계값
                    stress_points.append(i)
        
        return stress_points
    
    def _find_neighbor_vertices(self, vertex_idx: int, faces: np.ndarray) -> List[int]:
        """인접 정점 찾기"""
        
        neighbors = set()
        
        for face in faces:
            if vertex_idx in face:
                for v in face:
                    if v != vertex_idx:
                        neighbors.add(v)
        
        return list(neighbors)
    
    def _calculate_curvature(self, vertex_idx: int, neighbors: List[int], vertices: np.ndarray) -> float:
        """곡률 계산"""
        
        if len(neighbors) < 3:
            return 0.0
        
        center = vertices[vertex_idx]
        neighbor_points = vertices[neighbors]
        
        # 평균 거리를 기반으로 간단한 곡률 추정
        distances = [np.linalg.norm(point - center) for point in neighbor_points]
        curvature = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0.0
        
        return curvature

# 전역 3D 모델러
clothing_3d_modeler = Clothing3DModeler()