"""
MyCloset AI 8단계: 품질 평가 (Quality Assessment)
SSIM, LPIPS, FID, Custom Fit Score 기반 자동 점수 및 추천 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import math

class PerceptualLoss(nn.Module):
    """LPIPS (Learned Perceptual Image Patch Similarity) 계산"""
    
    def __init__(self, net='vgg', device='cpu'):
        super().__init__()
        self.device = device
        
        if net == 'vgg':
            self.net = models.vgg16(pretrained=True).features
        elif net == 'alex':
            self.net = models.alexnet(pretrained=True).features
        else:
            raise ValueError("지원되지 않는 네트워크")
        
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
        
        # 정규화 파라미터
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        # 특징 추출 레이어
        self.feature_layers = [4, 9, 16, 23, 30]  # VGG16 기준

    def normalize_tensor(self, x):
        """입력 텐서 정규화"""
        return (x - self.mean) / self.std

    def extract_features(self, x):
        """다층 특징 추출"""
        x = self.normalize_tensor(x)
        features = []
        
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        
        return features

    def forward(self, x1, x2):
        """LPIPS 거리 계산"""
        features1 = self.extract_features(x1)
        features2 = self.extract_features(x2)
        
        lpips_distance = 0
        
        for f1, f2 in zip(features1, features2):
            # 채널별 정규화
            f1_norm = F.normalize(f1, dim=1)
            f2_norm = F.normalize(f2, dim=1)
            
            # 코사인 거리 계산
            distance = (f1_norm - f2_norm) ** 2
            distance = torch.mean(distance, dim=[2, 3])  # 공간 차원 평균
            distance = torch.mean(distance, dim=1)       # 배치 차원 평균
            
            lpips_distance += distance
        
        return lpips_distance / len(features1)

class FIDCalculator:
    """FID (Fréchet Inception Distance) 계산기"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # InceptionV3 모델 로드
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()  # 마지막 fc 레이어 제거
        self.inception.eval()
        self.inception.to(device)
        
        for param in self.inception.parameters():
            param.requires_grad = False
        
        # 전처리 변환
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def get_inception_features(self, images):
        """Inception 특징 추출"""
        features = []
        
        with torch.no_grad():
            for img in images:
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                
                img = self.transform(img)
                feat = self.inception(img)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)

    def calculate_fid(self, real_images, fake_images):
        """FID 계산"""
        try:
            # 특징 추출
            real_features = self.get_inception_features(real_images)
            fake_features = self.get_inception_features(fake_images)
            
            # 평균과 공분산 계산
            mu_real = np.mean(real_features, axis=0)
            mu_fake = np.mean(fake_features, axis=0)
            
            sigma_real = np.cov(real_features, rowvar=False)
            sigma_fake = np.cov(fake_features, rowvar=False)
            
            # FID 계산
            diff = mu_real - mu_fake
            
            # 수치적 안정성을 위한 정규화
            covmean = self._matrix_sqrt(sigma_real.dot(sigma_fake))
            
            fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
            
            return float(fid)
            
        except Exception as e:
            logging.warning(f"FID 계산 실패: {e}")
            return float('inf')

    def _matrix_sqrt(self, matrix):
        """행렬의 제곱근 계산"""
        try:
            # 특이값 분해 사용
            u, s, vh = np.linalg.svd(matrix)
            sqrt_s = np.sqrt(np.maximum(s, 1e-8))  # 수치적 안정성
            return u.dot(np.diag(sqrt_s)).dot(vh)
        except:
            # 실패 시 단위 행렬 반환
            return np.eye(matrix.shape[0])

class CustomFitScorer:
    """맞춤형 피팅 점수 계산기"""
    
    def __init__(self):
        pass
    
    def calculate_fit_score(self, result_data: Dict[str, Any]) -> Dict[str, float]:
        """종합 피팅 점수 계산"""
        scores = {}
        
        # 1. 의류 커버리지 점수
        scores['coverage'] = self._calculate_coverage_score(result_data)
        
        # 2. 형태 일치 점수
        scores['shape_consistency'] = self._calculate_shape_consistency(result_data)
        
        # 3. 색상 보존 점수
        scores['color_preservation'] = self._calculate_color_preservation(result_data)
        
        # 4. 경계 자연스러움 점수
        scores['boundary_naturalness'] = self._calculate_boundary_naturalness(result_data)
        
        # 5. 포즈 일치 점수
        scores['pose_alignment'] = self._calculate_pose_alignment(result_data)
        
        # 6. 전체 현실감 점수
        scores['realism'] = self._calculate_realism_score(result_data)
        
        # 종합 점수 계산
        weights = {
            'coverage': 0.2,
            'shape_consistency': 0.2,
            'color_preservation': 0.15,
            'boundary_naturalness': 0.15,
            'pose_alignment': 0.15,
            'realism': 0.15
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights)
        scores['overall'] = overall_score
        
        return scores
    
    def _calculate_coverage_score(self, result_data: Dict[str, Any]) -> float:
        """의류 커버리지 점수"""
        try:
            if 'cloth_mask' in result_data and 'target_area' in result_data:
                cloth_mask = result_data['cloth_mask']
                target_area = result_data['target_area']
                
                # 목표 영역 대비 의류 커버리지 비율
                coverage_ratio = np.sum(cloth_mask) / max(np.sum(target_area), 1)
                
                # 0.8~1.2 범위에서 최적
                optimal_coverage = 1.0
                score = 1.0 - abs(coverage_ratio - optimal_coverage) / optimal_coverage
                
                return max(0.0, min(1.0, score))
            
            return 0.7  # 기본값
            
        except Exception as e:
            logging.warning(f"커버리지 점수 계산 실패: {e}")
            return 0.5
    
    def _calculate_shape_consistency(self, result_data: Dict[str, Any]) -> float:
        """형태 일치 점수"""
        try:
            if 'original_cloth' in result_data and 'warped_cloth' in result_data:
                original = result_data['original_cloth']
                warped = result_data['warped_cloth']
                
                # 윤곽선 비교
                orig_contours = self._extract_contours(original)
                warp_contours = self._extract_contours(warped)
                
                if orig_contours and warp_contours:
                    # 후보군 매칭
                    similarity = cv2.matchShapes(
                        orig_contours[0], warp_contours[0], 
                        cv2.CONTOURS_MATCH_I1, 0
                    )
                    
                    # 유사도를 점수로 변환 (낮을수록 좋음)
                    score = 1.0 / (1.0 + similarity)
                    return score
            
            return 0.8  # 기본값
            
        except Exception as e:
            logging.warning(f"형태 일치 점수 계산 실패: {e}")
            return 0.6
    
    def _calculate_color_preservation(self, result_data: Dict[str, Any]) -> float:
        """색상 보존 점수"""
        try:
            if 'original_cloth' in result_data and 'result_image' in result_data:
                original = result_data['original_cloth']
                result = result_data['result_image']
                
                # 히스토그램 비교
                hist_similarity = 0
                
                for c in range(3):  # RGB 채널
                    hist1 = cv2.calcHist([original], [c], None, [256], [0, 256])
                    hist2 = cv2.calcHist([result], [c], None, [256], [0, 256])
                    
                    # 정규화
                    hist1 = hist1 / np.sum(hist1)
                    hist2 = hist2 / np.sum(hist2)
                    
                    # 상관계수
                    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    hist_similarity += correlation
                
                return hist_similarity / 3.0
            
            return 0.8  # 기본값
            
        except Exception as e:
            logging.warning(f"색상 보존 점수 계산 실패: {e}")
            return 0.7
    
    def _calculate_boundary_naturalness(self, result_data: Dict[str, Any]) -> float:
        """경계 자연스러움 점수"""
        try:
            if 'result_image' in result_data and 'cloth_mask' in result_data:
                image = result_data['result_image']
                mask = result_data['cloth_mask']
                
                # 경계 그래디언트 계산
                boundary = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
                
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                
                # 경계 영역에서의 그래디언트 변화
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
                
                # 경계에서의 평균 그래디언트
                boundary_gradient = gradient_mag[boundary > 0]
                
                if len(boundary_gradient) > 0:
                    avg_gradient = np.mean(boundary_gradient)
                    # 적당한 그래디언트가 자연스러움 (너무 급격하지 않게)
                    naturalness = 1.0 / (1.0 + avg_gradient / 50.0)
                    return naturalness
            
            return 0.8  # 기본값
            
        except Exception as e:
            logging.warning(f"경계 자연스러움 점수 계산 실패: {e}")
            return 0.7
    
    def _calculate_pose_alignment(self, result_data: Dict[str, Any]) -> float:
        """포즈 일치 점수"""
        try:
            if 'pose_keypoints' in result_data and 'cloth_keypoints' in result_data:
                person_kp = result_data['pose_keypoints']
                cloth_kp = result_data['cloth_keypoints']
                
                # 주요 키포인트들의 거리 계산
                important_points = [2, 5, 8, 11]  # 어깨, 엉덩이
                distances = []
                
                for idx in important_points:
                    if (idx < len(person_kp) and idx < len(cloth_kp) and 
                        person_kp[idx][2] > 0.5 and cloth_kp[idx][2] > 0.5):
                        
                        dist = np.linalg.norm(
                            np.array(person_kp[idx][:2]) - np.array(cloth_kp[idx][:2])
                        )
                        distances.append(dist)
                
                if distances:
                    avg_distance = np.mean(distances)
                    # 거리가 작을수록 좋은 점수
                    alignment_score = 1.0 / (1.0 + avg_distance / 20.0)
                    return alignment_score
            
            return 0.8  # 기본값
            
        except Exception as e:
            logging.warning(f"포즈 일치 점수 계산 실패: {e}")
            return 0.7
    
    def _calculate_realism_score(self, result_data: Dict[str, Any]) -> float:
        """전체 현실감 점수"""
        try:
            if 'result_image' in result_data:
                image = result_data['result_image']
                
                # 노이즈 레벨 평가
                noise_score = self._evaluate_noise_level(image)
                
                # 블러 정도 평가
                sharpness_score = self._evaluate_sharpness(image)
                
                # 아티팩트 검출
                artifact_score = self._detect_artifacts(image)
                
                # 종합 현실감 점수
                realism = (noise_score + sharpness_score + artifact_score) / 3.0
                
                return realism
            
            return 0.8  # 기본값
            
        except Exception as e:
            logging.warning(f"현실감 점수 계산 실패: {e}")
            return 0.7
    
    def _extract_contours(self, image: np.ndarray) -> List:
        """윤곽선 추출"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 이진화
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def _evaluate_noise_level(self, image: np.ndarray) -> float:
        """노이즈 레벨 평가"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 라플라시안 분산으로 노이즈 추정
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 적당한 분산이 좋은 점수 (너무 높으면 노이즈, 너무 낮으면 과도한 스무딩)
        optimal_var = 500
        noise_score = 1.0 - abs(laplacian_var - optimal_var) / optimal_var
        
        return max(0.0, min(1.0, noise_score))
    
    def _evaluate_sharpness(self, image: np.ndarray) -> float:
        """선명도 평가"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 소벨 그래디언트로 선명도 측정
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        sharpness = np.mean(gradient_magnitude)
        
        # 정규화 (일반적으로 20-80 범위)
        normalized_sharpness = min(sharpness / 100.0, 1.0)
        
        return normalized_sharpness
    
    def _detect_artifacts(self, image: np.ndarray) -> float:
        """아티팩트 검출"""
        # 간단한 아티팩트 검출 (고주파 노이즈 패턴)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 고주파 필터
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_freq = cv2.filter2D(gray, -1, kernel)
        
        # 비정상적인 고주파 성분 검출
        artifact_level = np.std(high_freq)
        
        # 적당한 레벨이 좋음
        artifact_score = 1.0 / (1.0 + artifact_level / 20.0)
        
        return artifact_score

class QualityAssessmentStep:
    """8단계: 품질 평가 실행 클래스"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # 품질 평가 컴포넌트 초기화
        self.perceptual_loss = PerceptualLoss(device=device)
        self.fid_calculator = FIDCalculator(device=device)
        self.fit_scorer = CustomFitScorer()
        
        # 품질 기준 임계값
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'fair': 0.6,
            'poor': 0.4
        }

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """품질 평가 메인 처리"""
        try:
            # 입력 데이터 추출
            result_image = input_data["result_image"]
            original_person = input_data["original_person"] 
            original_cloth = input_data["original_cloth"]
            
            # 다중 메트릭 평가 실행
            quality_metrics = await self._comprehensive_quality_assessment(
                result_image, original_person, original_cloth, input_data
            )
            
            # 품질 등급 분류
            quality_grade = self._classify_quality_grade(quality_metrics)
            
            # 개선 추천사항 생성
            recommendations = await self._generate_recommendations(quality_metrics, input_data)
            
            # 사용자 친화적 점수 변환
            user_scores = self._convert_to_user_scores(quality_metrics)
            
            return {
                "quality_metrics": quality_metrics,
                "quality_grade": quality_grade,
                "user_scores": user_scores,
                "recommendations": recommendations,
                "overall_score": quality_metrics.get("overall_score", 0.0),
                "metadata": {
                    "assessment_method": "Multi-metric Evaluation",
                    "metrics_used": list(quality_metrics.keys()),
                    "processing_time": quality_metrics.get("processing_time", 0.0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"품질 평가 처리 중 오류: {str(e)}")
            raise

    async def _comprehensive_quality_assessment(
        self, 
        result_image: torch.Tensor,
        original_person: Image.Image,
        original_cloth: Image.Image,
        input_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """종합 품질 평가"""
        import time
        start_time = time.time()
        
        metrics = {}
        
        # 텐서를 PIL Image로 변환
        result_pil = self._tensor_to_pil(result_image)
        
        # 1. SSIM 계산
        metrics["ssim"] = await self._calculate_ssim(result_pil, original_person)
        
        # 2. LPIPS 계산
        metrics["lpips"] = await self._calculate_lpips(result_image, original_person)
        
        # 3. FID 계산 (배치가 있는 경우)
        metrics["fid"] = await self._calculate_fid(result_pil, original_person)
        
        # 4. 커스텀 피팅 점수
        fit_scores = self.fit_scorer.calculate_fit_score(input_data)
        metrics.update({f"fit_{k}": v for k, v in fit_scores.items()})
        
        # 5. 추가 메트릭들
        metrics["color_fidelity"] = await self._calculate_color_fidelity(result_pil, original_cloth)
        metrics["texture_preservation"] = await self._calculate_texture_preservation(result_pil, original_cloth)
        metrics["geometric_accuracy"] = await self._calculate_geometric_accuracy(input_data)
        
        # 종합 점수 계산
        metrics["overall_score"] = self._calculate_overall_score(metrics)
        
        metrics["processing_time"] = time.time() - start_time
        
        return metrics

    async def _calculate_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """SSIM 계산"""
        try:
            # PIL을 numpy로 변환
            arr1 = np.array(img1.convert('RGB'))
            arr2 = np.array(img2.convert('RGB'))
            
            # 크기 맞추기
            if arr1.shape != arr2.shape:
                arr2 = cv2.resize(arr2, (arr1.shape[1], arr1.shape[0]))
            
            # 그레이스케일 변환
            gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)
            
            # SSIM 계산
            mu1 = cv2.GaussianBlur(gray1.astype(np.float32), (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(gray2.astype(np.float32), (11, 11), 1.5)
            
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = cv2.GaussianBlur(gray1.astype(np.float32) * gray1.astype(np.float32), (11, 11), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(gray2.astype(np.float32) * gray2.astype(np.float32), (11, 11), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(gray1.astype(np.float32) * gray2.astype(np.float32), (11, 11), 1.5) - mu1_mu2
            
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            return float(np.mean(ssim_map))
            
        except Exception as e:
            self.logger.warning(f"SSIM 계산 실패: {e}")
            return 0.5

    async def _calculate_lpips(self, result_tensor: torch.Tensor, reference_image: Image.Image) -> float:
        """LPIPS 계산"""
        try:
            # reference 이미지를 텐서로 변환
            transform = transforms.Compose([
                transforms.Resize((result_tensor.shape[2], result_tensor.shape[3])),
                transforms.ToTensor()
            ])
            
            reference_tensor = transform(reference_image.convert('RGB')).unsqueeze(0).to(self.device)
            
            # LPIPS 계산
            with torch.no_grad():
                lpips_distance = self.perceptual_loss(result_tensor, reference_tensor)
            
            # 거리를 유사도로 변환 (낮을수록 좋음)
            lpips_similarity = 1.0 / (1.0 + lpips_distance.item())
            
            return float(lpips_similarity)
            
        except Exception as e:
            self.logger.warning(f"LPIPS 계산 실패: {e}")
            return 0.5

    async def _calculate_fid(self, result_image: Image.Image, reference_image: Image.Image) -> float:
        """FID 계산"""
        try:
            # 이미지를 텐서로 변환
            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor()
            ])
            
            result_tensor = transform(result_image.convert('RGB'))
            reference_tensor = transform(reference_image.convert('RGB'))
            
            # FID 계산 (단일 이미지이므로 간소화된 버전)
            fid_score = self.fid_calculator.calculate_fid([reference_tensor], [result_tensor])
            
            # FID를 유사도로 변환 (낮을수록 좋음)
            fid_similarity = 1.0 / (1.0 + fid_score / 50.0)
            
            return float(fid_similarity)
            
        except Exception as e:
            self.logger.warning(f"FID 계산 실패: {e}")
            return 0.5

    async def _calculate_color_fidelity(self, result_image: Image.Image, original_cloth: Image.Image) -> float:
        """색상 충실도 계산"""
        try:
            result_arr = np.array(result_image.convert('RGB'))
            cloth_arr = np.array(original_cloth.convert('RGB'))
            
            # 크기 맞추기
            if result_arr.shape[:2] != cloth_arr.shape[:2]:
                cloth_arr = cv2.resize(cloth_arr, (result_arr.shape[1], result_arr.shape[0]))
            
            # 히스토그램 비교
            color_similarity = 0
            for c in range(3):  # RGB 채널
                hist1 = cv2.calcHist([result_arr], [c], None, [256], [0, 256])
                hist2 = cv2.calcHist([cloth_arr], [c], None, [256], [0, 256])
                
                # 정규화
                hist1 = hist1 / np.sum(hist1)
                hist2 = hist2 / np.sum(hist2)
                
                # 상관계수
                correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                color_similarity += correlation
            
            return float(color_similarity / 3.0)
            
        except Exception as e:
            self.logger.warning(f"색상 충실도 계산 실패: {e}")
            return 0.7

    async def _calculate_texture_preservation(self, result_image: Image.Image, original_cloth: Image.Image) -> float:
        """텍스처 보존도 계산"""
        try:
            result_arr = np.array(result_image.convert('RGB'))
            cloth_arr = np.array(original_cloth.convert('RGB'))
            
            # 그레이스케일 변환
            result_gray = cv2.cvtColor(result_arr, cv2.COLOR_RGB2GRAY)
            cloth_gray = cv2.cvtColor(cloth_arr, cv2.COLOR_RGB2GRAY)
            
            # 크기 맞추기
            if result_gray.shape != cloth_gray.shape:
                cloth_gray = cv2.resize(cloth_gray, (result_gray.shape[1], result_gray.shape[0]))
            
            # LBP (Local Binary Pattern) 비교
            lbp1 = self._calculate_lbp(result_gray)
            lbp2 = self._calculate_lbp(cloth_gray)
            
            # 히스토그램 비교
            hist1 = cv2.calcHist([lbp1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([lbp2], [0], None, [256], [0, 256])
            
            # 정규화
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)
            
            # 상관계수
            texture_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            return float(texture_similarity)
            
        except Exception as e:
            self.logger.warning(f"텍스처 보존도 계산 실패: {e}")
            return 0.7

    def _calculate_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """Local Binary Pattern 계산"""
        def get_pixel(img, center, x, y):
            new_value = 0
            try:
                if img[x][y] >= center:
                    new_value = 1
            except:
                pass
            return new_value
        
        lbp = np.zeros_like(image)
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                val = 0
                val |= get_pixel(image, center, i-1, j-1) << 7
                val |= get_pixel(image, center, i-1, j) << 6
                val |= get_pixel(image, center, i-1, j+1) << 5
                val |= get_pixel(image, center, i, j+1) << 4
                val |= get_pixel(image, center, i+1, j+1) << 3
                val |= get_pixel(image, center, i+1, j) << 2
                val |= get_pixel(image, center, i+1, j-1) << 1
                val |= get_pixel(image, center, i, j-1) << 0
                lbp[i, j] = val
        
        return lbp.astype(np.uint8)

    async def _calculate_geometric_accuracy(self, input_data: Dict[str, Any]) -> float:
        """기하학적 정확도 계산"""
        try:
            if 'tps_transform' in input_data:
                tps_metrics = input_data['tps_transform'].get('quality_metrics', {})
                
                # TPS 변환 품질에서 기하학적 정확도 추출
                rmse = tps_metrics.get('rmse', 10.0)
                quality_score = tps_metrics.get('quality_score', 0.5)
                
                # RMSE를 정확도로 변환
                accuracy = 1.0 / (1.0 + rmse / 10.0)
                
                # TPS 품질 점수와 결합
                geometric_accuracy = (accuracy + quality_score) / 2.0
                
                return float(geometric_accuracy)
            
            return 0.7  # 기본값
            
        except Exception as e:
            self.logger.warning(f"기하학적 정확도 계산 실패: {e}")
            return 0.6

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """종합 점수 계산"""
        # 가중치 정의
        weights = {
            'ssim': 0.15,
            'lpips': 0.15,
            'fid': 0.1,
            'fit_overall': 0.25,
            'color_fidelity': 0.1,
            'texture_preservation': 0.1,
            'geometric_accuracy': 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                weighted_sum += metrics[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 0.5
        
        return float(overall_score)

    def _classify_quality_grade(self, metrics: Dict[str, float]) -> str:
        """품질 등급 분류"""
        overall_score = metrics.get('overall_score', 0.0)
        
        if overall_score >= self.quality_thresholds['excellent']:
            return 'Excellent'
        elif overall_score >= self.quality_thresholds['good']:
            return 'Good'
        elif overall_score >= self.quality_thresholds['fair']:
            return 'Fair'
        else:
            return 'Poor'

    async def _generate_recommendations(self, metrics: Dict[str, float], input_data: Dict[str, Any]) -> List[str]:
        """개선 추천사항 생성"""
        recommendations = []
        
        # SSIM 기반 추천
        if metrics.get('ssim', 0) < 0.7:
            recommendations.append("이미지 구조적 유사성이 낮습니다. 포즈 매칭을 개선해보세요.")
        
        # LPIPS 기반 추천
        if metrics.get('lpips', 0) < 0.6:
            recommendations.append("지각적 유사성이 낮습니다. 색상과 텍스처 보존을 강화해보세요.")
        
        # 색상 충실도 기반 추천
        if metrics.get('color_fidelity', 0) < 0.7:
            recommendations.append("색상 재현도가 낮습니다. 색상 보정 단계를 강화해보세요.")
        
        # 텍스처 보존 기반 추천
        if metrics.get('texture_preservation', 0) < 0.6:
            recommendations.append("텍스처 보존이 부족합니다. 의류 디테일 보존 기법을 적용해보세요.")
        
        # 기하학적 정확도 기반 추천
        if metrics.get('geometric_accuracy', 0) < 0.6:
            recommendations.append("기하학적 정확도가 낮습니다. TPS 변환 품질을 개선해보세요.")
        
        # 피팅 점수 기반 추천
        fit_overall = metrics.get('fit_overall', 0)
        if fit_overall < 0.7:
            if metrics.get('fit_coverage', 0) < 0.7:
                recommendations.append("의류 커버리지가 부족합니다. 세그멘테이션 품질을 개선해보세요.")
            
            if metrics.get('fit_shape_consistency', 0) < 0.7:
                recommendations.append("형태 일치도가 낮습니다. 워핑 알고리즘을 조정해보세요.")
            
            if metrics.get('fit_boundary_naturalness', 0) < 0.7:
                recommendations.append("경계가 부자연스럽습니다. 블렌딩 기법을 개선해보세요.")
        
        # 종합 점수 기반 추천
        overall_score = metrics.get('overall_score', 0)
        if overall_score < 0.6:
            recommendations.append("전반적인 품질 개선이 필요합니다. 고품질 모드를 사용해보세요.")
        elif overall_score < 0.8:
            recommendations.append("좋은 결과입니다! 미세 조정으로 더 나은 품질을 얻을 수 있습니다.")
        
        # 추천사항이 없는 경우
        if not recommendations:
            recommendations.append("훌륭한 결과입니다! 현재 설정을 유지하세요.")
        
        return recommendations

    def _convert_to_user_scores(self, metrics: Dict[str, float]) -> Dict[str, int]:
        """사용자 친화적 점수로 변환 (0-100점)"""
        user_scores = {}
        
        # 주요 메트릭들을 100점 만점으로 변환
        score_mapping = {
            'overall_quality': 'overall_score',
            'structural_similarity': 'ssim', 
            'perceptual_quality': 'lpips',
            'color_accuracy': 'color_fidelity',
            'texture_quality': 'texture_preservation',
            'fitting_accuracy': 'fit_overall',
            'geometric_precision': 'geometric_accuracy'
        }
        
        for user_name, metric_name in score_mapping.items():
            if metric_name in metrics:
                # 0-1 범위를 0-100으로 변환
                score = int(metrics[metric_name] * 100)
                user_scores[user_name] = max(0, min(100, score))
        
        return user_scores

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """텐서를 PIL Image로 변환"""
        if tensor.dim() == 4:
            tensor = tensor[0]  # 배치 차원 제거
        
        if tensor.dim() == 3 and tensor.shape[0] == 3:
            # [C, H, W] -> [H, W, C]
            tensor = tensor.permute(1, 2, 0)
        
        # [0, 1] -> [0, 255] 변환
        if tensor.max() <= 1.0:
            tensor = tensor * 255
        
        # numpy로 변환 후 PIL Image 생성
        numpy_array = tensor.cpu().numpy().astype(np.uint8)
        
        if len(numpy_array.shape) == 3:
            return Image.fromarray(numpy_array, 'RGB')
        else:
            return Image.fromarray(numpy_array, 'L')

    def generate_quality_report(self, assessment_result: Dict[str, Any], save_path: Optional[str] = None) -> Dict[str, Any]:
        """품질 평가 리포트 생성"""
        
        metrics = assessment_result['quality_metrics']
        user_scores = assessment_result['user_scores']
        recommendations = assessment_result['recommendations']
        grade = assessment_result['quality_grade']
        
        # 리포트 데이터 구성
        report = {
            "summary": {
                "overall_grade": grade,
                "overall_score": f"{user_scores.get('overall_quality', 0)}/100",
                "processing_time": f"{metrics.get('processing_time', 0):.2f}초"
            },
            "detailed_scores": {
                "구조적 유사성": f"{user_scores.get('structural_similarity', 0)}/100",
                "지각적 품질": f"{user_scores.get('perceptual_quality', 0)}/100", 
                "색상 정확도": f"{user_scores.get('color_accuracy', 0)}/100",
                "텍스처 품질": f"{user_scores.get('texture_quality', 0)}/100",
                "피팅 정확도": f"{user_scores.get('fitting_accuracy', 0)}/100",
                "기하학적 정밀도": f"{user_scores.get('geometric_precision', 0)}/100"
            },
            "recommendations": recommendations,
            "technical_metrics": {
                "SSIM": f"{metrics.get('ssim', 0):.3f}",
                "LPIPS": f"{metrics.get('lpips', 0):.3f}",
                "FID": f"{metrics.get('fid', 0):.3f}",
                "Color Fidelity": f"{metrics.get('color_fidelity', 0):.3f}",
                "Texture Preservation": f"{metrics.get('texture_preservation', 0):.3f}"
            }
        }
        
        # 파일로 저장 (옵션)
        if save_path:
            import json
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report

    def visualize_quality_metrics(self, assessment_result: Dict[str, Any], save_path: Optional[str] = None) -> np.ndarray:
        """품질 메트릭 시각화"""
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        user_scores = assessment_result['user_scores']
        
        # 메트릭 이름과 점수
        metrics = list(user_scores.keys())
        scores = list(user_scores.values())
        
        # 한글 폰트 설정 (가능한 경우)
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
        except:
            pass
        
        # 레이더 차트 생성
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        # 각도 계산
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 닫힌 다각형을 위해
        
        scores += scores[:1]  # 닫힌 다각형을 위해
        
        # 플롯
        ax.plot(angles, scores, 'o-', linewidth=2, label='Quality Scores')
        ax.fill(angles, scores, alpha=0.25)
        
        # 축 설정
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'])
        ax.grid(True)
        
        # 제목
        grade = assessment_result['quality_grade']
        overall_score = user_scores.get('overall_quality', 0)
        plt.title(f'Quality Assessment Report\nGrade: {grade} (Score: {overall_score}/100)', 
                 size=16, weight='bold', pad=20)
        
        # 범례
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        plt.tight_layout()
        
        # 저장 (옵션)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # numpy 배열로 변환
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return buf

# 사용 예시
async def example_usage():
    """품질 평가 사용 예시"""
    
    # 설정
    class Config:
        pass
    
    config = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 품질 평가 단계 초기화
    quality_assessment = QualityAssessmentStep(config, device)
    
    # 더미 입력 데이터
    dummy_result = torch.randn(1, 3, 512, 512).to(device)
    dummy_person = Image.new('RGB', (512, 512), color='red')
    dummy_cloth = Image.new('RGB', (512, 512), color='blue')
    
    input_data = {
        "result_image": dummy_result,
        "original_person": dummy_person,
        "original_cloth": dummy_cloth,
        "cloth_mask": np.ones((512, 512)),
        "target_area": np.ones((512, 512)),
        "pose_keypoints": [(100, 100, 0.9), (150, 120, 0.8)],
        "cloth_keypoints": [(105, 105, 0.8), (155, 125, 0.7)]
    }
    
    # 처리
    result = await quality_assessment.process(input_data)
    
    print(f"품질 평가 완료!")
    print(f"품질 등급: {result['quality_grade']}")
    print(f"종합 점수: {result['overall_score']:.3f}")
    
    # 사용자 점수 출력
    print("\n사용자 친화적 점수:")
    for metric, score in result['user_scores'].items():
        print(f"  {metric}: {score}/100")
    
    # 추천사항 출력
    print("\n개선 추천사항:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # 품질 리포트 생성
    report = quality_assessment.generate_quality_report(result)
    print(f"\n품질 리포트 생성 완료")

if __name__ == "__main__":
    asyncio.run(example_usage())