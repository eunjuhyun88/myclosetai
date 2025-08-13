#!/usr/bin/env python3
"""
🔥 MyCloset AI - Step 03: 의류 세그멘테이션 - Hybrid Ensemble
=====================================================================

앙상블 관련 함수들을 분리한 모듈

Author: MyCloset AI Team  
Date: 2025-08-01
Version: 1.0
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# 공통 imports 시스템 사용
try:
    from app.ai_pipeline.utils.common_imports import (
        np, cv2, PIL_AVAILABLE, CV2_AVAILABLE, NUMPY_AVAILABLE
    )
except ImportError:
    import numpy as np
    import cv2

logger = logging.getLogger(__name__)

def _run_hybrid_ensemble_sync(
    self, 
    image: np.ndarray, 
    person_parsing: Dict[str, Any],
    pose_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    하이브리드 앙상블 세그멘테이션 실행 (동기)
    
    Args:
        image: 입력 이미지
        person_parsing: 인체 파싱 결과
        pose_info: 포즈 정보
        
    Returns:
        앙상블 결과
    """
    try:
        logger.info("🔥 하이브리드 앙상블 세그멘테이션 시작")
        
        # 사용 가능한 모델들 확인
        available_models = self._detect_available_methods()
        if not available_models:
            logger.warning("⚠️ 사용 가능한 모델이 없음")
            return self._create_fallback_segmentation_result(image.shape)
        
        # 앙상블 실행
        ensemble_results = []
        methods_used = []
        execution_times = []
        
        for method in available_models[:3]:  # 최대 3개 모델만 사용
            try:
                start_time = time.time()
                
                if method == SegmentationMethod.U2NET_CLOTH:
                    result = self._run_u2net_segmentation(image, person_parsing, pose_info)
                elif method == SegmentationMethod.SAM_HUGE:
                    result = self._run_sam_segmentation(image, person_parsing, pose_info)
                elif method == SegmentationMethod.DEEPLABV3_PLUS:
                    result = self._run_deeplabv3plus_segmentation(image, person_parsing, pose_info)
                else:
                    continue
                
                execution_time = time.time() - start_time
                
                if result and result.get('success', False):
                    ensemble_results.append(result)
                    methods_used.append(method.value)
                    execution_times.append(execution_time)
                    logger.info(f"✅ {method.value} 완료 ({execution_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"❌ {method.value} 실행 실패: {e}")
                continue
        
        if not ensemble_results:
            logger.warning("⚠️ 모든 모델 실행 실패")
            return self._create_fallback_segmentation_result(image.shape)
        
        # 앙상블 결과 결합
        final_result = self._combine_ensemble_results(
            ensemble_results, methods_used, execution_times, image, person_parsing
        )
        
        logger.info(f"🔥 하이브리드 앙상블 완료 (사용된 모델: {methods_used})")
        return final_result
        
    except Exception as e:
        logger.error(f"❌ 하이브리드 앙상블 실패: {e}")
        return self._create_fallback_segmentation_result(image.shape)

def _combine_ensemble_results(
    self,
    results: List[Dict[str, Any]],
    methods_used: List[str],
    execution_times: List[float],
    image: np.ndarray,
    person_parsing: Dict[str, Any]
) -> Dict[str, Any]:
    """
    앙상블 결과 결합 (개선된 버전)
    
    Args:
        results: 개별 모델 결과들
        methods_used: 사용된 모델들
        execution_times: 실행 시간들
        image: 원본 이미지
        person_parsing: 인체 파싱 결과
        
    Returns:
        결합된 결과
    """
    try:
        logger.info("🔥 앙상블 결과 결합 시작")
        
        if len(results) == 1:
            # 단일 모델 결과 반환
            result = results[0]
            result['ensemble_methods'] = methods_used
            result['ensemble_times'] = execution_times
            return result
        
        # 다중 모델 결과 결합 (개선된 가중치 계산)
        combined_masks = {}
        combined_confidence = 0.0
        total_weight = 0.0
        
        # 모델별 가중치 계산 (개선된 버전)
        model_weights = []
        for i, (result, method, exec_time) in enumerate(zip(results, methods_used, execution_times)):
            # 1. 신뢰도 기반 가중치 (0.3)
            confidence = result.get('confidence', 0.5)
            confidence_weight = confidence * 0.3
            
            # 2. 실행 시간 기반 가중치 (0.2) - 빠른 모델에 보너스
            time_weight = 1.0 / (exec_time + 1e-6) * 0.2
            
            # 3. 모델 타입별 가중치 (0.3)
            type_weight = 0.0
            if 'deeplabv3' in method.lower():
                type_weight = 1.0 * 0.3  # 최고 가중치
            elif 'sam' in method.lower():
                type_weight = 0.9 * 0.3  # 높은 가중치
            elif 'u2net' in method.lower():
                type_weight = 0.8 * 0.3  # 중간 가중치
            else:
                type_weight = 0.5 * 0.3  # 기본 가중치
            
            # 4. 결과 품질 기반 가중치 (0.2)
            quality_weight = 0.0
            if 'masks' in result and result['masks']:
                # 마스크 품질 평가
                mask_quality = self._evaluate_mask_quality(result['masks'], image)
                quality_weight = mask_quality * 0.2
            
            # 총 가중치 계산
            total_model_weight = confidence_weight + time_weight + type_weight + quality_weight
            model_weights.append(total_model_weight)
            total_weight += total_model_weight
        
        # 가중치 정규화
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in model_weights]
        else:
            normalized_weights = [1.0 / len(results)] * len(results)
        
        # 가중 평균으로 결과 결합
        for i, result in enumerate(results):
            weight = normalized_weights[i]
            
            if 'masks' in result:
                for mask_type, mask in result['masks'].items():
                    if mask_type not in combined_masks:
                        combined_masks[mask_type] = np.zeros_like(mask, dtype=np.float32)
                    combined_masks[mask_type] += mask.astype(np.float32) * weight
            
            if 'confidence' in result:
                combined_confidence += result['confidence'] * weight
        
        # 마스크 정규화 및 임계값 적용
        for mask_type in combined_masks:
            combined_masks[mask_type] = (combined_masks[mask_type] > 0.5).astype(np.uint8)
        
        # 후처리 적용
        refined_masks = self._apply_ensemble_postprocessing(combined_masks, image)
        
        # 최종 결과 생성
        final_result = {
            'success': True,
            'masks': refined_masks,
            'confidence': combined_confidence,
            'ensemble_methods': methods_used,
            'ensemble_times': execution_times,
            'ensemble_weights': normalized_weights,
            'ensemble_count': len(results),
            'method': 'hybrid_ensemble'
        }
        
        logger.info(f"🔥 앙상블 결과 결합 완료 (가중치: {normalized_weights})")
        return final_result
        
    except Exception as e:
        logger.error(f"❌ 앙상블 결과 결합 실패: {e}")
        return self._create_fallback_segmentation_result(image.shape)

def _calculate_adaptive_threshold(
    self, 
    ensemble_mask: np.ndarray, 
    image: np.ndarray
) -> float:
    """
    적응형 임계값 계산
    
    Args:
        ensemble_mask: 앙상블 마스크
        image: 원본 이미지
        
    Returns:
        적응형 임계값
    """
    try:
        # 이미지 통계 계산
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # 마스크 통계 계산
        mask_mean = np.mean(ensemble_mask) if ensemble_mask.size > 0 else 0.5
        mask_std = np.std(ensemble_mask) if ensemble_mask.size > 0 else 0.1
        
        # 적응형 임계값 계산
        base_threshold = 0.5
        intensity_factor = (mean_intensity - 128) / 128  # -1 to 1
        contrast_factor = (std_intensity - 50) / 50      # -1 to 1
        
        adaptive_threshold = base_threshold + intensity_factor * 0.1 + contrast_factor * 0.05
        
        # 범위 제한
        adaptive_threshold = np.clip(adaptive_threshold, 0.3, 0.7)
        
        logger.debug(f"적응형 임계값: {adaptive_threshold:.3f}")
        return adaptive_threshold
        
    except Exception as e:
        logger.warning(f"적응형 임계값 계산 실패: {e}")
        return 0.5

def _apply_ensemble_postprocessing(
    self, 
    mask: np.ndarray, 
    image: np.ndarray
) -> np.ndarray:
    """
    앙상블 후처리 적용
    
    Args:
        mask: 원본 마스크
        image: 원본 이미지
        
    Returns:
        후처리된 마스크
    """
    try:
        logger.debug("앙상블 후처리 시작")
        
        # 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 경계 정제
        mask = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)
        mask = (mask > 0.5).astype(np.uint8)
        
        # 홀 채우기
        mask = cv2.fillPoly(mask, [cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]], 1)
        
        logger.debug("앙상블 후처리 완료")
        return mask
        
    except Exception as e:
        logger.warning(f"앙상블 후처리 실패: {e}")
        return mask

def _run_u2net_segmentation(
    self,
    image: np.ndarray,
    person_parsing: Dict[str, Any],
    pose_info: Dict[str, Any]
) -> Dict[str, Any]:
    """U2Net 세그멘테이션 실행"""
    try:
        if 'u2net_cloth' not in self.segmentation_models:
            return None
        
        model = self.segmentation_models['u2net_cloth']
        result = model.predict(image)
        
        if result and result.get('success', False):
            result['method'] = 'u2net_cloth'
            return result
        
        return None
        
    except Exception as e:
        logger.error(f"U2Net 세그멘테이션 실패: {e}")
        return None

def _run_sam_segmentation(
    self,
    image: np.ndarray,
    person_parsing: Dict[str, Any],
    pose_info: Dict[str, Any]
) -> Dict[str, Any]:
    """SAM 세그멘테이션 실행"""
    try:
        if 'sam_huge' not in self.segmentation_models:
            return None
        
        model = self.segmentation_models['sam_huge']
        
        # SAM 프롬프트 생성
        prompts = self._generate_sam_prompts(image, person_parsing, pose_info)
        
        result = model.predict(image, prompts)
        
        if result and result.get('success', False):
            result['method'] = 'sam_huge'
            return result
        
        return None
        
    except Exception as e:
        logger.error(f"SAM 세그멘테이션 실패: {e}")
        return None

def _run_deeplabv3plus_segmentation(
    self,
    image: np.ndarray,
    person_parsing: Dict[str, Any],
    pose_info: Dict[str, Any]
) -> Dict[str, Any]:
    """DeepLabV3+ 세그멘테이션 실행"""
    try:
        if 'deeplabv3plus' not in self.segmentation_models:
            return None
        
        model = self.segmentation_models['deeplabv3plus']
        result = model.predict(image)
        
        if result and result.get('success', False):
            result['method'] = 'deeplabv3plus'
            return result
        
        return None
        
    except Exception as e:
        logger.error(f"DeepLabV3+ 세그멘테이션 실패: {e}")
        return None

def _generate_sam_prompts(
    self,
    image: np.ndarray,
    person_parsing: Dict[str, Any],
    pose_info: Dict[str, Any]
) -> Dict[str, Any]:
    """SAM 프롬프트 생성"""
    try:
        prompts = {
            'points': [],
            'boxes': [],
            'masks': []
        }
        
        # 인체 파싱에서 의류 영역 추출
        if 'parsing_map' in person_parsing:
            parsing_map = person_parsing['parsing_map']
            
            # 의류 카테고리들 (상의, 하의, 전신)
            clothing_categories = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11]
            
            for category in clothing_categories:
                if category in parsing_map:
                    mask = (parsing_map == category).astype(np.uint8)
                    
                    # 컨투어 찾기
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        if cv2.contourArea(contour) > 100:  # 최소 면적
                            # 바운딩 박스 추가
                            x, y, w, h = cv2.boundingRect(contour)
                            prompts['boxes'].append([x, y, x + w, y + h])
                            
                            # 중심점 추가
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                prompts['points'].append([cx, cy])
        
        return prompts
        
    except Exception as e:
        logger.warning(f"SAM 프롬프트 생성 실패: {e}")
        return {'points': [], 'boxes': [], 'masks': []}

def _evaluate_mask_quality(self, masks: Dict[str, np.ndarray], image: np.ndarray) -> float:
    """마스크 품질 평가"""
    try:
        if not masks:
            return 0.5
        
        total_quality = 0.0
        mask_count = 0
        
        for mask_type, mask in masks.items():
            if mask is None or mask.size == 0:
                continue
            
            # 1. 면적 비율 평가
            area_ratio = np.sum(mask) / mask.size
            area_score = min(area_ratio * 10, 1.0)  # 적절한 면적 비율에 높은 점수
            
            # 2. 경계 품질 평가
            edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150)
            edge_density = np.sum(edges) / (edges.size * 255)
            edge_score = 1.0 - min(edge_density * 5, 1.0)  # 낮은 edge density에 높은 점수
            
            # 3. 연결성 평가
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            connectivity_score = 1.0 / (len(contours) + 1)  # 컨투어가 적을수록 좋음
            
            # 4. 원형도 평가
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                contour_perimeter = cv2.arcLength(largest_contour, True)
                
                if contour_perimeter > 0:
                    circularity = 4 * np.pi * contour_area / (contour_perimeter ** 2)
                else:
                    circularity = 0.0
            else:
                circularity = 0.0
            
            # 종합 품질 점수
            quality_score = (area_score * 0.3 + edge_score * 0.3 + 
                           connectivity_score * 0.2 + circularity * 0.2)
            
            total_quality += quality_score
            mask_count += 1
        
        return total_quality / mask_count if mask_count > 0 else 0.5
        
    except Exception as e:
        logger.warning(f"마스크 품질 평가 실패: {e}")
        return 0.5
