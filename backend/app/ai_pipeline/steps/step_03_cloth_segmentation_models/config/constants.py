"""
Cloth Segmentation Constants
의류 세그멘테이션 상수들
"""

# SAM 모델 상수
SAM_CONSTANTS = {
    "MODEL_TYPE": "vit_h",
    "CHECKPOINT_PATH": "sam_vit_h_4b8939.pth",
    "DEVICE": "mps",
    "POINTS_PER_SIDE": 32,
    "PRED_IOU_THRESH": 0.88,
    "STABILITY_SCORE_THRESH": 0.95,
    "BOX_NMS_THRESH": 0.7,
    "CROP_N_LAYERS": 0,
    "CROP_NMS_THRESH": 0.7,
    "CROP_OVERLAP_RATIO": 512 / 1500,
    "CROP_N_POINTS_DOWNSCALE_FACTOR": 1,
    "POINT_GRIDS": None,
    "MIN_MASK_REGION_AREA": 0,
    "OUTPUT_MODE": "binary_mask"
}

# 세그멘테이션 품질 임계값
SEGMENTATION_QUALITY_THRESHOLDS = {
    "MIN_CONFIDENCE": 0.8,
    "MIN_MASK_AREA": 1000,
    "MAX_MASK_AREA": 1000000,
    "MIN_ASPECT_RATIO": 0.1,
    "MAX_ASPECT_RATIO": 10.0
}

# 의류 클래스 정의
CLOTHING_CLASSES = {
    "TOP": 1,
    "BOTTOM": 2,
    "DRESS": 3,
    "OUTERWEAR": 4,
    "UNDERWEAR": 5,
    "ACCESSORIES": 6
}

# 색상 매핑
CLOTHING_COLORS = {
    "TOP": [255, 0, 0],      # 빨강
    "BOTTOM": [0, 255, 0],   # 초록
    "DRESS": [0, 0, 255],    # 파랑
    "OUTERWEAR": [255, 255, 0], # 노랑
    "UNDERWEAR": [255, 0, 255], # 마젠타
    "ACCESSORIES": [0, 255, 255] # 시안
}

# 후처리 설정
POSTPROCESSING_CONFIG = {
    "MORPHOLOGY_KERNEL_SIZE": 5,
    "GAUSSIAN_BLUR_KERNEL_SIZE": 3,
    "CONTOUR_APPROXIMATION_EPSILON": 0.02,
    "MIN_CONTOUR_AREA": 500
}

# 입력 이미지 설정
INPUT_IMAGE_CONFIG = {
    "MAX_SIZE": 1024,
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
    "NORMALIZE": True
}

# 출력 설정
OUTPUT_CONFIG = {
    "FORMAT": "binary_mask",
    "SAVE_ANNOTATIONS": True,
    "SAVE_VISUALIZATION": True,
    "OVERLAY_ALPHA": 0.6
}

# 성능 최적화 설정
PERFORMANCE_CONFIG = {
    "BATCH_SIZE": 1,
    "USE_AMP": True,
    "USE_CACHE": True,
    "MAX_WORKERS": 4
}

# 기본 설정
DEFAULT_CONFIG = {
    "model_type": "vit_h",
    "device": "mps",
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95,
    "box_nms_thresh": 0.7,
    "crop_n_layers": 0,
    "crop_nms_thresh": 0.7,
    "crop_overlap_ratio": 512 / 1500,
    "crop_n_points_downscale_factor": 1,
    "point_grids": None,
    "min_mask_region_area": 0,
    "output_mode": "binary_mask"
}
