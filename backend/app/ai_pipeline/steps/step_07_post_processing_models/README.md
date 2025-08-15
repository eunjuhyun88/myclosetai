# Post Processing Models

후처리 작업을 위한 AI 모델들을 제공하는 패키지입니다.

## 개요

이 패키지는 이미지 후처리를 위한 4가지 주요 AI 모델을 포함합니다:

- **SwinIR**: 이미지 초해상도 (Super-Resolution)
- **Real-ESRGAN**: 이미지 향상 (Enhancement)
- **GFPGAN**: 얼굴 복원 (Face Restoration)
- **CodeFormer**: 강건한 얼굴 복원 (Robust Face Restoration)

## 구조

```
step_07_post_processing_models/
├── __init__.py                          # 메인 패키지 초기화
├── step_07_post_processing.py          # 메인 스텝 클래스
├── post_processing_model_loader.py     # 모델 로더
├── README.md                           # 이 파일
├── checkpoints/                        # 모델 체크포인트 저장소
├── config/                             # 설정 파일들
│   ├── __init__.py
│   ├── model_config.py                # 모델 설정
│   └── inference_config.py            # 추론 설정
├── core/                              # 핵심 기능
├── ensemble/                          # 앙상블 모델
├── inference/                         # 추론 엔진
│   ├── __init__.py
│   └── inference_engine.py
├── models/                            # 신경망 모델들
│   ├── __init__.py
│   ├── swinir_model.py               # SwinIR 모델
│   ├── realesrgan_model.py           # Real-ESRGAN 모델
│   ├── gfpgan_model.py               # GFPGAN 모델
│   └── codeformer_model.py           # CodeFormer 모델
├── postprocessing/                    # 후처리 기능
├── preprocessing/                     # 전처리 기능
├── processors/                        # 프로세서들
├── services/                          # 서비스 레이어
├── utils/                             # 유틸리티 함수들
└── visualizers/                       # 시각화 도구들
```

## 주요 클래스

### PostProcessingStep
후처리 파이프라인의 메인 진입점입니다.

### PostProcessingModelLoader
모델들을 로드하고 관리하는 클래스입니다.

### PostProcessingInferenceEngine
로드된 모델들을 사용하여 추론을 수행하는 엔진입니다.

## 사용법

### 기본 사용법

```python
from backend.app.ai_pipeline.steps.step_07_post_processing_models import PostProcessingStep

# 후처리 스텝 초기화
post_processing = PostProcessingStep()

# 이미지 처리
result = post_processing.process_image(
    image=image,
    model_type='swinir',
    upscale=4
)
```

### 모델 로더 직접 사용

```python
from backend.app.ai_pipeline.steps.step_07_post_processing_models import PostProcessingModelLoader

# 모델 로더 초기화
loader = PostProcessingModelLoader()

# 모델 로드
model = loader.load_model('swinir', upscale=4)

# 모델 정보 확인
info = loader.get_model_info('swinir')
print(f"Model parameters: {info['total_parameters']:,}")
```

### 추론 엔진 직접 사용

```python
from backend.app.ai_pipeline.steps.step_07_post_processing_models import PostProcessingInferenceEngine

# 추론 엔진 초기화
engine = PostProcessingInferenceEngine(model_loader=loader)

# 이미지 처리
result = engine.process_image(
    image=image,
    model_type='realesrgan',
    tile_size=400
)
```

## 설정

### 모델 설정

```python
from backend.app.ai_pipeline.steps.step_07_post_processing_models.config import PostProcessingModelConfig

# 기본 설정 로드
config = PostProcessingModelConfig()

# 설정 커스터마이징
config.swinir.upscale = 2
config.realesrgan.num_block = 16

# 설정 저장
config.save_to_file('custom_config.json')
```

### 추론 설정

```python
from backend.app.ai_pipeline.steps.step_07_post_processing_models.config import PostProcessingInferenceConfig

# 추론 설정 로드
inference_config = PostProcessingInferenceConfig()

# 타일링 설정
inference_config.enable_tiling = True
inference_config.tile_size_threshold = 1024

# 설정 저장
inference_config.save_to_file('inference_config.json')
```

## 지원 모델

### SwinIR
- **용도**: 이미지 초해상도
- **특징**: Swin Transformer 기반, 고품질 결과
- **적용 분야**: 저해상도 이미지 개선, 디테일 복원

### Real-ESRGAN
- **용도**: 이미지 향상 및 초해상도
- **특징**: GAN 기반, 실세계 이미지에 최적화
- **적용 분야**: 노이즈 제거, 선명도 향상

### GFPGAN
- **용도**: 얼굴 복원
- **특징**: StyleGAN2 기반, 자연스러운 얼굴 생성
- **적용 분야**: 얼굴 품질 개선, 노화 복원

### CodeFormer
- **용도**: 강건한 얼굴 복원
- **특징**: Transformer + Codebook 기반, 다양한 얼굴 상태 지원
- **적용 분야**: 심각한 손상된 얼굴 복원

## 메모리 관리

이 패키지는 메모리 효율적인 처리를 위해 다음 기능을 제공합니다:

- **타일링 처리**: 큰 이미지를 작은 타일로 분할하여 처리
- **모델 캐싱**: 자주 사용되는 모델을 메모리에 유지
- **동적 메모리 할당**: 사용 가능한 메모리에 따른 설정 자동 조정

## 성능 최적화

### GPU 메모리 최적화

```python
# 메모리 효율적인 설정
config = PostProcessingInferenceConfig()
config.max_memory_usage = 0.7  # GPU 메모리 70% 사용 제한
config.enable_tiling = True     # 타일링 활성화
```

### 배치 처리

```python
# 여러 이미지 배치 처리
results = engine.batch_process(
    images=image_list,
    model_type='swinir',
    batch_size=4
)
```

## 에러 처리

모든 주요 함수는 적절한 에러 처리를 포함합니다:

```python
try:
    result = post_processing.process_image(image, 'swinir')
except ValueError as e:
    print(f"Invalid model type: {e}")
except RuntimeError as e:
    print(f"Processing failed: {e}")
```

## 로깅

프로젝트의 통합 로깅 시스템을 사용합니다:

```python
from backend.app.ai_pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
logger.error("Processing failed")
```

## 의존성

- PyTorch >= 1.8.0
- torchvision
- numpy
- PIL (Pillow)
- opencv-python

## 라이선스

이 구현은 각 모델의 원본 논문과 구현체를 기반으로 합니다.
