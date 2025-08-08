# MyCloset-AI AI 모델 인벤토리

## 전체 통계
- **총 체크포인트 파일 수**: 185개
- **총 크기**: 약 50GB+
- **스텝별 분포**: 8개 스텝

## Step 01: Human Parsing (인간 파싱)
### 주요 모델들
| 모델명 | 크기 | 용도 | 상태 |
|--------|------|------|------|
| `graphonomy.pth` | 1.1GB | Graphonomy 메인 모델 | ✅ |
| `graphonomy_root.pth` | 1.1GB | Graphonomy 루트 모델 | ✅ |
| `exp-schp-201908261155-lip.pth` | 255MB | SCHP LIP 데이터셋 | ✅ |
| `exp-schp-201908301523-atr.pth` | 255MB | SCHP ATR 데이터셋 | ✅ |
| `graphonomy_fixed.pth` | 255MB | 수정된 Graphonomy | ✅ |
| `deeplabv3plus.pth` | 233MB | DeepLabV3+ 모델 | ✅ |
| `pytorch_model.bin` | 105MB | PyTorch 변환 모델 | ✅ |

### 총 크기: 약 3.5GB

## Step 02: Pose Estimation (포즈 추정)
### 주요 모델들
| 모델명 | 크기 | 용도 | 상태 |
|--------|------|------|------|
| `diffusion_pytorch_model.bin` | 1.3GB | Diffusion 모델 | ✅ |
| `diffusion_pytorch_model.fp16.bin` | 689MB | FP16 최적화 모델 | ✅ |
| `hrnet_w48_coco_384x288.pth` | 243MB | HRNet 48-384x288 | ✅ |
| `hrnet_w48_coco_256x192.pth` | 243MB | HRNet 48-256x192 | ✅ |
| `hrnet_w32_coco_256x192.pth` | 109MB | HRNet 32-256x192 | ✅ |
| `body_pose_model.pth` | 98MB | OpenPose 바디 모델 | ✅ |
| `yolov8m-pose.pt` | 51MB | YOLOv8 포즈 모델 | ✅ |

### 총 크기: 약 2.6GB

## Step 03: Cloth Segmentation (의류 분할)
### 주요 모델들
| 모델명 | 크기 | 용도 | 상태 |
|--------|------|------|------|
| `pytorch_model.bin` | 2.4GB | PyTorch 변환 모델 | ✅ |
| `sam_vit_h_4b8939.pth` | 2.4GB | SAM ViT-H 모델 | ✅ |
| `sam_vit_l_0b3195.pth` | 1.2GB | SAM ViT-L 모델 | ✅ |
| `u2net.pth` | 528MB | U²-Net 모델 | ✅ |
| `u2net_fallback.pth` | 528MB | U²-Net 백업 | ✅ |
| `mobile_sam.pt` | 358MB | Mobile SAM | ✅ |
| `mobile_sam_alternative.pt` | 358MB | Mobile SAM 대안 | ✅ |
| `deeplabv3_resnet101_coco.pth` | 233MB | DeepLabV3 ResNet101 | ✅ |
| `deeplabv3_resnet101_ultra.pth` | 233MB | DeepLabV3 Ultra | ✅ |

### 총 크기: 약 8.2GB

## Step 04: Geometric Matching (기하학적 매칭)
### 주요 모델들
| 모델명 | 크기 | 용도 | 상태 |
|--------|------|------|------|
| `sam_vit_h_4b8939.pth` | 2.4GB | SAM ViT-H 모델 | ✅ |
| `diffusion_pytorch_model.bin` | 1.3GB | Diffusion 모델 | ✅ |
| `gmm_final.pth` | 1.3GB | GMM 최종 모델 | ✅ |
| `pytorch_model.bin` | 1.1GB | PyTorch 변환 모델 | ✅ |
| `ViT-L-14.pt` | 577MB | ViT-L-14 모델 | ✅ |
| `tps_network.pth` | 548MB | TPS 네트워크 | ✅ |
| `efficientnet_b0_ultra.pth` | 548MB | EfficientNet B0 Ultra | ✅ |
| `raft-things.pth` | 548MB | RAFT Things 데이터셋 | ✅ |
| `resnet101_geometric.pth` | 528MB | ResNet101 기하학적 | ✅ |
| `resnet50_geometric_ultra.pth` | 98MB | ResNet50 Ultra | ✅ |
| `RealESRGAN_x4plus.pth` | 64MB | RealESRGAN x4+ | ✅ |

### 총 크기: 약 8.5GB

## Step 05: Cloth Warping (의류 변형)
### 주요 모델들
| 모델명 | 크기 | 용도 | 상태 |
|--------|------|------|------|
| `RealVisXL_V4.0.safetensors` | 6.5GB | RealVisXL V4.0 | ✅ |
| `diffusion_pytorch_model.bin` | 3.2GB | Diffusion 모델 | ✅ |
| `viton_hd_warping.pth` | 1.3GB | VITON-HD 변형 | ✅ |
| `vgg19_warping.pth` | 548MB | VGG19 변형 | ✅ |
| `vgg16_warping_ultra.pth` | 528MB | VGG16 Ultra 변형 | ✅ |
| `tps_transformation.pth` | 470MB | TPS 변형 | ✅ |
| `dpt_hybrid_midas.pth` | 470MB | DPT Hybrid MiDaS | ✅ |
| `tom_final.pth` | 83MB | TOM 최종 모델 | ✅ |
| `u2net_warping.pth` | 39MB | U²-Net 변형 | ✅ |
| `densenet121_ultra.pth` | 31MB | DenseNet121 Ultra | ✅ |
| `RealESRGAN_x4plus.pth` | 64MB | RealESRGAN x4+ | ✅ |

### 총 크기: 약 13.5GB

## Step 06: Virtual Fitting (가상 피팅)
### 주요 모델들
| 모델명 | 크기 | 용도 | 상태 |
|--------|------|------|------|
| `diffusion_pytorch_model.safetensors` | 3.2GB | Diffusion SafeTensors | ✅ |
| `pytorch_model.bin` | 3.2GB | PyTorch 모델 | ✅ |
| `ootd_checkpoint.pth` | 3.2GB | OOTD 체크포인트 | ✅ |
| `stable_diffusion_4.8gb.pth` | 3.2GB | Stable Diffusion | ✅ |
| `ootd_3.2gb.pth` | 3.2GB | OOTD 3.2GB | ✅ |
| `hrviton_final.pth` | 230MB | HR-VITON 최종 | ✅ |
| `viton_hd_2.1gb.pth` | 230MB | VITON-HD 2.1GB | ✅ |
| `text_encoder_pytorch_model.bin` | 492MB | 텍스트 인코더 | ✅ |
| `vae_diffusion_pytorch_model.bin` | 335MB | VAE 모델 | ✅ |

### 총 크기: 약 17GB

## Step 07: Post Processing (후처리)
### 주요 모델들
| 모델명 | 크기 | 용도 | 상태 |
|--------|------|------|------|
| `densenet161_enhance.pth` | 233MB | DenseNet161 향상 | ✅ |
| `mobilenet_v3_ultra.pth` | 233MB | MobileNet V3 Ultra | ✅ |
| `GFPGAN.pth` | 233MB | GFPGAN | ✅ |
| `resnet101_enhance_ultra.pth` | 233MB | ResNet101 Ultra 향상 | ✅ |
| `RealESRGAN_x2plus.pth` | 233MB | RealESRGAN x2+ | ✅ |
| `ESRGAN_x8.pth` | 233MB | ESRGAN x8 | ✅ |
| `swinir_real_sr_x4_large.pth` | 233MB | SwinIR Real SR x4 | ✅ |
| `RealESRGAN_x4plus.pth` | 233MB | RealESRGAN x4+ | ✅ |
| `pytorch_model.bin` | 233MB | PyTorch 모델 | ✅ |

### 총 크기: 약 2.1GB

## Step 08: Quality Assessment (품질 평가)
### 주요 모델들
| 모델명 | 크기 | 용도 | 상태 |
|--------|------|------|------|
| `open_clip_pytorch_model.bin` | 1.6GB | OpenCLIP PyTorch | ✅ |
| `pytorch_model.bin` | 1.6GB | PyTorch 모델 | ✅ |
| `clip_vit_b32.pth` | 233MB | CLIP ViT-B/32 | ✅ |
| `alex.pth` | 233MB | AlexNet | ✅ |
| `lpips_alex.pth` | 233MB | LPIPS AlexNet | ✅ |
| `ViT-B-32.pt` | 233MB | ViT-B/32 | ✅ |
| `ViT-L-14.pt` | 233MB | ViT-L/14 | ✅ |

### 총 크기: 약 4.2GB

## 전체 요약
- **총 모델 수**: 185개
- **총 크기**: 약 60GB
- **주요 모델 타입**:
  - Diffusion 모델들 (Stable Diffusion, OOTD)
  - Vision Transformer 모델들 (ViT, SAM)
  - CNN 모델들 (ResNet, VGG, DenseNet)
  - 특화 모델들 (Graphonomy, HRNet, U²-Net)

## 모델 로더 개선 방안
1. **지연 로딩 (Lazy Loading)**: 필요한 시점에만 모델 로드
2. **모델 캐싱**: 자주 사용되는 모델 메모리 캐싱
3. **모델 압축**: FP16, INT8 양자화 적용
4. **모델 공유**: 공통 레이어 공유로 메모리 절약
5. **체크포인트 검증**: 로드 전 체크포인트 무결성 검증
