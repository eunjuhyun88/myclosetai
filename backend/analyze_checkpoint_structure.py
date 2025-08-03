#!/usr/bin/env python3
"""
체크포인트 파일의 전체 구조 분석 스크립트
"""

import torch
import os
import sys
from collections import defaultdict

def analyze_checkpoint_structure(checkpoint_path):
    """체크포인트 파일의 구조를 분석합니다."""

    print("=" * 80)
    print("🔍 체크포인트 구조 분석 시작")
    print("=" * 80)

    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트 파일이 존재하지 않습니다: {checkpoint_path}")
        return

    try:
        # 체크포인트 로딩
        print(f"📁 체크포인트 로딩: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        print(f"✅ 체크포인트 로딩 성공: {type(checkpoint)}")

        # 체크포인트 구조 분석
        if isinstance(checkpoint, dict):
            print(f"📊 체크포인트 키: {list(checkpoint.keys())}")

            # state_dict가 있는지 확인
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
                print("✅ state_dict 키에서 체크포인트 추출")

            print(f"✅ state_dict 추출 성공: {len(checkpoint.keys())}개 키")

            # 전체 키를 카테고리별로 분류
            categories = {
                'backbone': [],
                'edge': [],
                'context_encoding': [],
                'decoder': [],
                'other': []
            }

            for key in sorted(checkpoint.keys()):
                tensor_shape = checkpoint[key].shape
                print(f"  - {key}: {tensor_shape}")
                
                if 'backbone' in key:
                    categories['backbone'].append((key, tensor_shape))
                elif 'edge' in key:
                    categories['edge'].append((key, tensor_shape))
                elif 'context_encoding' in key:
                    categories['context_encoding'].append((key, tensor_shape))
                elif 'decoder' in key:
                    categories['decoder'].append((key, tensor_shape))
                else:
                    categories['other'].append((key, tensor_shape))

            print("\n" + "=" * 60)
            print("📊 카테고리별 분석")
            print("=" * 60)

            for category, items in categories.items():
                if items:
                    print(f"\n🔍 {category.upper()} 모듈 ({len(items)}개):")
                    for key, shape in items:
                        print(f"  - {key}: {shape}")

            # Edge 모듈 상세 분석
            if categories['edge']:
                print(f"\n🔍 EDGE 모듈 상세 분석:")
                edge_structure = defaultdict(list)
                for key, shape in categories['edge']:
                    if 'conv' in key and '.weight' in key:
                        # Conv2d weight shape: [out_channels, in_channels, kH, kW]
                        out_channels = shape[0]
                        in_channels = shape[1]
                        kernel_size = f"{shape[2]}x{shape[3]}"
                        conv_name = key.split('.')[1]
                        edge_structure[conv_name].append(f"Conv2d({in_channels}, {out_channels}, {kernel_size})")

                print("  📝 Edge 모듈 요약:")
                for conv_name, details in edge_structure.items():
                    print(f"    {conv_name}: {', '.join(details)}")

            # Backbone 모듈 상세 분석
            if categories['backbone']:
                print(f"\n🔍 BACKBONE 모듈 상세 분석:")
                backbone_structure = defaultdict(list)
                for key, shape in categories['backbone']:
                    if 'conv' in key and '.weight' in key:
                        out_channels = shape[0]
                        in_channels = shape[1]
                        kernel_size = f"{shape[2]}x{shape[3]}"
                        conv_name = key.split('.')[1]
                        backbone_structure[conv_name].append(f"Conv2d({in_channels}, {out_channels}, {kernel_size})")

                print("  📝 Backbone 모듈 요약:")
                for conv_name, details in backbone_structure.items():
                    print(f"    {conv_name}: {', '.join(details)}")

        else:
            print("⚠️ 체크포인트가 딕셔너리 형태가 아닙니다. 직접 구조를 확인해야 합니다.")

    except Exception as e:
        print(f"❌ 체크포인트 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 80)
    print("✅ 체크포인트 구조 분석 완료")
    print("=" * 80)

if __name__ == "__main__":
    # 실제 체크포인트 파일 경로
    checkpoint_file_path = "ai_models/Self-Correction-Human-Parsing/exp-schp-201908261155-atr.pth"
    analyze_checkpoint_structure(checkpoint_file_path) 