"""
Model Utilities
모델과 관련된 공통 유틸리티 함수들을 제공하는 클래스
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import logging
import time
import os
from pathlib import Path

# 프로젝트 로깅 설정 import
from backend.app.ai_pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)

class ModelUtils:
    """
    모델과 관련된 공통 유틸리티 함수들을 제공하는 클래스
    """
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """
        모델의 파라미터 개수 계산
        
        Args:
            model: PyTorch 모델
            
        Returns:
            파라미터 개수 정보 딕셔너리
        """
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            non_trainable_params = total_params - trainable_params
            
            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'non_trainable_parameters': non_trainable_params
            }
            
        except Exception as e:
            logger.error(f"파라미터 개수 계산 중 오류 발생: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def get_model_size(model: nn.Module, precision: str = 'fp32') -> Dict[str, float]:
        """
        모델 크기 계산
        
        Args:
            model: PyTorch 모델
            precision: 정밀도 ('fp32', 'fp16', 'int8')
            
        Returns:
            모델 크기 정보 딕셔너리
        """
        try:
            # 파라미터 개수 계산
            param_info = ModelUtils.count_parameters(model)
            total_params = param_info.get('total_parameters', 0)
            
            # 정밀도별 바이트 수
            precision_bytes = {
                'fp32': 4,  # 32비트 = 4바이트
                'fp16': 2,  # 16비트 = 2바이트
                'int8': 1   # 8비트 = 1바이트
            }
            
            bytes_per_param = precision_bytes.get(precision, 4)
            
            # 모델 크기 계산
            model_size_bytes = total_params * bytes_per_param
            model_size_mb = model_size_bytes / (1024 * 1024)
            model_size_gb = model_size_mb / 1024
            
            return {
                'size_bytes': model_size_bytes,
                'size_mb': model_size_mb,
                'size_gb': model_size_gb,
                'precision': precision,
                'total_parameters': total_params
            }
            
        except Exception as e:
            logger.error(f"모델 크기 계산 중 오류 발생: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def save_model_checkpoint(model: nn.Module, 
                            filepath: str, 
                            optimizer: Optional[torch.optim.Optimizer] = None,
                            epoch: int = 0,
                            loss: float = 0.0,
                            additional_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        모델 체크포인트 저장
        
        Args:
            model: 저장할 모델
            filepath: 저장 경로
            optimizer: 옵티마이저 (선택사항)
            epoch: 현재 에포크
            loss: 현재 손실값
            additional_info: 추가 정보
            
        Returns:
            저장 성공 여부
        """
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 체크포인트 데이터 준비
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'model_info': {
                    'total_parameters': ModelUtils.count_parameters(model)['total_parameters'],
                    'model_size': ModelUtils.get_model_size(model)
                }
            }
            
            # 옵티마이저 상태 추가
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            # 추가 정보 추가
            if additional_info:
                checkpoint.update(additional_info)
            
            # 체크포인트 저장
            torch.save(checkpoint, filepath)
            
            logger.info(f"모델 체크포인트 저장 완료: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"모델 체크포인트 저장 중 오류 발생: {e}")
            return False
    
    @staticmethod
    def load_model_checkpoint(model: nn.Module, 
                            filepath: str, 
                            optimizer: Optional[torch.optim.Optimizer] = None,
                            device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        모델 체크포인트 로드
        
        Args:
            model: 로드할 모델
            filepath: 체크포인트 파일 경로
            optimizer: 옵티마이저 (선택사항)
            device: 디바이스
            
        Returns:
            로드된 체크포인트 정보
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {filepath}")
            
            # 디바이스 설정
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 체크포인트 로드
            checkpoint = torch.load(filepath, map_location=device)
            
            # 모델 상태 로드
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            # 옵티마이저 상태 로드
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            logger.info(f"모델 체크포인트 로드 완료: {filepath}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"모델 체크포인트 로드 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def convert_model_format(model: nn.Module, 
                           target_format: str,
                           device: Optional[torch.device] = None) -> nn.Module:
        """
        모델 형식 변환
        
        Args:
            model: 변환할 모델
            target_format: 목표 형식 ('fp32', 'fp16', 'int8')
            device: 디바이스
            
        Returns:
            변환된 모델
        """
        try:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if target_format == 'fp16':
                # FP16 변환
                if device.type == 'cuda':
                    model = model.half()
                else:
                    logger.warning("FP16 변환은 CUDA 디바이스에서만 지원됩니다")
                    
            elif target_format == 'int8':
                # INT8 양자화
                try:
                    model = torch.quantization.quantize_dynamic(
                        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                    )
                except Exception as e:
                    logger.warning(f"INT8 양자화 실패, 원본 모델 반환: {e}")
                    
            elif target_format == 'fp32':
                # FP32 변환
                model = model.float()
                
            else:
                logger.warning(f"지원하지 않는 형식: {target_format}")
            
            return model
            
        except Exception as e:
            logger.error(f"모델 형식 변환 중 오류 발생: {e}")
            return model
    
    @staticmethod
    def optimize_model_for_inference(model: nn.Module, 
                                   device: Optional[torch.device] = None) -> nn.Module:
        """
        추론을 위한 모델 최적화
        
        Args:
            model: 최적화할 모델
            device: 디바이스
            
        Returns:
            최적화된 모델
        """
        try:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 평가 모드로 설정
            model.eval()
            
            # CUDA 최적화
            if device.type == 'cuda':
                # JIT 컴파일 (가능한 경우)
                try:
                    if hasattr(torch.jit, 'script'):
                        model = torch.jit.script(model)
                except Exception as e:
                    logger.debug(f"JIT 컴파일 실패: {e}")
                
                # CUDA 그래프 최적화
                if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                    logger.info("CUDA AMP 자동 혼합 정밀도 활성화")
            
            # 추론 모드로 설정
            with torch.no_grad():
                model = model
            
            logger.info("모델 추론 최적화 완료")
            return model
            
        except Exception as e:
            logger.error(f"모델 최적화 중 오류 발생: {e}")
            return model
    
    @staticmethod
    def get_model_summary(model: nn.Module, 
                         input_shape: Tuple[int, ...],
                         device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        모델 요약 정보 생성
        
        Args:
            model: 분석할 모델
            input_shape: 입력 텐서 형태
            device: 디바이스
            
        Returns:
            모델 요약 정보
        """
        try:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 모델을 디바이스로 이동
            model = model.to(device)
            
            # 더미 입력 생성
            dummy_input = torch.randn(input_shape, device=device)
            
            # 모델 정보 수집
            summary = {
                'model_name': model.__class__.__name__,
                'input_shape': input_shape,
                'device': str(device),
                'parameters': ModelUtils.count_parameters(model),
                'model_size': ModelUtils.get_model_size(model),
                'layers': []
            }
            
            # 레이어별 정보 수집
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # 리프 노드만
                    layer_info = {
                        'name': name,
                        'type': module.__class__.__name__,
                        'parameters': sum(p.numel() for p in module.parameters()),
                        'shape': list(module.weight.shape) if hasattr(module, 'weight') else None
                    }
                    summary['layers'].append(layer_info)
            
            # 메모리 사용량 추정
            try:
                # 입력을 모델에 통과시켜 출력 형태 확인
                with torch.no_grad():
                    output = model(dummy_input)
                
                summary['output_shape'] = list(output.shape)
                
                # 메모리 사용량 추정
                input_memory = dummy_input.element_size() * dummy_input.numel()
                output_memory = output.element_size() * output.numel()
                
                summary['memory_usage'] = {
                    'input_memory_mb': input_memory / (1024 * 1024),
                    'output_memory_mb': output_memory / (1024 * 1024),
                    'total_memory_mb': (input_memory + output_memory) / (1024 * 1024)
                }
                
            except Exception as e:
                logger.warning(f"메모리 사용량 추정 실패: {e}")
                summary['output_shape'] = None
                summary['memory_usage'] = None
            
            return summary
            
        except Exception as e:
            logger.error(f"모델 요약 생성 중 오류 발생: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def benchmark_model(model: nn.Module, 
                       input_shape: Tuple[int, ...],
                       num_runs: int = 100,
                       device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        모델 성능 벤치마크
        
        Args:
            model: 벤치마크할 모델
            input_shape: 입력 텐서 형태
            num_runs: 실행 횟수
            device: 디바이스
            
        Returns:
            벤치마크 결과
        """
        try:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 모델을 디바이스로 이동 및 최적화
            model = model.to(device)
            model = ModelUtils.optimize_model_for_inference(model, device)
            
            # 더미 입력 생성
            dummy_input = torch.randn(input_shape, device=device)
            
            # 워밍업
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # CUDA 동기화
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 벤치마크 실행
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(dummy_input)
            
            # CUDA 동기화
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            # 결과 계산
            total_time = end_time - start_time
            avg_time = total_time / num_runs
            fps = num_runs / total_time
            
            # 메모리 사용량
            memory_usage = ModelUtils._get_memory_usage(device)
            
            benchmark_results = {
                'total_runs': num_runs,
                'total_time_seconds': total_time,
                'average_time_seconds': avg_time,
                'fps': fps,
                'device': str(device),
                'memory_usage': memory_usage
            }
            
            logger.info(f"모델 벤치마크 완료: 평균 {avg_time*1000:.2f}ms, FPS: {fps:.2f}")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"모델 벤치마크 중 오류 발생: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _get_memory_usage(device: torch.device) -> Dict[str, float]:
        """메모리 사용량 조회"""
        try:
            if device.type == 'cuda':
                memory_info = {
                    'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                    'cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                    'total_mb': torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                }
            else:
                memory_info = {
                    'allocated_mb': 0.0,
                    'cached_mb': 0.0,
                    'total_mb': 0.0
                }
            
            return memory_info
            
        except Exception as e:
            logger.error(f"메모리 사용량 조회 중 오류 발생: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def export_model(model: nn.Module, 
                    filepath: str, 
                    input_shape: Tuple[int, ...],
                    export_format: str = 'onnx',
                    device: Optional[torch.device] = None) -> bool:
        """
        모델 내보내기
        
        Args:
            model: 내보낼 모델
            filepath: 저장 경로
            input_shape: 입력 텐서 형태
            export_format: 내보내기 형식 ('onnx', 'torchscript')
            device: 디바이스
            
        Returns:
            내보내기 성공 여부
        """
        try:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 모델을 디바이스로 이동 및 최적화
            model = model.to(device)
            model = ModelUtils.optimize_model_for_inference(model, device)
            
            # 더미 입력 생성
            dummy_input = torch.randn(input_shape, device=device)
            
            if export_format == 'onnx':
                # ONNX 형식으로 내보내기
                torch.onnx.export(
                    model,
                    dummy_input,
                    filepath,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
            elif export_format == 'torchscript':
                # TorchScript 형식으로 내보내기
                scripted_model = torch.jit.script(model)
                torch.jit.save(scripted_model, filepath)
                
            else:
                raise ValueError(f"지원하지 않는 내보내기 형식: {export_format}")
            
            logger.info(f"모델 내보내기 완료: {filepath} ({export_format})")
            return True
            
        except Exception as e:
            logger.error(f"모델 내보내기 중 오류 발생: {e}")
            return False
    
    @staticmethod
    def validate_model_output(model: nn.Module, 
                            input_tensor: torch.Tensor,
                            expected_output_shape: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
        """
        모델 출력 검증
        
        Args:
            model: 검증할 모델
            input_tensor: 입력 텐서
            expected_output_shape: 예상 출력 형태
            
        Returns:
            검증 결과
        """
        try:
            # 모델을 평가 모드로 설정
            model.eval()
            
            # 추론 실행
            with torch.no_grad():
                output = model(input_tensor)
            
            # 출력 검증
            validation_result = {
                'input_shape': list(input_tensor.shape),
                'output_shape': list(output.shape),
                'output_dtype': str(output.dtype),
                'output_device': str(output.device),
                'output_range': {
                    'min': float(output.min()),
                    'max': float(output.max()),
                    'mean': float(output.mean()),
                    'std': float(output.std())
                },
                'is_valid': True,
                'validation_messages': []
            }
            
            # 출력 형태 검증
            if expected_output_shape:
                if list(output.shape) != list(expected_output_shape):
                    validation_result['is_valid'] = False
                    validation_result['validation_messages'].append(
                        f"출력 형태 불일치: 예상 {expected_output_shape}, 실제 {list(output.shape)}"
                    )
            
            # NaN/Inf 검증
            if torch.isnan(output).any():
                validation_result['is_valid'] = False
                validation_result['validation_messages'].append("출력에 NaN 값이 포함되어 있습니다")
            
            if torch.isinf(output).any():
                validation_result['is_valid'] = False
                validation_result['validation_messages'].append("출력에 Inf 값이 포함되어 있습니다")
            
            # 값 범위 검증
            if output.min() < -1e6 or output.max() > 1e6:
                validation_result['validation_messages'].append("출력 값의 범위가 비정상적으로 큽니다")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"모델 출력 검증 중 오류 발생: {e}")
            return {
                'is_valid': False,
                'validation_messages': [f"검증 중 오류 발생: {str(e)}"]
            }
