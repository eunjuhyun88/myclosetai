#!/usr/bin/env python3
"""
MyCloset AI - 8단계 파이프라인 성능 벤치마크 도구
M3 Max 최적화 성능 측정 및 분석

성능 측정 항목:
- 각 단계별 처리 시간
- 메모리 사용량 (RAM/GPU)
- CPU/GPU 사용률
- 품질 점수
- 전체 파이프라인 처리량
"""

import os
import sys
import time
import json
import psutil
import platform
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import numpy as np
from PIL import Image
import cv2

# 색상 출력
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def log_info(msg): print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")
def log_success(msg): print(f"{Colors.GREEN}✅ {msg}{Colors.END}")
def log_warning(msg): print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")
def log_error(msg): print(f"{Colors.RED}❌ {msg}{Colors.END}")
def log_step(msg): print(f"{Colors.PURPLE}🔄 {msg}{Colors.END}")

@dataclass
class BenchmarkResult:
    """벤치마크 결과 데이터 클래스"""
    step_name: str
    processing_time: float
    memory_before: float
    memory_after: float
    memory_peak: float
    cpu_usage: float
    gpu_usage: Optional[float]
    input_size: Tuple[int, int]
    output_size: Tuple[int, int]
    quality_score: float
    error_message: Optional[str] = None

@dataclass
class SystemInfo:
    """시스템 정보 데이터 클래스"""
    os_type: str
    os_version: str
    architecture: str
    cpu_model: str
    cpu_cores: int
    total_ram: float
    gpu_model: Optional[str]
    gpu_memory: Optional[float]
    pytorch_version: str
    device_type: str
    mps_available: bool
    cuda_available: bool

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.memory_tracker = []
        
    def start_monitoring(self):
        """모니터링 시작"""
        self.start_time = time.perf_counter()
        self.memory_tracker = []
        
    def get_memory_usage(self) -> float:
        """현재 메모리 사용량 (GB)"""
        return self.process.memory_info().rss / (1024**3)
    
    def get_cpu_usage(self) -> float:
        """CPU 사용률 (%)"""
        return self.process.cpu_percent(interval=0.1)
    
    def get_gpu_usage(self) -> Optional[float]:
        """GPU 사용률 (%) - MPS/CUDA"""
        try:
            if torch.backends.mps.is_available():
                # MPS는 직접적인 사용률 측정이 어려움
                return None
            elif torch.cuda.is_available():
                return torch.cuda.utilization()
            return None
        except:
            return None
    
    def stop_monitoring(self) -> float:
        """모니터링 종료 및 경과 시간 반환"""
        if self.start_time:
            return time.perf_counter() - self.start_time
        return 0.0

class MockPipelineStep:
    """실제 AI 모델 대신 사용할 모킹 클래스"""
    
    def __init__(self, step_name: str, complexity: float = 1.0):
        self.step_name = step_name
        self.complexity = complexity
        
    def process(self, input_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """가상 처리 수행"""
        # 복잡도에 따른 처리 시간 시뮬레이션
        processing_time = self.complexity * 0.5  # 기본 0.5초 * 복잡도
        
        # CPU 부하 시뮬레이션
        for _ in range(int(self.complexity * 1000000)):
            _ = np.random.random()
        
        # 메모리 할당 시뮬레이션
        temp_data = np.random.random((
            int(input_data.shape[0] * self.complexity),
            int(input_data.shape[1] * self.complexity),
            input_data.shape[2]
        )).astype(np.float32)
        
        time.sleep(processing_time)
        
        # 품질 점수 시뮬레이션 (복잡도가 높을수록 품질 좋음)
        quality_score = min(0.95, 0.6 + self.complexity * 0.2)
        
        # 결과 크기는 입력과 동일하게 유지
        result = input_data.copy()
        
        return result, quality_score

class PipelineBenchmark:
    """8단계 파이프라인 벤치마크 클래스"""
    
    def __init__(self, device: str = "auto"):
        self.device = self._setup_device(device)
        self.monitor = PerformanceMonitor()
        self.results: List[BenchmarkResult] = []
        
        # 8단계 파이프라인 스텝들 (실제 모델 대신 모킹)
        self.pipeline_steps = {
            "01_human_parsing": MockPipelineStep("Human Parsing", 2.0),
            "02_pose_estimation": MockPipelineStep("Pose Estimation", 1.5),
            "03_cloth_segmentation": MockPipelineStep("Cloth Segmentation", 1.8),
            "04_geometric_matching": MockPipelineStep("Geometric Matching", 1.2),
            "05_cloth_warping": MockPipelineStep("Cloth Warping", 1.0),
            "06_virtual_fitting": MockPipelineStep("Virtual Fitting", 3.0),
            "07_post_processing": MockPipelineStep("Post Processing", 0.8),
            "08_quality_assessment": MockPipelineStep("Quality Assessment", 0.5)
        }
        
    def _setup_device(self, device: str) -> torch.device:
        """디바이스 설정"""
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        torch_device = torch.device(device)
        log_info(f"사용 디바이스: {torch_device}")
        return torch_device
    
    def get_system_info(self) -> SystemInfo:
        """시스템 정보 수집"""
        gpu_model = None
        gpu_memory = None
        
        if torch.backends.mps.is_available():
            gpu_model = "Apple Silicon GPU (MPS)"
            # MPS 메모리는 시스템 RAM과 공유되므로 별도 측정 어려움
            gpu_memory = None
        elif torch.cuda.is_available():
            gpu_model = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return SystemInfo(
            os_type=platform.system(),
            os_version=platform.release(),
            architecture=platform.machine(),
            cpu_model=platform.processor() or "Unknown",
            cpu_cores=psutil.cpu_count(),
            total_ram=psutil.virtual_memory().total / (1024**3),
            gpu_model=gpu_model,
            gpu_memory=gpu_memory,
            pytorch_version=torch.__version__,
            device_type=str(self.device),
            mps_available=torch.backends.mps.is_available(),
            cuda_available=torch.cuda.is_available()
        )
    
    def create_test_input(self, size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """테스트용 입력 이미지 생성"""
        # RGB 이미지 생성 (사람 실루엣 형태)
        image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # 간단한 사람 형태 그리기
        center_x, center_y = size[0] // 2, size[1] // 2
        
        # 머리 (원)
        cv2.circle(image, (center_x, center_y - 100), 30, (255, 255, 255), -1)
        
        # 몸통 (사각형)
        cv2.rectangle(image, 
                     (center_x - 40, center_y - 70),
                     (center_x + 40, center_y + 50),
                     (255, 255, 255), -1)
        
        # 팔
        cv2.rectangle(image,
                     (center_x - 70, center_y - 50),
                     (center_x - 40, center_y),
                     (255, 255, 255), -1)
        cv2.rectangle(image,
                     (center_x + 40, center_y - 50),
                     (center_x + 70, center_y),
                     (255, 255, 255), -1)
        
        # 다리
        cv2.rectangle(image,
                     (center_x - 25, center_y + 50),
                     (center_x - 10, center_y + 120),
                     (255, 255, 255), -1)
        cv2.rectangle(image,
                     (center_x + 10, center_y + 50),
                     (center_x + 25, center_y + 120),
                     (255, 255, 255), -1)
        
        return image.astype(np.float32) / 255.0
    
    def benchmark_step(self, step_name: str, input_data: np.ndarray) -> BenchmarkResult:
        """개별 파이프라인 스텝 벤치마크"""
        step = self.pipeline_steps[step_name]
        
        # 메모리 정리
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        # 초기 상태 측정
        memory_before = self.monitor.get_memory_usage()
        
        # 모니터링 시작
        self.monitor.start_monitoring()
        
        try:
            # 스텝 실행
            output_data, quality_score = step.process(input_data)
            
            # 처리 시간 측정
            processing_time = self.monitor.stop_monitoring()
            
            # 최종 상태 측정
            memory_after = self.monitor.get_memory_usage()
            memory_peak = max(memory_before, memory_after)
            cpu_usage = self.monitor.get_cpu_usage()
            gpu_usage = self.monitor.get_gpu_usage()
            
            return BenchmarkResult(
                step_name=step_name,
                processing_time=processing_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=memory_peak,
                cpu_usage=cpu_usage,
                gpu_usage=gpu_usage,
                input_size=input_data.shape[:2],
                output_size=output_data.shape[:2],
                quality_score=quality_score
            )
            
        except Exception as e:
            processing_time = self.monitor.stop_monitoring()
            return BenchmarkResult(
                step_name=step_name,
                processing_time=processing_time,
                memory_before=memory_before,
                memory_after=self.monitor.get_memory_usage(),
                memory_peak=memory_before,
                cpu_usage=0.0,
                gpu_usage=None,
                input_size=input_data.shape[:2],
                output_size=(0, 0),
                quality_score=0.0,
                error_message=str(e)
            )
    
    def run_full_pipeline(self, input_sizes: List[Tuple[int, int]] = [(256, 256), (512, 512), (1024, 1024)]) -> Dict:
        """전체 파이프라인 벤치마크 실행"""
        all_results = {}
        
        for size in input_sizes:
            log_step(f"해상도 {size[0]}x{size[1]} 벤치마크 실행")
            
            # 테스트 입력 생성
            test_input = self.create_test_input(size)
            current_data = test_input
            
            size_results = []
            total_start_time = time.perf_counter()
            
            # 8단계 순차 실행
            for step_name in self.pipeline_steps.keys():
                log_info(f"  실행 중: {step_name}")
                
                result = self.benchmark_step(step_name, current_data)
                size_results.append(result)
                
                if result.error_message:
                    log_error(f"    오류: {result.error_message}")
                    break
                else:
                    log_success(f"    완료: {result.processing_time:.2f}s, 품질: {result.quality_score:.2f}")
            
            total_time = time.perf_counter() - total_start_time
            
            all_results[f"{size[0]}x{size[1]}"] = {
                "steps": size_results,
                "total_time": total_time,
                "average_quality": np.mean([r.quality_score for r in size_results if r.error_message is None]),
                "total_memory_used": max([r.memory_peak for r in size_results]) - min([r.memory_before for r in size_results]),
                "throughput": 1.0 / total_time if total_time > 0 else 0.0
            }
        
        return all_results
    
    def generate_report(self, results: Dict, output_file: Optional[str] = None) -> str:
        """벤치마크 결과 리포트 생성"""
        system_info = self.get_system_info()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
{Colors.BOLD}🔥 MyCloset AI - 8단계 파이프라인 성능 벤치마크 리포트{Colors.END}
{'='*80}

📅 실행 시간: {timestamp}
🖥️  시스템 정보:
   OS: {system_info.os_type} {system_info.os_version} ({system_info.architecture})
   CPU: {system_info.cpu_model} ({system_info.cpu_cores} cores)
   RAM: {system_info.total_ram:.1f} GB
   GPU: {system_info.gpu_model or 'N/A'}
   GPU 메모리: {f'{system_info.gpu_memory:.1f} GB' if system_info.gpu_memory else 'N/A'}
   PyTorch: {system_info.pytorch_version}
   디바이스: {system_info.device_type}
   MPS 지원: {'✅' if system_info.mps_available else '❌'}
   CUDA 지원: {'✅' if system_info.cuda_available else '❌'}

📊 성능 결과 요약:
"""
        
        # 해상도별 결과 테이블
        report += f"\n{'해상도':<12} {'총 시간':<10} {'처리량':<12} {'평균 품질':<12} {'메모리 사용':<12}\n"
        report += "-" * 70 + "\n"
        
        for resolution, data in results.items():
            throughput = data['throughput']
            quality = data['average_quality']
            memory = data['total_memory_used']
            total_time = data['total_time']
            
            report += f"{resolution:<12} {total_time:<10.2f}s {throughput:<12.2f}/s {quality:<12.2f} {memory:<12.2f}GB\n"
        
        # 단계별 상세 분석
        report += f"\n📋 단계별 상세 분석:\n"
        
        for resolution, data in results.items():
            report += f"\n🔸 {resolution} 해상도:\n"
            report += f"{'단계':<25} {'시간':<8} {'메모리':<10} CPU{'%':<4} {'품질':<6}\n"
            report += "-" * 55 + "\n"
            
            for step_result in data['steps']:
                if step_result.error_message:
                    report += f"{step_result.step_name:<25} ERROR    -         -    -\n"
                else:
                    memory_delta = step_result.memory_after - step_result.memory_before
                    report += f"{step_result.step_name:<25} {step_result.processing_time:<8.2f}s {memory_delta:<+8.2f}GB {step_result.cpu_usage:<5.1f} {step_result.quality_score:<6.2f}\n"
        
        # 성능 분석 및 권장사항
        report += f"\n💡 성능 분석 및 권장사항:\n"
        
        # 최적 해상도 찾기
        best_resolution = max(results.keys(), key=lambda x: results[x]['average_quality'] / results[x]['total_time'])
        report += f"   🎯 권장 해상도: {best_resolution} (품질/시간 비율 최적)\n"
        
        # 병목 단계 찾기
        all_steps_time = {}
        for data in results.values():
            for step in data['steps']:
                if step.step_name not in all_steps_time:
                    all_steps_time[step.step_name] = []
                all_steps_time[step.step_name].append(step.processing_time)
        
        bottleneck_step = max(all_steps_time.keys(), key=lambda x: np.mean(all_steps_time[x]))
        bottleneck_time = np.mean(all_steps_time[bottleneck_step])
        report += f"   ⚠️  병목 단계: {bottleneck_step} (평균 {bottleneck_time:.2f}s)\n"
        
        # 메모리 사용량 분석
        max_memory = max([data['total_memory_used'] for data in results.values()])
        if max_memory > 8.0:
            report += f"   🔴 높은 메모리 사용량: {max_memory:.1f}GB (최적화 필요)\n"
        elif max_memory > 4.0:
            report += f"   🟡 보통 메모리 사용량: {max_memory:.1f}GB (허용 범위)\n"
        else:
            report += f"   🟢 낮은 메모리 사용량: {max_memory:.1f}GB (효율적)\n"
        
        # 디바이스별 권장사항
        if system_info.device_type == "mps":
            report += f"\n🍎 M3 Max 최적화 권장사항:\n"
            report += f"   • FP16 모드 활성화로 메모리 사용량 50% 절약\n"
            report += f"   • 배치 크기 조정으로 처리량 향상\n"
            report += f"   • 모델 양자화로 속도 개선\n"
        elif system_info.device_type == "cuda":
            report += f"\n🚀 CUDA 최적화 권장사항:\n"
            report += f"   • CUDA Streams 활용으로 병렬 처리\n"
            report += f"   • TensorRT 최적화 적용\n"
            report += f"   • 혼합 정밀도 훈련 활용\n"
        else:
            report += f"\n💻 CPU 최적화 권장사항:\n"
            report += f"   • OpenMP 스레드 수 조정\n"
            report += f"   • ONNX 런타임 활용\n"
            report += f"   • 모델 경량화 적용\n"
        
        # 파일로 저장
        if output_file:
            # 색상 코드 제거한 버전으로 저장
            clean_report = report
            for color in [Colors.BLUE, Colors.GREEN, Colors.YELLOW, Colors.RED, Colors.PURPLE, Colors.CYAN, Colors.BOLD, Colors.END]:
                clean_report = clean_report.replace(color, "")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(clean_report)
            log_success(f"리포트 저장됨: {output_file}")
        
        return report

def stress_test(benchmark: PipelineBenchmark, duration: int = 60) -> Dict:
    """스트레스 테스트 (지정된 시간 동안 연속 실행)"""
    log_step(f"{duration}초 스트레스 테스트 시작")
    
    start_time = time.time()
    iteration_count = 0
    results = []
    
    test_input = benchmark.create_test_input((512, 512))
    
    while time.time() - start_time < duration:
        iteration_start = time.time()
        
        # 랜덤하게 몇 개 스텝만 실행 (부하 조절)
        steps_to_run = np.random.choice(list(benchmark.pipeline_steps.keys()), 
                                       size=np.random.randint(3, 6), 
                                       replace=False)
        
        iteration_results = []
        for step_name in steps_to_run:
            result = benchmark.benchmark_step(step_name, test_input)
            iteration_results.append(result)
        
        iteration_time = time.time() - iteration_start
        results.append({
            'iteration': iteration_count,
            'time': iteration_time,
            'steps': len(steps_to_run),
            'memory_peak': max([r.memory_peak for r in iteration_results])
        })
        
        iteration_count += 1
        
        if iteration_count % 10 == 0:
            log_info(f"반복: {iteration_count}, 평균 시간: {np.mean([r['time'] for r in results[-10:]]):.2f}s")
    
    total_time = time.time() - start_time
    
    return {
        'total_time': total_time,
        'iterations': iteration_count,
        'average_iteration_time': np.mean([r['time'] for r in results]),
        'max_memory': max([r['memory_peak'] for r in results]),
        'throughput': iteration_count / total_time,
        'stability_score': 1.0 - (np.std([r['time'] for r in results]) / np.mean([r['time'] for r in results]))
    }

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="MyCloset AI - 8단계 파이프라인 성능 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="사용할 디바이스 (기본값: auto)"
    )
    parser.add_argument(
        "--resolutions",
        default="256x256,512x512,1024x1024",
        help="테스트할 해상도들 (기본값: 256x256,512x512,1024x1024)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="결과 리포트 저장 파일명"
    )
    parser.add_argument(
        "--stress-test",
        type=int,
        metavar="SECONDS",
        help="스트레스 테스트 실행 (초 단위)"
    )
    parser.add_argument(
        "--json-output",
        type=str,
        help="JSON 형태로 결과 저장"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="빠른 테스트 (512x512만 실행)"
    )
    
    args = parser.parse_args()
    
    # 헤더
    print(f"\n{Colors.BOLD}{Colors.BLUE}⚡ MyCloset AI - 파이프라인 벤치마크{Colors.END}")
    print("=" * 60)
    
    # 해상도 파싱
    if args.quick:
        resolutions = [(512, 512)]
    else:
        resolutions = []
        for res_str in args.resolutions.split(','):
            width, height = map(int, res_str.split('x'))
            resolutions.append((width, height))
    
    # 벤치마크 객체 생성
    benchmark = PipelineBenchmark(device=args.device)
    
    # 스트레스 테스트
    if args.stress_test:
        stress_results = stress_test(benchmark, args.stress_test)
        
        print(f"\n📊 스트레스 테스트 결과:")
        print(f"   총 시간: {stress_results['total_time']:.1f}초")
        print(f"   반복 횟수: {stress_results['iterations']}")
        print(f"   평균 반복 시간: {stress_results['average_iteration_time']:.2f}초")
        print(f"   처리량: {stress_results['throughput']:.2f} iterations/sec")
        print(f"   안정성 점수: {stress_results['stability_score']:.2f}")
        print(f"   최대 메모리: {stress_results['max_memory']:.2f}GB")
        
        # JSON 저장
        if args.json_output:
            stress_results['timestamp'] = datetime.now().isoformat()
            stress_results['system_info'] = asdict(benchmark.get_system_info())
            
            with open(args.json_output, 'w') as f:
                json.dump(stress_results, f, indent=2)
            log_success(f"JSON 결과 저장됨: {args.json_output}")
    
    else:
        # 일반 벤치마크
        log_info(f"테스트 해상도: {[f'{w}x{h}' for w, h in resolutions]}")
        
        results = benchmark.run_full_pipeline(resolutions)
        
        # 리포트 생성
        output_file = args.output or f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report = benchmark.generate_report(results, output_file)
        
        # 화면에 출력
        print(report)
        
        # JSON 저장
        if args.json_output:
            json_data = {
                'timestamp': datetime.now().isoformat(),
                'system_info': asdict(benchmark.get_system_info()),
                'results': {}
            }
            
            for resolution, data in results.items():
                json_data['results'][resolution] = {
                    'total_time': data['total_time'],
                    'average_quality': data['average_quality'],
                    'total_memory_used': data['total_memory_used'],
                    'throughput': data['throughput'],
                    'steps': [asdict(step) for step in data['steps']]
                }
            
            with open(args.json_output, 'w') as f:
                json.dump(json_data, f, indent=2)
            log_success(f"JSON 결과 저장됨: {args.json_output}")
    
    log_success("벤치마크 완료! 🎉")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_warning("\n사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        log_error(f"예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)