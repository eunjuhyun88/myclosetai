#!/usr/bin/env python3
"""
🔥 MyCloset AI - 메모리 모니터링 시스템
========================================

각 단계별 메모리 사용량을 실시간으로 모니터링하고 관리하는 시스템

Author: MyCloset AI Team
Date: 2025-08-07
Version: 1.0
"""

import os
import psutil
import gc
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

# PyTorch 관련
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """메모리 스냅샷 데이터 클래스"""
    timestamp: float = field(default_factory=time.time)
    step_name: str = ""
    step_id: str = ""
    
    # 시스템 메모리
    system_total_gb: float = 0.0
    system_used_gb: float = 0.0
    system_available_gb: float = 0.0
    system_percent: float = 0.0
    
    # 프로세스 메모리
    process_rss_gb: float = 0.0
    process_vms_gb: float = 0.0
    process_percent: float = 0.0
    
    # GPU 메모리 (가능한 경우)
    gpu_total_gb: float = 0.0
    gpu_used_gb: float = 0.0
    gpu_available_gb: float = 0.0
    gpu_percent: float = 0.0
    
    # 가비지 컬렉션 정보
    gc_objects: int = 0
    gc_collections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'timestamp': self.timestamp,
            'step_name': self.step_name,
            'step_id': self.step_id,
            'system': {
                'total_gb': self.system_total_gb,
                'used_gb': self.system_used_gb,
                'available_gb': self.system_available_gb,
                'percent': self.system_percent
            },
            'process': {
                'rss_gb': self.process_rss_gb,
                'vms_gb': self.process_vms_gb,
                'percent': self.process_percent
            },
            'gpu': {
                'total_gb': self.gpu_total_gb,
                'used_gb': self.gpu_used_gb,
                'available_gb': self.gpu_available_gb,
                'percent': self.gpu_percent
            },
            'gc': {
                'objects': self.gc_objects,
                'collections': self.gc_collections
            }
        }

class MemoryMonitor:
    """메모리 모니터링 클래스"""
    
    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.warning_threshold = 0.7  # 70% 경고
        self.critical_threshold = 0.85  # 85% 위험
        self.optimal_range = (0.2, 0.4)  # 20-40% 적정 범위
        
    def take_snapshot(self, step_name: str = "", step_id: str = "") -> MemorySnapshot:
        """메모리 스냅샷 생성"""
        snapshot = MemorySnapshot(step_name=step_name, step_id=step_id)
        
        try:
            # 시스템 메모리 정보
            memory = psutil.virtual_memory()
            snapshot.system_total_gb = memory.total / (1024**3)
            snapshot.system_used_gb = memory.used / (1024**3)
            snapshot.system_available_gb = memory.available / (1024**3)
            snapshot.system_percent = memory.percent / 100.0
            
            # 프로세스 메모리 정보
            process = psutil.Process()
            memory_info = process.memory_info()
            snapshot.process_rss_gb = memory_info.rss / (1024**3)
            snapshot.process_vms_gb = memory_info.vms / (1024**3)
            snapshot.process_percent = process.memory_percent() / 100.0
            
            # GPU 메모리 정보 (가능한 경우)
            if TORCH_AVAILABLE:
                try:
                    if torch.backends.mps.is_available():
                        # M3 Max MPS 메모리 정보
                        snapshot.gpu_total_gb = 16.0  # M3 Max 통합 메모리
                        snapshot.gpu_used_gb = snapshot.system_used_gb * 0.3  # 추정치
                        snapshot.gpu_available_gb = snapshot.gpu_total_gb - snapshot.gpu_used_gb
                        snapshot.gpu_percent = snapshot.gpu_used_gb / snapshot.gpu_total_gb
                    elif torch.cuda.is_available():
                        # CUDA 메모리 정보
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory
                        snapshot.gpu_total_gb = gpu_memory / (1024**3)
                        snapshot.gpu_used_gb = torch.cuda.memory_allocated() / (1024**3)
                        snapshot.gpu_available_gb = snapshot.gpu_total_gb - snapshot.gpu_used_gb
                        snapshot.gpu_percent = snapshot.gpu_used_gb / snapshot.gpu_total_gb
                except Exception as e:
                    logger.debug(f"GPU 메모리 정보 수집 실패: {e}")
            
            # 가비지 컬렉션 정보
            snapshot.gc_objects = len(gc.get_objects())
            snapshot.gc_collections = gc.get_count()[0]
            
        except Exception as e:
            logger.error(f"메모리 스냅샷 생성 실패: {e}")
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def log_memory_status(self, step_name: str = "", step_id: str = "") -> None:
        """메모리 상태 로깅"""
        snapshot = self.take_snapshot(step_name, step_id)
        
        # 메모리 상태 평가
        status = self._evaluate_memory_status(snapshot)
        
        # 로깅
        logger.info(f"🔥 [메모리 모니터] {step_name} ({step_id}) - {status}")
        logger.info(f"   💾 시스템: {snapshot.system_used_gb:.1f}GB/{snapshot.system_total_gb:.1f}GB ({snapshot.system_percent*100:.1f}%)")
        logger.info(f"   🖥️  프로세스: {snapshot.process_rss_gb:.1f}GB ({snapshot.process_percent*100:.1f}%)")
        
        if snapshot.gpu_total_gb > 0:
            logger.info(f"   🎮 GPU: {snapshot.gpu_used_gb:.1f}GB/{snapshot.gpu_total_gb:.1f}GB ({snapshot.gpu_percent*100:.1f}%)")
        
        logger.info(f"   🗑️  GC 객체: {snapshot.gc_objects:,}개")
        
        # 경고/위험 상태 알림
        if snapshot.system_percent > self.critical_threshold:
            logger.warning(f"⚠️ [메모리 위험] 시스템 메모리 사용량이 {snapshot.system_percent*100:.1f}%로 위험 수준입니다!")
        elif snapshot.system_percent > self.warning_threshold:
            logger.warning(f"⚠️ [메모리 경고] 시스템 메모리 사용량이 {snapshot.system_percent*100:.1f}%로 높습니다.")
        
        # 적정 범위 확인
        if self.optimal_range[0] <= snapshot.system_percent <= self.optimal_range[1]:
            logger.info(f"✅ [메모리 적정] 시스템 메모리 사용량이 적정 범위({self.optimal_range[0]*100:.0f}-{self.optimal_range[1]*100:.0f}%) 내에 있습니다.")
    
    def _evaluate_memory_status(self, snapshot: MemorySnapshot) -> str:
        """메모리 상태 평가"""
        if snapshot.system_percent > self.critical_threshold:
            return "위험"
        elif snapshot.system_percent > self.warning_threshold:
            return "경고"
        elif self.optimal_range[0] <= snapshot.system_percent <= self.optimal_range[1]:
            return "적정"
        else:
            return "양호"
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """메모리 정리"""
        start_time = time.time()
        
        # 정리 전 스냅샷
        before_snapshot = self.take_snapshot("before_cleanup", "cleanup")
        
        try:
            # Python 가비지 컬렉션
            collected_objects = gc.collect()
            
            # PyTorch 메모리 정리
            if TORCH_AVAILABLE:
                try:
                    if torch.backends.mps.is_available():
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                        if hasattr(torch.mps, 'synchronize'):
                            torch.mps.synchronize()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        if aggressive:
                            torch.cuda.synchronize()
                except Exception as e:
                    logger.warning(f"GPU 메모리 정리 실패: {e}")
            
            # 정리 후 스냅샷
            after_snapshot = self.take_snapshot("after_cleanup", "cleanup")
            
            # 정리 효과 계산
            memory_freed = before_snapshot.system_used_gb - after_snapshot.system_used_gb
            process_freed = before_snapshot.process_rss_gb - after_snapshot.process_rss_gb
            
            cleanup_time = time.time() - start_time
            
            result = {
                'success': True,
                'cleanup_time': cleanup_time,
                'memory_freed_gb': memory_freed,
                'process_freed_gb': process_freed,
                'objects_collected': collected_objects,
                'aggressive': aggressive
            }
            
            logger.info(f"✅ 메모리 정리 완료: {memory_freed:.2f}GB 해제, {cleanup_time:.2f}초 소요")
            return result
            
        except Exception as e:
            logger.error(f"❌ 메모리 정리 실패: {e}")
            return {
                'success': False,
                'error': str(e),
                'cleanup_time': time.time() - start_time
            }
    
    def get_memory_trend(self, steps: int = 10) -> Dict[str, Any]:
        """메모리 사용량 트렌드 분석"""
        if len(self.snapshots) < 2:
            return {'error': '충분한 스냅샷이 없습니다'}
        
        recent_snapshots = self.snapshots[-steps:]
        
        # 메모리 사용량 변화
        memory_changes = []
        for i in range(1, len(recent_snapshots)):
            change = recent_snapshots[i].system_used_gb - recent_snapshots[i-1].system_used_gb
            memory_changes.append(change)
        
        # 통계 계산
        avg_change = sum(memory_changes) / len(memory_changes) if memory_changes else 0
        max_change = max(memory_changes) if memory_changes else 0
        min_change = min(memory_changes) if memory_changes else 0
        
        return {
            'total_snapshots': len(self.snapshots),
            'analyzed_snapshots': len(recent_snapshots),
            'average_memory_change_gb': avg_change,
            'max_memory_increase_gb': max_change,
            'max_memory_decrease_gb': min_change,
            'trend': 'increasing' if avg_change > 0.1 else 'decreasing' if avg_change < -0.1 else 'stable'
        }
    
    def print_memory_report(self) -> None:
        """메모리 사용량 리포트 출력"""
        if not self.snapshots:
            logger.info("📊 메모리 리포트: 스냅샷이 없습니다")
            return
        
        latest = self.snapshots[-1]
        trend = self.get_memory_trend()
        
        logger.info("📊 === 메모리 사용량 리포트 ===")
        logger.info(f"   📈 총 스냅샷: {len(self.snapshots)}개")
        logger.info(f"   💾 현재 시스템 메모리: {latest.system_used_gb:.1f}GB/{latest.system_total_gb:.1f}GB ({latest.system_percent*100:.1f}%)")
        logger.info(f"   🖥️  현재 프로세스 메모리: {latest.process_rss_gb:.1f}GB ({latest.process_percent*100:.1f}%)")
        logger.info(f"   🗑️  현재 GC 객체: {latest.gc_objects:,}개")
        
        if trend.get('trend'):
            logger.info(f"   📊 메모리 트렌드: {trend['trend']} (평균 변화: {trend['average_memory_change_gb']:.2f}GB)")
        
        # 권장사항
        if latest.system_percent > self.critical_threshold:
            logger.warning("   ⚠️ 권장사항: 즉시 메모리 정리를 수행하세요!")
        elif latest.system_percent > self.warning_threshold:
            logger.warning("   ⚠️ 권장사항: 메모리 정리를 고려하세요.")
        elif latest.system_percent < self.optimal_range[0]:
            logger.info("   ✅ 메모리 사용량이 매우 낮습니다.")
        else:
            logger.info("   ✅ 메모리 사용량이 적정 범위입니다.")

# 전역 메모리 모니터 인스턴스
memory_monitor = MemoryMonitor()

def get_memory_monitor() -> MemoryMonitor:
    """전역 메모리 모니터 반환"""
    return memory_monitor

def log_step_memory(step_name: str, step_id: str = "") -> None:
    """단계별 메모리 로깅 (편의 함수)"""
    memory_monitor.log_memory_status(step_name, step_id)

def cleanup_step_memory(aggressive: bool = False) -> Dict[str, Any]:
    """단계별 메모리 정리 (편의 함수)"""
    return memory_monitor.cleanup_memory(aggressive)
