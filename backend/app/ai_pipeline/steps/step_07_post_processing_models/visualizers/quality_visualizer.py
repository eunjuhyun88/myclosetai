"""
Quality Visualizer
품질 메트릭과 결과를 시각화하는 클래스
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from typing import Dict, Any, List, Optional, Union, Tuple
import torch
from PIL import Image
import cv2
import logging
from pathlib import Path

# 프로젝트 로깅 설정 import
from backend.app.ai_pipeline.utils.logging_config import get_logger

logger = get_logger(__name__)

class QualityVisualizer:
    """
    품질 메트릭과 결과를 시각화하는 클래스
    """
    
    def __init__(self, output_dir: str = "quality_visualizations"):
        """
        Args:
            output_dir: 시각화 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 시각화 설정
        self.visualization_config = {
            'figure_size': (12, 8),
            'dpi': 300,
            'color_palette': 'viridis',
            'font_size': 12,
            'save_format': 'png',
            'show_plots': False
        }
        
        # matplotlib 스타일 설정
        plt.style.use('default')
        sns.set_palette(self.visualization_config['color_palette'])
        
        logger.info(f"QualityVisualizer initialized at {self.output_dir}")
    
    def plot_quality_metrics(self, metrics: Dict[str, float], 
                           title: str = "Quality Metrics", 
                           save_path: Optional[str] = None) -> str:
        """
        품질 메트릭을 막대 그래프로 시각화
        
        Args:
            metrics: 품질 메트릭 딕셔너리
            title: 그래프 제목
            save_path: 저장 경로 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        try:
            # 그래프 생성
            fig, ax = plt.subplots(figsize=self.visualization_config['figure_size'])
            
            # 메트릭 이름과 값 추출
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            # 막대 그래프 생성
            bars = ax.bar(metric_names, metric_values, 
                         color=sns.color_palette(self.visualization_config['color_palette']))
            
            # 그래프 꾸미기
            ax.set_title(title, fontsize=self.visualization_config['font_size'] + 2, fontweight='bold')
            ax.set_xlabel('Metrics', fontsize=self.visualization_config['font_size'])
            ax.set_ylabel('Score', fontsize=self.visualization_config['font_size'])
            ax.grid(True, alpha=0.3)
            
            # 막대 위에 값 표시
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            # y축 범위 설정
            ax.set_ylim(0, max(metric_values) * 1.1)
            
            # x축 레이블 회전
            plt.xticks(rotation=45, ha='right')
            
            # 레이아웃 조정
            plt.tight_layout()
            
            # 저장 경로 설정
            if save_path is None:
                save_path = self.output_dir / f"quality_metrics_{title.lower().replace(' ', '_')}.{self.visualization_config['save_format']}"
            else:
                save_path = Path(save_path)
            
            # 그래프 저장
            plt.savefig(save_path, dpi=self.visualization_config['dpi'], bbox_inches='tight')
            
            if self.visualization_config['show_plots']:
                plt.show()
            
            plt.close()
            
            logger.info(f"품질 메트릭 시각화 저장 완료: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"품질 메트릭 시각화 중 오류 발생: {e}")
            raise
    
    def plot_quality_comparison(self, original_metrics: Dict[str, float], 
                               processed_metrics: Dict[str, float],
                               title: str = "Quality Comparison",
                               save_path: Optional[str] = None) -> str:
        """
        원본과 처리된 이미지의 품질 메트릭을 비교하여 시각화
        
        Args:
            original_metrics: 원본 이미지 품질 메트릭
            processed_metrics: 처리된 이미지 품질 메트릭
            title: 그래프 제목
            save_path: 저장 경로
            
        Returns:
            저장된 파일 경로
        """
        try:
            # 그래프 생성
            fig, ax = plt.subplots(figsize=self.visualization_config['figure_size'])
            
            # 메트릭 이름 추출
            metric_names = list(original_metrics.keys())
            
            # 원본과 처리된 값 추출
            original_values = [original_metrics[name] for name in metric_names]
            processed_values = [processed_metrics[name] for name in metric_names]
            
            # x축 위치 설정
            x = np.arange(len(metric_names))
            width = 0.35
            
            # 막대 그래프 생성
            bars1 = ax.bar(x - width/2, original_values, width, 
                          label='Original', alpha=0.8, color='skyblue')
            bars2 = ax.bar(x + width/2, processed_values, width, 
                          label='Processed', alpha=0.8, color='lightcoral')
            
            # 그래프 꾸미기
            ax.set_title(title, fontsize=self.visualization_config['font_size'] + 2, fontweight='bold')
            ax.set_xlabel('Metrics', fontsize=self.visualization_config['font_size'])
            ax.set_ylabel('Score', fontsize=self.visualization_config['font_size'])
            ax.set_xticks(x)
            ax.set_xticklabels(metric_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 막대 위에 값 표시
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 레이아웃 조정
            plt.tight_layout()
            
            # 저장 경로 설정
            if save_path is None:
                save_path = self.output_dir / f"quality_comparison_{title.lower().replace(' ', '_')}.{self.visualization_config['save_format']}"
            else:
                save_path = Path(save_path)
            
            # 그래프 저장
            plt.savefig(save_path, dpi=self.visualization_config['dpi'], bbox_inches='tight')
            
            if self.visualization_config['show_plots']:
                plt.show()
            
            plt.close()
            
            logger.info(f"품질 비교 시각화 저장 완료: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"품질 비교 시각화 중 오류 발생: {e}")
            raise
    
    def plot_quality_trends(self, metrics_history: List[Dict[str, float]], 
                           metric_name: str = "overall_score",
                           title: str = "Quality Trends Over Time",
                           save_path: Optional[str] = None) -> str:
        """
        시간에 따른 품질 메트릭 변화를 시각화
        
        Args:
            metrics_history: 시간순 품질 메트릭 리스트
            metric_name: 시각화할 메트릭 이름
            title: 그래프 제목
            save_path: 저장 경로
            
        Returns:
            저장된 파일 경로
        """
        try:
            # 그래프 생성
            fig, ax = plt.subplots(figsize=self.visualization_config['figure_size'])
            
            # 시간 인덱스와 메트릭 값 추출
            time_indices = list(range(len(metrics_history)))
            metric_values = [metrics.get(metric_name, 0.0) for metrics in metrics_history]
            
            # 선 그래프 생성
            ax.plot(time_indices, metric_values, marker='o', linewidth=2, markersize=6,
                   color=sns.color_palette(self.visualization_config['color_palette'])[0])
            
            # 그래프 꾸미기
            ax.set_title(title, fontsize=self.visualization_config['font_size'] + 2, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=self.visualization_config['font_size'])
            ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=self.visualization_config['font_size'])
            ax.grid(True, alpha=0.3)
            
            # 데이터 포인트에 값 표시
            for i, value in enumerate(metric_values):
                ax.annotate(f'{value:.3f}', (i, value), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
            
            # y축 범위 설정
            if metric_values:
                min_val, max_val = min(metric_values), max(metric_values)
                margin = (max_val - min_val) * 0.1
                ax.set_ylim(max(0, min_val - margin), max_val + margin)
            
            # 레이아웃 조정
            plt.tight_layout()
            
            # 저장 경로 설정
            if save_path is None:
                save_path = self.output_dir / f"quality_trends_{metric_name}_{title.lower().replace(' ', '_')}.{self.visualization_config['save_format']}"
            else:
                save_path = Path(save_path)
            
            # 그래프 저장
            plt.savefig(save_path, dpi=self.visualization_config['dpi'], bbox_inches='tight')
            
            if self.visualization_config['show_plots']:
                plt.show()
            
            plt.close()
            
            logger.info(f"품질 트렌드 시각화 저장 완료: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"품질 트렌드 시각화 중 오류 발생: {e}")
            raise
    
    def create_quality_heatmap(self, metrics_matrix: np.ndarray, 
                              row_labels: List[str], 
                              col_labels: List[str],
                              title: str = "Quality Metrics Heatmap",
                              save_path: Optional[str] = None) -> str:
        """
        품질 메트릭을 히트맵으로 시각화
        
        Args:
            metrics_matrix: 메트릭 행렬 (n_samples x n_metrics)
            row_labels: 행 레이블 (샘플 이름)
            col_labels: 열 레이블 (메트릭 이름)
            title: 히트맵 제목
            save_path: 저장 경로
            
        Returns:
            저장된 파일 경로
        """
        try:
            # 그래프 생성
            fig, ax = plt.subplots(figsize=self.visualization_config['figure_size'])
            
            # 히트맵 생성
            sns.heatmap(metrics_matrix, 
                       annot=True, 
                       fmt='.3f',
                       cmap=self.visualization_config['color_palette'],
                       xticklabels=col_labels,
                       yticklabels=row_labels,
                       ax=ax,
                       cbar_kws={'label': 'Score'})
            
            # 그래프 꾸미기
            ax.set_title(title, fontsize=self.visualization_config['font_size'] + 2, fontweight='bold')
            ax.set_xlabel('Metrics', fontsize=self.visualization_config['font_size'])
            ax.set_ylabel('Samples', fontsize=self.visualization_config['font_size'])
            
            # 레이아웃 조정
            plt.tight_layout()
            
            # 저장 경로 설정
            if save_path is None:
                save_path = self.output_dir / f"quality_heatmap_{title.lower().replace(' ', '_')}.{self.visualization_config['save_format']}"
            else:
                save_path = Path(save_path)
            
            # 그래프 저장
            plt.savefig(save_path, dpi=self.visualization_config['dpi'], bbox_inches='tight')
            
            if self.visualization_config['show_plots']:
                plt.show()
            
            plt.close()
            
            logger.info(f"품질 히트맵 시각화 저장 완료: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"품질 히트맵 시각화 중 오류 발생: {e}")
            raise
    
    def plot_quality_distribution(self, metrics_data: Dict[str, List[float]], 
                                 title: str = "Quality Metrics Distribution",
                                 save_path: Optional[str] = None) -> str:
        """
        품질 메트릭의 분포를 박스플롯과 히스토그램으로 시각화
        
        Args:
            metrics_data: 메트릭별 데이터 딕셔너리
            title: 그래프 제목
            save_path: 저장 경로
            
        Returns:
            저장된 파일 경로
        """
        try:
            # 서브플롯 생성
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 박스플롯
            data_for_boxplot = [metrics_data[name] for name in metrics_data.keys()]
            ax1.boxplot(data_for_boxplot, labels=list(metrics_data.keys()))
            ax1.set_title('Quality Metrics Boxplot', fontsize=self.visualization_config['font_size'], fontweight='bold')
            ax1.set_ylabel('Score', fontsize=self.visualization_config['font_size'])
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # 히스토그램
            for metric_name, values in metrics_data.items():
                ax2.hist(values, alpha=0.7, label=metric_name, bins=20)
            
            ax2.set_title('Quality Metrics Histogram', fontsize=self.visualization_config['font_size'], fontweight='bold')
            ax2.set_xlabel('Score', fontsize=self.visualization_config['font_size'])
            ax2.set_ylabel('Frequency', fontsize=self.visualization_config['font_size'])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 전체 제목
            fig.suptitle(title, fontsize=self.visualization_config['font_size'] + 2, fontweight='bold')
            
            # 레이아웃 조정
            plt.tight_layout()
            
            # 저장 경로 설정
            if save_path is None:
                save_path = self.output_dir / f"quality_distribution_{title.lower().replace(' ', '_')}.{self.visualization_config['save_format']}"
            else:
                save_path = Path(save_path)
            
            # 그래프 저장
            plt.savefig(save_path, dpi=self.visualization_config['dpi'], bbox_inches='tight')
            
            if self.visualization_config['show_plots']:
                plt.show()
            
            plt.close()
            
            logger.info(f"품질 분포 시각화 저장 완료: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"품질 분포 시각화 중 오류 발생: {e}")
            raise
    
    def create_comprehensive_report(self, all_metrics: List[Dict[str, float]], 
                                  sample_names: Optional[List[str]] = None,
                                  output_filename: str = "comprehensive_quality_report") -> str:
        """
        종합적인 품질 보고서 생성
        
        Args:
            all_metrics: 모든 샘플의 메트릭 리스트
            sample_names: 샘플 이름 리스트
            output_filename: 출력 파일명
            
        Returns:
            저장된 파일 경로
        """
        try:
            if sample_names is None:
                sample_names = [f"Sample_{i+1}" for i in range(len(all_metrics))]
            
            # 메트릭 이름 추출
            metric_names = list(all_metrics[0].keys()) if all_metrics else []
            
            # 메트릭 행렬 생성
            metrics_matrix = np.array([[metrics[name] for name in metric_names] for metrics in all_metrics])
            
            # 서브플롯 생성
            fig = plt.figure(figsize=(20, 15))
            
            # 1. 품질 메트릭 히트맵
            ax1 = plt.subplot(2, 3, 1)
            sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='viridis',
                       xticklabels=metric_names, yticklabels=sample_names, ax=ax1)
            ax1.set_title('Quality Metrics Heatmap', fontweight='bold')
            
            # 2. 메트릭별 박스플롯
            ax2 = plt.subplot(2, 3, 2)
            data_for_boxplot = [metrics_matrix[:, i] for i in range(len(metric_names))]
            ax2.boxplot(data_for_boxplot, labels=metric_names)
            ax2.set_title('Metrics Distribution', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. 전체 점수 트렌드
            ax3 = plt.subplot(2, 3, 3)
            overall_scores = [metrics.get('overall_score', 0.0) for metrics in all_metrics]
            ax3.plot(range(len(overall_scores)), overall_scores, marker='o', linewidth=2)
            ax3.set_title('Overall Quality Trend', fontweight='bold')
            ax3.set_xlabel('Sample Index')
            ax3.set_ylabel('Overall Score')
            ax3.grid(True, alpha=0.3)
            
            # 4. 메트릭별 평균 비교
            ax4 = plt.subplot(2, 3, 4)
            metric_means = [np.mean(metrics_matrix[:, i]) for i in range(len(metric_names))]
            bars = ax4.bar(metric_names, metric_means, color='skyblue')
            ax4.set_title('Average Metrics', fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            
            # 막대 위에 값 표시
            for bar, value in zip(bars, metric_means):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # 5. 품질 등급 분포
            ax5 = plt.subplot(2, 3, 5)
            quality_grades = []
            for score in overall_scores:
                if score >= 0.8:
                    quality_grades.append('Excellent')
                elif score >= 0.6:
                    quality_grades.append('Good')
                elif score >= 0.4:
                    quality_grades.append('Fair')
                else:
                    quality_grades.append('Poor')
            
            grade_counts = {}
            for grade in quality_grades:
                grade_counts[grade] = grade_counts.get(grade, 0) + 1
            
            if grade_counts:
                ax5.pie(grade_counts.values(), labels=grade_counts.keys(), autopct='%1.1f%%')
                ax5.set_title('Quality Grade Distribution', fontweight='bold')
            
            # 6. 통계 요약
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            
            # 통계 정보 텍스트
            stats_text = f"""
            Total Samples: {len(all_metrics)}
            
            Average Scores:
            """
            for i, name in enumerate(metric_names):
                mean_val = np.mean(metrics_matrix[:, i])
                std_val = np.std(metrics_matrix[:, i])
                stats_text += f"{name}: {mean_val:.3f} ± {std_val:.3f}\n"
            
            stats_text += f"\nOverall Quality: {np.mean(overall_scores):.3f} ± {np.std(overall_scores):.3f}"
            
            ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
            
            # 전체 제목
            fig.suptitle('Comprehensive Quality Assessment Report', fontsize=16, fontweight='bold')
            
            # 레이아웃 조정
            plt.tight_layout()
            
            # 저장
            save_path = self.output_dir / f"{output_filename}.{self.visualization_config['save_format']}"
            plt.savefig(save_path, dpi=self.visualization_config['dpi'], bbox_inches='tight')
            
            if self.visualization_config['show_plots']:
                plt.show()
            
            plt.close()
            
            logger.info(f"종합 품질 보고서 생성 완료: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"종합 품질 보고서 생성 중 오류 발생: {e}")
            raise
    
    def set_visualization_config(self, **kwargs):
        """시각화 설정 업데이트"""
        self.visualization_config.update(kwargs)
        logger.info("시각화 설정 업데이트 완료")
    
    def get_output_directory(self) -> Path:
        """출력 디렉토리 반환"""
        return self.output_dir
