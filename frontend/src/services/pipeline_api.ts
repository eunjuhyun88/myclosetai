/**
 * MyCloset AI 프론트엔드 API 서비스
 * 백엔드와 통신하는 TypeScript 클라이언트
 */

// 타입 정의
export interface UserMeasurements {
  height: number;
  weight: number;
  chest?: number;
  waist?: number;
  hip?: number;
}

export interface ClothingAnalysis {
  category: string;
  style: string;
  dominant_color: number[];
  material?: string;
  confidence?: number;
}

export interface QualityMetrics {
  ssim: number;
  lpips: number;
  fid?: number;
  fit_overall: number;
  fit_coverage?: number;
  fit_shape_consistency?: number;
  color_preservation?: number;
  boundary_naturalness?: number;
}

export interface VirtualTryOnRequest {
  person_image: File;
  clothing_image: File;
  height: number;
  weight: number;
  quality_mode?: 'fast' | 'balanced' | 'quality';
}

export interface VirtualTryOnResponse {
  success: boolean;
  fitted_image?: string; // base64
  processing_time: number;
  confidence: number;
  measurements: Record<string, number>;
  clothing_analysis: ClothingAnalysis;
  fit_score: number;
  recommendations: string[];
  quality_metrics?: QualityMetrics;
  memory_usage?: Record<string, number>;
  step_times?: Record<string, number>;
  error?: string;
}

export interface PipelineProgress {
  step_id: number;
  progress: number;
  message: string;
  timestamp: number;
}

export interface PipelineStatus {
  status: string;
  device: string;
  memory_usage: Record<string, number>;
  models_loaded: string[];
  active_connections: number;
}

export interface SystemStats {
  total_requests: number;
  successful_requests: number;
  average_processing_time: number;
  average_quality_score: number;
  peak_memory_usage: number;
  uptime: number;
}

// API 클라이언트 클래스
class PipelineAPIClient {
  private baseURL: string;
  private wsURL: string;
  private currentWS: WebSocket | null = null;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.wsURL = baseURL.replace('http', 'ws');
  }

  /**
   * 가상 피팅 처리 요청
   */
  async processVirtualTryOn(
    request: VirtualTryOnRequest,
    onProgress?: (progress: PipelineProgress) => void
  ): Promise<VirtualTryOnResponse> {
    try {
      // WebSocket 연결 설정 (진행률 업데이트용)
      if (onProgress) {
        await this.setupProgressWebSocket(onProgress);
      }

      // FormData 준비
      const formData = new FormData();
      formData.append('person_image', request.person_image);
      formData.append('clothing_image', request.clothing_image);
      formData.append('height', request.height.toString());
      formData.append('weight', request.weight.toString());
      
      if (request.quality_mode) {
        formData.append('quality_mode', request.quality_mode);
      }

      console.log('🚀 가상 피팅 API 요청 시작...');

      // API 요청
      const response = await fetch(`${this.baseURL}/api/virtual-tryon`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP ${response.status}`);
      }

      const result: VirtualTryOnResponse = await response.json();
      
      // WebSocket 연결 정리
      this.closeProgressWebSocket();
      
      console.log('✅ 가상 피팅 API 응답 성공:', result);
      return result;

    } catch (error) {
      this.closeProgressWebSocket();
      console.error('❌ 가상 피팅 API 오류:', error);
      throw new Error(`가상 피팅 처리 실패: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * 진행률 WebSocket 설정
   */
  private async setupProgressWebSocket(
    onProgress: (progress: PipelineProgress) => void
  ): Promise<void> {
    try {
      const ws = new WebSocket(`${this.wsURL}/api/ws/pipeline-progress`);
      
      return new Promise((resolve, reject) => {
        ws.onopen = () => {
          console.log('🔗 WebSocket 연결 성공');
          this.currentWS = ws;
          resolve();
        };

        ws.onmessage = (event) => {
          try {
            const progress: PipelineProgress = JSON.parse(event.data);
            console.log('📊 진행률 업데이트:', progress);
            onProgress(progress);
          } catch (error) {
            console.error('WebSocket 메시지 파싱 오류:', error);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket 오류:', error);
          reject(error);
        };

        ws.onclose = () => {
          console.log('🔌 WebSocket 연결 종료');
          this.currentWS = null;
        };

        // 연결 타임아웃
        setTimeout(() => {
          if (ws.readyState !== WebSocket.OPEN) {
            reject(new Error('WebSocket 연결 타임아웃'));
          }
        }, 5000);
      });

    } catch (error) {
      console.error('WebSocket 설정 실패:', error);
    }
  }

  /**
   * WebSocket 연결 종료
   */
  private closeProgressWebSocket(): void {
    if (this.currentWS) {
      this.currentWS.close();
      this.currentWS = null;
    }
  }

  /**
   * 파이프라인 상태 조회
   */
  async getPipelineStatus(): Promise<PipelineStatus> {
    const response = await fetch(`${this.baseURL}/api/pipeline/status`);
    
    if (!response.ok) {
      throw new Error(`상태 조회 실패: ${response.status}`);
    }
    
    return await response.json();
  }

  /**
   * 파이프라인 워밍업
   */
  async warmupPipeline(qualityMode: string = 'balanced'): Promise<void> {
    const formData = new FormData();
    formData.append('quality_mode', qualityMode);

    const response = await fetch(`${this.baseURL}/api/pipeline/warmup`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || '워밍업 실패');
    }
  }

  /**
   * 시스템 통계 조회
   */
  async getSystemStats(): Promise<SystemStats> {
    const response = await fetch(`${this.baseURL}/stats`);
    
    if (!response.ok) {
      throw new Error(`통계 조회 실패: ${response.status}`);
    }
    
    return await response.json();
  }

  /**
   * 헬스 체크
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * 테스트용 더미 처리
   */
  async testDummyProcess(
    onProgress?: (progress: PipelineProgress) => void
  ): Promise<VirtualTryOnResponse> {
    try {
      if (onProgress) {
        await this.setupProgressWebSocket(onProgress);
      }
      
      const response = await fetch(`${this.baseURL}/test`, {
        method: 'GET',
      });

      if (!response.ok) {
        throw new Error(`테스트 실패: ${response.status}`);
      }

      // 더미 응답 생성
      const result: VirtualTryOnResponse = {
        success: true,
        fitted_image: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
        processing_time: 2.5,
        confidence: 0.85,
        measurements: {
          chest: 95,
          waist: 80,
          hip: 98,
          bmi: 23.4
        },
        clothing_analysis: {
          category: "shirt",
          style: "casual",
          dominant_color: [120, 150, 180]
        },
        fit_score: 0.88,
        recommendations: [
          "🧪 테스트 모드로 처리되었습니다",
          "실제 AI 파이프라인에서 더 정확한 결과를 확인하세요",
          "이 스타일이 잘 어울립니다!"
        ]
      };
      
      this.closeProgressWebSocket();
      return result;

    } catch (error) {
      this.closeProgressWebSocket();
      throw error;
    }
  }

  /**
   * 품질 메트릭 정보 조회
   */
  async getQualityMetricsInfo(): Promise<any> {
    const response = await fetch(`${this.baseURL}/api/quality/metrics`);
    
    if (!response.ok) {
      throw new Error(`메트릭 정보 조회 실패: ${response.status}`);
    }
    
    return await response.json();
  }

  /**
   * 예시 이미지 목록 조회
   */
  async getExampleImages(): Promise<any> {
    const response = await fetch(`${this.baseURL}/api/examples`);
    
    if (!response.ok) {
      throw new Error(`예시 이미지 조회 실패: ${response.status}`);
    }
    
    return await response.json();
  }

  /**
   * 피드백 제출
   */
  async submitFeedback(feedback: {
    rating: number;
    comment: string;
    result_id?: string;
  }): Promise<void> {
    const response = await fetch(`${this.baseURL}/api/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(feedback),
    });

    if (!response.ok) {
      throw new Error(`피드백 제출 실패: ${response.status}`);
    }
  }
}

// 유틸리티 함수들
export class PipelineUtils {
  /**
   * 이미지 파일을 base64로 변환
   */
  static async fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        resolve(result.split(',')[1]); // "data:image/...;base64," 부분 제거
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  /**
   * base64를 이미지 URL로 변환
   */
  static base64ToImageURL(base64: string): string {
    return `data:image/png;base64,${base64}`;
  }

  /**
   * 파일 크기 검증
   */
  static validateFileSize(file: File, maxSizeMB: number = 10): boolean {
    const maxSizeBytes = maxSizeMB * 1024 * 1024;
    return file.size <= maxSizeBytes;
  }

  /**
   * 이미지 파일 타입 검증
   */
  static validateImageType(file: File): boolean {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    return allowedTypes.includes(file.type);
  }

  /**
   * 처리 시간을 사용자 친화적 형식으로 변환
   */
  static formatProcessingTime(seconds: number): string {
    if (seconds < 1) {
      return `${Math.round(seconds * 1000)}ms`;
    } else if (seconds < 60) {
      return `${seconds.toFixed(1)}초`;
    } else {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = Math.round(seconds % 60);
      return `${minutes}분 ${remainingSeconds}초`;
    }
  }

  /**
   * 품질 점수를 등급으로 변환
   */
  static getQualityGrade(score: number): {
    grade: string;
    color: string;
    description: string;
  } {
    if (score >= 0.9) {
      return {
        grade: 'Excellent',
        color: 'text-green-600',
        description: '완벽한 품질'
      };
    } else if (score >= 0.8) {
      return {
        grade: 'Good',
        color: 'text-blue-600',
        description: '우수한 품질'
      };
    } else if (score >= 0.6) {
      return {
        grade: 'Fair',
        color: 'text-yellow-600',
        description: '양호한 품질'
      };
    } else {
      return {
        grade: 'Poor',
        color: 'text-red-600',
        description: '개선 필요'
      };
    }
  }

  /**
   * 메모리 사용량을 사용자 친화적 형식으로 변환
   */
  static formatMemoryUsage(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }

  /**
   * 에러 메시지를 사용자 친화적으로 변환
   */
  static getUserFriendlyError(error: string): string {
    const errorMappings: Record<string, string> = {
      'Network Error': '네트워크 연결을 확인해주세요.',
      'timeout': '처리 시간이 초과되었습니다. 다시 시도해주세요.',
      'invalid image': '지원되지 않는 이미지 형식입니다.',
      'file too large': '파일 크기가 너무 큽니다. 10MB 이하로 업로드해주세요.',
      'server error': '서버에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요.',
    };

    const lowerError = error.toLowerCase();
    for (const [key, message] of Object.entries(errorMappings)) {
      if (lowerError.includes(key)) {
        return message;
      }
    }

    return '알 수 없는 오류가 발생했습니다. 지원팀에 문의해주세요.';
  }
}

// 기본 export
export default PipelineAPIClient;