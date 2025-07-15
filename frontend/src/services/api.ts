/**
 * 실제 백엔드에 맞춘 API 클라이언트
 * routes.py와 virtual_tryon.py 구조에 완전 호환
 */

// 실제 백엔드 응답 타입 (schemas.py 기반)
export interface VirtualTryOnRequest {
  person_image: File;
  clothing_image: File;
  height: number;
  weight: number;
  chest?: number;
  waist?: number;
  hips?: number;
  clothing_type?: string;
  fabric_type?: string;
  quality_level?: 'fast' | 'balanced' | 'high' | 'ultra';
  style_preferences?: string;
  save_intermediate?: boolean;
  async_processing?: boolean;
}

export interface VirtualTryOnResponse {
  success: boolean;
  task_id?: string;
  message?: string;
  processing_time: number;
  async_processing?: boolean;
  
  // 동기 처리 결과
  result_image_base64?: string;
  result_image_url?: string;
  quality_score?: number;
  fit_score?: number;
  confidence?: number;
  
  // 상세 정보
  steps_completed?: number;
  processing_details?: any;
  recommendations?: any;
  intermediate_results?: any;
}

export interface ProcessingStatus {
  task_id: string;
  status: 'processing' | 'completed' | 'failed';
  progress: number;
  current_step: string;
  elapsed_time: number;
  result?: VirtualTryOnResponse;
  error?: string;
}

export interface SystemStatus {
  pipeline_ready: boolean;
  pipeline_status: any;
  active_tasks: number;
  system_health: 'healthy' | 'degraded' | 'unhealthy';
}

/**
 * 실제 백엔드 API 클라이언트
 */
export default class RealBackendAPIClient {
  private baseURL: string;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.baseURL = baseURL.replace(/\/$/, '');
  }

  /**
   * 🎯 메인 가상 피팅 API (실제 백엔드 routes.py)
   */
  async processVirtualTryOn(request: VirtualTryOnRequest): Promise<VirtualTryOnResponse> {
    const formData = new FormData();
    
    // 필수 필드
    formData.append('person_image', request.person_image);
    formData.append('clothing_image', request.clothing_image);
    formData.append('height', request.height.toString());
    formData.append('weight', request.weight.toString());
    
    // 선택적 필드들
    if (request.chest) formData.append('chest', request.chest.toString());
    if (request.waist) formData.append('waist', request.waist.toString());
    if (request.hips) formData.append('hips', request.hips.toString());
    if (request.clothing_type) formData.append('clothing_type', request.clothing_type);
    if (request.fabric_type) formData.append('fabric_type', request.fabric_type);
    if (request.quality_level) formData.append('quality_level', request.quality_level);
    if (request.style_preferences) formData.append('style_preferences', request.style_preferences);
    if (request.save_intermediate !== undefined) formData.append('save_intermediate', request.save_intermediate.toString());
    if (request.async_processing !== undefined) formData.append('async_processing', request.async_processing.toString());

    // 실제 백엔드 API 호출
    const response = await fetch(`${this.baseURL}/api/virtual-tryon`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    return await response.json();
  }

  /**
   * 📊 태스크 상태 조회 (실제 백엔드 routes.py)
   */
  async getTaskStatus(taskId: string): Promise<ProcessingStatus> {
    const response = await fetch(`${this.baseURL}/api/status/${taskId}`);
    
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error('존재하지 않는 태스크입니다.');
      }
      throw new Error(`태스크 상태 조회 실패: ${response.status}`);
    }
    
    return await response.json();
  }

  /**
   * 🚀 빠른 가상 피팅 (실제 백엔드 routes.py)
   */
  async quickVirtualFitting(
    personImage: File,
    clothingImage: File,
    height: number = 170,
    weight: number = 65
  ): Promise<any> {
    const formData = new FormData();
    formData.append('person_image', personImage);
    formData.append('clothing_image', clothingImage);
    formData.append('height', height.toString());
    formData.append('weight', weight.toString());

    const response = await fetch(`${this.baseURL}/api/quick-fit`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || '빠른 피팅 실패');
    }

    return await response.json();
  }

  /**
   * 🔍 파이프라인 상태 조회 (실제 백엔드 routes.py)
   */
  async getPipelineStatus(): Promise<SystemStatus> {
    const response = await fetch(`${this.baseURL}/api/pipeline/status`);
    
    if (!response.ok) {
      throw new Error(`파이프라인 상태 조회 실패: ${response.status}`);
    }
    
    return await response.json();
  }

  /**
   * 🧪 인체 파싱만 테스트 (실제 백엔드 routes.py)
   */
  async parseHumanOnly(personImage: File): Promise<any> {
    const formData = new FormData();
    formData.append('person_image', personImage);

    const response = await fetch(`${this.baseURL}/api/parse-human`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || '인체 파싱 실패');
    }

    return await response.json();
  }

  /**
   * 🏥 헬스체크 (기본 FastAPI)
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
   * 🔄 태스크 취소 (실제 백엔드 routes.py)
   */
  async cancelTask(taskId: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/api/tasks/${taskId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || '태스크 취소 실패');
    }

    return await response.json();
  }

  /**
   * 📋 활성 태스크 목록 (실제 백엔드 routes.py)
   */
  async listActiveTasks(): Promise<any> {
    const response = await fetch(`${this.baseURL}/api/tasks`);
    
    if (!response.ok) {
      throw new Error(`활성 태스크 조회 실패: ${response.status}`);
    }
    
    return await response.json();
  }

  /**
   * 🔄 태스크 상태 폴링 (비동기 처리용)
   */
  async pollTaskStatus(
    taskId: string, 
    onProgress?: (status: ProcessingStatus) => void,
    pollInterval: number = 1000,
    maxWaitTime: number = 300000 // 5분
  ): Promise<VirtualTryOnResponse> {
    const startTime = Date.now();

    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          // 시간 초과 체크
          if (Date.now() - startTime > maxWaitTime) {
            reject(new Error('처리 시간이 초과되었습니다.'));
            return;
          }

          const status = await this.getTaskStatus(taskId);
          
          // 진행률 콜백 호출
          if (onProgress) {
            onProgress(status);
          }
          
          // 완료 체크
          if (status.status === 'completed' && status.result) {
            resolve(status.result);
            return;
          }
          
          // 실패 체크
          if (status.status === 'failed') {
            reject(new Error(status.error || '처리 중 오류가 발생했습니다.'));
            return;
          }
          
          // 계속 폴링
          if (status.status === 'processing') {
            setTimeout(poll, pollInterval);
          }
          
        } catch (error) {
          reject(error);
        }
      };
      
      poll();
    });
  }

  /**
   * 🔄 통합 처리 함수 (요청 + 폴링)
   */
  async processAndWait(
    request: VirtualTryOnRequest,
    onProgress?: (status: ProcessingStatus) => void
  ): Promise<VirtualTryOnResponse> {
    console.log('🚀 실제 백엔드 가상 피팅 처리 시작');
    
    // 비동기 처리 강제 설정
    const asyncRequest = { ...request, async_processing: true };
    
    // 1. 처리 요청
    const initialResponse = await this.processVirtualTryOn(asyncRequest);
    
    if (!initialResponse.task_id) {
      // 동기 처리된 경우 바로 반환
      return initialResponse;
    }
    
    console.log(`📋 태스크 생성: ${initialResponse.task_id}`);
    
    // 2. 상태 폴링
    return await this.pollTaskStatus(initialResponse.task_id, onProgress);
  }

  /**
   * 🌍 한국어 에러 메시지 변환
   */
  private translateError(error: string): string {
    const errorMappings: Record<string, string> = {
      'connection failed': '서버에 연결할 수 없습니다.',
      'pipeline not ready': 'AI 파이프라인이 준비되지 않았습니다.',
      'invalid image': '올바르지 않은 이미지입니다.',
      'file too large': '파일 크기가 너무 큽니다.',
      'processing failed': '처리 중 오류가 발생했습니다.',
      'task not found': '존재하지 않는 작업입니다.',
      'server error': '서버에 일시적인 문제가 발생했습니다.',
    };

    const lowerError = error.toLowerCase();
    for (const [key, message] of Object.entries(errorMappings)) {
      if (lowerError.includes(key)) {
        return message;
      }
    }

    return error;
  }
}

// React Hook
export const useRealBackendAPI = () => {
  const apiClient = new RealBackendAPIClient();

  return {
    processVirtualTryOn: apiClient.processVirtualTryOn.bind(apiClient),
    processAndWait: apiClient.processAndWait.bind(apiClient),
    getTaskStatus: apiClient.getTaskStatus.bind(apiClient),
    quickVirtualFitting: apiClient.quickVirtualFitting.bind(apiClient),
    getPipelineStatus: apiClient.getPipelineStatus.bind(apiClient),
    parseHumanOnly: apiClient.parseHumanOnly.bind(apiClient),
    healthCheck: apiClient.healthCheck.bind(apiClient),
    cancelTask: apiClient.cancelTask.bind(apiClient),
    listActiveTasks: apiClient.listActiveTasks.bind(apiClient),
  };
};

// 싱글톤 인스턴스
export const realBackendClient = new RealBackendAPIClient();