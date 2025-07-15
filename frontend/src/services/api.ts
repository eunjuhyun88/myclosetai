/**
 * MyCloset AI API 서비스 메인 진입점
 * 기존 코드와의 호환성을 위한 re-export 파일
 * - 기존 import 구문 유지
 * - 새로운 PipelineAPIClient와 통합
 * - 하위 호환성 보장
 */

// 새로운 PipelineAPIClient를 기본으로 사용
import PipelineAPIClient from './PipelineAPIClient';

// 타입들을 re-export
export type {
  VirtualTryOnRequest,
  VirtualTryOnResponse,
  PipelineProgress,
  PipelineStatus,
  SystemStats,
  SystemHealth,
  TaskInfo,
  ProcessingStatus,
  BrandSizeData,
  SizeRecommendation,
  UsePipelineOptions,
  QualityLevel,
  DeviceType,
  ClothingCategory,
  FabricType,
  StylePreference,
} from '../types/pipeline';

// 유틸리티 re-export
export { PipelineUtils } from '../utils/pipelineUtils';

// =================================================================
// 🔧 기본 API 클라이언트 인스턴스 (싱글톤)
// =================================================================

// 기본 설정으로 초기화된 API 클라이언트
let _apiClientInstance: PipelineAPIClient | null = null;

export function getApiClient(): PipelineAPIClient {
  if (!_apiClientInstance) {
    _apiClientInstance = new PipelineAPIClient({
      baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
      enableCaching: true,
      enableRetry: true,
      maxRetryAttempts: 3,
      timeout: 30000,
      enableDebugMode: process.env.NODE_ENV === 'development',
    });
  }
  return _apiClientInstance;
}

// 기존 코드 호환성을 위한 apiClient export
export const apiClient = getApiClient();

// =================================================================
// 🔧 하위 호환성을 위한 개별 함수들
// =================================================================

/**
 * @deprecated 직접 apiClient.processVirtualTryOn() 사용을 권장
 */
export async function processVirtualTryOn(
  request: any,
  onProgress?: (progress: any) => void
): Promise<any> {
  console.warn('⚠️ processVirtualTryOn 함수는 deprecated입니다. apiClient.processVirtualTryOn()을 사용하세요.');
  return await apiClient.processVirtualTryOn(request, onProgress);
}

/**
 * @deprecated 직접 apiClient.healthCheck() 사용을 권장
 */
export async function healthCheck(): Promise<boolean> {
  console.warn('⚠️ healthCheck 함수는 deprecated입니다. apiClient.healthCheck()을 사용하세요.');
  return await apiClient.healthCheck();
}

/**
 * @deprecated 직접 apiClient.getPipelineStatus() 사용을 권장
 */
export async function getPipelineStatus(): Promise<any> {
  console.warn('⚠️ getPipelineStatus 함수는 deprecated입니다. apiClient.getPipelineStatus()을 사용하세요.');
  return await apiClient.getPipelineStatus();
}

/**
 * @deprecated 직접 apiClient.getSystemStats() 사용을 권장
 */
export async function getSystemStats(): Promise<any> {
  console.warn('⚠️ getSystemStats 함수는 deprecated입니다. apiClient.getSystemStats()을 사용하세요.');
  return await apiClient.getSystemStats();
}

/**
 * @deprecated 직접 apiClient.warmupPipeline() 사용을 권장
 */
export async function warmupPipeline(qualityMode: string = 'balanced'): Promise<void> {
  console.warn('⚠️ warmupPipeline 함수는 deprecated입니다. apiClient.warmupPipeline()을 사용하세요.');
  await apiClient.warmupPipeline(qualityMode as any);
}

// =================================================================
// 🔧 기존 pipeline_api.ts 내용과의 호환성
// =================================================================

// 기존 PipelineAPIClient 클래스 재구성 (하위 호환)
export class LegacyPipelineAPIClient {
  private client: PipelineAPIClient;

  constructor(baseURL: string = 'http://localhost:8000') {
    console.warn('⚠️ LegacyPipelineAPIClient는 deprecated입니다. 새로운 PipelineAPIClient를 사용하세요.');
    this.client = new PipelineAPIClient({ baseURL });
  }

  async processVirtualTryOn(request: any, onProgress?: (progress: any) => void): Promise<any> {
    return await this.client.processVirtualTryOn(request, onProgress);
  }

  async getPipelineStatus(): Promise<any> {
    return await this.client.getPipelineStatus();
  }

  async warmupPipeline(qualityMode: string = 'balanced'): Promise<void> {
    await this.client.warmupPipeline(qualityMode as any);
  }

  async getSystemStats(): Promise<any> {
    return await this.client.getSystemStats();
  }

  async healthCheck(): Promise<boolean> {
    return await this.client.healthCheck();
  }

  // 기존 더미 프로세스 (하위 호환)
  async testDummyProcess(
    onProgress?: (progress: any) => void,
    duration: number = 5000
  ): Promise<any> {
    console.warn('⚠️ testDummyProcess는 더 이상 지원되지 않습니다.');
    
    if (onProgress) {
      for (let i = 0; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, duration / 10));
        onProgress({
          step_id: Math.floor(i / 12.5) + 1,
          progress: i,
          message: `더미 처리 중... ${i}%`,
          timestamp: Date.now()
        });
      }
    }

    return {
      success: true,
      fitted_image: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkR1bW15IEltYWdlPC90ZXh0Pjwvc3ZnPg==',
      processing_time: duration / 1000,
      confidence: 0.95,
      measurements: { chest: 95, waist: 80, hip: 90, bmi: 22.5 },
      clothing_analysis: { category: 'shirt', style: 'casual', dominant_color: [255, 255, 255] },
      fit_score: 0.88,
      recommendations: ['좋은 핏입니다!', '색상이 잘 어울립니다.'],
      quality_metrics: { ssim: 0.85, lpips: 0.15, fid: 25.5, fit_overall: 0.88 }
    };
  }

  async submitFeedback(feedback: any): Promise<any> {
    console.warn('⚠️ submitFeedback는 더 이상 지원되지 않습니다.');
    console.log('피드백 제출됨:', feedback);
    return { success: true, message: '피드백이 제출되었습니다.' };
  }

  private getUserFriendlyError(error: string): string {
    const errorMappings: Record<string, string> = {
      'connection failed': '서버에 연결할 수 없습니다. 네트워크를 확인해주세요.',
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

// =================================================================
// 🔧 React Hook을 위한 래퍼 (하위 호환)
// =================================================================

export const usePipelineAPI = () => {
  console.warn('⚠️ usePipelineAPI 훅은 deprecated입니다. usePipeline 훅을 사용하세요.');
  
  const client = getApiClient();

  return {
    processVirtualTryOn: client.processVirtualTryOn.bind(client),
    getPipelineStatus: client.getPipelineStatus.bind(client),
    warmupPipeline: client.warmupPipeline.bind(client),
    getSystemStats: client.getSystemStats.bind(client),
    healthCheck: client.healthCheck.bind(client),
    // 더미 함수들 (하위 호환)
    testDummyProcess: async (onProgress?: any) => {
      const legacyClient = new LegacyPipelineAPIClient();
      return await legacyClient.testDummyProcess(onProgress);
    },
    submitFeedback: async (feedback: any) => {
      const legacyClient = new LegacyPipelineAPIClient();
      return await legacyClient.submitFeedback(feedback);
    },
  };
};

// =================================================================
// 🔧 메인 export들
// =================================================================

// 새로운 클라이언트를 기본으로 export
export { PipelineAPIClient };
export default PipelineAPIClient;

// 하위 호환성을 위한 추가 export들
export { LegacyPipelineAPIClient };

// 환경 설정 헬퍼
export const config = {
  API_BASE_URL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
  WS_BASE_URL: process.env.REACT_APP_WS_BASE_URL || 'ws://localhost:8000',
  ENABLE_DEBUG: process.env.NODE_ENV === 'development',
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  SUPPORTED_IMAGE_TYPES: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'],
  DEFAULT_TIMEOUT: 30000,
  MAX_RETRY_ATTEMPTS: 3,
};

// 유틸리티 함수들
export const utils = {
  validateImageFile: (file: File): boolean => {
    return config.SUPPORTED_IMAGE_TYPES.includes(file.type) && 
           file.size <= config.MAX_FILE_SIZE;
  },
  
  formatFileSize: (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },
  
  generateSessionId: (): string => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  },
  
  isValidURL: (url: string): boolean => {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }
};

// 환경 검증
export const validateEnvironment = (): {
  valid: boolean;
  errors: string[];
  warnings: string[];
} => {
  const errors: string[] = [];
  const warnings: string[] = [];

  // API URL 검증
  if (!utils.isValidURL(config.API_BASE_URL)) {
    errors.push('Invalid API_BASE_URL in environment variables');
  }

  // 브라우저 기능 검증
  if (typeof window !== 'undefined') {
    if (!window.fetch) {
      errors.push('Fetch API not supported');
    }
    if (!window.WebSocket) {
      warnings.push('WebSocket not supported - real-time features disabled');
    }
    if (!window.File) {
      errors.push('File API not supported');
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
};

// 초기화 함수
export const initializeAPI = async (): Promise<boolean> => {
  try {
    console.log('🚀 MyCloset AI API 초기화 중...');
    
    // 환경 검증
    const envCheck = validateEnvironment();
    if (!envCheck.valid) {
      console.error('❌ 환경 검증 실패:', envCheck.errors);
      return false;
    }
    
    if (envCheck.warnings.length > 0) {
      console.warn('⚠️ 환경 경고:', envCheck.warnings);
    }

    // API 클라이언트 초기화
    const client = getApiClient();
    const initialized = await client.initialize();
    
    if (initialized) {
      console.log('✅ MyCloset AI API 초기화 완료');
      return true;
    } else {
      console.error('❌ API 클라이언트 초기화 실패');
      return false;
    }
  } catch (error) {
    console.error('❌ API 초기화 중 오류:', error);
    return false;
  }
};