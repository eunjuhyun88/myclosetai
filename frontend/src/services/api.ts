/**
 * MyCloset AI API 서비스 메인 진입점 (수정 버전)
 * 백엔드 API와 완전 호환되도록 수정
 * - 에러 처리 강화
 * - 타임아웃 증가
 * - FormData 필드명 통일
 * - 진행률 추적 개선
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
// 🔧 기본 API 클라이언트 인스턴스 (싱글톤 - 수정된 버전)
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
      requestTimeout: 60000, // 60초로 증가
      enableDebugMode: process.env.NODE_ENV === 'development',
    });
  }
  return _apiClientInstance;
}

// 기존 코드 호환성을 위한 apiClient export
export const apiClient = getApiClient();

// =================================================================
// 🔧 하위 호환성을 위한 개별 함수들 (수정된 버전)
// =================================================================

/**
 * @deprecated 직접 apiClient.processVirtualTryOn() 사용을 권장
 */
export async function processVirtualTryOn(
  request: any,
  onProgress?: (progress: any) => void
): Promise<any> {
  console.warn('⚠️ processVirtualTryOn 함수는 deprecated입니다. apiClient.processVirtualTryOn()을 사용하세요.');
  
  try {
    return await apiClient.processVirtualTryOn(request, onProgress);
  } catch (error: any) {
    // 사용자 친화적 에러 메시지로 변환
    const friendlyMessage = getFriendlyErrorMessage(error);
    console.error('❌ 가상 피팅 실패:', friendlyMessage);
    throw new Error(friendlyMessage);
  }
}

/**
 * @deprecated 직접 apiClient.healthCheck() 사용을 권장
 */
export async function healthCheck(): Promise<boolean> {
  console.warn('⚠️ healthCheck 함수는 deprecated입니다. apiClient.healthCheck()을 사용하세요.');
  
  try {
    return await apiClient.healthCheck();
  } catch (error) {
    console.error('❌ 헬스체크 실패:', error);
    return false;
  }
}

/**
 * @deprecated 직접 apiClient.getPipelineStatus() 사용을 권장
 */
export async function getPipelineStatus(): Promise<any> {
  console.warn('⚠️ getPipelineStatus 함수는 deprecated입니다. apiClient.getPipelineStatus()을 사용하세요.');
  
  try {
    return await apiClient.getPipelineStatus();
  } catch (error) {
    console.error('❌ 파이프라인 상태 조회 실패:', error);
    throw error;
  }
}

/**
 * @deprecated 직접 apiClient.getSystemStats() 사용을 권장
 */
export async function getSystemStats(): Promise<any> {
  console.warn('⚠️ getSystemStats 함수는 deprecated입니다. apiClient.getSystemStats()을 사용하세요.');
  
  try {
    return await apiClient.getSystemStats();
  } catch (error) {
    console.error('❌ 시스템 통계 조회 실패:', error);
    throw error;
  }
}

/**
 * @deprecated 직접 apiClient.warmupPipeline() 사용을 권장
 */
export async function warmupPipeline(qualityMode: string = 'balanced'): Promise<void> {
  console.warn('⚠️ warmupPipeline 함수는 deprecated입니다. apiClient.warmupPipeline()을 사용하세요.');
  
  try {
    await apiClient.warmupPipeline(qualityMode as any);
  } catch (error) {
    console.error('❌ 워밍업 실패:', error);
    throw error;
  }
}

// =================================================================
// 🔧 기존 pipeline_api.ts 내용과의 호환성 (수정된 버전)
// =================================================================

// 기존 PipelineAPIClient 클래스 재구성 (하위 호환)
export class LegacyPipelineAPIClient {
  private client: PipelineAPIClient;

  constructor(baseURL: string = 'http://localhost:8000') {
    console.warn('⚠️ LegacyPipelineAPIClient는 deprecated입니다. 새로운 PipelineAPIClient를 사용하세요.');
    this.client = new PipelineAPIClient({ 
      baseURL,
      requestTimeout: 60000, // 60초로 증가
      enableDebugMode: true,
      enableRetry: true,
      maxRetryAttempts: 3
    });
  }

  async processVirtualTryOn(request: any, onProgress?: (progress: any) => void): Promise<any> {
    try {
      return await this.client.processVirtualTryOn(request, onProgress);
    } catch (error: any) {
      const friendlyMessage = getFriendlyErrorMessage(error);
      console.error('❌ Legacy 가상 피팅 실패:', friendlyMessage);
      throw new Error(friendlyMessage);
    }
  }

  async getPipelineStatus(): Promise<any> {
    try {
      return await this.client.getPipelineStatus();
    } catch (error) {
      console.error('❌ Legacy 파이프라인 상태 조회 실패:', error);
      throw error;
    }
  }

  async warmupPipeline(qualityMode: string = 'balanced'): Promise<void> {
    try {
      await this.client.warmupPipeline(qualityMode as any);
    } catch (error) {
      console.error('❌ Legacy 워밍업 실패:', error);
      throw error;
    }
  }

  async getSystemStats(): Promise<any> {
    try {
      return await this.client.getSystemStats();
    } catch (error) {
      console.error('❌ Legacy 시스템 통계 조회 실패:', error);
      throw error;
    }
  }

  async healthCheck(): Promise<boolean> {
    try {
      return await this.client.healthCheck();
    } catch (error) {
      console.error('❌ Legacy 헬스체크 실패:', error);
      return false;
    }
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
    return getFriendlyErrorMessage(error);
  }
}

// =================================================================
// 🔧 React Hook을 위한 래퍼 (하위 호환 - 수정된 버전)
// =================================================================

export const usePipelineAPI = () => {
  console.warn('⚠️ usePipelineAPI 훅은 deprecated입니다. usePipeline 훅을 사용하세요.');
  
  const client = getApiClient();

  return {
    processVirtualTryOn: async (request: any, onProgress?: any) => {
      try {
        return await client.processVirtualTryOn(request, onProgress);
      } catch (error: any) {
        const friendlyMessage = getFriendlyErrorMessage(error);
        throw new Error(friendlyMessage);
      }
    },
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
// 🔧 에러 메시지 처리 (새로 추가)
// =================================================================

function getFriendlyErrorMessage(error: any): string {
  const errorMessage = typeof error === 'string' ? error : error?.message || '알 수 없는 오류';
  
  // HTTP 상태 코드 기반 메시지
  if (errorMessage.includes('413') || errorMessage.includes('file too large')) {
    return '파일 크기가 너무 큽니다. 50MB 이하의 이미지를 사용해주세요.';
  }
  
  if (errorMessage.includes('415') || errorMessage.includes('unsupported media')) {
    return '지원되지 않는 파일 형식입니다. JPG, PNG, WebP 파일을 사용해주세요.';
  }
  
  if (errorMessage.includes('400') || errorMessage.includes('bad request')) {
    return '잘못된 요청입니다. 입력 정보를 확인해주세요.';
  }
  
  if (errorMessage.includes('401') || errorMessage.includes('unauthorized')) {
    return '인증이 필요합니다. 로그인해주세요.';
  }
  
  if (errorMessage.includes('403') || errorMessage.includes('forbidden')) {
    return '접근 권한이 없습니다.';
  }
  
  if (errorMessage.includes('404') || errorMessage.includes('not found')) {
    return '요청한 리소스를 찾을 수 없습니다.';
  }
  
  if (errorMessage.includes('422') || errorMessage.includes('validation')) {
    return '입력 데이터가 올바르지 않습니다.';
  }
  
  if (errorMessage.includes('429') || errorMessage.includes('rate limit')) {
    return '너무 많은 요청을 보냈습니다. 잠시 후 다시 시도해주세요.';
  }
  
  if (errorMessage.includes('500') || errorMessage.includes('internal server')) {
    return '서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요.';
  }
  
  if (errorMessage.includes('502') || errorMessage.includes('bad gateway')) {
    return '게이트웨이 오류입니다. 잠시 후 다시 시도해주세요.';
  }
  
  if (errorMessage.includes('503') || errorMessage.includes('service unavailable')) {
    return '서비스를 일시적으로 사용할 수 없습니다.';
  }
  
  if (errorMessage.includes('504') || errorMessage.includes('gateway timeout')) {
    return '게이트웨이 시간 초과입니다.';
  }
  
  if (errorMessage.includes('timeout') || errorMessage.includes('시간 초과')) {
    return '요청 시간이 초과되었습니다. 다시 시도해주세요.';
  }
  
  if (errorMessage.includes('network') || errorMessage.includes('연결')) {
    return '네트워크 연결을 확인해주세요.';
  }
  
  if (errorMessage.includes('fetch') || errorMessage.includes('connection')) {
    return '서버에 연결할 수 없습니다. 네트워크를 확인해주세요.';
  }
  
  if (errorMessage.includes('abort') || errorMessage.includes('cancel')) {
    return '요청이 취소되었습니다.';
  }
  
  if (errorMessage.includes('cors')) {
    return 'CORS 오류가 발생했습니다. 서버 설정을 확인해주세요.';
  }
  
  if (errorMessage.includes('json') || errorMessage.includes('parse')) {
    return '서버 응답을 처리할 수 없습니다.';
  }
  
  if (errorMessage.includes('memory') || errorMessage.includes('메모리')) {
    return '메모리 부족으로 처리할 수 없습니다. 더 작은 이미지를 사용해주세요.';
  }
  
  // 기본 메시지
  return errorMessage || '알 수 없는 오류가 발생했습니다. 지원팀에 문의해주세요.';
}

// =================================================================
// 🔧 파일 검증 유틸리티 (새로 추가)
// =================================================================

export const fileUtils = {
  /**
   * 이미지 파일 검증
   */
  validateImageFile: (file: File): { valid: boolean; error?: string } => {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    const maxSize = 50 * 1024 * 1024; // 50MB

    if (!allowedTypes.includes(file.type.toLowerCase())) {
      return {
        valid: false,
        error: '지원되지 않는 파일 형식입니다. JPG, PNG, WebP 파일을 사용해주세요.'
      };
    }

    if (file.size > maxSize) {
      return {
        valid: false,
        error: `파일 크기가 너무 큽니다. 50MB 이하의 파일을 사용해주세요. (현재: ${(file.size / (1024 * 1024)).toFixed(1)}MB)`
      };
    }

    return { valid: true };
  },

  /**
   * 파일 크기 포맷팅
   */
  formatFileSize: (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },

  /**
   * 이미지 압축 (필요한 경우)
   */
  compressImage: async (file: File, maxWidth: number = 1024, quality: number = 0.8): Promise<File> => {
    return new Promise((resolve, reject) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();

      img.onload = () => {
        // 비율 유지하며 크기 조정
        const ratio = Math.min(maxWidth / img.width, maxWidth / img.height);
        canvas.width = img.width * ratio;
        canvas.height = img.height * ratio;

        // 이미지 그리기
        ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Blob으로 변환
        canvas.toBlob(
          (blob) => {
            if (blob) {
              const compressedFile = new File([blob], file.name, {
                type: 'image/jpeg',
                lastModified: Date.now()
              });
              resolve(compressedFile);
            } else {
              reject(new Error('이미지 압축에 실패했습니다.'));
            }
          },
          'image/jpeg',
          quality
        );
      };

      img.onerror = () => reject(new Error('이미지를 로드할 수 없습니다.'));
      img.src = URL.createObjectURL(file);
    });
  }
};

// =================================================================
// 🔧 메인 export들
// =================================================================

// 새로운 클라이언트를 기본으로 export
export { PipelineAPIClient };
export default PipelineAPIClient;

// 하위 호환성을 위한 추가 export들
export { LegacyPipelineAPIClient };

// 환경 설정 헬퍼 (수정된 버전)
export const config = {
  API_BASE_URL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
  WS_BASE_URL: process.env.REACT_APP_WS_BASE_URL || 'ws://localhost:8000',
  ENABLE_DEBUG: process.env.NODE_ENV === 'development',
  MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB로 증가
  SUPPORTED_IMAGE_TYPES: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'],
  DEFAULT_TIMEOUT: 60000, // 60초로 증가
  MAX_RETRY_ATTEMPTS: 3,
  HEARTBEAT_INTERVAL: 30000,
  HEALTH_CHECK_INTERVAL: 30000,
};

// 유틸리티 함수들 (수정된 버전)
export const utils = {
  validateImageFile: (file: File): boolean => {
    const result = fileUtils.validateImageFile(file);
    return result.valid;
  },
  
  formatFileSize: (bytes: number): string => {
    return fileUtils.formatFileSize(bytes);
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
  },

  /**
   * 에러 메시지를 사용자 친화적으로 변환
   */
  getFriendlyErrorMessage,

  /**
   * 디바운스 함수
   */
  debounce: <T extends (...args: any[]) => any>(func: T, wait: number): T => {
    let timeout: NodeJS.Timeout;
    return ((...args: any[]) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => func(...args), wait);
    }) as T;
  },

  /**
   * 재시도 로직
   */
  retry: async <T>(fn: () => Promise<T>, maxAttempts: number = 3, delay: number = 1000): Promise<T> => {
    let lastError: Error;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (error: any) {
        lastError = error;
        console.warn(`재시도 ${attempt}/${maxAttempts} 실패:`, error.message);

        if (attempt < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, delay * attempt));
        }
      }
    }

    throw lastError!;
  },

  /**
   * 파일을 Base64로 변환
   */
  fileToBase64: (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        resolve(result.split(',')[1]); // data:image/jpeg;base64, 부분 제거
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  },

  /**
   * 브라우저 호환성 체크
   */
  checkBrowserCompatibility: (): {
    websocket: boolean;
    fileApi: boolean;
    formData: boolean;
    fetch: boolean;
    overall: boolean;
  } => {
    const features = {
      websocket: 'WebSocket' in window,
      fileApi: 'File' in window && 'FileReader' in window,
      formData: 'FormData' in window,
      fetch: 'fetch' in window
    };

    return {
      ...features,
      overall: features.websocket && features.fileApi && features.formData && features.fetch
    };
  }
};

// 환경 검증 (수정된 버전)
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
  const compatibility = utils.checkBrowserCompatibility();
  if (!compatibility.overall) {
    if (!compatibility.fetch) errors.push('Fetch API not supported');
    if (!compatibility.fileApi) errors.push('File API not supported');
    if (!compatibility.formData) errors.push('FormData not supported');
    if (!compatibility.websocket) warnings.push('WebSocket not supported - real-time features disabled');
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
};

// 초기화 함수 (수정된 버전)
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

// =================================================================
// 🔧 개발 및 디버깅 도구 (새로 추가)
// =================================================================

export const devTools = {
  /**
   * API 연결 테스트
   */
  testAPI: async (): Promise<any> => {
    try {
      console.log('🧪 API 연결 테스트 시작...');
      const client = getApiClient();
      const isHealthy = await client.healthCheck();
      
      if (isHealthy) {
        console.log('✅ API 연결 테스트 성공');
        return { success: true, message: 'API 서버가 정상 작동 중입니다.' };
      } else {
        console.log('❌ API 연결 테스트 실패');
        return { success: false, message: 'API 서버에 연결할 수 없습니다.' };
      }
    } catch (error: any) {
      console.error('❌ API 테스트 오류:', error);
      return { 
        success: false, 
        message: getFriendlyErrorMessage(error),
        error: error.message 
      };
    }
  },

  /**
   * 파이프라인 워밍업 테스트
   */
  testWarmup: async (qualityMode: string = 'balanced'): Promise<any> => {
    try {
      console.log('🔥 파이프라인 워밍업 테스트 시작...');
      const client = getApiClient();
      await client.warmupPipeline(qualityMode as any);
      
      console.log('✅ 파이프라인 워밍업 성공');
      return { success: true, message: '파이프라인이 준비되었습니다.' };
    } catch (error: any) {
      console.error('❌ 워밍업 테스트 오류:', error);
      return { 
        success: false, 
        message: getFriendlyErrorMessage(error),
        error: error.message 
      };
    }
  },

  /**
   * 더미 가상 피팅 테스트
   */
  testDummyVirtualTryOn: async (): Promise<any> => {
    try {
      console.log('🎭 더미 가상 피팅 테스트 시작...');
      
      // 더미 파일 생성
      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext('2d');
      
      if (ctx) {
        ctx.fillStyle = '#ff0000';
        ctx.fillRect(0, 0, 256, 512);
        ctx.fillStyle = '#0000ff';
        ctx.fillRect(256, 0, 256, 512);
      }
      
      const dummyBlob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((blob) => resolve(blob!), 'image/jpeg', 0.8);
      });
      
      const personImage = new File([dummyBlob], 'person.jpg', { type: 'image/jpeg' });
      const clothingImage = new File([dummyBlob], 'clothing.jpg', { type: 'image/jpeg' });
      
      const request = {
        person_image: personImage,
        clothing_image: clothingImage,
        height: 170,
        weight: 65,
        quality_mode: 'fast' as const
      };
      
      const client = getApiClient();
      const result = await client.processVirtualTryOn(request);
      
      console.log('✅ 더미 가상 피팅 테스트 성공:', result);
      return { success: true, message: '더미 테스트가 완료되었습니다.', result };
    } catch (error: any) {
      console.error('❌ 더미 테스트 오류:', error);
      return { 
        success: false, 
        message: getFriendlyErrorMessage(error),
        error: error.message 
      };
    }
  },

  /**
   * 시스템 정보 조회
   */
  getSystemInfo: async (): Promise<any> => {
    try {
      const client = getApiClient();
      const [stats, info] = await Promise.all([
        client.getSystemStats(),
        client.getServerInfo()
      ]);
      
      return { success: true, stats, info };
    } catch (error: any) {
      console.error('❌ 시스템 정보 조회 오류:', error);
      return { 
        success: false, 
        message: getFriendlyErrorMessage(error) 
      };
    }
  },

  /**
   * 디버그 정보 내보내기
   */
  exportDebugInfo: (): string => {
    const client = getApiClient();
    const debugInfo = {
      timestamp: new Date().toISOString(),
      config,
      environment: validateEnvironment(),
      browserCompatibility: utils.checkBrowserCompatibility(),
      clientInfo: client.getClientInfo(),
      clientConfig: client.getConfig(),
      metrics: client.getMetrics(),
      userAgent: navigator.userAgent,
      url: window.location.href
    };
    
    return JSON.stringify(debugInfo, null, 2);
  }
};

// 전역 개발 도구 등록 (개발 모드에서만)
if (process.env.NODE_ENV === 'development' && typeof window !== 'undefined') {
  (window as any).myClosetDevTools = devTools;
  console.log('🛠️ MyCloset AI 개발 도구가 window.myClosetDevTools에 등록되었습니다.');
}