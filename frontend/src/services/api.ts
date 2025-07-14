/**
 * MyCloset AI 일반 API 서비스
 * 기본 HTTP 요청 및 공통 API 함수들
 */

import { APIResponse, APIError, SystemStats } from '../types';

export class APIService {
  private baseURL: string;
  private defaultHeaders: Record<string, string>;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.baseURL = baseURL;
    this.defaultHeaders = {
      'Accept': 'application/json',
    };
  }

  /**
   * 기본 fetch 래퍼
   */
  private async request<T = any>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      ...options,
      headers: {
        ...this.defaultHeaders,
        ...options.headers,
      },
    };

    console.log(`🌐 API 요청: ${config.method || 'GET'} ${url}`);

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const error: APIError = {
          status: response.status,
          message: errorData.detail || errorData.error || response.statusText,
          details: errorData
        };
        throw error;
      }

      const data = await response.json();
      console.log(`✅ API 응답 성공: ${endpoint}`, data);
      return data;

    } catch (error) {
      console.error(`❌ API 요청 실패: ${endpoint}`, error);
      throw error;
    }
  }

  /**
   * GET 요청
   */
  async get<T = any>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  /**
   * POST 요청 (JSON)
   */
  async post<T = any>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  /**
   * POST 요청 (FormData)
   */
  async postFormData<T = any>(endpoint: string, formData: FormData): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: formData,
    });
  }

  /**
   * PUT 요청
   */
  async put<T = any>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  /**
   * DELETE 요청
   */
  async delete<T = any>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }

  /**
   * 헬스 체크
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.get('/health');
      return response.status === 'healthy';
    } catch {
      return false;
    }
  }

  /**
   * 시스템 통계 조회
   */
  async getSystemStats(): Promise<SystemStats> {
    return this.get<SystemStats>('/stats');
  }

  /**
   * 서버 정보 조회
   */
  async getServerInfo(): Promise<any> {
    return this.get('/');
  }

  /**
   * 테스트 엔드포인트
   */
  async testEndpoint(): Promise<any> {
    return this.get('/test');
  }

  /**
   * 파일 업로드 헬퍼
   */
  async uploadFile(
    endpoint: string,
    file: File,
    fieldName: string = 'file',
    additionalFields?: Record<string, string>
  ): Promise<any> {
    const formData = new FormData();
    formData.append(fieldName, file);

    if (additionalFields) {
      Object.entries(additionalFields).forEach(([key, value]) => {
        formData.append(key, value);
      });
    }

    return this.postFormData(endpoint, formData);
  }

  /**
   * 다중 파일 업로드
   */
  async uploadMultipleFiles(
    endpoint: string,
    files: { [fieldName: string]: File },
    additionalFields?: Record<string, string>
  ): Promise<any> {
    const formData = new FormData();

    Object.entries(files).forEach(([fieldName, file]) => {
      formData.append(fieldName, file);
    });

    if (additionalFields) {
      Object.entries(additionalFields).forEach(([key, value]) => {
        formData.append(key, value);
      });
    }

    return this.postFormData(endpoint, formData);
  }

  /**
   * 다운로드 헬퍼
   */
  async downloadFile(endpoint: string, filename?: string): Promise<void> {
    const url = `${this.baseURL}${endpoint}`;
    
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Download failed: ${response.status}`);

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename || 'download';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      window.URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      console.error('다운로드 실패:', error);
      throw error;
    }
  }
}

// 기본 API 인스턴스
export const apiService = new APIService();

// 편의 함수들
export const api = {
  // 기본 요청들
  get: (endpoint: string) => apiService.get(endpoint),
  post: (endpoint: string, data?: any) => apiService.post(endpoint, data),
  put: (endpoint: string, data?: any) => apiService.put(endpoint, data),
  delete: (endpoint: string) => apiService.delete(endpoint),
  
  // 특수 요청들
  uploadFile: (endpoint: string, file: File, fieldName?: string, additionalFields?: Record<string, string>) => 
    apiService.uploadFile(endpoint, file, fieldName, additionalFields),
  
  uploadFiles: (endpoint: string, files: { [fieldName: string]: File }, additionalFields?: Record<string, string>) =>
    apiService.uploadMultipleFiles(endpoint, files, additionalFields),
  
  downloadFile: (endpoint: string, filename?: string) =>
    apiService.downloadFile(endpoint, filename),
  
  // 시스템 API들
  health: () => apiService.healthCheck(),
  stats: () => apiService.getSystemStats(),
  serverInfo: () => apiService.getServerInfo(),
  test: () => apiService.testEndpoint(),
};

// 에러 처리 유틸리티
export const handleAPIError = (error: any): string => {
  if (error.status) {
    // APIError 타입
    switch (error.status) {
      case 400:
        return `잘못된 요청: ${error.message}`;
      case 401:
        return '인증이 필요합니다.';
      case 403:
        return '접근 권한이 없습니다.';
      case 404:
        return '요청한 리소스를 찾을 수 없습니다.';
      case 500:
        return `서버 오류: ${error.message}`;
      case 503:
        return '서비스를 일시적으로 사용할 수 없습니다.';
      default:
        return `API 오류 (${error.status}): ${error.message}`;
    }
  }
  
  if (error.name === 'TypeError' && error.message.includes('fetch')) {
    return '네트워크 연결을 확인해주세요.';
  }
  
  return error.message || '알 수 없는 오류가 발생했습니다.';
};

// 재시도 로직을 포함한 API 호출
export const withRetry = async <T>(
  apiCall: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> => {
  let lastError: any;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await apiCall();
    } catch (error) {
      lastError = error;
      
      if (attempt === maxRetries) {
        break;
      }
      
      console.warn(`API 호출 실패 (${attempt}/${maxRetries}), ${delay}ms 후 재시도...`);
      await new Promise(resolve => setTimeout(resolve, delay * attempt));
    }
  }
  
  throw lastError;
};

export default apiService;