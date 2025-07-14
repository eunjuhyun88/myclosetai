/**
 * MyCloset AI 파이프라인 React Hook
 * 가상 피팅 기능을 위한 커스텀 훅
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import PipelineAPIClient, {
  VirtualTryOnRequest,
  VirtualTryOnResponse,
  PipelineProgress,
  PipelineStatus,
  PipelineUtils
} from '../services/pipeline_api';

export interface UsePipelineOptions {
  baseURL?: string;
  autoHealthCheck?: boolean;
  healthCheckInterval?: number;
}

export interface UsePipelineState {
  isProcessing: boolean;
  progress: number;
  progressMessage: string;
  result: VirtualTryOnResponse | null;
  error: string | null;
  isHealthy: boolean;
  pipelineStatus: PipelineStatus | null;
}

export interface UsePipelineActions {
  processVirtualTryOn: (request: VirtualTryOnRequest) => Promise<void>;
  clearResult: () => void;
  clearError: () => void;
  checkHealth: () => Promise<boolean>;
  getPipelineStatus: () => Promise<void>;
  warmupPipeline: (qualityMode?: string) => Promise<void>;
  testConnection: () => Promise<void>;
}

export const usePipeline = (options: UsePipelineOptions = {}): UsePipelineState & UsePipelineActions => {
  // 상태 관리
  const [state, setState] = useState<UsePipelineState>({
    isProcessing: false,
    progress: 0,
    progressMessage: '',
    result: null,
    error: null,
    isHealthy: false,
    pipelineStatus: null,
  });

  // API 클라이언트 인스턴스
  const apiClient = useRef<PipelineAPIClient>(
    new PipelineAPIClient(options.baseURL || 'http://localhost:8000')
  );

  // 헬스체크 인터벌 참조
  const healthCheckInterval = useRef<NodeJS.Timeout | null>(null);

  // 상태 업데이트 헬퍼
  const updateState = useCallback((updates: Partial<UsePipelineState>) => {
    setState(prev => ({ ...prev, ...updates }));
  }, []);

  // 진행률 콜백
  const handleProgress = useCallback((progress: PipelineProgress) => {
    updateState({
      progress: progress.progress,
      progressMessage: progress.message
    });
  }, [updateState]);

  // 가상 피팅 처리
  const processVirtualTryOn = useCallback(async (request: VirtualTryOnRequest) => {
    try {
      // 입력 검증
      if (!PipelineUtils.validateImageType(request.person_image)) {
        throw new Error('사용자 이미지 형식이 올바르지 않습니다.');
      }
      
      if (!PipelineUtils.validateImageType(request.clothing_image)) {
        throw new Error('의류 이미지 형식이 올바르지 않습니다.');
      }

      if (!PipelineUtils.validateFileSize(request.person_image)) {
        throw new Error('사용자 이미지 크기가 너무 큽니다. (최대 10MB)');
      }

      if (!PipelineUtils.validateFileSize(request.clothing_image)) {
        throw new Error('의류 이미지 크기가 너무 큽니다. (최대 10MB)');
      }

      // 처리 시작
      updateState({
        isProcessing: true,
        progress: 0,
        progressMessage: '처리를 시작합니다...',
        result: null,
        error: null
      });

      console.log('🎯 가상 피팅 처리 시작:', request);

      // API 호출
      const result = await apiClient.current.processVirtualTryOn(request, handleProgress);

      updateState({
        isProcessing: false,
        result,
        progress: 100,
        progressMessage: '완료!'
      });

      console.log('✅ 가상 피팅 처리 완료:', result);

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
      const friendlyError = PipelineUtils.getUserFriendlyError(errorMessage);
      
      updateState({
        isProcessing: false,
        error: friendlyError,
        progress: 0,
        progressMessage: ''
      });

      console.error('❌ 가상 피팅 처리 실패:', error);
    }
  }, [updateState, handleProgress]);

  // 결과 초기화
  const clearResult = useCallback(() => {
    updateState({
      result: null,
      progress: 0,
      progressMessage: ''
    });
  }, [updateState]);

  // 에러 초기화
  const clearError = useCallback(() => {
    updateState({ error: null });
  }, [updateState]);

  // 헬스체크
  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      const isHealthy = await apiClient.current.healthCheck();
      updateState({ isHealthy });
      return isHealthy;
    } catch (error) {
      console.error('헬스체크 실패:', error);
      updateState({ isHealthy: false });
      return false;
    }
  }, [updateState]);

  // 파이프라인 상태 조회
  const getPipelineStatus = useCallback(async () => {
    try {
      const pipelineStatus = await apiClient.current.getPipelineStatus();
      updateState({ pipelineStatus });
    } catch (error) {
      console.error('파이프라인 상태 조회 실패:', error);
      updateState({ pipelineStatus: null });
    }
  }, [updateState]);

  // 파이프라인 워밍업
  const warmupPipeline = useCallback(async (qualityMode: string = 'balanced') => {
    try {
      updateState({
        isProcessing: true,
        progressMessage: '파이프라인 워밍업 중...'
      });

      await apiClient.current.warmupPipeline(qualityMode);
      
      updateState({
        isProcessing: false,
        progressMessage: '워밍업 완료'
      });

      console.log('✅ 파이프라인 워밍업 완료');

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '워밍업 실패';
      updateState({
        isProcessing: false,
        error: errorMessage,
        progressMessage: ''
      });

      console.error('❌ 파이프라인 워밍업 실패:', error);
    }
  }, [updateState]);

  // 연결 테스트
  const testConnection = useCallback(async () => {
    try {
      updateState({
        isProcessing: true,
        progressMessage: '연결 테스트 중...'
      });

      const result = await apiClient.current.testDummyProcess(handleProgress);
      
      updateState({
        isProcessing: false,
        result,
        progressMessage: '테스트 완료'
      });

      console.log('✅ 연결 테스트 완료');

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '연결 테스트 실패';
      updateState({
        isProcessing: false,
        error: errorMessage,
        progressMessage: ''
      });

      console.error('❌ 연결 테스트 실패:', error);
    }
  }, [updateState, handleProgress]);

  // 자동 헬스체크 설정
  useEffect(() => {
    if (options.autoHealthCheck) {
      // 즉시 헬스체크 실행
      checkHealth();

      // 주기적 헬스체크 설정
      const interval = options.healthCheckInterval || 30000; // 기본 30초
      healthCheckInterval.current = setInterval(checkHealth, interval);

      return () => {
        if (healthCheckInterval.current) {
          clearInterval(healthCheckInterval.current);
        }
      };
    }
  }, [options.autoHealthCheck, options.healthCheckInterval, checkHealth]);

  // 컴포넌트 언마운트 시 정리
  useEffect(() => {
    return () => {
      if (healthCheckInterval.current) {
        clearInterval(healthCheckInterval.current);
      }
    };
  }, []);

  return {
    // 상태
    ...state,
    
    // 액션
    processVirtualTryOn,
    clearResult,
    clearError,
    checkHealth,
    getPipelineStatus,
    warmupPipeline,
    testConnection
  };
};

// 편의 함수들
export const usePipelineStatus = (options: UsePipelineOptions = {}) => {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const apiClient = useRef<PipelineAPIClient>(
    new PipelineAPIClient(options.baseURL || 'http://localhost:8000')
  );

  const fetchStatus = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const pipelineStatus = await apiClient.current.getPipelineStatus();
      setStatus(pipelineStatus);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '상태 조회 실패';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  return {
    status,
    isLoading,
    error,
    refetch: fetchStatus
  };
};

export const usePipelineHealth = (options: UsePipelineOptions = {}) => {
  const [isHealthy, setIsHealthy] = useState<boolean>(false);
  const [isChecking, setIsChecking] = useState(false);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);

  const apiClient = useRef<PipelineAPIClient>(
    new PipelineAPIClient(options.baseURL || 'http://localhost:8000')
  );

  const checkHealth = useCallback(async () => {
    setIsChecking(true);

    try {
      const healthy = await apiClient.current.healthCheck();
      setIsHealthy(healthy);
      setLastCheck(new Date());
    } catch (error) {
      setIsHealthy(false);
      console.error('헬스체크 실패:', error);
    } finally {
      setIsChecking(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();

    // 자동 헬스체크
    if (options.autoHealthCheck) {
      const interval = options.healthCheckInterval || 30000;
      const intervalId = setInterval(checkHealth, interval);
      return () => clearInterval(intervalId);
    }
  }, [checkHealth, options.autoHealthCheck, options.healthCheckInterval]);

  return {
    isHealthy,
    isChecking,
    lastCheck,
    checkHealth
  };
};

export default usePipeline;