/**
 * MyCloset AI usePipeline Hook - 초기화 오류 수정 버전
 * 순환 참조와 변수 호이스팅 문제 해결
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';

// 타입 import
import type {
  UsePipelineOptions,
  UsePipelineState,
  UsePipelineActions,
  VirtualTryOnRequest,
  VirtualTryOnResponse,
  PipelineProgress,
  PipelineStep,
  PipelineStatus,
  SystemStats,
  SystemHealth,
  TaskInfo,
  ProcessingStatus,
} from '../types/pipeline';

// 서비스 import
import WebSocketManager from '../services/WebSocketManager';
import PipelineAPIClient from '../services/PipelineAPIClient';
import { PipelineUtils } from '../utils/pipelineUtils';

// =================================================================
// 🔧 메인 usePipeline Hook (순환 참조 문제 해결)
// =================================================================

export const usePipeline = (
  options: UsePipelineOptions = {}
): UsePipelineState & UsePipelineActions => {

  // 설정 먼저 생성 (다른 것들보다 우선)
  const config = useMemo(() => ({
    baseURL: options.baseURL || process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
    wsURL: options.wsURL || 
           options.baseURL?.replace('http', 'ws') || 
           process.env.REACT_APP_WS_BASE_URL || 
           'ws://localhost:8000',
    
    // 연결 설정
    autoReconnect: options.autoReconnect ?? true,
    maxReconnectAttempts: options.maxReconnectAttempts || 10,
    reconnectInterval: options.reconnectInterval || 3000,
    heartbeatInterval: options.heartbeatInterval || 30000,
    connectionTimeout: options.connectionTimeout || 15000,
    
    // 시스템 최적화
    device: options.device || 'auto',
    device_type: options.device_type || 'auto',
    memory_gb: options.memory_gb || 16.0,
    is_m3_max: options.is_m3_max ?? false,
    optimization_enabled: options.optimization_enabled ?? true,
    quality_level: options.quality_level || 'balanced',
    
    // Hook 기능 설정
    autoHealthCheck: options.autoHealthCheck ?? true,
    healthCheckInterval: options.healthCheckInterval || 60000,
    persistSession: options.persistSession ?? true,
    enableDetailedProgress: options.enableDetailedProgress ?? true,
    enableRetry: options.enableRetry ?? true,
    maxRetryAttempts: options.maxRetryAttempts || 3,
    
    // 성능 설정
    enableCaching: options.enableCaching ?? true,
    cacheTimeout: options.cacheTimeout || 300000,
    requestTimeout: options.requestTimeout || 30000,
    maxConcurrentRequests: options.maxConcurrentRequests || 3,
    
    // 기능 플래그
    enableTaskTracking: options.enableTaskTracking ?? true,
    enableBrandIntegration: options.enableBrandIntegration ?? true,
    enableDebugMode: options.enableDebugMode ?? (process.env.NODE_ENV === 'development'),
    
    ...options,
  }), [options]);

  // 초기 상태 정의 (안전한 기본값들)
  const initialState: UsePipelineState = useMemo(() => ({
    // 처리 상태
    isProcessing: false,
    progress: 0,
    progressMessage: '',
    currentStep: '',
    stepProgress: 0,
    
    // 결과 상태
    result: null,
    error: null,
    
    // 연결 상태
    isConnected: false,
    isHealthy: false,
    connectionAttempts: 0,
    lastConnectionAttempt: null,
    
    // 시스템 상태
    pipelineStatus: null,
    systemStats: null,
    systemHealth: null,
    
    // 세션 관리
    sessionId: null,
    currentTaskId: null,
    
    // 상세 진행률
    steps: [],
    activeTask: null,
    
    // 메타데이터
    totalRequestsCount: 0,
    successfulRequestsCount: 0,
    
    // 캐시 상태
    cachedResults: new Map(),
    
    // 브랜드 데이터
    brandSizeData: new Map(),
  }), []);

  // Hook 상태 관리
  const [state, setState] = useState<UsePipelineState>(initialState);

  // 서비스 인스턴스들 (ref로 관리)
  const wsManager = useRef<WebSocketManager | null>(null);
  const apiClient = useRef<PipelineAPIClient | null>(null);
  const healthCheckTimer = useRef<NodeJS.Timeout | null>(null);
  const isInitialized = useRef<boolean>(false);
  const eventListeners = useRef<Map<string, Function[]>>(new Map());

  // =================================================================
  // 🔧 안전한 상태 업데이트 헬퍼
  // =================================================================

  const updateState = useCallback((updates: Partial<UsePipelineState>) => {
    setState(prev => {
      try {
        const newState = { ...prev, ...updates };
        
        // 메트릭 자동 업데이트 (안전하게)
        if (updates.result?.success && prev.successfulRequestsCount !== undefined) {
          newState.successfulRequestsCount = prev.successfulRequestsCount + 1;
        }
        if (updates.isProcessing === false && prev.totalRequestsCount !== undefined) {
          newState.totalRequestsCount = prev.totalRequestsCount + 1;
        }
        
        return newState;
      } catch (error) {
        console.error('상태 업데이트 오류:', error);
        return prev; // 오류 시 이전 상태 유지
      }
    });
  }, []);

  // =================================================================
  // 🔧 이벤트 시스템 (안전한 구현)
  // =================================================================

  const emitEvent = useCallback((event: string, data?: any) => {
    try {
      const listeners = eventListeners.current;
      const handlers = listeners.get(event);
      if (handlers) {
        handlers.forEach(handler => {
          try {
            handler({ type: event, data, timestamp: Date.now() });
          } catch (error) {
            console.error('이벤트 핸들러 오류:', { event, error });
          }
        });
      }
    } catch (error) {
      console.error('이벤트 발생 오류:', error);
    }
  }, []);

  // =================================================================
  // 🔧 서비스 초기화 (지연 초기화로 순환 참조 방지)
  // =================================================================

  const getAPIClient = useCallback(() => {
    if (!apiClient.current) {
      try {
        apiClient.current = new PipelineAPIClient(config);
        console.log('✅ PipelineAPIClient 초기화 완료');
      } catch (error) {
        console.error('❌ PipelineAPIClient 초기화 실패:', error);
      }
    }
    return apiClient.current;
  }, [config]);

  const getWebSocketManager = useCallback(() => {
    if (!wsManager.current) {
      try {
        const wsUrl = `${config.wsURL}/api/ws/pipeline-progress`;
        wsManager.current = new WebSocketManager(wsUrl, config);
        console.log('✅ WebSocketManager 초기화 완료');
      } catch (error) {
        console.error('❌ WebSocketManager 초기화 실패:', error);
      }
    }
    return wsManager.current;
  }, [config]);

  // =================================================================
  // 🔧 WebSocket 메시지 처리 (안전한 구현)
  // =================================================================

  const handleWebSocketMessage = useCallback((data: PipelineProgress) => {
    try {
      if (config.enableDebugMode) {
        console.log('📨 WebSocket 메시지:', data);
      }

      switch (data.type) {
        case 'connection_established':
          updateState({
            isConnected: true,
            sessionId: data.session_id || state.sessionId,
            error: null,
            connectionAttempts: 0
          });
          emitEvent('connected', data);
          break;

        case 'pipeline_progress':
          updateState({
            progress: data.progress || 0,
            progressMessage: data.message || '',
            currentStep: data.step_name || state.currentStep
          });
          emitEvent('progress', data);
          break;

        case 'completed':
          updateState({
            isProcessing: false,
            progress: 100,
            progressMessage: '처리 완료!'
          });
          emitEvent('completed', data);
          break;

        case 'error':
          const errorMessage = typeof data.message === 'string' 
            ? data.message 
            : '알 수 없는 오류가 발생했습니다.';
          
          updateState({
            isProcessing: false,
            error: errorMessage,
            progress: 0,
            progressMessage: ''
          });
          emitEvent('error', data);
          break;

        default:
          console.log('알 수 없는 메시지 타입:', data.type);
      }
    } catch (error) {
      console.error('WebSocket 메시지 처리 오류:', error);
    }
  }, [config.enableDebugMode, state.sessionId, state.currentStep, updateState, emitEvent]);

  // =================================================================
  // 🔧 연결 관리 (안전한 구현)
  // =================================================================

  const connect = useCallback(async (): Promise<boolean> => {
    try {
      const wsManagerInstance = getWebSocketManager();
      if (!wsManagerInstance) {
        throw new Error('WebSocketManager 초기화 실패');
      }

      if (wsManagerInstance.isConnected()) {
        return true;
      }

      updateState({
        connectionAttempts: (state.connectionAttempts || 0) + 1,
        lastConnectionAttempt: new Date()
      });

      // 이벤트 핸들러 설정
      wsManagerInstance.setOnMessage(handleWebSocketMessage);
      wsManagerInstance.setOnConnected(() => {
        updateState({ isConnected: true, error: null });
        emitEvent('ws_connected', {});
      });
      wsManagerInstance.setOnDisconnected(() => {
        updateState({ isConnected: false });
        emitEvent('ws_disconnected', {});
      });
      wsManagerInstance.setOnError((error) => {
        updateState({
          isConnected: false,
          error: '연결 오류가 발생했습니다.'
        });
        emitEvent('ws_error', { error });
      });

      const connected = await wsManagerInstance.connect();
      return connected;

    } catch (error) {
      console.error('WebSocket 연결 실패:', error);
      updateState({
        isConnected: false,
        error: '서버에 연결할 수 없습니다.'
      });
      return false;
    }
  }, [state.connectionAttempts, handleWebSocketMessage, getWebSocketManager, updateState, emitEvent]);

  const disconnect = useCallback(() => {
    try {
      wsManager.current?.disconnect();
      updateState({ isConnected: false });
      console.log('🔌 WebSocket 연결 해제됨');
    } catch (error) {
      console.error('연결 해제 오류:', error);
    }
  }, [updateState]);

  // =================================================================
  // 🔧 가상 피팅 처리 (단순화된 안전한 버전)
  // =================================================================

  const processVirtualTryOn = useCallback(async (
    request: VirtualTryOnRequest
  ): Promise<VirtualTryOnResponse | void> => {
    try {
      const client = getAPIClient();
      if (!client) {
        throw new Error('API 클라이언트를 초기화할 수 없습니다.');
      }

      // 세션 ID 생성
      const sessionId = request.session_id || `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      // 초기 상태 설정
      updateState({
        isProcessing: true,
        progress: 0,
        progressMessage: '가상 피팅을 시작합니다...',
        result: null,
        error: null,
        sessionId
      });

      console.log('🎯 가상 피팅 처리 시작');

      // WebSocket 연결 시도 (선택적)
      if (config.enableDetailedProgress) {
        await connect().catch(err => {
          console.warn('WebSocket 연결 실패 (진행률 업데이트 없이 계속):', err);
        });
      }

      // API 처리
      const result = await client.processVirtualTryOn({
        ...request,
        session_id: sessionId
      });

      // 성공 시 상태 업데이트
      updateState({
        isProcessing: false,
        result,
        progress: 100,
        progressMessage: '가상 피팅 완료!'
      });

      console.log('✅ 가상 피팅 처리 완료');
      emitEvent('processing_complete', { result });
      return result;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
      
      updateState({
        isProcessing: false,
        error: errorMessage,
        progress: 0,
        progressMessage: ''
      });

      console.error('❌ 가상 피팅 처리 실패:', error);
      emitEvent('processing_error', { error: errorMessage });
      throw error;
    }
  }, [config.enableDetailedProgress, getAPIClient, connect, updateState, emitEvent]);

  // =================================================================
  // 🔧 기본 액션들 (간단하고 안전한 구현)
  // =================================================================

  const clearResult = useCallback(() => {
    updateState({
      result: null,
      progress: 0,
      progressMessage: ''
    });
  }, [updateState]);

  const clearError = useCallback(() => {
    updateState({ error: null });
  }, [updateState]);

  const reset = useCallback(() => {
    updateState({
      isProcessing: false,
      progress: 0,
      progressMessage: '',
      currentStep: '',
      stepProgress: 0,
      result: null,
      error: null
    });
  }, [updateState]);

  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      const client = getAPIClient();
      if (!client) return false;
      
      const isHealthy = await client.healthCheck();
      updateState({ isHealthy });
      return isHealthy;
    } catch (error) {
      console.error('헬스체크 실패:', error);
      updateState({ isHealthy: false });
      return false;
    }
  }, [getAPIClient, updateState]);

  const warmupPipeline = useCallback(async (qualityMode: string = 'balanced'): Promise<void> => {
    try {
      updateState({ isProcessing: true, progressMessage: '파이프라인 워밍업 중...' });
      
      const client = getAPIClient();
      if (!client) throw new Error('API 클라이언트 없음');
      
      await client.warmupPipeline(qualityMode as any);
      
      updateState({ isProcessing: false, progressMessage: '워밍업 완료' });
      console.log('✅ 파이프라인 워밍업 완료');
    } catch (error) {
      updateState({
        isProcessing: false,
        error: '워밍업에 실패했습니다.',
        progressMessage: ''
      });
      console.error('❌ 워밍업 실패:', error);
    }
  }, [getAPIClient, updateState]);

  const testConnection = useCallback(async (): Promise<void> => {
    try {
      updateState({ isProcessing: true, progressMessage: '연결 테스트 중...' });
      
      const [wsConnected, healthOk] = await Promise.all([
        connect(),
        checkHealth()
      ]);

      if (!wsConnected || !healthOk) {
        throw new Error('연결 테스트 실패');
      }

      updateState({
        isProcessing: false,
        progressMessage: '연결 테스트 완료',
        error: null
      });

      console.log('✅ 연결 테스트 완료');
    } catch (error) {
      updateState({
        isProcessing: false,
        error: '연결 테스트에 실패했습니다.',
        progressMessage: ''
      });
      console.error('❌ 연결 테스트 실패:', error);
    }
  }, [connect, checkHealth, updateState]);

  // =================================================================
  // 🔧 더미 구현들 (에러 방지용)
  // =================================================================

  const reconnect = useCallback(async (): Promise<boolean> => {
    disconnect();
    await new Promise(resolve => setTimeout(resolve, 1000));
    return await connect();
  }, [disconnect, connect]);

  const getPipelineStatus = useCallback(async (): Promise<void> => {
    try {
      const client = getAPIClient();
      if (client) {
        const status = await client.getPipelineStatus();
        updateState({ pipelineStatus: status });
      }
    } catch (error) {
      console.error('파이프라인 상태 조회 실패:', error);
    }
  }, [getAPIClient, updateState]);

  const getSystemStats = useCallback(async (): Promise<void> => {
    try {
      const client = getAPIClient();
      if (client) {
        const stats = await client.getSystemStats();
        updateState({ systemStats: stats });
      }
    } catch (error) {
      console.error('시스템 통계 조회 실패:', error);
    }
  }, [getAPIClient, updateState]);

  const getSystemHealth = useCallback(async (): Promise<void> => {
    // 더미 구현
  }, []);

  const getTaskStatus = useCallback(async (taskId: string): Promise<ProcessingStatus | null> => {
    return null; // 더미 구현
  }, []);

  const cancelTask = useCallback(async (taskId: string): Promise<boolean> => {
    return false; // 더미 구현
  }, []);

  const retryTask = useCallback(async (taskId: string): Promise<boolean> => {
    return false; // 더미 구현
  }, []);

  const getTaskHistory = useCallback((): TaskInfo[] => {
    return []; // 더미 구현
  }, []);

  const getBrandSizes = useCallback(async (brand: string): Promise<any> => {
    return null; // 더미 구현
  }, []);

  const getSizeRecommendation = useCallback(async (measurements: any, brand: string, item: string): Promise<any> => {
    return null; // 더미 구현
  }, []);

  const sendHeartbeat = useCallback(() => {
    // 더미 구현
  }, []);

  const exportLogs = useCallback(() => {
    const logs = { state, timestamp: new Date().toISOString() };
    console.log('로그 내보내기:', logs);
  }, [state]);

  const addEventListener = useCallback((event: string, handler: Function) => {
    const listeners = eventListeners.current;
    if (!listeners.has(event)) {
      listeners.set(event, []);
    }
    listeners.get(event)!.push(handler);
  }, []);

  const removeEventListener = useCallback((event: string, handler: Function) => {
    const listeners = eventListeners.current;
    const handlers = listeners.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }, []);

  // =================================================================
  // 🔧 Effect들 (최소한으로)
  // =================================================================

  useEffect(() => {
    if (!isInitialized.current) {
      console.log('🚀 usePipeline Hook 초기화');
      isInitialized.current = true;
      
      // 자동 헬스체크
      if (config.autoHealthCheck) {
        checkHealth();
        healthCheckTimer.current = setInterval(checkHealth, config.healthCheckInterval);
      }
    }

    return () => {
      if (healthCheckTimer.current) {
        clearInterval(healthCheckTimer.current);
        healthCheckTimer.current = null;
      }
      wsManager.current?.cleanup();
      apiClient.current?.cleanup();
    };
  }, [config.autoHealthCheck, config.healthCheckInterval, checkHealth]);

  // =================================================================
  // 🔧 Hook 반환값
  // =================================================================

  return {
    // 상태
    ...state,
    
    // 메인 액션
    processVirtualTryOn,
    
    // 결과 관리
    clearResult,
    clearError,
    reset,
    
    // 연결 관리
    connect,
    disconnect,
    reconnect,
    
    // 상태 조회
    checkHealth,
    getPipelineStatus,
    getSystemStats,
    getSystemHealth,
    
    // 파이프라인 관리
    warmupPipeline,
    testConnection,
    
    // Task 관리
    getTaskStatus,
    cancelTask,
    retryTask,
    getTaskHistory,
    
    // 브랜드/사이즈 기능
    getBrandSizes,
    getSizeRecommendation,
    
    // 유틸리티
    sendHeartbeat,
    exportLogs,
    
    // 이벤트 시스템
    addEventListener,
    removeEventListener,
  };
};

// =================================================================
// 🔧 편의 Hook들 (단순화)
// =================================================================

export const usePipelineHealth = (options: UsePipelineOptions = {}) => {
  const [isHealthy, setIsHealthy] = useState<boolean>(false);
  const [isChecking, setIsChecking] = useState(false);
  const [lastCheck, setLastCheck] = useState<Date | null>(null);

  const checkHealth = useCallback(async () => {
    setIsChecking(true);
    try {
      // 간단한 헬스체크
      const response = await fetch(`${options.baseURL || 'http://localhost:8000'}/health`);
      const healthy = response.ok;
      setIsHealthy(healthy);
      setLastCheck(new Date());
    } catch (error) {
      setIsHealthy(false);
    } finally {
      setIsChecking(false);
    }
  }, [options.baseURL]);

  useEffect(() => {
    checkHealth();
    if (options.autoHealthCheck) {
      const interval = setInterval(checkHealth, options.healthCheckInterval || 30000);
      return () => clearInterval(interval);
    }
  }, [checkHealth, options.autoHealthCheck, options.healthCheckInterval]);

  return { isHealthy, isChecking, lastCheck, checkHealth };
};

export default usePipeline;