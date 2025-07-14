/**
 * MyCloset AI 메인 파이프라인 React Hook
 * 백엔드 통일된 생성자 패턴을 따른 React Hook
 * - 모듈화된 구조로 순환참조 제거
 * - 안정적인 WebSocket 및 API 연결
 * - M3 Max 최적화 및 백엔드 완전 호환
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
  PipelineStatus,
  SystemStats,
  PipelineStep,
} from '../types/pipeline';

// 서비스 import
import WebSocketManager from '../services/WebSocketManager';
import PipelineAPIClient from '../services/PipelineAPIClient';
import { PipelineUtils } from '../utils/pipelineUtils';

// =================================================================
// 🔧 메인 Hook - 백엔드 통일된 패턴 적용
// =================================================================

export const usePipeline = (
  options: UsePipelineOptions = {},
  ...kwargs: any[] // 🎯 백엔드 패턴과 호환
): UsePipelineState & UsePipelineActions => {
  
  // 💡 지능적 설정 통합 (백엔드 패턴 호환)
  const config = useMemo(() => {
    const baseConfig: UsePipelineOptions = {
      baseURL: options.baseURL || 'http://localhost:8000',
      wsURL: options.wsURL || options.baseURL?.replace('http', 'ws') || 'ws://localhost:8000',
      autoReconnect: options.autoReconnect ?? true,
      maxReconnectAttempts: options.maxReconnectAttempts || 5,
      reconnectInterval: options.reconnectInterval || 3000,
      heartbeatInterval: options.heartbeatInterval || 30000,
      connectionTimeout: options.connectionTimeout || 10000,
      
      // 🔧 백엔드 호환 시스템 파라미터
      device: options.device || PipelineUtils.autoDetectDevice(),
      device_type: options.device_type || PipelineUtils.autoDetectDeviceType(),
      memory_gb: options.memory_gb || 16.0,
      is_m3_max: options.is_m3_max ?? PipelineUtils.detectM3Max(),
      optimization_enabled: options.optimization_enabled ?? true,
      quality_level: options.quality_level || 'balanced',
      
      // Hook 전용 설정들
      autoHealthCheck: options.autoHealthCheck ?? true,
      healthCheckInterval: options.healthCheckInterval || 30000,
      persistSession: options.persistSession ?? true,
      enableDetailedProgress: options.enableDetailedProgress ?? true,
      enableRetry: options.enableRetry ?? true,
      maxRetryAttempts: options.maxRetryAttempts || 3,
      
      ...options,
    };
    
    // ⚙️ 추가 파라미터 병합 (백엔드 패턴)
    return PipelineUtils.mergeStepSpecificConfig(
      baseConfig,
      kwargs.reduce((acc, kwarg) => ({ ...acc, ...kwarg }), {}),
      PipelineUtils.getSystemParams()
    );
  }, [options, kwargs]);

  // 상태 관리 (백엔드 패턴과 호환)
  const [state, setState] = useState<UsePipelineState>({
    isProcessing: false,
    progress: 0,
    progressMessage: '',
    currentStep: '',
    stepProgress: 0,
    result: null,
    error: null,
    isConnected: false,
    isHealthy: false,
    connectionAttempts: 0,
    lastConnectionAttempt: null,
    pipelineStatus: null,
    systemStats: null,
    sessionId: null,
    steps: []
  });

  // 서비스 인스턴스들 (백엔드 패턴 적용)
  const wsManager = useRef<WebSocketManager | null>(null);
  const apiClient = useRef<PipelineAPIClient | null>(null);
  const healthCheckTimer = useRef<NodeJS.Timeout | null>(null);
  const messageLog = useRef<any[]>([]);
  const isInitialized = useRef<boolean>(false);

  // 상태 업데이트 헬퍼
  const updateState = useCallback((updates: Partial<UsePipelineState>) => {
    setState(prev => ({ ...prev, ...updates }));
  }, []);

  // =================================================================
  // 🔧 서비스 초기화 메서드들 (백엔드 패턴 적용)
  // =================================================================

  /**
   * API 클라이언트 초기화 (백엔드 패턴 적용)
   */
  const initializeAPIClient = useCallback(() => {
    if (!apiClient.current) {
      // 백엔드 통일된 생성자 패턴 적용
      apiClient.current = new PipelineAPIClient(config, ...kwargs);
      PipelineUtils.log('info', '✅ API 클라이언트 초기화 완료');
    }
  }, [config, kwargs]);

  /**
   * WebSocket 관리자 초기화 (백엔드 패턴 적용)
   */
  const initializeWebSocketManager = useCallback(() => {
    if (!wsManager.current) {
      const wsUrl = `${config.wsURL}/api/ws/pipeline-progress`;
      // 백엔드 통일된 생성자 패턴 적용
      wsManager.current = new WebSocketManager(wsUrl, config, ...kwargs);
      PipelineUtils.log('info', '✅ WebSocket 관리자 초기화 완료');
    }
  }, [config, kwargs]);

  // =================================================================
  // 🔧 WebSocket 메시지 처리
  // =================================================================

  const handleWebSocketMessage = useCallback((data: PipelineProgress) => {
    PipelineUtils.log('info', '📨 WebSocket 메시지 수신', data.type);
    messageLog.current.push({ ...data, receivedAt: new Date() });

    switch (data.type) {
      case 'connection_established':
        updateState({
          isConnected: true,
          sessionId: data.session_id || state.sessionId,
          error: null
        });
        break;

      case 'pipeline_progress':
        updateState({
          progress: data.progress,
          progressMessage: data.message,
          currentStep: data.step_name || state.currentStep
        });
        break;

      case 'step_update':
        updateState({
          currentStep: data.step_name || '',
          stepProgress: data.progress,
          steps: data.steps || state.steps
        });
        break;

      case 'completed':
        updateState({
          isProcessing: false,
          progress: 100,
          progressMessage: '처리 완료!'
        });
        break;

      case 'error':
        updateState({
          isProcessing: false,
          error: PipelineUtils.getUserFriendlyError(data.message),
          progress: 0,
          progressMessage: ''
        });
        break;

      default:
        PipelineUtils.log('warn', '알 수 없는 메시지 타입', data.type);
    }
  }, [updateState, state.sessionId, state.currentStep, state.steps]);

  // =================================================================
  // 🔧 연결 관리 메서드들
  // =================================================================

  /**
   * WebSocket 연결 설정 (백엔드 패턴 적용)
   */
  const connect = useCallback(async (): Promise<boolean> => {
    if (wsManager.current?.isConnected()) {
      return true;
    }

    updateState({
      connectionAttempts: state.connectionAttempts + 1,
      lastConnectionAttempt: new Date()
    });

    try {
      initializeWebSocketManager();

      if (!wsManager.current) {
        throw new Error('WebSocket 관리자 초기화 실패');
      }

      // 메시지 핸들러 설정
      wsManager.current.setOnMessage(handleWebSocketMessage);
      wsManager.current.setOnConnected(() => {
        updateState({ isConnected: true, error: null });
      });
      wsManager.current.setOnDisconnected(() => {
        updateState({ isConnected: false });
      });
      wsManager.current.setOnError((error) => {
        updateState({
          isConnected: false,
          error: PipelineUtils.getUserFriendlyError('connection failed')
        });
      });

      const connected = await wsManager.current.connect();
      
      if (connected && state.sessionId) {
        wsManager.current.subscribeToSession(state.sessionId);
      }

      return connected;
    } catch (error) {
      PipelineUtils.log('error', '❌ WebSocket 연결 실패', error);
      updateState({
        isConnected: false,
        error: PipelineUtils.getUserFriendlyError('connection failed')
      });
      return false;
    }
  }, [
    config, 
    handleWebSocketMessage, 
    state.connectionAttempts, 
    state.sessionId, 
    updateState,
    initializeWebSocketManager
  ]);

  const disconnect = useCallback(() => {
    wsManager.current?.disconnect();
    updateState({ isConnected: false });
  }, [updateState]);

  const reconnect = useCallback(async (): Promise<boolean> => {
    disconnect();
    await new Promise(resolve => setTimeout(resolve, 1000));
    return await connect();
  }, [disconnect, connect]);

  // =================================================================
  // 🔧 메인 처리 메서드 (백엔드 패턴 호환)
  // =================================================================

  /**
   * 가상 피팅 처리 (백엔드 패턴 호환)
   */
  const processVirtualTryOn = useCallback(async (request: VirtualTryOnRequest) => {
    const timer = PipelineUtils.createPerformanceTimer('가상 피팅 전체 처리');
    
    try {
      // 서비스 초기화
      initializeAPIClient();
      
      // WebSocket 연결 확인
      if (!wsManager.current?.isConnected()) {
        PipelineUtils.log('info', '🔄 WebSocket 재연결 시도...');
        const connected = await connect();
        if (!connected) {
          PipelineUtils.log('warn', '⚠️ WebSocket 연결 실패, 진행률 업데이트 없이 계속 진행');
        }
      }

      // 세션 ID 생성 또는 재사용
      const sessionId = request.session_id || 
                       (config.persistSession && state.sessionId) || 
                       PipelineUtils.generateSessionId();

      // 처리 시작 상태 설정
      updateState({
        isProcessing: true,
        progress: 0,
        progressMessage: '처리를 시작합니다...',
        result: null,
        error: null,
        sessionId,
        steps: config.enableDetailedProgress ? [
          { id: 1, name: 'Human Parsing', status: 'pending', progress: 0 },
          { id: 2, name: 'Pose Estimation', status: 'pending', progress: 0 },
          { id: 3, name: 'Cloth Segmentation', status: 'pending', progress: 0 },
          { id: 4, name: 'Geometric Matching', status: 'pending', progress: 0 },
          { id: 5, name: 'Cloth Warping', status: 'pending', progress: 0 },
          { id: 6, name: 'Virtual Fitting', status: 'pending', progress: 0 },
          { id: 7, name: 'Post Processing', status: 'pending', progress: 0 },
          { id: 8, name: 'Quality Assessment', status: 'pending', progress: 0 }
        ] : []
      });

      PipelineUtils.log('info', '🎯 가상 피팅 처리 시작', { sessionId });

      // 세션 구독
      wsManager.current?.subscribeToSession(sessionId);

      // API 처리 (백엔드 통일된 인터페이스 사용)
      const result = await apiClient.current!.processVirtualTryOn({
        ...request,
        session_id: sessionId
      }, ...kwargs);

      const processingTime = timer.end();

      updateState({
        isProcessing: false,
        result,
        progress: 100,
        progressMessage: '완료!',
        sessionId: result.session_id || sessionId
      });

      PipelineUtils.log('info', '✅ 가상 피팅 처리 완료', { 
        processingTime: processingTime / 1000 
      });

    } catch (error) {
      timer.end();
      const errorMessage = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.';
      const friendlyError = PipelineUtils.getUserFriendlyError(errorMessage);
      
      updateState({
        isProcessing: false,
        error: friendlyError,
        progress: 0,
        progressMessage: ''
      });

      PipelineUtils.log('error', '❌ 가상 피팅 처리 실패', error);
    }
  }, [
    config, 
    connect, 
    state.sessionId, 
    updateState, 
    initializeAPIClient,
    kwargs
  ]);

  // =================================================================
  // 🔧 상태 관리 액션들
  // =================================================================

  const clearResult = useCallback(() => {
    updateState({
      result: null,
      progress: 0,
      progressMessage: '',
      steps: []
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
      error: null,
      steps: []
    });
  }, [updateState]);

  // =================================================================
  // 🔧 정보 조회 액션들
  // =================================================================

  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      initializeAPIClient();
      const isHealthy = await apiClient.current!.healthCheck();
      updateState({ isHealthy });
      return isHealthy;
    } catch (error) {
      PipelineUtils.log('error', '헬스체크 실패', error);
      updateState({ isHealthy: false });
      return false;
    }
  }, [updateState, initializeAPIClient]);

  const getPipelineStatus = useCallback(async () => {
    try {
      initializeAPIClient();
      const pipelineStatus = await apiClient.current!.getPipelineStatus();
      updateState({ pipelineStatus });
    } catch (error) {
      PipelineUtils.log('error', '파이프라인 상태 조회 실패', error);
    }
  }, [updateState, initializeAPIClient]);

  const getSystemStats = useCallback(async () => {
    try {
      initializeAPIClient();
      const systemStats = await apiClient.current!.getSystemStats();
      updateState({ systemStats });
    } catch (error) {
      PipelineUtils.log('error', '시스템 통계 조회 실패', error);
    }
  }, [updateState, initializeAPIClient]);

  // =================================================================
  // 🔧 파이프라인 관리 액션들
  // =================================================================

  const warmupPipeline = useCallback(async (qualityMode: string = 'balanced') => {
    try {
      updateState({
        isProcessing: true,
        progressMessage: '파이프라인 워밍업 중...'
      });

      initializeAPIClient();
      const success = await apiClient.current!.warmupPipeline(qualityMode);
      
      updateState({
        isProcessing: false,
        progressMessage: success ? '워밍업 완료' : '워밍업 실패'
      });

      PipelineUtils.log(
        success ? 'info' : 'error', 
        success ? '✅ 파이프라인 워밍업 완료' : '❌ 파이프라인 워밍업 실패'
      );

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '워밍업 실패';
      updateState({
        isProcessing: false,
        error: errorMessage,
        progressMessage: ''
      });

      PipelineUtils.log('error', '❌ 파이프라인 워밍업 실패', error);
    }
  }, [updateState, initializeAPIClient]);

  const testConnection = useCallback(async () => {
    try {
      updateState({
        isProcessing: true,
        progressMessage: '연결 테스트 중...'
      });

      const wsConnected = await connect();
      const healthOk = await checkHealth();

      if (!wsConnected) {
        throw new Error('WebSocket 연결 실패');
      }

      if (!healthOk) {
        throw new Error('API 헬스체크 실패');
      }

      updateState({
        isProcessing: false,
        progressMessage: '연결 테스트 완료',
        error: null
      });

      PipelineUtils.log('info', '✅ 연결 테스트 완료');

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '연결 테스트 실패';
      updateState({
        isProcessing: false,
        error: errorMessage,
        progressMessage: ''
      });

      PipelineUtils.log('error', '❌ 연결 테스트 실패', error);
    }
  }, [connect, checkHealth, updateState]);

  // =================================================================
  // 🔧 유틸리티 액션들
  // =================================================================

  const sendHeartbeat = useCallback(() => {
    wsManager.current?.send({ type: 'ping', timestamp: Date.now() });
  }, []);

  const exportLogs = useCallback(() => {
    const logs = {
      state,
      messageLog: messageLog.current,
      connectionAttempts: wsManager.current?.getReconnectAttempts() || 0,
      wsInfo: wsManager.current?.getWebSocketInfo(),
      clientInfo: apiClient.current?.getClientInfo(),
      config,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(logs, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pipeline_logs_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [state, config]);

  // =================================================================
  // 🔧 생명주기 관리 (백엔드 패턴 적용)
  // =================================================================

  /**
   * ✅ 통일된 초기화 (백엔드 패턴 호환)
   */
  const initializeHook = useCallback(async () => {
    if (isInitialized.current) return;
    
    PipelineUtils.log('info', '🔄 usePipeline Hook 초기화 중...');
    
    try {
      // 브라우저 호환성 확인
      const compatibility = PipelineUtils.checkBrowserCompatibility();
      if (!compatibility.overall) {
        PipelineUtils.log('error', '❌ 브라우저 호환성 부족', compatibility);
        updateState({
          error: '브라우저가 필요한 기능을 지원하지 않습니다.'
        });
        return;
      }
      
      // 서비스들 초기화
      initializeAPIClient();
      initializeWebSocketManager();
      
      // 자동 연결
      await connect();
      
      isInitialized.current = true;
      PipelineUtils.log('info', '✅ usePipeline Hook 초기화 완료');
      
    } catch (error) {
      PipelineUtils.log('error', '❌ usePipeline Hook 초기화 실패', error);
      updateState({
        error: PipelineUtils.getUserFriendlyError('initialization failed')
      });
    }
  }, [connect, initializeAPIClient, initializeWebSocketManager, updateState]);

  /**
   * ✅ 백엔드 패턴: 리소스 정리
   */
  const cleanupHook = useCallback(async () => {
    PipelineUtils.log('info', '🧹 usePipeline Hook: 리소스 정리 중...');
    
    try {
      // 헬스체크 타이머 정리
      if (healthCheckTimer.current) {
        clearInterval(healthCheckTimer.current);
        healthCheckTimer.current = null;
      }
      
      // 서비스들 정리
      await wsManager.current?.cleanup();
      await apiClient.current?.cleanup();
      
      // 연결 해제
      disconnect();
      
      // 상태 초기화
      isInitialized.current = false;
      
      PipelineUtils.log('info', '✅ usePipeline Hook 리소스 정리 완료');
    } catch (error) {
      PipelineUtils.log('warn', '⚠️ usePipeline Hook 리소스 정리 중 오류', error);
    }
  }, [disconnect]);

  // =================================================================
  // 🔧 Effect들
  // =================================================================

  // 초기화 Effect
  useEffect(() => {
    initializeHook();
    
    return () => {
      cleanupHook();
    };
  }, [initializeHook, cleanupHook]);

  // 자동 헬스체크 Effect
  useEffect(() => {
    if (config.autoHealthCheck) {
      checkHealth();

      healthCheckTimer.current = setInterval(checkHealth, config.healthCheckInterval);

      return () => {
        if (healthCheckTimer.current) {
          clearInterval(healthCheckTimer.current);
          healthCheckTimer.current = null;
        }
      };
    }
  }, [config.autoHealthCheck, config.healthCheckInterval, checkHealth]);

  // =================================================================
  // 🔧 Hook 반환값
  // =================================================================

  return {
    // 상태
    ...state,
    
    // 액션
    processVirtualTryOn,
    clearResult,
    clearError,
    reset,
    connect,
    disconnect,
    reconnect,
    checkHealth,
    getPipelineStatus,
    getSystemStats,
    warmupPipeline,
    testConnection,
    sendHeartbeat,
    exportLogs
  };
};

// =================================================================
// 🔧 편의 Hook들 (백엔드 패턴 호환)
// =================================================================

export const usePipelineStatus = (options: UsePipelineOptions = {}) => {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const apiClient = useMemo(() => new PipelineAPIClient(options), [options]);

  const fetchStatus = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const pipelineStatus = await apiClient.getPipelineStatus();
      setStatus(pipelineStatus);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '상태 조회 실패';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [apiClient]);

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

  const apiClient = useMemo(() => new PipelineAPIClient(options), [options]);

  const checkHealth = useCallback(async () => {
    setIsChecking(true);

    try {
      const healthy = await apiClient.healthCheck();
      setIsHealthy(healthy);
      setLastCheck(new Date());
    } catch (error) {
      setIsHealthy(false);
      PipelineUtils.log('error', '헬스체크 실패', error);
    } finally {
      setIsChecking(false);
    }
  }, [apiClient]);

  useEffect(() => {
    checkHealth();

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

// =================================================================
// 🔄 하위 호환성 지원 (백엔드 패턴 호환)
// =================================================================

export const createPipelineHook = (
  config: UsePipelineOptions = {}
) => {
  /**
   * 🔄 기존 팩토리 함수 호환 (기존 프론트엔드 호환)
   */
  return () => usePipeline(config);
};

export default usePipeline;