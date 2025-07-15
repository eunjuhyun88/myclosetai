/**
 * 파일명: frontend/src/services/WebSocketManager.ts
 * MyCloset AI WebSocket 관리자
 * 실제 백엔드 WebSocket과 완전 호환되는 프로덕션 수준 WebSocket 클라이언트
 * - 자동 재연결 및 상태 관리
 * - 실시간 파이프라인 진행률 수신
 * - 메시지 큐잉 및 재전송
 * - 연결 품질 모니터링
 */

import type {
  PipelineProgress,
  UsePipelineOptions,
  PipelineEvent,
  ProcessingStatusEnum,
  PipelineStep,
} from '../types/pipeline';
import { PipelineUtils } from '../utils/pipelineUtils';

// =================================================================
// 🔧 WebSocket 상태 및 설정 타입들
// =================================================================

export enum WebSocketState {
  DISCONNECTED = 0,
  CONNECTING = 1,
  CONNECTED = 2,
  RECONNECTING = 3,
  FAILED = 4,
  CLOSING = 5
}

export interface WebSocketConfig {
  url: string;
  protocols?: string[];
  autoReconnect: boolean;
  maxReconnectAttempts: number;
  reconnectInterval: number;
  heartbeatInterval: number;
  connectionTimeout: number;
  messageQueueSize: number;
  enableCompression: boolean;
  enableDebugMode: boolean;
}

export interface ConnectionMetrics {
  connectionAttempts: number;
  successfulConnections: number;
  failedConnections: number;
  totalReconnects: number;
  averageLatency: number;
  lastConnectionTime?: Date;
  lastDisconnectionTime?: Date;
  lastError?: string;
  dataTransferred: number;
  messagesReceived: number;
  messagesSent: number;
}

export interface QueuedMessage {
  id: string;
  data: any;
  timestamp: number;
  attempts: number;
  priority: number;
  callback?: (success: boolean, error?: Error) => void;
}

// =================================================================
// 🔧 메인 WebSocketManager 클래스
// =================================================================

export default class WebSocketManager {
  private ws: WebSocket | null = null;
  private config: WebSocketConfig;
  private state: WebSocketState = WebSocketState.DISCONNECTED;
  
  // 재연결 관리
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private connectionTimer: NodeJS.Timeout | null = null;
  
  // 하트비트 관리
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private lastHeartbeatReceived: number = 0;
  private heartbeatMissed = 0;
  private readonly maxHeartbeatMissed = 3;
  
  // 메시지 큐잉
  private messageQueue: QueuedMessage[] = [];
  private pendingMessages: Map<string, QueuedMessage> = new Map();
  
  // 메트릭 및 모니터링
  private metrics: ConnectionMetrics = {
    connectionAttempts: 0,
    successfulConnections: 0,
    failedConnections: 0,
    totalReconnects: 0,
    averageLatency: 0,
    dataTransferred: 0,
    messagesReceived: 0,
    messagesSent: 0
  };
  
  // 지연 측정
  private latencyMeasurements: number[] = [];
  private pingStart: number = 0;
  
  // 이벤트 핸들러들
  private onMessageCallback?: (data: PipelineProgress) => void;
  private onConnectedCallback?: () => void;
  private onDisconnectedCallback?: () => void;
  private onErrorCallback?: (error: any) => void;
  private onStateChangeCallback?: (state: WebSocketState) => void;
  private onMetricsUpdateCallback?: (metrics: ConnectionMetrics) => void;
  
  // 세션 관리
  private currentSessionId: string | null = null;
  private subscribedSessions: Set<string> = new Set();
  
  // 메시지 필터링 및 처리
  private messageFilters: Array<(message: any) => boolean> = [];
  private messageTransformers: Array<(message: any) => any> = [];

  constructor(url: string, options: UsePipelineOptions = {}, ...kwargs: any[]) {
    this.config = {
      url,
      protocols: options.wsProtocols || [],
      autoReconnect: options.autoReconnect ?? true,
      maxReconnectAttempts: options.maxReconnectAttempts || 10,
      reconnectInterval: options.reconnectInterval || 3000,
      heartbeatInterval: options.heartbeatInterval || 30000,
      connectionTimeout: options.connectionTimeout || 15000,
      messageQueueSize: options.messageQueueSize || 100,
      enableCompression: options.enableCompression ?? true,
      enableDebugMode: options.enableDebugMode ?? false,
    };
    
    // 추가 설정 병합
    this.mergeAdditionalConfig(kwargs);
    
    PipelineUtils.info('🔧 WebSocketManager 초기화', {
      url: this.config.url,
      autoReconnect: this.config.autoReconnect,
      maxAttempts: this.config.maxReconnectAttempts
    });
    
    // 정리 이벤트 등록
    if (typeof window !== 'undefined') {
      window.addEventListener('beforeunload', () => this.cleanup());
      window.addEventListener('online', () => this.handleNetworkChange(true));
      window.addEventListener('offline', () => this.handleNetworkChange(false));
    }
  }

  // =================================================================
  // 🔧 설정 관리
  // =================================================================

  private mergeAdditionalConfig(kwargs: any[]): void {
    for (const kwarg of kwargs) {
      if (typeof kwarg === 'object' && kwarg !== null) {
        Object.assign(this.config, kwarg);
      }
    }
  }

  updateConfig(newConfig: Partial<WebSocketConfig>): void {
    const wasConnected = this.isConnected();
    
    Object.assign(this.config, newConfig);
    
    if (newConfig.url && newConfig.url !== this.config.url) {
      PipelineUtils.info('🔄 WebSocket URL 변경', { 
        oldUrl: this.config.url, 
        newUrl: newConfig.url 
      });
      
      if (wasConnected) {
        this.disconnect();
        setTimeout(() => this.connect(), 1000);
      }
    }
    
    PipelineUtils.debug('⚙️ WebSocket 설정 업데이트', newConfig);
  }

  // =================================================================
  // 🔧 연결 관리
  // =================================================================

  async connect(): Promise<boolean> {
    if (this.state === WebSocketState.CONNECTING || this.state === WebSocketState.CONNECTED) {
      PipelineUtils.warn('⚠️ 이미 연결 중이거나 연결됨');
      return this.state === WebSocketState.CONNECTED;
    }

    this.setState(WebSocketState.CONNECTING);
    this.metrics.connectionAttempts++;

    try {
      await this.establishConnection();
      return true;
    } catch (error) {
      this.metrics.failedConnections++;
      this.handleConnectionError(error);
      
      if (this.config.autoReconnect && this.reconnectAttempts < this.config.maxReconnectAttempts) {
        this.scheduleReconnect();
      } else {
        this.setState(WebSocketState.FAILED);
      }
      
      return false;
    }
  }

  private async establishConnection(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // WebSocket 생성
        const protocols = this.config.protocols?.length ? this.config.protocols : undefined;
        this.ws = new WebSocket(this.config.url, protocols);
        
        // 바이너리 데이터 타입 설정
        this.ws.binaryType = 'arraybuffer';
        
        // 연결 타임아웃 설정
        const timeout = setTimeout(() => {
          this.ws?.close();
          reject(new Error('Connection timeout'));
        }, this.config.connectionTimeout);

        // 연결 성공
        this.ws.onopen = (event) => {
          clearTimeout(timeout);
          this.handleConnectionOpen(event);
          resolve();
        };

        // 메시지 수신
        this.ws.onmessage = (event) => {
          this.handleMessage(event);
        };

        // 연결 종료
        this.ws.onclose = (event) => {
          clearTimeout(timeout);
          this.handleConnectionClose(event);
          if (this.state === WebSocketState.CONNECTING) {
            reject(new Error(`Connection failed: ${event.code} ${event.reason}`));
          }
        };

        // 오류 발생
        this.ws.onerror = (event) => {
          clearTimeout(timeout);
          const error = new Error('WebSocket error occurred');
          this.handleConnectionError(error);
          if (this.state === WebSocketState.CONNECTING) {
            reject(error);
          }
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  disconnect(): void {
    this.setState(WebSocketState.CLOSING);
    this.config.autoReconnect = false;
    
    this.clearTimers();
    
    if (this.ws) {
      if (this.ws.readyState === WebSocket.OPEN) {
        // 정상적인 종료 메시지 전송
        this.sendMessage({
          type: 'disconnect',
          session_id: this.currentSessionId,
          timestamp: Date.now()
        }, false);
      }
      
      this.ws.close(1000, 'Normal closure');
      this.ws = null;
    }
    
    this.setState(WebSocketState.DISCONNECTED);
    PipelineUtils.info('🔌 WebSocket 연결 종료');
  }

  async reconnect(): Promise<boolean> {
    PipelineUtils.info('🔄 WebSocket 수동 재연결 시도');
    
    this.disconnect();
    await PipelineUtils.sleep(1000);
    this.config.autoReconnect = true;
    this.reconnectAttempts = 0;
    
    return await this.connect();
  }

  // =================================================================
  // 🔧 이벤트 핸들러들
  // =================================================================

  private handleConnectionOpen(event: Event): void {
    this.setState(WebSocketState.CONNECTED);
    this.metrics.successfulConnections++;
    this.metrics.lastConnectionTime = new Date();
    this.reconnectAttempts = 0;
    
    PipelineUtils.info('✅ WebSocket 연결 성공', {
      url: this.config.url,
      protocol: this.ws?.protocol,
      extensions: this.ws?.extensions
    });
    
    // 하트비트 시작
    this.startHeartbeat();
    
    // 큐된 메시지 전송
    this.processMessageQueue();
    
    // 세션 재구독
    this.resubscribeToSessions();
    
    // 연결 품질 테스트
    this.testConnectionQuality();
    
    // 콜백 실행
    this.onConnectedCallback?.();
    this.emitEvent('connected', { 
      attempt: this.metrics.connectionAttempts,
      metrics: { ...this.metrics }
    });
  }

  private handleConnectionClose(event: CloseEvent): void {
    const wasConnected = this.state === WebSocketState.CONNECTED;
    
    this.setState(WebSocketState.DISCONNECTED);
    this.metrics.lastDisconnectionTime = new Date();
    this.clearTimers();
    
    PipelineUtils.info('🔌 WebSocket 연결 종료', {
      code: event.code,
      reason: event.reason,
      wasClean: event.wasClean,
      wasConnected
    });
    
    // 연결이 예상치 못하게 끊어진 경우 재연결 시도
    if (wasConnected && this.config.autoReconnect && !event.wasClean) {
      this.scheduleReconnect();
    }
    
    // 콜백 실행
    this.onDisconnectedCallback?.();
    this.emitEvent('disconnected', {
      code: event.code,
      reason: event.reason,
      wasClean: event.wasClean
    });
  }

  private handleConnectionError(error: any): void {
    this.metrics.lastError = error?.message || 'Unknown error';
    
    PipelineUtils.error('❌ WebSocket 오류', {
      error: error?.message,
      state: WebSocketState[this.state],
      attempts: this.reconnectAttempts
    });
    
    // 콜백 실행
    this.onErrorCallback?.(error);
    this.emitEvent('error', { 
      error: error?.message,
      state: this.state,
      attempts: this.reconnectAttempts
    });
  }

  private handleMessage(event: MessageEvent): void {
    try {
      this.metrics.messagesReceived++;
      this.metrics.dataTransferred += this.getMessageSize(event.data);
      
      let data: any;
      
      // 메시지 데이터 파싱
      if (typeof event.data === 'string') {
        data = JSON.parse(event.data);
      } else if (event.data instanceof ArrayBuffer) {
        // 바이너리 데이터 처리 (필요한 경우)
        const decoder = new TextDecoder();
        data = JSON.parse(decoder.decode(event.data));
      } else {
        throw new Error('Unsupported message format');
      }
      
      if (this.config.enableDebugMode) {
        PipelineUtils.debug('📨 WebSocket 메시지 수신', data);
      }
      
      // 특별한 메시지 타입 처리
      if (this.handleSpecialMessage(data)) {
        return;
      }
      
      // 메시지 필터링
      if (!this.applyMessageFilters(data)) {
        return;
      }
      
      // 메시지 변환
      data = this.applyMessageTransformers(data);
      
      // 메인 메시지 콜백 실행
      this.onMessageCallback?.(data);
      this.emitEvent('message', data);
      
    } catch (error) {
      PipelineUtils.error('❌ 메시지 처리 오류', {
        error: error?.message,
        rawData: event.data
      });
      
      this.emitEvent('message_error', {
        error: error?.message,
        rawData: event.data
      });
    }
  }

  private handleSpecialMessage(data: any): boolean {
    switch (data.type) {
      case 'pong':
        this.handlePongMessage(data);
        return true;
        
      case 'heartbeat':
        this.handleHeartbeatMessage(data);
        return true;
        
      case 'connection_established':
        this.handleConnectionAck(data);
        return true;
        
      case 'subscription_ack':
        this.handleSubscriptionAck(data);
        return true;
        
      case 'error':
        this.handleServerError(data);
        return false; // 에러는 메인 콜백으로도 전달
        
      default:
        return false;
    }
  }

  // =================================================================
  // 🔧 메시지 전송 및 큐잉
  // =================================================================

  send(data: any, priority: number = 0): Promise<boolean> {
    return new Promise((resolve, reject) => {
      const message: QueuedMessage = {
        id: this.generateMessageId(),
        data,
        timestamp: Date.now(),
        attempts: 0,
        priority,
        callback: (success, error) => {
          if (success) {
            resolve(true);
          } else {
            reject(error || new Error('Message sending failed'));
          }
        }
      };
      
      if (this.isConnected()) {
        this.sendMessageDirect(message);
      } else {
        this.queueMessage(message);
        resolve(true); // 큐에 추가는 성공
      }
    });
  }

  sendMessage(data: any, queue: boolean = true): boolean {
    const message: QueuedMessage = {
      id: this.generateMessageId(),
      data,
      timestamp: Date.now(),
      attempts: 0,
      priority: 0
    };
    
    if (this.isConnected()) {
      return this.sendMessageDirect(message);
    } else if (queue) {
      this.queueMessage(message);
      return true;
    } else {
      PipelineUtils.warn('⚠️ WebSocket 연결되지 않음 - 메시지 무시됨');
      return false;
    }
  }

  private sendMessageDirect(message: QueuedMessage): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return false;
    }
    
    try {
      const payload = JSON.stringify(message.data);
      this.ws.send(payload);
      
      this.metrics.messagesSent++;
      this.metrics.dataTransferred += payload.length;
      message.attempts++;
      
      if (this.config.enableDebugMode) {
        PipelineUtils.debug('📤 WebSocket 메시지 전송', message.data);
      }
      
      // 메시지 전송 성공 콜백
      message.callback?.(true);
      
      return true;
      
    } catch (error) {
      PipelineUtils.error('❌ 메시지 전송 실패', {
        error: error?.message,
        messageId: message.id
      });
      
      message.callback?.(false, error as Error);
      return false;
    }
  }

  private queueMessage(message: QueuedMessage): void {
    // 큐 크기 제한 확인
    if (this.messageQueue.length >= this.config.messageQueueSize) {
      // 우선순위가 낮은 메시지 제거
      this.messageQueue.sort((a, b) => b.priority - a.priority);
      const removed = this.messageQueue.pop();
      if (removed) {
        PipelineUtils.warn('⚠️ 메시지 큐 오버플로우 - 메시지 제거됨', { messageId: removed.id });
        removed.callback?.(false, new Error('Queue overflow'));
      }
    }
    
    this.messageQueue.push(message);
    this.messageQueue.sort((a, b) => b.priority - a.priority);
    
    PipelineUtils.debug('📥 메시지 큐에 추가됨', {
      messageId: message.id,
      queueSize: this.messageQueue.length
    });
  }

  private processMessageQueue(): void {
    if (this.messageQueue.length === 0) return;
    
    PipelineUtils.info('📤 큐된 메시지 전송 시작', {
      count: this.messageQueue.length
    });
    
    const messages = [...this.messageQueue];
    this.messageQueue = [];
    
    for (const message of messages) {
      if (!this.sendMessageDirect(message)) {
        // 전송 실패 시 다시 큐에 추가
        this.queueMessage(message);
      }
    }
  }

  // =================================================================
  // 🔧 세션 및 구독 관리
  // =================================================================

  subscribeToSession(sessionId: string): void {
    this.currentSessionId = sessionId;
    this.subscribedSessions.add(sessionId);
    
    const subscribeMessage = {
      type: 'subscribe',
      session_id: sessionId,
      timestamp: Date.now(),
      client_info: this.getClientInfo()
    };
    
    this.sendMessage(subscribeMessage);
    
    PipelineUtils.info('🔔 세션 구독', { sessionId });
  }

  unsubscribeFromSession(sessionId: string): void {
    this.subscribedSessions.delete(sessionId);
    
    if (this.currentSessionId === sessionId) {
      this.currentSessionId = null;
    }
    
    const unsubscribeMessage = {
      type: 'unsubscribe',
      session_id: sessionId,
      timestamp: Date.now()
    };
    
    this.sendMessage(unsubscribeMessage);
    
    PipelineUtils.info('🔕 세션 구독 해제', { sessionId });
  }

  private resubscribeToSessions(): void {
    for (const sessionId of this.subscribedSessions) {
      this.subscribeToSession(sessionId);
    }
  }

  // =================================================================
  // 🔧 하트비트 및 연결 품질 관리
  // =================================================================

  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    if (this.config.heartbeatInterval > 0) {
      this.heartbeatTimer = setInterval(() => {
        this.sendHeartbeat();
      }, this.config.heartbeatInterval);
      
      this.lastHeartbeatReceived = Date.now();
    }
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private sendHeartbeat(): void {
    if (!this.isConnected()) return;
    
    this.pingStart = performance.now();
    
    const heartbeatMessage = {
      type: 'ping',
      timestamp: Date.now(),
      session_id: this.currentSessionId
    };
    
    if (!this.sendMessage(heartbeatMessage, false)) {
      this.heartbeatMissed++;
      
      if (this.heartbeatMissed >= this.maxHeartbeatMissed) {
        PipelineUtils.warn('⚠️ 하트비트 실패 - 연결 재시도');
        this.handleConnectionTimeout();
      }
    }
  }

  private handlePongMessage(data: any): void {
    if (this.pingStart > 0) {
      const latency = performance.now() - this.pingStart;
      this.recordLatency(latency);
      this.pingStart = 0;
    }
    
    this.lastHeartbeatReceived = Date.now();
    this.heartbeatMissed = 0;
  }

  private handleHeartbeatMessage(data: any): void {
    this.lastHeartbeatReceived = Date.now();
    this.heartbeatMissed = 0;
    
    // 하트비트 응답
    const response = {
      type: 'heartbeat_ack',
      timestamp: Date.now(),
      session_id: this.currentSessionId
    };
    
    this.sendMessage(response, false);
  }

  private handleConnectionTimeout(): void {
    PipelineUtils.error('❌ 연결 타임아웃 감지');
    
    if (this.ws) {
      this.ws.close(1002, 'Connection timeout');
    }
  }

  // =================================================================
  // 🔧 연결 품질 및 성능 모니터링
  // =================================================================

  private testConnectionQuality(): void {
    // 연결 품질 테스트를 위한 ping 전송
    for (let i = 0; i < 3; i++) {
      setTimeout(() => {
        this.sendHeartbeat();
      }, i * 1000);
    }
  }

  private recordLatency(latency: number): void {
    this.latencyMeasurements.push(latency);
    
    // 최대 50개 측정값만 유지
    if (this.latencyMeasurements.length > 50) {
      this.latencyMeasurements.shift();
    }
    
    // 평균 지연시간 계산
    this.metrics.averageLatency = this.latencyMeasurements.reduce((a, b) => a + b, 0) / this.latencyMeasurements.length;
    
    this.onMetricsUpdateCallback?.(this.metrics);
  }

  getConnectionQuality(): {
    state: string;
    latency: number;
    stability: number;
    throughput: number;
    health: 'excellent' | 'good' | 'fair' | 'poor';
  } {
    const now = Date.now();
    const stability = this.metrics.successfulConnections / Math.max(this.metrics.connectionAttempts, 1);
    const recentLatency = this.latencyMeasurements.slice(-10);
    const avgLatency = recentLatency.length > 0 ? 
      recentLatency.reduce((a, b) => a + b, 0) / recentLatency.length : 0;
    
    // 처리량 계산 (메시지/초)
    const connectionDuration = this.metrics.lastConnectionTime ? 
      (now - this.metrics.lastConnectionTime.getTime()) / 1000 : 1;
    const throughput = this.metrics.messagesReceived / Math.max(connectionDuration, 1);
    
    // 전반적인 상태 평가
    let health: 'excellent' | 'good' | 'fair' | 'poor' = 'poor';
    if (avgLatency < 100 && stability > 0.95 && throughput > 0.1) {
      health = 'excellent';
    } else if (avgLatency < 300 && stability > 0.8 && throughput > 0.05) {
      health = 'good';
    } else if (avgLatency < 1000 && stability > 0.5) {
      health = 'fair';
    }
    
    return {
      state: WebSocketState[this.state],
      latency: Math.round(avgLatency),
      stability: Math.round(stability * 100) / 100,
      throughput: Math.round(throughput * 100) / 100,
      health
    };
  }

  // =================================================================
  // 🔧 재연결 로직
  // =================================================================

  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    
    this.setState(WebSocketState.RECONNECTING);
    this.reconnectAttempts++;
    this.metrics.totalReconnects++;
    
    // 지수 백오프 적용
    const baseDelay = this.config.reconnectInterval;
    const backoffMultiplier = Math.min(Math.pow(2, this.reconnectAttempts - 1), 8);
    const jitter = Math.random() * 1000;
    const delay = baseDelay * backoffMultiplier + jitter;
    
    PipelineUtils.info('🔄 재연결 예약됨', {
      attempt: this.reconnectAttempts,
      maxAttempts: this.config.maxReconnectAttempts,
      delay: Math.round(delay)
    });
    
    this.reconnectTimer = setTimeout(async () => {
      try {
        await this.connect();
      } catch (error) {
        PipelineUtils.error('❌ 재연결 실패', error);
      }
    }, delay);
    
    this.emitEvent('reconnecting', {
      attempt: this.reconnectAttempts,
      maxAttempts: this.config.maxReconnectAttempts,
      delay
    });
  }

  // =================================================================
  // 🔧 네트워크 상태 관리
  // =================================================================

  private handleNetworkChange(isOnline: boolean): void {
    PipelineUtils.info(`🌐 네트워크 상태 변경: ${isOnline ? 'Online' : 'Offline'}`);
    
    if (isOnline && this.state === WebSocketState.DISCONNECTED && this.config.autoReconnect) {
      // 네트워크가 복구되면 재연결 시도
      setTimeout(() => this.connect(), 2000);
    } else if (!isOnline && this.isConnected()) {
      // 네트워크가 끊어지면 연결 상태 업데이트
      this.setState(WebSocketState.DISCONNECTED);
    }
    
    this.emitEvent('network_change', { isOnline });
  }

  // =================================================================
  // 🔧 메시지 필터링 및 변환
  // =================================================================

  addMessageFilter(filter: (message: any) => boolean): void {
    this.messageFilters.push(filter);
  }

  removeMessageFilter(filter: (message: any) => boolean): void {
    const index = this.messageFilters.indexOf(filter);
    if (index > -1) {
      this.messageFilters.splice(index, 1);
    }
  }

  addMessageTransformer(transformer: (message: any) => any): void {
    this.messageTransformers.push(transformer);
  }

  removeMessageTransformer(transformer: (message: any) => any): void {
    const index = this.messageTransformers.indexOf(transformer);
    if (index > -1) {
      this.messageTransformers.splice(index, 1);
    }
  }

  private applyMessageFilters(message: any): boolean {
    return this.messageFilters.every(filter => {
      try {
        return filter(message);
      } catch (error) {
        PipelineUtils.warn('⚠️ 메시지 필터 오류', error);
        return true; // 필터 오류 시 메시지 통과
      }
    });
  }

  private applyMessageTransformers(message: any): any {
    return this.messageTransformers.reduce((msg, transformer) => {
      try {
        return transformer(msg);
      } catch (error) {
        PipelineUtils.warn('⚠️ 메시지 변환 오류', error);
        return msg; // 변환 오류 시 원본 반환
      }
    }, message);
  }

  // =================================================================
  // 🔧 이벤트 핸들러 설정
  // =================================================================

  setOnMessage(callback: (data: PipelineProgress) => void): void {
    this.onMessageCallback = callback;
  }

  setOnConnected(callback: () => void): void {
    this.onConnectedCallback = callback;
  }

  setOnDisconnected(callback: () => void): void {
    this.onDisconnectedCallback = callback;
  }

  setOnError(callback: (error: any) => void): void {
    this.onErrorCallback = callback;
  }

  setOnStateChange(callback: (state: WebSocketState) => void): void {
    this.onStateChangeCallback = callback;
  }

  setOnMetricsUpdate(callback: (metrics: ConnectionMetrics) => void): void {
    this.onMetricsUpdateCallback = callback;
  }

  // =================================================================
  // 🔧 상태 조회 및 정보 메서드들
  // =================================================================

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN && this.state === WebSocketState.CONNECTED;
  }

  isConnecting(): boolean {
    return this.state === WebSocketState.CONNECTING;
  }

  isReconnecting(): boolean {
    return this.state === WebSocketState.RECONNECTING;
  }

  getState(): WebSocketState {
    return this.state;
  }

  getStateString(): string {
    return WebSocketState[this.state];
  }

  private setState(newState: WebSocketState): void {
    if (this.state !== newState) {
      const oldState = this.state;
      this.state = newState;
      
      PipelineUtils.debug('🔄 WebSocket 상태 변경', {
        from: WebSocketState[oldState],
        to: WebSocketState[newState]
      });
      
      this.onStateChangeCallback?.(newState);
      this.emitEvent('state_change', {
        from: oldState,
        to: newState
      });
    }
  }

  getMetrics(): ConnectionMetrics {
    return { ...this.metrics };
  }

  getReconnectAttempts(): number {
    return this.reconnectAttempts;
  }

  getWebSocketInfo(): any {
    return {
      url: this.config.url,
      state: WebSocketState[this.state],
      readyState: this.ws?.readyState,
      protocol: this.ws?.protocol,
      extensions: this.ws?.extensions,
      reconnectAttempts: this.reconnectAttempts,
      queueSize: this.messageQueue.length,
      subscribedSessions: Array.from(this.subscribedSessions),
      currentSessionId: this.currentSessionId,
      connectionQuality: this.getConnectionQuality(),
      config: { ...this.config },
      metrics: { ...this.metrics }
    };
  }

  getClientInfo(): any {
    return {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      language: navigator.language,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      screen: `${screen.width}x${screen.height}`,
      deviceMemory: (navigator as any).deviceMemory,
      hardwareConcurrency: navigator.hardwareConcurrency,
      timestamp: Date.now()
    };
  }

  // =================================================================
  // 🔧 유틸리티 메서드들
  // =================================================================

  private generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private getMessageSize(data: any): number {
    if (typeof data === 'string') {
      return data.length;
    } else if (data instanceof ArrayBuffer) {
      return data.byteLength;
    } else {
      return JSON.stringify(data).length;
    }
  }

  private clearTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    if (this.connectionTimer) {
      clearTimeout(this.connectionTimer);
      this.connectionTimer = null;
    }
    
    this.stopHeartbeat();
  }

  private handleConnectionAck(data: any): void {
    PipelineUtils.info('✅ 연결 확인 수신', data);
    this.emitEvent('connection_ack', data);
  }

  private handleSubscriptionAck(data: any): void {
    PipelineUtils.info('✅ 구독 확인 수신', { sessionId: data.session_id });
    this.emitEvent('subscription_ack', data);
  }

  private handleServerError(data: any): void {
    PipelineUtils.error('❌ 서버 오류 수신', data);
    this.emitEvent('server_error', data);
  }

  private emitEvent(type: string, data?: any): void {
    const event: PipelineEvent = {
      type,
      data,
      timestamp: Date.now(),
      source: 'websocket'
    };
    
    PipelineUtils.emitEvent(`websocket:${type}`, event);
  }

  // =================================================================
  // 🔧 정리 및 종료
  // =================================================================

  async cleanup(): Promise<void> {
    PipelineUtils.info('🧹 WebSocketManager 리소스 정리 중...');
    
    this.config.autoReconnect = false;
    this.clearTimers();
    
    // 큐된 메시지들에 대한 콜백 실행
    for (const message of this.messageQueue) {
      message.callback?.(false, new Error('WebSocket cleanup'));
    }
    this.messageQueue = [];
    
    for (const [id, message] of this.pendingMessages) {
      message.callback?.(false, new Error('WebSocket cleanup'));
    }
    this.pendingMessages.clear();
    
    // WebSocket 연결 종료
    if (this.ws) {
      this.ws.close(1001, 'Going away');
      this.ws = null;
    }
    
    // 이벤트 리스너 정리
    this.messageFilters = [];
    this.messageTransformers = [];
    this.subscribedSessions.clear();
    
    this.setState(WebSocketState.DISCONNECTED);
    
    PipelineUtils.info('✅ WebSocketManager 리소스 정리 완료');
  }
}