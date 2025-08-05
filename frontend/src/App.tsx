import React, { useState, useRef, useEffect, useCallback } from 'react';
// 🔥 기존 import 구문들 아래에 추가
// 🔥 VirtualFittingResultVisualization 컴포넌트 추가
const VirtualFittingResultVisualization = ({ result }: { result: TryOnResult }) => {
  const [activeTab, setActiveTab] = useState('result');
  
  return (
    <div className="bg-white rounded-xl shadow-xl p-6" style={{
      backgroundColor: '#ffffff',
      borderRadius: '0.75rem',
      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
      padding: '1.5rem'
    }}>
      {/* 탭 헤더 */}
      <div style={{ borderBottom: '1px solid #e5e7eb', marginBottom: '1.5rem' }}>
        <h2 style={{
          fontSize: '1.875rem',
          fontWeight: '700',
          color: '#111827',
          marginBottom: '1rem',
          textAlign: 'center'
        }}>🎭 AI 가상 피팅 결과</h2>
        
        <div style={{ display: 'flex', gap: '0.25rem', flexWrap: 'wrap' }}>
          {[
            { id: 'result', label: '최종 결과', icon: '🎯', desc: '피팅된 이미지와 점수' },
            { id: 'process', label: 'AI 처리과정', icon: '⚙️', desc: '14GB 모델 처리 단계' },
            { id: 'analysis', label: '상세 분석', icon: '📊', desc: '키포인트와 비교 분석' },
            { id: 'quality', label: '품질 평가', icon: '⭐', desc: 'AI 품질 메트릭' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '0.75rem 1rem',
                borderRadius: '0.5rem 0.5rem 0 0',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                fontSize: '0.875rem',
                fontWeight: '500',
                transition: 'all 0.2s',
                border: 'none',
                cursor: 'pointer',
                backgroundColor: activeTab === tab.id ? '#3b82f6' : '#f8fafc',
                color: activeTab === tab.id ? '#ffffff' : '#64748b',
                borderBottom: activeTab === tab.id ? '2px solid #3b82f6' : '2px solid transparent'
              }}
              title={tab.desc}
            >
              <span style={{ fontSize: '1.125rem' }}>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>
      </div>
      
      {/* 탭 컨텐츠 */}
      <div style={{ minHeight: '32rem' }}>
        {activeTab === 'result' && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: window.innerWidth < 1024 ? '1fr' : '2fr 1fr',
            gap: '2rem'
          }}>
            {/* 메인 결과 이미지 */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ position: 'relative' }}>
                <img 
                  src={result.fitted_image?.startsWith('data:') ? result.fitted_image : `data:image/jpeg;base64,${result.fitted_image}`}
                  alt="Virtual Fitting Result"
                  style={{
                    width: '100%',
                    borderRadius: '1rem',
                    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
                    border: '3px solid #10b981'
                  }}
                  onError={(e) => {
                    console.error('이미지 로드 실패:', e);
                    const target = e.target as HTMLImageElement;
                    target.style.display = 'none';
                  }}
                />
                <div style={{
                  position: 'absolute',
                  top: '1rem',
                  right: '1rem',
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  borderRadius: '0.5rem',
                  padding: '0.5rem 0.75rem',
                  backdropFilter: 'blur(10px)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div style={{
                      width: '0.75rem',
                      height: '0.75rem',
                      backgroundColor: '#10b981',
                      borderRadius: '50%',
                      animation: 'pulse 2s infinite'
                    }}></div>
                    <span style={{ fontSize: '0.875rem', fontWeight: '500', color: '#374151' }}>
                      AI Generated
                    </span>
                  </div>
                </div>
              </div>
              
              {/* 처리 시간과 신뢰도 */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div style={{
                  background: 'linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)',
                  padding: '1rem',
                  borderRadius: '0.75rem'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ color: '#1e40af', fontWeight: '500' }}>⏱️ 처리 시간</span>
                    <span style={{ fontSize: '1.5rem', fontWeight: '700', color: '#1d4ed8' }}>
                      {result.processing_time?.toFixed(1)}초
                    </span>
                  </div>
                </div>
                <div style={{
                  background: 'linear-gradient(135deg, #f3e8ff 0%, #ddd6fe 100%)',
                  padding: '1rem',
                  borderRadius: '0.75rem'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ color: '#7c3aed', fontWeight: '500' }}>🎯 신뢰도</span>
                    <span style={{ fontSize: '1.5rem', fontWeight: '700', color: '#6d28d9' }}>
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* 사이드 패널 */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {/* Fit Score */}
              <div style={{
                background: 'linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)',
                padding: '1.5rem',
                borderRadius: '1rem',
                border: '1px solid #a7f3d0'
              }}>
                <h3 style={{
                  fontSize: '1.125rem',
                  fontWeight: '600',
                  color: '#065f46',
                  marginBottom: '1rem',
                  textAlign: 'center'
                }}>🏆 피팅 점수</h3>
                <div style={{ textAlign: 'center' }}>
                  <div style={{
                    fontSize: '3rem',
                    fontWeight: '800',
                    color: '#059669',
                    marginBottom: '0.5rem',
                    textShadow: '0 2px 4px rgba(0,0,0,0.1)'
                  }}>
                    {(result.fit_score * 100).toFixed(1)}%
                  </div>
                  <div style={{
                    width: '100%',
                    backgroundColor: '#a7f3d0',
                    borderRadius: '9999px',
                    height: '0.75rem',
                    marginBottom: '1rem',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      background: 'linear-gradient(90deg, #10b981 0%, #059669 100%)',
                      height: '0.75rem',
                      borderRadius: '9999px',
                      transition: 'width 1s ease-out',
                      width: `${result.fit_score * 100}%`,
                      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                    }}></div>
                  </div>
                  <p style={{
                    fontSize: '0.875rem',
                    color: '#047857',
                    fontWeight: '500'
                  }}>
                    {result.fit_score > 0.9 ? '🌟 완벽한 핏!' : 
                     result.fit_score > 0.8 ? '✨ 매우 좋은 핏' : 
                     result.fit_score > 0.7 ? '👍 좋은 핏' : '👌 보통 핏'}
                  </p>
                </div>
              </div>
              
              {/* AI 추천사항 */}
              <div style={{
                backgroundColor: '#f8fafc',
                padding: '1.5rem',
                borderRadius: '1rem',
                border: '1px solid #e2e8f0'
              }}>
                <h3 style={{
                  fontSize: '1.125rem',
                  fontWeight: '600',
                  color: '#1e293b',
                  marginBottom: '1rem'
                }}>💡 AI 추천</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {result.recommendations?.slice(0, 3).map((rec, idx) => (
                    <div key={idx} style={{
                      backgroundColor: '#ffffff',
                      padding: '0.75rem',
                      borderRadius: '0.5rem',
                      boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
                      borderLeft: '4px solid #3b82f6'
                    }}>
                      <p style={{
                        fontSize: '0.875rem',
                        color: '#374151',
                        margin: 0,
                        lineHeight: '1.4'
                      }}>{rec}</p>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* 사용된 AI 모델 정보 */}
              <div style={{
                background: 'linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%)',
                padding: '1.5rem',
                borderRadius: '1rem',
                border: '1px solid #c7d2fe'
              }}>
                <h3 style={{
                  fontSize: '1.125rem',
                  fontWeight: '600',
                  color: '#3730a3',
                  marginBottom: '1rem'
                }}>🧠 AI 모델</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {[
                    { label: '모델', value: 'OOTDiffusion 14GB' },
                    { label: '디바이스', value: 'MPS (M3 Max)' },
                    { label: '해상도', value: '512x512' },
                    { label: '품질', value: 'High Quality' }
                  ].map((item, idx) => (
                    <div key={idx} style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      fontSize: '0.875rem'
                    }}>
                      <span style={{ color: '#4338ca', fontWeight: '500' }}>{item.label}:</span>
                      <span style={{ color: '#1e1b4b', fontWeight: '600' }}>{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* 다른 탭들 */}
        {activeTab === 'process' && (
          <div style={{ textAlign: 'center', padding: '3rem' }}>
            <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>⚙️</div>
            <h3 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#374151', marginBottom: '0.5rem' }}>
              AI 처리 과정
            </h3>
            <p style={{ color: '#6b7280' }}>
              14GB OOTDiffusion 모델의 실제 처리 단계가 여기에 표시됩니다
            </p>
          </div>
        )}
        
        {activeTab === 'analysis' && (
          <div style={{ textAlign: 'center', padding: '3rem' }}>
            <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>📊</div>
            <h3 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#374151', marginBottom: '0.5rem' }}>
              상세 분석
            </h3>
            <p style={{ color: '#6b7280' }}>
              키포인트 분석과 비교 결과가 여기에 표시됩니다
            </p>
          </div>
        )}
        
        {activeTab === 'quality' && (
          <div style={{ textAlign: 'center', padding: '3rem' }}>
            <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>⭐</div>
            <h3 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#374151', marginBottom: '0.5rem' }}>
              품질 평가
            </h3>
            <p style={{ color: '#6b7280' }}>
              AI 품질 메트릭과 성능 분석이 여기에 표시됩니다
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

// =================================================================
// 🔧 완전한 API 클라이언트 (모든 기능 포함 + 오류 수정)
// =================================================================

interface APIClientConfig {
  baseURL: string;
  wsURL?: string;
  apiKey?: string;
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
  enableCaching: boolean;
  cacheTimeout: number;
  enableCompression: boolean;
  enableRetry: boolean;
  uploadChunkSize: number;
  maxConcurrentRequests: number;
  requestQueueSize: number;
  enableMetrics: boolean;
  enableDebug: boolean;
  enableWebSocket: boolean;
  heartbeatInterval: number;
  reconnectInterval: number;
  maxReconnectAttempts: number;
}

interface RequestMetrics {
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageResponseTime: number;
  totalBytesTransferred: number;
  cacheHitRate: number;
  retryRate: number;
  errorBreakdown: Record<string, number>;
  uptime: number;
  lastError?: string;
  lastErrorTime?: number;
}

interface CacheEntry<T = any> {
  data: T;
  timestamp: number;
  expiry: number;
  size: number;
  hits: number;
  etag?: string;
}

interface QueuedRequest {
  id: string;
  url: string;
  options: RequestInit;
  resolve: (value: any) => void;
  reject: (reason: any) => void;
  priority: number;
  attempts: number;
  timestamp: number;
  maxRetries: number;
}

interface VirtualTryOnRequest {
  person_image: File;
  clothing_image: File;
  height: number;
  weight: number;
  chest?: number;
  waist?: number;
  hip?: number;
  shoulder_width?: number;
  clothing_type?: string;
  fabric_type?: string;
  style_preference?: string;
  quality_mode?: 'low' | 'balanced' | 'high';
  session_id?: string;
  enable_realtime?: boolean;
  save_intermediate?: boolean;
  pose_adjustment?: boolean;
  color_preservation?: boolean;
  texture_enhancement?: boolean;
}

interface VirtualTryOnResponse {
  success: boolean;
  fitted_image: string;
  confidence: number;
  fit_score: number;
  processing_time: number;
  session_id: string;
  recommendations: string[];
  details: Record<string, any>;
  metadata?: {
    step_name: string;
    device: string;
    timestamp: string;
    unified_service_manager?: boolean;
  };
}

interface PipelineProgress {
  type: string;
  progress: number;
  message: string;
  timestamp: number;
  step_id?: number;
  step_name?: string;
}

// =================================================================
// 🔧 유틸리티 클래스 (PipelineUtils)
// =================================================================

class PipelineUtils {
  static info(message: string, data?: any): void {
    console.log(`ℹ️ ${message}`, data ? data : '');
  }

  static warn(message: string, data?: any): void {
    console.warn(`⚠️ ${message}`, data ? data : '');
  } 

  static error(message: string, data?: any): void {
    console.error(`❌ ${message}`, data ? data : '');
  }

  static debug(message: string, data?: any): void {
    console.log(`🐛 ${message}`, data ? data : '');
  }

  static createPerformanceTimer(label: string) {
    const startTime = performance.now();
    return {
      end: () => performance.now() - startTime,
      label
    };
  }

  static validateImageType(file: File): boolean {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    return validTypes.includes(file.type);
  }

  static validateFileSize(file: File, maxMB: number): boolean {
    return file.size <= maxMB * 1024 * 1024;
  }

  static formatBytes(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  static sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  static autoDetectDevice(): string {
    if (typeof navigator !== 'undefined') {
      const userAgent = navigator.userAgent;
      if (userAgent.includes('Mac')) return 'mps';
      if (userAgent.includes('NVIDIA')) return 'cuda';
    }
    return 'cpu';
  }

  static getSystemParams(): Map<string, any> {
    return new Map([
      ['client_version', '2.0.0'],
      ['user_agent', navigator.userAgent],
      ['platform', navigator.platform],
      ['language', navigator.language],
      ['hardware_concurrency', navigator.hardwareConcurrency],
      ['device_memory', (navigator as any).deviceMemory || 'unknown'],
      ['connection', (navigator as any).connection?.effectiveType || 'unknown']
    ]);
  }

  static getUserFriendlyError(error: any): string {
    if (typeof error === 'string') return error;
    if (error?.message) return error.message;
    if (error?.detail) return error.detail;
    return 'Unknown error occurred';
  }

  static emitEvent(eventName: string, data?: any): void {
    const event = new CustomEvent(eventName, { detail: data });
    window.dispatchEvent(event);
  }
}

// =================================================================
// 🔧 WebSocket 관리자 클래스 (완전한 기능형)
// =================================================================

class EnhancedWebSocketManager {
  private ws: WebSocket | null = null;
  private url: string;
  private protocols?: string[];
  private isConnecting = false;
  private isDestroyed = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private connectionTimeout: NodeJS.Timeout | null = null;
  private messageQueue: Array<{ data: any; timestamp: number }> = [];
  private subscriptions = new Set<string>();
  
  private messageHandlers = new Map<string, Function[]>();
  private eventHandlers = new Map<string, Function[]>();
  
  private latencyMeasurements: number[] = [];
  private lastPingTime = 0;
  private connectionQuality = 0;
  private totalReconnects = 0;
  private lastDisconnectTime = 0;

  constructor(url: string, options: Partial<APIClientConfig> = {}) {
    this.url = url;
    this.protocols = options.wsURL ? [options.wsURL] : undefined;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 10;
    
    console.log('🔧 EnhancedWebSocketManager 생성:', url);
  }

  onMessage(type: string, handler: Function): void {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, []);
    }
    this.messageHandlers.get(type)!.push(handler);
  }

  onEvent(event: string, handler: Function): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)!.push(handler);
  }

  async connect(): Promise<boolean> {
    if (this.isDestroyed) return false;
    if (this.isConnected()) return true;
    if (this.isConnecting) return false;

    this.isConnecting = true;
    console.log('🔗 WebSocket 연결 시도:', this.url);

    try {
      this.ws = new WebSocket(this.url, this.protocols);
      
      return new Promise((resolve) => {
        if (!this.ws || this.isDestroyed) {
          this.isConnecting = false;
          resolve(false);
          return;
        }

        this.connectionTimeout = setTimeout(() => {
          console.log('⏰ WebSocket 연결 타임아웃');
          this.ws?.close();
          this.isConnecting = false;
          resolve(false);
        }, 15000);

        this.ws.onopen = () => {
          if (this.isDestroyed) return;
          
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          
          if (this.connectionTimeout) {
            clearTimeout(this.connectionTimeout);
            this.connectionTimeout = null;
          }
          
          console.log('✅ WebSocket 연결 성공');
          this.startHeartbeat();
          this.processMessageQueue();
          this.resubscribeAll();
          this.emitEvent('connected');
          resolve(true);
        };

        this.ws.onmessage = (event) => {
          if (this.isDestroyed) return;
          this.handleMessage(event);
        };

        this.ws.onclose = (event) => {
          this.isConnecting = false;
          this.stopHeartbeat();
          this.lastDisconnectTime = Date.now();
          
          if (this.connectionTimeout) {
            clearTimeout(this.connectionTimeout);
            this.connectionTimeout = null;
          }
          
          if (!this.isDestroyed) {
            console.log('🔌 WebSocket 연결 종료:', event.code, event.reason);
            this.emitEvent('disconnected', { code: event.code, reason: event.reason });
            
            if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
              this.scheduleReconnect();
            }
          }
        };

        this.ws.onerror = (error) => {
          this.isConnecting = false;
          
          if (!this.isDestroyed) {
            console.error('❌ WebSocket 오류:', error);
            this.emitEvent('error', error);
          }
          resolve(false);
        };
      });
    } catch (error) {
      this.isConnecting = false;
      console.error('❌ WebSocket 연결 실패:', error);
      return false;
    }
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const data = JSON.parse(event.data);
      
      if (data.type === 'pong') {
        this.handlePong();
        return;
      }
      
      const handlers = this.messageHandlers.get(data.type) || [];
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error('❌ 메시지 핸들러 오류:', error);
        }
      });
      
    } catch (error) {
      console.error('❌ WebSocket 메시지 파싱 오류:', error);
    }
  }

  private scheduleReconnect(): void {
    if (this.isDestroyed || this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('🚫 재연결 중단');
      return;
    }

    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;
    this.totalReconnects++;
    
    console.log(`🔄 ${delay}ms 후 재연결 시도 (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
    
    this.reconnectTimer = setTimeout(() => {
      if (!this.isDestroyed) {
        this.connect();
      }
    }, delay);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      if (this.isConnected()) {
        this.lastPingTime = performance.now();
        this.send({ type: 'ping', timestamp: Date.now() });
      }
    }, 30000);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private handlePong(): void {
    if (this.lastPingTime > 0) {
      const latency = performance.now() - this.lastPingTime;
      this.latencyMeasurements.push(latency);
      
      if (this.latencyMeasurements.length > 10) {
        this.latencyMeasurements.shift();
      }
      
      this.lastPingTime = 0;
    }
  }

  private processMessageQueue(): void {
    const messages = [...this.messageQueue];
    this.messageQueue = [];
    
    messages.forEach(({ data }) => {
      this.send(data);
    });
  }

  private resubscribeAll(): void {
    this.subscriptions.forEach(sessionId => {
      this.send({
        type: 'subscribe',
        session_id: sessionId,
        timestamp: Date.now()
      });
    });
  }

  private emitEvent(event: string, data?: any): void {
    const handlers = this.eventHandlers.get(event) || [];
    handlers.forEach(handler => {
      try {
        handler(data);
      } catch (error) {
        console.error('❌ 이벤트 핸들러 오류:', error);
      }
    });
  }

  isConnected(): boolean {
    return !this.isDestroyed && this.ws?.readyState === WebSocket.OPEN;
  }

  send(data: any): boolean {
    if (!this.isConnected()) {
      this.messageQueue.push({ data, timestamp: Date.now() });
      return false;
    }

    try {
      this.ws!.send(JSON.stringify(data));
      return true;
    } catch (error) {
      console.error('❌ WebSocket 메시지 전송 실패:', error);
      return false;
    }
  }

  subscribe(sessionId: string): void {
    this.subscriptions.add(sessionId);
    this.send({
      type: 'subscribe',
      session_id: sessionId,
      timestamp: Date.now()
    });
  }

  unsubscribe(sessionId: string): void {
    this.subscriptions.delete(sessionId);
    this.send({
      type: 'unsubscribe',
      session_id: sessionId,
      timestamp: Date.now()
    });
  }

  disconnect(): void {
    console.log('🔌 WebSocket 연결 해제');
    
    this.stopHeartbeat();
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws && this.ws.readyState !== WebSocket.CLOSED) {
      this.ws.close(1000, 'Normal closure');
    }
    
    this.ws = null;
  }

  cleanup(): void {
    console.log('🧹 EnhancedWebSocketManager 정리 시작');
    
    this.isDestroyed = true;
    this.disconnect();
    
    this.messageHandlers.clear();
    this.eventHandlers.clear();
    this.subscriptions.clear();
    this.messageQueue = [];
    
    console.log('✅ EnhancedWebSocketManager 정리 완료');
  }
}

// =================================================================
// 🔧 메인 PipelineAPIClient 클래스 (완전한 기능형)
// =================================================================

class PipelineAPIClient {
  private config: APIClientConfig;
  private defaultHeaders: Record<string, string>;
  private metrics: RequestMetrics;
  private cache: Map<string, CacheEntry> = new Map();
  private requestQueue: QueuedRequest[] = [];
  private activeRequests: Set<string> = new Set();
  private abortControllers: Map<string, AbortController> = new Map();
  private wsManager: EnhancedWebSocketManager | null = null;
  
  private retryDelays: number[] = [1000, 2000, 4000, 8000, 16000];
  private circuitBreakerFailures = 0;
  private circuitBreakerLastFailure = 0;
  private readonly circuitBreakerThreshold = 5;
  private readonly circuitBreakerTimeout = 60000;
  
  private uploadProgressCallbacks: Map<string, (progress: number) => void> = new Map();
  private startTime = Date.now();

  constructor(options: Partial<APIClientConfig> = {}) {
    this.config = {
      baseURL: options.baseURL || 'http://localhost:8000',
      wsURL: options.wsURL || 'ws://localhost:8000/api/ws/ai-pipeline',
      apiKey: options.apiKey,
      timeout: options.timeout || 60000,
      retryAttempts: options.retryAttempts || 3,
      retryDelay: options.retryDelay || 1000,
      enableCaching: options.enableCaching ?? true,
      cacheTimeout: options.cacheTimeout || 300000,
      enableCompression: options.enableCompression ?? true,
      enableRetry: options.enableRetry ?? true,
      uploadChunkSize: 1024 * 1024,
      maxConcurrentRequests: options.maxConcurrentRequests || 3,
      requestQueueSize: 100,
      enableMetrics: true,
      enableDebug: options.enableDebug ?? false,
      enableWebSocket: options.enableWebSocket ?? true,
      heartbeatInterval: options.heartbeatInterval || 30000,
      reconnectInterval: options.reconnectInterval || 3000,
      maxReconnectAttempts: options.maxReconnectAttempts || 10,
    };

    
    this.defaultHeaders = {
      'Accept': 'application/json',
      'User-Agent': `MyClosetAI-Client/2.0.0 (${navigator.userAgent})`,
      'X-Client-Version': '2.0.0',
      'X-Client-Platform': navigator.platform,
      'X-Request-ID': this.generateRequestId(),
      'X-Session-ID': this.generateSessionId(),
    };

    if (this.config.apiKey) {
      this.defaultHeaders['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    if (this.config.enableCompression) {
      this.defaultHeaders['Accept-Encoding'] = 'gzip, deflate, br';
    }

    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageResponseTime: 0,
      totalBytesTransferred: 0,
      cacheHitRate: 0,
      retryRate: 0,
      errorBreakdown: {},
      uptime: 0,
    };

    PipelineUtils.info('🔧 PipelineAPIClient 초기화', {
      baseURL: this.config.baseURL,
      enableWebSocket: this.config.enableWebSocket,
      enableCaching: this.config.enableCaching,
      timeout: this.config.timeout
    });

    this.startBackgroundTasks();
  }

  get baseURL(): string {
    return this.config.baseURL;
  }

  private startBackgroundTasks(): void {
    setInterval(() => this.cleanupExpiredCache(), 60000);
    setInterval(() => this.processRequestQueue(), 100);
    
    if (this.config.enableWebSocket) {
      this.initializeWebSocket();
    }
  }

  private initializeWebSocket(): void {
    if (!this.wsManager) {
      this.wsManager = new EnhancedWebSocketManager(this.config.wsURL!, this.config);
      
      this.wsManager.onMessage('pipeline_progress', (data: PipelineProgress) => {
        PipelineUtils.emitEvent('pipeline:progress', data);
      });
      
      this.wsManager.onEvent('connected', () => {
        PipelineUtils.info('✅ WebSocket 연결됨');
      });
      
      this.wsManager.onEvent('disconnected', () => {
        PipelineUtils.warn('❌ WebSocket 연결 해제됨');
      });
    }
  }

  async initialize(): Promise<boolean> {
    PipelineUtils.info('🔄 PipelineAPIClient 초기화 중...');
    
    try {
      const isHealthy = await this.healthCheck();
      if (!isHealthy) {
        PipelineUtils.error('❌ 서버 헬스체크 실패');
        return false;
      }
      
      if (this.config.enableWebSocket && this.wsManager) {
        await this.wsManager.connect();
      }
      
      PipelineUtils.info('✅ PipelineAPIClient 초기화 완료');
      return true;
    } catch (error) {
      PipelineUtils.error('❌ PipelineAPIClient 초기화 중 오류', error);
      return false;
    }
  }

  private async request<T = any>(
    endpoint: string,
    options: RequestInit = {},
    skipCache: boolean = false
  ): Promise<T> {
    const url = this.buildURL(endpoint);
    const cacheKey = this.generateCacheKey(url, options);
    
    if (!skipCache && this.config.enableCaching && options.method !== 'POST') {
      const cached = this.getFromCache<T>(cacheKey);
      if (cached) {
        return cached;
      }
    }

    if (this.activeRequests.size >= this.config.maxConcurrentRequests) {
      return this.queueRequest<T>(url, options);
    }

    return this.executeRequest<T>(url, options, cacheKey);
  }

 
  private async executeRequest<T>(
    url: string,
    options: RequestInit,
    cacheKey: string,
    attemptNum: number = 1
  ): Promise<T> {
    const requestId = this.generateRequestId();
    const timer = PipelineUtils.createPerformanceTimer(`API Request: ${url}`);
    
    try {
      this.activeRequests.add(requestId);
      this.metrics.totalRequests++;

      const abortController = new AbortController();
      this.abortControllers.set(requestId, abortController);
      
      const timeoutId = setTimeout(() => {
        abortController.abort();
        PipelineUtils.warn('⏰ 요청 타임아웃', { url, timeout: this.config.timeout });
      }, this.config.timeout);

      const requestOptions: RequestInit = {
        ...options,
        headers: {
          ...this.defaultHeaders,
          ...options.headers,
          'X-Request-ID': requestId,
          'X-Attempt-Number': attemptNum.toString(),
          'X-Timestamp': Date.now().toString(),
        },
        signal: abortController.signal,
      };

      if (options.body instanceof FormData) {
        if (requestOptions.headers && 'Content-Type' in requestOptions.headers) {
          const headers = requestOptions.headers as Record<string, string>;
          delete headers['Content-Type'];
        }
      }

      const response = await fetch(url, requestOptions);
      clearTimeout(timeoutId);

      const duration = timer.end();

      const result = await this.processResponse<T>(response, requestId);
      
      if (this.config.enableCaching && requestOptions.method !== 'POST') {
        this.saveToCache(cacheKey, result, this.calculateCacheSize(result), response.headers.get('etag'));
      }

      this.metrics.successfulRequests++;

      return result;

    } catch (error: any) {
      timer.end();
      return this.handleRequestError<T>(error, url, options, cacheKey, attemptNum);
      
    } finally {
      this.activeRequests.delete(requestId);
      this.abortControllers.delete(requestId);
      this.processRequestQueue();
    }
  }

  private async processResponse<T>(response: Response, requestId: string): Promise<T> {
    if (!response.ok) {
      const errorData = await this.parseErrorResponse(response);
      
      PipelineUtils.error('❌ HTTP 오류 응답', {
        status: response.status,
        statusText: response.statusText,
        url: response.url,
        errorData,
        requestId
      });

      throw this.createAPIError(
        `http_${response.status}`,
        errorData.message || response.statusText,
        errorData
      );
    }

    const contentType = response.headers.get('content-type');
    if (contentType?.includes('application/json')) {
      return await response.json();
    } else if (contentType?.includes('text/')) {
      return await response.text() as unknown as T;
    } else {
      return await response.blob() as unknown as T;
    }
  }

  // 가상 피팅 API
  async processVirtualTryOn(
    request: VirtualTryOnRequest,
    onProgress?: (progress: PipelineProgress) => void
  ): Promise<VirtualTryOnResponse> {
    const timer = PipelineUtils.createPerformanceTimer('가상 피팅 API 전체 처리');

    try {
      this.validateVirtualTryOnRequest(request);

      const formData = this.buildVirtualTryOnFormData(request);
      const requestId = this.generateRequestId();
      
      if (onProgress) {
        this.uploadProgressCallbacks.set(requestId, (progress: number) => {
          onProgress({
            type: 'upload_progress',
            progress,
            message: `업로드 중... ${progress}%`,
            timestamp: Date.now()
          });
        });
      }

      if (this.wsManager && this.wsManager.isConnected()) {
        this.wsManager.subscribe(request.session_id || requestId);
      }

      const result = await this.uploadWithProgress<VirtualTryOnResponse>(
        '/api/step/complete',
        formData,
        requestId,
        onProgress
      );

      const duration = timer.end();
      
      PipelineUtils.info('✅ 가상 피팅 API 성공', {
        processingTime: duration / 1000,
        fitScore: result.fit_score,
        confidence: result.confidence
      });

      return result;

    } catch (error: any) {
      timer.end();
      const friendlyError = PipelineUtils.getUserFriendlyError(error);
      PipelineUtils.error('❌ 가상 피팅 API 실패', friendlyError);
      throw error;
    }
  }

  private buildVirtualTryOnFormData(request: VirtualTryOnRequest): FormData {
    const formData = new FormData();
    
    formData.append('person_image', request.person_image);
    formData.append('clothing_image', request.clothing_image);
    
    formData.append('height', request.height.toString());
    formData.append('weight', request.weight.toString());
    
    if (request.chest) formData.append('chest', request.chest.toString());
    if (request.waist) formData.append('waist', request.waist.toString());
    if (request.hip) formData.append('hip', request.hip.toString());
    if (request.shoulder_width) formData.append('shoulder_width', request.shoulder_width.toString());
    
    formData.append('clothing_type', request.clothing_type || 'upper_body');
    formData.append('fabric_type', request.fabric_type || 'cotton');
    formData.append('style_preference', request.style_preference || 'regular');
    
    formData.append('quality_mode', request.quality_mode || 'balanced');
    formData.append('session_id', request.session_id || this.generateSessionId());
    formData.append('enable_realtime', String(request.enable_realtime || false));
    formData.append('save_intermediate', String(request.save_intermediate || false));
    
    if (request.pose_adjustment !== undefined) {
      formData.append('pose_adjustment', String(request.pose_adjustment));
    }
    if (request.color_preservation !== undefined) {
      formData.append('color_preservation', String(request.color_preservation));
    }
    if (request.texture_enhancement !== undefined) {
      formData.append('texture_enhancement', String(request.texture_enhancement));
    }
    
    const systemParams = PipelineUtils.getSystemParams();
    for (const [key, value] of systemParams) {
      formData.append(key, String(value));
    }
    
    formData.append('client_version', '2.0.0');
    formData.append('platform', navigator.platform);
    formData.append('timestamp', new Date().toISOString());
    formData.append('user_agent', navigator.userAgent);
    
    return formData;
  }

  // 헬스체크
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.request('/health', {}, true);
      return response.status === 'healthy' || response.success === true;
    } catch (error) {
      PipelineUtils.debug('❌ 헬스체크 실패', error);
      return false;
    }
  }

  // WebSocket 관련 메서드들
  connectWebSocket(): Promise<boolean> {
    if (!this.wsManager) {
      this.initializeWebSocket();
    }
    return this.wsManager?.connect() || Promise.resolve(false);
  }

  disconnectWebSocket(): void {
    this.wsManager?.disconnect();
  }

  isWebSocketConnected(): boolean {
    return this.wsManager?.isConnected() || false;
  }

  subscribeToSession(sessionId: string): void {
    this.wsManager?.subscribe(sessionId);
  }

  // 유틸리티 메서드들
  private buildURL(endpoint: string): string {
    const baseURL = this.config.baseURL.replace(/\/$/, '');
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    return `${baseURL}${cleanEndpoint}`;
  }

  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private generateSessionId(): string {
    return `ses_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private validateVirtualTryOnRequest(request: VirtualTryOnRequest): void {
    if (!request.person_image || !request.clothing_image) {
      throw this.createAPIError('validation_error', '사용자 이미지와 의류 이미지는 필수입니다.');
    }

    this.validateImageFile(request.person_image, '사용자 이미지');
    this.validateImageFile(request.clothing_image, '의류 이미지');

    if (request.height <= 0 || request.height > 300) {
      throw this.createAPIError('validation_error', '키는 1-300cm 범위여야 합니다.');
    }

    if (request.weight <= 0 || request.weight > 500) {
      throw this.createAPIError('validation_error', '몸무게는 1-500kg 범위여야 합니다.');
    }
  }

  private validateImageFile(file: File, fieldName: string = '이미지'): void {
    if (!PipelineUtils.validateImageType(file)) {
      throw this.createAPIError('invalid_file', `${fieldName}: 지원되지 않는 파일 형식입니다. JPG, PNG, WebP 파일을 사용해주세요.`);
    }

    if (!PipelineUtils.validateFileSize(file, 50)) {
      throw this.createAPIError('file_too_large', `${fieldName}: 파일 크기가 너무 큽니다. 50MB 이하의 파일을 사용해주세요.`);
    }
  }

  private createAPIError(code: string, message: string, details?: any): any {
    return {
      code,
      message,
      details,
      timestamp: new Date().toISOString(),
      request_id: this.generateRequestId()
    };
  }

  // 캐싱 시스템
  private generateCacheKey(url: string, options: RequestInit): string {
    const method = options.method || 'GET';
    const headers = JSON.stringify(options.headers || {});
    const body = options.body instanceof FormData ? 'FormData' : JSON.stringify(options.body || '');
    return `${method}:${url}:${headers}:${body}`;
  }

  private getFromCache<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    if (Date.now() > entry.expiry) {
      this.cache.delete(key);
      return null;
    }

    entry.hits++;
    return entry.data;
  }

  private saveToCache<T>(key: string, data: T, size: number, etag?: string | null): void {
    const expiry = Date.now() + this.config.cacheTimeout;
    
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      expiry,
      size,
      hits: 0,
      etag: etag || undefined
    });

    if (this.cache.size > 200) {
      this.evictOldestCacheEntries();
    }
  }

  private evictOldestCacheEntries(): void {
    const entries = Array.from(this.cache.entries());
    entries.sort((a, b) => {
      const scoreA = a[1].hits / (Date.now() - a[1].timestamp);
      const scoreB = b[1].hits / (Date.now() - b[1].timestamp);
      return scoreA - scoreB;
    });
    
    const toRemove = Math.min(50, entries.length);
    for (let i = 0; i < toRemove; i++) {
      this.cache.delete(entries[i][0]);
    }
  }

  private cleanupExpiredCache(): void {
    const now = Date.now();
    const expiredKeys: string[] = [];
    
    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiry) {
        expiredKeys.push(key);
      }
    }
    
    for (const key of expiredKeys) {
      this.cache.delete(key);
    }
    
    if (expiredKeys.length > 0) {
      PipelineUtils.debug('🗑️ 만료된 캐시 항목 정리됨', { count: expiredKeys.length });
    }
  }

  // 요청 큐잉 시스템
  private async queueRequest<T>(url: string, options: RequestInit): Promise<T> {
    return new Promise((resolve, reject) => {
      const queuedRequest: QueuedRequest = {
        id: this.generateRequestId(),
        url,
        options,
        resolve,
        reject,
        priority: this.getRequestPriority(url),
        attempts: 0,
        timestamp: Date.now(),
        maxRetries: this.config.retryAttempts
      };

      if (this.requestQueue.length >= this.config.requestQueueSize) {
        reject(new Error('Request queue is full'));
        return;
      }

      this.requestQueue.push(queuedRequest);
      this.requestQueue.sort((a, b) => b.priority - a.priority);
    });
  }

  private getRequestPriority(url: string): number {
    if (url.includes('/health')) return 10;
    if (url.includes('/virtual-tryon')) return 9;
    if (url.includes('/step/complete')) return 9;
    if (url.includes('/step/')) return 8;
    return 5;
  }

  private processRequestQueue(): void {
    while (
      this.requestQueue.length > 0 && 
      this.activeRequests.size < this.config.maxConcurrentRequests
    ) {
      const queuedRequest = this.requestQueue.shift()!;
      
      this.executeRequest(
        queuedRequest.url,
        queuedRequest.options,
        this.generateCacheKey(queuedRequest.url, queuedRequest.options)
      )
        .then(queuedRequest.resolve)
        .catch(queuedRequest.reject);
    }
  }

  // 재시도 로직
  private shouldRetry(error: any, attemptNum: number): boolean {
    if (!this.config.enableRetry || attemptNum >= this.config.retryAttempts) {
      return false;
    }

    const errorCode = this.getErrorCode(error);
    
    const nonRetryableErrors = [
      'http_400', 'http_401', 'http_403', 'http_404', 
      'http_422', 'validation_error', 'invalid_file'
    ];
    
    if (nonRetryableErrors.includes(errorCode)) {
      return false;
    }

    const retryableErrors = [
      'http_500', 'http_502', 'http_503', 'http_504',
      'network_error', 'timeout', 'connection_failed'
    ];
    
    return retryableErrors.includes(errorCode) || error.name === 'AbortError';
  }

  private calculateRetryDelay(attemptNum: number): number {
    const baseDelay = this.config.retryDelay;
    const exponentialDelay = Math.min(
      baseDelay * Math.pow(2, attemptNum - 1),
      this.retryDelays[Math.min(attemptNum - 1, this.retryDelays.length - 1)]
    );
    
    const jitter = exponentialDelay * 0.25 * (Math.random() - 0.5);
    return Math.max(1000, exponentialDelay + jitter);
  }

  private async handleRequestError<T>(
    error: any,
    url: string,
    options: RequestInit,
    cacheKey: string,
    attemptNum: number
  ): Promise<T> {
    this.metrics.failedRequests++;

    const errorCode = this.getErrorCode(error);

    PipelineUtils.error('❌ API 요청 실패', {
      url,
      error: error.message,
      attempt: attemptNum,
      errorCode
    });

    if (this.shouldRetry(error, attemptNum)) {
      const delay = this.calculateRetryDelay(attemptNum);
      
      PipelineUtils.info(`🔄 재시도 예약됨 (${attemptNum}/${this.config.retryAttempts})`, {
        delay,
        url
      });
      
      await PipelineUtils.sleep(delay);
      return this.executeRequest<T>(url, options, cacheKey, attemptNum + 1);
    }

    throw error;
  }

  private getErrorCode(error: any): string {
    if (error?.code) return error.code;
    if (error?.name === 'AbortError') return 'timeout';
    if (error?.message?.includes('fetch')) return 'network_error';
    if (error?.message?.includes('timeout')) return 'timeout';
    if (error?.message?.includes('JSON')) return 'parse_error';
    return 'unknown_error';
  }

  private calculateCacheSize(data: any): number {
    try {
      return JSON.stringify(data).length;
    } catch {
      return 0;
    }
  }

  private async parseErrorResponse(response: Response): Promise<any> {
    try {
      const contentType = response.headers.get('content-type');
      if (contentType?.includes('application/json')) {
        return await response.json();
      } else {
        const text = await response.text();
        return { message: text || response.statusText };
      }
    } catch (parseError) {
      PipelineUtils.warn('⚠️ 에러 응답 파싱 실패', parseError);
      return { message: response.statusText };
    }
  }

  private async uploadWithProgress<T>(
    endpoint: string,
    formData: FormData,
    requestId: string,
    onProgress?: (progress: PipelineProgress) => void
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const url = this.buildURL(endpoint);

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const progress = Math.round((event.loaded / event.total) * 100);
          const callback = this.uploadProgressCallbacks.get(requestId);
          if (callback) {
            callback(progress);
          }
        }
      });

      xhr.addEventListener('load', () => {
        this.uploadProgressCallbacks.delete(requestId);
        
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const result = JSON.parse(xhr.responseText);
            resolve(result);
          } catch (error) {
            reject(new Error('Invalid JSON response'));
          }
        } else {
          try {
            const errorData = JSON.parse(xhr.responseText);
            reject(this.createAPIError(
              `http_${xhr.status}`,
              errorData.message || xhr.statusText,
              errorData
            ));
          } catch {
            reject(this.createAPIError(`http_${xhr.status}`, xhr.statusText));
          }
        }
      });

      xhr.addEventListener('error', (event) => {
        this.uploadProgressCallbacks.delete(requestId);
        reject(new Error('Network error during upload'));
      });

      xhr.addEventListener('timeout', () => {
        this.uploadProgressCallbacks.delete(requestId);
        reject(new Error('Upload timeout'));
      });

      xhr.addEventListener('abort', () => {
        this.uploadProgressCallbacks.delete(requestId);
        reject(new Error('Upload aborted'));
      });

      xhr.timeout = this.config.timeout;
      xhr.open('POST', url);

      for (const [key, value] of Object.entries(this.defaultHeaders)) {
        if (key !== 'Content-Type') {
          xhr.setRequestHeader(key, value);
        }
      }
      xhr.setRequestHeader('X-Request-ID', requestId);

      xhr.send(formData);
    });
  }

  // 정리 및 종료
  async cleanup(): Promise<void> {
    PipelineUtils.info('🧹 PipelineAPIClient: 리소스 정리 중...');
    
    try {
      if (this.wsManager) {
        this.wsManager.cleanup();
        this.wsManager = null;
      }
      
      this.uploadProgressCallbacks.clear();
      this.cache.clear();
      
      PipelineUtils.info('✅ PipelineAPIClient 리소스 정리 완료');
    } catch (error) {
      PipelineUtils.warn('⚠️ PipelineAPIClient 리소스 정리 중 오류', error);
    }
  }
}

// =================================================================
// 🔧 백엔드 호환 API 클라이언트 (완전한 기능형)
// =================================================================

interface UserMeasurements {
  height: number;
  weight: number;
}

interface StepResult {
  success: boolean;
  message: string;
  processing_time: number;
  session_id?: string;  // 🔥 추가: 최상위 레벨 session_id

  confidence: number;
  error?: string;
  details?: {
    session_id?: string;
    result_image?: string;
    visualization?: string;
    overlay_image?: string;
    detected_parts?: number;
    total_parts?: number;
    detected_keypoints?: number;
    total_keypoints?: number;
    category?: string;
    style?: string;
    clothing_info?: {
      category: string;
      style: string;
      colors: string[];
    };
    body_parts?: string[];
    pose_confidence?: number;
    matching_score?: number;
    alignment_points?: number;
    fitting_quality?: string;
  };
  fitted_image?: string;
  warped_cloth?: string;
  fit_score?: number;
  recommendations?: string[];
}

interface TryOnResult {
  success: boolean;
  message: string;
  processing_time: number;
  confidence: number;
  session_id: string;
  fitted_image?: string;
  fit_score: number;
  measurements: {
    chest: number;
    waist: number;
    hip: number;
    bmi: number;
  };
  clothing_analysis: {
    category: string;
    style: string;
    dominant_color: number[];
    color_name?: string;
    material?: string;
    pattern?: string;
  };
  recommendations: string[];
}

interface SystemInfo {
  app_name: string;
  app_version: string;
  device: string;
  device_name: string;
  is_m3_max: boolean;
  total_memory_gb: number;
  available_memory_gb: number;
  timestamp: number;
}

interface PipelineStep {
  id: number;
  name: string;
  description: string;
  endpoint: string;
  processing_time: number;
}

// 🔥 새로운 8단계 AI 파이프라인 정의
const PIPELINE_STEPS: PipelineStep[] = [
  {
    id: 1,
    name: "인체 파싱",
    description: "AI가 신체 부위를 20개 영역으로 분석합니다",
    endpoint: "/api/step/1/human-parsing",
    processing_time: 1.2
  },
  {
    id: 2,
    name: "포즈 추정",
    description: "18개 키포인트로 자세를 분석합니다",
    endpoint: "/api/step/2/pose-estimation",
    processing_time: 0.8
  },
  {
    id: 3,
    name: "의류 세그멘테이션",
    description: "의류 영역을 정확히 분리합니다",
    endpoint: "/api/step/3/cloth-segmentation",
    processing_time: 0.6
  },
  {
    id: 4,
    name: "기하학적 매칭",
    description: "신체와 의류를 정확히 매칭합니다",
    endpoint: "/api/step/4/geometric-matching",
    processing_time: 1.5
  },
  {
    id: 5,
    name: "의류 워핑",
    description: "의류를 신체에 맞게 변형합니다",
    endpoint: "/api/step/5/cloth-warping",
    processing_time: 1.0
  },
  {
    id: 6,
    name: "가상 피팅",
    description: "AI로 가상 착용 결과를 생성합니다",
    endpoint: "/api/step/6/virtual-fitting",
    processing_time: 2.5
  },
  {
    id: 7,
    name: "후처리",
    description: "이미지 품질을 개선합니다",
    endpoint: "/api/step/7/post-processing",
    processing_time: 0.8
  },
  {
    id: 8,
    name: "품질 평가",
    description: "최종 결과를 분석하고 평가합니다",
    endpoint: "/api/step/8/quality-assessment",
    processing_time: 0.3
  }
];

// 백엔드 완전 호환 API 클라이언트
class APIClient {
  private baseURL: string;
  private currentSessionId: string | null = null;
  private websocket: WebSocket | null = null;
  private progressCallback: ((step: number, progress: number, message: string) => void) | null = null;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.baseURL = baseURL;
  }

  setSessionId(sessionId: string) {
    this.currentSessionId = sessionId;
  }

  getSessionId(): string | null {
    return this.currentSessionId;
  }

  setProgressCallback(callback: (step: number, progress: number, message: string) => void) {
    this.progressCallback = callback;
  }

  // WebSocket 연결
  connectWebSocket(sessionId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const wsURL = `ws://localhost:8000/api/ws/ai-pipeline`;
        this.websocket = new WebSocket(wsURL);

        this.websocket.onopen = () => {
          console.log('🔗 WebSocket 연결됨');
          if (this.websocket) {
            this.websocket.send(JSON.stringify({ 
              type: 'subscribe', 
              session_id: sessionId 
            }));
          }
          resolve();
        };

        this.websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            if (data.type === 'ai_progress' && this.progressCallback) {
              this.progressCallback(data.step || 0, data.progress || 0, data.message || '');
            }
            
            console.log('📡 WebSocket 메시지:', data);
          } catch (error) {
            console.error('WebSocket 메시지 파싱 오류:', error);
          }
        };

        this.websocket.onerror = (error) => {
          console.error('WebSocket 오류:', error);
          reject(error);
        };

        this.websocket.onclose = () => {
          console.log('🔌 WebSocket 연결 해제됨');
          this.websocket = null;
        };

        setTimeout(() => {
          if (this.websocket?.readyState !== WebSocket.OPEN) {
            reject(new Error('WebSocket 연결 타임아웃'));
          }
        }, 5000);

      } catch (error) {
        reject(error);
      }
    });
  }

  disconnectWebSocket() {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }

  // 헬스체크
  async healthCheck(): Promise<{ success: boolean; data?: any; error?: string }> {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      
      // 응답 상태 확인
      if (!response.ok) {
        return { 
          success: false, 
          error: `HTTP ${response.status}: ${response.statusText}` 
        };
      }
      
      // Content-Type 확인
      const contentType = response.headers.get('content-type');
      let data;
      
      if (contentType && contentType.includes('application/json')) {
        data = await response.json();
      } else {
        // JSON이 아닌 경우 텍스트로 읽기
        const text = await response.text();
        try {
          data = JSON.parse(text);
        } catch {
          data = { message: text };
        }
      }
      
      return { success: true, data };
    } catch (error) {
      console.error('Health check error:', error);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Network error' 
      };
    }
  }

  // 시스템 정보 조회
  async getSystemInfo(): Promise<SystemInfo> {
    const response = await fetch(`${this.baseURL}/api/system/info`);
    if (!response.ok) {
      throw new Error(`시스템 정보 조회 실패: ${response.status}`);
    }
    return await response.json();
  }

  // 개별 단계 API 호출
  async callStepAPI(stepId: number, formData: FormData): Promise<StepResult> {
    const step = PIPELINE_STEPS.find(s => s.id === stepId);
    if (!step) {
      throw new Error(`Invalid step ID: ${stepId}`);
    }

    if (this.currentSessionId) {
      formData.append('session_id', this.currentSessionId);
    }

    try {
      console.log(`🚀 Step ${stepId} API 호출: ${step.endpoint}`);
      console.log(`🌐 요청 URL: ${this.baseURL}${step.endpoint}`);
      console.log(`📤 FormData 내용:`, {
        entries: Array.from(formData.entries()).map(([key, value]) => ({
          key,
          valueType: value instanceof File ? 'File' : typeof value,
          fileName: value instanceof File ? value.name : value,
          fileSize: value instanceof File ? value.size : 'N/A'
        }))
      });
      
      const response = await fetch(`${this.baseURL}${step.endpoint}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage;
        
        try {
          const errorJson = JSON.parse(errorText);
          errorMessage = errorJson.detail || errorJson.error || errorJson.message || `HTTP ${response.status}`;
        } catch {
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        
        throw new Error(errorMessage);
      }

      const result: StepResult = await response.json();
      
      // 세션 ID 업데이트
      if (stepId === 1 && result.details?.session_id) {
        this.setSessionId(result.details.session_id);
      }

      console.log(`✅ Step ${stepId} 완료:`, result);
      return result;
      
    } catch (error) {
      console.error(`❌ Step ${stepId} 실패:`, error);
      throw error;
    }
  }

  // 전체 파이프라인 실행
  async runCompletePipeline(
    personImage: File, 
    clothingImage: File, 
    measurements: UserMeasurements
  ): Promise<TryOnResult> {
    const formData = new FormData();
    formData.append('person_image', personImage);
    formData.append('clothing_image', clothingImage);
    formData.append('height', measurements.height.toString());
    formData.append('weight', measurements.weight.toString());
    
    if (this.currentSessionId) {
      formData.append('session_id', this.currentSessionId);
    }

    try {
      console.log('🚀 전체 파이프라인 실행 시작');
      
      const response = await fetch(`${this.baseURL}/api/step/complete`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Pipeline failed: ${response.status} - ${errorText}`);
      }

      const result: TryOnResult = await response.json();
      
      if (result.session_id) {
        this.setSessionId(result.session_id);
      }

      console.log('✅ 전체 파이프라인 완료:', result);
      return result;
      
    } catch (error) {
      console.error('❌ 전체 파이프라인 실패:', error);
      throw error;
    }
  }
}

// =================================================================
// 🔧 유틸리티 함수들
// =================================================================

const fileUtils = {
  validateImageFile: (file: File) => {
    const maxSize = 50 * 1024 * 1024; // 50MB
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    
    if (file.size > maxSize) {
      return { valid: false, error: '파일 크기가 50MB를 초과합니다.' };
    }
    
    if (!allowedTypes.includes(file.type)) {
      return { valid: false, error: '지원되지 않는 파일 형식입니다.' };
    }
    
    return { valid: true };
  },
  
  formatFileSize: (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },

  createImagePreview: (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }
};

// =================================================================
// 🔧 메인 App 컴포넌트 (완전한 수정 버전)
// =================================================================

const App: React.FC = () => {
  // usePipeline hook 사용 - 자동 실행 제거로 인해 더 이상 필요하지 않음

  // API 클라이언트 초기화
  const [apiClient] = useState(() => {
    const client = new APIClient();
    
    // 🔥 새로고침 후 세션 복구: localStorage에서 세션 ID 복원
    const savedSessionId = localStorage.getItem('mycloset_session_id');
    if (savedSessionId) {
      client.setSessionId(savedSessionId);
      console.log('🔄 세션 복구됨:', savedSessionId);
    }
    
    return client;
  });
  const [pipelineClient] = useState(() => new PipelineAPIClient());

  // 현재 단계 관리 (0: 업로드/측정, 1-8: AI 파이프라인)
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  
  // 파일 상태
  const [personImage, setPersonImage] = useState<File | null>(null);
  const [clothingImage, setClothingImage] = useState<File | null>(null);
  const [personImagePreview, setPersonImagePreview] = useState<string | null>(null);
  const [clothingImagePreview, setClothingImagePreview] = useState<string | null>(null);
  
  // 측정값
  const [measurements, setMeasurements] = useState<UserMeasurements>({
    height: 170,
    weight: 65
  });

  // 단계별 결과 저장
  const [stepResults, setStepResults] = useState<{[key: number]: StepResult}>({});
  
  // 최종 결과
  const [result, setResult] = useState<TryOnResult | null>(null);
  
  // 파일 검증 에러
  const [fileErrors, setFileErrors] = useState<{
    person?: string;
    clothing?: string;
  }>({});

  // 처리 상태
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [error, setError] = useState<string | null>(null);

  // 서버 상태
  const [isServerHealthy, setIsServerHealthy] = useState(true);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);

  // 반응형 상태
  const [isMobile, setIsMobile] = useState(false);

  // 파일 참조
  const personImageRef = useRef<HTMLInputElement>(null);
  const clothingImageRef = useRef<HTMLInputElement>(null);

  // Step 0 완료 후 자동 실행
  
  const executeFullPipeline = async () => {
    if (!personImage || !clothingImage) {
      setError('이미지를 먼저 업로드해주세요.');
      return;
    }
    
    setIsProcessing(true);
    setProgress(0);
    setProgressMessage('전체 파이프라인 시작...');
    setError(null);
    setResult(null);
    
    try {
      // �� pipeline API 호출 (전체 파이프라인 실행)
      const result = await apiClient.runCompletePipeline(
        personImage, 
        clothingImage, 
        measurements
      );
      
      if (result.success) {
        setResult(result);
        setProgress(100);
        setProgressMessage('🎉 전체 파이프라인 완료!');
      } else {
        throw new Error(result.message || '파이프라인 실행 실패');
      }
      
    } catch (error: any) {
      console.error('❌ 전체 파이프라인 실패:', error);
      setError(`전체 파이프라인 실패: ${error.message}`);
    } finally {
      setIsProcessing(false);
      setProgress(0);
      setProgressMessage('');
    }
  };
  // Step 1 실행 함수 (인체 파싱)
  const executeStep1 = async () => {
    setIsProcessing(true);
    setProgress(0);
    setProgressMessage('Step 1: 인체 파싱 중...');
    
    try {
      // 🔥 세션 ID 확인 (localStorage에서 복구됨)
      const currentSessionId = apiClient.getSessionId();
      
      if (!currentSessionId) {
        throw new Error('세션 ID가 없습니다. 이미지 업로드를 먼저 완료해주세요.');
      }
      
      console.log('🔍 Step 1 실행 - 세션 ID:', currentSessionId);
      
      // 🔥 Step 1: apiClient.callStepAPI 사용하여 실제 AI 추론 호출
      const formData = new FormData();
      
      // 세션 ID 추가
      formData.append('session_id', currentSessionId);
      
      // 이미지 추가
      if (personImage) formData.append('person_image', personImage);
      if (clothingImage) formData.append('clothing_image', clothingImage);
      
      // 측정값 추가
      formData.append('height', measurements.height.toString());
      formData.append('weight', measurements.weight.toString());
      
      // 기타 파라미터
      formData.append('confidence_threshold', '0.7');
      formData.append('enhance_quality', 'true');
      formData.append('force_ai_processing', 'true');
      
      console.log('🚀 Step 1 인체 파싱 시작:', {
        person_image: personImage?.name,
        clothing_image: clothingImage?.name,
        height: measurements.height,
        weight: measurements.weight,
        endpoint: '/api/step/1/human-parsing'
      });
      
      // 🔥 apiClient.callStepAPI 사용하여 실제 AI 추론 호출
      const result = await apiClient.callStepAPI(1, formData);
      
      if (!result.success) {
        throw new Error(`Step 1 실패: ${result.error || result.message}`);
      }
      
      // 세션 ID 업데이트
      if (result.session_id) {
        apiClient.setSessionId(result.session_id);
      }
      
      setStepResults(prev => ({ ...prev, 1: result }));
      setCompletedSteps(prev => [...prev, 1]);
      
      // 🔥 Step 1 완료 후 currentStep을 1로 설정
      setCurrentStep(1);
      
      // 결과 이미지가 있으면 표시
      if (result.details?.result_image) {
        setPersonImagePreview(result.details.result_image);
      }
      
      console.log('✅ Step 1 인체 파싱 완료:', result);
      
    } catch (error) {
      console.error('Step 1 실행 실패:', error);
      setError(error instanceof Error ? error.message : 'Step 1 실행 중 오류가 발생했습니다.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Step 2 실행 함수 (포즈 추정)
  const executeStep2 = async () => {
    setIsProcessing(true);
    setProgress(0);
    setProgressMessage('Step 2: 포즈 추정 중...');
    
    try {
      // 🔥 세션 ID 확인 (더 안전한 방식)
      let currentSessionId = stepResults[1]?.session_id || 
                           stepResults[1]?.details?.session_id || 
                           apiClient.getSessionId() ||
                           localStorage.getItem('mycloset_session_id');
      
      if (!currentSessionId) {
        throw new Error('세션 ID가 없습니다. Step 1을 먼저 완료해주세요.');
      }
      
      // 세션 ID가 있으면 apiClient에 설정
      apiClient.setSessionId(currentSessionId);
      
      console.log('🔍 Step 2 실행 - 세션 ID:', currentSessionId);
      
      // 🔥 Step 2: 세션 ID만 전송 (이미지와 측정값은 백엔드에서 세션에서 로드)
      const formData = new FormData();
      formData.append('session_id', currentSessionId);
      
      console.log('🚀 Step 2 포즈 추정 시작:', {
        session_id: currentSessionId,
        endpoint: '/api/step/2/pose-estimation'
      });
      
      const result = await apiClient.callStepAPI(2, formData);
      
      if (!result.success) {
        throw new Error(`Step 2 실패: ${result.error || result.message}`);
      }
      
      setStepResults(prev => ({ ...prev, 2: result }));
      setCompletedSteps(prev => [...prev, 2]);
      
      // 🔥 Step 2 완료 후 currentStep을 2로 설정
      setCurrentStep(2);
      
      // 결과 이미지가 있으면 표시
      if (result.details?.result_image) {
        setPersonImagePreview(result.details.result_image);
      }
      
    } catch (error) {
      console.error('Step 2 실행 실패:', error);
      setError(error instanceof Error ? error.message : 'Step 2 실행 중 오류가 발생했습니다.');
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Step 3 실행 함수 (의류 세그멘테이션)
  const executeStep3 = async () => {
    setIsProcessing(true);
    setProgress(0);
    setProgressMessage('Step 3: 의류 세그멘테이션 중...');
    
    try {
      // 🔥 세션 ID 확인 (더 안전한 방식)
      let currentSessionId = stepResults[2]?.session_id || 
                           stepResults[2]?.details?.session_id || 
                           stepResults[1]?.session_id ||
                           stepResults[1]?.details?.session_id ||
                           apiClient.getSessionId() ||
                           localStorage.getItem('mycloset_session_id');
      
      if (!currentSessionId) {
        throw new Error('세션 ID가 없습니다. Step 2를 먼저 완료해주세요.');
      }
      
      // 세션 ID가 있으면 apiClient에 설정
      apiClient.setSessionId(currentSessionId);
      console.log('🔍 Step 3 실행 - 세션 ID:', currentSessionId);
      
      // 🔥 Step 3: 세션 ID만 전송 (이미지와 측정값은 백엔드에서 세션에서 로드)
      const formData = new FormData();
      formData.append('session_id', currentSessionId);
      
      console.log('🚀 Step 3 의류 세그멘테이션 시작:', {
        session_id: currentSessionId,
        endpoint: '/api/step/3/cloth-segmentation'
      });
      
      const result = await apiClient.callStepAPI(3, formData);
      
      if (!result.success) {
        throw new Error(`Step 3 실패: ${result.error || result.message}`);
      }
      
      setStepResults(prev => ({ ...prev, 3: result }));
      setCompletedSteps(prev => [...prev, 3]);
      
      // 🔥 Step 3 완료 후 currentStep을 3으로 설정
      setCurrentStep(3);
      
      // 결과 이미지가 있으면 표시
      if (result.details?.result_image) {
        setPersonImagePreview(result.details.result_image);
      }
      
    } catch (error) {
      console.error('Step 3 실행 실패:', error);
      setError(error instanceof Error ? error.message : 'Step 3 실행 중 오류가 발생했습니다.');
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Step 4 실행 함수 (기하학적 매칭)
  const executeStep4 = async () => {
    setIsProcessing(true);
    setProgress(0);
    setProgressMessage('Step 4: 기하학적 매칭 중...');
    
    try {
      // 🔥 세션 ID 확인 (더 안전한 방식)
      let currentSessionId = stepResults[3]?.session_id || 
                           stepResults[3]?.details?.session_id || 
                           stepResults[2]?.session_id ||
                           stepResults[2]?.details?.session_id ||
                           stepResults[1]?.session_id ||
                           stepResults[1]?.details?.session_id ||
                           apiClient.getSessionId() ||
                           localStorage.getItem('mycloset_session_id');
      
      if (!currentSessionId) {
        throw new Error('세션 ID가 없습니다. Step 3을 먼저 완료해주세요.');
      }
      
      // 세션 ID가 있으면 apiClient에 설정
      apiClient.setSessionId(currentSessionId);
      console.log('🔍 Step 4 실행 - 세션 ID:', currentSessionId);
      
      // 🔥 Step 4: 세션 ID만 전송 (이미지와 측정값은 백엔드에서 세션에서 로드)
      const formData = new FormData();
      formData.append('session_id', currentSessionId);
      
      console.log('🚀 Step 4 기하학적 매칭 시작:', {
        session_id: currentSessionId,
        endpoint: '/api/step/4/geometric-matching'
      });
      
      const result = await apiClient.callStepAPI(4, formData);
      
      if (!result.success) {
        throw new Error(`Step 4 실패: ${result.error || result.message}`);
      }
      
      setStepResults(prev => ({ ...prev, 4: result }));
      setCompletedSteps(prev => [...prev, 4]);
      
      // 🔥 Step 4 완료 후 currentStep을 4로 설정
      setCurrentStep(4);
      
      // 결과 이미지가 있으면 표시
      if (result.details?.result_image) {
        setPersonImagePreview(result.details.result_image);
      }
      
    } catch (error) {
      console.error('Step 4 실행 실패:', error);
      setError(error instanceof Error ? error.message : 'Step 4 실행 중 오류가 발생했습니다.');
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Step 5 실행 함수 (의류 워핑)
  const executeStep5 = async () => {
    setIsProcessing(true);
    setProgress(0);
    setProgressMessage('Step 5: 의류 워핑 중...');
    
    try {
      // 🔥 세션 ID 확인 (더 안전한 방식)
      let currentSessionId = stepResults[4]?.session_id || 
                           stepResults[4]?.details?.session_id || 
                           stepResults[3]?.session_id ||
                           stepResults[3]?.details?.session_id ||
                           stepResults[2]?.session_id ||
                           stepResults[2]?.details?.session_id ||
                           stepResults[1]?.session_id ||
                           stepResults[1]?.details?.session_id ||
                           apiClient.getSessionId() ||
                           localStorage.getItem('mycloset_session_id');
      
      if (!currentSessionId) {
        throw new Error('세션 ID가 없습니다. Step 4를 먼저 완료해주세요.');
      }
      
      // 세션 ID가 있으면 apiClient에 설정
      apiClient.setSessionId(currentSessionId);
      console.log('🔍 Step 5 실행 - 세션 ID:', currentSessionId);
      
      // 🔥 Step 5: 세션 ID만 전송 (이미지와 측정값은 백엔드에서 세션에서 로드)
      const formData = new FormData();
      formData.append('session_id', currentSessionId);
      
      console.log('🚀 Step 5 의류 워핑 시작:', {
        session_id: currentSessionId,
        endpoint: '/api/step/5/cloth-warping'
      });
      
      const result = await apiClient.callStepAPI(5, formData);
      
      if (!result.success) {
        throw new Error(`Step 5 실패: ${result.error || result.message}`);
      }
      
      setStepResults(prev => ({ ...prev, 5: result }));
      setCompletedSteps(prev => [...prev, 5]);
      
      // 🔥 Step 5 완료 후 currentStep을 5로 설정
      setCurrentStep(5);
      
      // 결과 이미지가 있으면 표시
      if (result.details?.result_image) {
        setPersonImagePreview(result.details.result_image);
      }
      
    } catch (error) {
      console.error('Step 5 실행 실패:', error);
      setError(error instanceof Error ? error.message : 'Step 5 실행 중 오류가 발생했습니다.');
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Step 6 실행 함수 (가상 피팅)
  const executeStep6 = async () => {
    setIsProcessing(true);
    setProgress(0);
    setProgressMessage('Step 6: 가상 피팅 중...');
    
    try {
      // 🔥 세션 ID 확인 (더 안전한 방식)
      let currentSessionId = stepResults[5]?.session_id || 
                           stepResults[5]?.details?.session_id || 
                           stepResults[4]?.session_id ||
                           stepResults[4]?.details?.session_id ||
                           stepResults[3]?.session_id ||
                           stepResults[3]?.details?.session_id ||
                           stepResults[2]?.session_id ||
                           stepResults[2]?.details?.session_id ||
                           stepResults[1]?.session_id ||
                           stepResults[1]?.details?.session_id ||
                           apiClient.getSessionId() ||
                           localStorage.getItem('mycloset_session_id');
      
      if (!currentSessionId) {
        throw new Error('세션 ID가 없습니다. Step 5를 먼저 완료해주세요.');
      }
      
      // 세션 ID가 있으면 apiClient에 설정
      apiClient.setSessionId(currentSessionId);
      console.log('🔍 Step 6 실행 - 세션 ID:', currentSessionId);
      
      // 🔥 Step 6: 세션 ID만 전송 (이미지와 측정값은 백엔드에서 세션에서 로드)
      const formData = new FormData();
      formData.append('session_id', currentSessionId);
      
      console.log('🚀 Step 6 가상 피팅 시작:', {
        session_id: currentSessionId,
        endpoint: '/api/step/6/virtual-fitting'
      });
      
      const result = await apiClient.callStepAPI(6, formData);
      
      if (!result.success) {
        throw new Error(`Step 6 실패: ${result.error || result.message}`);
      }
      
      setStepResults(prev => ({ ...prev, 6: result }));
      setCompletedSteps(prev => [...prev, 6]);
      
      // 🔥 Step 6 완료 후 currentStep을 6으로 설정
      setCurrentStep(6);
      
      // 결과 이미지가 있으면 표시
      if (result.details?.result_image) {
        setPersonImagePreview(result.details.result_image);
      }
      
    } catch (error) {
      console.error('Step 6 실행 실패:', error);
      setError(error instanceof Error ? error.message : 'Step 6 실행 중 오류가 발생했습니다.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Step 7 실행 함수 (후처리)
  const executeStep7 = async () => {
    setIsProcessing(true);
    setProgress(0);
    setProgressMessage('Step 7: 후처리 중...');
    
    try {
      // 🔥 세션 ID 확인 (더 안전한 방식)
      let currentSessionId = stepResults[6]?.session_id || 
                           stepResults[6]?.details?.session_id || 
                           stepResults[5]?.session_id ||
                           stepResults[5]?.details?.session_id ||
                           stepResults[4]?.session_id ||
                           stepResults[4]?.details?.session_id ||
                           stepResults[3]?.session_id ||
                           stepResults[3]?.details?.session_id ||
                           stepResults[2]?.session_id ||
                           stepResults[2]?.details?.session_id ||
                           stepResults[1]?.session_id ||
                           stepResults[1]?.details?.session_id ||
                           apiClient.getSessionId() ||
                           localStorage.getItem('mycloset_session_id');
      
      if (!currentSessionId) {
        throw new Error('세션 ID가 없습니다. Step 6을 먼저 완료해주세요.');
      }
      
      // 세션 ID가 있으면 apiClient에 설정
      apiClient.setSessionId(currentSessionId);
      console.log('🔍 Step 7 실행 - 세션 ID:', currentSessionId);
      
      // 🔥 Step 7: 세션 ID만 전송 (이미지와 측정값은 백엔드에서 세션에서 로드)
      const formData = new FormData();
      formData.append('session_id', currentSessionId);
      
      console.log('🚀 Step 7 후처리 시작:', {
        session_id: currentSessionId,
        endpoint: '/api/step/7/post-processing'
      });
      
      const result = await apiClient.callStepAPI(7, formData);
      
      if (!result.success) {
        throw new Error(`Step 7 실패: ${result.error || result.message}`);
      }
      
      setStepResults(prev => ({ ...prev, 7: result }));
      setCompletedSteps(prev => [...prev, 7]);
      
      // 🔥 Step 7 완료 후 currentStep을 7로 설정
      setCurrentStep(7);
      
      // 결과 이미지가 있으면 표시
      if (result.details?.result_image) {
        setPersonImagePreview(result.details.result_image);
      }
      
    } catch (error) {
      console.error('Step 7 실행 실패:', error);
      setError(error instanceof Error ? error.message : 'Step 7 실행 중 오류가 발생했습니다.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Step 8 실행 함수 (품질 평가)
  const executeStep8 = async () => {
    setIsProcessing(true);
    setProgress(0);
    setProgressMessage('Step 8: 품질 평가 중...');
    
    try {
      const formData = new FormData();
      formData.append('session_id', stepResults[7]?.session_id || apiClient.getSessionId() || '');
      
      const result = await apiClient.callStepAPI(8, formData);
      
      if (!result.success) {
        throw new Error(`Step 8 실패: ${result.error || result.message}`);
      }
      
      setStepResults(prev => ({ ...prev, 8: result }));
      setCompletedSteps(prev => [...prev, 8]);
      
      // 🔥 Step 8 완료 후 currentStep을 8로 설정
      setCurrentStep(8);
      
      // 최종 결과 설정
      if (result.fitted_image) {
        setResult({
          success: true,
          message: '가상 피팅 완료!',
          processing_time: result.processing_time,
          confidence: result.confidence,
          session_id: result.session_id || '',
          fit_score: result.fit_score || 0.85,
          measurements: {
            chest: measurements.height * 0.5,
            waist: measurements.height * 0.45,
            hip: measurements.height * 0.55,
            bmi: measurements.weight / Math.pow(measurements.height / 100, 2)
          },
          clothing_analysis: {
            category: 'shirt',
            style: 'casual',
            dominant_color: [255, 255, 255],
            material: 'cotton',
            pattern: 'solid'
          },
          recommendations: result.recommendations || ['피팅이 완료되었습니다!']
        });
      }
      
    } catch (error) {
      console.error('Step 8 실행 실패:', error);
      setError(error instanceof Error ? error.message : 'Step 8 실행 중 오류가 발생했습니다.');
    } finally {
      setIsProcessing(false);
    }
  };


  // =================================================================
  // 🔧 이펙트들
  // =================================================================

  // 반응형 처리
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // 서버 헬스체크
  useEffect(() => {
    const checkHealth = async () => {
      setIsCheckingHealth(true);
      try {
        console.log('🔍 서버 헬스체크 시작...');
        const result = await apiClient.healthCheck();
        
        console.log('📊 헬스체크 결과:', result);
        setIsServerHealthy(result.success);
        
        if (result.success && result.data) {
          console.log('✅ 서버 상태:', result.data);
        } else {
          console.log('❌ 서버 연결 실패:', result.error);
        }
      } catch (error) {
        console.error('❌ 헬스체크 실패:', error);
        setIsServerHealthy(false);
      } finally {
        setIsCheckingHealth(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [apiClient]);

  // 시스템 정보 조회
  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        const info = await apiClient.getSystemInfo();
        setSystemInfo(info);
        console.log('📊 시스템 정보:', info);
      } catch (error) {
        console.error('시스템 정보 조회 실패:', error);
      }
    };

    if (isServerHealthy) {
      fetchSystemInfo();
    }
  }, [isServerHealthy, apiClient]);

  // 진행률 콜백 설정
  useEffect(() => {
    apiClient.setProgressCallback((step, progressValue, message) => {
      setProgress(progressValue);
      setProgressMessage(message);
      console.log(`📊 Step ${step}: ${progressValue}% - ${message}`);
    });
  }, [apiClient]);


  // 컴포넌트 언마운트 시 WebSocket 정리
  useEffect(() => {
    return () => {
      apiClient.disconnectWebSocket();
      pipelineClient.cleanup();
    };
  }, [apiClient, pipelineClient]);

  // 개발자 콘솔 로그 (컴포넌트 마운트 시)
  useEffect(() => {
    console.log(`
🎉 MyCloset AI 완전 수정 버전 로드 완료!

✅ 모든 기능 포함:
- 8단계 AI 파이프라인 완전 지원
- 세션 기반 이미지 관리 (재업로드 방지)
- WebSocket 실시간 진행률 추적
- 완전한 에러 처리 및 재시도 로직
- 서킷 브레이커 패턴
- LRU 캐싱 시스템
- 요청 큐잉 시스템
- 모바일 완전 최적화
- M3 Max 128GB 최적화
- conda 환경 우선 지원

🔧 개발 도구 (데스크톱):
- Test: 서버 연결 테스트
- System: 시스템 정보 조회
- Complete: 전체 파이프라인 실행

📱 모바일 기능:
- 반응형 UI/UX
- 터치 최적화 인터페이스
- 진행률 오버레이
- 드래그 앤 드롭 지원

🚀 백엔드 호환성:
- FastAPI 완전 호환
- 모든 API 엔드포인트 지원
- WebSocket 실시간 통신
- SessionManager 연동

현재 상태: ${isServerHealthy ? '서버 연결됨' : '서버 연결 안됨'}
    `);
  }, [isServerHealthy]);


  // =================================================================
  // 🔧 이벤트 핸들러들
  // =================================================================

  // 파일 업로드 핸들러
  const handleImageUpload = useCallback(async (file: File, type: 'person' | 'clothing') => {
    const validation = fileUtils.validateImageFile(file);
    
    if (!validation.valid) {
      setFileErrors(prev => ({
        ...prev,
        [type]: validation.error
      }));
      return;
    }

    setFileErrors(prev => ({
      ...prev,
      [type]: undefined
    }));

    try {
      const preview = await fileUtils.createImagePreview(file);
      
      if (type === 'person') {
        setPersonImage(file);
        setPersonImagePreview(preview);
        console.log('✅ 사용자 이미지 업로드:', {
          name: file.name,
          size: fileUtils.formatFileSize(file.size),
          type: file.type
        });
      } else {
        setClothingImage(file);
        setClothingImagePreview(preview);
        console.log('✅ 의류 이미지 업로드:', {
          name: file.name,
          size: fileUtils.formatFileSize(file.size),
          type: file.type
        });
      }
      
      setError(null);
    } catch (error) {
      console.error('이미지 미리보기 생성 실패:', error);
      setFileErrors(prev => ({
        ...prev,
        [type]: '이미지 미리보기를 생성할 수 없습니다.'
      }));
    }
  }, []);

  // 드래그 앤 드롭
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent, type: 'person' | 'clothing') => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      handleImageUpload(files[0], type);
    }
  }, [handleImageUpload]);

  const goToNextStep = useCallback(() => {
    // 단순히 다음 단계로 이동만 (실행하지 않음)
    if (currentStep < 8) {
      setCurrentStep(prev => prev + 1);
    }
  }, [currentStep]);


  const goToPreviousStep = useCallback(() => {
    if (currentStep > 1) {
      setCurrentStep(prev => prev - 1);
      setCompletedSteps(prev => prev.filter(step => step < currentStep - 1));
    }
  }, [currentStep]);

  // 리셋
  const reset = useCallback(() => {
    setCurrentStep(0);
    setCompletedSteps([]);
    setPersonImage(null);
    setClothingImage(null);
    setPersonImagePreview(null);
    setClothingImagePreview(null);
    setStepResults({});
    setResult(null);
    setFileErrors({});
    setError(null);
    setIsProcessing(false);
    setProgress(0);
    setProgressMessage('');
    setHasValidatedImages(false); // 🔥 이미지 검증 상태 초기화
    apiClient.disconnectWebSocket();
    apiClient.setSessionId('');
    
    // 🔥 localStorage에서 세션 ID 제거
    localStorage.removeItem('mycloset_session_id');
    console.log('🗑️ 세션 ID 제거됨');
  }, [apiClient]);

  // 🔥 이미지 업로드 후 자동으로 검증 실행 (한 번만 실행되도록 수정)
  const [hasValidatedImages, setHasValidatedImages] = useState(false);
  
  useEffect(() => {
    if (personImage && clothingImage && measurements.height > 0 && measurements.weight > 0 && !hasValidatedImages) {
      console.log('🔄 이미지와 측정값이 모두 준비됨 - 자동 검증 시작');
      setHasValidatedImages(true);
      validateAndUploadImages();
    }
  }, [personImage, clothingImage, measurements.height, measurements.weight, hasValidatedImages]);

  const clearError = useCallback(() => setError(null), []);


  // =================================================================
  // 🔧 단계별 처리 함수들
  // =================================================================

  // 1단계: 이미지 업로드 검증
  const processStep1 = useCallback(async () => {
    console.log('🚀 processStep1 시작:', {
      personImage: !!personImage,
      clothingImage: !!clothingImage,
      personImageName: personImage?.name,
      clothingImageName: clothingImage?.name
    });

    if (!personImage || !clothingImage) {
      console.error('❌ 이미지 누락:', { personImage: !!personImage, clothingImage: !!clothingImage });
      setError('사용자 이미지와 의류 이미지를 모두 업로드해주세요.');
      return;
    }

    setIsProcessing(true);
    setProgress(10);
    setProgressMessage('이미지 검증 중...');

    try {
      const formData = new FormData();
      formData.append('person_image', personImage);
      formData.append('clothing_image', clothingImage);
      
      console.log('📤 Step 1 API 호출 준비 완료:', {
        formDataEntries: Array.from(formData.entries()).map(([key, value]) => ({
          key,
          valueType: value instanceof File ? 'File' : typeof value,
          fileName: value instanceof File ? value.name : value
        }))
      });
      
      setProgress(50);
      console.log('🌐 Step 1 API 호출 시작...');
      const stepResult = await apiClient.callStepAPI(1, formData);
      
      if (!stepResult.success) {
        throw new Error(stepResult.error || '1단계 검증 실패');
      }
      
      setStepResults(prev => ({ ...prev, 1: stepResult }));
      setProgress(100);
      setProgressMessage('이미지 검증 완료!');
      
      setTimeout(() => {
        setIsProcessing(false);
        goToNextStep();
      }, 1500);
      
    } catch (error: any) {
      console.error('❌ 1단계 실패:', error);
      setError(`1단계 실패: ${error.message}`);
      setIsProcessing(false);
      setProgress(0);
    }
  }, [personImage, clothingImage, apiClient, goToNextStep]);

 
  const processStep2 = useCallback(async () => {
  if (measurements.height <= 0 || measurements.weight <= 0) {
    setError('올바른 키와 몸무게를 입력해주세요.');
    return;
  }

  // ✅ 이제 TypeScript 오류 없이 작동
  const sessionId = 
    stepResults[1]?.session_id ||           // 최상위 레벨에서 먼저 확인
    stepResults[1]?.details?.session_id ||  // details에서 확인
    apiClient.getSessionId();               // API 클라이언트에서 확인
  
  console.log('🔍 Step 2 세션 ID 디버깅:', {
    'stepResults[1]': stepResults[1],
    'stepResults[1]?.session_id': stepResults[1]?.session_id,
    'stepResults[1]?.details?.session_id': stepResults[1]?.details?.session_id,
    'apiClient.getSessionId()': apiClient.getSessionId(),
    '최종_사용할_세션_ID': sessionId
  });
  
  if (!sessionId) {
    setError('세션 ID가 없습니다. 1단계부터 다시 시작해주세요.');
    return;
  }

  setIsProcessing(true);
  setProgress(10);
  setProgressMessage('신체 측정값 검증 중...');

  try {
    const formData = new FormData();
    
    formData.append('height', measurements.height.toString());
    formData.append('weight', measurements.weight.toString());
    formData.append('session_id', sessionId);
    
    formData.append('chest', '0');
    formData.append('waist', '0');
    formData.append('hips', '0');
    
    setProgress(50);
    const stepResult = await apiClient.callStepAPI(2, formData);
    
    if (!stepResult.success) {
      throw new Error(stepResult.error || '2단계 검증 실패');
    }
    
    setStepResults(prev => ({ ...prev, 2: stepResult }));
    setProgress(100);
    setProgressMessage('신체 측정값 검증 완료!');
    
    setTimeout(() => {
      setIsProcessing(false);
      goToNextStep();
    }, 1500);
    
  } catch (error: any) {
    console.error('❌ 2단계 실패:', error);
    
    let errorMessage = error.message;
    if (error.message.includes('422')) {
      errorMessage = '입력 데이터 형식이 올바르지 않습니다. 키와 몸무게를 다시 확인해주세요.';
    } else if (error.message.includes('404')) {
      errorMessage = '세션을 찾을 수 없습니다. 1단계부터 다시 시작해주세요.';
    }
    
    setError(`2단계 실패: ${errorMessage}`);
    setIsProcessing(false);
    setProgress(0);
  }
}, [measurements, apiClient, goToNextStep, stepResults]);

  // 유효성 검사 함수들
  const canProceedToNext = useCallback(() => {
    console.log('🔍 canProceedToNext 디버깅:', {
      currentStep,
      personImage: !!personImage,
      clothingImage: !!clothingImage,
      fileErrors,
      measurements,
      stepResults: Object.keys(stepResults),
      result: !!result
    });

    switch (currentStep) {
      case 0:
        // 이미지 업로드 단계: 두 이미지가 모두 업로드되고 에러가 없으면 진행 가능
        const step0CanProceed = personImage && clothingImage && 
               !fileErrors.person && !fileErrors.clothing;
        console.log('Step 0 조건 확인:', {
          personImage: !!personImage,
          clothingImage: !!clothingImage,
          fileErrors_person: fileErrors.person,
          fileErrors_clothing: fileErrors.clothing,
          canProceed: step0CanProceed
        });
        return step0CanProceed;
      case 1:
        // 치수 입력 단계: 키와 몸무게가 유효하면 진행 가능
        const step1CanProceed = measurements.height > 0 && measurements.weight > 0 &&
               measurements.height >= 100 && measurements.height <= 250 &&
               measurements.weight >= 30 && measurements.weight <= 300;
        console.log('Step 1 조건 확인:', {
          height: measurements.height,
          weight: measurements.weight,
          canProceed: step1CanProceed
        });
        return step1CanProceed;
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
        return stepResults[currentStep]?.success;
      case 8:
        return result?.success;
      case 9:
        return true;
      default:
        return false;
    }
  }, [currentStep, personImage, clothingImage, fileErrors, measurements, stepResults, result]);

  // 서버 상태 관련
  const getServerStatusColor = useCallback(() => {
    if (isCheckingHealth) return '#f59e0b';
    return isServerHealthy ? '#4ade80' : '#ef4444';
  }, [isCheckingHealth, isServerHealthy]);

  const getServerStatusText = useCallback(() => {
    if (isCheckingHealth) return 'Checking...';
    return isServerHealthy ? 'Server Online' : 'Server Offline';
  }, [isCheckingHealth, isServerHealthy]);

  // 개발 도구 함수들
  const handleTestConnection = useCallback(async () => {
    try {
      const result = await apiClient.healthCheck();
      console.log('연결 테스트 결과:', result);
      alert(result.success ? '✅ 연결 성공!' : `❌ 연결 실패: ${result.error}`);
    } catch (error) {
      console.error('연결 테스트 실패:', error);
      alert(`❌ 연결 테스트 실패: ${error}`);
    }
  }, [apiClient]);

  const handleSystemInfo = useCallback(async () => {
    try {
      const info = await apiClient.getSystemInfo();
      console.log('시스템 정보:', info);
      alert(`✅ ${info.app_name} v${info.app_version}\n🎯 ${info.device_name}\n💾 ${info.available_memory_gb}GB 사용가능`);
    } catch (error) {
      console.error('시스템 정보 실패:', error);
      alert(`❌ 시스템 정보 조회 실패: ${error}`);
    }
  }, [apiClient]);

  const handleCompletePipeline = useCallback(async () => {
  if (!personImage || !clothingImage) {
    setError('이미지를 먼저 업로드해주세요.');
    return;
  }
  
  setIsProcessing(true);
  setProgress(0);
  setProgressMessage('전체 파이프라인 시작...');
  setError(null);
  setResult(null);
  
  try {
    // �� pipeline API 호출 (전체 파이프라인 실행)
    const result = await apiClient.runCompletePipeline(
      personImage, 
      clothingImage, 
      measurements
    );
    
    if (result.success) {
      setResult(result);
      setProgress(100);
      setProgressMessage('🎉 전체 파이프라인 완료!');
    } else {
      throw new Error(result.message || '파이프라인 실행 실패');
    }
    
  } catch (error: any) {
    console.error('❌ 전체 파이프라인 실패:', error);
    setError(`전체 파이프라인 실패: ${error.message}`);
  } finally {
    setIsProcessing(false);
    setProgress(0);
    setProgressMessage('');
  }
}, [personImage, clothingImage, measurements, apiClient]);

// 🔥 이미지 업로드 검증 함수 추가

  const validateAndUploadImages = async () => {
    if (!personImage || !clothingImage) return;
    
    setIsProcessing(true);
    setProgress(0);
    setProgressMessage('이미지 업로드 검증 중...');
    
    try {
      // Step 1: 이미지 업로드 검증
      const validationFormData = new FormData();
      validationFormData.append('person_image', personImage);
      validationFormData.append('clothing_image', clothingImage);
      
      console.log('🚀 이미지 업로드 검증 시작');
      
      const validationResponse = await fetch('http://localhost:8000/api/step/upload/validation', {
        method: 'POST',
        body: validationFormData,
      });

      if (!validationResponse.ok) {
        const errorText = await validationResponse.text();
        throw new Error(`이미지 검증 실패: ${errorText}`);
      }

      const validationResult = await validationResponse.json();
      
      if (!validationResult.success) {
        throw new Error(`이미지 검증 실패: ${validationResult.error || validationResult.message}`);
      }
      
      // 세션 ID 저장
      if (validationResult.session_id) {
        apiClient.setSessionId(validationResult.session_id);
        // 🔥 localStorage에 세션 ID 저장 (새로고침 후 복구용)
        localStorage.setItem('mycloset_session_id', validationResult.session_id);
        console.log('💾 세션 ID 저장됨:', validationResult.session_id);
      }
      
      console.log('✅ 이미지 업로드 검증 완료:', validationResult);
      
      // Step 2: 신체 측정값 검증
      setProgress(50);
      setProgressMessage('신체 측정값 검증 중...');
      
      const measurementsFormData = new FormData();
      measurementsFormData.append('height', measurements.height.toString());
      measurementsFormData.append('weight', measurements.weight.toString());
      measurementsFormData.append('session_id', apiClient.getSessionId() || '');
      
      const measurementsResponse = await fetch('http://localhost:8000/api/step/upload/measurements', {
        method: 'POST',
        body: measurementsFormData,
      });

      if (!measurementsResponse.ok) {
        const errorText = await measurementsResponse.text();
        throw new Error(`측정값 검증 실패: ${errorText}`);
      }

      const measurementsResult = await measurementsResponse.json();
      
      if (!measurementsResult.success) {
        throw new Error(`측정값 검증 실패: ${measurementsResult.error || measurementsResult.message}`);
      }
      
      console.log('✅ 신체 측정값 검증 완료:', measurementsResult);
      
      setProgress(100);
      setProgressMessage('이미지 업로드 완료! Step 1을 실행할 수 있습니다.');
      
    } catch (error) {
      console.error('이미지 업로드 검증 실패:', error);
      setError(error instanceof Error ? error.message : '이미지 업로드 검증 중 오류가 발생했습니다.');
    } finally {
    setIsProcessing(false);
  }
  };

  // 요청 취소
  const handleCancelRequest = useCallback(() => {
    if (isProcessing) {
      setIsProcessing(false);
      setProgress(0);
      setProgressMessage('');
      apiClient.disconnectWebSocket();
    }
  }, [isProcessing, apiClient]);

  // 단계별 처리 함수 매핑
  const processCurrentStep = useCallback(async () => {
    const processors = {
      1: processStep1,
      2: processStep2
    };

    const processor = processors[currentStep as keyof typeof processors];
    if (processor) {
      await processor();
    }
  }, [currentStep, processStep1, processStep2]);

  // =================================================================
  // 🔧 렌더링 함수들 (모바일 최적화)
  // =================================================================

  const renderImageUploadStep = () => (
    <div style={{ 
      display: 'grid', 
      gridTemplateColumns: isMobile ? '1fr' : 'repeat(2, 1fr)', 
      gap: '1.5rem', 
      marginBottom: '2rem' 
    }}>
      {/* 이미지 업로드 섹션 */}
      {['person', 'clothing'].map((type) => (
        <div key={type} style={{ 
          backgroundColor: '#ffffff', 
          borderRadius: '0.75rem', 
          border: fileErrors[type as keyof typeof fileErrors] ? '2px solid #ef4444' : '1px solid #e5e7eb', 
          padding: '1.5rem' 
        }}>
          <h3 style={{ 
            fontSize: '1.125rem', 
            fontWeight: '500', 
            color: '#111827', 
            marginBottom: '1rem' 
          }}>{type === 'person' ? 'Your Photo' : 'Clothing Item'}</h3>
          
          {(type === 'person' ? personImagePreview : clothingImagePreview) ? (
            <div style={{ position: 'relative' }}>
              <img
                src={(type === 'person' ? personImagePreview : clothingImagePreview)!}
                alt={type}
                style={{ 
                  width: '100%', 
                  height: '16rem', 
                  objectFit: 'cover', 
                  borderRadius: '0.5rem' 
                }}
              />
              <button
                onClick={() => (type === 'person' ? personImageRef : clothingImageRef).current?.click()}
                style={{ 
                  position: 'absolute', 
                  top: '0.5rem', 
                  right: '0.5rem', 
                  backgroundColor: '#ffffff', 
                  borderRadius: '50%', 
                  padding: '0.5rem', 
                  boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)', 
                  border: 'none',
                  cursor: 'pointer'
                }}
              >
                📷
              </button>
            </div>
          ) : (
            <div 
              onClick={() => (type === 'person' ? personImageRef : clothingImageRef).current?.click()}
              onDragOver={handleDragOver}
              onDrop={(e) => handleDrop(e, type as 'person' | 'clothing')}
              style={{ 
                border: '2px dashed #d1d5db', 
                borderRadius: '0.5rem', 
                padding: '3rem', 
                textAlign: 'center', 
                cursor: 'pointer'
              }}
            >
              <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📷</div>
              <p style={{ fontSize: '0.875rem', color: '#4b5563', margin: 0 }}>
                Upload {type === 'person' ? 'your photo' : 'clothing item'}
              </p>
              <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: 0 }}>
                PNG, JPG, WebP up to 50MB
              </p>
            </div>
          )}
          
          {fileErrors[type as keyof typeof fileErrors] && (
            <div style={{ 
              marginTop: '0.5rem', 
              padding: '0.5rem', 
              backgroundColor: '#fef2f2', 
              border: '1px solid #fecaca', 
              borderRadius: '0.25rem', 
              fontSize: '0.875rem', 
              color: '#b91c1c' 
            }}>
              {fileErrors[type as keyof typeof fileErrors]}
            </div>
          )}
          
          <input
            ref={type === 'person' ? personImageRef : clothingImageRef}
            type="file"
            accept="image/*"
            onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0], type as 'person' | 'clothing')}
            style={{ display: 'none' }}
          />
        </div>
      ))}
      
      {/* 치수 입력 섹션 */}
      <div style={{ 
        gridColumn: isMobile ? '1' : '1 / -1',
        backgroundColor: '#ffffff', 
        borderRadius: '0.75rem', 
        border: '1px solid #e5e7eb', 
        padding: '1.5rem',
        marginTop: '1.5rem'
      }}>
        <h3 style={{ 
          fontSize: '1.125rem', 
          fontWeight: '500', 
          color: '#111827', 
          marginBottom: '1rem' 
        }}>Body Measurements</h3>
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: isMobile ? '1fr' : 'repeat(2, 1fr)', 
          gap: '1rem' 
        }}>
          <div>
            <label style={{ 
              display: 'block', 
              fontSize: '0.875rem', 
              fontWeight: '500', 
              color: '#374151', 
              marginBottom: '0.5rem' 
            }}>Height (cm)</label>
            <input
              type="number"
              value={measurements.height}
              onChange={(e) => setMeasurements(prev => ({ ...prev, height: Number(e.target.value) }))}
              style={{ 
                width: '100%', 
                padding: '0.5rem 0.75rem', 
                border: '1px solid #d1d5db', 
                borderRadius: '0.5rem', 
                fontSize: '0.875rem',
                outline: 'none'
              }}
              min="100"
              max="250"
              placeholder="170"
            />
          </div>
          <div>
            <label style={{ 
              display: 'block', 
              fontSize: '0.875rem', 
              fontWeight: '500', 
              color: '#374151', 
              marginBottom: '0.5rem' 
            }}>Weight (kg)</label>
            <input
              type="number"
              value={measurements.weight}
              onChange={(e) => setMeasurements(prev => ({ ...prev, weight: Number(e.target.value) }))}
              style={{ 
                width: '100%', 
                padding: '0.5rem 0.75rem', 
                border: '1px solid #d1d5db', 
                borderRadius: '0.5rem', 
                fontSize: '0.875rem',
                outline: 'none'
              }}
              min="30"
              max="300"
              placeholder="65"
            />
          </div>
        </div>
        
        {measurements.height > 0 && measurements.weight > 0 && (
          <div style={{ 
            marginTop: '1rem', 
            padding: '0.75rem', 
            backgroundColor: '#f9fafb', 
            borderRadius: '0.5rem' 
          }}>
            <div style={{ 
              fontSize: '0.875rem', 
              color: '#4b5563' 
            }}>
              BMI: {(measurements.weight / Math.pow(measurements.height / 100, 2)).toFixed(1)}
            </div>
          </div>
        )}
      </div>
      
      {/* Step 1 시작 버튼 */}
      <div style={{ 
        gridColumn: isMobile ? '1' : '1 / -1',
        marginTop: '2rem', 
        textAlign: 'center' 
      }}>
        <button
          onClick={() => setCurrentStep(1)}  // ✅ Step 1 (인체 파싱)으로 이동
          disabled={!personImage || !clothingImage || measurements.height <= 0 || measurements.weight <= 0 || isProcessing}
          style={{
            backgroundColor: '#000000',
            color: '#ffffff',
            border: 'none',
            borderRadius: '0.5rem',
            padding: '0.75rem 1.5rem',
            fontSize: '0.875rem',
            fontWeight: '500',
            cursor: 'pointer',
            transition: 'all 0.2s',
            width: isMobile ? '100%' : 'auto',
            minWidth: '200px'
          }}
        >
          {isProcessing ? 'AI 파이프라인 처리 중...' : '🚀 Step 1 시작 (인체 파싱)'}
        </button>
      </div>
    </div>
  );

  const renderMeasurementsStep = () => (
    <div style={{ 
      backgroundColor: '#ffffff', 
      borderRadius: '0.75rem', 
      border: '1px solid #e5e7eb', 
      padding: '1.5rem', 
      maxWidth: '28rem',
      margin: '0 auto'
    }}>
      <h3 style={{ 
        fontSize: '1.125rem', 
        fontWeight: '500', 
        color: '#111827', 
        marginBottom: '1rem' 
      }}>Body Measurements</h3>
      
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: isMobile ? '1fr' : 'repeat(2, 1fr)', 
        gap: '1rem' 
      }}>
        <div>
          <label style={{ 
            display: 'block', 
            fontSize: '0.875rem', 
            fontWeight: '500', 
            color: '#374151', 
            marginBottom: '0.5rem' 
          }}>Height (cm)</label>
          <input
            type="number"
            value={measurements.height}
            onChange={(e) => setMeasurements(prev => ({ ...prev, height: Number(e.target.value) }))}
            style={{ 
              width: '100%', 
              padding: '0.5rem 0.75rem', 
              border: '1px solid #d1d5db', 
              borderRadius: '0.5rem', 
              fontSize: '0.875rem',
              outline: 'none'
            }}
            min="100"
            max="250"
            placeholder="170"
          />
        </div>
        <div>
          <label style={{ 
            display: 'block', 
            fontSize: '0.875rem', 
            fontWeight: '500', 
            color: '#374151', 
            marginBottom: '0.5rem' 
          }}>Weight (kg)</label>
          <input
            type="number"
            value={measurements.weight}
            onChange={(e) => setMeasurements(prev => ({ ...prev, weight: Number(e.target.value) }))}
            style={{ 
              width: '100%', 
              padding: '0.5rem 0.75rem', 
              border: '1px solid #d1d5db', 
              borderRadius: '0.5rem', 
              fontSize: '0.875rem',
              outline: 'none'
            }}
            min="30"
            max="300"
            placeholder="65"
          />
        </div>
      </div>
      
      {measurements.height > 0 && measurements.weight > 0 && (
        <div style={{ 
          marginTop: '1rem', 
          padding: '0.75rem', 
          backgroundColor: '#f9fafb', 
          borderRadius: '0.5rem' 
        }}>
          <div style={{ 
            fontSize: '0.875rem', 
            color: '#4b5563' 
          }}>
            BMI: {(measurements.weight / Math.pow(measurements.height / 100, 2)).toFixed(1)}
          </div>
          
          {/* AI 가상 피팅 시작 버튼 */}
          <button
            onClick={executeStep1}
            style={{
              backgroundColor: '#000000',
              color: '#ffffff',
              border: 'none',
              borderRadius: '0.5rem',
              padding: '0.75rem 1.5rem',
              fontSize: '0.875rem',
              fontWeight: '500',
              cursor: 'pointer',
              transition: 'all 0.2s',
              width: '100%',
              marginTop: '1rem'
            }}
          >
            �� AI 가상 피팅 시작 (Step 1)
          </button>
        </div>
      )}
    </div>
  );

  const renderProcessingStep = () => {
    const stepData = PIPELINE_STEPS[currentStep - 1];
    const stepResult = stepResults[currentStep];

    return (
      <div style={{ 
        textAlign: 'center', 
        maxWidth: '40rem', 
        margin: '0 auto' 
      }}>
        <div style={{ 
          backgroundColor: '#ffffff', 
          borderRadius: '0.75rem', 
          border: '1px solid #e5e7eb', 
          padding: '2rem' 
        }}>
          <div style={{ 
            width: '4rem', 
            height: '4rem', 
            margin: '0 auto', 
            backgroundColor: '#eff6ff', 
            borderRadius: '50%', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            marginBottom: '1rem' 
          }}>
            {stepResult?.success ? (
              <span style={{ fontSize: '2rem' }}>✅</span>
            ) : isProcessing ? (
              <div style={{ 
                width: '2rem', 
                height: '2rem', 
                border: '4px solid #3b82f6', 
                borderTop: '4px solid transparent', 
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }}></div>
            ) : (
              <span style={{ fontSize: '1.25rem', fontWeight: '600', color: '#3b82f6' }}>
                {currentStep}
              </span>
            )}
          </div>
          
          <h3 style={{ 
            fontSize: '1.25rem', 
            fontWeight: '600', 
            color: '#111827' 
          }}>{stepData.name}</h3>
          
          <p style={{ 
            color: '#4b5563', 
            marginTop: '0.5rem'
          }}>{stepData.description}</p>

          {isProcessing && !stepResult && (
            <div style={{ 
              marginTop: '1rem', 
              padding: '1rem', 
              backgroundColor: '#fef3c7', 
              borderRadius: '0.5rem',
              border: '1px solid #f59e0b'
            }}>
              <p style={{ 
                fontSize: '0.875rem', 
                color: '#92400e', 
                margin: 0 
              }}>
                {progressMessage}
              </p>
              <div style={{ 
                width: '100%', 
                backgroundColor: '#f3f4f6', 
                borderRadius: '0.5rem', 
                height: '0.5rem',
                marginTop: '0.5rem'
              }}>
                <div 
                  style={{ 
                    backgroundColor: '#f59e0b', 
                    height: '0.5rem', 
                    borderRadius: '0.5rem', 
                    transition: 'width 0.3s',
                    width: `${progress}%`
                  }}
                ></div>
              </div>
            </div>
          )}

          {stepResult && (
            <div style={{ 
              marginTop: '1rem', 
              padding: '1rem', 
              backgroundColor: stepResult.success ? '#f0fdf4' : '#fef2f2', 
              borderRadius: '0.5rem',
              border: stepResult.success ? '1px solid #22c55e' : '1px solid #ef4444'
            }}>
              <p style={{ 
                fontSize: '0.875rem', 
                color: stepResult.success ? '#15803d' : '#dc2626',
                margin: '0 0 0.5rem 0',
                fontWeight: '500'
              }}>
                {stepResult.success ? '✅ ' : '❌ '}{stepResult.message}
              </p>
              
              {stepResult.success && (
                <p style={{ 
                  fontSize: '0.75rem', 
                  color: '#16a34a', 
                  margin: '0 0 0.5rem 0' 
                }}>
                  신뢰도: {(stepResult.confidence * 100).toFixed(1)}% | 
                  처리시간: {stepResult.processing_time.toFixed(2)}초
                </p>
              )}
              
              {stepResult.error && (
                <p style={{ 
                  fontSize: '0.75rem', 
                  color: '#dc2626', 
                  margin: '0.25rem 0 0 0' 
                }}>
                  오류: {stepResult.error}
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderVirtualFittingStep = () => (
    <div style={{ 
      textAlign: 'center', 
      maxWidth: '28rem', 
      margin: '0 auto' 
    }}>
      <div style={{ 
        backgroundColor: '#ffffff', 
        borderRadius: '0.75rem', 
        border: '1px solid #e5e7eb', 
        padding: '2rem' 
      }}>
        <div style={{ 
          width: '4rem', 
          height: '4rem', 
          margin: '0 auto', 
          backgroundColor: '#f3e8ff', 
          borderRadius: '50%', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          marginBottom: '1rem' 
        }}>
          {result?.success ? (
            <span style={{ fontSize: '2rem' }}>✅</span>
          ) : (
            <div style={{ 
              width: '2rem', 
              height: '2rem', 
              border: '4px solid #7c3aed', 
              borderTop: '4px solid transparent', 
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }}></div>
          )}
        </div>
        
        <h3 style={{ 
          fontSize: '1.25rem', 
          fontWeight: '600', 
          color: '#111827' 
        }}>AI 가상 피팅 생성</h3>
        
        <p style={{ 
          color: '#4b5563', 
          marginTop: '0.5rem'
        }}>딥러닝 모델이 최종 결과를 생성하고 있습니다</p>

        {isProcessing && (
          <div style={{ marginTop: '1rem' }}>
            <div style={{ 
              width: '100%', 
              backgroundColor: '#f3f4f6', 
              borderRadius: '0.5rem', 
              height: '0.75rem',
              marginBottom: '0.5rem'
            }}>
              <div 
                style={{ 
                  backgroundColor: '#7c3aed', 
                  height: '0.75rem', 
                  borderRadius: '0.5rem', 
                  transition: 'width 0.3s',
                  width: `${progress}%`
                }}
              ></div>
            </div>
            <p style={{ 
              fontSize: '0.875rem', 
              color: '#4b5563' 
            }}>{progressMessage}</p>
            
            <button
              onClick={handleCancelRequest}
              style={{
                marginTop: '1rem',
                padding: '0.5rem 1rem',
                backgroundColor: '#ef4444',
                color: '#ffffff',
                borderRadius: '0.5rem',
                border: 'none',
                cursor: 'pointer',
                fontSize: '0.875rem'
              }}
            >
              취소
            </button>
          </div>
        )}

        {result && (
          <div style={{ 
            marginTop: '1rem', 
            padding: '1rem', 
            backgroundColor: '#f0fdf4', 
            borderRadius: '0.5rem' 
          }}>
            <p style={{ 
              fontSize: '0.875rem', 
              color: '#15803d' 
            }}>가상 피팅 완성!</p>
            <p style={{ 
              fontSize: '0.75rem', 
              color: '#16a34a', 
              marginTop: '0.25rem' 
            }}>
              품질 점수: {(result.fit_score * 100).toFixed(1)}% | 
              처리시간: {result.processing_time.toFixed(1)}초
            </p>
          </div>
        )}
      </div>
    </div>
  );

  // 🔥 각 단계별 결과 표시 함수
  const renderStepResult = () => {
    const stepData = PIPELINE_STEPS[currentStep - 1];
    const stepResult = stepResults[currentStep];
    const isCompleted = completedSteps.includes(currentStep);
    const isCurrentStep = !isCompleted && !isProcessing;

    return (
      <div style={{ 
        textAlign: 'center', 
        maxWidth: '50rem', 
        margin: '0 auto' 
      }}>
        <div style={{ 
          backgroundColor: '#ffffff', 
          borderRadius: '0.75rem', 
          border: '1px solid #e5e7eb', 
          padding: '2rem' 
        }}>
          {/* Step 헤더 */}
          <div style={{ 
            width: '4rem', 
            height: '4rem', 
            margin: '0 auto', 
            backgroundColor: isCompleted ? '#dcfce7' : '#eff6ff', 
            borderRadius: '50%', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            marginBottom: '1rem' 
          }}>
            {isCompleted ? (
              <span style={{ fontSize: '2rem' }}>✅</span>
            ) : isProcessing ? (
              <div style={{ 
                width: '2rem', 
                height: '2rem', 
                border: '4px solid #3b82f6', 
                borderTop: '4px solid transparent', 
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }}></div>
            ) : (
              <span style={{ fontSize: '1.25rem', fontWeight: '600', color: '#3b82f6' }}>
                {currentStep}
              </span>
            )}
          </div>
          
          <h3 style={{ 
            fontSize: '1.25rem', 
            fontWeight: '600', 
            color: '#111827' 
          }}>{stepData.name}</h3>
          
          <p style={{ 
            color: '#4b5563', 
            marginTop: '0.5rem'
          }}>{stepData.description}</p>

          {/* 진행 상태 표시 */}
          {isProcessing && (
            <div style={{ 
              marginTop: '1rem', 
              padding: '1rem', 
              backgroundColor: '#fef3c7', 
              borderRadius: '0.5rem',
              border: '1px solid #f59e0b'
            }}>
              <p style={{ 
                fontSize: '0.875rem', 
                color: '#92400e', 
                margin: 0 
              }}>
                {progressMessage}
              </p>
              <div style={{ 
                width: '100%', 
                backgroundColor: '#f3f4f6', 
                borderRadius: '0.5rem', 
                height: '0.5rem',
                marginTop: '0.5rem'
              }}>
                <div 
                  style={{ 
                    backgroundColor: '#f59e0b', 
                    height: '0.5rem', 
                    borderRadius: '0.5rem', 
                    transition: 'width 0.3s',
                    width: `${progress}%`
                  }}
                ></div>
              </div>
            </div>
          )}

          {/* 결과 이미지 표시 */}
          {stepResult?.details?.result_image && (
            <div style={{ 
              marginTop: '1.5rem',
              padding: '1rem',
              backgroundColor: '#f9fafb',
              borderRadius: '0.5rem'
            }}>
              <h4 style={{ 
                fontSize: '1rem', 
                fontWeight: '500', 
                color: '#374151',
                marginBottom: '1rem'
              }}>처리 결과</h4>
              <img 
                src={stepResult.details.result_image} 
                alt={`Step ${currentStep} 결과`}
                style={{ 
                  maxWidth: '100%', 
                  height: 'auto',
                  borderRadius: '0.5rem',
                  border: '1px solid #e5e7eb'
                }}
              />
            </div>
          )}

          {/* 버튼들 */}
          <div style={{ 
            marginTop: '1.5rem',
            display: 'flex',
            gap: '1rem',
            justifyContent: 'center',
            flexWrap: 'wrap'
          }}>
            {/* 이전 단계 버튼 */}
            {currentStep > 1 && (
              <button
                onClick={() => setCurrentStep(currentStep - 1)}
                style={{
                  backgroundColor: '#f3f4f6',
                  color: '#374151',
                  border: '1px solid #d1d5db',
                  borderRadius: '0.5rem',
                  padding: '0.75rem 1.5rem',
                  fontSize: '0.875rem',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                ← 이전 단계
              </button>
            )}

            {/* 현재 단계 실행 버튼 */}
            {isCurrentStep && (
              <button
                onClick={() => {
                  const stepFunctions = [
                    executeStep1, executeStep2, executeStep3, executeStep4,
                    executeStep5, executeStep6, executeStep7, executeStep8
                  ];
                  stepFunctions[currentStep - 1]();
                }}
                style={{
                  backgroundColor: '#000000',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '0.5rem',
                  padding: '0.75rem 1.5rem',
                  fontSize: '0.875rem',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                🚀 Step {currentStep} 실행
              </button>
            )}

            {/* 다음 단계 버튼 */}
            {isCompleted && currentStep < 8 && (
              <button
                onClick={() => setCurrentStep(currentStep + 1)}
                style={{
                  backgroundColor: '#000000',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '0.5rem',
                  padding: '0.75rem 1.5rem',
                  fontSize: '0.875rem',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                다음 단계 →
              </button>
            )}

            {/* 최종 결과 보기 버튼 */}
            {isCompleted && currentStep === 8 && (
              <button
                onClick={() => setCurrentStep(9)}
                style={{
                  backgroundColor: '#059669',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '0.5rem',
                  padding: '0.75rem 1.5rem',
                  fontSize: '0.875rem',
                  fontWeight: '500',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                🎉 최종 결과 보기
              </button>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderResultStep = () => {
    if (!result) return null;

    return (
      <div style={{ 
        maxWidth: '80rem', 
        margin: '0 auto' 
      }}>
        <VirtualFittingResultVisualization result={result} />
      </div>
    );
  };

  // 🔥 새로운 단계별 컨텐츠 렌더링 (Step 1-8)
  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return renderImageUploadStep();     // Step 0: 이미지 업로드 + 치수 입력
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
        return renderStepResult();          // Step 1-8: 각 단계별 결과 표시
      case 9:
        return renderResultStep();          // Step 9: 최종 결과
      default:
        return renderImageUploadStep();
    }
  };

  // =================================================================
  // 🔧 메인 렌더링 (완전한 UI/UX + 모바일 최적화)
  // =================================================================

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#f9fafb', 
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif' 
    }}>
      {/* CSS Animations */}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>

      {/* Header */}
      <header style={{ 
        backgroundColor: '#ffffff', 
        borderBottom: '1px solid #e5e7eb',
        position: 'sticky',
        top: 0,
        zIndex: 50
      }}>
        <div style={{ 
          maxWidth: '80rem', 
          margin: '0 auto', 
          padding: isMobile ? '0 0.75rem' : '0 1rem' 
        }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            height: isMobile ? '3.5rem' : '4rem' 
          }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{ flexShrink: 0 }}>
                <div style={{ 
                  width: isMobile ? '1.75rem' : '2rem', 
                  height: isMobile ? '1.75rem' : '2rem', 
                  backgroundColor: '#000000', 
                  borderRadius: '0.5rem', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center' 
                }}>
                  <svg style={{ 
                    width: isMobile ? '1rem' : '1.25rem', 
                    height: isMobile ? '1rem' : '1.25rem', 
                    color: '#ffffff' 
                  }} fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2L2 7v10c0 5.55 3.84 10 9 10s9-4.45 9-10V7l-10-5z"/>
                  </svg>
                </div>
              </div>
              <div style={{ marginLeft: '0.75rem' }}>
                <h1 style={{ 
                  fontSize: isMobile ? '1.125rem' : '1.25rem', 
                  fontWeight: '600', 
                  color: '#111827', 
                  margin: 0 
                }}>MyCloset AI</h1>
                <p style={{ 
                  fontSize: isMobile ? '0.625rem' : '0.75rem', 
                  color: '#6b7280', 
                  margin: 0 
                }}>
                  {systemInfo ? 
                    `${systemInfo.device_name} ${systemInfo.is_m3_max ? '🍎' : ''}` : 
                    '완전 수정 버전 (포트 8000)'
                  }
                </p>
              </div>
            </div>
            
            {/* 서버 상태 및 개발 도구 */}
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: isMobile ? '0.5rem' : '1rem' 
            }}>
              {/* 개발 도구 버튼들 - 데스크톱에서만 표시 */}
              {!isMobile && (
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    onClick={handleTestConnection}
                    disabled={isProcessing}
                    style={{
                      padding: '0.25rem 0.5rem',
                      fontSize: '0.75rem',
                      backgroundColor: '#e5e7eb',
                      color: '#374151',
                      border: 'none',
                      borderRadius: '0.25rem',
                      cursor: isProcessing ? 'not-allowed' : 'pointer',
                      opacity: isProcessing ? 0.5 : 1
                    }}
                  >
                    Test
                  </button>
                  <button
                    onClick={handleSystemInfo}
                    disabled={isProcessing}
                    style={{
                      padding: '0.25rem 0.5rem',
                      fontSize: '0.75rem',
                      backgroundColor: '#e5e7eb',
                      color: '#374151',
                      border: 'none',
                      borderRadius: '0.25rem',
                      cursor: isProcessing ? 'not-allowed' : 'pointer',
                      opacity: isProcessing ? 0.5 : 1
                    }}
                  >
                    System
                  </button>
                  <button
                    onClick={handleCompletePipeline}
                    disabled={isProcessing || !personImage || !clothingImage}
                    style={{
                      padding: '0.25rem 0.5rem',
                      fontSize: '0.75rem',
                      backgroundColor: isProcessing || !personImage || !clothingImage ? '#d1d5db' : '#3b82f6',
                      color: '#ffffff',
                      border: 'none',
                      borderRadius: '0.25rem',
                      cursor: isProcessing || !personImage || !clothingImage ? 'not-allowed' : 'pointer',
                      opacity: isProcessing || !personImage || !clothingImage ? 0.5 : 1
                    }}
                  >
                    Complete
                  </button>
                </div>
              )}

              {/* 진행률 표시 (처리 중일 때) */}
              {isProcessing && (
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '0.5rem' 
                }}>
                  <div style={{ 
                    width: isMobile ? '0.625rem' : '0.75rem', 
                    height: isMobile ? '0.625rem' : '0.75rem', 
                    border: '2px solid #3b82f6', 
                    borderTop: '2px solid transparent', 
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite'
                  }}></div>
                  <span style={{ 
                    fontSize: isMobile ? '0.75rem' : '0.875rem', 
                    color: '#4b5563' 
                  }}>
                    {progress}%
                  </span>
                </div>
              )}

              {/* 서버 상태 */}
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '0.5rem' 
              }}>
                <div style={{ 
                  height: '0.5rem', 
                  width: '0.5rem', 
                  backgroundColor: getServerStatusColor(),
                  borderRadius: '50%',
                  transition: 'background-color 0.3s'
                }}></div>
                {!isMobile && (
                  <span style={{ 
                    fontSize: '0.875rem', 
                    color: '#4b5563' 
                  }}>
                    {getServerStatusText()}
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div style={{
          backgroundColor: '#fef2f2',
          borderBottom: '1px solid #fecaca',
          padding: '0.75rem'
        }}>
          <div style={{
            maxWidth: '80rem',
            margin: '0 auto',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <span style={{ fontSize: '1.25rem', marginRight: '0.5rem' }}>⚠️</span>
              <span style={{ color: '#b91c1c', fontSize: '0.875rem' }}>{error}</span>
            </div>
            <button
              onClick={clearError}
              style={{
                color: '#b91c1c',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                fontSize: '1.25rem'
              }}
            >
              ×
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main style={{ 
        maxWidth: '80rem', 
        margin: '0 auto', 
        padding: isMobile ? '1rem 0.75rem' : '2rem 1rem' 
      }}>
        {/* Progress Bar */}
        <div style={{ marginBottom: isMobile ? '1.5rem' : '2rem' }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between', 
            marginBottom: '1rem',
            flexDirection: isMobile ? 'column' : 'row',
            gap: isMobile ? '0.5rem' : '0'
          }}>
            <h2 style={{ 
              fontSize: isMobile ? '1.5rem' : '1.875rem', 
              fontWeight: '700', 
              color: '#111827', 
              margin: 0 
            }}>AI Virtual Try-On</h2>
            <span style={{ 
              fontSize: isMobile ? '0.75rem' : '0.875rem', 
              color: '#4b5563' 
            }}>Step {currentStep} of 8</span>
          </div>
          
          {/* Step Progress */}
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: isMobile ? '0.5rem' : '1rem', 
            marginBottom: '1.5rem',
            overflowX: 'auto',
            paddingBottom: isMobile ? '0.5rem' : '0'
          }}>
            {PIPELINE_STEPS.map((step, index) => (
              <div key={step.id} style={{ 
                display: 'flex', 
                alignItems: 'center',
                flexShrink: 0
              }}>
                <div 
                  style={{
                    width: isMobile ? '1.5rem' : '2rem', 
                    height: isMobile ? '1.5rem' : '2rem', 
                    borderRadius: '50%', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    fontSize: isMobile ? '0.75rem' : '0.875rem', 
                    fontWeight: '500',
                    backgroundColor: completedSteps.includes(step.id) 
                      ? '#22c55e' 
                      : currentStep === step.id 
                        ? '#3b82f6' 
                        : '#e5e7eb',
                    color: completedSteps.includes(step.id) || currentStep === step.id 
                      ? '#ffffff' 
                      : '#4b5563'
                  }}
                >
                  {completedSteps.includes(step.id) ? (
                    '✓'
                  ) : (
                    step.id
                  )}
                </div>
                {index < PIPELINE_STEPS.length - 1 && (
                  <div 
                    style={{
                      width: isMobile ? '1.5rem' : '3rem', 
                      height: '2px', 
                      marginLeft: isMobile ? '0.25rem' : '0.5rem', 
                      marginRight: isMobile ? '0.25rem' : '0.5rem',
                      backgroundColor: completedSteps.includes(step.id) ? '#22c55e' : '#e5e7eb'
                    }}
                  ></div>
                )}
              </div>
            ))}
          </div>

          {/* Current Step Info */}
          <div style={{ 
            backgroundColor: '#eff6ff', 
            border: '1px solid #bfdbfe', 
            borderRadius: '0.5rem', 
            padding: isMobile ? '0.75rem' : '1rem' 
          }}>
            <h3 style={{ 
              fontWeight: '600', 
              color: '#1e40af', 
              margin: 0,
              fontSize: isMobile ? '0.875rem' : '1rem'
            }}>{PIPELINE_STEPS[currentStep - 1]?.name}</h3>
            <p style={{ 
              color: '#1d4ed8', 
              fontSize: isMobile ? '0.75rem' : '0.875rem', 
              marginTop: '0.25rem', 
              margin: 0 
            }}>{PIPELINE_STEPS[currentStep - 1]?.description}</p>
          </div>
        </div>

        {/* Step Content */}
        <div style={{ marginBottom: isMobile ? '1.5rem' : '2rem' }}>
          {renderStepContent()}
        </div>

        {/* Navigation Buttons */}
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          flexDirection: isMobile ? 'column' : 'row',
          gap: isMobile ? '1rem' : '0'
        }}>
          <button
            onClick={goToPreviousStep}
            disabled={currentStep === 1 || isProcessing}            style={{
              padding: isMobile ? '0.875rem 1.5rem' : '0.75rem 1.5rem',
              backgroundColor: '#f3f4f6',
              color: '#374151',
              borderRadius: '0.5rem',
              fontWeight: '500',
              border: 'none',
              cursor: (currentStep === 1 || isProcessing) ? 'not-allowed' : 'pointer',
              opacity: (currentStep === 1 || isProcessing) ? 0.5 : 1,
              transition: 'all 0.2s',
              order: isMobile ? 2 : 1,
              width: isMobile ? '100%' : 'auto'
            }}
          >
            이전 단계
          </button>

          <div style={{ 
            display: 'flex', 
            gap: '0.75rem',
            order: isMobile ? 1 : 2,
            flexDirection: isMobile ? 'column' : 'row'
          }}>
            {/* 리셋 버튼 */}
            {!isProcessing && (currentStep > 1 || result) && (
              <button
                onClick={reset}
                style={{
                  padding: isMobile ? '0.875rem 1.5rem' : '0.75rem 1.5rem',
                  backgroundColor: '#6b7280',
                  color: '#ffffff',
                  borderRadius: '0.5rem',
                  fontWeight: '500',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  width: isMobile ? '100%' : 'auto'
                }}
              >
                처음부터
              </button>
            )}

            {currentStep <= 2 && (
              <button
                onClick={() => {
                      if (currentStep === 0) executeStep1();  // Step 0에서 Step 1 실행
                      else if (currentStep === 1) executeStep2();  // Step 1에서 Step 2 실행
                    }}               
                disabled={!canProceedToNext() || isProcessing}
                style={{
                  padding: isMobile ? '0.875rem 1.5rem' : '0.75rem 1.5rem',
                  backgroundColor: (!canProceedToNext() || isProcessing) ? '#d1d5db' : '#3b82f6',
                  color: '#ffffff',
                  borderRadius: '0.5rem',
                  fontWeight: '500',
                  border: 'none',
                  cursor: (!canProceedToNext() || isProcessing ) ? 'not-allowed' : 'pointer',
                  opacity: (!canProceedToNext() || isProcessing ) ? 0.5 : 1,
                  transition: 'all 0.2s',
                  width: isMobile ? '100%' : 'auto'
                }}
              >
                {isProcessing ? '처리 중...' : '다음 단계'}
              </button>
            )}

            {currentStep >=2 && currentStep < 8 && !isProcessing && (
              <button
                onClick={() => {
                  // 단순히 다음 단계로 이동만 (실행하지 않음)
                  if (currentStep < 8) {
                    setCurrentStep(prev => prev + 1);
                  }
                }}
                disabled={!canProceedToNext() || isProcessing}
                style={{
                  padding: isMobile ? '0.875rem 1.5rem' : '0.75rem 1.5rem',
                  backgroundColor: (!canProceedToNext() || isProcessing) ? '#d1d5db' : '#10b981',
                  color: '#ffffff',
                  borderRadius: '0.5rem',
                  fontWeight: '500',
                  border: 'none',
                  cursor: (!canProceedToNext() || isProcessing) ? 'not-allowed' : 'pointer',
                  opacity: (!canProceedToNext() || isProcessing) ? 0.5 : 1,
                  transition: 'all 0.2s',
                  width: isMobile ? '100%' : 'auto'
                }}
              >
                다음 단계
              </button>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer style={{
        backgroundColor: '#ffffff',
        borderTop: '1px solid #e5e7eb',
        padding: isMobile ? '1rem 0.75rem' : '1.5rem 1rem',
        marginTop: '2rem'
      }}>
        <div style={{
          maxWidth: '80rem',
          margin: '0 auto',
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
          justifyContent: 'space-between',
          alignItems: isMobile ? 'flex-start' : 'center',
          gap: isMobile ? '1rem' : '0'
        }}>
          <div style={{
            fontSize: '0.875rem',
            color: '#6b7280'
          }}>
            <p style={{ margin: 0 }}>
              🤖 MyCloset AI v2.0.0 - 완전 수정 버전
            </p>
            <p style={{ margin: 0, fontSize: '0.75rem' }}>
              {systemInfo ? 
                `M3 Max 128GB 최적화 • ${systemInfo.available_memory_gb}GB 사용가능` : 
                'AI 파이프라인 완전 연동 • 8단계 처리 지원'
              }
            </p>
          </div>
          
          {!isMobile && (
            <div style={{
              display: 'flex',
              gap: '1rem',
              fontSize: '0.875rem',
              color: '#6b7280'
            }}>
              <span>포트: 8000</span>
              <span>•</span>
              <span>WebSocket: {isServerHealthy ? '연결됨' : '연결 안됨'}</span>
              <span>•</span>
              <span>캐시: 활성화</span>
            </div>
          )}
        </div>
      </footer>

      {/* 모바일 진행률 오버레이 */}
      {isMobile && isProcessing && (
        <div style={{
          position: 'fixed',
          bottom: '1rem',
          left: '1rem',
          right: '1rem',
          backgroundColor: '#ffffff',
          borderRadius: '0.75rem',
          boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
          padding: '1rem',
          border: '1px solid #e5e7eb',
          zIndex: 100
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem'
          }}>
            <div style={{
              width: '1rem',
              height: '1rem',
              border: '2px solid #3b82f6',
              borderTop: '2px solid transparent',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }}></div>
            <div style={{ flex: 1 }}>
              <div style={{
                fontSize: '0.875rem',
                fontWeight: '500',
                color: '#111827',
                marginBottom: '0.25rem'
              }}>
                {progressMessage}
              </div>
              <div style={{
                width: '100%',
                backgroundColor: '#f3f4f6',
                borderRadius: '9999px',
                height: '0.5rem'
              }}>
                <div style={{
                  backgroundColor: '#3b82f6',
                  height: '0.5rem',
                  borderRadius: '9999px',
                  transition: 'width 0.3s',
                  width: `${progress}%`
                }}></div>
              </div>
            </div>
            <button
              onClick={handleCancelRequest}
              style={{
                padding: '0.5rem',
                backgroundColor: '#ef4444',
                color: '#ffffff',
                borderRadius: '0.375rem',
                border: 'none',
                cursor: 'pointer',
                fontSize: '0.75rem'
              }}
            >
              취소
            </button>
          </div>
        </div>
      )}


    </div>
  );
};

export default App;