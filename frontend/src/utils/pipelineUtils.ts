/**
 * MyCloset AI 파이프라인 유틸리티 클래스
 * 실제 백엔드와 완전 호환되는 프로덕션 수준 유틸리티
 * - 시스템 감지 및 최적화
 * - 에러 처리 및 로깅
 * - 성능 모니터링
 * - 브라우저 호환성 체크
 */

import type {
  BrowserCompatibility,
  SystemRequirements,
  DeviceType,
  QualityLevel,
  APIError,
  PipelineEvent,
  UsePipelineOptions,
  ProcessingStatusEnum,
} from '../types/pipeline';

// =================================================================
// 🔧 로깅 시스템
// =================================================================

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
  CRITICAL = 4
}

export interface LogEntry {
  level: LogLevel;
  message: string;
  data?: any;
  timestamp: string;
  source: string;
  sessionId?: string;
  taskId?: string;
}

class Logger {
  private static instance: Logger;
  private logs: LogEntry[] = [];
  private maxLogs = 1000;
  private logLevel = LogLevel.INFO;
  private enableConsole = true;
  private enableStorage = true;

  static getInstance(): Logger {
    if (!Logger.instance) {
      Logger.instance = new Logger();
    }
    return Logger.instance;
  }

  setLogLevel(level: LogLevel): void {
    this.logLevel = level;
  }

  setMaxLogs(max: number): void {
    this.maxLogs = max;
  }

  enableConsoleOutput(enable: boolean): void {
    this.enableConsole = enable;
  }

  enableStorageLogging(enable: boolean): void {
    this.enableStorage = enable;
  }

  log(level: LogLevel, message: string, data?: any, source = 'PipelineUtils'): void {
    if (level < this.logLevel) return;

    const entry: LogEntry = {
      level,
      message,
      data,
      timestamp: new Date().toISOString(),
      source,
      sessionId: PipelineUtils.getCurrentSessionId(),
      taskId: PipelineUtils.getCurrentTaskId(),
    };

    // 콘솔 출력
    if (this.enableConsole) {
      const levelName = LogLevel[level];
      const emoji = this.getLevelEmoji(level);
      const style = this.getLevelStyle(level);
      
      if (typeof window !== 'undefined' && console) {
        console[this.getConsoleMethod(level)](
          `%c${emoji} [${levelName}] ${entry.timestamp} [${source}]`,
          style,
          message,
          data || ''
        );
      }
    }

    // 메모리 저장
    if (this.enableStorage) {
      this.logs.push(entry);
      if (this.logs.length > this.maxLogs) {
        this.logs = this.logs.slice(-this.maxLogs);
      }
    }
  }

  private getLevelEmoji(level: LogLevel): string {
    const emojis = {
      [LogLevel.DEBUG]: '🐛',
      [LogLevel.INFO]: 'ℹ️',
      [LogLevel.WARN]: '⚠️',
      [LogLevel.ERROR]: '❌',
      [LogLevel.CRITICAL]: '🚨'
    };
    return emojis[level] || 'ℹ️';
  }

  private getLevelStyle(level: LogLevel): string {
    const styles = {
      [LogLevel.DEBUG]: 'color: #888; font-size: 11px;',
      [LogLevel.INFO]: 'color: #2196F3; font-weight: bold;',
      [LogLevel.WARN]: 'color: #FF9800; font-weight: bold;',
      [LogLevel.ERROR]: 'color: #F44336; font-weight: bold;',
      [LogLevel.CRITICAL]: 'color: #F44336; font-weight: bold; background: #FFEBEE; padding: 2px 4px;'
    };
    return styles[level] || styles[LogLevel.INFO];
  }

  private getConsoleMethod(level: LogLevel): 'log' | 'warn' | 'error' {
    if (level >= LogLevel.ERROR) return 'error';
    if (level >= LogLevel.WARN) return 'warn';
    return 'log';
  }

  getLogs(level?: LogLevel): LogEntry[] {
    if (level === undefined) return [...this.logs];
    return this.logs.filter(log => log.level >= level);
  }

  clearLogs(): void {
    this.logs = [];
  }

  exportLogs(): string {
    return JSON.stringify(this.logs, null, 2);
  }
}

// =================================================================
// 🔧 성능 모니터링
// =================================================================

export interface PerformanceTimer {
  name: string;
  start: number;
  end(): number;
  mark(label: string): void;
  getMarks(): Record<string, number>;
  getDuration(): number;
}

class PerformanceMonitor {
  private static instance: PerformanceMonitor;
  private timers: Map<string, PerformanceTimer> = new Map();
  private metrics: Map<string, number[]> = new Map();

  static getInstance(): PerformanceMonitor {
    if (!PerformanceMonitor.instance) {
      PerformanceMonitor.instance = new PerformanceMonitor();
    }
    return PerformanceMonitor.instance;
  }

  createTimer(name: string): PerformanceTimer {
    const start = performance.now();
    const marks: Record<string, number> = {};

    const timer: PerformanceTimer = {
      name,
      start,
      end: () => {
        const duration = performance.now() - start;
        this.recordMetric(name, duration);
        this.timers.delete(name);
        return duration;
      },
      mark: (label: string) => {
        marks[label] = performance.now() - start;
      },
      getMarks: () => ({ ...marks }),
      getDuration: () => performance.now() - start
    };

    this.timers.set(name, timer);
    return timer;
  }

  recordMetric(name: string, value: number): void {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    
    const values = this.metrics.get(name)!;
    values.push(value);
    
    // 최대 100개 값만 유지
    if (values.length > 100) {
      values.shift();
    }
  }

  getMetricStats(name: string): {
    count: number;
    average: number;
    min: number;
    max: number;
    median: number;
    p95: number;
  } | null {
    const values = this.metrics.get(name);
    if (!values || values.length === 0) return null;

    const sorted = [...values].sort((a, b) => a - b);
    const count = values.length;
    const sum = values.reduce((a, b) => a + b, 0);

    return {
      count,
      average: sum / count,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      median: sorted[Math.floor(count / 2)],
      p95: sorted[Math.floor(count * 0.95)]
    };
  }

  getAllMetrics(): Record<string, any> {
    const result: Record<string, any> = {};
    for (const [name] of this.metrics) {
      result[name] = this.getMetricStats(name);
    }
    return result;
  }

  clearMetrics(): void {
    this.metrics.clear();
  }
}

// =================================================================
// 🔧 메인 PipelineUtils 클래스
// =================================================================

export class PipelineUtils {
  private static logger = Logger.getInstance();
  private static perfMonitor = PerformanceMonitor.getInstance();
  private static currentSessionId: string | null = null;
  private static currentTaskId: string | null = null;
  private static eventListeners: Map<string, Function[]> = new Map();

  // =================================================================
  // 🔧 로깅 메서드들
  // =================================================================

  static log(level: 'debug' | 'info' | 'warn' | 'error' | 'critical', message: string, data?: any): void {
    const logLevel = {
      debug: LogLevel.DEBUG,
      info: LogLevel.INFO,
      warn: LogLevel.WARN,
      error: LogLevel.ERROR,
      critical: LogLevel.CRITICAL
    }[level];

    this.logger.log(logLevel, message, data);
  }

  static debug(message: string, data?: any): void {
    this.log('debug', message, data);
  }

  static info(message: string, data?: any): void {
    this.log('info', message, data);
  }

  static warn(message: string, data?: any): void {
    this.log('warn', message, data);
  }

  static error(message: string, data?: any): void {
    this.log('error', message, data);
  }

  static critical(message: string, data?: any): void {
    this.log('critical', message, data);
  }

  static setLogLevel(level: LogLevel): void {
    this.logger.setLogLevel(level);
  }

  static getLogs(level?: LogLevel): LogEntry[] {
    return this.logger.getLogs(level);
  }

  static clearLogs(): void {
    this.logger.clearLogs();
  }

  static exportLogs(): string {
    return this.logger.exportLogs();
  }

  // =================================================================
  // 🔧 성능 모니터링 메서드들
  // =================================================================

  static createPerformanceTimer(name: string): PerformanceTimer {
    return this.perfMonitor.createTimer(name);
  }

  static recordMetric(name: string, value: number): void {
    this.perfMonitor.recordMetric(name, value);
  }

  static getPerformanceStats(name: string): any {
    return this.perfMonitor.getMetricStats(name);
  }

  static getAllPerformanceMetrics(): Record<string, any> {
    return this.perfMonitor.getAllMetrics();
  }

  // =================================================================
  // 🔧 시스템 감지 메서드들
  // =================================================================

  static autoDetectDevice(): DeviceType {
    if (typeof window === 'undefined') return 'auto';

    // GPU 감지 시도
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      
      if (gl) {
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (debugInfo) {
          const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
          if (renderer.toLowerCase().includes('nvidia') || 
              renderer.toLowerCase().includes('amd') || 
              renderer.toLowerCase().includes('radeon')) {
            return 'cuda';
          }
        }
      }
    } catch (error) {
      this.warn('GPU 감지 실패', error);
    }

    // Apple Silicon 감지
    if (this.detectM3Max()) {
      return 'mps';
    }

    // 기본값
    return 'cpu';
  }

  static autoDetectDeviceType(): DeviceType {
    if (typeof navigator === 'undefined') return 'auto';

    const platform = navigator.platform.toLowerCase();
    const userAgent = navigator.userAgent.toLowerCase();

    if (platform.includes('mac') || userAgent.includes('macintosh')) {
      return 'mac';
    } else if (platform.includes('win') || userAgent.includes('windows')) {
      return 'pc';
    } else if (platform.includes('linux') || userAgent.includes('linux')) {
      return 'pc';
    }

    return 'auto';
  }

  static detectM3Max(): boolean {
    if (typeof navigator === 'undefined') return false;

    const platform = navigator.platform.toLowerCase();
    const userAgent = navigator.userAgent.toLowerCase();
    
    // Mac 플랫폼 확인
    const isMac = platform.includes('mac') || userAgent.includes('macintosh');
    
    // 고성능 Mac 감지 (M1/M2/M3 Pro/Max)
    const hasHighCoreCount = navigator.hardwareConcurrency >= 8;
    const hasHighMemory = (navigator as any).deviceMemory >= 16;
    
    return isMac && hasHighCoreCount && (hasHighMemory || navigator.hardwareConcurrency >= 10);
  }

  static getSystemParams(): Map<string, any> {
    const params = new Map<string, any>();
    
    params.set('device', this.autoDetectDevice());
    params.set('device_type', this.autoDetectDeviceType());
    params.set('is_m3_max', this.detectM3Max());
    params.set('hardware_concurrency', navigator.hardwareConcurrency || 4);
    params.set('memory_gb', this.estimateMemoryGB());
    params.set('optimization_enabled', true);
    params.set('user_agent', navigator.userAgent);
    params.set('platform', navigator.platform);
    params.set('language', navigator.language);
    params.set('timezone', Intl.DateTimeFormat().resolvedOptions().timeZone);
    
    return params;
  }

  static estimateMemoryGB(): number {
    if (typeof navigator === 'undefined') return 8.0;

    // navigator.deviceMemory가 있으면 사용
    if ('deviceMemory' in navigator) {
      return (navigator as any).deviceMemory;
    }

    // 하드웨어 코어 수로 추정
    const cores = navigator.hardwareConcurrency || 4;
    if (cores >= 16) return 32.0;
    if (cores >= 12) return 24.0;
    if (cores >= 8) return 16.0;
    if (cores >= 4) return 8.0;
    return 4.0;
  }

  // =================================================================
  // 🔧 브라우저 호환성 체크
  // =================================================================

  static checkBrowserCompatibility(): BrowserCompatibility {
    if (typeof window === 'undefined') {
      return {
        websocket: false,
        fileApi: false,
        formData: false,
        fetch: false,
        webgl: false,
        webassembly: false,
        overall: false
      };
    }

    const features = {
      websocket: 'WebSocket' in window,
      fileApi: 'File' in window && 'FileReader' in window && 'FileList' in window,
      formData: 'FormData' in window,
      fetch: 'fetch' in window,
      webgl: this.checkWebGLSupport(),
      webassembly: 'WebAssembly' in window
    };

    const overall = features.websocket && features.fileApi && features.formData && features.fetch;

    return { ...features, overall };
  }

  private static checkWebGLSupport(): boolean {
    try {
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      return !!context;
    } catch {
      return false;
    }
  }

  static getSystemRequirements(): SystemRequirements {
    return {
      minMemory: 4,
      recommendedMemory: 8,
      supportedBrowsers: [
        'Chrome 90+',
        'Firefox 88+',
        'Safari 14+',
        'Edge 90+'
      ],
      requiredFeatures: [
        'WebSocket',
        'File API',
        'FormData',
        'Fetch API'
      ]
    };
  }

  // =================================================================
  // 🔧 세션 및 ID 관리
  // =================================================================

  static generateSessionId(): string {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 15);
    const sessionId = `session_${timestamp}_${random}`;
    this.currentSessionId = sessionId;
    return sessionId;
  }

  static generateTaskId(): string {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 15);
    const taskId = `task_${timestamp}_${random}`;
    this.currentTaskId = taskId;
    return taskId;
  }

  static getCurrentSessionId(): string | null {
    return this.currentSessionId;
  }

  static getCurrentTaskId(): string | null {
    return this.currentTaskId;
  }

  static setCurrentSessionId(sessionId: string | null): void {
    this.currentSessionId = sessionId;
  }

  static setCurrentTaskId(taskId: string | null): void {
    this.currentTaskId = taskId;
  }

  // =================================================================
  // 🔧 설정 관리
  // =================================================================

  static mergeStepSpecificConfig(
    baseConfig: UsePipelineOptions,
    additionalConfig: Record<string, any>,
    systemParams: Map<string, any>
  ): UsePipelineOptions {
    const systemConfig: Partial<UsePipelineOptions> = {};
    
    for (const [key, value] of systemParams) {
      if (key in baseConfig) {
        (systemConfig as any)[key] = value;
      }
    }

    return {
      ...baseConfig,
      ...systemConfig,
      ...additionalConfig
    };
  }

  static validateConfiguration(config: UsePipelineOptions): {
    valid: boolean;
    errors: string[];
    warnings: string[];
  } {
    const errors: string[] = [];
    const warnings: string[] = [];

    // 필수 설정 확인
    if (!config.baseURL) {
      errors.push('baseURL is required');
    } else if (!this.isValidURL(config.baseURL)) {
      errors.push('baseURL must be a valid URL');
    }

    // 수치 범위 확인
    if (config.memory_gb && (config.memory_gb < 1 || config.memory_gb > 128)) {
      warnings.push('memory_gb should be between 1 and 128');
    }

    if (config.maxRetryAttempts && (config.maxRetryAttempts < 0 || config.maxRetryAttempts > 10)) {
      warnings.push('maxRetryAttempts should be between 0 and 10');
    }

    // 브라우저 호환성 확인
    const compatibility = this.checkBrowserCompatibility();
    if (!compatibility.overall) {
      errors.push('Browser does not support required features');
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings
    };
  }

  private static isValidURL(url: string): boolean {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }

  // =================================================================
  // 🔧 파일 처리 및 검증
  // =================================================================

  static validateImageType(file: File): boolean {
    const allowedTypes = [
      'image/jpeg',
      'image/jpg', 
      'image/png',
      'image/webp',
      'image/bmp',
      'image/gif'
    ];
    return allowedTypes.includes(file.type.toLowerCase());
  }

  static validateFileSize(file: File, maxSizeMB: number = 10): boolean {
    const maxBytes = maxSizeMB * 1024 * 1024;
    return file.size <= maxBytes;
  }

  static async getImageDimensions(file: File): Promise<{ width: number; height: number }> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }

  static async compressImage(
    file: File, 
    maxWidth: number = 1024, 
    maxHeight: number = 1024, 
    quality: number = 0.8
  ): Promise<File> {
    return new Promise((resolve, reject) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        reject(new Error('Canvas context not available'));
        return;
      }

      const img = new Image();
      img.onload = () => {
        // 비율 유지하며 크기 조정
        const ratio = Math.min(maxWidth / img.width, maxHeight / img.height);
        canvas.width = img.width * ratio;
        canvas.height = img.height * ratio;

        // 이미지 그리기
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

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
              reject(new Error('Image compression failed'));
            }
          },
          'image/jpeg',
          quality
        );
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }

  // =================================================================
  // 🔧 에러 처리
  // =================================================================

  static getUserFriendlyError(error: string | Error | any): string {
    let errorMessage = '';
    
    if (typeof error === 'string') {
      errorMessage = error;
    } else if (error instanceof Error) {
      errorMessage = error.message;
    } else if (error && typeof error === 'object') {
      errorMessage = error.message || error.detail || error.error || 'Unknown error';
    } else {
      errorMessage = 'Unknown error occurred';
    }

    const errorMappings: Record<string, string> = {
      // 연결 오류
      'connection failed': '서버에 연결할 수 없습니다. 네트워크 연결을 확인해주세요.',
      'network error': '네트워크 오류가 발생했습니다. 인터넷 연결을 확인해주세요.',
      'timeout': '요청 시간이 초과되었습니다. 다시 시도해주세요.',
      'websocket': 'WebSocket 연결에 문제가 있습니다.',
      
      // 파일 오류
      'invalid image': '지원되지 않는 이미지 형식입니다. JPG, PNG, WebP 파일을 사용해주세요.',
      'file too large': '파일 크기가 너무 큽니다. 10MB 이하의 파일을 업로드해주세요.',
      'invalid file': '잘못된 파일입니다. 이미지 파일인지 확인해주세요.',
      
      // 서버 오류
      'server error': '서버에 일시적인 문제가 발생했습니다. 잠시 후 다시 시도해주세요.',
      'internal server error': '서버 내부 오류가 발생했습니다. 지원팀에 문의해주세요.',
      'service unavailable': '서비스를 일시적으로 사용할 수 없습니다.',
      'bad gateway': '게이트웨이 오류가 발생했습니다.',
      
      // 인증/권한 오류
      'unauthorized': '인증이 필요합니다. 다시 로그인해주세요.',
      'forbidden': '접근 권한이 없습니다.',
      'rate limit': '요청 횟수 제한에 도달했습니다. 잠시 후 다시 시도해주세요.',
      
      // 요청 오류
      'bad request': '잘못된 요청입니다. 입력 정보를 확인해주세요.',
      'validation error': '입력값이 올바르지 않습니다.',
      'missing parameter': '필수 정보가 누락되었습니다.',
      
      // AI 처리 오류
      'processing failed': 'AI 처리 중 오류가 발생했습니다. 다른 이미지로 시도해보세요.',
      'model error': 'AI 모델에 문제가 있습니다. 잠시 후 다시 시도해주세요.',
      'out of memory': '메모리 부족으로 처리할 수 없습니다. 더 작은 이미지를 사용해주세요.',
      
      // 시스템 오류
      'initialization failed': '시스템 초기화에 실패했습니다. 페이지를 새로고침해주세요.',
      'configuration error': '설정 오류가 발생했습니다.',
      'dependency error': '필요한 구성 요소를 찾을 수 없습니다.',
    };

    const lowerError = errorMessage.toLowerCase();
    for (const [key, message] of Object.entries(errorMappings)) {
      if (lowerError.includes(key)) {
        return message;
      }
    }

    // HTTP 상태 코드 기반 메시지
    const httpCodeMatch = errorMessage.match(/(\d{3})/);
    if (httpCodeMatch) {
      const statusCode = parseInt(httpCodeMatch[1]);
      const httpMessage = this.getHTTPErrorMessage(statusCode);
      if (httpMessage) return httpMessage;
    }

    return '알 수 없는 오류가 발생했습니다. 지원팀에 문의해주세요.';
  }

  static getHTTPErrorMessage(status: number): string {
    const statusMessages: Record<number, string> = {
      400: '잘못된 요청입니다. 입력 정보를 확인해주세요.',
      401: '인증이 필요합니다. 로그인해주세요.',
      403: '접근 권한이 없습니다.',
      404: '요청한 리소스를 찾을 수 없습니다.',
      408: '요청 시간이 초과되었습니다.',
      409: '요청이 현재 서버 상태와 충돌합니다.',
      422: '입력 데이터가 올바르지 않습니다.',
      429: '너무 많은 요청을 보냈습니다. 잠시 후 다시 시도해주세요.',
      500: '서버 내부 오류가 발생했습니다.',
      502: '게이트웨이 오류입니다.',
      503: '서비스를 일시적으로 사용할 수 없습니다.',
      504: '게이트웨이 시간 초과입니다.'
    };

    return statusMessages[status] || `HTTP ${status} 오류가 발생했습니다.`;
  }

  static createAPIError(
    code: string,
    message: string,
    details?: Record<string, any>,
    retryAfter?: number
  ): APIError {
    return {
      code,
      message,
      details,
      timestamp: new Date().toISOString(),
      request_id: this.generateRequestId(),
      retry_after: retryAfter
    };
  }

  private static generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  // =================================================================
  // 🔧 이벤트 시스템
  // =================================================================

  static addEventListener(event: string, handler: Function): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, []);
    }
    this.eventListeners.get(event)!.push(handler);
  }

  static removeEventListener(event: string, handler: Function): void {
    const handlers = this.eventListeners.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  static emitEvent(event: string, data?: any): void {
    const handlers = this.eventListeners.get(event);
    if (handlers) {
      const eventObj: PipelineEvent = {
        type: event,
        data,
        timestamp: Date.now(),
        source: 'hook'
      };
      
      handlers.forEach(handler => {
        try {
          handler(eventObj);
        } catch (error) {
          this.error('Event handler error', { event, error });
        }
      });
    }
  }

  // =================================================================
  // 🔧 유틸리티 메서드들
  // =================================================================

  static sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  static debounce<T extends (...args: any[]) => any>(
    func: T,
    wait: number
  ): (...args: Parameters<T>) => void {
    let timeout: NodeJS.Timeout;
    return (...args: Parameters<T>) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => func(...args), wait);
    };
  }

  static throttle<T extends (...args: any[]) => any>(
    func: T,
    limit: number
  ): (...args: Parameters<T>) => void {
    let inThrottle: boolean;
    return (...args: Parameters<T>) => {
      if (!inThrottle) {
        func(...args);
        inThrottle = true;
        setTimeout(() => (inThrottle = false), limit);
      }
    };
  }

  static formatBytes(bytes: number, decimals: number = 2): string {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];

    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  }

  static formatDuration(ms: number): string {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  }

  static getQualityLevelConfig(level: QualityLevel): Record<string, any> {
    const configs = {
      fast: {
        resolution: 512,
        steps: 20,
        guidance_scale: 7.5,
        use_fp16: true,
        enable_xformers: true
      },
      balanced: {
        resolution: 768,
        steps: 30,
        guidance_scale: 7.5,
        use_fp16: true,
        enable_xformers: true
      },
      quality: {
        resolution: 1024,
        steps: 50,
        guidance_scale: 10.0,
        use_fp16: false,
        enable_xformers: false
      },
      ultra: {
        resolution: 1024,
        steps: 100,
        guidance_scale: 12.0,
        use_fp16: false,
        enable_xformers: false
      }
    };

    return configs[level] || configs.balanced;
  }

  // =================================================================
  // 🔧 정리 메서드
  // =================================================================

  static cleanup(): void {
    this.logger.clearLogs();
    this.perfMonitor.clearMetrics();
    this.eventListeners.clear();
    this.currentSessionId = null;
    this.currentTaskId = null;
  }
}

export default PipelineUtils;