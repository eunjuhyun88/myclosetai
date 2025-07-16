// frontend/src/main.tsx - 완전한 수정 버전
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'
//import App from './App_Working'  // 테스트용

// =================================================================
// 🔧 React 18 StrictMode 문제 해결
// =================================================================

/**
 * React 18의 StrictMode는 개발 모드에서 의도적으로 컴포넌트를 두 번 렌더링합니다.
 * 이로 인해 다음 문제들이 발생합니다:
 * 
 * 1. useEffect가 두 번 실행됨
 * 2. API 클라이언트가 중복 생성/파괴됨
 * 3. WebSocket 연결이 반복적으로 중단됨
 * 4. "Fetch is aborted" 오류 발생
 * 
 * 해결책:
 * - 개발 중에는 StrictMode 비활성화
 * - 프로덕션에서는 StrictMode 유지 (선택사항)
 */

const isDevelopment = process.env.NODE_ENV === 'development';
const isProduction = process.env.NODE_ENV === 'production';

// 환경별 렌더링 설정
const renderApp = () => {
  const rootElement = document.getElementById('root');
  
  if (!rootElement) {
    throw new Error('Root element not found');
  }

  const root = ReactDOM.createRoot(rootElement);

  if (isDevelopment) {
    // 🔴 개발 모드: StrictMode 비활성화 (API 연동 안정화)
    console.log('🔧 개발 모드: StrictMode 비활성화');
    root.render(<App />);
  } else {
    // ✅ 프로덕션 모드: StrictMode 활성화 (React 베스트 프랙티스)
    console.log('🚀 프로덕션 모드: StrictMode 활성화');
    root.render(
      <React.StrictMode>
        <App />
      </React.StrictMode>
    );
  }
};

// 애플리케이션 렌더링
renderApp();

// =================================================================
// 🔧 개발 도구 (개발 모드에서만)
// =================================================================

if (isDevelopment) {
  // 전역 개발 도구 함수들
  (window as any).devTools = {
    // API 테스트
    testAPI: async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        const data = await response.json();
        console.log('✅ API 테스트 성공:', data);
        return data;
      } catch (error) {
        console.error('❌ API 테스트 실패:', error);
        return null;
      }
    },

    // CORS 테스트
    testCORS: async () => {
      try {
        const response = await fetch('http://localhost:8000/health', {
          method: 'GET',
          headers: {
            'Origin': window.location.origin,
            'Content-Type': 'application/json'
          }
        });
        console.log('✅ CORS 테스트 성공:', response.status);
        return response.ok;
      } catch (error) {
        console.error('❌ CORS 테스트 실패:', error);
        return false;
      }
    },

    // WebSocket 테스트
    testWebSocket: () => {
      try {
        const ws = new WebSocket('ws://localhost:8000/api/ws/test');
        
        ws.onopen = () => {
          console.log('✅ WebSocket 연결 성공');
          ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
        };
        
        ws.onmessage = (event) => {
          console.log('📨 WebSocket 메시지:', JSON.parse(event.data));
        };
        
        ws.onerror = (error) => {
          console.error('❌ WebSocket 오류:', error);
        };
        
        ws.onclose = () => {
          console.log('🔌 WebSocket 연결 종료');
        };

        // 5초 후 연결 종료
        setTimeout(() => {
          ws.close();
        }, 5000);

        return ws;
      } catch (error) {
        console.error('❌ WebSocket 테스트 실패:', error);
        return null;
      }
    },

    // 앱 상태 리셋
    resetApp: () => {
      window.location.reload();
    }
  };

  // 개발 도구 사용법 출력
  console.log(`
🛠️ MyCloset AI 개발 도구 사용법:

1. API 테스트:        devTools.testAPI()
2. CORS 테스트:       devTools.testCORS()  
3. WebSocket 테스트:  devTools.testWebSocket()
4. 앱 리셋:          devTools.resetApp()

예시:
- await devTools.testAPI()
- await devTools.testCORS()
`);
}

// =================================================================
// 🔧 에러 핸들링
// =================================================================

// 전역 에러 핸들러
window.addEventListener('error', (event) => {
  console.error('전역 에러:', event.error);
  
  if (isDevelopment) {
    // 개발 모드에서는 더 자세한 정보 출력
    console.error('에러 상세 정보:', {
      message: event.message,
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
      error: event.error
    });
  }
});

// Promise 거부 핸들러
window.addEventListener('unhandledrejection', (event) => {
  console.error('처리되지 않은 Promise 거부:', event.reason);
  
  if (isDevelopment) {
    console.error('Promise 거부 상세:', event);
  }
});

// =================================================================
// 🔧 성능 모니터링 (개발 모드)
// =================================================================

if (isDevelopment && 'performance' in window) {
  // 페이지 로드 성능 측정
  window.addEventListener('load', () => {
    setTimeout(() => {
      const perfData = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      
      console.log('📊 페이지 로드 성능:', {
        'DOM 로드': `${perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart}ms`,
        '전체 로드': `${perfData.loadEventEnd - perfData.loadEventStart}ms`,
        'DNS 조회': `${perfData.domainLookupEnd - perfData.domainLookupStart}ms`,
        'TCP 연결': `${perfData.connectEnd - perfData.connectStart}ms`,
        '요청-응답': `${perfData.responseEnd - perfData.requestStart}ms`
      });
    }, 0);
  });
}

// =================================================================
// 🔧 브라우저 호환성 체크
// =================================================================

const checkBrowserCompatibility = () => {
  const features = {
    'WebSocket': 'WebSocket' in window,
    'Fetch API': 'fetch' in window,
    'Local Storage': 'localStorage' in window,
    'Session Storage': 'sessionStorage' in window,
    'File API': 'File' in window,
    'FormData': 'FormData' in window,
    'URL.createObjectURL': 'URL' in window && 'createObjectURL' in URL,
    'Web Workers': 'Worker' in window,
    'Service Workers': 'serviceWorker' in navigator
  };

  const unsupported = Object.entries(features)
    .filter(([, supported]) => !supported)
    .map(([feature]) => feature);

  return unsupported.length === 0;
};

// 호환성 체크 실행
checkBrowserCompatibility();

// =================================================================
// 🔧 환경 정보 출력
// =================================================================

console.log(`
🚀 MyCloset AI Frontend 시작됨

환경 정보:
- 모드: ${process.env.NODE_ENV}
- React 버전: ${React.version}
- StrictMode: ${isProduction ? '✅ 활성화' : '❌ 비활성화'}
- User Agent: ${navigator.userAgent}
- 화면 크기: ${window.screen.width}x${window.screen.height}
- 뷰포트: ${window.innerWidth}x${window.innerHeight}

API 정보:
- 백엔드 URL: http://localhost:8000
- WebSocket URL: ws://localhost:8000
- Health Check: http://localhost:8000/health
`);

export {};