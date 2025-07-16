import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react({
      // React Fast Refresh 설정
      fastRefresh: true,
      // JSX 런타임 설정
      jsxRuntime: 'automatic'
    })
  ],
  
  // 서버 설정
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    open: false, // 자동으로 브라우저 열지 않음
    cors: true,
    // 백엔드 API 프록시 설정
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        ws: true, // WebSocket 지원
        timeout: 60000, // 60초 타임아웃
        configure: (proxy, _options) => {
          proxy.on('error', (err, _req, _res) => {
            console.log('proxy error', err);
          });
          proxy.on('proxyReq', (proxyReq, req, _res) => {
            console.log('Sending Request to the Target:', req.method, req.url);
          });
          proxy.on('proxyRes', (proxyRes, req, _res) => {
            console.log('Received Response from the Target:', proxyRes.statusCode, req.url);
          });
        },
      },
      // WebSocket 프록시 (실시간 통신)
      '/api/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true
      }
    }
  },
  
  // 빌드 설정
  build: {
    target: 'es2020',
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true,
        drop_debugger: true
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          utils: ['src/utils']
        }
      }
    },
    // 청크 크기 경고 제한
    chunkSizeWarningLimit: 1000
  },
  
  // 미리보기 서버 설정
  preview: {
    host: '0.0.0.0',
    port: 4173,
    strictPort: true,
    open: false
  },
  
  // 경로 해결 설정
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@utils': resolve(__dirname, 'src/utils'),
      '@types': resolve(__dirname, 'src/types'),
      '@services': resolve(__dirname, 'src/services'),
      '@hooks': resolve(__dirname, 'src/hooks')
    }
  },
  
  // CSS 설정
  css: {
    devSourcemap: true,
    preprocessorOptions: {
      scss: {
        additionalData: `@import "@/styles/variables.scss";`
      }
    }
  },
  
  // 최적화 설정
  optimizeDeps: {
    include: ['react', 'react-dom'],
    exclude: []
  },
  
  // 환경 변수 설정
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
    __BUILD_TIME__: JSON.stringify(new Date().toISOString())
  },
  
  // 공용 디렉토리
  publicDir: 'public',
  
  // 에셋 포함 패턴
  assetsInclude: ['**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.gif', '**/*.svg', '**/*.webp'],
  
  // 로거 설정
  logLevel: 'info',
  clearScreen: false,
  
  // 실험적 기능
  experimental: {
    renderBuiltUrl(filename, { hostType }) {
      if (hostType === 'js') {
        return { js: `/${filename}` };
      }
      return { relative: true };
    }
  }
});